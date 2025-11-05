from __future__ import annotations

import math, random, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
from tqdm import tqdm
import wandb

from cambrian.tokenizer import Tokenizer
from cambrian.data import load_splits, make_loader
from cambrian.distributed import setup_ddp, cleanup_ddp, is_main_process
from cambrian.training import set_torch_defaults, sdpa_flash, cosine_lr, auto_microbatch, save_checkpoint
from cambrian.utils import parse_number

# -----------------------------------------------------------------------------
# Minimal configs (about param counts; "nano" ~31M)
# -----------------------------------------------------------------------------
CONFIGS: dict[str, dict] = {
    "nano":   dict(n_layer=8,  n_head=6, n_kv_head=1, n_embd=384,  max_seq_len=256, rope_theta=10_000.0),
    "micro":  dict(n_layer=10, n_head=6, n_kv_head=1, n_embd=384,  max_seq_len=256, rope_theta=10_000.0),
    "tiny":   dict(n_layer=12, n_head=6, n_kv_head=1, n_embd=384,  max_seq_len=256, rope_theta=10_000.0),
    "small":  dict(n_layer=12, n_head=8, n_kv_head=2, n_embd=768,  max_seq_len=512, rope_theta=10_000.0),
    "medium": dict(n_layer=18, n_head=12,n_kv_head=2, n_embd=1024, max_seq_len=1024,rope_theta=10_000.0),
}

# -----------------------------------------------------------------------------
# Tiny GPT with RoPE, QK-norm, MQA, ReLU^2 MLP, paramless RMSNorm, no biases
# Uses PyTorch SDPA => FlashAttention on Ampere+ when enabled
# -----------------------------------------------------------------------------
def rms_norm(x: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

def rotary_cache(T: int, hd: int, base: float=10_000.0, device=None, dtype=torch.bfloat16):
    t = torch.arange(T, device=device, dtype=torch.float32)
    inv = base ** (-torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd)
    freqs = torch.outer(t, inv)
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos[None, :, None, :].to(dtype), sin[None, :, None, :].to(dtype)  # (1,T,1,hd/2)
    return cos, sin

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B,T,H,hd) ; cos/sin: (1,T,1,hd/2)
    hd = x.shape[-1] // 2
    x1, x2 = x[..., :hd], x[..., hd:]
    y1 = x1 * cos + x2 * sin
    y2 = x2 * cos - x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)

class MHA(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % n_kv_head == 0
        self.n_head, self.n_kv_head = n_head, n_kv_head
        self.hd = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_head   * self.hd, bias=False)
        self.wk = nn.Linear(n_embd, n_kv_head* self.hd, bias=False)
        self.wv = nn.Linear(n_embd, n_kv_head* self.hd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head,   self.hd)
        k = self.wk(x).view(B, T, self.n_kv_head,self.hd)
        v = self.wv(x).view(B, T, self.n_kv_head,self.hd)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q, k = rms_norm(q), rms_norm(k)  # QK-norm
        # GQA/MQA: expand kv heads to match query heads
        g = self.n_head // self.n_kv_head
        if g > 1:
            k = k.repeat_interleave(g, dim=2)
            v = v.repeat_interleave(g, dim=2)
        # (B,T,H,D) -> (B,H,T,D)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # FlashAttention via SDPA
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y)

class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)
    def forward(self, x): return self.proj(F.relu(self.fc(x)).square())

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int):
        super().__init__()
        self.attn = MHA(n_embd, n_head, n_kv_head)
        self.mlp  = MLP(n_embd)
    def forward(self, x, cos, sin):
        x = x + self.attn(rms_norm(x), cos, sin)
        x = x + self.mlp(rms_norm(x))
        return x

class GPT(nn.Module):
    SOFTCAP = 15.0
    
    def __init__(self, vocab_size: int, n_layer: int, n_head: int, n_kv_head: int, n_embd: int, max_seq_len: int, rope_theta: float):
        super().__init__()
        self.vocab_size, self.n_embd, self.max_seq_len = vocab_size, n_embd, max_seq_len
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.h   = nn.ModuleList([Block(n_embd, n_head, n_kv_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # rotary cache
        cos, sin = rotary_cache(max_seq_len * 2, n_embd // n_head, base=rope_theta)  # small overprovision
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Tie input/output embeddings
        self.lm_head.weight = self.wte.weight
        
        # Init (simple, effective)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.size(1)
                std = 1.0 / math.sqrt(fan_in)
                nn.init.normal_(m.weight, mean=0.0, std=std)
        nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"seq len {T} > max_seq_len {self.max_seq_len}"
        x = self.wte(idx)
        cos, sin = self.cos[:, :T], self.sin[:, :T]
        for blk in self.h: x = blk(x, cos, sin)
        x = rms_norm(x)
        logits = self.lm_head(x)
        if targets is None: return logits
        # softcap for stability
        logits = self.SOFTCAP * torch.tanh(logits / self.SOFTCAP)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss

    def count_params(self) -> int: return sum(p.numel() for p in self.parameters())

# -----------------------------------------------------------------------------
# Main training
# -----------------------------------------------------------------------------
def main(
    run_name: str,
    dataset: str,
    model: str = "nano",
    max_iters: str | None = None,
    max_steps: str | None = None,
    max_tokens: str | None = None,
    warmup_iters: int = 2_000,
    base_lr: float = 3e-4,
    weight_decay: float = 0.1,
    eval_interval: int = 500,
    log_interval: int = 10,
    grad_clip: float = 1.0,
    target_mem_frac: float = 0.85,
    out_dir: Path = Path("out"),
    seed: int = 1337,
):
    max_iters_int = parse_number(max_steps or max_iters)
    max_tokens_int = parse_number(max_tokens)
    
    # Initialize DDP
    rank, world_size, local_rank = setup_ddp()
    set_torch_defaults()
    random.seed(seed + rank); torch.manual_seed(seed + rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process():
        logger.info(f"Device: cuda  GPUs: {world_size}")

    # tokenizer and dataset
    tok = Tokenizer("gpt2")
    cfg = dict(vocab_size=tok.vocab_size, **CONFIGS[model])
    tr_texts, va_texts = load_splits(dataset)

    # model
    net = GPT(**cfg)
    if is_main_process():
        logger.info(f"Model={model}: {net.count_params()/1e6:.1f}M params | {cfg['n_layer']}L × {cfg['n_embd']}D × {cfg['n_head']}H (KV={cfg['n_kv_head']})")

    amp_dtype = torch.bfloat16
    if is_main_process():
        logger.info(f"AMP dtype: bf16")

    # auto micro-batch per GPU
    bpg = auto_microbatch(net, cfg["max_seq_len"], cfg["vocab_size"], target_mem_frac, amp_dtype, device)
    
    # Synchronize bpg across ranks to avoid DDP hangs
    bpg_tensor = torch.tensor([bpg], device=device)
    if dist.is_initialized():
        dist.all_reduce(bpg_tensor, op=dist.ReduceOp.MIN)
    bpg = int(bpg_tensor.item())
    
    global_bs = bpg * world_size
    if is_main_process():
        logger.info(f"Auto micro-batch per GPU: {bpg}  -> Global batch: {global_bs}")

    tokens_per_step = global_bs * cfg["max_seq_len"]
    if max_tokens_int:
        max_iters_int = max_tokens_int // tokens_per_step
    if not max_iters_int:
        max_iters_int = 100_000
    
    params = net.count_params()
    flops_per_step = 6 * params * tokens_per_step
    if is_main_process():
        logger.info(f"Training: {max_iters_int:,} steps, {params/1e6:.1f}M params, {flops_per_step:,.2e} FLOPS/step")

    use_ddp = world_size > 1
    train_loader = make_loader(tr_texts, tok, cfg["max_seq_len"], bpg, use_ddp, is_train=True)
    val_loader = make_loader(va_texts, tok, cfg["max_seq_len"], bpg, use_ddp, is_train=False)
    net = net.to(device)
    if use_ddp: net = DDP(net, device_ids=[local_rank])

    no_decay_names = {"wte.weight", "lm_head.weight", "module.wte.weight", "module.lm_head.weight"}
    param_groups = [
        {"params": [p for n, p in net.named_parameters() if n not in no_decay_names], "weight_decay": weight_decay},
        {"params": [p for n, p in net.named_parameters() if n in no_decay_names], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.95), fused=True)

    if is_main_process():
        wandb.init(project="cambrian", name=run_name, config=dict(run_name=run_name, dataset=dataset, **cfg,
                                                                  base_lr=base_lr, weight_decay=weight_decay,
                                                                  max_iters=max_iters_int, warmup_iters=warmup_iters,
                                                                  global_batch_size=global_bs))
    step = 0
    total_flops_so_far = 0
    best_val = float("inf")
    out_run = out_dir / run_name
    checkpoint_interval = max(1, int(max_iters_int * 0.05))
    last_log_time = time.time()
    steps_since_log = 0

    pbar = tqdm(total=max_iters_int, desc="training", dynamic_ncols=True, disable=not is_main_process())
    epoch = 0
    while step < max_iters_int:
        if use_ddp:
            train_loader.sampler.set_epoch(epoch)
        train_loader.dataset.set_epoch(epoch)
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            lr = cosine_lr(step, max_iters_int, warmup_iters, base_lr)
            for g in opt.param_groups: g["lr"] = lr

            with sdpa_flash(), torch.autocast(device_type='cuda', dtype=amp_dtype):
                loss = net(xb, yb)
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()
            
            steps_since_log += 1
            total_flops_so_far += flops_per_step

            if is_main_process() and step % log_interval == 0:
                elapsed = max(time.time() - last_log_time, 1e-6)
                tok_per_s = tokens_per_step * steps_since_log / elapsed
                flops_per_s = flops_per_step * steps_since_log / elapsed
                mem = torch.cuda.max_memory_reserved(local_rank) / torch.cuda.get_device_properties(local_rank).total_memory
                
                wandb.log({
                    "loss/train": loss.item(), 
                    "lr": lr, 
                    "tokens_per_s": tok_per_s, 
                    "flops_per_s": flops_per_s,
                    "total_flops": total_flops_so_far,
                    "mem_frac": mem, 
                    "step": step
                })
                logger.info(f"step={step:,} loss={loss.item():.4f} lr={lr:.2e} tok/s={tok_per_s:,.0f} FLOPS/s={flops_per_s:.2e}")
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}", tok_s=f"{tok_per_s:,.0f}", mem=f"{mem:.2f}")
                last_log_time = time.time()
                steps_since_log = 0

            should_eval = False
            if step > 0:
                if eval_interval and eval_interval > 0 and step % eval_interval == 0:
                    should_eval = True
                if step % checkpoint_interval == 0:
                    should_eval = True

            if step % checkpoint_interval == 0:
                progress_pct = int(100 * step / max_iters_int)
                ckpt_dir = out_run / f"checkpoint_step_{step}_progress_{progress_pct}pct"
                save_checkpoint(ckpt_dir, net, dict(model=model, **cfg), dict(step=step, progress_pct=progress_pct))
            
            if should_eval:
                net.eval()
                losses = []
                with torch.no_grad():
                    for i, (xb, yb) in enumerate(val_loader):
                        if i >= 50: break
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                        with sdpa_flash(), torch.autocast(device_type='cuda', dtype=amp_dtype):
                            losses.append(net(xb, yb).item())
                loss_sum = torch.tensor([sum(losses)], device=device, dtype=torch.float32)
                count_tensor = torch.tensor([len(losses)], device=device, dtype=torch.float32)
                if dist.is_initialized():
                    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                total_count = int(count_tensor.item())
                if total_count == 0:
                    if is_main_process():
                        logger.warning("Validation loader produced no batches; skipping eval.")
                        wandb.log({"loss/val": float("nan"), "step": step})
                    net.train()
                else:
                    val_loss = (loss_sum / count_tensor).item()
                    if is_main_process():
                        ppl = math.exp(min(20, val_loss)) if math.isfinite(val_loss) else float("nan")
                        wandb.log({"loss/val": val_loss, "ppl/val": ppl, "step": step})
                        if math.isfinite(val_loss):
                            logger.info(f"[eval] step {step}  val_loss={val_loss:.4f}  ppl={ppl:.1f}")
                        else:
                            logger.info(f"[eval] step {step}  val_loss=nan  (no valid batches)")
                        if val_loss < best_val:
                            best_val = val_loss
                            ckpt_dir = out_run / "best"
                            save_checkpoint(ckpt_dir, net, dict(model=model, **cfg), dict(step=step, val_loss=val_loss, ppl=ppl))
                    net.train()

            step += 1
            pbar.update(1)
            if step >= max_iters_int: break
        if step >= max_iters_int: break
        epoch += 1

    pbar.close()
    del train_loader, val_loader
    
    # final save
    final_dir = out_run / "final"
    save_checkpoint(final_dir, net, dict(model=model, **cfg), dict(step=step, finished=True))
    if is_main_process():
        logger.success("Training complete.")
        wandb.finish()
    cleanup_ddp()

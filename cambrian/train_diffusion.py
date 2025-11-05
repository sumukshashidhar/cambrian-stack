from __future__ import annotations

import math, random, time
from pathlib import Path

import torch
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
from cambrian.diffusion import DiffusionLM, DiffusionConfig

# -----------------------------------------------------------------------------
# Config presets (match your Transformer sizes; no kv-sharing here)
# -----------------------------------------------------------------------------
CONFIGS: dict[str, dict] = {
    "nano":   dict(n_layer=8,  n_head=6, n_embd=384,  max_seq_len=256, rope_theta=10_000.0),
    "micro":  dict(n_layer=10, n_head=6, n_embd=384,  max_seq_len=256, rope_theta=10_000.0),
    "tiny":   dict(n_layer=12, n_head=6, n_embd=384,  max_seq_len=256, rope_theta=10_000.0),
    "small":  dict(n_layer=12, n_head=8, n_embd=768,  max_seq_len=512, rope_theta=10_000.0),
    "medium": dict(n_layer=18, n_head=12,n_embd=1024, max_seq_len=1024,rope_theta=10_000.0),
}

def main(
    run_name: str,
    dataset: str,
    model: str = "nano",
    max_iters: str | None = None,
    max_steps: str | None = None,
    max_tokens: str | None = None,
    diffusion_steps: int = 128,
    context_len: int = 0,
    warmup_iters: int = 2_000,
    base_lr: float = 3e-4,
    weight_decay: float = 0.1,
    eval_interval: int = 500,
    log_interval: int = 10,
    grad_clip: float = 1.0,
    target_mem_frac: float = 0.30,
    out_dir: Path = Path("out"),
    seed: int = 1337,
):
    max_iters_int = parse_number(max_steps or max_iters)
    max_tokens_int = parse_number(max_tokens)

    # DDP & defaults
    rank, world_size, local_rank = setup_ddp()
    set_torch_defaults()
    random.seed(seed + rank); torch.manual_seed(seed + rank)
    device = torch.device(f"cuda:{local_rank}")
    if is_main_process(): logger.info(f"Device: cuda  GPUs: {world_size}")

    # Tokenizer / data
    tok = Tokenizer("gpt2")  # keep the same tokenizer
    tr_texts, va_texts = load_splits(dataset)

    # Model config
    base = CONFIGS[model]
    vocab_with_mask = tok.vocab_size + 1
    mask_token_id = tok.vocab_size
    dcfg = DiffusionConfig(
        vocab_size=vocab_with_mask,
        mask_token_id=mask_token_id,
        diffusion_steps=diffusion_steps,
        context_len=context_len,
        **base,
    )
    net = DiffusionLM(dcfg)
    if is_main_process():
        logger.info(f"Diffusion {model}: {net.count_params()/1e6:.1f}M params | "
                    f"{base['n_layer']}L × {base['n_embd']}D × {base['n_head']}H | "
                    f"S={diffusion_steps} context={context_len}")

    amp_dtype = torch.bfloat16
    if is_main_process(): logger.info("AMP dtype: bf16")

    # Auto micro-batch per GPU
    # Use the *base tokenizer size* for the synthetic inputs in the tuner to avoid sampling the mask id.
    bpg = auto_microbatch(net, base["max_seq_len"], tok.vocab_size, target_mem_frac, amp_dtype, device)
    bpg_tensor = torch.tensor([bpg], device=device)
    if dist.is_initialized(): dist.all_reduce(bpg_tensor, op=dist.ReduceOp.MIN)
    bpg = int(bpg_tensor.item())
    global_bs = bpg * world_size
    if is_main_process(): logger.info(f"Auto micro-batch per GPU: {bpg}  -> Global batch: {global_bs}")

    tokens_per_step = global_bs * base["max_seq_len"]
    if max_tokens_int: max_iters_int = max_tokens_int // tokens_per_step
    if not max_iters_int: max_iters_int = 100_000

    params = net.count_params()
    flops_per_step = 6 * params * tokens_per_step
    if is_main_process():
        logger.info(f"Training: {max_iters_int:,} steps, {params/1e6:.1f}M params, {flops_per_step:,.2e} FLOPS/step")

    use_ddp = world_size > 1
    train_loader = make_loader(tr_texts, tok, base["max_seq_len"], bpg, use_ddp, is_train=True)
    val_loader   = make_loader(va_texts, tok, base["max_seq_len"], bpg, use_ddp, is_train=False)
    net = net.to(device)
    if use_ddp: net = DDP(net, device_ids=[local_rank])

    # Weight decay: exclude embeddings & tied head
    no_decay = {
        "wte.weight", "head.weight",
        "net.wte.weight", "net.head.weight",
        "module.wte.weight", "module.head.weight",
        "module.net.wte.weight", "module.net.head.weight",
    }
    param_groups = [
        {"params": [p for n, p in net.named_parameters() if n not in no_decay], "weight_decay": weight_decay},
        {"params": [p for n, p in net.named_parameters() if n in no_decay], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.95), fused=True)

    if is_main_process():
        wandb.init(project="cambrian", name=run_name, config=dict(
            run_name=run_name, dataset=dataset, **base,
            base_lr=base_lr, weight_decay=weight_decay,
            max_iters=max_iters_int, warmup_iters=warmup_iters,
            global_batch_size=global_bs, diffusion_steps=diffusion_steps, context_len=context_len
        ))

    step = 0
    total_flops_so_far = 0
    best_val = float("inf")
    out_run = out_dir / run_name
    checkpoint_interval = max(1, int(max_iters_int * 0.05))
    last_log_time = time.time()
    steps_since_log = 0

    pbar = tqdm(total=max_iters_int, desc="training(diffusion)", dynamic_ncols=True, disable=not is_main_process())
    epoch = 0
    while step < max_iters_int:
        if use_ddp: train_loader.sampler.set_epoch(epoch)
        train_loader.dataset.set_epoch(epoch)
        for xb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)

            lr = cosine_lr(step, max_iters_int, warmup_iters, base_lr)
            for g in opt.param_groups: g["lr"] = lr

            with sdpa_flash(), torch.autocast(device_type='cuda', dtype=amp_dtype):
                loss = net(xb, xb)  # targets unused; API compatible

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
                if eval_interval and eval_interval > 0 and step % eval_interval == 0: should_eval = True
                if step % checkpoint_interval == 0: should_eval = True

            if step % checkpoint_interval == 0:
                progress_pct = int(100 * step / max_iters_int)
                ckpt_dir = out_run / f"checkpoint_step_{step}_progress_{progress_pct}pct"
                save_checkpoint(ckpt_dir, net, dict(model=f"diffusion-{model}", **base,
                                                    diffusion_steps=diffusion_steps, context_len=context_len,
                                                    vocab_size=vocab_with_mask),
                                dict(step=step, progress_pct=progress_pct))

            if should_eval:
                net.eval()
                losses = []
                with torch.no_grad():
                    for i, (xb, _) in enumerate(val_loader):
                        if i >= 50: break
                        xb = xb.to(device, non_blocking=True)
                        with sdpa_flash(), torch.autocast(device_type='cuda', dtype=amp_dtype):
                            losses.append(net(xb, xb).item())
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
                        wandb.log({"loss/val": val_loss, "step": step})
                        logger.info(f"[eval] step {step}  val_loss={val_loss:.4f}")
                        if val_loss < best_val:
                            best_val = val_loss
                            ckpt_dir = out_run / "best"
                            save_checkpoint(ckpt_dir, net, dict(model=f"diffusion-{model}", **base,
                                                                diffusion_steps=diffusion_steps, context_len=context_len,
                                                                vocab_size=vocab_with_mask),
                                            dict(step=step, val_loss=val_loss))
                    net.train()

            step += 1
            pbar.update(1)
            if step >= max_iters_int: break
        if step >= max_iters_int: break
        epoch += 1

    pbar.close()
    del train_loader, val_loader

    final_dir = out_run / "final"
    save_checkpoint(final_dir, net, dict(model=f"diffusion-{model}", **base,
                                         diffusion_steps=diffusion_steps, context_len=context_len,
                                         vocab_size=vocab_with_mask),
                    dict(step=step, finished=True))
    if is_main_process():
        logger.success("Training complete.")
        wandb.finish()
    cleanup_ddp()

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Simple Discrete Diffusion Transformer
# - Bidirectional attention (no causal mask)
# - Time-step conditioning via learned time embeddings
# - RoPE + QK RMSNorm
# - ReLU^2 MLP, biasless linears (matches your GPT style)
# - One extra vocab id is reserved as [MASK] and used only in inputs
# -----------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

def rotary_cache(T: int, hd: int, base: float = 10_000.0, device=None, dtype=torch.bfloat16):
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

class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)
    def forward(self, x): return self.proj(F.relu(self.fc(x)).square())

class BidirectionalMHA(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.hd = n_head, n_embd // n_head
        self.wq = nn.Linear(n_embd, n_head * self.hd, bias=False)
        self.wk = nn.Linear(n_embd, n_head * self.hd, bias=False)
        self.wv = nn.Linear(n_embd, n_head * self.hd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.hd)
        k = self.wk(x).view(B, T, self.n_head, self.hd)
        v = self.wv(x).view(B, T, self.n_head, self.hd)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q, k = rms_norm(q), rms_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B,H,T,hd)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y)

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.attn = BidirectionalMHA(n_embd, n_head)
        self.mlp  = MLP(n_embd)
    def forward(self, x, cos, sin):
        x = x + self.attn(rms_norm(x), cos, sin)
        x = x + self.mlp(rms_norm(x))
        return x

@dataclass
class DiffusionConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    max_seq_len: int
    rope_theta: float = 10_000.0
    diffusion_steps: int = 128
    context_len: int = 0
    mask_token_id: int | None = None

class DiffusionTransformer(nn.Module):
    """Backbone transformer that maps (x_t, t) -> logits over tokens."""
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        vs = cfg.vocab_size
        assert cfg.mask_token_id is not None and 0 <= cfg.mask_token_id < vs

        self.wte = nn.Embedding(vs, cfg.n_embd)
        self.tte = nn.Embedding(cfg.diffusion_steps, cfg.n_embd)
        self.h   = nn.ModuleList([Block(cfg.n_embd, cfg.n_head) for _ in range(cfg.n_layer)])
        self.head = nn.Linear(cfg.n_embd, vs, bias=False)

        # share token embedding and output head
        self.head.weight = self.wte.weight

        # RoPE cache (small over-provision)
        cos, sin = rotary_cache(cfg.max_seq_len * 2, cfg.n_embd // cfg.n_head, base=cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Init similar to your GPT init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.size(1)
                std = 1.0 / math.sqrt(fan_in)
                nn.init.normal_(m.weight, mean=0.0, std=std)
        nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: (B,T) int64 ; t: (B,) int64
        B, T = x_t.shape
        assert T <= self.cfg.max_seq_len
        x = self.wte(x_t) + self.tte(t).unsqueeze(1)  # (B,T,C)
        cos, sin = self.cos[:, :T], self.sin[:, :T]
        for blk in self.h: x = blk(x, cos, sin)
        x = rms_norm(x)
        return self.head(x)  # (B,T,V)

class DiffusionLM(nn.Module):
    """Thin wrapper that exposes a training-friendly forward(idx, targets)->loss."""
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.net = DiffusionTransformer(cfg)

    @torch.no_grad()
    def _make_noisy_inputs(self, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x_t, t, mask) where mask marks positions to supervise."""
        B, T = clean.shape
        device = clean.device
        # sample a timestep per batch element, uniform in [0, S-1]
        t = torch.randint(low=0, high=self.cfg.diffusion_steps, size=(B,), device=device, dtype=torch.long)
        # mask probability grows with t (simple linear schedule)
        p = (t.to(torch.float32) + 1.0) / float(self.cfg.diffusion_steps)  # (B,)
        p = p.clamp_(0.0, 1.0).unsqueeze(1)  # (B,1)

        # eligible positions exclude prefix context
        maskable = torch.ones(B, T, device=device, dtype=torch.bool)
        if self.cfg.context_len > 0:
            maskable[:, : self.cfg.context_len] = False

        # Bernoulli per-token, per-sample
        u = torch.rand(B, T, device=device)
        mask = (u < p) & maskable  # (B,T)

        # ensure at least one masked token per sequence when possible
        for b in range(B):
            if maskable[b].any() and not mask[b].any():
                # pick a random eligible position
                idx = torch.randint(self.cfg.context_len, T, (1,), device=device)
                mask[b, idx] = True

        x_t = clean.clone()
        x_t[mask] = self.cfg.mask_token_id
        return x_t, t, mask

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        """
        Training loss. (targets is ignored; kept for API compatibility)
        idx: (B,T) clean tokens from dataset
        returns: scalar CE over masked positions
        """
        clean = idx
        x_t, t, mask = self._make_noisy_inputs(clean)
        logits = self.net(x_t, t)  # (B,T,V)
        if not mask.any():
            # degenerate corner case (should be rare due to fixup above)
            # supervise last position to avoid NaN
            mask[:, -1] = True
        logits_m = logits[mask]           # (#M, V)
        targets_m = clean[mask]           # (#M,)
        loss = F.cross_entropy(logits_m, targets_m)
        return loss

    def count_params(self) -> int: return sum(p.numel() for p in self.parameters())

    # Optional generation entrypoints if you want quick parallel decoding later.
    @torch.inference_mode()
    def sample_confidence(
        self,
        batch_size: int,
        seq_len: int,
        confidence_threshold: float = 0.95,
        num_steps: int | None = None,
        temperature: float = 1.0,
        device: torch.device | None = None,
        context_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if device is None: device = next(self.parameters()).device
        if num_steps is None: num_steps = seq_len

        x = torch.full((batch_size, seq_len), self.cfg.mask_token_id, dtype=torch.long, device=device)
        if context_tokens is not None:
            Lc = context_tokens.size(1)
            x[:, :Lc] = context_tokens.to(device)
        masked = torch.ones_like(x, dtype=torch.bool)
        if context_tokens is not None:
            masked[:, :Lc] = False

        for step in range(num_steps):
            if not masked.any(): break
            t = torch.full((batch_size,), min(step, self.cfg.diffusion_steps - 1), device=device, dtype=torch.long)
            logits = self.net(x, t)
            probs = F.softmax(logits / temperature, dim=-1)
            conf, pred = probs.max(dim=-1)  # (B,T)
            accept = (conf >= confidence_threshold) & masked
            # ensure progress per sample
            for b in range(batch_size):
                if masked[b].any() and not accept[b].any():
                    mm = conf[b].masked_fill(~masked[b], -float("inf"))
                    j = torch.argmax(mm)
                    accept[b, j] = True
            x = torch.where(accept, pred, x)
            masked = masked & ~accept
        return x

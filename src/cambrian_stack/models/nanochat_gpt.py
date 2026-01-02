"""Nanochat-style GPT model with rotary embeddings and RMSNorm (no params).

Features aligned with karpathy/nanochat:
- rotary embeddings (no positional embeddings)
- RMSNorm without learnable params
- QK norm
- ReLU^2 MLP
- no bias in linear layers
- logit softcap
- zero init of output projections
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger

from cambrian_stack.models.base import BaseModel


def rms_norm(x: Tensor) -> Tensor:
    # Pure functional RMSNorm (no learnable params)
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to last dimension of x.

    x shape: (B, T, H, D)
    cos/sin shape: (1, T, 1, D/2)
    """
    assert x.ndim == 4
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], dim=-1)
    return out.to(dtype=x.dtype)


@dataclass
class NanochatGPTConfig:
    depth: int
    max_seq_len: int
    vocab_size: int
    dropout: float = 0.0
    n_kv_head: int | None = None
    rope_base: int = 10000
    softcap: float = 15.0

    @property
    def d_model(self) -> int:
        return self.depth * 64

    @property
    def n_heads(self) -> int:
        return max(1, (self.d_model + 127) // 128)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_ff(self) -> int:
        return 4 * self.d_model


class CausalSelfAttention(nn.Module):
    def __init__(self, config: NanochatGPTConfig):
        super().__init__()
        self.n_head = config.n_heads
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_heads
        self.n_embd = config.d_model
        self.head_dim = config.head_dim

        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        assert self.head_dim % 2 == 0, "Rotary head_dim must be even"

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape

        q = self.c_q(x).view(bsz, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(bsz, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(bsz, seq_len, self.n_kv_head, self.head_dim)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_kv_head != self.n_head:
            repeat_factor = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        out = self.c_proj(attn)
        return self.dropout(out)


class MLP(nn.Module):
    def __init__(self, config: NanochatGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: NanochatGPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(rms_norm(x), cos, sin)
        x = x + self.mlp(rms_norm(x))
        return x


class NanochatGPT(BaseModel):
    def __init__(self, config: NanochatGPTConfig):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.depth)])
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.rotary_seq_len = config.max_seq_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, config.head_dim, config.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        self._zero_last_layers()

        logger.info(
            f"Initialized NanochatGPT: depth={config.depth}, d_model={config.d_model}, "
            f"n_heads={config.n_heads}, params={self.get_num_params()/1e6:.1f}M"
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _zero_last_layers(self) -> None:
        nn.init.zeros_(self.lm_head.weight)
        for block in self.blocks:
            nn.init.zeros_(block.mlp.c_proj.weight)
            nn.init.zeros_(block.attn.c_proj.weight)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int = 10000):
        device = self.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: Tensor, targets: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        batch_size, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        if seq_len > self.cos.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds rotary cache {self.cos.size(1)}")

        cos = self.cos[:, :seq_len]
        sin = self.sin[:, :seq_len]

        h = self.wte(x)
        h = rms_norm(h)
        for block in self.blocks:
            h = block(h, cos, sin)
        h = rms_norm(h)

        logits = self.lm_head(h)
        logits = logits.float()
        softcap = self.config.softcap
        if softcap > 0:
            logits = softcap * torch.tanh(logits / softcap)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.view(-1),
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, x: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> Tensor:
        generated = x
        for _ in range(max_new_tokens):
            if generated.size(1) > self.max_seq_len:
                generated = generated[:, -self.max_seq_len :]

            logits = self(generated)
            logits = logits[:, -1, :]

            if top_k is not None and top_k > 0:
                top_values, _ = torch.topk(logits, top_k)
                kth_values = top_values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < kth_values, torch.full_like(logits, -float("inf")), logits)

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

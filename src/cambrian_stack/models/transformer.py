"""Simple GPT-style transformer following Karpathy's architecture formula.

This is the BASELINE model. To experiment with a new architecture:
1. Copy this file as a starting point
2. Modify the architecture
3. Keep the same interface (forward, generate, get_num_params)
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger

from cambrian_stack.models.base import BaseModel


@dataclass
class TransformerConfig:
    """Model configuration derived from depth."""
    depth: int
    max_seq_len: int
    vocab_size: int
    dropout: float = 0.0
    
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
    """Multi-head causal self-attention."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv(x)  # (B, T, 3*d_model)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )  # (B, heads, T, head_dim)
        
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.proj(attn)
        out = self.dropout(out)
        return out


class MLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class Block(nn.Module):
    """Transformer block: attention + MLP with residual connections."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(BaseModel):
    """GPT-style autoregressive transformer."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.depth)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.wte.weight
        
        self.apply(self._init_weights)
        logger.info(
            f"Initialized Transformer: depth={config.depth}, d_model={config.d_model}, "
            f"n_heads={config.n_heads}, params={self.get_num_params()/1e6:.1f}M"
        )
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x: Tensor, targets: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        batch_size, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        
        pos = torch.arange(0, seq_len, device=x.device, dtype=torch.long)
        tok_emb = self.wte(x)
        pos_emb = self.wpe(pos)[None, :, :]
        h = self.drop(tok_emb + pos_emb)
        
        for block in self.blocks:
            h = block(h)
        
        h = self.ln_f(h)
        logits = self.lm_head(h)
        
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
            logits = logits[:, -1, :]  # last position
            
            if temperature <= 0:
                raise ValueError("Temperature must be positive.")
            logits = logits / temperature
            
            if top_k is not None and top_k > 0:
                top_values, _ = torch.topk(logits, top_k)
                kth_values = top_values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < kth_values, torch.full_like(logits, -float("inf")), logits)
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated

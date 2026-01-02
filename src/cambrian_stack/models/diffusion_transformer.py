"""Discrete diffusion transformer for text generation.

Key differences vs autoregressive Transformer:
- Bidirectional attention (no causal mask)
- Timestep conditioning
- Parallel denoising sampling instead of left-to-right
"""
from dataclasses import dataclass


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger

from cambrian_stack.models.base import BaseModel


@dataclass
class DiffusionTransformerConfig:
    """Configuration for diffusion transformer."""
    
    depth: int
    max_seq_len: int
    vocab_size: int
    diffusion_steps: int = 128
    mask_token_id: int = 0
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


class BidirectionalAttention(nn.Module):
    """Multi-head bidirectional self-attention."""
    
    def __init__(self, config: DiffusionTransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.dropout(self.proj(attn))


class MLP(nn.Module):
    """Feed-forward network with ReLU^2 activation."""
    
    def __init__(self, config: DiffusionTransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.relu(x).square()
        x = self.fc2(x)
        return self.dropout(x)


class Block(nn.Module):
    """Pre-norm transformer block."""
    
    def __init__(self, config: DiffusionTransformerConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.d_model)
        self.norm2 = nn.RMSNorm(config.d_model)
        self.attn = BidirectionalAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DiffusionTransformer(BaseModel):
    """Discrete diffusion transformer for text."""
    
    def __init__(self, config: DiffusionTransformerConfig):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.time_emb = nn.Embedding(config.diffusion_steps, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.depth)])
        self.norm_f = nn.RMSNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        self._zero_last_layers()
        
        logger.info(
            f"Initialized DiffusionTransformer: depth={config.depth}, "
            f"d_model={config.d_model}, diffusion_steps={config.diffusion_steps}, "
            f"params={self.get_num_params()/1e6:.1f}M"
        )
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)
    
    def _zero_last_layers(self) -> None:
        """Zero-init output projections for stability early in training."""
        nn.init.zeros_(self.output_head.weight)
        for block in self.blocks:
            nn.init.zeros_(block.mlp.fc2.weight)
            nn.init.zeros_(block.attn.proj.weight)
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        targets: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward diffusion denoising step."""
        batch, seq_len = x_t.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")
        
        x_t = x_t.clamp(min=0, max=self.config.vocab_size - 1)
        timesteps = timesteps.clamp(min=0, max=self.config.diffusion_steps - 1)
        
        pos = torch.arange(seq_len, device=x_t.device)
        h = self.token_emb(x_t) + self.pos_emb(pos)[None, :, :] + self.time_emb(timesteps)[:, None, :]
        h = self.drop(h)
        
        for block in self.blocks:
            h = block(h)
        
        h = self.norm_f(h)
        logits = self.output_head(h)
        
        if targets is None:
            return logits
        
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            targets.view(-1),
        )
        return logits, loss
    
    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        temperature: float = 1.0,
        method: str = "confidence",
        confidence_threshold: float = 0.9,
    ) -> Tensor:
        """Parallel denoising sampling."""
        mask_id = self.config.mask_token_id
        x = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
        is_masked = torch.ones_like(x, dtype=torch.bool)
        
        for step in range(self.config.diffusion_steps):
            if not is_masked.any():
                break
            
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            logits = self.forward(x, t)  # type: ignore[arg-type]
            logits = logits / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)
            confidences, predictions = probs.max(dim=-1)
            
            confidences = confidences.masked_fill(~is_masked, -float("inf"))
            
            if method == "confidence":
                to_decode = (confidences >= confidence_threshold) & is_masked
                for b in range(batch_size):
                    if is_masked[b].any() and not to_decode[b].any():
                        best_idx = confidences[b].argmax()
                        to_decode[b, best_idx] = True
            else:
                n_remaining = is_masked.sum(dim=1)
                n_to_decode = (n_remaining / max(self.config.diffusion_steps - step, 1)).ceil().long()
                to_decode = torch.zeros_like(is_masked)
                for b in range(batch_size):
                    if n_remaining[b] > 0:
                        _, top_idx = confidences[b].topk(min(n_to_decode[b].item(), n_remaining[b].item()))
                        to_decode[b, top_idx] = True
            
            x = torch.where(to_decode, predictions, x)
            is_masked = is_masked & ~to_decode
        
        return x
    
    @torch.inference_mode()
    def generate(self, x: Tensor, max_new_tokens: int, **kwargs) -> Tensor:
        """Compatibility wrapper."""
        return self.sample(
            batch_size=x.size(0),
            seq_len=max_new_tokens,
            device=x.device,
            **kwargs,
        )

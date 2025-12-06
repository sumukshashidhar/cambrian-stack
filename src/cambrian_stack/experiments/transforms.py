"""Reusable data transforms for experiments."""
from __future__ import annotations

import torch


def corrupt_tokens(
    tokens: torch.Tensor,
    mask_token_id: int,
    corruption_rate: float,
    diffusion_steps: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Corrupt tokens for diffusion training."""
    batch, seq_len = tokens.shape
    device = tokens.device
    
    mask = torch.rand(batch, seq_len, device=device) < corruption_rate
    corrupted = torch.where(mask, mask_token_id, tokens)
    
    mask_ratio = mask.float().mean(dim=1)
    timesteps = (mask_ratio * (diffusion_steps - 1)).long()
    timesteps = timesteps.clamp(min=0, max=diffusion_steps - 1)
    
    return corrupted, tokens, timesteps


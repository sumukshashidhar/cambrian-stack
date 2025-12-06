"""Tests for diffusion data corruption utilities."""
import torch

from cambrian_stack.experiments.transforms import corrupt_tokens


def test_corrupt_tokens_masks_fraction():
    tokens = torch.arange(10).unsqueeze(0)
    corrupted, targets, timesteps = corrupt_tokens(tokens, mask_token_id=999, corruption_rate=0.5, diffusion_steps=10)
    
    assert corrupted.shape == tokens.shape
    assert targets is tokens
    assert timesteps.shape == (1,)
    # Ensure some masks applied
    assert (corrupted == 999).any()


def test_corrupt_tokens_no_mask_when_zero_rate():
    tokens = torch.arange(6).unsqueeze(0)
    corrupted, targets, timesteps = corrupt_tokens(tokens, mask_token_id=7, corruption_rate=0.0, diffusion_steps=5)
    
    assert torch.equal(corrupted, tokens)
    assert timesteps.item() == 0

"""Tests for diffusion transformer model."""
import torch
import pytest
from omegaconf import OmegaConf

from cambrian_stack.models import create_model, MODEL_REGISTRY, BaseModel
from cambrian_stack.models.diffusion_transformer import DiffusionTransformer, DiffusionTransformerConfig


@pytest.fixture
def tiny_diffusion_cfg():
    return OmegaConf.create({
        "type": "diffusion_transformer",
        "depth": 2,
        "max_seq_len": 32,
        "vocab_size": 500,
        "diffusion_steps": 8,
        "mask_token_id": 0,
        "dropout": 0.0,
    })


def test_registry_contains_diffusion():
    assert "diffusion_transformer" in MODEL_REGISTRY
    assert issubclass(MODEL_REGISTRY["diffusion_transformer"], BaseModel)


def test_config_derivations():
    cfg = DiffusionTransformerConfig(depth=2, max_seq_len=32, vocab_size=500)
    assert cfg.d_model == 128
    assert cfg.n_heads == 1
    assert cfg.d_ff == 512


def test_forward_and_loss(tiny_diffusion_cfg):
    model = create_model(tiny_diffusion_cfg)
    batch_size = 2
    seq_len = tiny_diffusion_cfg.max_seq_len
    x_t = torch.randint(0, tiny_diffusion_cfg.vocab_size, (batch_size, seq_len))
    t = torch.randint(0, tiny_diffusion_cfg.diffusion_steps, (batch_size,))
    logits, loss = model(x_t, t, x_t)
    assert logits.shape == (batch_size, seq_len, tiny_diffusion_cfg.vocab_size)
    assert loss.ndim == 0


def test_generate(tiny_diffusion_cfg):
    model = create_model(tiny_diffusion_cfg)
    x = torch.zeros((1, 4), dtype=torch.long)
    out = model.generate(x, max_new_tokens=8)
    assert out.shape == (1, 8)


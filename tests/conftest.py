"""Pytest fixtures for all tests.

All tests run on CPU by default for speed and CI compatibility.
"""
import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def device():
    """Always CPU for fast, CI-compatible tests."""
    return torch.device("cpu")


@pytest.fixture
def tiny_model_cfg():
    """Minimal model config for fast tests (~200K params)."""
    return OmegaConf.create({
        "type": "transformer",
        "depth": 2,
        "max_seq_len": 32,
        "vocab_size": 1000,
        "dropout": 0.0,
    })


@pytest.fixture
def tiny_training_cfg():
    """Minimal training config for tests."""
    return OmegaConf.create({
        "max_steps": 100,
        "learning_rate": 6e-4,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "warmup_ratio": 0.1,
        "warmdown_ratio": 0.2,
        "final_lr_frac": 0.0,
        "device_batch_size": 2,
        "total_batch_size": 64,
        "eval_every": 50,
        "eval_batches": 2,
        "sample_every": 50,
        "save_every": 100,
    })


@pytest.fixture
def sample_batch(tiny_model_cfg, device):
    """Sample input batch for testing."""
    batch_size = 2
    seq_len = tiny_model_cfg.max_seq_len
    vocab_size = tiny_model_cfg.vocab_size
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    return {"input_ids": x, "labels": y}


@pytest.fixture
def mock_wandb(mocker):
    """Mock wandb for tests - no network calls."""
    mock_init = mocker.patch("wandb.init")
    mock_run = mocker.MagicMock()
    mock_init.return_value = mock_run
    return mock_run


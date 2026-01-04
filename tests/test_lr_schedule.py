"""Unit tests for learning rate schedule helpers."""
import math

import pytest
from omegaconf import OmegaConf

from cambrian_stack.training.trainer import get_lr_multiplier


def make_cfg(**overrides):
    """Create a minimal config object for scheduler tests."""
    training_defaults = {
        "max_steps": 100,
        "learning_rate": 1e-3,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "warmup_ratio": 0.1,
        "warmdown_ratio": 0.2,
        "final_lr_frac": 0.1,
        "device_batch_size": 2,
        "total_batch_size": 64,
        "eval_every": 50,
        "eval_batches": 2,
        "sample_every": 50,
        "save_every": 100,
    }
    training_defaults.update(overrides)
    return OmegaConf.create({"training": training_defaults})


def test_warmdown_reaches_final_fraction():
    cfg = make_cfg(final_lr_frac=0.05, warmdown_ratio=0.3, max_steps=20)
    last_step = cfg.training.max_steps - 1
    multiplier = get_lr_multiplier(last_step, cfg)
    assert math.isclose(multiplier, cfg.training.final_lr_frac, rel_tol=0, abs_tol=1e-6)


@pytest.mark.parametrize("step", [0, 4, 9])
def test_warmup_monotonic(step):
    cfg = make_cfg(max_steps=10, warmup_ratio=0.5, warmdown_ratio=0.0)
    multiplier = get_lr_multiplier(step, cfg)
    assert 0.0 <= multiplier <= 1.0
    if step > 0:
        prev = get_lr_multiplier(step - 1, cfg)
        assert multiplier >= prev

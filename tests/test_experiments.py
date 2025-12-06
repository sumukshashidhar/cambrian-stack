"""Tests for experiment registry and hooks."""
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from cambrian_stack.experiments import create_experiment, EXPERIMENT_REGISTRY
from cambrian_stack.experiments.autoregressive import AutoregressiveExperiment
from cambrian_stack.experiments.diffusion import DiffusionExperiment


class DummyAccelerator:
    def __init__(self, device: torch.device):
        self.device = device
        self.num_processes = 1
        self.is_main_process = True
    
    def backward(self, loss):
        loss.backward()
    
    def unwrap_model(self, model):
        return model
    
    def clip_grad_norm_(self, params, max_norm):
        torch.nn.utils.clip_grad_norm_(params, max_norm)
    
    def gather(self, tensor):
        return tensor.unsqueeze(0)


def test_registry_contains_default_experiments():
    assert "autoregressive" in EXPERIMENT_REGISTRY
    assert "diffusion" in EXPERIMENT_REGISTRY


def test_create_experiment_returns_correct_type():
    ar = create_experiment(OmegaConf.create({"type": "autoregressive"}))
    diff = create_experiment(OmegaConf.create({"type": "diffusion"}))
    assert isinstance(ar, AutoregressiveExperiment)
    assert isinstance(diff, DiffusionExperiment)


def test_autoregressive_training_step_backward(tiny_model_cfg, sample_batch, device):
    exp = AutoregressiveExperiment(OmegaConf.create({"type": "autoregressive"}))
    model = exp.build_model(tiny_model_cfg)
    accelerator = DummyAccelerator(device)
    loss = exp.training_step(model, sample_batch, OmegaConf.create({"training": {}}), accelerator, grad_accum_steps=1)
    assert loss.ndim == 0
    for _, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_diffusion_training_step_backward(device):
    cfg = OmegaConf.create({
        "data": {"corruption_rate": 0.5},
        "training": {},
        "model": {
            "type": "diffusion_transformer",
            "depth": 2,
            "max_seq_len": 8,
            "vocab_size": 50,
            "diffusion_steps": 4,
            "mask_token_id": 0,
            "dropout": 0.0,
        },
    })
    exp = DiffusionExperiment(OmegaConf.create({"type": "diffusion"}))
    model = exp.build_model(cfg.model)
    batch = {
        "input_ids": torch.randint(0, cfg.model.vocab_size, (2, cfg.model.max_seq_len), device=device),
    }
    accelerator = DummyAccelerator(device)
    loss = exp.training_step(model, batch, cfg, accelerator, grad_accum_steps=1)
    assert loss.ndim == 0
    for _, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None

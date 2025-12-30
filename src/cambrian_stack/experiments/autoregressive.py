"""Autoregressive (causal LM) experiment wiring."""
from __future__ import annotations

import torch
from omegaconf import DictConfig

from cambrian_stack.experiments.base import Experiment
from cambrian_stack.models import create_model, BaseModel
from cambrian_stack.data_loaders import get_dataloaders
from cambrian_stack.evaluation.metrics import evaluate_loss, generate_samples, EVAL_PROMPTS


class AutoregressiveExperiment(Experiment):
    """Standard causal language modeling."""
    
    def build_model(self, model_cfg: DictConfig) -> BaseModel:
        return create_model(model_cfg)
    
    def build_dataloaders(self, cfg: DictConfig, accelerator):
        return get_dataloaders(cfg, accelerator=accelerator)
    
    def training_step(self, model, batch, cfg: DictConfig, accelerator, grad_accum_steps: int):
        x = batch["input_ids"].to(accelerator.device)
        y = batch["labels"].to(accelerator.device)
        _, loss = model(x, y)
        loss = loss / grad_accum_steps
        accelerator.backward(loss)
        return loss
    
    def evaluate(self, model, val_loader, cfg: DictConfig, device, accelerator) -> dict[str, float]:
        val_loss, val_ppl = evaluate_loss(model, val_loader, cfg.training.eval_batches, device, accelerator)
        return {
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
        }
    
    @torch.no_grad()
    def sample(self, base_model, tokenizer, cfg: DictConfig, device):
        samples = generate_samples(
            base_model,
            tokenizer,
            EVAL_PROMPTS,
            max_tokens=cfg.training.sample_tokens if hasattr(cfg.training, "sample_tokens") else 50,
            temperature=cfg.training.get("sample_temperature", 0.8) if hasattr(cfg, "training") else 0.8,
            device=device,
        )
        return samples


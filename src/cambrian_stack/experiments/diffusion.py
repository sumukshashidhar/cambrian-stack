"""Diffusion experiment wiring."""
from __future__ import annotations

import torch
from omegaconf import DictConfig

from cambrian_stack.experiments.base import Experiment
from cambrian_stack.experiments.transforms import corrupt_tokens
from cambrian_stack.models import create_model, BaseModel
from cambrian_stack.data_loaders import get_dataloaders


class DiffusionExperiment(Experiment):
    """Discrete diffusion masked-token recovery."""
    
    def build_model(self, model_cfg: DictConfig) -> BaseModel:
        return create_model(model_cfg)
    
    def build_dataloaders(self, cfg: DictConfig, accelerator):
        return get_dataloaders(cfg, accelerator=accelerator)
    
    def training_step(self, model, batch, cfg: DictConfig, accelerator, grad_accum_steps: int):
        base_model = accelerator.unwrap_model(model)
        tokens = batch["input_ids"].to(accelerator.device)
        corruption_rate = getattr(cfg.data, "corruption_rate", 0.15)
        
        corrupted, targets, timesteps = corrupt_tokens(
            tokens,
            mask_token_id=base_model.config.mask_token_id,
            corruption_rate=corruption_rate,
            diffusion_steps=base_model.config.diffusion_steps,
        )
        
        logits, loss = model(corrupted, timesteps, targets)
        loss = loss / grad_accum_steps
        accelerator.backward(loss)
        return loss
    
    def evaluate(self, model, val_loader, cfg: DictConfig, device, accelerator) -> dict[str, float]:
        base_model = accelerator.unwrap_model(model)
        model_was_training = model.training
        model.eval()
        total_loss = 0.0
        total_masked_acc = 0.0
        batches = 0
        corruption_rate = getattr(cfg.data, "corruption_rate", 0.15)
        
        with torch.no_grad():
            for batches, batch in enumerate(val_loader, start=1):
                if batches > cfg.training.eval_batches:
                    batches -= 1
                    break
                tokens = batch["input_ids"].to(device)
                corrupted, targets, timesteps = corrupt_tokens(
                    tokens,
                    mask_token_id=base_model.config.mask_token_id,
                    corruption_rate=corruption_rate,
                    diffusion_steps=base_model.config.diffusion_steps,
                )
                logits, loss = model(corrupted, timesteps, targets)
                total_loss += loss.item()
                
                preds = logits.argmax(dim=-1)
                mask = corrupted.eq(base_model.config.mask_token_id)
                masked_acc = ((preds == targets) & mask).float().sum() / mask.sum().clamp(min=1)
                total_masked_acc += masked_acc.item()
        
        if model_was_training:
            model.train()
        
        avg_loss = total_loss / max(batches, 1)
        avg_masked_acc = total_masked_acc / max(batches, 1)
        
        return {
            "val_loss": avg_loss,
            "val_masked_accuracy": avg_masked_acc,
        }

    @torch.no_grad()
    def sample(self, base_model, tokenizer, cfg: DictConfig, device):
        sample_tokens = getattr(cfg.training, "sample_tokens", 64)
        sample_temperature = cfg.training.get("sample_temperature", 0.8) if hasattr(cfg, "training") else 0.8
        samples = base_model.sample(
            batch_size=4,
            seq_len=sample_tokens,
            device=device,
            temperature=sample_temperature,
        )
        return [tokenizer.decode(s, skip_special_tokens=True) for s in samples]

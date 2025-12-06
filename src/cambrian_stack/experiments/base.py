"""Base experiment interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from omegaconf import DictConfig


class Experiment(ABC):
    """Contract for plugging different training objectives into the core loop."""
    
    def __init__(self, exp_cfg: DictConfig):
        self.exp_cfg = exp_cfg
    
    @abstractmethod
    def build_model(self, model_cfg: DictConfig):
        """Create and return the model instance."""
    
    @abstractmethod
    def build_dataloaders(self, cfg: DictConfig, accelerator):
        """Return (train_loader, val_loader, tokenizer)."""
    
    @abstractmethod
    def training_step(self, model, batch, cfg: DictConfig, accelerator, grad_accum_steps: int):
        """Run one micro-step and accumulate gradients via accelerator.backward."""
    
    @abstractmethod
    def evaluate(self, model, val_loader, cfg: DictConfig, device, accelerator) -> dict[str, float]:
        """Return validation metrics."""
    
    def supports_sampling(self) -> bool:
        """Whether this experiment implements sample logging."""
        return hasattr(self, "sample")
    

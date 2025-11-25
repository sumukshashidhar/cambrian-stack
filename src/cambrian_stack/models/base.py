"""Base model interface - all models must implement this.

To add a new model architecture:
1. Create a new file in models/ (e.g., diffusion_transformer.py)
2. Subclass BaseModel and implement all abstract methods
3. Register in models/__init__.py MODEL_REGISTRY
4. Create a config file in configs/

That's it. No changes to trainer, data loading, or evaluation needed.
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def forward(self, x: Tensor, targets: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass.
        
        Args:
            x: Input token ids, shape (batch, seq_len)
            targets: Target token ids, shape (batch, seq_len), optional
        
        Returns:
            If targets is None: logits of shape (batch, seq_len, vocab_size)
            If targets provided: tuple of (logits, loss)
        """
        pass
    
    @abstractmethod
    def generate(self, x: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> Tensor:
        """Autoregressive generation."""
        pass
    
    @abstractmethod
    def get_num_params(self) -> int:
        """Return total number of parameters."""
        pass


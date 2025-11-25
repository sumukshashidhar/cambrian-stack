"""Tests for model implementations.

All tests use tiny models on CPU for speed.
"""
import pytest
import torch
from omegaconf import OmegaConf

from cambrian_stack.models import create_model, MODEL_REGISTRY, BaseModel
from cambrian_stack.models.transformer import Transformer, TransformerConfig


class TestTransformerConfig:
    def test_dimension_derivation(self):
        config = TransformerConfig(depth=12, max_seq_len=1024, vocab_size=50257)
        assert config.d_model == 768
        assert config.n_heads == 6
        assert config.d_ff == 3072
    
    def test_small_config(self):
        config = TransformerConfig(depth=2, max_seq_len=32, vocab_size=1000)
        assert config.d_model == 128
        assert config.n_heads == 1


class TestTransformer:
    def test_instantiation(self, tiny_model_cfg):
        model = create_model(tiny_model_cfg)
        assert model is not None
        assert isinstance(model, BaseModel)
    
    def test_forward_logits_only(self, tiny_model_cfg, sample_batch):
        model = create_model(tiny_model_cfg)
        x = sample_batch["input_ids"]
        
        logits = model(x)
        
        assert logits.shape == (2, 32, 1000)
    
    def test_forward_with_loss(self, tiny_model_cfg, sample_batch):
        model = create_model(tiny_model_cfg)
        x = sample_batch["input_ids"]
        y = sample_batch["labels"]
        
        logits, loss = model(x, y)
        
        assert logits.shape == (2, 32, 1000)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() > 0
    
    def test_gradient_flow(self, tiny_model_cfg, sample_batch):
        model = create_model(tiny_model_cfg)
        x = sample_batch["input_ids"]
        y = sample_batch["labels"]
        
        _, loss = model(x, y)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_generate(self, tiny_model_cfg, device):
        model = create_model(tiny_model_cfg)
        x = torch.randint(0, 1000, (1, 4), device=device)
        
        out = model.generate(x, max_new_tokens=8, temperature=1.0)
        
        assert out.shape == (1, 12)
    
    def test_param_count(self, tiny_model_cfg):
        model = create_model(tiny_model_cfg)
        params = model.get_num_params()
        
        assert params > 0
        assert params < 10_000_000  # Tiny model < 10M


class TestModelRegistry:
    def test_transformer_in_registry(self):
        assert "transformer" in MODEL_REGISTRY
    
    def test_all_models_inherit_base(self):
        for name, model_class in MODEL_REGISTRY.items():
            assert issubclass(model_class, BaseModel), f"{name} must inherit BaseModel"
    
    def test_create_with_type(self, tiny_model_cfg):
        model = create_model(tiny_model_cfg)
        assert isinstance(model, Transformer)
    
    def test_unknown_type_raises(self):
        cfg = OmegaConf.create({"type": "nonexistent", "depth": 2, "max_seq_len": 32, "vocab_size": 1000})
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model(cfg)


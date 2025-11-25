"""Model registry - add new models here."""
from cambrian_stack.models.base import BaseModel
from cambrian_stack.models.transformer import Transformer, TransformerConfig
from cambrian_stack.models.diffusion_transformer import DiffusionTransformer, DiffusionTransformerConfig

# ============================================================================
# MODEL REGISTRY - Add new models here
# ============================================================================
MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "transformer": Transformer,
    "diffusion_transformer": DiffusionTransformer,
}

CONFIG_REGISTRY: dict[str, type] = {
    "transformer": TransformerConfig,
    "diffusion_transformer": DiffusionTransformerConfig,
}
# ============================================================================


def create_model(model_cfg) -> BaseModel:
    """Factory function to create any registered model from config."""
    model_type = model_cfg.get("type", "transformer")
    
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    
    model_class = MODEL_REGISTRY[model_type]
    config_class = CONFIG_REGISTRY[model_type]
    
    config_kwargs = {k: v for k, v in model_cfg.items() if k != "type"}
    config = config_class(**config_kwargs)
    
    return model_class(config)


__all__ = [
    "BaseModel",
    "Transformer",
    "DiffusionTransformer",
    "create_model",
    "MODEL_REGISTRY",
]

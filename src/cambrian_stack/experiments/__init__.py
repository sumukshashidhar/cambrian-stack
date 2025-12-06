"""Experiment registry and factory."""
from cambrian_stack.experiments.base import Experiment
from cambrian_stack.experiments.autoregressive import AutoregressiveExperiment
from cambrian_stack.experiments.diffusion import DiffusionExperiment

EXPERIMENT_REGISTRY: dict[str, type[Experiment]] = {
    "autoregressive": AutoregressiveExperiment,
    "diffusion": DiffusionExperiment,
}


def create_experiment(cfg) -> Experiment:
    """Instantiate an experiment from config."""
    exp_type = cfg.get("type", "autoregressive")
    if exp_type not in EXPERIMENT_REGISTRY:
        available = ", ".join(EXPERIMENT_REGISTRY.keys())
        raise ValueError(f"Unknown experiment type: {exp_type!r}. Available: {available}")
    return EXPERIMENT_REGISTRY[exp_type](cfg)


__all__ = ["Experiment", "AutoregressiveExperiment", "DiffusionExperiment", "EXPERIMENT_REGISTRY", "create_experiment"]

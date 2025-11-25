"""Evaluation utilities."""
from cambrian_stack.evaluation.metrics import evaluate_loss, generate_samples, EVAL_PROMPTS

# Optional diffusion metrics
try:
    from cambrian_stack.evaluation.diffusion_metrics import (
        evaluate_denoising_accuracy,
        evaluate_generation_quality,
        measure_generation_speed,
    )
except Exception:  # pragma: no cover
    evaluate_denoising_accuracy = None
    evaluate_generation_quality = None
    measure_generation_speed = None

__all__ = [
    "evaluate_loss",
    "generate_samples",
    "EVAL_PROMPTS",
    "evaluate_denoising_accuracy",
    "evaluate_generation_quality",
    "measure_generation_speed",
]

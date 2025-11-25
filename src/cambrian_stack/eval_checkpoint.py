"""Evaluate a model checkpoint.

Usage:
    python -m cambrian_stack.eval_checkpoint --checkpoint out/baseline-d12/checkpoint_010000.pt
"""
import argparse
from pathlib import Path

import torch
from loguru import logger
from omegaconf import OmegaConf
from dotenv import load_dotenv

load_dotenv()

from cambrian_stack.models import create_model
from cambrian_stack.data_loaders import get_tokenizer
from cambrian_stack.evaluation import evaluate_loss, generate_samples, EVAL_PROMPTS


def main():
    """Evaluate checkpoint - high level."""
    args = parse_args()
    model, tokenizer, cfg, step = load_checkpoint(args.checkpoint)
    
    device = select_device()
    model = model.to(device)
    
    log_model_info(model, step)
    samples = run_generation(model, tokenizer, device)
    save_results(args.checkpoint, step, model, samples)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    return parser.parse_args()


def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["config"])
    
    model = create_model(cfg.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    tokenizer = get_tokenizer(cfg.data.tokenizer_name)
    step = checkpoint["step"]
    
    return model, tokenizer, cfg, step


def log_model_info(model, step: int):
    logger.info(f"Loaded checkpoint from step {step}")
    logger.info(f"Model: {model.get_num_params() / 1e6:.1f}M params")


def run_generation(model, tokenizer, device) -> list[str]:
    logger.info("Generating samples...")
    samples = generate_samples(model, tokenizer, EVAL_PROMPTS, device=device)
    
    for prompt, sample in zip(EVAL_PROMPTS, samples):
        logger.info(f"\n{'='*60}\nPrompt: {prompt}\n{'='*60}\n{sample}\n")
    
    return samples


def save_results(checkpoint_path: str, step: int, model, samples: list[str]):
    output_path = Path(checkpoint_path).parent / "eval_results.txt"
    
    with open(output_path, "w") as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Step: {step}\n")
        f.write(f"Parameters: {model.get_num_params()}\n\n")
        for prompt, sample in zip(EVAL_PROMPTS, samples):
            f.write(f"Prompt: {prompt}\n{sample}\n\n")
    
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()


"""Evaluate a diffusion model checkpoint.

Usage:
    python -m cambrian_stack.eval_diffusion --checkpoint out/diffusion-d12/checkpoint_010000.pt
"""
import argparse
from pathlib import Path

import torch
from loguru import logger
from omegaconf import OmegaConf
from dotenv import load_dotenv

load_dotenv()

from cambrian_stack.models import create_model
from cambrian_stack.data_loaders import get_tokenizer, get_dataloaders
from cambrian_stack.evaluation.diffusion_metrics import (
    evaluate_denoising_accuracy,
    evaluate_generation_quality,
    measure_generation_speed,
)


def main():
    args = parse_args()
    model, tokenizer, cfg, step = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    logger.info(f"Loaded checkpoint from step {step}")
    logger.info(f"Model params: {model.get_num_params()/1e6:.1f}M")
    logger.info(f"Diffusion steps: {model.config.diffusion_steps}")
    
    _, val_loader, _ = get_dataloaders(cfg)
    
    denoise = evaluate_denoising_accuracy(
        model, val_loader, num_batches=5, device=device, corruption_rate=0.15
    )
    gen = evaluate_generation_quality(
        model, tokenizer, num_samples=4, seq_len=64, device=device
    )
    speed = measure_generation_speed(model, batch_size=2, seq_len=64, num_runs=2, device=device)
    
    log_results(denoise, gen, speed)
    save_results(args.checkpoint, step, denoise, gen, speed)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate diffusion checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def load_checkpoint(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["config"])
    model = create_model(cfg.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    tokenizer = get_tokenizer(cfg.data.tokenizer_name)
    step = checkpoint.get("step", 0)
    return model, tokenizer, cfg, step


def log_results(denoise: dict, gen: dict, speed: dict) -> None:
    logger.info("\n" + "=" * 40)
    logger.info("Diffusion Evaluation Summary")
    logger.info("=" * 40)
    logger.info(f"Denoising accuracy: {denoise['denoising_accuracy']:.3f}")
    logger.info(f"Denoising loss: {denoise['denoising_loss']:.4f}")
    logger.info(f"Unique bigrams: {gen['unique_bigrams']}")
    logger.info(f"Unique trigrams: {gen['unique_trigrams']}")
    logger.info(f"Avg length: {gen['avg_length']:.1f}")
    logger.info(f"Tokens/sec: {speed['tokens_per_second']:.0f}")
    logger.info("=" * 40)
    for i, sample in enumerate(gen["samples"]):
        logger.info(f"\n--- Sample {i+1} ---\n{sample[:400]}...")


def save_results(checkpoint_path: str, step: int, denoise: dict, gen: dict, speed: dict) -> None:
    out_dir = Path(checkpoint_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = out_dir / "eval_results_diffusion.txt"
    with open(results_path, "w") as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Step: {step}\n\n")
        f.write(f"Denoising accuracy: {denoise['denoising_accuracy']:.4f}\n")
        f.write(f"Denoising loss: {denoise['denoising_loss']:.4f}\n")
        f.write(f"Unique bigrams: {gen['unique_bigrams']}\n")
        f.write(f"Unique trigrams: {gen['unique_trigrams']}\n")
        f.write(f"Avg length: {gen['avg_length']:.2f}\n")
        f.write(f"Tokens/sec: {speed['tokens_per_second']:.0f}\n")
    
    samples_path = out_dir / "eval_samples_diffusion.txt"
    with open(samples_path, "w") as f:
        for i, sample in enumerate(gen["samples"]):
            f.write(f"=== Sample {i+1} ===\n{sample}\n\n")
    
    logger.info(f"Saved eval results to {results_path}")
    logger.info(f"Saved samples to {samples_path}")


if __name__ == "__main__":
    main()


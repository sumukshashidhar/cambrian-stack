"""Evaluation metrics for diffusion models."""
from typing import Any
import time
import torch
import torch.nn.functional as F
from torch import Tensor
from loguru import logger


def evaluate_denoising_accuracy(
    model,
    val_loader,
    num_batches: int,
    device: torch.device,
    corruption_rate: float = 0.15,
) -> dict[str, float]:
    """Evaluate masked-token recovery accuracy."""
    model_was_training = model.training
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    
    mask_token_id = model.config.mask_token_id
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_batches:
                break
            
            tokens = batch["input_ids"].to(device)
            mask = torch.rand_like(tokens.float()) < corruption_rate
            corrupted = torch.where(mask, mask_token_id, tokens)
            t = torch.full((tokens.size(0),), int(corruption_rate * (model.config.diffusion_steps - 1)), device=device, dtype=torch.long)
            
            logits, loss = model(corrupted, t, tokens)
            total_loss += loss.item()
            
            predictions = logits.argmax(dim=-1)
            correct = (predictions == tokens) & mask
            total_correct += correct.sum().item()
            total_masked += mask.sum().item()
    
    accuracy = total_correct / max(total_masked, 1)
    avg_loss = total_loss / max(num_batches, 1)
    
    if model_was_training:
        model.train()
    
    return {
        "denoising_accuracy": accuracy,
        "denoising_loss": avg_loss,
        "masked_tokens_evaluated": total_masked,
    }


def evaluate_generation_quality(
    model,
    tokenizer,
    num_samples: int = 8,
    seq_len: int = 64,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Generate samples and compute simple diversity stats."""
    model_was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    
    with torch.no_grad():
        samples = model.sample(
            batch_size=num_samples,
            seq_len=seq_len,
            device=device,
            method="confidence",
            temperature=0.8,
        )
    
    texts = [tokenizer.decode(s, skip_special_tokens=True) for s in samples]
    
    def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
        words = text.split()
        return set(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    
    bigrams = set().union(*[ngrams(t, 2) for t in texts])
    trigrams = set().union(*[ngrams(t, 3) for t in texts])
    
    if model_was_training:
        model.train()
    
    return {
        "samples": texts,
        "unique_bigrams": len(bigrams),
        "unique_trigrams": len(trigrams),
        "avg_length": sum(len(t.split()) for t in texts) / max(len(texts), 1),
    }


def measure_generation_speed(
    model,
    batch_size: int = 4,
    seq_len: int = 64,
    num_runs: int = 3,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Rough throughput measurement for diffusion sampling."""
    model.eval()
    device = device or next(model.parameters()).device
    
    with torch.no_grad():
        _ = model.sample(batch_size=2, seq_len=32, device=device)
    
    times: list[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.sample(batch_size=batch_size, seq_len=seq_len, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    avg = sum(times) / max(len(times), 1)
    tokens = batch_size * seq_len
    
    return {
        "tokens_per_second": tokens / avg if avg > 0 else 0.0,
        "time_per_sample": avg / max(batch_size, 1),
    }


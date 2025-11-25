"""Evaluation metrics and generation utilities."""
import math
import torch
from torch import Tensor
from loguru import logger

from cambrian_stack.models.base import BaseModel


# Standard prompts for qualitative evaluation
EVAL_PROMPTS = [
    "Once upon a time",
    "The little girl",
    "There was a",
    "One day, a",
    "A boy named",
]


@torch.no_grad()
def evaluate_loss(model: BaseModel, dataloader, num_batches: int, device, accelerator=None) -> tuple[float, float]:
    """Compute average loss and perplexity on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader to evaluate on
        num_batches: Number of batches to evaluate
        device: Device to run on
    
    Returns:
        (average_loss, perplexity)
    """
    model_was_training = model.training
    model.eval()
    total_loss = 0.0
    batches_run = 0
    
    with torch.no_grad():
        for batches_run, batch in enumerate(dataloader, start=1):
            if batches_run > num_batches:
                batches_run -= 1
                break
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
    
    if accelerator is not None and accelerator.num_processes > 1:
        total_loss_tensor = torch.tensor(total_loss, device=device)
        count_tensor = torch.tensor(batches_run, device=device, dtype=torch.long)
        gathered_loss = accelerator.gather_for_metrics(total_loss_tensor)
        gathered_count = accelerator.gather_for_metrics(count_tensor)
        total_loss = gathered_loss.sum().item()
        batches_run = gathered_count.sum().item()
    
    if batches_run == 0:
        avg_loss = float("inf")
    else:
        avg_loss = total_loss / batches_run
    perplexity = math.exp(avg_loss) if math.isfinite(avg_loss) else float("inf")
    
    if model_was_training:
        model.train()
    
    return avg_loss, perplexity


@torch.no_grad()
def generate_samples(
    model: BaseModel,
    tokenizer,
    prompts: list[str],
    max_tokens: int = 50,
    temperature: float = 0.8,
    device = None,
) -> list[str]:
    """Generate samples from prompts.
    
    Args:
        model: The model to generate from
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of prompt strings
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to run on
    
    Returns:
        List of generated strings (including prompts)
    """
    model_was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    
    samples: list[str] = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = model.generate(tokens, max_new_tokens=max_tokens, temperature=temperature)
        text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        samples.append(text)
    
    if model_was_training:
        model.train()
    
    return samples

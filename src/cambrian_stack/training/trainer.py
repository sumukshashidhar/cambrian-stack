"""Training loop - top-down style.

The train() function reads like a high-level story.
Each step is a well-named function that can be understood independently.
"""
import math
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import AdamW
from accelerate import Accelerator
from loguru import logger
from omegaconf import DictConfig

from cambrian_stack.models import create_model, BaseModel
from cambrian_stack.data_loaders import get_dataloaders, corrupt_tokens
from cambrian_stack.utils.logging import setup_logging, setup_wandb
from cambrian_stack.evaluation.metrics import evaluate_loss, generate_samples, EVAL_PROMPTS
from omegaconf import OmegaConf


# =============================================================================
# MAIN ENTRY POINT - This is the high-level story
# =============================================================================

def train(cfg: DictConfig) -> None:
    """Train a model on TinyStories.
    
    This function reads like a story - each step is a chapter.
    """
    # Chapter 1: Setup
    accelerator = setup_accelerator()
    logs_dir = Path("logs") / Path(cfg.output.dir).name
    setup_logging(logs_dir)
    wandb_run = setup_wandb(cfg, accelerator.is_main_process)
    
    # Chapter 2: Create components
    model = create_model(cfg.model)
    train_loader, val_loader, tokenizer = get_dataloaders(cfg, accelerator=accelerator)
    optimizer = create_optimizer(model, cfg.training)
    
    # Chapter 3: Prepare for distributed training
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Chapter 4: Calculate training parameters
    grad_accum_steps = calculate_grad_accum_steps(cfg, accelerator)
    
    # Chapter 5: Training loop
    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        tokenizer=tokenizer,
        cfg=cfg,
        accelerator=accelerator,
        wandb_run=wandb_run,
        grad_accum_steps=grad_accum_steps,
    )
    
    # Chapter 6: Cleanup
    wandb_run.finish()
    logger.info("Training complete!")


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_accelerator() -> Accelerator:
    """Initialize accelerator for distributed training."""
    return Accelerator()


# =============================================================================
# DIFFUSION HELPERS
# =============================================================================

def is_diffusion_model(model: BaseModel) -> bool:
    """Check whether model is a diffusion transformer."""
    if hasattr(model, "config") and hasattr(model.config, "diffusion_steps"):
        return True
    if hasattr(model, "module"):
        mod = model.module
        return hasattr(mod, "config") and hasattr(mod.config, "diffusion_steps")
    return False


def diffusion_train_step(model, batch, cfg, device, accelerator, grad_accum_steps: int) -> torch.Tensor:
    """Single micro-step for diffusion training."""
    base_model = accelerator.unwrap_model(model)
    tokens = batch["input_ids"].to(device)
    corruption_rate = getattr(cfg.data, "corruption_rate", 0.15)
    
    corrupted, targets, timesteps = corrupt_tokens(
        tokens,
        mask_token_id=base_model.config.mask_token_id,
        corruption_rate=corruption_rate,
        diffusion_steps=base_model.config.diffusion_steps,
    )
    
    logits, loss = model(corrupted, timesteps, targets)
    loss = loss / grad_accum_steps
    accelerator.backward(loss)
    return loss


def diffusion_eval_step(model, val_loader, cfg, device, accelerator) -> dict[str, float]:
    """Evaluate diffusion model on masked recovery."""
    base_model = accelerator.unwrap_model(model)
    model_was_training = model.training
    model.eval()
    total_loss = 0.0
    total_masked_acc = 0.0
    batches = 0
    
    with torch.no_grad():
        for batches, batch in enumerate(val_loader, start=1):
            if batches > cfg.training.eval_batches:
                batches -= 1
                break
            tokens = batch["input_ids"].to(device)
            corruption_rate = getattr(cfg.data, "corruption_rate", 0.15)
            corrupted, targets, timesteps = corrupt_tokens(
                tokens,
                mask_token_id=base_model.config.mask_token_id,
                corruption_rate=corruption_rate,
                diffusion_steps=base_model.config.diffusion_steps,
            )
            logits, loss = model(corrupted, timesteps, targets)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            mask = corrupted.eq(model.config.mask_token_id)
            masked_acc = ((preds == targets) & mask).float().sum() / mask.sum().clamp(min=1)
            total_masked_acc += masked_acc.item()
    
    avg_loss = total_loss / max(batches, 1)
    avg_masked_acc = total_masked_acc / max(batches, 1)
    
    if model_was_training:
        model.train()
    
    return {
        "val_loss": avg_loss,
        "val_masked_accuracy": avg_masked_acc,
    }


def create_optimizer(model: BaseModel, training_cfg) -> AdamW:
    """Create AdamW optimizer."""
    return AdamW(
        model.parameters(),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
        betas=(0.9, 0.95),
    )


def calculate_grad_accum_steps(cfg: DictConfig, accelerator: Accelerator) -> int:
    """Calculate gradient accumulation steps to reach total_batch_size."""
    tokens_per_micro = cfg.training.device_batch_size * cfg.model.max_seq_len
    tokens_per_step = tokens_per_micro * max(1, accelerator.num_processes)
    if tokens_per_step == 0:
        raise ValueError("tokens_per_step computed as zero")
    grad_accum = math.ceil(cfg.training.total_batch_size / tokens_per_step)
    return max(1, grad_accum)


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def get_lr_multiplier(step: int, cfg: DictConfig) -> float:
    """Cosine decay with optional warmup.
    
    Schedule:
    - Warmup: linear from 0 to 1 over warmup_ratio * max_steps
    - Constant: 1.0
    - Warmdown: cosine decay from 1 to final_lr_frac over warmdown_ratio * max_steps
    """
    max_steps = cfg.training.max_steps
    warmup_steps = int(cfg.training.warmup_ratio * max_steps)
    warmdown_steps = int(cfg.training.warmdown_ratio * max_steps)
    warmdown_start = max_steps - warmdown_steps
    
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(1, warmup_steps)
    
    if warmdown_steps > 0 and step >= warmdown_start:
        progress = (step - warmdown_start) / max(1, warmdown_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return cfg.training.final_lr_frac + cosine * (1 - cfg.training.final_lr_frac)
    
    return 1.0


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_loop(
    model: BaseModel,
    train_loader,
    val_loader,
    optimizer: AdamW,
    tokenizer,
    cfg: DictConfig,
    accelerator: Accelerator,
    wandb_run,
    grad_accum_steps: int,
) -> None:
    """Main training loop."""
    device = accelerator.device
    model.train()
    train_iter = iter(train_loader)
    tokens_per_step = (
        cfg.training.device_batch_size * cfg.model.max_seq_len * max(1, accelerator.num_processes) * grad_accum_steps
    )
    is_diffusion = is_diffusion_model(model)
    
    for step in range(cfg.training.max_steps):
        step_start = time.perf_counter()
        
        # 1. Update learning rate
        lr_mult = get_lr_multiplier(step, cfg)
        base_lr = cfg.training.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr * lr_mult
        
        # 2. Accumulate gradients
        total_loss = torch.tensor(0.0, device=device)
        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            if is_diffusion:
                loss = diffusion_train_step(model, batch, cfg, device, accelerator, grad_accum_steps)
            else:
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)
                _, loss = model(x, y)
                loss = loss / grad_accum_steps
                accelerator.backward(loss)
            total_loss += loss.detach()
        
        # 3. Clip gradients
        accelerator.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        
        # 4. Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Sync losses across processes
        loss_scalar = accelerator.gather(total_loss).mean().item()
        step_time = time.perf_counter() - step_start
        tokens_per_sec = tokens_per_step / max(step_time, 1e-6)
        
        # 5. Log metrics
        if accelerator.is_main_process and (step % cfg.logging.log_every == 0 or step == 0):
            log_payload = {
                "step": step,
                "train/loss": loss_scalar,
                "train/lr": base_lr * lr_mult,
                "train/tokens_per_sec": tokens_per_sec,
            }
            logger.info(
                f"step {step:05d} | loss {loss_scalar:.4f} | lr {base_lr * lr_mult:.2e} | "
                f"tok/s {tokens_per_sec:,.0f}"
            )
            wandb_run.log(log_payload)
        
        # 6. Evaluate
        if cfg.training.eval_every > 0 and (step + 1) % cfg.training.eval_every == 0:
            if is_diffusion:
                val_metrics = diffusion_eval_step(model, val_loader, cfg, device, accelerator)
                if accelerator.is_main_process:
                    logger.info(
                        f"[eval] step {step:05d} | val_loss {val_metrics['val_loss']:.4f} "
                        f"| val_masked_acc {val_metrics['val_masked_accuracy']:.3f}"
                    )
                    wandb_run.log({"step": step, **val_metrics})
            else:
                val_loss, val_ppl = evaluate_loss(model, val_loader, cfg.training.eval_batches, device, accelerator)
                if accelerator.is_main_process:
                    logger.info(f"[eval] step {step:05d} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
                    wandb_run.log({"step": step, "val/loss": val_loss, "val/perplexity": val_ppl})
        
        # 7. Generate samples
        if (
            accelerator.is_main_process
            and cfg.training.sample_every > 0
            and (step + 1) % cfg.training.sample_every == 0
        ):
            if not is_diffusion:
                base_model = accelerator.unwrap_model(model)
                samples = generate_samples(base_model, tokenizer, EVAL_PROMPTS, device=device)
                for prompt, sample in zip(EVAL_PROMPTS, samples):
                    logger.info(f"\n{'='*20} Prompt: {prompt}\n{sample}\n")
                wandb_run.log({f"samples/{i}": s for i, s in enumerate(samples)})
        
        # 8. Save checkpoint
        if cfg.training.save_every > 0 and (step + 1) % cfg.training.save_every == 0:
            save_checkpoint(model, optimizer, step + 1, cfg, accelerator)

    # Final checkpoint
    save_checkpoint(model, optimizer, cfg.training.max_steps, cfg, accelerator)


# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_checkpoint(
    model: BaseModel,
    optimizer: AdamW,
    step: int,
    cfg: DictConfig,
    accelerator: Accelerator,
) -> None:
    """Save model checkpoint."""
    if accelerator.is_main_process:
        checkpoint_dir = Path(cfg.output.dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_{step:06d}.pt"
        
        unwrapped = accelerator.unwrap_model(model)
        state = {
            "step": step,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "model_state_dict": unwrapped.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        accelerator.save(state, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    accelerator.wait_for_everyone()


def load_checkpoint(path: Path, model: BaseModel, optimizer: AdamW | None = None) -> int:
    """Load model checkpoint. Returns the step number."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("step", 0))

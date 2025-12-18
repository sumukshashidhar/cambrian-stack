"""Training loop - top-down style.

The train() function reads like a high-level story.
Each step is a well-named function that can be understood independently.
"""
import math
import time
from pathlib import Path
from typing import List

import torch
from torch.optim import AdamW
from accelerate import Accelerator
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cambrian_stack.utils.logging import setup_logging, setup_wandb
from cambrian_stack.experiments import create_experiment
from cambrian_stack.models import BaseModel
from cambrian_stack.optim import Muon


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
    
    # Chapter 2: Create experiment + components
    exp_cfg = cfg.experiment if "experiment" in cfg else {"type": "autoregressive"}
    experiment = create_experiment(exp_cfg)
    model = experiment.build_model(cfg.model)
    train_loader, val_loader, tokenizer = experiment.build_dataloaders(cfg, accelerator=accelerator)
    optimizers = create_optimizers(model, cfg.training)
    
    # Chapter 3: Prepare for distributed training
    prepared = accelerator.prepare(model, *optimizers, train_loader, val_loader)
    model = prepared[0]
    optimizers = list(prepared[1 : 1 + len(optimizers)])  # type: ignore[assignment]
    train_loader, val_loader = prepared[-2], prepared[-1]
    
    # Chapter 4: Calculate training parameters
    grad_accum_steps = calculate_grad_accum_steps(cfg, accelerator)
    
    # Chapter 5: Training loop
    train_loop(
        experiment=experiment,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
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


def create_optimizers(model: BaseModel, training_cfg) -> List[torch.optim.Optimizer]:
    """Create Muon (matrix params) + AdamW (embedding/positional) optimizers."""
    # Learning rates (fallback to single lr if not provided)
    emb_lr = training_cfg.get("embedding_lr", training_cfg.learning_rate)
    unemb_lr = training_cfg.get("unembedding_lr", training_cfg.learning_rate)
    matrix_lr = training_cfg.get("matrix_lr", training_cfg.learning_rate)
    muon_momentum = training_cfg.get("muon_momentum", 0.95)
    betas = training_cfg.get("adam_betas", (0.8, 0.95))
    adam_eps = training_cfg.get("adam_eps", 1e-10)
    
    # Identify parameter groups (by identity, not equality to avoid tensor broadcast)
    embedding_params: list[torch.nn.Parameter] = []
    for attr in ("wte", "wpe", "lm_head"):
        if hasattr(model, attr):
            embedding_params.extend(list(getattr(model, attr).parameters()))
    embed_ids = {id(p) for p in embedding_params}

    other_params = [p for p in model.parameters() if id(p) not in embed_ids]
    # Muon expects 2D params; route others to AdamW
    muon_params = [p for p in other_params if p.ndim >= 2]
    adam_extra_params = [p for p in other_params if p.ndim < 2]
    embedding_params.extend(adam_extra_params)
    
    # Scale lr by sqrt dimension like nanochat
    d_model = getattr(getattr(model, "config", None), "d_model", 768)
    dmodel_lr_scale = (d_model / 768) ** -0.5
    
    # AdamW on embeddings (and tied lm_head if applicable)
    fused_ok = False
    try:
        AdamW(embedding_params, lr=1e-3, fused=True)
        fused_ok = True
    except Exception:
        fused_ok = False
    adamw = AdamW(
        [
            dict(params=embedding_params, lr=emb_lr * dmodel_lr_scale),
        ],
        lr=emb_lr * dmodel_lr_scale,
        betas=betas,
        eps=adam_eps,
        weight_decay=training_cfg.weight_decay,
        fused=fused_ok,
    )
    for g in adamw.param_groups:
        g["initial_lr"] = g["lr"]
    
    # Muon on matrix / rest
    optimizers: List[torch.optim.Optimizer] = [adamw]
    if len(muon_params) > 0:
        muon = Muon(
            muon_params,
            lr=matrix_lr,
            momentum=muon_momentum,
            nesterov=True,
            ns_steps=5,
        )
        optimizers.append(muon)
    
    return optimizers


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
    experiment,
    model,
    train_loader,
    val_loader,
    optimizers: List[torch.optim.Optimizer],
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
    can_sample = experiment.supports_sampling()
    
    for step in range(cfg.training.max_steps):
        step_start = time.perf_counter()
        
        # 1. Update learning rate
        lr_mult = get_lr_multiplier(step, cfg)
        for opt in optimizers:
            for param_group in opt.param_groups:
                base_lr = param_group.get("initial_lr", param_group["lr"])
                param_group["lr"] = base_lr * lr_mult
        
        # 2. Accumulate gradients
        total_loss = torch.tensor(0.0, device=device)
        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            loss = experiment.training_step(model, batch, cfg, accelerator, grad_accum_steps)
            total_loss += loss.detach()
        
        # 3. Clip gradients
        accelerator.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        
        # 4. Optimizer step
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
        
        # Sync losses across processes
        loss_scalar = accelerator.gather(total_loss).mean().item()
        step_time = time.perf_counter() - step_start
        tokens_per_sec = tokens_per_step / max(step_time, 1e-6)
        
        # 5. Log metrics
        if accelerator.is_main_process and (step % cfg.logging.log_every == 0 or step == 0):
            log_payload = {
                "step": step,
                "train/loss": loss_scalar,
                "train/lr": optimizers[0].param_groups[0]["lr"],
                "train/tokens_per_sec": tokens_per_sec,
            }
            logger.info(
                f"step {step:05d} | loss {loss_scalar:.4f} | lr {base_lr * lr_mult:.2e} | "
                f"tok/s {tokens_per_sec:,.0f}"
            )
            wandb_run.log(log_payload)
        
        # 6. Evaluate
        if cfg.training.eval_every > 0 and (step + 1) % cfg.training.eval_every == 0:
            val_metrics = experiment.evaluate(model, val_loader, cfg, device, accelerator)
            if accelerator.is_main_process:
                formatted = " | ".join(f"{k} {v:.4f}" for k, v in val_metrics.items())
                logger.info(f"[eval] step {step:05d} | {formatted}")
                wandb_run.log({"step": step, **val_metrics})
        
        # 7. Generate samples
        if (
            can_sample
            and accelerator.is_main_process
            and cfg.training.sample_every > 0
            and (step + 1) % cfg.training.sample_every == 0
        ):
            base_model = accelerator.unwrap_model(model)
            samples = experiment.sample(base_model, tokenizer, cfg, device=device)  # type: ignore[attr-defined]
            prompts = getattr(cfg.training, "sample_prompts", None)
            if prompts:
                for prompt, sample in zip(prompts, samples):
                    logger.info(f"\n{'='*20} Prompt: {prompt}\n{sample}\n")
            else:
                for i, sample in enumerate(samples):
                    logger.info(f"\n{'='*20} Sample {i}\n{sample}\n")
            wandb_run.log({f"samples/{i}": s for i, s in enumerate(samples)})
        
        # 8. Save checkpoint
        if cfg.training.save_every > 0 and (step + 1) % cfg.training.save_every == 0:
            save_checkpoint(model, optimizers, step + 1, cfg, accelerator)

    # Final checkpoint
    save_checkpoint(model, optimizers, cfg.training.max_steps, cfg, accelerator)


# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_checkpoint(
    model: BaseModel,
    optimizers: List[torch.optim.Optimizer],
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
            "optimizers_state_dict": [opt.state_dict() for opt in optimizers],
        }
        accelerator.save(state, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    accelerator.wait_for_everyone()


def load_checkpoint(path: Path, model: BaseModel, optimizers: List[torch.optim.Optimizer] | None = None) -> int:
    """Load model checkpoint. Returns the step number."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizers is not None and "optimizers_state_dict" in checkpoint:
        for opt, sd in zip(optimizers, checkpoint["optimizers_state_dict"]):
            opt.load_state_dict(sd)
    return int(checkpoint.get("step", 0))

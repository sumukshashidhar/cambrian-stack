"""Minimal speedrun trainer (nanochat-like, pretraining only)."""
from __future__ import annotations

import math
import time
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cambrian_stack.data_loaders import get_dataloaders
from cambrian_stack.models import create_model
from cambrian_stack.training.trainer import create_optimizers, save_checkpoint
from cambrian_stack.evaluation.metrics import evaluate_loss, generate_samples, EVAL_PROMPTS
from cambrian_stack.utils.logging import setup_logging, setup_wandb
from cambrian_stack.optim import Muon


def setup_accelerator() -> Accelerator:
    return Accelerator(mixed_precision="bf16")


def compute_grad_accum(cfg: DictConfig, accelerator: Accelerator) -> tuple[int, int]:
    tokens_per_micro = cfg.training.device_batch_size * cfg.model.max_seq_len
    tokens_per_step = tokens_per_micro * max(1, accelerator.num_processes)
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step computed as zero")
    total_batch = cfg.training.total_batch_size
    grad_accum = max(1, (total_batch + tokens_per_step - 1) // tokens_per_step)
    effective_tokens = tokens_per_step * grad_accum
    if effective_tokens != total_batch and accelerator.is_main_process:
        logger.warning(
            f"total_batch_size={total_batch} not divisible by tokens_per_step={tokens_per_step}; "
            f"using grad_accum={grad_accum} (effective tokens/step={effective_tokens})"
        )
    return grad_accum, effective_tokens


def compute_max_steps(cfg: DictConfig, model, effective_tokens: int, accelerator: Accelerator) -> int:
    max_steps = int(cfg.training.max_steps)
    target_ratio = float(cfg.training.get("target_param_ratio", -1))
    if max_steps > 0:
        return max_steps
    if target_ratio <= 0:
        raise ValueError("Set training.max_steps > 0 or training.target_param_ratio > 0")
    num_params = accelerator.unwrap_model(model).get_num_params()
    max_steps = max(1, int((target_ratio * num_params) // effective_tokens))
    cfg.training.max_steps = max_steps
    if accelerator.is_main_process:
        logger.info(
            f"Computed max_steps={max_steps} using target_param_ratio={target_ratio} "
            f"(params={num_params:,}, tokens/step={effective_tokens:,})"
        )
    return max_steps


def lr_multiplier(step: int, max_steps: int, cfg: DictConfig) -> float:
    warmup_steps = int(round(cfg.training.warmup_ratio * max_steps))
    warmdown_steps = int(round(cfg.training.warmdown_ratio * max_steps))
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if warmdown_steps > 0 and step >= max_steps - warmdown_steps:
        progress = (max_steps - step) / warmdown_steps
        return progress * 1.0 + (1.0 - progress) * cfg.training.final_lr_frac
    return 1.0


def muon_momentum(step: int, cfg: DictConfig) -> float | None:
    start = cfg.training.get("muon_momentum_start", None)
    end = cfg.training.get("muon_momentum_end", None)
    warmup_steps = cfg.training.get("muon_momentum_warmup_steps", None)
    if start is None or end is None or warmup_steps is None:
        return None
    if warmup_steps <= 0:
        return float(end)
    frac = min(step / warmup_steps, 1.0)
    return float((1 - frac) * start + frac * end)


def train_speedrun(cfg: DictConfig) -> None:
    accelerator = setup_accelerator()
    logs_dir = Path("logs") / Path(cfg.output.dir).name
    setup_logging(logs_dir)
    wandb_run = setup_wandb(cfg, accelerator.is_main_process)

    train_loader, val_loader, tokenizer = get_dataloaders(cfg, accelerator=accelerator)
    if hasattr(tokenizer, "vocab_size") and cfg.model.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size={tokenizer.vocab_size} does not match model.vocab_size={cfg.model.vocab_size}. "
            "Set model.vocab_size to match the tokenizer or use a tokenizer with the desired vocab size."
        )

    model = create_model(cfg.model)
    optimizers = create_optimizers(model, cfg.training)

    prepared = accelerator.prepare(model, *optimizers, train_loader, val_loader)
    model = prepared[0]
    optimizers = list(prepared[1 : 1 + len(optimizers)])
    train_loader, val_loader = prepared[-2], prepared[-1]

    grad_accum_steps, effective_tokens = compute_grad_accum(cfg, accelerator)
    max_steps = compute_max_steps(cfg, model, effective_tokens, accelerator)

    device = accelerator.device
    model.train()
    train_iter = iter(train_loader)

    for step in range(max_steps):
        step_start = time.perf_counter()

        lr_mult = lr_multiplier(step, max_steps, cfg)
        for opt in optimizers:
            for param_group in opt.param_groups:
                base_lr = param_group.get("initial_lr", param_group["lr"])
                param_group["lr"] = base_lr * lr_mult

        mom = muon_momentum(step, cfg)
        if mom is not None:
            for opt in optimizers:
                if isinstance(opt, Muon):
                    for param_group in opt.param_groups:
                        param_group["momentum"] = mom

        total_loss = torch.tensor(0.0, device=device)
        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            _, loss = model(x, y)
            loss = loss / grad_accum_steps
            accelerator.backward(loss)
            total_loss += loss.detach()

        accelerator.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        for opt in optimizers:
            opt.step()
            opt.zero_grad()

        loss_scalar = accelerator.gather(total_loss).mean().item()
        step_time = time.perf_counter() - step_start
        tokens_per_sec = effective_tokens / max(step_time, 1e-6)

        if accelerator.is_main_process and (step % cfg.logging.log_every == 0 or step == 0):
            adam_lr = optimizers[0].param_groups[0]["lr"]
            muon_lr = next((pg["lr"] for opt in optimizers if isinstance(opt, Muon) for pg in opt.param_groups), adam_lr)
            logger.info(
                f"step {step:05d}/{max_steps} | loss {loss_scalar:.4f} | lr_adam {adam_lr:.2e} "
                f"| lr_muon {muon_lr:.2e} | tok/s {tokens_per_sec:,.0f}"
            )
            wandb_run.log(
                {
                    "step": step,
                    "train/loss": loss_scalar,
                    "train/lr_adam": adam_lr,
                    "train/lr_muon": muon_lr,
                    "train/tokens_per_sec": tokens_per_sec,
                }
            )

        if cfg.training.eval_every > 0 and (step + 1) % cfg.training.eval_every == 0:
            val_loss, val_ppl = evaluate_loss(model, val_loader, cfg.training.eval_batches, device, accelerator)
            if accelerator.is_main_process:
                logger.info(f"[eval] step {step:05d} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
                wandb_run.log({"step": step, "val_loss": val_loss, "val_perplexity": val_ppl})

        if cfg.training.sample_every > 0 and (step + 1) % cfg.training.sample_every == 0 and accelerator.is_main_process:
            base_model = accelerator.unwrap_model(model)
            samples = generate_samples(base_model, tokenizer, EVAL_PROMPTS, max_tokens=cfg.training.sample_tokens, device=device)
            for i, sample in enumerate(samples):
                logger.info(f"\n{'='*20} Sample {i}\n{sample}\n")
            wandb_run.log({f"samples/{i}": s for i, s in enumerate(samples)})

        if cfg.training.save_every > 0 and (step + 1) % cfg.training.save_every == 0:
            save_checkpoint(model, optimizers, step + 1, cfg, accelerator)

    save_checkpoint(model, optimizers, max_steps, cfg, accelerator)
    wandb_run.finish()
    if accelerator.is_main_process:
        logger.info("Speedrun training complete!")


@hydra.main(version_base=None, config_path="conf", config_name="speedrun")
def main(cfg: DictConfig) -> None:
    train_speedrun(cfg)


if __name__ == "__main__":
    main()

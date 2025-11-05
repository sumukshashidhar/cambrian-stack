from __future__ import annotations

from pathlib import Path
import typer

from cambrian.train_transformer import main as train_transformer_main
from cambrian.train_diffusion import main as train_diffusion_main  # <-- NEW

app = typer.Typer(help="Cambrian: Training stack for various architectures")

@app.command(name="train-transformer")
def train_transformer(
    run_name: str = typer.Option(..., help="Run name; results in out/<run-name>/"),
    dataset: str = typer.Option(..., help="HF dataset name, e.g. roneneldan/TinyStories"),
    model: str = typer.Option("nano", "--model", "--model-size", help="Model size: nano, micro, tiny, small, medium"),
    max_iters: str | None = typer.Option(None, help="Total optimizer steps (e.g., 100000, 100K)"),
    max_steps: str | None = typer.Option(None, help="Alias for max_iters - total training steps (e.g., 100K, 1M)"),
    max_tokens: str | None = typer.Option(None, help="Total token budget (e.g., 10B, 100M) - global_batch_size * seq_len * steps"),
    warmup_iters: int = typer.Option(2_000, help="Warmup steps"),
    base_lr: float = typer.Option(3e-4, help="Base LR"),
    weight_decay: float = typer.Option(0.1, help="AdamW weight decay"),
    eval_interval: int = typer.Option(500, help="Eval every N steps"),
    log_interval: int = typer.Option(10, help="Log every N steps"),
    grad_clip: float = typer.Option(1.0, help="Grad clipping norm"),
    target_mem_frac: float = typer.Option(0.85, help="Target per-GPU memory fraction for auto micro-batch"),
    out_dir: str = typer.Option("out", help="Output root directory"),
    seed: int = typer.Option(1337, help="Seed for RNG"),
):
    """Train a transformer language model"""
    train_transformer_main(
        run_name=run_name,
        dataset=dataset,
        model=model,
        max_iters=max_iters,
        max_steps=max_steps,
        max_tokens=max_tokens,
        warmup_iters=warmup_iters,
        base_lr=base_lr,
        weight_decay=weight_decay,
        eval_interval=eval_interval,
        log_interval=log_interval,
        grad_clip=grad_clip,
        target_mem_frac=target_mem_frac,
        out_dir=Path(out_dir),
        seed=seed,
    )

@app.command(name="train-diffusion")
def train_diffusion(
    run_name: str = typer.Option(..., help="Run name; results in out/<run-name>/"),
    dataset: str = typer.Option(..., help="HF dataset name, e.g. roneneldan/TinyStories"),
    model: str = typer.Option("nano", "--model", "--model-size", help="Model size: nano, micro, tiny, small, medium"),
    max_iters: str | None = typer.Option(None, help="Total optimizer steps (e.g., 100000, 100K)"),
    max_steps: str | None = typer.Option(None, help="Alias for max_iters - total training steps (e.g., 100K, 1M)"),
    max_tokens: str | None = typer.Option(None, help="Total token budget (e.g., 10B, 100M) - global_batch_size * seq_len * steps"),
    diffusion_steps: int = typer.Option(128, help="Number of discrete diffusion steps S"),
    context_len: int = typer.Option(0, help="Number of prefix tokens never masked"),
    warmup_iters: int = typer.Option(2_000, help="Warmup steps"),
    base_lr: float = typer.Option(3e-4, help="Base LR"),
    weight_decay: float = typer.Option(0.1, help="AdamW weight decay"),
    eval_interval: int = typer.Option(500, help="Eval every N steps"),
    log_interval: int = typer.Option(10, help="Log every N steps"),
    grad_clip: float = typer.Option(1.0, help="Grad clipping norm"),
    target_mem_frac: float = typer.Option(0.85, help="Target per-GPU memory fraction for auto micro-batch"),
    out_dir: str = typer.Option("out", help="Output root directory"),
    seed: int = typer.Option(1337, help="Seed for RNG"),
):
    """Train a discrete diffusion transformer (masked denoising)."""
    train_diffusion_main(
        run_name=run_name,
        dataset=dataset,
        model=model,
        max_iters=max_iters,
        max_steps=max_steps,
        max_tokens=max_tokens,
        diffusion_steps=diffusion_steps,
        context_len=context_len,
        warmup_iters=warmup_iters,
        base_lr=base_lr,
        weight_decay=weight_decay,
        eval_interval=eval_interval,
        log_interval=log_interval,
        grad_clip=grad_clip,
        target_mem_frac=target_mem_frac,
        out_dir=Path(out_dir),
        seed=seed,
    )

if __name__ == "__main__":
    app()

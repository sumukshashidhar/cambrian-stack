from __future__ import annotations

import json, math
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from loguru import logger

from cambrian.distributed import is_main_process

# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------
def set_torch_defaults():
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'

@contextmanager
def sdpa_flash():
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        yield

def cosine_lr(step: int, max_steps: int, warmup: int, base_lr: float) -> float:
    if step < warmup: return base_lr * (step + 1) / warmup
    if step >= max_steps: return 0.0
    t = (step - warmup) / (max_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))

def auto_microbatch(model: nn.Module, seq_len: int, vocab: int, target_mem_frac: float, amp_dtype: torch.dtype, device: torch.device) -> int:
    model = model.to(device)
    mb = 1
    best = 1
    torch.cuda.empty_cache()
    total_mem = torch.cuda.get_device_properties(device).total_memory
    def mem_frac() -> float:
        return torch.cuda.memory_reserved(device) / total_mem
    for _ in range(20):
        try:
            x = torch.randint(0, vocab, (mb, seq_len), device=device)
            y = torch.randint(0, vocab, (mb, seq_len), device=device)
            with sdpa_flash(), torch.autocast(device_type='cuda', dtype=amp_dtype):
                loss = model(x, y)
            loss.backward()
            frac = mem_frac()
            model.zero_grad(set_to_none=True)
            del x, y, loss
            torch.cuda.synchronize()
            if frac < target_mem_frac * 0.9:
                best = mb
                mb *= 2
                torch.cuda.empty_cache()
                continue
            if frac <= target_mem_frac:
                best = mb
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            raise
    return best

def save_checkpoint(out_dir: Path, model: nn.Module, cfg: dict, run_meta: dict):
    if not is_main_process(): return
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(sd, out_dir / "model.pt")
    with open(out_dir / "config.json", "w") as f: json.dump(cfg, f, indent=2)
    with open(out_dir / "meta.json",   "w") as f: json.dump(run_meta, f, indent=2)
    logger.success(f"Saved checkpoint to {out_dir}")

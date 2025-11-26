#!/usr/bin/env python
"""
Quick batch-size tuner for the baseline Transformer.

Finds the largest device_batch_size that keeps peak VRAM under a target
fraction of total memory on a single GPU. Use this to pick per-GPU
microbatch for multi-GPU training (total_batch_size is set separately).

Usage:
  python scripts/find_batch_size.py --seq-len 256 --depth 12 --target-frac 0.7
"""
from __future__ import annotations

import argparse
import math
import torch
from loguru import logger

from cambrian_stack.models.transformer import Transformer, TransformerConfig


def try_batch(batch_size: int, cfg, device) -> tuple[bool, float]:
    torch.cuda.reset_peak_memory_stats(device)
    model = Transformer(cfg).to(device)
    x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.max_seq_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.max_seq_len), device=device)
    try:
        _, loss = model(x, y)
        loss.backward()
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return False, math.inf
        raise
    peak = torch.cuda.max_memory_allocated(device)
    return True, peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--target-frac", type=float, default=0.7, help="target fraction of total VRAM to stay under")
    parser.add_argument("--max-try", type=int, default=128, help="max batch size to probe")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this tuner.")
    device = torch.device("cuda:0")
    total_mem = torch.cuda.get_device_properties(device).total_memory
    target_bytes = total_mem * args.target_frac

    cfg = TransformerConfig(
        depth=args.depth,
        max_seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        dropout=0.0,
    )

    best = 0
    lo, hi = 1, args.max_try
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, peak = try_batch(mid, cfg, device)
        if ok and peak <= target_bytes:
            best = mid
            lo = mid + 1
            logger.info(f"batch={mid} OK | peak {(peak/1e9):.2f} GB")
        else:
            hi = mid - 1
            if ok:
                logger.info(f"batch={mid} exceeds target: {(peak/1e9):.2f} GB > {(target_bytes/1e9):.2f} GB")
            else:
                logger.info(f"batch={mid} OOM")

    logger.info(
        f"Recommended device_batch_size: {best} (target {(args.target_frac*100):.0f}% of {total_mem/1e9:.2f} GB)"
    )


if __name__ == "__main__":
    main()


#!/bin/bash
# Example training run with a small dataset

# Multi-GPU training with torchrun (uses ALL available GPUs):
WANDB_PROJECT=cambrian-stack uv run torchrun --standalone --nproc_per_node=gpu -m cambrian.cli \
    --run-name "nano-tinystories" \
    --dataset "roneneldan/TinyStories" \
    --model-size nano \
    --max-iters 100K

# For single GPU training (if needed):
# WANDB_PROJECT=cambrian-stack CUDA_VISIBLE_DEVICES=0 uv run cambrian \
#     --run-name "nano-tinystories" \
#     --dataset "roneneldan/TinyStories" \
#     --model-size nano \
#     --max-iters 100K

# Other dataset options:
# - "roneneldan/TinyStories" (small, good for testing)
# - "wikitext" (medium size)
# - "allenai/c4" (large, production)

# Model size options: nano, micro, tiny, small, medium

# Budget options (specify exactly ONE):
# --max-iters 100K            # Run for N optimizer steps (e.g., 100K, 1M)
# --max-steps 100K            # Alias for max-iters (e.g., 100K, 1M)
# --max-tokens 10B            # Run for N tokens (e.g., 10B, 100M) - global_batch_size * seq_len * steps
#
# Note: FLOPs (floating point operations) are calculated and logged automatically
#
# Supported suffixes: K (thousand), M (million), B (billion), T (trillion)
# Examples:
#   --max-iters 50000  or  --max-iters 50K
#   --max-steps 100K           # Same as --max-iters 100K
#   --max-tokens 1000000000  or  --max-tokens 1B


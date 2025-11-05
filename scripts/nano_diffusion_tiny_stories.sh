#!/bin/bash
# Multi-GPU diffusion training with torchrun (uses ALL available GPUs)
WANDB_PROJECT=cambrian-stack uv run torchrun --standalone --nproc_per_node=gpu -m cambrian.cli train-diffusion \
  --run-name dt-nano-tinystories \
  --dataset roneneldan/TinyStories \
  --model nano \
  --max-steps 100K \
  --diffusion-steps 128 \
  --context-len 0
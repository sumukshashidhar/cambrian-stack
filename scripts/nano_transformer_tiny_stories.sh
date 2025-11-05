#!/bin/bash
WANDB_PROJECT=cambrian-stack uv run torchrun --standalone --nproc_per_node=gpu -m cambrian.cli \
    --run-name "nano-tinystories" \
    --dataset "roneneldan/TinyStories" \
    --model-size nano \
    --max-iters 100K

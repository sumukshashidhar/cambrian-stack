#!/bin/bash
# Evaluate a diffusion transformer checkpoint
# Usage: ./scripts/eval_diffusion.sh <checkpoint_path>

set -e

cd /home/sumukshashidhar/workdir/cambrian-stack
source .venv/bin/activate

if [ -f .env ]; then
    set -a && source .env && set +a
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if [ -z "$1" ]; then
    echo "Usage: ./scripts/eval_diffusion.sh <checkpoint_path>"
    exit 1
fi

python -m cambrian_stack.eval_diffusion --checkpoint "$1"

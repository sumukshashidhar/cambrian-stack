#!/bin/bash
# Evaluate a model checkpoint
# Usage: ./scripts/eval_checkpoint.sh out/baseline-d12/checkpoint_010000.pt

set -e

cd /home/sumukshashidhar/workdir/cambrian-stack
source .venv/bin/activate

# Load environment variables
if [ -f .env ]; then
    set -a && source .env && set +a
fi

# Use first GPU for eval
export CUDA_VISIBLE_DEVICES=0

if [ -z "$1" ]; then
    echo "Usage: ./scripts/eval_checkpoint.sh <checkpoint_path>"
    exit 1
fi

python -m cambrian_stack.eval_checkpoint --checkpoint "$1"


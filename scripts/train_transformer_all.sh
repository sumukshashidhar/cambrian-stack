#!/bin/bash
# Simple all-GPU transformer training launcher.
# Hydra configs now live under src/cambrian_stack/conf/.
# Usage: ./scripts/train_transformer_all.sh [hydra overrides]

set -e
cd /home/sumukshashidhar/workdir/cambrian-stack
source .venv/bin/activate

if [ -f .env ]; then
    set -a && source .env && set +a
    echo "✓ Loaded .env"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}
echo "✓ Using GPUs: $CUDA_VISIBLE_DEVICES"

mkdir -p logs out

accelerate launch --multi_gpu --num_processes=4 --main_process_port=${MAIN_PROCESS_PORT} \
    src/cambrian_stack/train.py \
    "$@"

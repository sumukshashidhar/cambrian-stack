#!/bin/bash
# Train diffusion transformer on TinyStories
# Usage: ./scripts/train_diffusion.sh [extra hydra args]

set -e

cd /home/sumukshashidhar/workdir/cambrian-stack
source .venv/bin/activate

if [ -f .env ]; then
    set -a && source .env && set +a
    echo "✓ Loaded .env"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}
echo "✓ Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "✓ Using main process port: $MAIN_PROCESS_PORT"

mkdir -p logs out

accelerate launch --multi_gpu --num_processes=2 --main_process_port=${MAIN_PROCESS_PORT} \
    src/cambrian_stack/train.py \
    --config-name=diffusion_transformer \
    "$@"

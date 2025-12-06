#!/bin/bash
# Train baseline transformer on TinyStories using all available GPUs.
# Hydra configs now live under src/cambrian_stack/conf/.
# Usage: ./scripts/train_baseline_transformer.sh [extra hydra args]

set -e

cd /home/sumukshashidhar/workdir/cambrian-stack
source .venv/bin/activate

# Load environment variables (HF_TOKEN, WANDB_PROJECT, WANDB_API_KEY)
if [ -f .env ]; then
    set -a && source .env && set +a
    echo "✓ Loaded .env"
fi

# Use all GPUs available
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}
echo "✓ Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "✓ Using main process port: $MAIN_PROCESS_PORT"

# Show GPU status
nvidia-smi --query-gpu=index,name,memory.free --format=csv

# Ensure output directories exist
mkdir -p logs out

# Train with accelerate (multi-GPU)
echo "Starting training..."
accelerate launch --multi_gpu --num_processes=4 --main_process_port=${MAIN_PROCESS_PORT} \
    src/cambrian_stack/train.py \
    "$@"

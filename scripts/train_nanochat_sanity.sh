#!/bin/bash
# Nanochat-ish sanity run sized for 4×24GB GPUs.
# - Model: depth=16, d_model=1024, vocab=65536, seq=2048
# - Batch: auto-tuned device_batch_size (target 70% VRAM) → grad_accum computed
# - Total batch size fixed at 524288 tokens (nanochat-style)
# - Steps: 8000 (~4.2B tokens, ~20× params for ~200M param model)
# Usage: ./scripts/train_nanochat_sanity.sh [extra hydra overrides]

set -e

cd /home/sumukshashidhar/workdir/cambrian-stack
source .venv/bin/activate

# Load env (WANDB, HF)
[ -f .env ] && set -a && source .env && set +a

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}

mkdir -p logs out

DBS=$(python scripts/find_batch_size.py --seq-len 2048 --depth 16 --vocab-size 65536 --target-frac 0.7 --max-try 12 | tail -n1 | awk '{print $NF}')
[ -z "$DBS" ] && DBS=4

accelerate launch --multi_gpu --num_processes=4 --main_process_port=${MAIN_PROCESS_PORT} \
  src/cambrian_stack/train.py \
  --config-name=baseline_fineweb \
  model.depth=16 \
  model.max_seq_len=2048 \
  model.vocab_size=65536 \
  training.device_batch_size=${DBS} \
  training.total_batch_size=524288 \
  training.max_steps=8000 \
  output.dir=out/fineweb-d16-sanity \
  logging.wandb_run=fineweb-d16-sanity \
  "$@"

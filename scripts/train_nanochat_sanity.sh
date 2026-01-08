#!/bin/bash
# Nanochat-ish sanity run sized for 4×24GB GPUs.
# - Model: depth=16, d_model=1024, vocab=65536, seq=2048
# - Batch: auto-tuned device_batch_size (target 70% VRAM) → grad_accum computed
# - Total batch size fixed at 524288 tokens (nanochat-style)
# - Steps: 8000 (override target_param_ratio for a short sanity run)
# Usage: ./scripts/train_nanochat_sanity.sh [extra hydra overrides]

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_DIR"

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# Load env (WANDB, HF)
[ -f .env ] && set -a && source .env && set +a

# Detect GPUs if CUDA_VISIBLE_DEVICES is unset.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
  if [ "${GPU_COUNT:-0}" -gt 0 ]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
    export CUDA_VISIBLE_DEVICES
  fi
fi

export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}
NUM_MACHINES=${NUM_MACHINES:-1}
MACHINE_RANK=${MACHINE_RANK:-0}
MAIN_PROCESS_IP=${MAIN_PROCESS_IP:-${MASTER_ADDR:-127.0.0.1}}

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
  DEFAULT_NUM_PROCESSES=${#DEVICES[@]}
else
  DEFAULT_NUM_PROCESSES=1
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
    if [ "${GPU_COUNT:-0}" -gt 0 ]; then
      DEFAULT_NUM_PROCESSES=$GPU_COUNT
    fi
  fi
fi
NUM_PROCESSES=${NUM_PROCESSES:-$DEFAULT_NUM_PROCESSES}

mkdir -p logs out

DBS=$(python scripts/find_batch_size.py \
  --seq-len 2048 \
  --depth 16 \
  --vocab-size 65536 \
  --target-frac 0.7 \
  --max-try 12 \
  --model-type nanochat_gpt 2>&1 | awk -F 'Recommended device_batch_size: ' '/Recommended device_batch_size/ {print $2}' | awk '{print $1}')
if [[ ! "${DBS}" =~ ^[0-9]+$ ]]; then
  DBS=4
fi

ACCEL_ARGS=(--num_processes="${NUM_PROCESSES}" --main_process_port="${MAIN_PROCESS_PORT}")
if [ "${NUM_PROCESSES}" -ge 2 ]; then
  ACCEL_ARGS=(--multi_gpu "${ACCEL_ARGS[@]}")
fi
if [ "${NUM_MACHINES}" -ge 2 ]; then
  ACCEL_ARGS+=(--num_machines="${NUM_MACHINES}" --machine_rank="${MACHINE_RANK}" --main_process_ip="${MAIN_PROCESS_IP}")
fi

accelerate launch "${ACCEL_ARGS[@]}" \
  -m cambrian_stack.speedrun \
  model.depth=16 \
  model.max_seq_len=2048 \
  model.vocab_size=65536 \
  training.device_batch_size=${DBS} \
  training.total_batch_size=524288 \
  training.max_steps=8000 \
  output.dir=out/fineweb-d16-sanity \
  logging.wandb_run=fineweb-d16-sanity \
  "$@"

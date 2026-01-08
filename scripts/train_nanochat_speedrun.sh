#!/bin/bash
# NanoChat-style speedrun for Cambrian Stack on FineWeb-Edu (streaming).
# - Model: depth=20, seq=2048, vocab=50257 (GPT-2 tokenizer)
# - Batch: auto-tuned per-GPU device_batch_size (target 70% VRAM)
# - Steps: computed inside speedrun runner from target tokens:param ratio (default 20x)
# Usage: ./scripts/train_nanochat_speedrun.sh [extra hydra overrides]

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}
cd "$REPO_DIR"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH}"

if [ -z "${SKIP_VENV:-}" ] && [ -f ".venv/bin/activate" ]; then
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

DEPTH=${DEPTH:-20}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
VOCAB_SIZE=${VOCAB_SIZE:-50257}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-524288}
TARGET_PARAM_RATIO=${TARGET_PARAM_RATIO:-20}
TARGET_FRAC=${TARGET_FRAC:-0.7}
MAX_TRY=${MAX_TRY:-32}
if [ -z "${RUN_NAME:-}" ]; then
  if [ -n "${WANDB_RUN:-}" ]; then
    RUN_NAME="${WANDB_RUN}"
  elif [ -n "${WANDB_API_KEY:-}" ]; then
    RUN_NAME="fineweb-d${DEPTH}-speedrun"
  else
    RUN_NAME="dummy"
  fi
fi
OUTPUT_DIR=${OUTPUT_DIR:-out/${RUN_NAME}}
RUN_EVAL=${RUN_EVAL:-1}

if [ -z "${NUM_PROCESSES:-}" ]; then
  IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_PROCESSES=${#DEVICES[@]}
fi

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "ERROR: python not found in PATH."
  exit 1
fi

if [ -n "${DEVICE_BATCH_SIZE:-}" ]; then
  DBS="${DEVICE_BATCH_SIZE}"
else
  DBS=$("${PYTHON_BIN}" scripts/find_batch_size.py \
    --seq-len "${MAX_SEQ_LEN}" \
    --depth "${DEPTH}" \
    --vocab-size "${VOCAB_SIZE}" \
    --model-type nanochat_gpt \
    --target-frac "${TARGET_FRAC}" \
    --max-try "${MAX_TRY}" 2>&1 | awk -F 'Recommended device_batch_size: ' '/Recommended device_batch_size/ {print $2}' | awk '{print $1}')
  if [[ ! "${DBS}" =~ ^[0-9]+$ ]]; then
    DBS=4
  fi
fi

echo "INFO depth=${DEPTH} | seq=${MAX_SEQ_LEN} | vocab=${VOCAB_SIZE}"
echo "INFO device_batch_size=${DBS} | num_processes=${NUM_PROCESSES}"

mkdir -p logs out

ACCEL_ARGS=(--num_processes="${NUM_PROCESSES}" --main_process_port="${MAIN_PROCESS_PORT}")
if [ "${NUM_PROCESSES}" -ge 2 ]; then
  ACCEL_ARGS=(--multi_gpu "${ACCEL_ARGS[@]}")
fi
if [ "${NUM_MACHINES}" -ge 2 ]; then
  ACCEL_ARGS+=(--num_machines="${NUM_MACHINES}" --machine_rank="${MACHINE_RANK}" --main_process_ip="${MAIN_PROCESS_IP}")
fi

ACCEL_BIN=$(command -v accelerate || true)
if [ -z "${ACCEL_BIN}" ]; then
  echo "ERROR: accelerate not found in PATH. Activate .venv or install accelerate."
  exit 1
fi

"${ACCEL_BIN}" launch "${ACCEL_ARGS[@]}" \
  -m cambrian_stack.speedrun \
  model.depth="${DEPTH}" \
  model.max_seq_len="${MAX_SEQ_LEN}" \
  model.vocab_size="${VOCAB_SIZE}" \
  training.device_batch_size="${DBS}" \
  training.total_batch_size="${TOTAL_BATCH_SIZE}" \
  training.target_param_ratio="${TARGET_PARAM_RATIO}" \
  training.max_steps=-1 \
  output.dir="${OUTPUT_DIR}" \
  logging.wandb_run="${RUN_NAME}" \
  "$@"

if [ "${RUN_EVAL}" != "0" ]; then
  FINAL_CKPT=$(ls -1 "${OUTPUT_DIR}"/checkpoint_*.pt 2>/dev/null | sort | tail -n1)
  if [ -n "${FINAL_CKPT}" ]; then
    python -m cambrian_stack.eval_checkpoint --checkpoint "${FINAL_CKPT}"
  else
    echo "No checkpoint found in ${OUTPUT_DIR}, skipping eval."
  fi
fi

#!/bin/bash
# Simple all-GPU transformer training launcher.
# Hydra configs now live under src/cambrian_stack/conf/.
# Usage: ./scripts/train_transformer_all.sh [hydra overrides]

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_DIR"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

if [ -f .env ]; then
    set -a && source .env && set +a
    echo "✓ Loaded .env"
fi

# Detect GPUs if CUDA_VISIBLE_DEVICES is unset.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
    if [ "${GPU_COUNT:-0}" -gt 0 ]; then
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
        export CUDA_VISIBLE_DEVICES
    fi
fi

MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}
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

echo "✓ Using GPUs: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "✓ Using num processes: $NUM_PROCESSES"

mkdir -p logs out

ACCEL_BIN=".venv/bin/accelerate"
if [ ! -x "$ACCEL_BIN" ]; then
    ACCEL_BIN=$(command -v accelerate || true)
fi
if [ -z "${ACCEL_BIN}" ]; then
    echo "ERROR: accelerate not found. Activate .venv or install accelerate."
    exit 1
fi

ACCEL_ARGS=(--num_processes="${NUM_PROCESSES}" --main_process_port="${MAIN_PROCESS_PORT}")
if [ "${NUM_PROCESSES}" -ge 2 ]; then
    ACCEL_ARGS=(--multi_gpu "${ACCEL_ARGS[@]}")
fi
if [ "${NUM_MACHINES}" -ge 2 ]; then
    ACCEL_ARGS+=(--num_machines="${NUM_MACHINES}" --machine_rank="${MACHINE_RANK}" --main_process_ip="${MAIN_PROCESS_IP}")
fi

"$ACCEL_BIN" launch "${ACCEL_ARGS[@]}" \
    src/cambrian_stack/train.py \
    "$@"

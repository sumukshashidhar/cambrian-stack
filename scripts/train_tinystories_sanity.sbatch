#!/bin/bash
#SBATCH -J cambrian-sanity
#SBATCH -A beig-dtai-gh
#SBATCH --qos=beig-dtai-gh
#SBATCH -p ghx4
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o logs/slurm-sanity-%j.out
#SBATCH -e logs/slurm-sanity-%j.err

# =============================================================================
# TinyStories Sanity Check - NCSA Delta AI (GH200)
# =============================================================================
# A quick sanity check script to validate the training pipeline works.
# Trains a small model (depth=8, 76M params) on TinyStories for 100 steps.
#
# Usage:
#   sbatch scripts/train_tinystories_sanity.sbatch
#
# Environment variables (optional):
#   SLURM_ACCOUNT    - SLURM account (default: beig-dtai-gh)
#   SIF              - Path to PyTorch Apptainer image
#   DEPTH            - Model depth (default: 8)
#   MAX_STEPS        - Training steps (default: 100)
#   NUM_GPUS         - Number of GPUs (default: 2)
#   WANDB_RUN        - WandB run name (default: dummy, set to enable logging)
#
# Expected results (~1 min on 2x GH200):
#   - Loss: 10.8 -> 2.5
#   - Val perplexity: 52 -> 14
#   - Checkpoint saved to out/sanity-check/
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
SIF="${SIF:-/work/nvme/beig/sumukshashidhar/containers/pytorch_25_08.sif}"

# Model/training config
DEPTH="${DEPTH:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
MAX_STEPS="${MAX_STEPS:-100}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32768}"
WANDB_RUN="${WANDB_RUN:-dummy}"
OUTPUT_DIR="${OUTPUT_DIR:-out/sanity-check}"

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
if [[ ! -f "$REPO_DIR/pyproject.toml" ]]; then
  echo "ERROR: repo not found at $REPO_DIR" >&2
  exit 1
fi

if [[ ! -f "$SIF" ]]; then
  echo "ERROR: Apptainer image not found at $SIF" >&2
  echo "Please set SIF=/path/to/pytorch.sif or place image at default location" >&2
  exit 1
fi

cd "$REPO_DIR"
mkdir -p logs out

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
# Load .env if present (for WANDB_API_KEY, HF_TOKEN, etc.)
if [[ -f "$REPO_DIR/.env" ]]; then
  set -a
  source "$REPO_DIR/.env"
  set +a
fi

# Cache directories - keep off $HOME for performance on NVMe
GLOBAL_CACHE="${GLOBAL_CACHE:-/work/nvme/beig/sumukshashidhar/.cache}"
export HF_HOME="$GLOBAL_CACHE/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$GLOBAL_CACHE/torchinductor"
export TRITON_CACHE_DIR="$GLOBAL_CACHE/triton"
export XDG_CACHE_HOME="$GLOBAL_CACHE/xdg"
export PIP_CACHE_DIR="$GLOBAL_CACHE/pip"
export PYTHONUSERBASE="$GLOBAL_CACHE/python_user"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$DATASETS_CACHE" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
         "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$PYTHONUSERBASE"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

# Detect GPU count from SLURM
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NUM_PROCESSES="$SLURM_GPUS_ON_NODE"
elif [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
  NUM_PROCESSES="$SLURM_GPUS_PER_NODE"
else
  NUM_PROCESSES="${NUM_GPUS:-2}"
fi
export NUM_PROCESSES

# -----------------------------------------------------------------------------
# Job info
# -----------------------------------------------------------------------------
echo "============================================================"
echo "[$(date)] Cambrian Stack - TinyStories Sanity Check"
echo "============================================================"
echo "Node:        ${SLURM_NODELIST:-localhost}"
echo "Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "GPUs:        $NUM_PROCESSES"
echo "Model:       NanochatGPT depth=$DEPTH"
echo "Steps:       $MAX_STEPS"
echo "Output:      $OUTPUT_DIR"
echo "Container:   $SIF"
echo "============================================================"
nvidia-smi -L 2>/dev/null || true
echo ""

# -----------------------------------------------------------------------------
# Run training in container
# -----------------------------------------------------------------------------
apptainer exec --nv \
  --bind "$REPO_DIR:$REPO_DIR" \
  --bind "$HOME:$HOME" \
  --bind "$GLOBAL_CACHE:$GLOBAL_CACHE" \
  --pwd "$REPO_DIR" \
  "$SIF" /bin/bash -lc '
    set -e

    # Environment inside container
    export SKIP_VENV=1
    export HF_HOME="'"$HF_HOME"'"
    export TRANSFORMERS_CACHE="'"$TRANSFORMERS_CACHE"'"
    export HF_HUB_CACHE="'"$HF_HUB_CACHE"'"
    export DATASETS_CACHE="'"$DATASETS_CACHE"'"
    export TORCHINDUCTOR_CACHE_DIR="'"$TORCHINDUCTOR_CACHE_DIR"'"
    export TRITON_CACHE_DIR="'"$TRITON_CACHE_DIR"'"
    export XDG_CACHE_HOME="'"$XDG_CACHE_HOME"'"
    export PIP_CACHE_DIR="'"$PIP_CACHE_DIR"'"
    export PYTHONUSERBASE="'"$PYTHONUSERBASE"'"
    export PATH="$PYTHONUSERBASE/bin:$PATH"
    export PYTHONPATH="'"$REPO_DIR"'/src:${PYTHONPATH:-}"
    export TOKENIZERS_PARALLELISM=false

    cd "'"$REPO_DIR"'"

    echo "[$(date)] Python: $(python3 --version)"
    python3 -c "import torch; print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}\")"

    # Install dependencies
    echo "[$(date)] Installing dependencies..."
    python3 -m pip install --user -q -e . accelerate

    echo "[$(date)] Starting training..."

    accelerate launch \
      --multi_gpu \
      --num_processes="'"$NUM_PROCESSES"'" \
      --main_process_port="'"$MAIN_PROCESS_PORT"'" \
      -m cambrian_stack.speedrun \
      model.depth="'"$DEPTH"'" \
      model.max_seq_len="'"$MAX_SEQ_LEN"'" \
      model.vocab_size=50257 \
      data=tiny_stories \
      training.device_batch_size="'"$DEVICE_BATCH_SIZE"'" \
      training.total_batch_size="'"$TOTAL_BATCH_SIZE"'" \
      training.max_steps="'"$MAX_STEPS"'" \
      training.eval_every=25 \
      training.sample_every=50 \
      training.save_every="'"$MAX_STEPS"'" \
      training.log_every=5 \
      output.dir="'"$OUTPUT_DIR"'" \
      logging.wandb_run="'"$WANDB_RUN"'"
  '

echo ""
echo "============================================================"
echo "[$(date)] Sanity check complete!"
echo "============================================================"
if [[ -d "$OUTPUT_DIR" ]]; then
  echo "Checkpoint: $(ls -1 "$OUTPUT_DIR"/checkpoint_*.pt 2>/dev/null | tail -1 || echo 'none')"
fi

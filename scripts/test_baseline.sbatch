#!/bin/bash
#SBATCH -J cambrian-baseline-test
#SBATCH -p ghx4-interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o logs/slurm-baseline-test-%j.out
#SBATCH -e logs/slurm-baseline-test-%j.err

# NOTE: You must specify your SLURM account when submitting:
#   sbatch -A <your-account> scripts/test_baseline.sbatch
# Or set SBATCH_ACCOUNT in your environment.

set -euo pipefail

# Repo directory: use SLURM_SUBMIT_DIR (where sbatch was run from)
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-.}}"

# Container: must be set via environment or override
SIF="${SIF:-}"
if [[ -z "$SIF" ]]; then
  echo "ERROR: SIF environment variable not set. Please set it to your PyTorch Apptainer image path." >&2
  echo "  Example: SIF=/path/to/pytorch.sif sbatch scripts/test_baseline.sbatch" >&2
  exit 1
fi

if [[ ! -f "$REPO_DIR/pyproject.toml" ]]; then
  echo "ERROR: repo not found at $REPO_DIR" >&2
  exit 1
fi

cd "$REPO_DIR"
mkdir -p logs out

# Load env (WANDB, HF) if present
if [[ -f "$REPO_DIR/.env" ]]; then
  set -a
  source "$REPO_DIR/.env"
  set +a
fi

# Cache directory: use CACHE_DIR env var, or default to .cache in repo
GLOBAL_CACHE="${CACHE_DIR:-$REPO_DIR/.cache}"
export HF_HOME="$GLOBAL_CACHE/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$GLOBAL_CACHE/torchinductor"
export TRITON_CACHE_DIR="$GLOBAL_CACHE/triton"
export XDG_CACHE_HOME="$GLOBAL_CACHE/xdg"
export PIP_CACHE_DIR="$GLOBAL_CACHE/pip"
export PYTHONUSERBASE="$GLOBAL_CACHE/python_user"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$DATASETS_CACHE" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$PYTHONUSERBASE"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
export NUM_PROCESSES=1

echo "============================================================"
echo "[$(date)] Starting Cambrian baseline test"
echo "[$(date)] Node list: ${SLURM_NODELIST:-unknown}"
echo "[$(date)] Job ID: ${SLURM_JOB_ID:-unknown}"
echo "[$(date)] Repo: $REPO_DIR"
echo "[$(date)] SIF:  $SIF"
echo "[$(date)] Cache: $GLOBAL_CACHE"
echo "============================================================"
echo "[$(date)] GPUs visible:"
nvidia-smi -L || true
nvidia-smi || true
echo ""

if [[ ! -f "$SIF" ]]; then
  echo "[$(date)] ERROR: SIF not found at $SIF" >&2
  exit 1
fi

apptainer exec --nv \
  --bind "$REPO_DIR:$REPO_DIR" \
  --bind "$HOME:$HOME" \
  --bind "$GLOBAL_CACHE:$GLOBAL_CACHE" \
  --pwd "$REPO_DIR" \
  "$SIF" /bin/bash -lc '
    set -e
    export SKIP_VENV=1
    export PYTHON_BIN=python3
    export HF_HOME='"$HF_HOME"'
    export TRANSFORMERS_CACHE='"$TRANSFORMERS_CACHE"'
    export HF_HUB_CACHE='"$HF_HUB_CACHE"'
    export DATASETS_CACHE='"$DATASETS_CACHE"'
    export TORCHINDUCTOR_CACHE_DIR='"$TORCHINDUCTOR_CACHE_DIR"'
    export TRITON_CACHE_DIR='"$TRITON_CACHE_DIR"'
    export XDG_CACHE_HOME='"$XDG_CACHE_HOME"'
    export PIP_CACHE_DIR='"$PIP_CACHE_DIR"'
    export PYTHONUSERBASE='"$PYTHONUSERBASE"'
    export PATH="$PYTHONUSERBASE/bin:$PATH"
    export PYTHONPATH="$(python3 -m site --user-site):$PYTHONPATH"

    cd "'"$REPO_DIR"'"
    echo "[$(date)] Python version:"
    python3 --version
    echo "[$(date)] Torch version:"
    python3 -c "import torch; print(torch.__version__); print(f\"CUDA available: {torch.cuda.is_available()}\")"

    # Install cambrian deps into user base inside cache
    echo "[$(date)] Installing dependencies..."
    python3 -m pip install --user -e . 2>&1 | tail -5
    python3 -m pip install --user accelerate 2>&1 | tail -3

    echo "[$(date)] Starting baseline training (1 hour test)..."
    python3 -m cambrian_stack.train \
      training.max_steps=3000 \
      training.eval_every=500 \
      training.save_every=0 \
      training.sample_every=0 \
      logging.wandb_run=baseline-test-$(date +%Y%m%d-%H%M%S) \
      output.dir=out/baseline-test
  '

echo "[$(date)] Done!"

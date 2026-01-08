#!/bin/bash
#SBATCH -J cambrian-speedrun
#SBATCH -A bcqu-dtai-gh
#SBATCH --qos=bcqu-dtai-gh
#SBATCH -p ghx4
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o logs/slurm-speedrun-%j.out
#SBATCH -e logs/slurm-speedrun-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-/work/nvme/bcqu/sumukshashidhar/projects/cambrian-stack}}"
SIF="${SIF:-/work/nvme/beig/sumukshashidhar/containers/pytorch_25_08.sif}"
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

# Keep caches off $HOME (NVMe BEIG)
GLOBAL_CACHE="/work/nvme/beig/sumukshashidhar/.cache"
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-144}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

# Match SLURM GPUs to accelerate
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  export NUM_PROCESSES="$SLURM_GPUS_ON_NODE"
elif [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
  export NUM_PROCESSES="$SLURM_GPUS_PER_NODE"
fi

echo "============================================================"
echo "[$(date)] Starting Cambrian speedrun"
echo "[$(date)] Node list: ${SLURM_NODELIST:-unknown}"
echo "[$(date)] Job ID: ${SLURM_JOB_ID:-unknown}"
echo "[$(date)] Repo: $REPO_DIR"
echo "[$(date)] SIF:  $SIF"
echo "============================================================"
echo "[$(date)] GPUs visible:"
nvidia-smi -L || true
nvidia-smi || true
echo ""

if [[ ! -f "$SIF" ]]; then
  echo "[$(date)] ERROR: SIF not found at $SIF. Please place the PyTorch Apptainer image there or set SIF=... when submitting." >&2
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
    python3 -c "import torch; print(\"torch\", torch.__version__)"

    # Install cambrian deps into user base inside NVMe cache
    python3 -m pip install --user -e .
    python3 -m pip install --user accelerate

    bash scripts/train_nanochat_speedrun.sh "$@"
  '

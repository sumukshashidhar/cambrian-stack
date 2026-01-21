#!/bin/bash
# Wrapper script for sumuk's environment on NCSA Delta
# Usage: ./scripts/run_speedrun_sumuk.sh

export SIF=/work/nvme/beig/sumukshashidhar/containers/pytorch_25_08.sif
export CACHE_DIR=/work/nvme/beig/sumukshashidhar/.cache

sbatch -A beig-dtai-gh scripts/train_nanochat_speedrun.sbatch "$@"

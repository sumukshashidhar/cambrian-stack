#!/bin/bash
# Launch parallel AR and Diffusion training jobs with 12h timeout.
# - Diffusion uses GPUs 0,1
# - Autoregressive baseline uses GPUs 2,3
# Wandb is enabled via .env (WANDB_API_KEY/WANDB_PROJECT).

set -e

ROOT="/home/sumukshashidhar/workdir/cambrian-stack"
cd "$ROOT"
source .venv/bin/activate

if [ -f .env ]; then
    set -a && source .env && set +a
    echo "✓ Loaded .env"
fi

mkdir -p logs out

RUN_ID=$(date +%Y%m%d-%H%M%S)
DIFF_RUN="diffusion-${RUN_ID}"
AR_RUN="baseline-${RUN_ID}"

# Diffusion (GPUs 0,1)
MASTER_PORT=29510 MAIN_PROCESS_PORT=29510 CUDA_VISIBLE_DEVICES=0,1 \
nohup timeout 12h ./scripts/train_diffusion.sh \
    logging.wandb_run="${DIFF_RUN}" \
    output.dir="out/${DIFF_RUN}" \
    model.max_seq_len=256 \
    training.device_batch_size=4 \
    training.total_batch_size=65536 \
    > "logs/${DIFF_RUN}.stdout" 2>&1 &
DIFF_PID=$!
echo "Started diffusion run ${DIFF_RUN} (PID ${DIFF_PID}) on GPUs 0,1"

# Autoregressive baseline (GPUs 2,3)
MASTER_PORT=29520 MAIN_PROCESS_PORT=29520 CUDA_VISIBLE_DEVICES=2,3 \
nohup timeout 12h ./scripts/train_baseline_transformer.sh \
    logging.wandb_run="${AR_RUN}" \
    output.dir="out/${AR_RUN}" \
    model.max_seq_len=256 \
    training.device_batch_size=4 \
    training.total_batch_size=65536 \
    > "logs/${AR_RUN}.stdout" 2>&1 &
AR_PID=$!
echo "Started baseline run ${AR_RUN} (PID ${AR_PID}) on GPUs 2,3"

echo "Logs: logs/${DIFF_RUN}.stdout , logs/${AR_RUN}.stdout"
echo "Outputs: out/${DIFF_RUN} , out/${AR_RUN}"

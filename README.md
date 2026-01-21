# Cambrian Stack

Minimal, hackable transformer experiments.

## Quick Start (Local)

```bash
git clone <repo-url> && cd cambrian-stack
uv venv && source .venv/bin/activate
uv pip install -e ".[ml]"   # or ".[dev]" for tooling; base install skips torch

./scripts/train_baseline_transformer.sh                # baseline AR
python -m cambrian_stack.train --config-name=diffusion_transformer  # diffusion
```

Outputs: checkpoints in `out/`, logs in `logs/`, optional W&B if `WANDB_API_KEY` is set.

---

## Running on SLURM (HPC Clusters)

This section explains how to run training jobs on SLURM-managed HPC clusters using Apptainer (formerly Singularity) containers.

### Prerequisites

Before you start, ensure you have:

1. **Apptainer/Singularity container** with PyTorch
   - Example: NGC PyTorch container converted to SIF format
   - The container must have PyTorch with CUDA support

2. **SLURM account** with GPU partition access
   - Know your account name (e.g., `mygroup-gpu`)
   - Know available partitions (e.g., `ghx4`, `ghx4-interactive`)

3. **Environment file** (optional but recommended)
   - Create `.env` in the repo root with:
     ```bash
     WANDB_API_KEY=your_wandb_key
     HF_TOKEN=your_huggingface_token
     ```

### Environment Variables

The SLURM scripts use these environment variables:

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `SIF` | **Yes** | Path to PyTorch Apptainer/Singularity image | `/scratch/containers/pytorch.sif` |
| `CACHE_DIR` | No | Cache directory for HuggingFace, pip, etc. Defaults to `.cache/` in repo | `/scratch/myuser/.cache` |
| `SBATCH_ACCOUNT` | No | Default SLURM account (alternative to `-A` flag) | `mygroup-gpu` |

### Option 1: Batch Job Submission

This is the standard way to run training jobs.

**Step 1:** Set your environment variables

```bash
export SIF=/path/to/your/pytorch.sif
export CACHE_DIR=/path/to/cache   # optional, but recommended for shared systems
```

**Step 2:** Submit the job

```bash
# From the repo root directory:
sbatch -A <your-account> scripts/test_baseline.sbatch
```

**Step 3:** Monitor the job

```bash
# Check job status
squeue -u $USER

# Watch training progress (training logs go to stderr)
tail -f logs/slurm-baseline-test-<jobid>.err

# Watch job output (setup, GPU info, etc.)
tail -f logs/slurm-baseline-test-<jobid>.out
```

#### Available Scripts

| Script | Description | Duration | GPUs |
|--------|-------------|----------|------|
| `scripts/test_baseline.sbatch` | Sanity check run | 1 hour | 1 |
| `scripts/train_nanochat_speedrun.sbatch` | Full training run | 24 hours | 2 |

### Option 2: Interactive Sessions

For debugging, development, or when you want to see output in real-time.

**Step 1:** Request an interactive GPU session

```bash
srun -A <your-account> \
     -p ghx4-interactive \
     --gres=gpu:1 \
     --mem=32G \
     --cpus-per-task=8 \
     --time=02:00:00 \
     --pty bash
```

**Step 2:** Once on the GPU node, start the container

```bash
apptainer exec --nv /path/to/pytorch.sif bash
```

The `--nv` flag is **required** - it enables GPU access inside the container.

**Step 3:** Inside the container, set up and run

```bash
cd /path/to/cambrian-stack

# Install the package (first time only, or after code changes)
pip install --user -e .
pip install --user accelerate

# Run training
python -m cambrian_stack.train training.max_steps=100
```

#### One-liner for Interactive Training

```bash
srun -A <your-account> -p ghx4-interactive --gres=gpu:1 --mem=32G --time=02:00:00 \
  apptainer exec --nv /path/to/pytorch.sif bash -c '
    cd /path/to/cambrian-stack
    pip install --user -e . && pip install --user accelerate
    python -m cambrian_stack.train training.max_steps=100
  '
```

### Creating a Wrapper Script

For convenience, create a personal wrapper script:

```bash
#!/bin/bash
# scripts/run_myname.sh - Personal wrapper for <your-name>

export SIF=/path/to/your/pytorch.sif
export CACHE_DIR=/path/to/your/cache

sbatch -A your-account scripts/test_baseline.sbatch "$@"
```

Make it executable: `chmod +x scripts/run_myname.sh`

Then run with: `./scripts/run_myname.sh`

### Quick Iteration Tips

For faster debugging cycles:

```bash
# Use a smaller model
python -m cambrian_stack.train model.depth=4

# Fewer training steps
python -m cambrian_stack.train training.max_steps=50 training.eval_every=25

# Disable saving and sampling
python -m cambrian_stack.train training.save_every=0 training.sample_every=0

# Combine for fast iteration
python -m cambrian_stack.train \
  model.depth=4 \
  training.max_steps=50 \
  training.eval_every=25 \
  training.save_every=0
```

---

## Configs

Hydra configs live in `src/cambrian_stack/conf/` (grouped by experiment/model/data/training/logging/output). Example overrides:

```bash
python -m cambrian_stack.train training.max_steps=2000 training.eval_every=200
python -m cambrian_stack.train --config-name=baseline_transformer model.depth=6
```

## Experiments

- Autoregressive (GPT-style) and diffusion strategies are registered in `cambrian_stack/experiments`. Add new ideas by adding an experiment module + config, without touching the trainer.

## Docs

- Build locally: `pip install -e ".[docs]" && sphinx-build -b html docs docs/_build/html`
- Deploy: GitHub Actions workflow `docs.yml` publishes to Pages.

## License

MIT

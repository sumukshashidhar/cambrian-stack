# cambrian-stack

A minimal, hackable framework for transformer experiments. Train language models on TinyStories with one command.

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Extending the Framework](#extending-the-framework)

---

## Quick Start

```bash
# Clone and setup
git clone <repo-url> && cd cambrian-stack
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Configure API keys (optional but recommended)
cp .env.example .env  # Add your WANDB_API_KEY and HF_TOKEN

# Train a baseline transformer
./scripts/train_baseline_transformer.sh
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | ≥ 3.12  |
| CUDA        | ≥ 11.8  |
| GPU VRAM    | ≥ 8GB   |

**Recommended:** 4× NVIDIA GPU for fast training.

---

## Installation

### 1. Create virtual environment

```bash
uv venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
uv pip install -e ".[dev]"
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```bash
# .env
HF_TOKEN=your_huggingface_token        # For gated datasets
WANDB_API_KEY=your_wandb_api_key       # For experiment tracking
WANDB_PROJECT=cambrian-stack           # Project name in W&B
```

---

## Training

### Train baseline transformer

```bash
./scripts/train_baseline_transformer.sh
```

This trains a 12-layer GPT-style transformer (~85M params) on TinyStories.

### Train with custom configuration

```bash
# Override parameters via command line
./scripts/train_baseline_transformer.sh model.depth=6 training.max_steps=5000

# Use a different config file
./scripts/train_baseline_transformer.sh --config-name=diffusion_transformer
```

### Train diffusion transformer

```bash
./scripts/train_baseline_transformer.sh --config-name=diffusion_transformer
```

### Single GPU training

```bash
source .venv/bin/activate
python -m cambrian_stack.train model.depth=6 training.device_batch_size=8
```

### Training outputs

| Directory | Contents |
|-----------|----------|
| `out/`    | Model checkpoints |
| `logs/`   | Training logs |
| `wandb/`  | W&B run data |

---

## Evaluation

### Evaluate a checkpoint

```bash
./scripts/eval_checkpoint.sh out/baseline-d12/checkpoint_010000.pt
```

### Interactive generation

```python
import torch
from cambrian_stack.models import create_model
from cambrian_stack.data_loaders import get_tokenizer

# Load checkpoint
checkpoint = torch.load("out/baseline-d12/checkpoint_010000.pt", weights_only=False)
model = create_model(checkpoint["config"]["model"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval().cuda()

# Generate text
tokenizer = get_tokenizer("gpt2")
prompt = tokenizer.encode("Once upon a time", return_tensors="pt").cuda()
output = model.generate(prompt, max_new_tokens=50, temperature=0.8)
print(tokenizer.decode(output[0]))
```

---

## Configuration

Configs live in `src/cambrian_stack/conf/`. Key parameters:

```yaml
# src/cambrian_stack/conf/baseline_transformer.yaml

model:
  type: transformer           # transformer | diffusion_transformer
  depth: 12                   # Layers (d_model = depth × 64)
  max_seq_len: 1024
  vocab_size: 50257           # GPT-2 tokenizer

training:
  device_batch_size: 24       # Per-GPU batch size (reduce if OOM)
  total_batch_size: 262144    # Total tokens per optimizer step
  max_steps: 10000
  learning_rate: 6.0e-4
  eval_every: 250
  save_every: 2500

output:
  dir: out/baseline-d12
```

### Create a new experiment

1. Copy an existing config: `cp src/cambrian_stack/conf/baseline_transformer.yaml src/cambrian_stack/conf/my_experiment.yaml`
2. Modify parameters
3. Run: `./scripts/train_baseline_transformer.sh --config-name=my_experiment`

---

## Project Structure

```
cambrian-stack/
├── src/cambrian_stack/conf/    # Hydra config files (grouped)
│   ├── baseline_transformer.yaml
│   ├── diffusion_transformer.yaml
│   ├── experiment/
│   ├── model/
│   ├── data/
│   ├── training/
│   ├── logging/
│   └── output/
├── src/cambrian_stack/
│   ├── models/
│   │   ├── transformer.py      # GPT-style autoregressive
│   │   └── diffusion_transformer.py
│   ├── data_loaders/
│   │   └── tiny_stories.py     # Streaming TinyStories dataset
│   ├── training/
│   │   └── trainer.py          # Training loop
│   ├── evaluation/
│   │   └── metrics.py          # Loss, perplexity, generation
│   └── train.py                # Entry point
├── scripts/                    # Shell scripts
├── out/                        # Checkpoints (gitignored)
└── logs/                       # Training logs (gitignored)
```

---

## Extending the Framework

### Add a new model

1. Create `src/cambrian_stack/models/my_model.py`:

```python
from dataclasses import dataclass
from cambrian_stack.models.base import BaseModel

@dataclass
class MyModelConfig:
    depth: int
    max_seq_len: int
    vocab_size: int
    # ... your params
    
    @property
    def d_model(self) -> int:
        return self.depth * 64

class MyModel(BaseModel):
    def __init__(self, config: MyModelConfig):
        super().__init__()
        self.config = config
        # ... build layers
    
    def forward(self, x, targets=None):
        # ... forward pass
        pass
    
    def generate(self, x, max_new_tokens, **kwargs):
        # ... generation logic
        pass
```

2. Register in `src/cambrian_stack/models/__init__.py`:

```python
from cambrian_stack.models.my_model import MyModel, MyModelConfig

MODEL_REGISTRY["my_model"] = MyModel
CONFIG_REGISTRY["my_model"] = MyModelConfig
```

3. Create a config file `src/cambrian_stack/conf/my_model.yaml` with `model.type: my_model`

---

## Available Models

| Model | Type | Description |
|-------|------|-------------|
| `transformer` | Autoregressive | GPT-style causal LM |
| `diffusion_transformer` | Diffusion | Discrete diffusion with bidirectional attention |

---

## License

MIT

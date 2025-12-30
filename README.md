# Cambrian Stack

Minimal, hackable transformer experiments. For full docs, see **https://<your-pages-url>** (built from `docs/` via GitHub Pages).

## Quick start

```bash
git clone <repo-url> && cd cambrian-stack
uv venv && source .venv/bin/activate
uv pip install -e ".[ml]"   # or ".[dev]" for tooling; base install skips torch

./scripts/train_baseline_transformer.sh                # baseline AR
python -m cambrian_stack.train --config-name=diffusion_transformer  # diffusion
```

Outputs: checkpoints in `out/`, logs in `logs/`, optional W&B if `WANDB_API_KEY` is set.

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

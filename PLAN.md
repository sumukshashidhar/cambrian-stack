# Plan — Architecture Refactor for Rapid Experiments

## Goals
- Make “one-file + config” experiment changes possible.
- Decouple model/task-specific logic from the core trainer and dataset loaders.
- Simplify CLI/config plumbing to reduce friction when adding new ideas.

## Workstream A — Experiment Strategy Layer
1) Introduce `cambrian_stack/experiments/` with an `EXPERIMENT_REGISTRY`.
2) Implement `AutoregressiveExperiment` and `DiffusionExperiment` owning:
   - `build_model`, `build_dataloaders`, `training_step`, `validation_step`, optional `sample_step`.
3) Update `train.py` to select experiment via `cfg.experiment` and delegate loop callbacks.

## Workstream B — Data/Transforms Separation
1) Extract diffusion corruption + timestep logic from `data_loaders/tiny_stories.py` into `experiments/diffusion_data.py` (or `data/transforms/`).
2) Keep `tiny_stories.py` dataset-only; experiments compose dataset + transforms.

## Workstream C — CLI & Config Packaging
1) Move script entrypoints to `scripts/` (or `cambrian_stack/cli/`) as thin shims.
2) Relocate Hydra configs to `src/cambrian_stack/conf/` with groups (`model/`, `data/`, `training/`, `logging/`, `experiment/`).
3) Point Hydra `config_path` to `conf` and update scripts accordingly.

## Workstream D — API Alignment & Tests
1) Align `BaseModel`/task interface (or shift responsibility to experiments) to remove trainer conditionals.
2) Add smoke tests per experiment (CPU-fast) covering `training_step`, `validation_step`, and registry wiring.

## Workstream E — Migration & Housekeeping
1) Provide migration notes: old config keys → new grouped paths; deprecated in-package CLIs.
2) Verify existing configs run via compatibility shim or concise README update.

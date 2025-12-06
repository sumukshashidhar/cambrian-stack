Experiments
===========

Experiment strategies live in ``cambrian_stack/experiments`` and plug into the shared trainer.

Autoregressive
--------------

* File: ``experiments/autoregressive.py``
* Uses GPT-style ``Transformer`` model
* Train/eval hooks: cross-entropy LM loss, optional prompt sampling

Diffusion
---------

* File: ``experiments/diffusion.py``
* Uses ``DiffusionTransformer``
* Train/eval hooks: masked-token corruption and recovery accuracy
* Transform: ``experiments/transforms.py::corrupt_tokens``

Adding a new experiment
-----------------------

1. Create ``experiments/<name>.py`` implementing ``Experiment`` interface (``build_model``, ``build_dataloaders``, ``training_step``, ``evaluate``; optionally ``sample``).
2. Register it in ``experiments/__init__.py``.
3. Add a config under ``conf/experiment/`` and an entry config at the root of ``conf`` if desired.
4. Run with ``python -m cambrian_stack.train experiment.type=<name>`` or via a new entry config.

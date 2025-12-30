Quick Start
===========

Train the baseline transformer on TinyStories (multi-GPU if available):

.. code-block:: bash

   ./scripts/train_baseline_transformer.sh

Train on FineWeb-Edu with nanochat-like depth-32 preset:

.. code-block:: bash

   python -m cambrian_stack.train --config-name=baseline_fineweb

Single GPU minimal run:

.. code-block:: bash

   python -m cambrian_stack.train model.depth=6 training.device_batch_size=8

Evaluate a checkpoint:

.. code-block:: bash

   ./scripts/eval_checkpoint.sh out/baseline-d12/checkpoint_010000.pt

Where outputs go
----------------

* Checkpoints: ``out/``
* Logs: ``logs/``
* W&B run data: ``wandb/`` (only if enabled)

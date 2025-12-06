Configuration
=============

Hydra configs live under ``src/cambrian_stack/conf`` and are grouped by concern.

Top-level entry configs
-----------------------

* ``baseline_transformer.yaml`` — autoregressive GPT-style run
* ``diffusion_transformer.yaml`` — discrete diffusion run
* ``baseline_fineweb.yaml`` — autoregressive on FineWeb-Edu (streaming HF dataset)
* ``baseline_fineweb_nanochat.yaml`` — FineWeb-Edu with nanochat-inspired training hyperparams
* ``baseline_fineweb_nanochat_d32.yaml`` — FineWeb-Edu with nanochat-like depth-32 model and schedule

Groups
------

* ``experiment/`` — selects experiment strategy (``autoregressive`` or ``diffusion``)
* ``model/`` — architecture parameters (``transformer``, ``diffusion_transformer``)
* ``data/`` — dataset/tokenizer settings (TinyStories defaults)
* ``training/`` — optimization, schedule, eval, sampling, checkpoint cadence
* ``logging/`` — W&B and log cadence
* ``output/`` — checkpoint directory

Override examples
-----------------

Short overrides on the CLI:

.. code-block:: bash

   python -m cambrian_stack.train training.max_steps=2000 training.eval_every=200

Switch to diffusion config:

.. code-block:: bash

   python -m cambrian_stack.train --config-name=diffusion_transformer

Create a new experiment config:

.. code-block:: bash

   cp src/cambrian_stack/conf/baseline_transformer.yaml src/cambrian_stack/conf/my_experiment.yaml
   # Edit the copy, then run:
   python -m cambrian_stack.train --config-name=my_experiment

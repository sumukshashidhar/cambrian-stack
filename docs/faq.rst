FAQ
===

Do I need to install torch?
   Only if you want to run the models. Use ``pip install .[ml]`` or provide torch from your container image and stick to ``pip install .`` for the rest.

Where are the configs?
   Under ``src/cambrian_stack/conf``. The old ``configs/`` directory was removed.

How do I run diffusion?
   ``python -m cambrian_stack.train --config-name=diffusion_transformer`` or set ``experiment.type=diffusion``.

How do I change output directories?
   Override ``output.dir`` on the command line: ``python -m cambrian_stack.train output.dir=out/my-run``.

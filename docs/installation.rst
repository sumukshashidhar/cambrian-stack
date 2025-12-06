Installation
============

Pick the dependency profile that matches your environment.

Base install (no torch)
-----------------------

.. code-block:: bash

   pip install .

ML extras (torch + accelerate)
------------------------------

.. code-block:: bash

   pip install .[ml]

Development extras (ML + testing tools)
---------------------------------------

.. code-block:: bash

   pip install .[dev]

Environment
-----------

.. code-block:: bash

   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"   # or choose one of the profiles above

Optional environment variables
------------------------------

``HF_TOKEN`` for gated datasets, ``WANDB_API_KEY`` and ``WANDB_PROJECT`` for W&B logging. Create a ``.env`` file in the project root if you prefer dotenv loading.

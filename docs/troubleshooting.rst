Troubleshooting
===============

Training hangs after model initialization
-----------------------------------------

If training appears to freeze after you see logs like:

.. code-block:: text

   INFO | Loaded tokenizer 'gpt2' (vocab=50257)
   INFO | Initialized Transformer: depth=12, d_model=768...
   WARNING | total_batch_size=262144 not divisible by tokens_per_step...
   # Then nothing happens for many minutes

**Likely causes:**

1. **Network issues with HuggingFace**: The streaming dataloader needs to connect to HuggingFace servers. Check your network connectivity.

2. **Multi-worker data loading**: Try setting ``num_workers: 0`` in your data config to rule out multi-process issues.

3. **Missing HF_TOKEN**: If using a gated dataset, ensure ``HF_TOKEN`` is set in your environment.

**Quick fix:**

.. code-block:: bash

   # Test with single-threaded data loading
   python -m cambrian_stack.train data.num_workers=0


pin_memory warning on CPU-only systems
--------------------------------------

You may see:

.. code-block:: text

   UserWarning: 'pin_memory' argument is set as true but no accelerator is found

This is harmless - PyTorch will simply not pin memory when no GPU is available.
Training will proceed normally on CPU.


Out of memory errors
--------------------

If you run out of GPU memory:

1. Reduce ``training.device_batch_size``
2. Reduce ``model.max_seq_len``
3. Reduce model size: ``model.depth`` or ``model.d_model``

.. code-block:: bash

   # Minimal memory configuration
   python -m cambrian_stack.train \
       training.device_batch_size=4 \
       model.max_seq_len=256 \
       model.depth=6


WandB issues
------------

If W&B logging fails:

1. Ensure ``WANDB_API_KEY`` is set
2. Check network connectivity to wandb.ai
3. Try ``wandb login`` to re-authenticate

To disable W&B entirely:

.. code-block:: bash

   python -m cambrian_stack.train logging.wandb_project=null

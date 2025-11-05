from __future__ import annotations

import os
import torch
import torch.distributed as dist

def setup_ddp():
    if "RANK" not in os.environ: return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def cleanup_ddp():
    if dist.is_initialized(): dist.destroy_process_group()

def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

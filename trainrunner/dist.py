from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def setup_distributed() -> DistInfo:
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    is_distributed = world_size > 1

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return DistInfo(
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier(info: DistInfo) -> None:
    if info.is_distributed and dist.is_initialized():
        dist.barrier()


def all_reduce_sum(info: DistInfo, tensor: torch.Tensor) -> torch.Tensor:
    if not info.is_distributed:
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def broadcast_object(info: DistInfo, obj: Any, src: int = 0) -> Any:
    if not info.is_distributed:
        return obj
    obj_list = [obj] if info.rank == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


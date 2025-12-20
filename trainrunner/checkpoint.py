from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch

from .dist import DistInfo
from .utils import ensure_dir


@dataclass
class CheckpointPaths:
    checkpoints_dir: str
    latest_path: str
    best_path: str


def default_checkpoint_paths(run_dir: str) -> CheckpointPaths:
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    return CheckpointPaths(
        checkpoints_dir=checkpoints_dir,
        latest_path=os.path.join(checkpoints_dir, "latest.pth"),
        best_path=os.path.join(checkpoints_dir, "best.pth"),
    )


def atomic_torch_save(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def save_checkpoint(
    dist: DistInfo,
    paths: CheckpointPaths,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    lr_scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_valid_loss: Optional[float],
    best_metric: str = "valid/loss",
    best_mode: str = "min",
    best_metric_value: Optional[float] = None,
    extra_state: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
) -> None:
    if not dist.is_main_process:
        return
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_valid_loss": best_valid_loss,
        "best_metric": best_metric,
        "best_mode": best_mode,
        "best_metric_value": best_metric_value,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra_state": extra_state or {},
    }
    atomic_torch_save(state, paths.latest_path)
    if is_best:
        atomic_torch_save(state, paths.best_path)


def load_checkpoint(path: str, map_location: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=map_location)

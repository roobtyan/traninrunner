from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .dist import DistInfo
from .utils import ensure_dir, json_dumps


@dataclass
class LogEvent:
    phase: str  # "train" | "valid"
    epoch: int
    step_in_epoch: int
    global_step: int
    loss: float
    metrics: Dict[str, float]
    wall_time_s: float
    extra: Dict[str, Any]


class MetricsLogger:
    def __init__(self, dist: DistInfo, run_dir: str, filename: str = "metrics.jsonl") -> None:
        self._dist = dist
        self._run_dir = run_dir
        self._path = os.path.join(run_dir, filename)
        if self._dist.is_main_process:
            ensure_dir(run_dir)

    @property
    def path(self) -> str:
        return self._path

    def write(self, event: LogEvent) -> None:
        if not self._dist.is_main_process:
            return
        record = {
            "phase": event.phase,
            "epoch": event.epoch,
            "step_in_epoch": event.step_in_epoch,
            "global_step": event.global_step,
            "loss": event.loss,
            "metrics": event.metrics,
            "wall_time_s": event.wall_time_s,
            **(event.extra or {}),
        }
        line = json_dumps(record)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def print_iter(
        self,
        phase: str,
        epoch: int,
        step_in_epoch: int,
        global_step: int,
        loss: float,
        metrics: Dict[str, float],
    ) -> None:
        if not self._dist.is_main_process:
            return
        metric_str = ""
        if metrics:
            parts = [f"{k}={v:.4f}" for k, v in sorted(metrics.items())]
            metric_str = " " + " ".join(parts)
        print(f"[{phase}] epoch={epoch} iter={step_in_epoch} step={global_step} loss={loss:.6f}{metric_str}", flush=True)

    def print_epoch(self, phase: str, epoch: int, avg_loss: float, metrics: Dict[str, float]) -> None:
        if not self._dist.is_main_process:
            return
        metric_str = ""
        if metrics:
            parts = [f"{k}={v:.4f}" for k, v in sorted(metrics.items())]
            metric_str = " " + " ".join(parts)
        print(f"[{phase}] epoch={epoch} avg_loss={avg_loss:.6f}{metric_str}", flush=True)


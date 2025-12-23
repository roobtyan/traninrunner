from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .utils import deep_merge, nested_set, parse_scalar


@dataclass
class RunnerConfig:
    task_entry: str = ""
    work_dir: str = "./runs"
    run_name: Optional[str] = None
    mode: str = "train"  # train | val | infer
    epochs: int = 1
    valid_every_n_epoch: int = 1
    log_every_n_iter: int = 50
    seed: int = 1337
    deterministic: bool = False
    resume: Optional[str] = None
    resume_optimizer: bool = True
    resume_scheduler: bool = True
    resume_scaler: bool = True
    plugins: List[str] = field(default_factory=list)
    max_total_train_iters: Optional[int] = None
    best_metric: str = "valid/loss"
    best_mode: str = "min"  # "min" | "max"
    freeze: Optional[Dict[str, Any]] = None


def _load_config_file(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            data = yaml.safe_load(f) or {}
        elif path.endswith(".json"):
            data = json.load(f) or {}
        else:
            raise ValueError(f"Unsupported config format: {path} (expected .yaml/.yml/.json)")
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping (got {type(data)})")
    return data


def _parse_dotlist(tokens: List[str]) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    for t in tokens:
        if t == "--":
            continue
        if t.startswith("+"):
            t = t[1:]
        if t.startswith("--"):
            t = t[2:]
        if "=" not in t:
            raise ValueError(f"Expected dotlist override 'a.b=val', got: {t}")
        k, v = t.split("=", 1)
        nested_set(d, k, parse_scalar(v))
    return d


def _split_runner_and_task(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    runner_cfg = cfg.get("runner", {}) if isinstance(cfg.get("runner"), dict) else {}
    task_cfg = cfg.get("task", {}) if isinstance(cfg.get("task"), dict) else {}

    top_level_task_kwargs: Dict[str, Any] = {}
    for k, v in cfg.items():
        if k in {"runner", "task"}:
            continue
        top_level_task_kwargs[k] = v

    task_entry = task_cfg.get("entry", "")
    task_kwargs = task_cfg.get("kwargs", {}) if isinstance(task_cfg.get("kwargs"), dict) else {}
    for k, v in task_cfg.items():
        if k in {"entry", "kwargs"}:
            continue
        task_kwargs[k] = v

    task_kwargs = deep_merge(task_kwargs, top_level_task_kwargs)
    runner_cfg = dict(runner_cfg)
    runner_cfg.setdefault("task_entry", task_entry)
    return runner_cfg, task_kwargs


def parse_config(argv: Optional[List[str]] = None) -> Tuple[RunnerConfig, Dict[str, Any], Dict[str, Any]]:
    parser = argparse.ArgumentParser(description="trainrunner")
    parser.add_argument("--config", type=str, default=None, help="Path to config (.yaml/.yml/.json)")
    parser.add_argument("--task-entry", type=str, default=None, help="Python path: module:attr")
    parser.add_argument("--work-dir", type=str, default=None, help="Run root directory")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run folder name override")
    parser.add_argument("--mode", type=str, default=None, choices=["train", "val", "infer"])
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs")
    parser.add_argument("--valid-every-n-epoch", type=int, default=None)
    parser.add_argument("--log-every-n-iter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--resume-optimizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume-scheduler", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume-scaler", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plugin", action="append", default=None, help="Plugin path (repeatable): module:PluginClass")
    parser.add_argument("--max-total-train-iters", type=int, default=None)
    parser.add_argument("--best-metric", type=str, default=None, help="Metric key used for best checkpoint (e.g. valid/acc1)")
    parser.add_argument("--best-mode", type=str, default=None, choices=["min", "max"], help="Best checkpoint mode: min|max")
    args, unknown = parser.parse_known_args(argv)

    file_cfg = _load_config_file(args.config) if args.config else {}

    flags_cfg: Dict[str, Any] = {"runner": {}, "task": {}}
    if args.task_entry is not None:
        flags_cfg["task"]["entry"] = args.task_entry
    if args.work_dir is not None:
        flags_cfg["runner"]["work_dir"] = args.work_dir
    if args.run_name is not None:
        flags_cfg["runner"]["run_name"] = args.run_name
    if args.mode is not None:
        flags_cfg["runner"]["mode"] = args.mode
    if args.epochs is not None:
        flags_cfg["runner"]["epochs"] = args.epochs
    if args.valid_every_n_epoch is not None:
        flags_cfg["runner"]["valid_every_n_epoch"] = args.valid_every_n_epoch
    if args.log_every_n_iter is not None:
        flags_cfg["runner"]["log_every_n_iter"] = args.log_every_n_iter
    if args.seed is not None:
        flags_cfg["runner"]["seed"] = args.seed
    if args.deterministic:
        flags_cfg["runner"]["deterministic"] = True
    if args.resume is not None:
        flags_cfg["runner"]["resume"] = args.resume
    if args.resume_optimizer is not None:
        flags_cfg["runner"]["resume_optimizer"] = args.resume_optimizer
    if args.resume_scheduler is not None:
        flags_cfg["runner"]["resume_scheduler"] = args.resume_scheduler
    if args.resume_scaler is not None:
        flags_cfg["runner"]["resume_scaler"] = args.resume_scaler
    if args.plugin:
        flags_cfg["runner"]["plugins"] = args.plugin
    if args.max_total_train_iters is not None:
        flags_cfg["runner"]["max_total_train_iters"] = args.max_total_train_iters
    if args.best_metric is not None:
        flags_cfg["runner"]["best_metric"] = args.best_metric
    if args.best_mode is not None:
        flags_cfg["runner"]["best_mode"] = args.best_mode

    dot_cfg = _parse_dotlist(unknown)

    merged = deep_merge(deep_merge(deep_merge({}, file_cfg), flags_cfg), dot_cfg)
    runner_dict, task_kwargs = _split_runner_and_task(merged)

    runner = RunnerConfig()
    for k, v in runner_dict.items():
        if hasattr(runner, k):
            setattr(runner, k, v)

    if not runner.task_entry:
        raise ValueError("Missing task.entry (set via config or --task-entry or dotlist task.entry=...)")

    resolved = {"merged": merged, "runner": asdict(runner), "task_kwargs": task_kwargs}
    return runner, task_kwargs, resolved

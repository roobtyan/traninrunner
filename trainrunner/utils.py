from __future__ import annotations

import json
import os
import random
import time
from dataclasses import is_dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch


def utcnow_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def generate_run_id(prefix: str = "run") -> str:
    rnd = f"{random.getrandbits(32):08x}"
    return f"{prefix}-{utcnow_compact()}-{rnd}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def to_builtin(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=to_builtin, separators=(",", ":"))


def parse_scalar(value: str) -> Any:
    v = value.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    if v.lower() in {"null", "none"}:
        return None
    if v.startswith("{") or v.startswith("["):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return value
    try:
        if v.startswith("0") and len(v) > 1 and v[1].isdigit():
            raise ValueError
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return value


def nested_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [p for p in dotted_key.split(".") if p]
    if not parts:
        return
    cur: Dict[str, Any] = d
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def now_s() -> float:
    return time.time()


def pretty_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m}m{s:.0f}s"


def detach_loss(loss: torch.Tensor) -> float:
    return float(loss.detach().float().cpu().item())


def detach_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (metrics or {}).items():
        if isinstance(v, torch.Tensor):
            out[k] = float(v.detach().float().cpu().item())
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    return out


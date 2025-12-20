from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class FreezeConfig:
    patterns: Optional[List[str]]
    bn_eval: bool = False
    strict: bool = True


@dataclass
class FreezeState:
    matched: List[str]
    modules: List[nn.Module]
    bn_eval: bool

    def enforce_train_mode(self) -> None:
        if not self.bn_eval:
            return
        for m in self.modules:
            _set_bn_eval(m)


def parse_freeze_config(obj: Any) -> Optional[FreezeConfig]:
    if obj is None:
        return None

    if isinstance(obj, str):
        return FreezeConfig(patterns=[obj])

    if isinstance(obj, (list, tuple)):
        return FreezeConfig(patterns=[str(x) for x in obj])

    if isinstance(obj, dict):
        raw_patterns = obj.get("targets", None)
        if "targets" not in obj:
            raw_patterns = obj.get("freeze", None)

        patterns: Optional[List[str]]
        if raw_patterns is None:
            patterns = None
        elif isinstance(raw_patterns, (list, tuple)):
            patterns = [str(x) for x in raw_patterns]
        else:
            patterns = [str(raw_patterns)]

        bn_eval = bool(obj.get("bn_eval", False))
        strict = bool(obj.get("strict", True))
        return FreezeConfig(patterns=patterns, bn_eval=bn_eval, strict=strict)

    raise TypeError(f"freeze config must be dict|list|str|None, got {type(obj)}")


def setup_freeze(model: nn.Module, cfg: Optional[FreezeConfig], task_kwargs: Dict[str, Any]) -> Optional[FreezeState]:
    if cfg is None:
        return None

    patterns = cfg.patterns
    if patterns is None:
        patterns = _get_default_freeze_patterns(model, task_kwargs)

    patterns = list(patterns or [])
    if not patterns:
        return None

    matched, modules = _resolve_target_modules(model, patterns, strict=cfg.strict)
    if not modules:
        return None
    _freeze_modules(modules)
    state = FreezeState(matched=matched, modules=modules, bn_eval=cfg.bn_eval)
    state.enforce_train_mode()
    return state


def _get_default_freeze_patterns(model: nn.Module, task_kwargs: Dict[str, Any]) -> List[str]:
    fn = getattr(model, "get_default_freeze_targets", None)
    if callable(fn):
        out = fn(**task_kwargs)
        if out is None:
            return []
        if isinstance(out, (list, tuple)):
            return [str(x) for x in out]
        return [str(out)]
    return []


def _get_named_target_map(model: nn.Module) -> Optional[Dict[str, nn.Module]]:
    fn = getattr(model, "get_freeze_targets", None)
    if callable(fn):
        targets = fn()
        if targets is None:
            return {}
        if not isinstance(targets, dict):
            raise TypeError(f"get_freeze_targets() must return dict[str, nn.Module], got {type(targets)}")
        out: Dict[str, nn.Module] = {}
        for k, v in targets.items():
            if not isinstance(k, str):
                raise TypeError(f"get_freeze_targets() keys must be str, got {type(k)}")
            if not isinstance(v, nn.Module):
                raise TypeError(f"get_freeze_targets()['{k}'] must be nn.Module, got {type(v)}")
            out[k] = v
        return out

    targets = getattr(model, "freeze_targets", None)
    if isinstance(targets, dict):
        out2: Dict[str, nn.Module] = {}
        for k, v in targets.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, nn.Module):
                out2[k] = v
        return out2

    return None


def _resolve_target_modules(model: nn.Module, patterns: Sequence[str], strict: bool) -> Tuple[List[str], List[nn.Module]]:
    named_map = _get_named_target_map(model)
    if named_map is not None:
        candidates = sorted(named_map.keys())
        resolver = lambda name: named_map[name]
    else:
        candidates = sorted([n for n, _m in model.named_modules() if n])
        resolver = lambda name: _get_submodule(model, name)

    matched_names: List[str] = []
    modules: List[nn.Module] = []
    seen_mods: set[int] = set()

    for pat in patterns:
        matches = [n for n in candidates if fnmatch.fnmatchcase(n, pat)]
        if not matches:
            if strict:
                sample = ", ".join(candidates[:25]) + (" ..." if len(candidates) > 25 else "")
                raise ValueError(f"freeze target pattern {pat!r} matched nothing; candidates: {sample}")
            continue
        for name in matches:
            m = resolver(name)
            mid = id(m)
            if mid in seen_mods:
                continue
            seen_mods.add(mid)
            matched_names.append(name)
            modules.append(m)

    return matched_names, modules


def _get_submodule(model: nn.Module, path: str) -> nn.Module:
    if hasattr(model, "get_submodule"):
        return model.get_submodule(path)
    cur: nn.Module = model
    for part in path.split("."):
        if not part:
            continue
        if not hasattr(cur, part):
            raise AttributeError(f"Module has no submodule at '{path}' (missing '{part}' under '{type(cur).__name__}')")
        cur = getattr(cur, part)
        if not isinstance(cur, nn.Module):
            raise AttributeError(f"Attribute '{part}' under '{type(model).__name__}' is not an nn.Module")
    return cur


def _freeze_modules(modules: Sequence[nn.Module]) -> None:
    seen_params: set[int] = set()
    for m in modules:
        for p in m.parameters(recurse=True):
            pid = id(p)
            if pid in seen_params:
                continue
            seen_params.add(pid)
            p.requires_grad_(False)


def _set_bn_eval(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) or isinstance(m, torch.nn.SyncBatchNorm):
            m.eval()

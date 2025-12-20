from __future__ import annotations

import importlib
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Dict, List, Optional


class Plugin:
    def wrap_model(self, model, cfg: Dict[str, Any]):
        return model

    def train_step_ctx(self, model, cfg: Dict[str, Any]) -> ContextManager:
        return nullcontext()

    def valid_step_ctx(self, model, cfg: Dict[str, Any]) -> ContextManager:
        return nullcontext()

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        return

    def on_checkpoint_save(self, checkpoint: Dict[str, Any]) -> None:
        return


def _import_from_path(path: str):
    if ":" in path:
        mod, attr = path.split(":", 1)
    else:
        mod, attr = path, None
    module = importlib.import_module(mod)
    if attr is None:
        return module
    return getattr(module, attr)


def load_plugins(paths: List[str]) -> List[Plugin]:
    plugins: List[Plugin] = []
    for p in paths or []:
        obj = _import_from_path(p)
        plugin = obj() if callable(obj) else obj
        if not isinstance(plugin, Plugin):
            raise TypeError(f"Plugin '{p}' must be an instance of trainrunner.plugins.Plugin (got {type(plugin)})")
        plugins.append(plugin)
    return plugins


def plugin_state_dict(plugins: List[Plugin]) -> Dict[str, Any]:
    return {type(p).__name__: p.state_dict() for p in plugins}


def plugin_load_state_dict(plugins: List[Plugin], state: Dict[str, Any]) -> None:
    state = state or {}
    for p in plugins:
        p.load_state_dict(state.get(type(p).__name__, {}))


def plugin_on_checkpoint_save(plugins: List[Plugin], checkpoint: Dict[str, Any]) -> None:
    for p in plugins:
        p.on_checkpoint_save(checkpoint)


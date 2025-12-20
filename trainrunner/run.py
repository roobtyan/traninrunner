from __future__ import annotations

import importlib
import os
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .checkpoint import default_checkpoint_paths, load_checkpoint, save_checkpoint
from .config import RunnerConfig, parse_config
from .dist import DistInfo, all_reduce_sum, barrier, broadcast_object, cleanup_distributed, setup_distributed
from .freeze import parse_freeze_config, setup_freeze
from .metrics import LogEvent, MetricsLogger
from .plugins import (
    Plugin,
    load_plugins,
    plugin_load_state_dict,
    plugin_on_checkpoint_save,
    plugin_state_dict,
)
from .utils import detach_loss, detach_metrics, ensure_dir, generate_run_id, json_dumps, now_s, pretty_seconds, seed_everything


def _looks_like_freeze_cfg(obj: Any) -> bool:
    if obj is None:
        return False
    if isinstance(obj, (str, list, tuple)):
        return True
    if isinstance(obj, dict):
        return any(k in obj for k in ("targets", "bn_eval", "strict", "freeze"))
    return False


def _import_from_path(path: str):
    if ":" in path:
        mod, attr = path.split(":", 1)
    else:
        mod, attr = path, None
    module = importlib.import_module(mod)
    if attr is None:
        return module
    return getattr(module, attr)


def _create_task(entry: str, kwargs: Dict[str, Any]):
    obj = _import_from_path(entry)
    if isinstance(obj, type):
        return obj(**kwargs)
    if callable(obj):
        return obj(**kwargs)
    raise TypeError(f"task.entry must resolve to a class or callable factory, got {type(obj)}")


def _maybe_set_sampler_epoch(sampler: Any, epoch: int) -> None:
    if sampler is None:
        return
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _reduce_epoch_loss(dist: DistInfo, sum_loss: float, count: int) -> float:
    device = dist.device
    t = torch.tensor([sum_loss, float(count)], device=device, dtype=torch.float64)
    t = all_reduce_sum(dist, t)
    total_loss, total_count = float(t[0].item()), float(t[1].item())
    return total_loss / max(total_count, 1.0)


def _metric_value(m: Any) -> float:
    if isinstance(m, torch.Tensor):
        return float(m.detach().float().cpu().item())
    if isinstance(m, (int, float)):
        return float(m)
    return 0.0


def _reduce_correct_counts(dist: DistInfo, acc1_correct: float, acc5_correct: float, n: float) -> Tuple[float, float, float]:
    device = dist.device
    t = torch.tensor([acc1_correct, acc5_correct, n], device=device, dtype=torch.float64)
    t = all_reduce_sum(dist, t)
    return float(t[0].item()), float(t[1].item()), float(t[2].item())


def _compare_best(mode: str, value: float, best_value: Optional[float]) -> bool:
    if best_value is None:
        return True
    if mode == "min":
        return value < best_value
    if mode == "max":
        return value > best_value
    raise ValueError(f"best_mode must be 'min' or 'max', got {mode!r}")


def _device_string(dist: DistInfo) -> str:
    if dist.device.type == "cuda":
        return f"cuda:{dist.local_rank}"
    return "cpu"


def main(argv: Optional[list[str]] = None) -> None:
    runner, task_kwargs, resolved = parse_config(argv)
    dist = setup_distributed()

    try:
        _main(runner, task_kwargs, resolved, dist)
    finally:
        cleanup_distributed()


def _main(runner: RunnerConfig, task_kwargs: Dict[str, Any], resolved: Dict[str, Any], dist: DistInfo) -> None:
    seed_everything(runner.seed + dist.rank, deterministic=runner.deterministic)

    run_id = runner.run_name or (generate_run_id("train") if dist.is_main_process else None)
    run_id = broadcast_object(dist, run_id, src=0)

    run_dir = os.path.join(runner.work_dir, run_id)
    if dist.is_main_process:
        ensure_dir(run_dir)
        with open(os.path.join(run_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
            f.write(json_dumps(resolved) + "\n")

    barrier(dist)

    logger = MetricsLogger(dist=dist, run_dir=run_dir, filename="metrics.jsonl")
    ckpt_paths = default_checkpoint_paths(run_dir)

    plugins = load_plugins(runner.plugins)

    freeze_raw = getattr(runner, "freeze", None)
    if freeze_raw is None and _looks_like_freeze_cfg(task_kwargs.get("freeze")):
        freeze_raw = task_kwargs.pop("freeze")
    freeze_cfg = parse_freeze_config(freeze_raw)

    task = _create_task(runner.task_entry, task_kwargs)
    for p in plugins:
        task = p.wrap_model(task, cfg={"runner": asdict(runner), "task": task_kwargs})

    freeze_state = setup_freeze(task, freeze_cfg, task_kwargs)
    if freeze_cfg is not None and freeze_state is not None:
        any_trainable = any(p.requires_grad for p in task.parameters())
        if not any_trainable:
            raise ValueError("freeze config froze all parameters; at least one parameter must remain trainable")
    if freeze_state is not None and dist.is_main_process:
        frozen_param_ids: set[int] = set()
        for m in freeze_state.modules:
            for param in m.parameters(recurse=True):
                frozen_param_ids.add(id(param))
        print(
            f"[freeze] matched={freeze_state.matched} bn_eval={freeze_state.bn_eval} frozen_params={len(frozen_param_ids)}",
            flush=True,
        )

    task.to(dist.device)

    if dist.is_distributed:
        task = DDP(task, device_ids=[dist.local_rank] if dist.device.type == "cuda" else None)

    build_train_dataloader = task.module.build_train_dataloader if isinstance(task, DDP) else task.build_train_dataloader
    build_valid_dataloader = task.module.build_valid_dataloader if isinstance(task, DDP) else task.build_valid_dataloader
    train_dl, train_sampler = build_train_dataloader(dist.is_distributed, **task_kwargs)
    valid_dl, valid_sampler = build_valid_dataloader(dist.is_distributed, **task_kwargs)

    configure_optimizers = task.module.configure_optimizers if isinstance(task, DDP) else task.configure_optimizers
    optimizer, lr_scheduler, scheduler_step = configure_optimizers(**task_kwargs)
    if optimizer is None:
        raise ValueError("configure_optimizers must return a non-None optimizer")
    if scheduler_step not in {"iter", "epoch", "none"}:
        raise ValueError(f"configure_optimizers must return scheduler_step in {{'iter','epoch','none'}}, got {scheduler_step}")

    scaler = None  # reserved for AMP/QAT plugins; stored in checkpoint if provided by plugins

    start_epoch = 0
    global_step = 0
    best_valid_loss: Optional[float] = None
    best_metric = str(getattr(runner, "best_metric", "valid/loss") or "valid/loss")
    best_mode = str(getattr(runner, "best_mode", "min") or "min").lower()
    best_value: Optional[float] = None

    if runner.resume:
        ckpt = load_checkpoint(runner.resume, map_location="cpu")
        if dist.is_main_process:
            print(f"[resume] loading checkpoint: {runner.resume}", flush=True)
        model_obj = task.module if isinstance(task, DDP) else task
        model_obj.load_state_dict(ckpt["model"], strict=True)
        if optimizer is not None and runner.resume_optimizer and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if lr_scheduler is not None and runner.resume_scheduler and ckpt.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if scaler is not None and runner.resume_scaler and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        plugin_load_state_dict(plugins, (ckpt.get("extra_state") or {}).get("plugins", {}))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_valid_loss = ckpt.get("best_valid_loss", None)
        best_metric = str(ckpt.get("best_metric", best_metric))
        best_mode = str(ckpt.get("best_mode", best_mode)).lower()
        best_value = ckpt.get("best_metric_value", None)
        if best_value is None:
            best_value = best_valid_loss

    if dist.is_main_process:
        ddp_str = f"ddp(world_size={dist.world_size}, rank={dist.rank})" if dist.is_distributed else "single"
        print(f"[run] id={run_id} device={_device_string(dist)} {ddp_str}", flush=True)
        print(f"[run] run_dir={run_dir}", flush=True)
        print(f"[run] metrics={logger.path}", flush=True)

    model_obj = task.module if isinstance(task, DDP) else task
    training_step = model_obj.training_step
    validation_step = model_obj.validation_step

    max_total_train_iters = runner.max_total_train_iters

    wall_start = now_s()
    for epoch in range(start_epoch, runner.epochs):
        _maybe_set_sampler_epoch(train_sampler, epoch)
        model_obj.train()
        if freeze_state is not None:
            freeze_state.enforce_train_mode()

        train_sum_loss = 0.0
        train_count = 0
        train_acc1_correct = 0.0
        train_acc5_correct = 0.0
        train_n = 0.0

        if dist.is_main_process:
            pbar = tqdm(total=len(train_dl), desc=f"train epoch {epoch}", dynamic_ncols=True)
        else:
            pbar = None

        for step_in_epoch, batch in enumerate(train_dl):
            global_step += 1
            with ExitStack() as stack:
                for p in plugins:
                    stack.enter_context(p.train_step_ctx(model_obj, cfg={"runner": asdict(runner), "task": task_kwargs}))
                out = training_step(batch, ctx={"epoch": epoch, "step_in_epoch": step_in_epoch, "global_step": global_step, "device": dist.device})

            loss = out["loss"]
            metrics = out.get("metrics", {})
            if not isinstance(loss, torch.Tensor):
                raise TypeError("training_step must return dict(loss=Tensor, ...)")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None and scheduler_step == "iter":
                lr_scheduler.step()

            loss_val = detach_loss(loss)
            train_sum_loss += loss_val
            train_count += 1
            if metrics:
                train_acc1_correct += _metric_value(metrics.get("acc1_correct", 0.0))
                train_acc5_correct += _metric_value(metrics.get("acc5_correct", 0.0))
                train_n += _metric_value(metrics.get("n", 0.0))

            if dist.is_main_process and pbar is not None:
                pbar.update(1)

            if runner.log_every_n_iter > 0 and (step_in_epoch % runner.log_every_n_iter == 0):
                m = detach_metrics(metrics)
                m = {k: v for k, v in m.items() if k != "n" and not k.endswith("_correct")}
                logger.print_iter("train", epoch, step_in_epoch, global_step, loss_val, m)
                logger.write(
                    LogEvent(
                        phase="train",
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        global_step=global_step,
                        loss=loss_val,
                        metrics=m,
                        wall_time_s=now_s() - wall_start,
                        extra={},
                    )
                )

            if max_total_train_iters is not None and global_step >= max_total_train_iters:
                break

        if dist.is_main_process and pbar is not None:
            pbar.close()

        train_avg_loss = _reduce_epoch_loss(dist, train_sum_loss, train_count)
        train_acc1_correct, train_acc5_correct, train_n = _reduce_correct_counts(
            dist, train_acc1_correct, train_acc5_correct, train_n
        )
        train_metrics: Dict[str, float] = {}
        if train_n > 0:
            train_metrics["acc1"] = train_acc1_correct / train_n
            train_metrics["acc5"] = train_acc5_correct / train_n
        logger.print_epoch("train", epoch, train_avg_loss, train_metrics)
        logger.write(
            LogEvent(
                phase="train_epoch",
                epoch=epoch,
                step_in_epoch=-1,
                global_step=global_step,
                loss=train_avg_loss,
                metrics=train_metrics,
                wall_time_s=now_s() - wall_start,
                extra={},
            )
        )

        do_valid = ((epoch + 1) % runner.valid_every_n_epoch == 0)
        valid_avg_loss = None
        valid_metrics: Dict[str, float] = {}
        if do_valid and valid_dl is not None:
            _maybe_set_sampler_epoch(valid_sampler, epoch)
            model_obj.eval()

            valid_sum_loss = 0.0
            valid_count = 0
            valid_acc1_correct = 0.0
            valid_acc5_correct = 0.0
            valid_n = 0.0
            with torch.no_grad():
                if dist.is_main_process:
                    vbar = tqdm(total=len(valid_dl), desc=f"valid epoch {epoch}", dynamic_ncols=True)
                else:
                    vbar = None
                for v_step, v_batch in enumerate(valid_dl):
                    with ExitStack() as stack:
                        for p in plugins:
                            stack.enter_context(
                                p.valid_step_ctx(model_obj, cfg={"runner": asdict(runner), "task": task_kwargs})
                            )
                        out = validation_step(v_batch, ctx={"epoch": epoch, "step_in_epoch": v_step, "global_step": global_step, "device": dist.device})

                    v_loss = out["loss"]
                    if not isinstance(v_loss, torch.Tensor):
                        raise TypeError("validation_step must return dict(loss=Tensor, ...)")
                    v_loss_val = detach_loss(v_loss)
                    valid_sum_loss += v_loss_val
                    valid_count += 1
                    v_metrics = out.get("metrics", {}) or {}
                    if v_metrics:
                        valid_acc1_correct += _metric_value(v_metrics.get("acc1_correct", 0.0))
                        valid_acc5_correct += _metric_value(v_metrics.get("acc5_correct", 0.0))
                        valid_n += _metric_value(v_metrics.get("n", 0.0))
                    if dist.is_main_process and vbar is not None:
                        vbar.update(1)
                if dist.is_main_process and vbar is not None:
                    vbar.close()

            valid_avg_loss = _reduce_epoch_loss(dist, valid_sum_loss, valid_count)
            valid_acc1_correct, valid_acc5_correct, valid_n = _reduce_correct_counts(
                dist, valid_acc1_correct, valid_acc5_correct, valid_n
            )
            if valid_n > 0:
                valid_metrics["acc1"] = valid_acc1_correct / valid_n
                valid_metrics["acc5"] = valid_acc5_correct / valid_n
            logger.print_epoch("valid", epoch, valid_avg_loss, valid_metrics)
            logger.write(
                LogEvent(
                    phase="valid_epoch",
                    epoch=epoch,
                    step_in_epoch=-1,
                    global_step=global_step,
                    loss=valid_avg_loss,
                    metrics=valid_metrics,
                    wall_time_s=now_s() - wall_start,
                    extra={},
                )
            )

        is_best = False
        metric_map: Dict[str, Optional[float]] = {
            "train/loss": train_avg_loss,
            "train/acc1": train_metrics.get("acc1", None),
            "train/acc5": train_metrics.get("acc5", None),
            "valid/loss": valid_avg_loss,
            "valid/acc1": valid_metrics.get("acc1", None),
            "valid/acc5": valid_metrics.get("acc5", None),
        }
        candidate = metric_map.get(best_metric, None)
        if candidate is not None and _compare_best(best_mode, float(candidate), best_value):
            best_value = float(candidate)
            is_best = True
            if best_metric == "valid/loss":
                best_valid_loss = valid_avg_loss

        extra_state = {"plugins": plugin_state_dict(plugins)}
        checkpoint_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "best_valid_loss": best_valid_loss,
            "best_metric": best_metric,
            "best_mode": best_mode,
            "best_metric_value": best_value,
            "run_id": run_id,
        }
        plugin_on_checkpoint_save(plugins, checkpoint_payload)

        save_checkpoint(
            dist=dist,
            paths=ckpt_paths,
            model=model_obj,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_valid_loss=best_valid_loss,
            best_metric=best_metric,
            best_mode=best_mode,
            best_metric_value=best_value,
            extra_state=extra_state,
            is_best=is_best,
        )

        if dist.is_main_process:
            took = pretty_seconds(now_s() - wall_start)
            best_str = " (best)" if is_best else ""
            v_str = f" valid/loss={valid_avg_loss:.6f}" if valid_avg_loss is not None else ""
            best_str2 = f"{best_metric}={best_value}" if best_value is not None else f"{best_metric}=None"
            print(f"[ckpt] saved latest.pth{best_str}; best={best_str2}{v_str} elapsed={took}", flush=True)

        if max_total_train_iters is not None and global_step >= max_total_train_iters:
            break


if __name__ == "__main__":
    main()

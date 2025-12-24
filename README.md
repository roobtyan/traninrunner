# trainrunner

`trainrunner` is a lightweight, script-style PyTorch training runner with:

- DDP via `torchrun`
- Resume from checkpoints
- Epoch-based validation
- Iter-based stdout logging + `metrics.jsonl`
- `latest.pth` + `best.pth` checkpoint convention (best by `valid/loss`, lower is better)
- Pluggable hooks/contexts to mount features like QAT (no built-in QAT dependency)

## Install

From the repository root:

```bash
pip install -e .
```

## Run (single node)

```bash
python -m trainrunner.run --config configs/minimal.yaml
```

## Validation-only / Inference-only

```bash
python -m trainrunner.run --config configs/minimal.yaml --mode val --resume /path/to/latest.pth
python -m trainrunner.run --config configs/minimal.yaml --mode infer --resume /path/to/latest.pth
```

## Run (DDP, recommended wrapper)

```bash
bash scripts/train.sh --config configs/minimal.yaml
```

## Multi-node (DDP)

Use standard env vars (scheme A):

```bash
export NNODES=2
export NODE_RANK=0              # 0..NNODES-1
export MASTER_ADDR=10.0.0.10
export MASTER_PORT=29500
bash scripts/train.sh --config configs/minimal.yaml
```

## Config

Config files support YAML/JSON. Merge precedence:

`defaults < config file < CLI flags < dotlist overrides`

Dotlist overrides are passed after `--`:

```bash
python -m trainrunner.run --config config.yaml -- optim.lr=1e-3 data.batch_size=64
```

## Freeze (optional)

You can freeze part of the model by passing a `freeze` config under `runner` (so it won't be forwarded into your task constructor). Targets are treated as glob patterns and are resolved in this order:

1. If the task defines `get_freeze_targets() -> dict[str, nn.Module]`, patterns match those keys.
2. Otherwise, patterns match `named_modules()` paths (e.g. `backbone`, `model.layer1`).

Example:

```yaml
runner:
  freeze:
    targets: ["backbone"]
    bn_eval: true   # keep BatchNorm in frozen targets in eval() during training
    strict: true    # error if a pattern matches nothing
```

## Task Interface

Provide `task.entry=some.module:TaskClassOrFactory`. The resolved object must satisfy:

- `build_train_dataloader(ddp: bool, **cfg) -> (DataLoader, sampler|None)`
- `build_valid_dataloader(ddp: bool, **cfg) -> (DataLoader, sampler|None)`
- `configure_optimizers(**cfg) -> (optimizer, lr_scheduler|None, scheduler_step: 'iter'|'epoch'|'none')`
- `prepare_batch(batch, ctx) -> batch` (optional; device move / preprocess)
- `training_step(batch, ctx) -> dict(loss=Tensor, metrics=dict)`
- `validation_step(batch, ctx) -> dict(loss=Tensor, metrics=dict)`

See `examples/minimal_task.py`.

## Classification example (ResNet + ImageFolder)

This repo also includes an ImageFolder-style image classification task under `train/classify/`.

Example config: `configs/classify_resnet18.yaml`

Install deps (optional extras):

```bash
pip install -e ".[classify]"
```

Best checkpoint selection is configurable via:

- `runner.best_metric`: e.g. `valid/loss` (default) or `valid/acc1`
- `runner.best_mode`: `min` or `max`

If you set `task.kwargs.model.pretrained=true` and the weights are not cached locally, torchvision may try to download them; use `task.kwargs.model.weights_path=/path/to.pth` to force a local weights file.

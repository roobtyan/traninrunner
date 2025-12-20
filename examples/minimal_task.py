from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class _RandomClassification(Dataset):
    def __init__(self, n: int, in_dim: int, num_classes: int, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, in_dim, generator=g)
        w = torch.randn(in_dim, num_classes, generator=g)
        logits = self.x @ w
        self.y = torch.argmax(logits, dim=-1)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class MinimalTask(nn.Module):
    def __init__(
        self,
        in_dim: int = 16,
        num_classes: int = 4,
        train_size: int = 2048,
        valid_size: int = 512,
        batch_size: int = 64,
        lr: float = 1e-3,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, num_classes))
        self.loss_fn = nn.CrossEntropyLoss()
        self._train_ds = _RandomClassification(train_size, in_dim, num_classes, seed=1)
        self._valid_ds = _RandomClassification(valid_size, in_dim, num_classes, seed=2)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def build_train_dataloader(self, ddp: bool, **cfg) -> Tuple[DataLoader, Optional[Any]]:
        sampler = None
        if ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(self._train_ds, shuffle=True)
        dl = DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self._num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return dl, sampler

    def build_valid_dataloader(self, ddp: bool, **cfg) -> Tuple[DataLoader, Optional[Any]]:
        sampler = None
        if ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(self._valid_ds, shuffle=False)
        dl = DataLoader(
            self._valid_ds,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self._num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return dl, sampler

    def configure_optimizers(self, **cfg):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._lr)
        return optimizer, None, "none"

    def training_step(self, batch, ctx: Dict[str, Any]):
        x, y = batch
        x = x.to(ctx["device"], non_blocking=True)
        y = y.to(ctx["device"], non_blocking=True)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        return {"loss": loss, "metrics": {"acc": acc}}

    def validation_step(self, batch, ctx: Dict[str, Any]):
        x, y = batch
        x = x.to(ctx["device"], non_blocking=True)
        y = y.to(ctx["device"], non_blocking=True)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        return {"loss": loss, "metrics": {"acc": acc}}


import torch

from torch import nn
from torch.utils.data import DataLoader
from typing import Any, Dict, Optional, Tuple
from .transform.cifar_dataloader import build_cifar100_dataloader
from .model.resnet18 import ResNet18
from .losses.cross_entropy import build_cross_entropy_loss
from torchvision import transforms as T


class ClassifyCIFAR10Task(nn.Module):
    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Optional[Dict[str, Any]] = None,
        train: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._data_cfg = dict(data or {})
        self._model_cfg = dict(model or {})
        self._train_cfg = dict(train or {})
        self._optim_cfg = dict(optim or {})
        self._loss_cfg = dict(loss or {})

        self.model = ResNet18(num_classes=100)
        self.loss = build_cross_entropy_loss(
            label_smoothing=float(self._loss_cfg.get("label_smoothing", 0.0))
        )

        self._batch_size = int(self._train_cfg.get("batch_size", 64))
        self._num_workers = int(self._train_cfg.get("num_workers", 8))
        self._lr = float(self._optim_cfg.get("lr", 1e-3))

    def build_train_dataloader(
        self, ddp: bool, **cfg
    ) -> Tuple[DataLoader, Optional[Any]]:
        transform = T.Compose([
            T.ToTensor(),  # PIL → Tensor (C,H,W)
        ])
        return build_cifar100_dataloader(
            is_ddp=ddp,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            train=True,
            transform=transform,
        )

    def build_valid_dataloader(
        self, ddp: bool, **cfg
    ) -> Tuple[DataLoader, Optional[Any]]:
        transform = T.Compose([
            T.ToTensor(),  # PIL → Tensor (C,H,W)
        ])
        return build_cifar100_dataloader(
            is_ddp=ddp,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            train=False,
            transform=transform,
        )

    def configure_optimizers(self, **cfg):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self._lr, momentum=0.9, weight_decay=5e-4
        )
        return optimizer, None, "none"

    def _step(self, batch, ctx: Dict[str, Any]):
        x, y = batch
        x = x.to(ctx["device"], non_blocking=True)
        y = y.to(ctx["device"], non_blocking=True)
        logits = self.model(x)
        loss = self.loss(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        return {"loss": loss, "metrics": {"acc": acc}}

    def training_step(self, batch, ctx: Dict[str, Any]):
        return self._step(batch, ctx)

    def validation_step(self, batch, ctx: Dict[str, Any]):
        return self._step(batch, ctx)

import torch

from torch import nn
from torch.utils.data import DataLoader
from typing import Any, Dict, Optional, Tuple
from .transform.nuscenes_bev_dataset import build_nuscenes_bev_dataloader
from torchvision import transforms as T


class FusionDet3DTask(nn.Module):
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

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 704 * 6, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10),
        )

        self.loss = torch.nn.CrossEntropyLoss()

        self._batch_size = int(self._train_cfg.get("batch_size", 64))
        self._num_workers = int(self._train_cfg.get("num_workers", 8))
        self._lr = float(self._optim_cfg.get("lr", 1e-3))
        
        # dataset
        self._data_root = self._data_cfg.get("data_root", "./data/nuscenes")
        self._data_version = self._data_cfg.get("version", "v1.0-trainval")
        self._queue_length = int(self._data_cfg.get("queue_length", 4))
        self._img_size = tuple(self._data_cfg.get("img_size", (256, 704)))
        self._num_cams = int(self._data_cfg.get("num_cams", 6))
        self._use_lidar = bool(self._data_cfg.get("use_lidar", True))
        self._class_names = self._data_cfg.get("class_names", None)

    def build_train_dataloader(
        self, ddp: bool, **cfg
    ) -> Tuple[DataLoader, Optional[Any]]:
        return build_nuscenes_bev_dataloader(
            data_root=self._data_root,
            version=self._data_version,
            split='train',
            is_ddp=ddp,
            queue_length=self._queue_length,
            img_size=self._img_size,
            num_cams=self._num_cams,
            use_lidar=self._use_lidar,
            class_names=self._class_names,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )

    def build_valid_dataloader(
        self, ddp: bool, **cfg
    ) -> Tuple[DataLoader, Optional[Any]]:
        return build_nuscenes_bev_dataloader(
            data_root=self._data_root,
            version=self._data_version,
            split='val',
            is_ddp=ddp,
            queue_length=self._queue_length,
            img_size=self._img_size,
            num_cams=self._num_cams,
            use_lidar=self._use_lidar,
            class_names=self._class_names,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
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

from __future__ import annotations

from torch import nn


def build_cross_entropy_loss(*, label_smoothing: float = 0.0) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))


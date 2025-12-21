from typing import Union

import torch
import torch.nn as nn


def _build_activation(act_layer: Union[nn.Module, type, None], inplace: bool = True):
    if act_layer is None:
        return None
    if isinstance(act_layer, nn.Module):
        return act_layer
    if hasattr(act_layer, 'inplace'):
        return act_layer(inplace=inplace)
    return act_layer()


class ConvBnAct(nn.Sequential):
    """Simple conv-bn-activation block."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            padding: int = 0,
            bias: bool = False,
            act_layer: Union[type, nn.Module, None] = nn.SiLU,
    ):
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        act = _build_activation(act_layer)
        if act is not None:
            layers.append(act)
        super().__init__(*layers)


class DepthwiseBlock(nn.Module):
    """
    Depthwise separable residual block (Inverted Residual).
    Follows MobileNetV2 style: Expand -> Depthwise -> Linear Project.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            expansion: float = 2.0,
            act_layer: Union[type, nn.Module] = nn.SiLU,
            use_shortcut_on_mismatch: bool = False  # 是否在 stride!=1 时强制使用 shortcut
    ):
        super().__init__()
        assert stride in (1, 2), "LightBlock only supports stride 1 or 2."
        hidden_dim = int(round(in_channels * expansion))

        # 只有 stride=1 且输入输出通道一致时，才默认使用残差连接
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        # 1. PW Expand (if needed)
        if expansion != 1:
            layers.append(
                ConvBnAct(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, act_layer=act_layer)
            )

        # 2. DW Conv
        layers.append(
            ConvBnAct(
                hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                groups=hidden_dim, act_layer=act_layer
            )
        )

        # 3. PW Linear Project (Note: No Activation here!)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

        # Shortcut 处理
        self.shortcut = None
        if self.use_res_connect:
            self.shortcut = nn.Identity()  # 直接相加
        elif use_shortcut_on_mismatch:
            self.shortcut = ConvBnAct(
                in_channels, out_channels, kernel_size=1, stride=stride,
                padding=0, act_layer=None  # Shortcut 路径通常也不加激活
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            bias: bool = False,
            use_bn: bool = True,
            act_layer: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
            stride=1,
            padding=0,
        )

        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.act_layer = _build_activation(act_layer, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.act_layer is not None:
            x = self.act_layer(x)
        return x
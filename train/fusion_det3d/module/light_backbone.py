from collections.abc import Sequence as SequenceCollection
from typing import Union, Sequence

import torch
import torch.nn as nn

from ...common.module.conv import ConvBnAct, DepthwiseBlock


class LightBackbone(nn.Module):
    def __init__(
        self,
        in_channels,
        stem_channels,
        stage_channels: Sequence[int] = (64, 96, 128, 256),
        num_blocks: Sequence[int] = (1, 2, 3, 2),
        stage_strides: Sequence[int] = (2, 2, 2, 2),
        expansion: float = 2.0,
        out_strides: Sequence[int] = (8, 16, 32),
        out_aligned_feature: bool = True,
        stride_out: int = 8,
        cat_dim: int = 0,
        norm_eval: bool = False,
        act_layer: Union[type, nn.Module] = nn.SiLU,
    ):
        super().__init__()
        assert len(stage_channels) == len(stage_strides) == len(num_blocks)

        self.out_strides = tuple(sorted(set(out_strides)))
        self.stage_channels = stage_channels
        self.stage_strides = stage_strides
        self.num_blocks = num_blocks
        self.expansion = expansion
        self.out_aligned_feature = out_aligned_feature
        self.stride_out = stride_out
        self.cat_dim = cat_dim
        self.norm_eval = norm_eval
        self.act_layer = act_layer

        self.stem = nn.Sequential(
            ConvBnAct(
                in_channels, stem_channels, 3, stride=2, padding=1, act_layer=act_layer
            ),
            ConvBnAct(
                stem_channels,
                stem_channels,
                3,
                stride=1,
                padding=1,
                act_layer=act_layer,
            ),
        )

        self.stages = nn.ModuleList()
        stage_in = stem_channels
        for stage_channel, stride, block in zip(
            stage_channels, stage_strides, num_blocks
        ):
            stage = self._make_stage(stage_in, stage_channel, block, stride)
            self.stages.append(stage)
            stage_in = stage_channel

        self._init_weights()

    def _make_stage(self, in_channels, out_channels, block, stride):
        blocks = []
        for idx in range(block):
            block_stride = stride if idx == 0 else 1
            blocks.append(
                DepthwiseBlock(
                    in_channels,
                    out_channels,
                    stride=block_stride,
                    expansion=self.expansion,
                    act_layer=self.act_layer,
                )
            )
            in_channels = out_channels
        return nn.Sequential(*blocks)

    def process_sequence_input(self, x: SequenceCollection) -> torch.Tensor:
        if len(x) == 1:
            return x[0]
        return torch.cat(x, dim=self.cat_dim)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if isinstance(x, SequenceCollection):
            x = self.process_sequence_input(x)
        x = self.stem(x)
        stage_outputs = {}
        cumulative_stride = 2
        for stage, stride in zip(self.stages, self.stage_strides):
            x = stage(x)
            cumulative_stride *= stride
            stage_outputs[cumulative_stride] = x

        feats = []
        for s in self.out_strides:
            if s in stage_outputs:
                feats.append(stage_outputs[s])
        return tuple(feats)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

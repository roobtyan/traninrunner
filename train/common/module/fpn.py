from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import ConvBnAct, DepthwiseSeparableConv

class FPN(nn.Module):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 out_index: Union[int, List[int]] = None,
                 start_level: int = 0,
                 sep_conv: bool = True,
                 add_extra_convs: bool = False,
                 extra_convs_on_inputs: bool = False,
                 relu_before_pooling: bool = False,
                 use_norm: bool = False,
                 kernel_size=3
                 ):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.out_index = out_index
        self.start_level = start_level
        self.sep_conv = sep_conv
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.relu_before_pooling = relu_before_pooling
        self.use_norm = use_norm
        self.kernel_size = kernel_size
        self.num_ins = len(in_channels)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # ---------------------------------------
        # 1. lateral convs (1x1)
        # ---------------------------------------
        for i in range(self.start_level, self.num_ins):
            if use_norm:
                l_conv = ConvBnAct(
                    in_channels[i], out_channels, kernel_size=1, stride=1, padding=0, act_layer=nn.ReLU(inplace=True),
                )
            else:
                l_conv = nn.Conv2d(
                    in_channels[i], out_channels, kernel_size=1, stride=1, padding=0,
                )
            self.lateral_convs.append(l_conv)

            if self.sep_conv:
                fpn_conv = DepthwiseSeparableConv(
                    out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False, use_bn=use_norm,
                    act_layer=nn.ReLU(inplace=True)
                )
            else:
                fpn_conv = nn.Conv2d(
                    out_channels, out_channels, kernel_size, padding=kernel_size // 2,
                )
            self.fpn_convs.append(fpn_conv)

        # ---------------------------------------
        # 2. extra convs (P6, P7)
        # ---------------------------------------
        extra_levels = num_outs - self.num_ins
        if add_extra_convs and extra_levels > 0:
            for i in range(extra_levels):
                if i == 0 and extra_convs_on_inputs:
                    in_ch = in_channels[-1]
                else:
                    in_ch = out_channels

                extra_conv = nn.Conv2d(
                    in_ch, out_channels, kernel_size, padding=kernel_size // 2, stride=2,
                )
                self.fpn_convs.append(extra_conv)
        else:
            self.extra_pool = nn.ModuleList()
            for _ in range(extra_levels):
                self.extra_pool.append(nn.MaxPool2d(kernel_size=1, stride=2))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        # 1. lateral
        laterals = [
            l_conv(inputs[i + self.start_level])
            for i, l_conv in enumerate(self.lateral_convs)
        ]

        # 2. top-down
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=(h, w), mode='nearest'
            )

        # 3. 3x3 conv to clean up
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        # 4. extra conv levels
        if self.num_outs > len(outs):
            if self.add_extra_convs:
                last_out = (inputs[-1] if self.extra_convs_on_inputs else outs[-1])
                outs.append(self.fpn_convs[len(laterals)](last_out))
                for i in range(len(laterals) + 1, self.num_outs):
                    outs.append(self.fpn_convs[i](outs[-1]))
            else:
                for i in range(self.num_outs - len(outs)):
                    if self.relu_before_pooling:
                        outs.append(self.extra_pool[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.extra_pool[i](outs[-1]))

        # 5. outputs
        if self.out_index is None:
            return tuple(outs)
        elif isinstance(self.out_index, list):
            return tuple(outs[i] for i in self.out_index)
        else:
            return outs[self.out_index]


class ViewSelector(nn.Module):
    def __init__(self, index=[0], total_views=6):
        super(ViewSelector, self).__init__()
        self.index = index
        self.total_views = total_views

    def forward(self, inputs, *args):
        outputs = []

        for feat in inputs:
            bv, c, h, w = feat.shape
            b = bv // self.total_views
            # reshape
            feat = feat.reshape(b, self.total_views, c, h, w)
            # slice
            feat = feat[:, self.index]
            # reshape bv
            feat = feat.reshape(-1, c, h, w)
            outputs.append(feat)

        ext = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                bv, *rest = arg.shape
                b = bv // self.total_views
                arg = arg.reshape(b, self.total_views, *rest)
                arg = arg[:, self.index]
                ext.append(arg.reshape(-1, *rest))
            else:
                raise TypeError
        if len(ext) > 0:
            return outputs, *ext
        return outputs

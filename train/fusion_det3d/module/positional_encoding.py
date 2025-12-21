"""Positional encoding utilities."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class SinePositionalEncoding3D(nn.Module):
    """3D sine-cosine positional encoding with projection."""

    def __init__(
        self,
        embed_dims: int,
        num_feats: int = 64,
        temperature: float = 10000.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        # num_feats: 内部生成正弦波的基础频率数，设为 64 或 128
        # embed_dims: 最终输出的维度，必须等于 BEVFormer 的 embed_dims (如 256)
        self.num_feats = num_feats
        self.embed_dims = embed_dims
        self.temperature = float(temperature)
        self.normalize = normalize

        # 添加一个投影层
        # 输入维度 = num_feats * 2(sin/cos) * 3(x,y,z轴) = num_feats * 6
        self.proj = nn.Linear(self.num_feats * 6, embed_dims)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Generate encoding for a (B, D, H, W) mask."""

        if mask.dim() != 4:
            raise ValueError("mask must have shape (B, D, H, W)")
        B, D, H, W = mask.shape
        device = mask.device

        # 防止 mask 全 0 导致 nan，虽然这里是用 range 生成的，但也加个保护
        eps = 1e-6

        z_range = (
            torch.linspace(0, 1, steps=D, device=device)
            .view(1, D, 1, 1)
            .expand(B, D, H, W)
        )
        y_range = (
            torch.linspace(0, 1, steps=H, device=device)
            .view(1, 1, H, 1)
            .expand(B, D, H, W)
        )
        x_range = (
            torch.linspace(0, 1, steps=W, device=device)
            .view(1, 1, 1, W)
            .expand(B, D, H, W)
        )

        scales = torch.arange(self.num_feats, dtype=torch.float32, device=device)
        scales = self.temperature ** (
            2 * torch.div(scales, 2, rounding_mode="floor") / self.num_feats
        )

        def encode(range_tensor: torch.Tensor) -> torch.Tensor:
            # 增加 eps 防止除零（虽然这里 scales 不会为 0）
            dim_t = range_tensor[..., None] / (scales + eps)
            sin = dim_t.sin()
            cos = dim_t.cos()
            out = torch.stack((sin, cos), dim=-1)
            return out.flatten(-2)  # [B, D, H, W, num_feats * 2]

        x_embed = encode(x_range)
        y_embed = encode(y_range)
        z_embed = encode(z_range)

        # 拼接三个轴: [B, D, H, W, num_feats * 6]
        pos = torch.cat([x_embed, y_embed, z_embed], dim=-1)

        # 核心修改：投影到目标维度 (256)
        pos = self.proj(pos)

        # 调整维度顺序: [B, D, H, W, C] -> [B, C, D, H, W]
        pos = pos.permute(0, 4, 1, 2, 3).contiguous()
        return pos

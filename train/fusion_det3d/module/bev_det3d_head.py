import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

class BEVDet3DHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        bev_h: int,
        bev_w: int,
        pc_range: List[float],
        max_objs: int = 500,
        min_overlap: float = 0.1,
        min_radius: int = 2,
        loss_weights: Dict[str, float] | None = None,
    ):
        super().__init__()
        self.bev_h = int(bev_h)
        self.bev_w = int(bev_w)
        self.pc_range = pc_range
        self.max_objs = int(max_objs)
        self.min_overlap = float(min_overlap)
        self.min_radius = int(min_radius)
        self.num_classes = int(num_classes)
        self.reg_dim = 10

        self.loss_weights = {
            "hm": 1.0,
            "offset": 1.0,
            "height": 1.0,
            "dim": 1.0,
            "rot": 1.0,
            "vel": 1.0,
        }
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)

        # CenterHead style
        self.heatmap_head = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(in_channels, self.reg_dim, kernel_size=3, padding=1)
        self._init_weights()

    def forward(self, x):
        # x: (B, C, H, W)
        hm = self.heatmap_head(x)  # (B, num_classes, H, W)
        reg = self.reg_head(x)  # (B, 10, H, W)
        return {'hm': hm, 'reg': reg}

    def loss(self, preds, gt_boxes, gt_labels):
        hm_pred = preds['hm'].sigmoid().clamp(1e-4, 1.0 - 1e-4)  # (B, num_classes, H, W)
        reg_pred = preds['reg']  # (B, 10, H, W)

        hm_target, reg_target, ind, reg_mask = self._build_targets(
            gt_boxes, gt_labels, device=hm_pred.device
        )
        # hm_target: (B, num_classes, H, W)
        # reg_target: (B, max_objs, 10), ind: (B, max_objs), reg_mask: (B, max_objs)

        loss_hm = self._gaussian_focal_loss(hm_pred, hm_target) * self.loss_weights["hm"]
        reg_pred = self._gather_feat(reg_pred, ind)  # (B, max_objs, 10)
        reg_mask = reg_mask.unsqueeze(-1).float()  # (B, max_objs, 1)

        loss_offset = self._l1_loss(
            reg_pred[..., 0:2], reg_target[..., 0:2], reg_mask
        ) * self.loss_weights["offset"]
        loss_height = self._l1_loss(
            reg_pred[..., 2:3], reg_target[..., 2:3], reg_mask
        ) * self.loss_weights["height"]
        loss_dim = self._l1_loss(
            reg_pred[..., 3:6], reg_target[..., 3:6], reg_mask
        ) * self.loss_weights["dim"]
        loss_rot = self._l1_loss(
            reg_pred[..., 6:8], reg_target[..., 6:8], reg_mask
        ) * self.loss_weights["rot"]
        loss_vel = self._l1_loss(
            reg_pred[..., 8:10], reg_target[..., 8:10], reg_mask
        ) * self.loss_weights["vel"]

        return {
            "loss_heatmap": loss_hm,
            "loss_offset": loss_offset,
            "loss_height": loss_height,
            "loss_dim": loss_dim,
            "loss_rot": loss_rot,
            "loss_vel": loss_vel,
        }

    def _init_weights(self):
        nn.init.constant_(self.heatmap_head.bias, -2.19)

    def _build_targets(
        self,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(gt_boxes)
        heatmap = torch.zeros(
            (batch_size, self.num_classes, self.bev_h, self.bev_w), device=device
        )
        reg_target = torch.zeros(
            (batch_size, self.max_objs, self.reg_dim), device=device
        )
        ind = torch.zeros((batch_size, self.max_objs), dtype=torch.long, device=device)
        reg_mask = torch.zeros((batch_size, self.max_objs), dtype=torch.uint8, device=device)

        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        x_range = x_max - x_min
        y_range = y_max - y_min

        for b in range(batch_size):
            if gt_boxes[b].numel() == 0:
                continue
            boxes = gt_boxes[b].cpu()
            labels = gt_labels[b].cpu()
            num_objs = min(boxes.shape[0], self.max_objs)
            obj_idx = 0
            for i in range(num_objs):
                x, y, z, w, l, h, yaw, vx, vy = boxes[i].tolist()
                cls = int(labels[i])
                if cls < 0 or cls >= self.num_classes:
                    continue

                if x < x_min or x >= x_max or y < y_min or y >= y_max:
                    continue

                cx = (x - x_min) / x_range * self.bev_w
                cy = (y - y_min) / y_range * self.bev_h

                if cx < 0 or cx >= self.bev_w or cy < 0 or cy >= self.bev_h:
                    continue
                
                # 取整作为中心点
                center_x = int(cx)
                center_y = int(cy)

                if obj_idx >= self.max_objs:
                    break

                length = l / x_range * self.bev_w
                width = w / y_range * self.bev_h
                radius = self._gaussian_radius((width, length), self.min_overlap)
                radius = max(self.min_radius, int(radius))

                self._draw_gaussian(heatmap[b, cls], (center_x, center_y), radius)  # (H, W)

                ind[b, obj_idx] = center_y * self.bev_w + center_x
                reg_mask[b, obj_idx] = 1
                reg_target[b, obj_idx, 0] = cx - center_x
                reg_target[b, obj_idx, 1] = cy - center_y
                reg_target[b, obj_idx, 2] = z
                reg_target[b, obj_idx, 3] = w
                reg_target[b, obj_idx, 4] = l
                reg_target[b, obj_idx, 5] = h
                reg_target[b, obj_idx, 6] = math.sin(yaw)
                reg_target[b, obj_idx, 7] = math.cos(yaw)
                reg_target[b, obj_idx, 8] = vx
                reg_target[b, obj_idx, 9] = vy
                obj_idx += 1

        return heatmap, reg_target, ind, reg_mask

    def _gaussian_focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        neg_weights = (1 - target).pow(4.0)

        pos_loss = -torch.log(pred) * (1 - pred).pow(2.0) * pos_mask
        neg_loss = -torch.log(1 - pred) * pred.pow(2.0) * neg_weights * neg_mask

        num_pos = pos_mask.sum()
        if num_pos > 0:
            return (pos_loss.sum() + neg_loss.sum()) / num_pos
        return neg_loss.sum()

    def _l1_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = torch.abs(pred - target) * mask
        denom = mask.sum().clamp(min=1.0)
        return loss.sum() / denom

    def _gather_feat(self, feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, H*W, C)
        ind = ind.unsqueeze(-1).expand(-1, -1, c)  # (B, max_objs, C)
        return feat.gather(1, ind)

    def _gaussian2d(self, shape: Tuple[int, int], sigma: float = 1.0) -> torch.Tensor:
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = torch.meshgrid(
            torch.linspace(-m, m, int(shape[0])),
            torch.linspace(-n, n, int(shape[1])),
            indexing="ij",
        )
        h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        return h

    def _draw_gaussian(self, heatmap: torch.Tensor, center: Tuple[int, int], radius: int) -> None:
        diameter = 2 * radius + 1
        gaussian = self._gaussian2d((diameter, diameter), sigma=diameter / 6.0).to(
            heatmap.device
        )

        x, y = center
        height, width = heatmap.shape[0], heatmap.shape[1]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian[
            radius - top : radius + bottom, radius - left : radius + right
        ]
        torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    def _gaussian_radius(self, det_size: Tuple[float, float], min_overlap: float) -> float:
        height, width = det_size

        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = math.sqrt(max(0.0, b1 * b1 - 4 * a1 * c1))
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = math.sqrt(max(0.0, b2 * b2 - 4 * a2 * c2))
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = math.sqrt(max(0.0, b3 * b3 - 4 * a3 * c3))
        r3 = (b3 + sq3) / 2

        return min(r1, r2, r3)

from __future__ import annotations

from typing import Dict, List

import torch


def _topk(scores: torch.Tensor, k: int):
    b, c, h, w = scores.shape
    k = min(int(k), h * w)
    if k <= 0:
        empty = scores.new_zeros((b, 0))
        empty_long = scores.new_zeros((b, 0), dtype=torch.long)
        return empty, empty_long, empty_long, empty_long, empty_long
    topk_scores, topk_inds = torch.topk(scores.view(b, c, -1), k)
    topk_inds = topk_inds % (h * w)
    topk_ys = (topk_inds // w).int()
    topk_xs = (topk_inds % w).int()

    topk_scores, topk_inds2 = torch.topk(topk_scores.view(b, -1), k)
    topk_clses = (topk_inds2 // k).int()
    topk_inds = topk_inds.view(b, -1).gather(1, topk_inds2)
    topk_ys = topk_ys.view(b, -1).gather(1, topk_inds2)
    topk_xs = topk_xs.view(b, -1).gather(1, topk_inds2)

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def _gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    feat = feat.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
    ind = ind.unsqueeze(-1).expand(-1, -1, c)
    return feat.gather(1, ind)


def decode_center_head(
    hm: torch.Tensor,
    reg: torch.Tensor,
    pc_range: List[float],
    score_thresh: float = 0.1,
    max_per_img: int = 100,
) -> List[Dict[str, torch.Tensor]]:
    hm = hm.sigmoid()
    b, c, h, w = hm.shape
    k = min(int(max_per_img), h * w)
    scores, inds, clses, ys, xs = _topk(hm, k)
    reg_feat = _gather_feat(reg, inds)

    x_min, y_min, _, x_max, y_max, _ = pc_range
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_scale = x_range / float(w)
    y_scale = y_range / float(h)

    results: List[Dict[str, torch.Tensor]] = []
    for i in range(b):
        score_i = scores[i]
        mask = score_i > float(score_thresh)
        if mask.sum() == 0:
            results.append(
                {
                    "boxes_3d": reg_feat.new_zeros((0, 9)),
                    "scores": score_i.new_zeros((0,)),
                    "labels": clses[i][mask],
                }
            )
            continue
        score_i = score_i[mask]
        cls_i = clses[i][mask]
        xs_i = xs[i][mask].float()
        ys_i = ys[i][mask].float()
        reg_i = reg_feat[i][mask]

        x = (xs_i + reg_i[:, 0]) * x_scale + x_min
        y = (ys_i + reg_i[:, 1]) * y_scale + y_min
        z = reg_i[:, 2]
        w_box = reg_i[:, 3]
        l_box = reg_i[:, 4]
        h_box = reg_i[:, 5]
        yaw = torch.atan2(reg_i[:, 6], reg_i[:, 7])
        vx = reg_i[:, 8]
        vy = reg_i[:, 9]

        boxes = torch.stack([x, y, z, w_box, l_box, h_box, yaw, vx, vy], dim=1)
        results.append({"boxes_3d": boxes, "scores": score_i, "labels": cls_i})
    return results

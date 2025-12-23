from __future__ import annotations

import os
from typing import Dict, List

import torch
from PIL import Image, ImageDraw


_CAM_ORDER = [2, 0, 1, 4, 3, 5]


def _box3d_corners(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 8, 3))
    x, y, z, w, l, h, yaw = (
        boxes[:, 0],
        boxes[:, 1],
        boxes[:, 2],
        boxes[:, 3],
        boxes[:, 4],
        boxes[:, 5],
        boxes[:, 6],
    )
    x_corners = torch.stack(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=1
    )
    y_corners = torch.stack(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=1
    )
    z_corners = torch.stack(
        [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], dim=1
    )

    cos_yaw = torch.cos(yaw).unsqueeze(1)
    sin_yaw = torch.sin(yaw).unsqueeze(1)
    x_rot = x_corners * cos_yaw - y_corners * sin_yaw
    y_rot = x_corners * sin_yaw + y_corners * cos_yaw
    z_rot = z_corners

    x_rot = x_rot + x.unsqueeze(1)
    y_rot = y_rot + y.unsqueeze(1)
    z_rot = z_rot + z.unsqueeze(1)
    return torch.stack([x_rot, y_rot, z_rot], dim=-1)


def _project_points(corners: torch.Tensor, lidar2img: torch.Tensor):
    num = corners.shape[0]
    ones = torch.ones((num, 8, 1), dtype=corners.dtype, device=corners.device)
    corners_hom = torch.cat([corners, ones], dim=-1)
    proj = torch.matmul(corners_hom, lidar2img.t())
    depth = proj[..., 2].clamp(min=1e-6)
    u = proj[..., 0] / depth
    v = proj[..., 1] / depth
    return u, v, depth


def _draw_boxes(
    img: Image.Image,
    boxes: torch.Tensor,
    lidar2img: torch.Tensor,
    color: tuple,
):
    if boxes.numel() == 0:
        return
    corners = _box3d_corners(boxes)
    u, v, depth = _project_points(corners, lidar2img)
    u = u.cpu().numpy()
    v = v.cpu().numpy()
    depth = depth.cpu().numpy()

    draw = ImageDraw.Draw(img)
    w_img, h_img = img.size
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i in range(u.shape[0]):
        for a, b in edges:
            if depth[i, a] <= 0 or depth[i, b] <= 0:
                continue
            x1, y1 = float(u[i, a]), float(v[i, a])
            x2, y2 = float(u[i, b]), float(v[i, b])
            if (
                (x1 < 0 and x2 < 0)
                or (x1 >= w_img and x2 >= w_img)
                or (y1 < 0 and y2 < 0)
                or (y1 >= h_img and y2 >= h_img)
            ):
                continue
            draw.line((x1, y1, x2, y2), fill=color, width=2)


def _build_heatmap_panel(
    hm: torch.Tensor, topk: int, out_size: tuple[int, int]
) -> Image.Image:
    hm = hm.sigmoid()
    c, h, w = hm.shape
    k = min(int(topk), c * h * w)
    img = Image.new("RGB", (w, h), (0, 0, 0))
    if k <= 0:
        return img.resize(out_size)
    scores, inds = torch.topk(hm.view(-1), k)
    colors = [
        (255, 64, 64),
        (64, 255, 64),
        (64, 128, 255),
        (255, 192, 64),
        (192, 64, 255),
        (64, 255, 192),
    ]
    draw = ImageDraw.Draw(img)
    for idx in inds.tolist():
        cls = idx // (h * w)
        rem = idx % (h * w)
        y = rem // w
        x = rem % w
        color = colors[int(cls) % len(colors)]
        r = 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    return img.resize(out_size)


def visualize_sample(
    meta: Dict,
    pred: Dict[str, torch.Tensor],
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    class_names: List[str],
    save_path: str,
    img_size: tuple[int, int],
    topk: int = 50,
) -> None:
    img_h, img_w = img_size
    img_paths = meta.get("img_paths", [])
    intrinsics = meta["intrinsics"]
    cam2egos = meta["cam2egos"]

    pred_boxes = pred["boxes_3d"]
    pred_scores = pred["scores"]
    if pred_scores.numel() > topk:
        topk_scores, topk_idx = torch.topk(pred_scores, topk)
        pred_boxes = pred_boxes[topk_idx]
        pred_scores = topk_scores

    cam_imgs = []
    for cam_idx in _CAM_ORDER:
        if cam_idx >= len(img_paths):
            continue
        img_path = img_paths[cam_idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((img_w, img_h))
        K = intrinsics[cam_idx].float()
        cam2ego = cam2egos[cam_idx].float()
        lidar2img = torch.matmul(K, torch.inverse(cam2ego))

        _draw_boxes(img, pred_boxes, lidar2img, color=(255, 64, 64))
        _draw_boxes(img, gt_boxes, lidar2img, color=(64, 255, 64))
        cam_imgs.append(img)

    while len(cam_imgs) < 6:
        cam_imgs.append(Image.new("RGB", (img_w, img_h), (0, 0, 0)))

    grid = Image.new("RGB", (img_w * 3, img_h * 2), (0, 0, 0))
    for i, img in enumerate(cam_imgs[:6]):
        r = i // 3
        c = i % 3
        grid.paste(img, (c * img_w, r * img_h))

    heatmap = _build_heatmap_panel(pred["hm"], topk=topk, out_size=(img_w * 3, img_h))
    canvas = Image.new("RGB", (img_w * 3, img_h * 3), (0, 0, 0))
    canvas.paste(grid, (0, 0))
    canvas.paste(heatmap, (0, img_h * 2))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)

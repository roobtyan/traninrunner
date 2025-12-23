from __future__ import annotations

from typing import Dict, List

import torch
from pyquaternion import Quaternion


def _get_attr_name(class_name: str, vx: float, vy: float, speed_thresh: float) -> str:
    speed = (vx * vx + vy * vy) ** 0.5
    if class_name in {
        "car",
        "truck",
        "bus",
        "trailer",
        "construction_vehicle",
    }:
        return "vehicle.moving" if speed > speed_thresh else "vehicle.parked"
    if class_name in {"pedestrian"}:
        return "pedestrian.moving" if speed > speed_thresh else "pedestrian.standing"
    if class_name in {"motorcycle", "bicycle"}:
        return "cycle.with_rider" if speed > speed_thresh else "cycle.without_rider"
    return ""


def format_nuscenes_results(
    samples: List[Dict[str, torch.Tensor]],
    class_names: List[str],
    score_thresh: float,
    speed_thresh: float = 0.2,
    use_lidar: bool = False,
) -> Dict:
    results: Dict[str, List[Dict]] = {}
    for sample in samples:
        sample_token = sample["sample_token"]
        boxes = sample["boxes_3d"]
        scores = sample["scores"]
        labels = sample["labels"]
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        dets: List[Dict] = []
        for i in range(len(scores)):
            score = float(scores[i])
            if score < float(score_thresh):
                continue
            label = int(labels[i])
            if label < 0 or label >= len(class_names):
                continue
            x, y, z, w, l, h, yaw, vx, vy = [float(v) for v in boxes[i]]
            quat = Quaternion(axis=[0, 0, 1], angle=yaw)
            dets.append(
                {
                    "sample_token": sample_token,
                    "translation": [x, y, z],
                    "size": [w, l, h],
                    "rotation": [quat.w, quat.x, quat.y, quat.z],
                    "velocity": [vx, vy],
                    "detection_name": class_names[label],
                    "detection_score": score,
                    "attribute_name": _get_attr_name(class_names[label], vx, vy, speed_thresh),
                }
            )
        results[sample_token] = dets

    meta = {
        "use_camera": True,
        "use_lidar": bool(use_lidar),
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }
    return {"results": results, "meta": meta}

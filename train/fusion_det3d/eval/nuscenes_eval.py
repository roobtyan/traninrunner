from __future__ import annotations

import json
import os
from typing import Dict, Tuple

from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.nuscenes import NuScenes


def evaluate_nuscenes(
    data_root: str,
    version: str,
    results: Dict,
    output_dir: str,
    eval_cfg: str = "detection_cvpr_2019",
    eval_set: str | None = None,
    verbose: bool = False,
) -> Tuple[Dict, str]:
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "results_nuscenes.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    if eval_set is None:
        eval_set = "mini_val" if "mini" in version else "val"

    nusc = NuScenes(version=version, dataroot=data_root, verbose=verbose)
    cfg = config_factory(eval_cfg)
    evaluator = DetectionEval(
        nusc=nusc,
        config=cfg,
        result_path=result_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=verbose,
    )
    metrics_summary = evaluator.main()
    return metrics_summary, result_path

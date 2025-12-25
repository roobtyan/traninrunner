import os
import torch
import torch.distributed as dist

from torch import nn
from torch.utils.data import DataLoader
from typing import Any, Dict, Optional, Tuple
from .transform.nuscenes_bev_dataset import build_nuscenes_bev_dataloader
from torchvision import transforms as T
from .module.resnet_backbone import ResNetBackbone
from ..common.module.fpn import FPN, ViewSelector
from .module.bevformer_encoder import BEVFormerEncoder
from .module.lidar_encoder import LidarEncoder
from .module.bev_det3d_head import BEVDet3DHead
from .eval import decode_center_head, format_nuscenes_results, evaluate_nuscenes, visualize_sample


class FusionDet3DTask(nn.Module):
    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Optional[Dict[str, Any]] = None,
        train: Optional[Dict[str, Any]] = None,
        val: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._data_cfg = dict(data or {})
        self._model_cfg = dict(model or {})
        self._train_cfg = dict(train or {})
        self._val_cfg = dict(val or {})
        self._optim_cfg = dict(optim or {})
        self._loss_cfg = dict(loss or {})

        self._batch_size = int(self._train_cfg.get("batch_size", 64))
        self._num_workers = int(self._train_cfg.get("num_workers", 8))
        self._lr = float(self._optim_cfg.get("lr", 1e-3))

        # dataset
        self._data_root = self._data_cfg.get("data_root", "./data/nuscenes")
        self._data_version = self._data_cfg.get("version", "v1.0-trainval")
        self._queue_length = int(self._data_cfg.get("queue_length", 4))
        self._img_size = tuple(self._data_cfg.get("img_size", (256, 704)))
        self._num_cams = int(self._data_cfg.get("num_cams", 6))
        self._use_lidar = bool(self._data_cfg.get("use_lidar", True))
        self._class_names = self._data_cfg.get("class_names", None)

        # model
        self._bev_shape = tuple(self._model_cfg.get("bev_shape", (75, 60)))  # H, W
        self._point_cloud_range = self._model_cfg.get(
            "pc_range", [-60.0, -60.0, -4.0, 140.0, 60.0, 4.0]
        )
        self._voxel_size = self._model_cfg.get("voxel_size", [1.0, 0.6, 2.0])
        self._n_voxels = self._model_cfg.get("n_voxels", [75, 60, 1])
        self._embed_dims = self._model_cfg.get("embed_dims", 256)
        self._num_heads = int(self._model_cfg.get("num_heads", 8))
        self._num_levels = int(self._model_cfg.get("fpn_num_outs", 3))
        self._num_points = int(self._model_cfg.get("num_points", 8))
        self._num_layers = int(self._model_cfg.get("num_layers", 1))
        self._num_z = int(self._model_cfg.get("num_z", 8))

        self._val_score_thresh = float(self._val_cfg.get("score_thresh", 0.1))
        self._val_max_per_img = int(self._val_cfg.get("max_per_img", 100))
        self._val_vis_enable = bool(self._val_cfg.get("vis_enable", False))
        self._val_vis_topk = int(self._val_cfg.get("vis_topk", 50))
        self._val_vis_dir = str(self._val_cfg.get("vis_dir", "vis"))
        self._val_speed_thresh = float(self._val_cfg.get("speed_thresh", 0.2))
        self._val_eval_cfg = str(self._val_cfg.get("eval_cfg", "detection_cvpr_2019"))
        self._val_eval_set = self._val_cfg.get("eval_set", None)
        self._val_results = []
        self._val_epoch = None

        self._backbone = ResNetBackbone(
            self._model_cfg.get("backbone_depth", 18),
            pretrained=self._model_cfg.get("backbone_pretrained", True),
            out_strides=self._model_cfg.get("backbone_out_strides", [8, 16, 32]),
        )

        self._neck = FPN(
            in_channels=self._model_cfg.get("fpn_in_channels", [128, 256, 512]),
            out_channels=self._model_cfg.get("fpn_out_channels", 256),
            num_outs=self._model_cfg.get("fpn_num_outs", 3),
            use_norm=self._model_cfg.get("fpn_use_norm", False),
        )

        self._img_bev_encoder = BEVFormerEncoder(
            bev_h=self._bev_shape[0],
            bev_w=self._bev_shape[1],
            embed_dims=self._embed_dims,
            num_heads=self._num_heads,
            num_levels=self._num_levels,
            num_points=self._num_points,
            num_layers=self._num_layers,
            num_z=self._num_z,
            pc_range=self._point_cloud_range,
            img_size=self._img_size,
            fpn_out_channels=self._model_cfg.get("fpn_out_channels", 256),
        )

        if self._use_lidar:
            self._lidar_encoder = (
                LidarEncoder(
                    out_channels=self._embed_dims,
                    pc_range=self._point_cloud_range,
                    voxel_size=self._voxel_size,
                    bev_shape=self._bev_shape,
                )
                if self._use_lidar
                else None
            )

            self.fusion_layer = nn.Sequential(
                nn.Conv2d(
                    self._embed_dims * 2,
                    self._embed_dims,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self._embed_dims),
                nn.ReLU(inplace=True),
            )

        self._head = BEVDet3DHead(
            in_channels=self._embed_dims,
            num_classes=len(self._class_names) if self._class_names is not None else 10,
            bev_h=self._bev_shape[0],
            bev_w=self._bev_shape[1],
            pc_range=self._point_cloud_range,
            max_objs=int(self._model_cfg.get("max_objs", 500)),
            min_overlap=float(self._model_cfg.get("min_overlap", 0.1)),
            min_radius=int(self._model_cfg.get("min_radius", 2)),
            loss_weights=self._model_cfg.get("loss_weights", None),
        )

    def build_train_dataloader(
        self, ddp: bool, **cfg
    ) -> Tuple[DataLoader, Optional[Any]]:
        return build_nuscenes_bev_dataloader(
            data_root=self._data_root,
            version=self._data_version,
            split="train",
            is_ddp=ddp,
            queue_length=self._queue_length,
            img_size=self._img_size,
            num_cams=self._num_cams,
            use_lidar=self._use_lidar,
            class_names=self._class_names,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )

    def build_valid_dataloader(
        self, ddp: bool, **cfg
    ) -> Tuple[DataLoader, Optional[Any]]:
        return build_nuscenes_bev_dataloader(
            data_root=self._data_root,
            version=self._data_version,
            split="val",
            is_ddp=ddp,
            queue_length=self._queue_length,
            img_size=self._img_size,
            num_cams=self._num_cams,
            use_lidar=self._use_lidar,
            class_names=self._class_names,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
        )

    def configure_optimizers(self, **cfg):
        # 推荐配置
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=2e-4,  # Transformer 学习率通常比 CNN 低
            weight_decay=1e-2,
        )
        return optimizer, None, "none"

    def prepare_batch(self, batch, ctx: Dict[str, Any]):
        device = ctx["device"]
        batch = dict(batch)
        batch["imgs"] = batch["imgs"].to(device, non_blocking=True)
        if "lidar_points" in batch and batch["lidar_points"] is not None:
            batch["lidar_points"] = [
                p.to(device, non_blocking=True) for p in batch["lidar_points"]
            ]
        batch["gt_boxes"] = [b.to(device, non_blocking=True) for b in batch["gt_boxes"]]
        batch["gt_labels"] = [b.to(device, non_blocking=True) for b in batch["gt_labels"]]
        return batch

    def _forward(self, batch):
        imgs = batch["imgs"]  # (B, T, N, C, H, W)
        lidar_points = batch.get("lidar_points", None)  # List[Tensor(N, 4)]
        metas = batch["metas"]  # List[List[Dict]] (B, T)
        gt_boxes = batch["gt_boxes"]  # List[Tensor]
        gt_labels = batch["gt_labels"]  # List[Tensor]

        B, T, N, C, H, W = imgs.shape
        imgs_reshaped = imgs.view(B * T * N, C, H, W)
        features = self._backbone(imgs_reshaped)  # Tuple of feature maps
        features = self._neck(features)  # FPN 输出特征图列表

        # 解析相机参数
        camera_params = self._parse_camera_params(metas, B, T, N)  # (B, T, N, 4, 4)
        ego_emotion = self._parse_ego_emotion(metas, B, T)  # (B, T, 4, 4)
        # Image BEV Encoding
        # 输入：多尺度特征，相机参数
        # 输出：(B, C, BEV_H, BEV_W)
        img_bev_feat = self._img_bev_encoder(
            features,
            camera_params,
            ego_emotion,
            batch_size=B,
            seq_len=T,
            img_size=(H, W),
        )

        # Lidar Encoding
        lidar_bev_feat = None
        if self._use_lidar and lidar_points:
            lidar_bev_feat = self._lidar_encoder(lidar_points)  # (B, C, BEV_H, BEV_W)

        # 融合 Image BEV 和 Lidar BEV 特征
        if lidar_bev_feat is not None:
            fused_bev_feat = torch.cat(
                [img_bev_feat, lidar_bev_feat], dim=1
            )  # (B, 2C, H, W)
            bev_feat = self.fusion_layer(fused_bev_feat)  # (B, C, H, W)
        else:
            bev_feat = img_bev_feat  # (B, C, H, W)

        # 检测头
        preds = self._head(bev_feat)
        loss_dict = self._head.loss(preds, gt_boxes, gt_labels)
        loss_sum = sum(loss_dict.values())
        metrics = {k: v.detach().item() for k, v in loss_dict.items()}
        return preds, loss_sum, metrics

    def _step(self, batch, ctx: Dict[str, Any]):
        _, loss_sum, metrics = self._forward(batch)
        return {"loss": loss_sum, "metrics": metrics}

    def _parse_camera_params(self, metas, B, T, N):
        """
        计算从 当前帧 Ego 坐标系 到 Image 像素坐标系 的投影矩阵。
            bev_point(ego) --> cam_point --> img_point
        矩阵形式为：
        lidar2img = K @ inv(cam2ego)

        Returns:
            lidar2img: (B, T, N, 4, 4)
        """
        lidar2img_list = []
        for b in range(B):
            t_list = []
            for t in range(T):
                n_list = []
                frame_meta = metas[b][t]
                frame_cam2egos = frame_meta["cam2egos"]  # (N, 4, 4)
                frame_intrinsics = frame_meta["intrinsics"]  # (N, 4, 4)

                for n in range(N):
                    K = frame_intrinsics[n]  # (4, 4)
                    cam2ego = frame_cam2egos[n]  # (4, 4
                    proj_matrix = torch.matmul(K, torch.inverse(cam2ego))  # (4, 4)
                    n_list.append(proj_matrix)

                t_list.append(torch.stack(n_list))  # (N, 4, 4)
            lidar2img_list.append(torch.stack(t_list))  # (T, N, 4, 4)

        return torch.stack(lidar2img_list).to(
            next(self.parameters()).device
        )  # (B, T, N, 4, 4)
    
    def _parse_ego_emotion(self, metas, B, T):
        '''
        T-1时刻坐标对齐到T时刻坐标系下的变换矩阵
        '''
        ego_emotion_list = []
        for b in range(B):
            t_list = []
            for t in range(T):
                if t == 0:
                    t_list.append(torch.eye(4))
                else:
                    curr_ego2global = metas[b][t]["ego2globals"][0]  # (4, 4)
                    prev_ego2global = metas[b][t-1]["ego2globals"][0]  # (4, 4)
                    if not isinstance(curr_ego2global, torch.Tensor):
                        curr_ego2global = torch.tensor(curr_ego2global)
                    if not isinstance(prev_ego2global, torch.Tensor):
                        prev_ego2global = torch.tensor(prev_ego2global)
                    ego2ego = torch.matmul(
                        torch.inverse(curr_ego2global), prev_ego2global
                    )  # (4, 4)
                    t_list.append(ego2ego)
            ego_emotion_list.append(torch.stack(t_list))  # (T, 4, 4)
        return torch.stack(ego_emotion_list).to(
            next(self.parameters()).device
        )  # (B, T, 4, 4)

    def _boxes_ego_to_global(self, boxes: torch.Tensor, ego2global: torch.Tensor):
        if boxes.numel() == 0:
            return boxes
        if not isinstance(ego2global, torch.Tensor):
            ego2global = torch.tensor(ego2global, dtype=boxes.dtype)
        ego2global = ego2global.to(device=boxes.device, dtype=boxes.dtype)

        centers = boxes[:, 0:3]
        ones = torch.ones((boxes.shape[0], 1), device=boxes.device, dtype=boxes.dtype)
        centers_h = torch.cat([centers, ones], dim=1)
        centers_global = (ego2global @ centers_h.t()).t()[:, :3]

        rot = ego2global[:3, :3]
        yaws = boxes[:, 6]
        dir_xy = torch.stack(
            [torch.cos(yaws), torch.sin(yaws), torch.zeros_like(yaws)], dim=1
        )
        dir_global = (rot @ dir_xy.t()).t()
        yaws_global = torch.atan2(dir_global[:, 1], dir_global[:, 0])

        vel = boxes[:, 7:9]
        vel3 = torch.cat(
            [vel, torch.zeros((boxes.shape[0], 1), device=boxes.device, dtype=boxes.dtype)],
            dim=1,
        )
        vel_global = (rot @ vel3.t()).t()[:, :2]

        out = boxes.clone()
        out[:, 0:3] = centers_global
        out[:, 6] = yaws_global
        out[:, 7:9] = vel_global
        return out

    def training_step(self, batch, ctx: Dict[str, Any]):
        return self._step(batch, ctx)

    def validation_step(self, batch, ctx: Dict[str, Any]):
        preds, loss_sum, metrics = self._forward(batch)
        imgs = batch["imgs"]
        metas = batch["metas"]
        gt_boxes = batch["gt_boxes"]
        gt_labels = batch["gt_labels"]

        if self._val_epoch != ctx.get("epoch"):
            self._val_epoch = ctx.get("epoch")
            self._val_results = []

        dets = decode_center_head(
            preds["hm"],
            preds["reg"],
            self._point_cloud_range,
            score_thresh=self._val_score_thresh,
            max_per_img=self._val_max_per_img,
        )

        b, t, n, _, h, w = imgs.shape
        for i in range(b):
            frame_meta = metas[i][-1]
            sample_token = frame_meta.get("sample_token", "")
            boxes_ego = dets[i]["boxes_3d"].detach().cpu()
            ego2global = frame_meta["ego2globals"][0]
            boxes_global = self._boxes_ego_to_global(boxes_ego, ego2global)
            self._val_results.append(
                {
                    "sample_token": sample_token,
                    "boxes_3d": boxes_global,
                    "scores": dets[i]["scores"].detach().cpu(),
                    "labels": dets[i]["labels"].detach().cpu(),
                }
            )

            if self._val_vis_enable and ctx.get("is_main_process", False):
                run_dir = ctx.get("run_dir", "")
                vis_dir = os.path.join(run_dir, self._val_vis_dir)
                file_name = f"{sample_token}.jpg" if sample_token else f"step_{ctx.get('step_in_epoch', 0)}.jpg"
                save_path = os.path.join(vis_dir, file_name)
                visualize_sample(
                    frame_meta,
                    {
                        "boxes_3d": boxes_ego,
                        "scores": dets[i]["scores"].detach().cpu(),
                        "labels": dets[i]["labels"].detach().cpu(),
                        "hm": preds["hm"][i].detach().cpu(),
                    },
                    gt_boxes[i].detach().cpu(),
                    gt_labels[i].detach().cpu(),
                    self._class_names,
                    save_path,
                    img_size=(h, w),
                    topk=self._val_vis_topk,
                )

        return {"loss": loss_sum, "metrics": metrics}

    def inference_step(self, batch, ctx: Dict[str, Any]):
        return self.validation_step(batch, ctx)

    def validation_epoch_end(self, ctx: Dict[str, Any]):
        if not self._val_results:
            return {}

        all_results = self._val_results
        if dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, self._val_results)
            all_results = []
            for part in gathered:
                all_results.extend(part or [])

        if not ctx.get("is_main_process", False):
            return {}

        run_dir = ctx.get("run_dir", "")
        output_dir = os.path.join(run_dir, "metrics")
        results = format_nuscenes_results(
            all_results,
            self._class_names,
            score_thresh=self._val_score_thresh,
            speed_thresh=self._val_speed_thresh,
            use_lidar=self._use_lidar,
        )
        metrics_summary, _ = evaluate_nuscenes(
            data_root=self._data_root,
            version=self._data_version,
            results=results,
            output_dir=output_dir,
            eval_cfg=self._val_eval_cfg,
            eval_set=self._val_eval_set,
        )
        summary = metrics_summary if isinstance(metrics_summary, dict) else {}
        tp_errors = summary.get("tp_errors", {})
        metrics = {
            "nuscenes/mAP": float(summary.get("mean_ap", 0.0)),
            "nuscenes/NDS": float(summary.get("nd_score", 0.0)),
            "nuscenes/ATE": float(tp_errors.get("trans_err", 0.0)),
            "nuscenes/ASE": float(tp_errors.get("scale_err", 0.0)),
            "nuscenes/AOE": float(tp_errors.get("orient_err", 0.0)),
            "nuscenes/AVE": float(tp_errors.get("vel_err", 0.0)),
            "nuscenes/AAE": float(tp_errors.get("attr_err", 0.0)),
        }
        return metrics
    
    def get_freeze_targets(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self._backbone,
            "neck": self._neck,
        }

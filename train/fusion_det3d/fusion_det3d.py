import torch

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


class FusionDet3DTask(nn.Module):
    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Optional[Dict[str, Any]] = None,
        train: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._data_cfg = dict(data or {})
        self._model_cfg = dict(model or {})
        self._train_cfg = dict(train or {})
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
            num_cams=self._num_cams,
            img_size=self._img_size,
        )

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

    def _step(self, batch, ctx: Dict[str, Any]):
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

    def training_step(self, batch, ctx: Dict[str, Any]):
        return self._step(batch, ctx)

    def validation_step(self, batch, ctx: Dict[str, Any]):
        return self._step(batch, ctx)
    
    def get_freeze_targets(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self._backbone,
            "neck": self._neck,
        }

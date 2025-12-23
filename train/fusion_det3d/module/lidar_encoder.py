import torch
import torch.nn as nn


class LidarEncoder(nn.Module):
    def __init__(self, out_channels=256, pc_range=None, voxel_size=None, bev_shape=None):
        super().__init__()
        # 实际应包含 Voxelization -> PillarFeatureNet -> Backbone2D
        # 这里用一个简单的 Conv 层模拟处理过程
        self.pre_process = nn.Linear(4, 64) 
        self.dummy_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        if bev_shape is None:
            self.out_h = 200
            self.out_w = 200
        else:
            self.out_h = int(bev_shape[0])
            self.out_w = int(bev_shape[1])

    def forward(self, lidar_points_list):
        # lidar_points_list: list of (N, 4) tensors
        batch_size = len(lidar_points_list)
        
        # 1. Voxelization (略)
        # 2. Scatter to BEV
        # 为了代码跑通，直接生成一个随机/全0特征图
        device = lidar_points_list[0].device
        dummy_bev = torch.zeros((batch_size, 64, self.out_h, self.out_w), device=device)
        
        # 3. 2D Backbone
        out = self.dummy_conv(dummy_bev)
        return out # (B, 256, 200, 200)

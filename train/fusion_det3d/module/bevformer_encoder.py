import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict
from mmcv.ops import multi_scale_deform_attn

def get_reference_points(bev_h, bev_w, bev_z, device):
    '''
    生成bev空间下的归一化网格点
    '''
    xs = torch.linspace(0 + 0.5, bev_w - 0.5, bev_w, device=device) / bev_w
    ys = torch.linspace(0 + 0.5, bev_h - 0.5, bev_h, device=device) / bev_h

    ref_y, ref_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)
    ref_x = ref_x.reshape(-1)[None]
    ref_y = ref_y.reshape(-1)[None]

    ref_z = torch.linspace(0 + 0.5, bev_z - 0.5, bev_z, device=device) / bev_z  # (D,)
    ref_z = ref_z[None, None, :].repeat(1, bev_h*bev_w, 1)

    ref_2d = torch.stack((ref_x, ref_y), -1).reshape(1, bev_h*bev_w, 1, 2).repeat(1, 1, bev_z, 1)
    ref_3d = torch.cat((ref_2d, ref_z.unsqueeze(-1)), -1)  # (1, H*W, D, 3)

    return ref_3d  # (1, H*W, D, 3)


def temporal_alignmnet(prev_bev, ego_emotion, pc_range):
    '''
    利用ego motion对齐BEV特征
    prev_bev: (B, C, H, W)
    ego_emotion: (B, 4, 4)
    '''
    B, C, H, W = prev_bev.shape
    device = prev_bev.device

    # 1. 生成当前帧网格
    xs = torch.linspace(0, W - 1, W, device=device, dtype=prev_bev.dtype)
    ys = torch.linspace(0, H - 1, H, device=device, dtype=prev_bev.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

    # 2. 像素坐标系 -> 世界坐标系
    real_y = (xx + 0.5) / W * (pc_range[3] - pc_range[0]) + pc_range[0]
    real_x = (yy + 0.5) / H * (pc_range[4] - pc_range[1]) + pc_range[1]

    grid_3d = torch.stack((real_x, real_y, torch.zeros_like(real_x), torch.ones_like(real_x)), dim=-1)  # (H, W, 4)
    grid_3d = grid_3d.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 4)

    # 3. 世界坐标系 -> 对齐后的像素坐标系
    matrix_inv = torch.inverse(ego_emotion).view(B, 1, 1, 4, 4)  # (B, 1, 1, 4, 4)
    grid_prev_3d = torch.matmul(matrix_inv, grid_3d.unsqueeze(-1)).squeeze(-1)  # (B, H, W, 4)

    # 4. 像素坐标系归一化
    norm_x = (grid_prev_3d[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0]) * 2 - 1
    norm_y = (grid_prev_3d[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1]) * 2 - 1

    sample_grid = torch.stack((norm_x, norm_y), dim=-1)  # (B, H, W, 2)

    # 5. 双线性采样
    aligned_bev = F.grid_sample(prev_bev, sample_grid, align_corners=False, padding_mode='zeros')  # (B, C, H, W)
    return aligned_bev



class BEVFormerEncoder(nn.Module):
    def __init__(self, bev_h, bev_w, embed_dims=256):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        
        # 定义 BEV Query (可学习参数)
        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)
        
        # 简单的 Attention 层 (实际 BEVFormer 这里是 Deformable Attention)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dims, nhead=8)

    def forward(self, mlvl_feats, camera_params, ego_emotion, batch_size, seq_len):
        """
        mlvl_feats: tuple of (B*T*N, C, H, W)
        """
        # 1. 准备 BEV Queries
        bev_queries = self.bev_embedding.weight.unsqueeze(1).repeat(1, batch_size, 1) # (HW, B, C)
        
        # 2. 这里应该进行 Temporal Self Attention (省略)
        
        # 3. 这里应该进行 Spatial Cross Attention (Image -> BEV)
        # 真正的 BEVFormer 会利用 camera_params 做投影采样
        # 这里为了跑通，我们做一个非常粗暴的 Global Average Pooling 模拟映射 (仅用于代码调试)
        # !!! 实际上这里必须替换为 Deformable Cross Attention !!!
        
        # 取最后一层特征，平均池化并扩展到 BEV 维度 (这只是为了让维度对齐，没有任何几何意义！)
        feat = mlvl_feats[-1] # (B*T*N, C, H, W)
        feat = torch.mean(feat, dim=[2, 3]) # (B*T*N, C)
        feat = feat.view(batch_size, -1, self.embed_dims) # (B, T*N, C)
        feat = torch.mean(feat, dim=1) # (B, C)
        
        # 广播到 BEV grid (再次强调，这是假的逻辑)
        bev_feat = feat.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.bev_h, self.bev_w)
        
        return bev_feat # (B, C, H, W)
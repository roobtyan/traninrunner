import torch
import torch.nn as nn
from typing import Any, Dict


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

    def forward(self, mlvl_feats, camera_params, batch_size, seq_len):
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
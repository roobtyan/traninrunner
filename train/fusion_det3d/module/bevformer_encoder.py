import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops import MultiScaleDeformableAttention


def get_reference_points(bev_h, bev_w, num_z, pc_range, device):
    '''
    生成BEV 3D参考点 (H*W*Z, 3)
    '''
    xs = torch.linspace(0.5, bev_w - 0.5, bev_w, device=device) / bev_w
    ys = torch.linspace(0.5, bev_h - 0.5, bev_h, device=device) / bev_h
    zs = torch.linspace(0.5, num_z - 0.5, num_z, device=device) / num_z

    ref_y, ref_x = torch.meshgrid(ys, xs, indexing='ij')
    ref_xy = torch.stack((ref_x, ref_y), -1).reshape(-1, 2)  # (H*W, 2)
    ref_xy = ref_xy[:, None, :].repeat(1, num_z, 1)  # (H*W, Z, 2)
    ref_z = zs[None, :, None].repeat(bev_h * bev_w, 1, 1)  # (H*W, Z, 1)

    ref_3d = torch.cat((ref_xy, ref_z), dim=-1)  # (H*W, Z, 3)
    # 还原到真实坐标系
    ref_3d[..., 0] = ref_3d[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    ref_3d[..., 1] = ref_3d[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    ref_3d[..., 2] = ref_3d[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

    return ref_3d.reshape(-1, 3)


def temporal_alignmnet(prev_bev, ego_motion, pc_range):
    '''
    利用ego motion对齐BEV特征
    prev_bev: (B, C, H, W)
    ego_motion: (B, 4, 4)
    '''
    B, C, H, W = prev_bev.shape
    device = prev_bev.device

    xs = torch.linspace(0, W - 1, W, device=device, dtype=prev_bev.dtype)
    ys = torch.linspace(0, H - 1, H, device=device, dtype=prev_bev.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')

    real_x = (xx + 0.5) / W * (pc_range[3] - pc_range[0]) + pc_range[0]
    real_y = (yy + 0.5) / H * (pc_range[4] - pc_range[1]) + pc_range[1]

    grid_3d = torch.stack((real_x, real_y, torch.zeros_like(real_x), torch.ones_like(real_x)), dim=-1)
    grid_3d = grid_3d.unsqueeze(0).repeat(B, 1, 1, 1) # (B, H, W, 4)

    matrix_inv = torch.inverse(ego_motion).view(B, 1, 1, 4, 4)
    grid_prev_3d = torch.matmul(matrix_inv, grid_3d.unsqueeze(-1)).squeeze(-1)

    norm_x = (grid_prev_3d[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0]) * 2 - 1
    norm_y = (grid_prev_3d[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1]) * 2 - 1
    sample_grid = torch.stack((norm_x, norm_y), dim=-1)

    aligned_bev = F.grid_sample(prev_bev, sample_grid, align_corners=False, padding_mode='zeros')
    return aligned_bev


class SpatialCrossAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, num_levels, num_points, num_cams):
        super().__init__()
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.ms_deform_attn = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=True,
        )

    def forward(self, query, query_pos, mlvl_feats, lidar2img, reference_points, img_size):
        '''
        BEV query (3D)
        ↓ 投影
        Image reference points (2D)
        ↓ deformable attention
        Multi-scale image features
        ↓ 多相机平均
        Updated BEV features
        '''
        # query: (B, H*W*Z, C)
        # mlvl_feats: list[(B, N, C, H, W)]
        # lidar2img: (B, N, 4, 4)
        B, num_query, _ = query.shape
        device = query.device

        # 3D参考点投影到图像平面，获取2D参考点和BEV掩码
        ref_2d, bev_mask = self._point_sampling(reference_points, lidar2img, img_size)

        spatial_shapes = torch.as_tensor(
            [[feat.shape[-2], feat.shape[-1]] for feat in mlvl_feats],
            device=device,
            dtype=torch.long,
        )
        level_start_index = torch.cat(
            [spatial_shapes.new_zeros(1), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        output = torch.zeros_like(query)
        valid = torch.zeros((B, num_query), device=device)
        for cam in range(self.num_cams):
            mask_cam = bev_mask[:, cam]
            if mask_cam.sum() == 0:
                continue

            ref_cam = ref_2d[:, cam].unsqueeze(2).repeat(1, 1, self.num_levels, 1)
            value = torch.cat(
                [feat[:, cam].flatten(2).transpose(1, 2) for feat in mlvl_feats],
                dim=1,
            )

            attn_out = self.ms_deform_attn(
                query=query,
                value=value,
                identity=torch.zeros_like(query),
                query_pos=query_pos,
                reference_points=ref_cam,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
            weight = mask_cam.unsqueeze(-1).float()
            output += attn_out * weight
            valid += mask_cam.float()

        valid = valid.clamp(min=1.0)
        output = output / valid.unsqueeze(-1)
        return output

    def _point_sampling(self, reference_points, lidar2img, img_size):
        # reference_points: (H*W*Z, 3)
        # lidar2img: (B, N, 4, 4)
        B, N = lidar2img.shape[:2]
        num_query = reference_points.shape[0] # H*W*Z
        device = reference_points.device

        ref_points = torch.cat(
            [reference_points, torch.ones((num_query, 1), device=device)], dim=-1
        ) # (H*W*Z, 4)
        ref_points = ref_points.view(1, 1, num_query, 4, 1).repeat(B, N, 1, 1, 1)
        lidar2img = lidar2img.view(B, N, 1, 4, 4)
        cam_points = torch.matmul(lidar2img, ref_points).squeeze(-1)

        eps = 1e-5
        depth = cam_points[..., 2]
        u = cam_points[..., 0] / (depth + eps)
        v = cam_points[..., 1] / (depth + eps)
        img_h, img_w = img_size
        u_norm = u / img_w
        v_norm = v / img_h

        ref_2d = torch.stack((u_norm, v_norm), dim=-1)
        bev_mask = (depth > eps) & (u_norm > 0.0) & (u_norm < 1.0) & (v_norm > 0.0) & (v_norm < 1.0)
        return ref_2d, bev_mask


class BEVFormerLayer(nn.Module):
    def __init__(self, embed_dims, num_heads, num_levels, num_points, num_cams, ffn_dim, num_z):
        super().__init__()
        self.num_z = num_z
        self.temporal_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
        self.spatial_attn = SpatialCrossAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            num_cams=num_cams,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(self, bev, bev_pos, prev_bev, mlvl_feats, lidar2img, img_size, reference_points):
        # bev: (B, H*W, C)
        if prev_bev is not None:
            q = bev + bev_pos
            k = prev_bev + bev_pos
            attn_out, _ = self.temporal_attn(q, k, prev_bev)
            bev = self.norm1(bev + attn_out)

        b, num_query, c = bev.shape
        bev_z = bev.unsqueeze(2).repeat(1, 1, self.num_z, 1).reshape(b, -1, c) # (B, H*W*Z, C)
        bev_pos_z = bev_pos.unsqueeze(2).repeat(1, 1, self.num_z, 1).reshape(b, -1, c) # (B, H*W*Z, C)

        spatial_out = self.spatial_attn(
            bev_z, bev_pos_z, mlvl_feats, lidar2img, reference_points, img_size
        )
        spatial_out = spatial_out.view(b, num_query, self.num_z, c).mean(dim=2)
        bev = self.norm2(bev + spatial_out)

        ffn_out = self.ffn(bev)
        bev = self.norm3(bev + ffn_out)
        return bev


class BEVFormerEncoder(nn.Module):
    def __init__(
        self,
        bev_h,
        bev_w,
        embed_dims=256,
        num_heads=8,
        num_levels=3,
        num_points=8,
        num_layers=1,
        num_z=8,
        pc_range=None,
        num_cams=6,
        img_size=None,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.num_z = num_z
        self.pc_range = pc_range or [-60.0, -60.0, -4.0, 140.0, 60.0, 4.0]
        self.num_cams = num_cams
        self.img_size = img_size

        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)
        self.bev_pos_embedding = nn.Embedding(bev_h * bev_w, embed_dims)

        ffn_dim = embed_dims * 2
        self.layers = nn.ModuleList([
            BEVFormerLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                num_cams=num_cams,
                ffn_dim=ffn_dim,
                num_z=num_z,
            )
            for _ in range(num_layers)
        ])

    def forward(self, mlvl_feats, lidar2img, ego_motion, batch_size, seq_len, img_size=None):
        # mlvl_feats: list[(B*T*N, C, H, W)]
        num_cams = lidar2img.shape[2]
        img_size = img_size or self.img_size

        mlvl_feats = [
            feat.view(batch_size, seq_len, num_cams, feat.shape[1], feat.shape[2], feat.shape[3])
            for feat in mlvl_feats
        ]

        bev = self.bev_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        bev_pos = self.bev_pos_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        reference_points = get_reference_points(
            self.bev_h, self.bev_w, self.num_z, self.pc_range, bev.device
        )

        prev_bev = None
        for t in range(seq_len):
            feats_t = [feat[:, t] for feat in mlvl_feats]
            lidar2img_t = lidar2img[:, t]

            prev_bev_aligned = None
            if prev_bev is not None and ego_motion is not None:
                motion_t = ego_motion[:, t]
                prev_bev_aligned = temporal_alignmnet(prev_bev, motion_t, self.pc_range)
                prev_bev_aligned = prev_bev_aligned.flatten(2).transpose(1, 2)

            bev_t = bev
            for layer in self.layers:
                bev_t = layer(
                    bev_t,
                    bev_pos,
                    prev_bev_aligned,
                    feats_t,
                    lidar2img_t,
                    img_size,
                    reference_points,
                )

            prev_bev = bev_t.transpose(1, 2).reshape(
                batch_size, self.embed_dims, self.bev_h, self.bev_w
            )

        return prev_bev

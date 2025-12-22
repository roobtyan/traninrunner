import torch
import torch.nn as nn

class BEVDet3DHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 这是一个简单的 CenterHead 风格
        self.heatmap_head = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1) # xyz, whl, rot, vel

    def forward(self, x):
        hm = self.heatmap_head(x)
        reg = self.reg_head(x)
        return {'hm': hm, 'reg': reg}

    def loss(self, preds, gt_boxes, gt_labels):
        # 这是一个 Mock 的 loss
        # 实际需要生成 Gaussian Target Map 并计算 Focal Loss + L1 Loss
        hm_pred = preds['hm']
        loss_val = hm_pred.sum() * 0.0 + 1.0 # 保持计算图连通
        
        return {'loss_heatmap': loss_val, 'loss_bbox': loss_val}
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        depth: int = 50,
        pretrained: bool = True,
        out_strides: list = [8, 16, 32],
        train_backbone: bool = False,
    ):
        super().__init__()
        self.out_strides = out_strides

        # 1. 加载 torchvision 的 ResNet
        if depth == 50:
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet50(weights=weights)
        elif depth == 18:  # 备选，显存不够时用
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported depth: {depth}")

        # 2. 拆解层 (ResNet 结构: Stem -> Layer1(s4) -> Layer2(s8) -> Layer3(s16) -> Layer4(s32))
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # stride 4,  dim=256 (ResNet50) / 64 (ResNet18)
        self.layer2 = backbone.layer2  # stride 8,  dim=512 (ResNet50) / 128 (ResNet18)
        self.layer3 = backbone.layer3  # stride 16, dim=1024 (ResNet50) / 256 (ResNet18)
        self.layer4 = backbone.layer4  # stride 32, dim=2048 (ResNet50) / 512 (ResNet18)

        # 3. 冻结权重逻辑（默认冻结，除非显式传入 train_backbone=True）
        if not train_backbone:
            for param in self.parameters():
                param.requires_grad = False

        # 删除不需要的全连接层以节省一点点内存
        del backbone.fc
        del backbone.avgpool

    def forward(self, x):
        # 假设输入已经被处理为 (B*N, 3, H, W)
        x = self.stem(x)  # Stride 4
        c2 = self.layer1(x)  # Stride 4
        c3 = self.layer2(c2)  # Stride 8
        c4 = self.layer3(c3)  # Stride 16
        c5 = self.layer4(c4)  # Stride 32

        outputs = []
        # 根据请求的 stride 返回对应的特征图
        if 8 in self.out_strides:
            outputs.append(c3)
        if 16 in self.out_strides:
            outputs.append(c4)
        if 32 in self.out_strides:
            outputs.append(c5)

        return tuple(outputs)

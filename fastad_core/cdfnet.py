import torch
import torch.nn as nn

class CDFNet(nn.Module):
    def __init__(self):
        super(CDFNet, self).__init__()

        # ---------------------------------------------------------
        # 1. Stem (起手式): 保持 ResNet 经典结构
        # 作用：快速将图像尺寸缩小 4 倍，捕捉大尺度边缘
        # Output: (B, 64, H/4, W/4)
        # ---------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ---------------------------------------------------------
        # 2. Body (主体): 3个连续的下采样层
        # 逻辑：每一层负责 "尺寸减半，通道翻倍"
        # ---------------------------------------------------------
        self.body = nn.Sequential(
            # Layer 1: /4 -> /8
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            # Layer 2: /8 -> /16
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),

            # Layer 3: /16 -> /32 (达到目标尺寸)
            # 这里直接升维到 1024
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, bias=True)
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.stem(x)      # -> (B, 64, H/4, W/4)
        x = self.body(x)      # -> (B, 1024, H/32, W/32)
        x = self.head(x)      # -> (B, 1024, H/32, W/32)
        return x
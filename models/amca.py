"""AMCA: adaptive multi-scale context aggregation module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AMCA(nn.Module):
    def __init__(self, channels, reduction=4):
        super(AMCA, self).__init__()
        hidden = max(channels // reduction, 32)

        # 3x3 local receptive field
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

        # 3x3 dilated convolution (dilation = 2)
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

        # 5x5 large receptive field
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=5,
                padding=2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

        # Predict spatially adaptive weights for the three scales
        self.scale_selector = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)

        multi_scale = torch.cat([f1, f2, f3], dim=1)
        weights = self.scale_selector(multi_scale)
        weights = F.softmax(weights, dim=1)

        # Keep detached weights available for optional visualization.
        self.last_scale_weights = weights.detach()

        w1 = weights[:, 0:1]
        w2 = weights[:, 1:2]
        w3 = weights[:, 2:3]

        out = w1 * f1 + w2 * f2 + w3 * f3
        out = self.out_proj(out)
        return x + out

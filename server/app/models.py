"""
Attention Residual U-Net — vendored from iteris/models.py so this Space has
no dependency on the training package (which pulls in MONAI). Architecture
must stay byte-identical to the training code or the checkpoint won't load.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        return self.relu(
            self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.skip(x)
        )


class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_x: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_x, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_up = F.interpolate(self.W_g(g), size=x.shape[2:], mode='bilinear', align_corners=False)
        alpha = self.psi(self.relu(g_up + self.W_x(x)))
        return x * alpha


class AttentionResUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 4, features=(64, 128, 256, 512)):
        super().__init__()
        F_ = features
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ResBlock(in_channels, F_[0])
        self.enc2 = ResBlock(F_[0], F_[1])
        self.enc3 = ResBlock(F_[1], F_[2])
        self.enc4 = ResBlock(F_[2], F_[3])
        self.bottleneck = ResBlock(F_[3], F_[3] * 2)

        self.att4 = AttentionGate(F_[3] * 2, F_[3], F_[3] // 2)
        self.att3 = AttentionGate(F_[3], F_[2], F_[2] // 2)
        self.att2 = AttentionGate(F_[2], F_[1], F_[1] // 2)
        self.att1 = AttentionGate(F_[1], F_[0], F_[0] // 2)

        self.up4 = nn.ConvTranspose2d(F_[3] * 2, F_[3], 2, stride=2)
        self.dec4 = ResBlock(F_[3] * 2, F_[3])
        self.up3 = nn.ConvTranspose2d(F_[3], F_[2], 2, stride=2)
        self.dec3 = ResBlock(F_[2] * 2, F_[2])
        self.up2 = nn.ConvTranspose2d(F_[2], F_[1], 2, stride=2)
        self.dec2 = ResBlock(F_[1] * 2, F_[1])
        self.up1 = nn.ConvTranspose2d(F_[1], F_[0], 2, stride=2)
        self.dec1 = ResBlock(F_[0] * 2, F_[0])

        self.head = nn.Conv2d(F_[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), self.att4(b, e4)], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), self.att3(d4, e3)], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.att2(d3, e2)], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.att1(d2, e1)], 1))

        return self.head(d1)

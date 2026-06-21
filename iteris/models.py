"""
Segmentation network architectures.

`AttentionResUNet` is the baseline for all datasets. ResNet-34 style
encoder, soft attention gates on skip connections (Oktay et al. 2018),
transposed-conv decoder. `in_channels` and `num_classes` are the only
parameters that differ between datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Two-conv residual block with projection shortcut for channel mismatch."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.skip  = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        return self.relu(
            self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.skip(x)
        )


class AttentionGate(nn.Module):
    """
    Soft attention gate (Oktay et al., 2018).
    g = gating signal from decoder; x = skip connection from encoder.
    """

    def __init__(self, F_g: int, F_x: int, F_int: int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(F_g,   F_int, 1, bias=False),
                                  nn.BatchNorm2d(F_int))
        self.W_x  = nn.Sequential(nn.Conv2d(F_x,   F_int, 1, bias=False),
                                  nn.BatchNorm2d(F_int))
        self.psi  = nn.Sequential(nn.Conv2d(F_int, 1,     1, bias=False),
                                  nn.BatchNorm2d(1),
                                  nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_up = F.interpolate(self.W_g(g), size=x.shape[2:],
                             mode='bilinear', align_corners=False)
        alpha = self.psi(self.relu(g_up + self.W_x(x)))
        return x * alpha


class AttentionResUNet(nn.Module):
    """
    Attention Residual U-Net — dataset-agnostic.

    Parameters
    ----------
    in_channels : 1 for greyscale (CAMUS / CHAOS / ACDC / DRIVE / ISIC),
                  4 for BraTS multi-parametric MRI.
    num_classes : from cfg['num_classes'].
    features    : channel widths at each encoder scale.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        features=(64, 128, 256, 512),
    ):
        super().__init__()
        F_ = features
        self.pool       = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = ResBlock(in_channels, F_[0])
        self.enc2 = ResBlock(F_[0], F_[1])
        self.enc3 = ResBlock(F_[1], F_[2])
        self.enc4 = ResBlock(F_[2], F_[3])
        self.bottleneck = ResBlock(F_[3], F_[3] * 2)

        # Attention gates
        self.att4 = AttentionGate(F_[3] * 2, F_[3], F_[3] // 2)
        self.att3 = AttentionGate(F_[3],     F_[2], F_[2] // 2)
        self.att2 = AttentionGate(F_[2],     F_[1], F_[1] // 2)
        self.att1 = AttentionGate(F_[1],     F_[0], F_[0] // 2)

        # Decoder
        self.up4  = nn.ConvTranspose2d(F_[3] * 2, F_[3], 2, stride=2)
        self.dec4 = ResBlock(F_[3] * 2, F_[3])
        self.up3  = nn.ConvTranspose2d(F_[3],     F_[2], 2, stride=2)
        self.dec3 = ResBlock(F_[2] * 2, F_[2])
        self.up2  = nn.ConvTranspose2d(F_[2],     F_[1], 2, stride=2)
        self.dec2 = ResBlock(F_[1] * 2, F_[1])
        self.up1  = nn.ConvTranspose2d(F_[1],     F_[0], 2, stride=2)
        self.dec1 = ResBlock(F_[0] * 2, F_[0])

        self.head = nn.Conv2d(F_[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  self.att4(b,  e4)], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), self.att3(d4, e3)], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.att2(d3, e2)], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.att1(d2, e1)], 1))

        return self.head(d1)


class LiteUNet(nn.Module):
    """
    Deliberately LIGHTWEIGHT plain U-Net — the RL warm-start baseline.

    No attention gates, no residual blocks, ~1/4 the channels of
    AttentionResUNet (≈ 20–30x fewer params). It is intentionally weaker so its
    segmentation errors are *systematic* (smooth, uniform over/under-segmentation)
    rather than the tiny scattered residuals of the strong attention net — which
    is exactly the error profile the RL contour agents CAN correct. This creates
    genuine headroom for refinement while staying a sane, recognisable baseline.

    Pair it with the attention net (kept as the upper-bound competitor): the story
    is "RL refinement lifts a lightweight baseline toward the heavyweight net".
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4,
                 features=(16, 32, 64, 128)):
        super().__init__()
        F_ = features
        self.pool = nn.MaxPool2d(2)

        def double_conv(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            )

        self.enc1 = double_conv(in_channels, F_[0])
        self.enc2 = double_conv(F_[0], F_[1])
        self.enc3 = double_conv(F_[1], F_[2])
        self.bottleneck = double_conv(F_[2], F_[3])

        self.up3  = nn.ConvTranspose2d(F_[3], F_[2], 2, stride=2)
        self.dec3 = double_conv(F_[2] * 2, F_[2])
        self.up2  = nn.ConvTranspose2d(F_[2], F_[1], 2, stride=2)
        self.dec2 = double_conv(F_[1] * 2, F_[1])
        self.up1  = nn.ConvTranspose2d(F_[1], F_[0], 2, stride=2)
        self.dec1 = double_conv(F_[0] * 2, F_[0])

        self.head = nn.Conv2d(F_[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


# Architecture registry — selected via cfg['model'] (default: attention Res-UNet).
_MODEL_REGISTRY = {
    'attn_resunet': AttentionResUNet,   # strong baseline / upper-bound competitor
    'lite_unet':    LiteUNet,           # lightweight RL warm-start baseline
}


def build_model(cfg: dict) -> nn.Module:
    """Factory. cfg['model'] selects the architecture (default attn_resunet).

    Both nets take the same (in_channels, num_classes), so the warm-start /
    DRL pipeline is architecture-agnostic — only the checkpoint + this key change.
    """
    in_channels = cfg.get('in_channels', 1)
    name = cfg.get('model', 'attn_resunet')
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](in_channels=in_channels, num_classes=cfg['num_classes'])

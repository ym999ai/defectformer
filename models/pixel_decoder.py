"""
Pixel Decoder.

Fuses {P2, P3, P4, P5} (all at channel d after MSFE) into a single
feature map E ∈ R^{H/8 × W/8 × d} via bilinear upsample + lateral 1×1
convolutions and element-wise summation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelDecoder(nn.Module):
    def __init__(self, d: int = 256):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(d, d, 1) for _ in range(4)])
        self.output_conv = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        features: [P2, P3, P4, P5]  — strides [4, 8, 16, 32], all with d channels
        returns:  E at stride-8 (P3 spatial size)
        """
        # Target = P3 spatial size (stride 8)
        target_h, target_w = features[1].shape[-2:]

        fused = None
        for feat, lat in zip(features, self.laterals):
            x = lat(feat)
            if x.shape[-2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)
            fused = x if fused is None else fused + x

        return self.output_conv(fused)

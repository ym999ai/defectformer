"""
Multi-Scale Feature Enhancement (MSFE) Module.

Contains:
  LTEC  - Local Texture Enhancement Convolution
  CSAG  - Cross-Scale Attention Gate
  MSFE  - bidirectional (top-down + bottom-up) feature enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTEC(nn.Module):
    """
    Local Texture Enhancement Convolution.
    Two sequential DW-Sep-Conv blocks; residual after the first block.

    F̂^(1) = PWConv(DWConv(F̂)) + F̂          (block 1 + Channel-Add)
    F̃     = PWConv(DWConv(F̂^(1)))             (block 2)
    """

    def __init__(self, d: int):
        super().__init__()
        self.dw1 = nn.Conv2d(d, d, 3, padding=1, groups=d, bias=False)
        self.pw1 = nn.Conv2d(d, d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(d)

        self.dw2 = nn.Conv2d(d, d, 3, padding=1, groups=d, bias=False)
        self.pw2 = nn.Conv2d(d, d, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(d)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.bn1(self.pw1(self.dw1(x)))) + x   # block 1 + residual
        return self.relu(self.bn2(self.pw2(self.dw2(h))))     # block 2


class CSAG(nn.Module):
    """
    Cross-Scale Attention Gate.

    α_{ℓk} = softmax_k( <F̄_ℓ, F̄_k> / √d )
    F̂_ℓ   = Σ_k  α_{ℓk} · Upsample(F_k, size(F_ℓ))
    """

    def __init__(self, d: int):
        super().__init__()
        self.scale = d ** -0.5

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        # Step 1: global average pooling → [B, d] per scale
        descriptors = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in features]

        # Step 2: similarity matrix [B, L, L]  (L = num_scales)
        desc = torch.stack(descriptors, dim=1)               # [B, L, d]
        A = torch.bmm(desc, desc.transpose(1, 2)) * self.scale  # [B, L, L]
        alpha = F.softmax(A, dim=-1)                          # row-wise softmax

        # Step 3: weighted aggregation
        result = []
        for l, feat_l in enumerate(features):
            target_size = feat_l.shape[-2:]
            agg = torch.zeros_like(feat_l)
            for k, feat_k in enumerate(features):
                up = F.interpolate(feat_k, size=target_size,
                                   mode='bilinear', align_corners=False)
                w = alpha[:, l, k].view(-1, 1, 1, 1)
                agg = agg + w * up
            result.append(agg)
        return result


class _MSFEPass(nn.Module):
    """Single CSAG + LTEC pass (used for both top-down and bottom-up)."""

    def __init__(self, d: int, num_scales: int):
        super().__init__()
        self.csag = CSAG(d)
        self.ltecs = nn.ModuleList([LTEC(d) for _ in range(num_scales)])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        aggregated = self.csag(features)
        return [self.ltecs[i](aggregated[i]) for i in range(len(features))]


class MSFE(nn.Module):
    """
    Multi-Scale Feature Enhancement Module.

    Projects backbone channels to uniform d, then applies CSAG+LTEC in
    top-down (C5→C2) and bottom-up (C2→C5) passes to yield {P_ℓ}.

    in_channels: [96, 192, 384, 768] for Swin-T
    """

    def __init__(self, in_channels: list[int], d: int = 256):
        super().__init__()
        num_scales = len(in_channels)

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, d, 1, bias=False),
                nn.BatchNorm2d(d),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        ])

        self.top_down = _MSFEPass(d, num_scales)
        self.bottom_up = _MSFEPass(d, num_scales)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        features: [F2, F3, F4, F5] from backbone  (order: fine→coarse)
        returns:  [P2, P3, P4, P5]
        """
        projected = [self.proj[i](features[i]) for i in range(len(features))]

        # Top-down: feed reversed list (coarse→fine), then restore order
        td_in = list(reversed(projected))
        td_out = list(reversed(self.top_down(td_in)))

        # Bottom-up: fine→coarse
        return self.bottom_up(td_out)

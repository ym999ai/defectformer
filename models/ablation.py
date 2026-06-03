"""
Configurable model variants for DefectFormer ablation studies.

Supports all 10 configurations from Table III of the paper:
  Baseline (ResNet-50, no MSFE, standard decoder)
  + Swin-T backbone
  + MSFE (full / csag_only / additive)
  + QDID (prior-init on/off, DMCA mask on/off, DCSA repulsion on/off)
  + Synthetic augmentation (handled at dataset level)

Entry point: DefectFormerAblation(config_name_or_kwargs)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .msfe import MSFE, CSAG, LTEC, _MSFEPass
from .pixel_decoder import PixelDecoder
from .qdid import QDID, DecoderLayer


# ─────────────────────────────────────────────────────────────────────────────
# Backbone registry
# ─────────────────────────────────────────────────────────────────────────────

_BACKBONE_CFGS = {
    "swin_tiny": {
        "timm_name":  "swin_tiny_patch4_window7_224",
        "out_indices": (0, 1, 2, 3),
        "nhwc": True,   # timm Swin outputs channels-last
    },
    "resnet50": {
        "timm_name":  "resnet50",
        "out_indices": (1, 2, 3, 4),   # strides 4,8,16,32
        "nhwc": False,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# MSFE variants
# ─────────────────────────────────────────────────────────────────────────────

class _AdditivePass(nn.Module):
    """Replaces CSAG with equal-weight aggregation; keeps LTEC."""

    def __init__(self, d: int, num_scales: int):
        super().__init__()
        self.ltecs = nn.ModuleList([LTEC(d) for _ in range(num_scales)])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        n = len(features)
        result = []
        for l, feat_l in enumerate(features):
            sz = feat_l.shape[-2:]
            agg = sum(
                F.interpolate(f, size=sz, mode="bilinear", align_corners=False)
                for f in features
            ) / n
            result.append(self.ltecs[l](agg))
        return result


class _CSAGOnlyPass(nn.Module):
    """CSAG without LTEC — used for 'LTEC removed' ablation."""

    def __init__(self, d: int):
        super().__init__()
        self.csag = CSAG(d)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        return self.csag(features)


class SimpleFPN(nn.Module):
    """
    Standard top-down FPN — used for 'baseline' and 'swin_backbone' configs
    (no MSFE at all: no CSAG, no LTEC, no bidirectional flow).
    """

    def __init__(self, in_channels: list[int], d: int = 256):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, d, 1, bias=False),
                nn.BatchNorm2d(d),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        ])
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, d, 3, padding=1, bias=False),
                nn.BatchNorm2d(d),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels
        ])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        projected = [self.proj[i](features[i]) for i in range(len(features))]
        # Simple top-down
        out = [None] * len(projected)
        out[-1] = projected[-1]
        for i in range(len(projected) - 2, -1, -1):
            up = F.interpolate(out[i + 1], size=projected[i].shape[-2:],
                               mode="bilinear", align_corners=False)
            out[i] = projected[i] + up
        return [self.output_convs[i](out[i]) for i in range(len(out))]


class MSFEVariant(nn.Module):
    """
    Factory that returns the correct MSFE/FPN variant.

    variant options:
      "full"      → full MSFE (CSAG + LTEC, bidirectional)
      "csag_only" → CSAG without LTEC
      "additive"  → bidirectional additive FPN + LTEC (no CSAG attention)
      "none"      → simple top-down FPN (baseline)
    """

    def __new__(cls, in_channels: list[int], d: int = 256,
                variant: str = "full") -> nn.Module:
        if variant == "full":
            return MSFE(in_channels, d)

        if variant == "none":
            return SimpleFPN(in_channels, d)

        # csag_only and additive both need the projection layers
        obj = super().__new__(cls)
        return obj

    def __init__(self, in_channels: list[int], d: int = 256,
                 variant: str = "full"):
        if variant in ("full", "none"):
            return   # already constructed via __new__

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

        if variant == "csag_only":
            self.top_down  = _CSAGOnlyPass(d)
            self.bottom_up = _CSAGOnlyPass(d)
        elif variant == "additive":
            self.top_down  = _AdditivePass(d, num_scales)
            self.bottom_up = _AdditivePass(d, num_scales)
        else:
            raise ValueError(f"Unknown MSFE variant: {variant}")

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        projected = [self.proj[i](features[i]) for i in range(len(features))]
        td_out = list(reversed(self.top_down(list(reversed(projected)))))
        return self.bottom_up(td_out)


# ─────────────────────────────────────────────────────────────────────────────
# QDID variant
# ─────────────────────────────────────────────────────────────────────────────

class QDIDVariant(QDID):
    """
    Configurable QDID that can disable:
      use_prior_init    → replace saliency seeding with learnable queries (DETR-style)
      use_dmca_mask     → pass all-ones mask (= standard unmasked cross-attention)
      use_dcsa_repulsion→ pass zero masks to DCSA  (= standard self-attention)
    """

    def __init__(self, d: int = 256, nq: int = 100,
                 num_layers: int = 6, num_classes: int = 15,
                 num_heads: int = 8,
                 use_prior_init: bool = True,
                 use_dmca_mask: bool = True,
                 use_dcsa_repulsion: bool = True):
        super().__init__(d, nq, num_layers, num_classes, num_heads)
        self.use_prior_init     = use_prior_init
        self.use_dmca_mask      = use_dmca_mask
        self.use_dcsa_repulsion = use_dcsa_repulsion

        if not use_prior_init:
            # Learnable content queries (DETR-style random init)
            self.learnable_queries = nn.Embedding(nq, d)

    def forward(self, E: torch.Tensor):
        B, d, Hf, Wf = E.shape
        L = Hf * Wf
        E_flat = E.view(B, d, L).permute(0, 2, 1)

        # ── Query initialisation ──────────────────────────────
        if self.use_prior_init:
            queries, M0 = self.initializer(E)
        else:
            queries = self.learnable_queries.weight.unsqueeze(0).expand(B, -1, -1)
            M0 = torch.full((B, self.nq, L), 1.0 / L, device=E.device)

        current_masks = M0
        all_cls_logits:  list[torch.Tensor] = []
        all_mask_logits: list[torch.Tensor] = []

        for layer in self.layers:
            # Attention mask for DMCA
            if self.use_dmca_mask:
                attn_mask = (current_masks > 0.5).float()
            else:
                attn_mask = torch.ones(B, self.nq, L, device=E.device)

            # Mask for DCSA IoU repulsion
            dcsa_masks = current_masks if self.use_dcsa_repulsion \
                         else torch.zeros(B, self.nq, L, device=E.device)

            queries     = layer(queries, E_flat, attn_mask, dcsa_masks)
            mask_logits = self._predict_mask_logits(queries, E_flat)
            current_masks = torch.sigmoid(mask_logits)

            all_mask_logits.append(mask_logits)
            all_cls_logits.append(self.class_head(queries))

        return all_cls_logits, all_mask_logits


# ─────────────────────────────────────────────────────────────────────────────
# Full configurable model
# ─────────────────────────────────────────────────────────────────────────────

class DefectFormerAblation(nn.Module):
    """
    Configurable DefectFormer for ablation studies.

    All flags default to the full model (DefectFormer paper settings).
    Override individual flags to reproduce each row of Table III.
    """

    def __init__(
        self,
        num_classes:        int  = 15,
        d:                  int  = 256,
        nq:                 int  = 100,
        num_decoder_layers: int  = 6,
        num_heads:          int  = 8,
        backbone:           str  = "swin_tiny",   # "swin_tiny" | "resnet50"
        msfe_variant:       str  = "full",        # "full"|"csag_only"|"additive"|"none"
        use_prior_init:     bool = True,
        use_dmca_mask:      bool = True,
        use_dcsa_repulsion: bool = True,
        pretrained:         bool = True,
    ):
        super().__init__()
        cfg = _BACKBONE_CFGS[backbone]

        # ── Backbone ──────────────────────────────────────────────────
        self.backbone = timm.create_model(
            cfg["timm_name"],
            pretrained=pretrained,
            features_only=True,
            out_indices=cfg["out_indices"],
        )
        self._nhwc = cfg["nhwc"]
        in_channels = self.backbone.feature_info.channels()

        # ── Feature pyramid ───────────────────────────────────────────
        self.msfe = MSFEVariant(in_channels, d, msfe_variant)

        # ── Pixel decoder ─────────────────────────────────────────────
        self.pixel_decoder = PixelDecoder(d)

        # ── Decoder ───────────────────────────────────────────────────
        self.qdid = QDIDVariant(
            d=d, nq=nq, num_layers=num_decoder_layers,
            num_classes=num_classes, num_heads=num_heads,
            use_prior_init=use_prior_init,
            use_dmca_mask=use_dmca_mask,
            use_dcsa_repulsion=use_dcsa_repulsion,
        )

        self.d  = d
        self.nq = nq

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        H, W = x.shape[-2:]

        features = self.backbone(x)

        if self._nhwc:
            features = [
                f.permute(0, 3, 1, 2).contiguous()
                if f.ndim == 4 and f.shape[-1] != f.shape[1] else f
                for f in features
            ]

        enhanced = self.msfe(features)
        E        = self.pixel_decoder(enhanced)
        Hf, Wf   = E.shape[-2:]

        all_cls, all_ml_feat = self.qdid(E)

        all_mask_logits = []
        for ml in all_ml_feat:
            ml_2d = ml.view(ml.shape[0], self.nq, Hf, Wf)
            ml_up = F.interpolate(ml_2d, size=(H, W),
                                  mode="bilinear", align_corners=False)
            all_mask_logits.append(ml_up.view(ml.shape[0], self.nq, H * W))

        return all_cls, all_mask_logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor, conf_threshold: float = 0.5):
        self.eval()
        H, W = x.shape[-2:]
        all_cls, all_ml = self(x)

        cls  = all_cls[-1]                                  # [B, Nq, C+1]
        ml   = all_ml[-1]                                   # [B, Nq, H*W]
        prob = cls.softmax(-1)[..., :-1]
        scores, labels = prob.max(-1)
        masks = torch.sigmoid(ml).view(x.shape[0], self.nq, H, W)

        return [
            {
                "labels": labels[b][scores[b] > conf_threshold],
                "scores": scores[b][scores[b] > conf_threshold],
                "masks":  (masks[b][scores[b] > conf_threshold] > 0.5),
            }
            for b in range(x.shape[0])
        ]

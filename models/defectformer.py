"""
DefectFormer: end-to-end industrial defect instance segmentation.

Pipeline: Swin-T backbone → MSFE → Pixel Decoder → QDID
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    raise ImportError("timm is required: pip install timm") from e

from .msfe import MSFE
from .pixel_decoder import PixelDecoder
from .qdid import QDID


class DefectFormer(nn.Module):
    def __init__(
        self,
        num_classes: int = 15,
        d: int = 256,
        nq: int = 100,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
    ):
        super().__init__()

        # --- Backbone ---
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),   # C2, C3, C4, C5
        )
        in_channels = self.backbone.feature_info.channels()  # [96,192,384,768] for Swin-T

        # --- MSFE ---
        self.msfe = MSFE(in_channels=in_channels, d=d)

        # --- Pixel Decoder ---
        self.pixel_decoder = PixelDecoder(d=d)

        # --- QDID ---
        self.qdid = QDID(
            d=d,
            nq=nq,
            num_layers=num_decoder_layers,
            num_classes=num_classes,
            num_heads=num_heads,
        )

        self.d = d
        self.nq = nq

    def forward(self, x: torch.Tensor):
        """
        x : [B, 3, H, W]
        Returns
          all_cls_logits  : list[L] of [B, Nq, C+1]
          all_mask_logits : list[L] of [B, Nq, H*W]  at *full* resolution
        """
        H, W = x.shape[-2:]

        # 1. Backbone: {F2, F3, F4, F5}
        features = self.backbone(x)
        # timm Swin outputs NHWC (channels-last); convert to NCHW for Conv2d
        features = [
            f.permute(0, 3, 1, 2).contiguous() if f.ndim == 4 and f.shape[-1] != f.shape[1]
            else f
            for f in features
        ]

        # 2. MSFE: {P2, P3, P4, P5}
        enhanced = self.msfe(features)

        # 3. Pixel Decoder → E  [B, d, Hf, Wf]  where Hf=H/8, Wf=W/8
        E = self.pixel_decoder(enhanced)
        Hf, Wf = E.shape[-2:]

        # 4. QDID → per-layer outputs
        all_cls_logits, all_mask_logits_feat = self.qdid(E)

        # Upsample mask logits from feature resolution to full image resolution
        all_mask_logits = []
        for ml in all_mask_logits_feat:
            ml_2d = ml.view(ml.shape[0], self.nq, Hf, Wf)        # [B, Nq, Hf, Wf]
            ml_up = F.interpolate(ml_2d, size=(H, W),
                                  mode='bilinear', align_corners=False)
            all_mask_logits.append(ml_up.view(ml.shape[0], self.nq, H * W))

        return all_cls_logits, all_mask_logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor, conf_threshold: float = 0.5):
        """
        Inference helper.
        Returns list of dicts (one per image):
          {'labels': LongTensor, 'scores': FloatTensor, 'masks': BoolTensor [N,H,W]}
        """
        self.eval()
        H, W = x.shape[-2:]
        all_cls_logits, all_mask_logits = self(x)

        cls_logits = all_cls_logits[-1]    # [B, Nq, C+1]
        mask_logits = all_mask_logits[-1]  # [B, Nq, H*W]

        probs = cls_logits.softmax(-1)[..., :-1]          # exclude no-object
        scores, labels = probs.max(-1)                     # [B, Nq]

        masks_2d = torch.sigmoid(mask_logits).view(
            x.shape[0], self.nq, H, W
        )

        results = []
        for b in range(x.shape[0]):
            keep = scores[b] > conf_threshold
            results.append({
                "labels": labels[b][keep],
                "scores": scores[b][keep],
                "masks":  (masks_2d[b][keep] > 0.5),
            })
        return results

"""
DRAEM-inspired synthetic anomaly augmentation.

With probability p_apply, pastes a color-distorted region onto the image
and records it as an additional GT instance, matching the augmentation
strategy described in §III-E of the paper.

Usage:
    aug = SyntheticAnomalyAug(p_apply=0.5)
    img_tensor, target = aug(img_tensor, target)
"""

import random
import math

import numpy as np
import torch
from PIL import Image, ImageDraw


class SyntheticAnomalyAug:
    """
    Synthetic anomaly paste augmentation.

    Generates a random filled shape (ellipse / rectangle / polygon),
    applies strong per-channel color distortion inside that region,
    and appends the region mask to the GT instance set.
    """

    def __init__(
        self,
        p_apply:    float = 0.5,
        min_scale:  float = 0.02,   # min fraction of image area
        max_scale:  float = 0.15,   # max fraction of image area
        min_pixels: int   = 16,
    ):
        self.p_apply    = p_apply
        self.min_scale  = min_scale
        self.max_scale  = max_scale
        self.min_pixels = min_pixels

    # ------------------------------------------------------------------

    def _random_mask(self, H: int, W: int) -> np.ndarray:
        """Return a (H, W) bool array with a filled random shape."""
        area_frac = random.uniform(self.min_scale, self.max_scale)
        h = max(4, int(math.sqrt(area_frac * H * W * random.uniform(0.5, 2.0))))
        w = max(4, int(area_frac * H * W / max(1, h)))
        h, w = min(h, H - 1), min(w, W - 1)

        y0 = random.randint(0, H - h - 1)
        x0 = random.randint(0, W - w - 1)

        canvas = Image.new("L", (W, H), 0)
        draw   = ImageDraw.Draw(canvas)
        shape  = random.choice(["ellipse", "rectangle", "polygon"])

        if shape == "polygon":
            # Random convex-ish polygon
            cx, cy = x0 + w // 2, y0 + h // 2
            n_pts  = random.randint(5, 9)
            pts = [
                (
                    int(cx + (w // 2) * math.cos(2 * math.pi * i / n_pts)
                              * random.uniform(0.5, 1.0)),
                    int(cy + (h // 2) * math.sin(2 * math.pi * i / n_pts)
                              * random.uniform(0.5, 1.0)),
                )
                for i in range(n_pts)
            ]
            draw.polygon(pts, fill=255)
        elif shape == "ellipse":
            draw.ellipse([x0, y0, x0 + w, y0 + h], fill=255)
        else:
            draw.rectangle([x0, y0, x0 + w, y0 + h], fill=255)

        return np.asarray(canvas) > 0

    def _distort_region(self, tensor: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        """
        Applies per-channel color distortion inside `mask`.
        Works on normalised tensors — values remain in a valid range.
        """
        aug = tensor.clone()
        # Random per-channel multiplicative + additive shift
        for c in range(tensor.shape[0]):
            scale = random.uniform(0.2, 0.8)
            shift = random.uniform(-0.5, 0.5)
            aug[c][mask] = aug[c][mask] * scale + shift
        aug.clamp_(-3.0, 3.0)
        return aug

    # ------------------------------------------------------------------

    def __call__(
        self,
        image_tensor: torch.Tensor,
        target:       dict,
    ) -> tuple[torch.Tensor, dict]:
        """
        image_tensor : [3, H, W] normalised FloatTensor
        target       : dict with 'labels' [N] and 'masks' [N, H*W]
        """
        if random.random() > self.p_apply:
            return image_tensor, target

        _, H, W = image_tensor.shape
        region = self._random_mask(H, W)

        if region.sum() < self.min_pixels:
            return image_tensor, target

        mask_t = torch.from_numpy(region)                    # [H, W] bool
        image_tensor = self._distort_region(image_tensor, mask_t)

        # Append synthetic instance
        cls_id = int(target["labels"][0]) if len(target["labels"]) > 0 else 0
        new_lbl  = torch.tensor([cls_id], dtype=torch.long)
        new_mask = mask_t.flatten().unsqueeze(0)             # [1, H*W]

        target = {
            "labels": torch.cat([target["labels"], new_lbl]),
            "masks":  torch.cat([target["masks"],  new_mask]),
        }
        return image_tensor, target

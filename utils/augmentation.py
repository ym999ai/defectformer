"""
Advanced data augmentation for DefectFormer training.

CopyPasteAug  - paste GT instances from other training images (COCO copy-paste)
MultiScaleResize - randomly sample training resolution per batch
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Copy-Paste Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class CopyPasteAug:
    """
    Copy-Paste instance augmentation.

    For each training sample, randomly samples k instances from another
    image in the dataset and pastes them onto the current image.

    This is particularly effective for rare defect types that have few
    training examples, as it synthesises new compositions without requiring
    additional labelling.

    Reference: "Simple Copy-Paste is a Strong Data Augmentation Method
                for Instance Segmentation" (Ghiasi et al., CVPR 2021)
    """

    def __init__(self, dataset: Dataset,
                 p_apply:    float = 0.5,
                 max_paste:  int   = 3,
                 blend_alpha: float = 1.0):
        """
        dataset     : the training dataset (accessed to fetch source instances)
        p_apply     : probability of applying augmentation per sample
        max_paste   : max number of instances to paste per image
        blend_alpha : 1.0 = hard paste; <1.0 = alpha-blend (softer boundary)
        """
        self.dataset     = dataset
        self.p_apply     = p_apply
        self.max_paste   = max_paste
        self.blend_alpha = blend_alpha

    def __call__(self, img: torch.Tensor,
                 target: dict) -> tuple[torch.Tensor, dict]:
        """
        img    : [3, H, W] normalised float tensor
        target : {'labels': [N], 'masks': [N, H*W]}
        """
        if random.random() > self.p_apply:
            return img, target

        # ── Pick a random source image from the dataset ───────────
        src_idx = random.randint(0, len(self.dataset) - 1)
        try:
            src_img, src_tgt = self.dataset[src_idx]
        except Exception:
            return img, target

        if len(src_tgt["labels"]) == 0:
            return img, target

        _, H, W = img.shape

        # ── Select random subset of source instances ──────────────
        n_avail = len(src_tgt["labels"])
        n_paste = min(random.randint(1, self.max_paste), n_avail)
        paste_idx = random.sample(range(n_avail), n_paste)

        aug_img    = img.clone()
        new_labels = list(target["labels"].tolist())
        new_masks  = list(target["masks"])

        for pi in paste_idx:
            src_mask = src_tgt["masks"][pi].bool()   # [H*W]
            if src_mask.sum() == 0:
                continue

            src_mask_2d = src_mask.view(H, W)        # [H, W]
            src_img_rs  = src_img                    # already same size

            if self.blend_alpha >= 1.0:
                # Hard paste: replace pixels
                aug_img[:, src_mask_2d] = src_img_rs[:, src_mask_2d]
            else:
                # Soft alpha-blend at mask boundary
                a = self.blend_alpha
                aug_img[:, src_mask_2d] = (
                    a * src_img_rs[:, src_mask_2d]
                    + (1 - a) * aug_img[:, src_mask_2d]
                )

            new_labels.append(int(src_tgt["labels"][pi]))
            new_masks.append(src_mask)

        return aug_img, {
            "labels": torch.tensor(new_labels, dtype=torch.long),
            "masks":  torch.stack(new_masks),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Scale Resize (applied at dataset level, before collation)
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleResize:
    """
    Randomly resizes each training image to one of the given scales.

    Helps the model generalise to different defect sizes and image
    resolutions without additional labelling.

    Usage (wrap dataset):
        ds = MultiScaleWrapper(train_ds, scales=[640, 768, 896, 1024])
    """

    def __init__(self, scales: list[int] = None):
        self.scales = scales or [640, 768, 896, 1024]

    def __call__(self, img: torch.Tensor,
                 target: dict) -> tuple[torch.Tensor, dict]:
        """
        img    : [3, H, W] float tensor
        target : {'labels': [N], 'masks': [N, H*W]}  — masks at original H×W
        """
        _, H, W = img.shape
        new_size = random.choice(self.scales)
        if new_size == H == W:
            return img, target

        # Resize image
        img_rs = F.interpolate(
            img.unsqueeze(0), size=(new_size, new_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        # Resize masks
        N = len(target["labels"])
        if N == 0:
            masks_rs = torch.zeros(0, new_size * new_size, dtype=torch.bool)
        else:
            masks_2d = target["masks"].float().view(N, 1, H, W)
            masks_rs = F.interpolate(
                masks_2d, size=(new_size, new_size), mode="nearest"
            ).view(N, new_size * new_size).bool()

        return img_rs, {"labels": target["labels"], "masks": masks_rs}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrappers
# ─────────────────────────────────────────────────────────────────────────────

class AugmentedDataset(Dataset):
    """
    Applies a chain of augmentation transforms to a base dataset.

    Each transform must accept (img_tensor, target) and return
    (img_tensor, target).
    """

    def __init__(self, base: Dataset, transforms: list):
        self.base       = base
        self.transforms = transforms

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        img, target = self.base[idx]
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

    def __getattr__(self, name):
        return getattr(self.base, name)

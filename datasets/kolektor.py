"""
KolektorSDD2 Dataset for DefectFormer instance segmentation.

Expected layout (official split):
  <root>/
    train/
      *.jpg  (or .png)  — defect + defect-free
      *_GT.png          — binary ground-truth masks (0=bg, 255=defect)
    test/
      *.jpg
      *_GT.png

Defect-free images (all-zero GT mask) are included as negatives
(0 instances).  Instance annotations are derived via connected
components; components < MIN_PIXELS pixels are discarded.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset
import torchvision.transforms as T


MIN_PIXELS = 16
DEFECT_CLASS = 0   # single defect class: crack/scratch


def _connected_instances(mask_arr: np.ndarray):
    binary = (mask_arr > 0).astype(np.uint8)
    labeled, n = ndimage.label(binary)
    return [
        (DEFECT_CLASS, (labeled == i).astype(np.uint8))
        for i in range(1, n + 1)
        if (labeled == i).sum() >= MIN_PIXELS
    ]


class KolektorSDD2Dataset(Dataset):
    """KolektorSDD2 for DefectFormer, following the official train/test split."""

    NUM_CLASSES = 1   # single defect class

    def __init__(self, root: str, split: str = "train", img_size: int = 1024):
        assert split in ("train", "test")
        self.root = Path(root)
        self.split = split
        self.img_size = img_size

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        self.img_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

        self.samples = self._build_index()

    def _build_index(self) -> list[dict]:
        split_dir = self.root / self.split
        samples = []

        for img_fp in sorted(split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.png")):
            if "_GT" in img_fp.stem:
                continue
            gt_fp = split_dir / (img_fp.stem + "_GT.png")
            if gt_fp.exists():
                samples.append({"image": str(img_fp), "mask": str(gt_fp)})

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s["image"]).convert("RGB")
        mask_arr = np.array(Image.open(s["mask"]).convert("L"))

        img_tensor = self.img_tf(img)
        S = self.img_size
        instances = _connected_instances(mask_arr)

        inst_masks = []
        inst_labels = []

        for cls_id, inst_m in instances:
            pil_m = Image.fromarray(inst_m * 255)
            pil_m = pil_m.resize((S, S), Image.NEAREST)
            t = torch.tensor(np.array(pil_m) > 0, dtype=torch.bool).flatten()
            inst_masks.append(t)
            inst_labels.append(cls_id)

        if inst_masks:
            target = {
                "labels": torch.tensor(inst_labels, dtype=torch.long),
                "masks":  torch.stack(inst_masks, 0),
            }
        else:
            target = {
                "labels": torch.zeros(0, dtype=torch.long),
                "masks":  torch.zeros(0, S * S, dtype=torch.bool),
            }

        return img_tensor, target

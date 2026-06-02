"""
MVTec AD Dataset for DefectFormer instance segmentation.

Directory layout expected:
  <root>/
    bottle/
      train/good/*.png
      test/good/*.png
      test/<defect_type>/*.png
      ground_truth/<defect_type>/*_mask.png
    cable/
      ...

Instance annotations are derived from connected-component analysis on
the provided pixel-level masks; components < MIN_PIXELS are discarded.

Split: anomalous test images are partitioned 60/20/20 per category.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


MIN_PIXELS = 16  # filter noise artifacts

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def _connected_instances(mask_arr: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """
    mask_arr : H×W uint8 binary mask (values 0 or 255)
    Returns [(class_id=0, binary_mask), ...]  — class_id is always 0 here;
    the caller maps it to the dataset class index.
    """
    binary = (mask_arr > 0).astype(np.uint8)
    labeled, n = ndimage.label(binary)
    instances = []
    for inst_id in range(1, n + 1):
        m = (labeled == inst_id).astype(np.uint8)
        if m.sum() >= MIN_PIXELS:
            instances.append((0, m))
    return instances


class MVTecDataset(Dataset):
    """MVTec AD — per-instance masks for DefectFormer."""

    def __init__(self, root: str, split: str = "train",
                 img_size: int = 1024, seed: int = 42):
        assert split in ("train", "val", "test")
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.seed = seed

        self.class_to_idx: dict[str, int] = {}
        self.samples: list[dict] = []
        self._build_index()

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        self.img_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    # ------------------------------------------------------------------

    def _build_index(self):
        rng = random.Random(self.seed)

        for category in MVTEC_CATEGORIES:
            cat_path = self.root / category
            test_path = cat_path / "test"
            if not test_path.exists():
                continue

            defect_types = sorted(
                d.name for d in test_path.iterdir()
                if d.is_dir() and d.name != "good"
            )

            for dt in defect_types:
                key = f"{category}/{dt}"
                if key not in self.class_to_idx:
                    self.class_to_idx[key] = len(self.class_to_idx)

            category_samples = []
            for dt in defect_types:
                img_dir  = test_path / dt
                gt_dir   = cat_path / "ground_truth" / dt
                cls_key  = f"{category}/{dt}"

                for img_fp in sorted(img_dir.glob("*.png")):
                    mask_fp = gt_dir / (img_fp.stem + "_mask.png")
                    if mask_fp.exists():
                        category_samples.append({
                            "image":   str(img_fp),
                            "mask":    str(mask_fp),
                            "cls_key": cls_key,
                        })

            # Shuffle then split 60 / 20 / 20
            rng.shuffle(category_samples)
            n = len(category_samples)
            n_train = int(0.6 * n)
            n_val   = int(0.2 * n)

            if self.split == "train":
                self.samples.extend(category_samples[:n_train])
            elif self.split == "val":
                self.samples.extend(category_samples[n_train: n_train + n_val])
            else:
                self.samples.extend(category_samples[n_train + n_val:])

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s["image"]).convert("RGB")
        mask_arr = np.array(Image.open(s["mask"]).convert("L"))

        img_tensor = self.img_tf(img)                              # [3, H, W]
        S = self.img_size
        cls_idx = self.class_to_idx[s["cls_key"]]

        instances = _connected_instances(mask_arr)
        inst_masks = []
        inst_labels = []

        for _, inst_m in instances:
            pil_m = Image.fromarray(inst_m * 255)
            pil_m = pil_m.resize((S, S), Image.NEAREST)
            t = torch.tensor(np.array(pil_m) > 0, dtype=torch.bool).flatten()
            inst_masks.append(t)
            inst_labels.append(cls_idx)

        if inst_masks:
            target = {
                "labels": torch.tensor(inst_labels, dtype=torch.long),
                "masks":  torch.stack(inst_masks, 0),               # [N, S*S]
            }
        else:
            target = {
                "labels": torch.zeros(0, dtype=torch.long),
                "masks":  torch.zeros(0, S * S, dtype=torch.bool),
            }

        return img_tensor, target

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

"""
Convert MVTec AD / KolektorSDD2 to COCO instance-segmentation JSON.

Needed by SOLOv2 / CondInst / Mask2Former (mmdetection) baselines.

Usage
-----
python baselines/convert_to_coco.py \\
    --dataset mvtec \\
    --data_root /data/mvtec_anomaly_detection \\
    --output_dir ./coco_data/mvtec

python baselines/convert_to_coco.py \\
    --dataset kolektor \\
    --data_root /data/kolektorsdd2 \\
    --output_dir ./coco_data/kolektor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.mvtec import MVTecDataset, _connected_instances
from datasets.kolektor import KolektorSDD2Dataset, _connected_instances as _kol_instances

# ── Polygon extraction (cv2 preferred, scipy fallback) ────────────────────────

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    """Binary H×W mask → list of COCO polygon point lists."""
    if _HAS_CV2:
        import cv2
        cnts, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_KCOS,
        )
        polys = []
        for c in cnts:
            flat = c.flatten().tolist()
            if len(flat) >= 6:          # ≥ 3 points
                polys.append(flat)
        return polys
    else:
        # Fallback: use bounding-box rectangle as degenerate polygon
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return []
        r_idx = np.where(rows)[0]; r0, r1 = int(r_idx[0]),  int(r_idx[-1])
        c_idx = np.where(cols)[0]; c0, c1 = int(c_idx[0]),  int(c_idx[-1])
        return [[float(c0), float(r0), float(c1), float(r0),
                 float(c1), float(r1), float(c0), float(r1)]]


# ── COCO builder ─────────────────────────────────────────────────────────────

def _build_coco_from_samples(samples: list[dict],
                              class_to_idx: dict[str, int],
                              get_cls_idx) -> dict:
    """
    samples        : list of {'image': path, 'mask': path, ...}
    class_to_idx   : maps class name → 0-based int
    get_cls_idx(s) : callable(sample) → 0-based class int
    """
    idx_to_name = {v: k for k, v in class_to_idx.items()}
    categories  = [{"id": i + 1, "name": idx_to_name[i]}
                   for i in range(len(class_to_idx))]

    coco      = {"images": [], "annotations": [], "categories": categories}
    ann_id    = 1

    for img_id, s in enumerate(samples, 1):
        img      = Image.open(s["image"])
        W, H     = img.size
        coco["images"].append({
            "id":        img_id,
            "file_name": s["image"],
            "width":     W,
            "height":    H,
        })

        mask_arr  = np.array(Image.open(s["mask"]).convert("L"))
        instances = _connected_instances(mask_arr)
        cls_idx   = get_cls_idx(s)              # 0-based

        for _, inst_m in instances:
            polys = _mask_to_polygons(inst_m)
            if not polys:
                continue
            rows  = np.any(inst_m, axis=1)
            cols  = np.any(inst_m, axis=0)
            r_idx = np.where(rows)[0]; r0, r1 = int(r_idx[0]),  int(r_idx[-1])
            c_idx = np.where(cols)[0]; c0, c1 = int(c_idx[0]),  int(c_idx[-1])

            coco["annotations"].append({
                "id":           ann_id,
                "image_id":     img_id,
                "category_id":  cls_idx + 1,    # 1-based
                "segmentation": polys,
                "area":         float(inst_m.sum()),
                "bbox":         [float(c0), float(r0),
                                 float(c1 - c0 + 1), float(r1 - r0 + 1)],
                "iscrowd":      0,
            })
            ann_id += 1

    return coco


def convert_mvtec(data_root: str, output_dir: str, img_size: int = 1024):
    """Convert all three MVTec AD splits to COCO JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    # Build datasets for all splits to collect samples
    for split in ("train", "val", "test"):
        ds = MVTecDataset(data_root, split=split, img_size=img_size)

        def _get_cls(s):
            return ds.class_to_idx[s["cls_key"]]

        coco = _build_coco_from_samples(ds.samples, ds.class_to_idx, _get_cls)
        out  = os.path.join(output_dir, f"{split}.json")
        with open(out, "w") as f:
            json.dump(coco, f)

        n_img = len(coco["images"])
        n_ann = len(coco["annotations"])
        print(f"  [{split}]  images={n_img}  annotations={n_ann}  → {out}")

    # Save category info separately for reference
    ds0 = MVTecDataset(data_root, split="train", img_size=img_size)
    meta = {"class_to_idx": ds0.class_to_idx,
            "num_classes":  ds0.num_classes}
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  num_classes={ds0.num_classes}")
    return ds0.num_classes


def convert_kolektor(data_root: str, output_dir: str, img_size: int = 1024):
    """Convert KolektorSDD2 train/test splits to COCO JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    class_to_idx = {"defect": 0}

    for split in ("train", "test"):
        ds = KolektorSDD2Dataset(data_root, split=split, img_size=img_size)

        def _get_cls(s):
            return 0

        coco = _build_coco_from_samples(ds.samples, class_to_idx, _get_cls)
        out  = os.path.join(output_dir, f"{split}.json")
        with open(out, "w") as f:
            json.dump(coco, f)

        n_img = len(coco["images"])
        n_ann = len(coco["annotations"])
        print(f"  [{split}]  images={n_img}  annotations={n_ann}  → {out}")

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump({"class_to_idx": class_to_idx, "num_classes": 1}, f)

    return 1


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    required=True, choices=["mvtec", "kolektor"])
    p.add_argument("--data_root",  required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--img_size",   type=int, default=1024)
    args = p.parse_args()

    print(f"Converting {args.dataset} → COCO JSON  (img_size={args.img_size})")
    if args.dataset == "mvtec":
        convert_mvtec(args.data_root, args.output_dir, args.img_size)
    else:
        convert_kolektor(args.data_root, args.output_dir, args.img_size)
    print("Done.")


if __name__ == "__main__":
    main()

"""
Shared utilities for SOLOv2 / CondInst / Mask2Former training scripts.

Provides:
  - ensure_coco_data()   : auto-runs convert_to_coco if JSON not found
  - write_cfg_file()     : writes a generated mmdet config .py to disk
  - run_mmdet_train()    : launches training via mmengine Runner
  - mmdet_evaluate()     : runs inference + computes our AP/mIoU metrics
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_coco_data(dataset: str, data_root: str,
                     coco_dir: str, img_size: int = 1024) -> dict:
    """
    Auto-convert to COCO JSON if not already done.
    Returns meta dict  {num_classes, class_to_idx, classes_tuple}.
    """
    meta_file = os.path.join(coco_dir, "meta.json")
    if not os.path.exists(meta_file):
        print(f"COCO data not found in {coco_dir} — converting now...")
        from baselines.convert_to_coco import convert_mvtec, convert_kolektor
        if dataset == "mvtec":
            convert_mvtec(data_root, coco_dir, img_size)
        else:
            convert_kolektor(data_root, coco_dir, img_size)

    with open(meta_file) as f:
        meta = json.load(f)

    meta["classes_tuple"] = tuple(
        k for k, _ in sorted(meta["class_to_idx"].items(),
                              key=lambda x: x[1])
    )
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Config writer
# ─────────────────────────────────────────────────────────────────────────────

def write_cfg_file(cfg_str: str, path: str) -> str:
    """Write a config string to a .py file and return its path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(cfg_str)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# mmdet training runner
# ─────────────────────────────────────────────────────────────────────────────

def run_mmdet_train(cfg_file: str):
    """Launch mmdet training via mmengine Runner."""
    try:
        from mmengine.config import Config
        from mmdet.utils import register_all_modules
        from mmengine.runner import Runner
    except ImportError as e:
        raise ImportError(
            "mmdetection not found.  Install with:\n"
            "  pip install -U openmim\n"
            "  mim install mmengine mmcv mmdet"
        ) from e

    register_all_modules()
    cfg    = Config.fromfile(cfg_file)
    runner = Runner.from_cfg(cfg)
    runner.train()


# ─────────────────────────────────────────────────────────────────────────────
# mmdet evaluation (our AP/mIoU metrics)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch

_IOU_THRS = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05).tolist()]


def _mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / (union + 1e-6)


def _ap_at_thr(pred_masks, pred_scores, gt_masks, thr):
    if len(gt_masks) == 0:
        return float(len(pred_masks) == 0)
    if len(pred_masks) == 0:
        return 0.0
    order      = torch.argsort(pred_scores, descending=True)
    pred_masks = pred_masks[order]
    tp, fp     = torch.zeros(len(pred_masks)), torch.zeros(len(pred_masks))
    matched    = torch.zeros(len(gt_masks), dtype=torch.bool)
    for pi, pm in enumerate(pred_masks):
        best_iou, best_gi = 0.0, -1
        for gi, gm in enumerate(gt_masks):
            if matched[gi]:
                continue
            iou = _mask_iou(pm, gm)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= thr and best_gi >= 0:
            tp[pi] = 1
            matched[best_gi] = True
        else:
            fp[pi] = 1
    cum_tp = torch.cumsum(tp, 0)
    cum_fp = torch.cumsum(fp, 0)
    prec   = cum_tp / (cum_tp + cum_fp + 1e-6)
    rec    = cum_tp / len(gt_masks)
    return float(torch.trapz(prec, rec))


def mmdet_evaluate(model_cfg: str, checkpoint: str,
                   test_ann: str, device: str = "cuda",
                   conf_thr: float = 0.3) -> dict:
    """
    Run mmdet inference on test_ann and compute AP / mIoU with our metrics.
    """
    try:
        from mmdet.apis import init_detector, inference_detector
        import mmcv
    except ImportError as e:
        raise ImportError("mmdetection required for evaluation") from e

    detector = init_detector(model_cfg, checkpoint, device=device)

    with open(test_ann) as f:
        coco = json.load(f)

    img_id_to_gt: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_id_to_gt.setdefault(ann["image_id"], []).append(ann)

    aps   = {t: [] for t in _IOU_THRS}
    mious = []

    for img_info in coco["images"]:
        img_id   = img_info["id"]
        img_path = img_info["file_name"]
        H, W     = img_info["height"], img_info["width"]

        result = inference_detector(detector, img_path)

        # Extract masks and scores from mmdet result
        if hasattr(result, "pred_instances"):
            pi   = result.pred_instances
            masks  = (pi.masks.cpu() > 0.5) if hasattr(pi, "masks") else torch.zeros(0, H, W, dtype=torch.bool)
            scores = pi.scores.cpu() if hasattr(pi, "scores") else torch.zeros(0)
        else:
            masks, scores = torch.zeros(0, H, W, dtype=torch.bool), torch.zeros(0)

        keep      = scores > conf_thr
        pred_masks = masks[keep].view(-1, H * W)
        pred_scores = scores[keep]

        # GT masks from COCO JSON
        gt_anns = img_id_to_gt.get(img_id, [])
        gt_masks_list = []
        for ann in gt_anns:
            # Decode polygon to mask
            from pycocotools import mask as coco_mask_util
            if isinstance(ann["segmentation"], list):
                rles = coco_mask_util.frPyObjects(ann["segmentation"], H, W)
                rle  = coco_mask_util.merge(rles)
            else:
                rle = ann["segmentation"]
            m = coco_mask_util.decode(rle).astype(bool)
            gt_masks_list.append(torch.from_numpy(m).flatten())

        gt_masks = torch.stack(gt_masks_list) if gt_masks_list \
                   else torch.zeros(0, H * W, dtype=torch.bool)

        for thr in _IOU_THRS:
            aps[thr].append(_ap_at_thr(pred_masks, pred_scores, gt_masks, thr))

        if len(gt_masks) == 0 and len(pred_masks) == 0:
            mious.append(1.0)
        elif len(gt_masks) == 0 or len(pred_masks) == 0:
            mious.append(0.0)
        else:
            matched = torch.zeros(len(pred_masks), dtype=torch.bool)
            row_ious = []
            for gm in gt_masks:
                best, bi = 0.0, -1
                for pi2, pm in enumerate(pred_masks):
                    if matched[pi2]:
                        continue
                    iou = _mask_iou(pm, gm)
                    if iou > best:
                        best, bi = iou, pi2
                if best >= 0.5 and bi >= 0:
                    matched[bi] = True
                row_ious.append(best)
            mious.append(float(np.mean(row_ious)))

    return {
        "AP":   float(np.mean([np.mean(v) for v in aps.values()])),
        "AP50": float(np.mean(aps[0.50])),
        "AP75": float(np.mean(aps[0.75])),
        "mIoU": float(np.mean(mious)),
    }

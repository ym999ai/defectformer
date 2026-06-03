"""
Unified evaluation script for all 4 instance-segmentation baselines.

Loads each trained model, runs inference on the test set, and reports
AP / AP50 / AP75 / mIoU using the same metric code as DefectFormer eval.py.

Usage
-----
# Evaluate all baselines at once
python baselines/eval_baselines.py \\
    --data_root /data/mvtec_anomaly_detection \\
    --maskrcnn_ckpt  ./runs/maskrcnn/best.pth \\
    --solov2_ckpt    ./runs/solov2/best.pth \\
    --solov2_cfg     ./runs/solov2/solov2_config.py \\
    --condinst_ckpt  ./runs/condinst/best.pth \\
    --condinst_cfg   ./runs/condinst/condinst_config.py \\
    --m2f_ckpt       ./runs/mask2former/best.pth \\
    --m2f_cfg        ./runs/mask2former/mask2former_config.py \\
    --coco_dir       ./coco_data/mvtec

# Evaluate Mask R-CNN only
python baselines/eval_baselines.py \\
    --data_root /data/mvtec_anomaly_detection \\
    --maskrcnn_ckpt ./runs/maskrcnn/best.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.mvtec    import MVTecDataset
from datasets.kolektor import KolektorSDD2Dataset
from baselines._mmdet_base import _IOU_THRS, _mask_iou, _ap_at_thr


# ─────────────────────────────────────────────────────────────────────────────
# Shared evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(pred_iter) -> dict:
    """
    pred_iter : yields (pred_masks [N, H*W] bool, pred_scores [N],
                        gt_masks   [M, H*W] bool)
    """
    aps   = {t: [] for t in _IOU_THRS}
    mious = []

    for pred_masks, pred_scores, gt_masks in pred_iter:
        for thr in _IOU_THRS:
            aps[thr].append(_ap_at_thr(pred_masks, pred_scores, gt_masks, thr))

        if len(gt_masks) == 0 and len(pred_masks) == 0:
            mious.append(1.0)
        elif len(gt_masks) == 0 or len(pred_masks) == 0:
            mious.append(0.0)
        else:
            matched  = torch.zeros(len(pred_masks), dtype=torch.bool)
            row_ious = []
            for gm in gt_masks:
                best, bi = 0.0, -1
                for pi, pm in enumerate(pred_masks):
                    if matched[pi]:
                        continue
                    iou = _mask_iou(pm, gm)
                    if iou > best:
                        best, bi = iou, pi
                if best >= 0.5 and bi >= 0:
                    matched[bi] = True
                row_ious.append(best)
            mious.append(float(np.mean(row_ious)))

    return {
        "AP":   round(float(np.mean([np.mean(v) for v in aps.values()])) * 100, 1),
        "AP50": round(float(np.mean(aps[0.50])) * 100, 1),
        "AP75": round(float(np.mean(aps[0.75])) * 100, 1),
        "mIoU": round(float(np.mean(mious))     * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mask R-CNN evaluator
# ─────────────────────────────────────────────────────────────────────────────

def eval_maskrcnn(ckpt: str, data_root: str, dataset: str,
                  img_size: int, batch_size: int,
                  conf_thr: float, device: torch.device) -> dict:
    from baselines.train_maskrcnn import (
        _MaskRCNNAdapter, _collate, build_model,
    )

    if dataset == "mvtec":
        base        = MVTecDataset(data_root, split="test", img_size=img_size)
        num_classes = base.num_classes
    else:
        base        = KolektorSDD2Dataset(data_root, split="test", img_size=img_size)
        num_classes = KolektorSDD2Dataset.NUM_CLASSES

    test_ds     = _MaskRCNNAdapter(base)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, collate_fn=_collate, num_workers=2)

    model = build_model(num_classes).to(device)
    ckpt_data = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt_data["model"] if "model" in ckpt_data else ckpt_data)
    model.eval()

    def _iter():
        with torch.no_grad():
            for images, targets in test_loader:
                images = [img.to(device) for img in images]
                preds  = model(images)
                for pred, tgt in zip(preds, targets):
                    H, W   = images[0].shape[-2:]
                    keep   = pred["scores"] > conf_thr
                    pm     = (pred["masks"][keep, 0] > 0.5).cpu().view(-1, H * W)
                    ps     = pred["scores"][keep].cpu()
                    gm     = tgt["masks"].bool().cpu().view(-1, H * W)
                    yield pm, ps, gm

    return _compute_metrics(_iter())


# ─────────────────────────────────────────────────────────────────────────────
# mmdet evaluator (SOLOv2, CondInst, Mask2Former)
# ─────────────────────────────────────────────────────────────────────────────

def eval_mmdet_model(cfg_file: str, ckpt: str, test_ann: str,
                     conf_thr: float, device_str: str) -> dict:
    try:
        from mmdet.apis import init_detector, inference_detector
    except ImportError as e:
        raise ImportError("mmdetection required") from e

    with open(test_ann) as f:
        coco = json.load(f)

    img_id_to_gt: dict = {}
    for ann in coco["annotations"]:
        img_id_to_gt.setdefault(ann["image_id"], []).append(ann)

    detector = init_detector(cfg_file, ckpt, device=device_str)

    def _iter():
        from pycocotools import mask as cmu
        for img_info in coco["images"]:
            H, W     = img_info["height"], img_info["width"]
            result   = inference_detector(detector, img_info["file_name"])
            pi       = result.pred_instances
            masks    = (pi.masks.cpu() > 0.5).view(-1, H * W) \
                       if hasattr(pi, "masks") else torch.zeros(0, H * W, dtype=torch.bool)
            scores   = pi.scores.cpu() if hasattr(pi, "scores") else torch.zeros(0)
            keep     = scores > conf_thr
            pred_m   = masks[keep]
            pred_s   = scores[keep]

            gt_anns  = img_id_to_gt.get(img_info["id"], [])
            gt_list  = []
            for ann in gt_anns:
                if isinstance(ann["segmentation"], list):
                    rles = cmu.frPyObjects(ann["segmentation"], H, W)
                    rle  = cmu.merge(rles)
                else:
                    rle  = ann["segmentation"]
                gt_list.append(torch.from_numpy(cmu.decode(rle).astype(bool)).flatten())
            gt_m = torch.stack(gt_list) if gt_list \
                   else torch.zeros(0, H * W, dtype=torch.bool)

            yield pred_m, pred_s, gt_m

    return _compute_metrics(_iter())


# ─────────────────────────────────────────────────────────────────────────────
# Print table
# ─────────────────────────────────────────────────────────────────────────────

def _print_table(results: dict):
    paper = {
        "Mask R-CNN":   {"AP": 29.4, "AP50": 51.2, "AP75": 28.7, "mIoU": 42.3},
        "SOLOv2":       {"AP": 32.1, "AP50": 54.8, "AP75": 31.5, "mIoU": 44.7},
        "CondInst":     {"AP": 35.6, "AP50": 58.3, "AP75": 34.8, "mIoU": 47.1},
        "Mask2Former":  {"AP": 41.3, "AP50": 63.7, "AP75": 40.8, "mIoU": 53.2},
        "DefectFormer": {"AP": 48.6, "AP50": 71.4, "AP75": 47.9, "mIoU": 59.8},
    }
    print("\n" + "=" * 75)
    print(f"  {'Method':<16}  {'AP':>6}  {'AP50':>6}  {'AP75':>6}  {'mIoU':>6}  "
          f"{'PaperAP':>8}")
    print("  " + "-" * 71)
    for name, m in results.items():
        pa = paper.get(name, {}).get("AP", "---")
        pa = f"{pa:.1f}" if isinstance(pa, float) else pa
        print(f"  {name:<16}  {m['AP']:6.1f}  {m['AP50']:6.1f}  "
              f"{m['AP75']:6.1f}  {m['mIoU']:6.1f}  {pa:>8}")
    print("=" * 75)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      default="mvtec", choices=["mvtec", "kolektor"])
    p.add_argument("--data_root",    default=None)
    p.add_argument("--coco_dir",     default=None,
                   help="COCO JSON directory (for mmdet models)")
    p.add_argument("--img_size",     type=int,   default=1024)
    p.add_argument("--batch_size",   type=int,   default=2)
    p.add_argument("--conf_thr",     type=float, default=0.3)

    # Per-model checkpoints + configs
    p.add_argument("--maskrcnn_ckpt",  default=None)
    p.add_argument("--solov2_ckpt",    default=None)
    p.add_argument("--solov2_cfg",     default=None)
    p.add_argument("--condinst_ckpt",  default=None)
    p.add_argument("--condinst_cfg",   default=None)
    p.add_argument("--m2f_ckpt",       default=None)
    p.add_argument("--m2f_cfg",        default=None)
    p.add_argument("--output",         default="./baselines_results.json")
    args = p.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    results    = {}

    # ── Mask R-CNN ────────────────────────────────────────────
    if args.maskrcnn_ckpt and args.data_root:
        print("\n[Mask R-CNN] evaluating...")
        results["Mask R-CNN"] = eval_maskrcnn(
            args.maskrcnn_ckpt, args.data_root, args.dataset,
            args.img_size, args.batch_size, args.conf_thr, device,
        )
        print(f"  → {results['Mask R-CNN']}")

    # ── SOLOv2 ───────────────────────────────────────────────
    if args.solov2_ckpt and args.solov2_cfg and args.coco_dir:
        print("\n[SOLOv2] evaluating...")
        test_ann = os.path.join(args.coco_dir, "test.json")
        results["SOLOv2"] = eval_mmdet_model(
            args.solov2_cfg, args.solov2_ckpt, test_ann,
            args.conf_thr, device_str,
        )
        print(f"  → {results['SOLOv2']}")

    # ── CondInst ─────────────────────────────────────────────
    if args.condinst_ckpt and args.condinst_cfg and args.coco_dir:
        print("\n[CondInst] evaluating...")
        test_ann = os.path.join(args.coco_dir, "test.json")
        results["CondInst"] = eval_mmdet_model(
            args.condinst_cfg, args.condinst_ckpt, test_ann,
            args.conf_thr, device_str,
        )
        print(f"  → {results['CondInst']}")

    # ── Mask2Former ──────────────────────────────────────────
    if args.m2f_ckpt and args.m2f_cfg and args.coco_dir:
        print("\n[Mask2Former] evaluating...")
        test_ann = os.path.join(args.coco_dir, "test.json")
        results["Mask2Former"] = eval_mmdet_model(
            args.m2f_cfg, args.m2f_ckpt, test_ann,
            args.conf_thr, device_str,
        )
        print(f"  → {results['Mask2Former']}")

    if results:
        _print_table(results)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {args.output}")
    else:
        print("No checkpoints provided. Pass --maskrcnn_ckpt / --solov2_ckpt etc.")


if __name__ == "__main__":
    main()

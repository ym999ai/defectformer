"""
Mask R-CNN baseline — torchvision, self-contained (no mmdetection needed).

Architecture : ResNet-50-FPN  (same as Table II / Table III of the paper)
Loss         : torchvision's built-in classification + bbox + mask losses
Optimizer    : AdamW, backbone lr = 0.1 × head lr  (mirrors DefectFormer)
Schedule     : linear warmup (500 steps) + cosine decay

Usage
-----
python baselines/train_maskrcnn.py \\
    --data_root /data/mvtec_anomaly_detection \\
    --dataset   mvtec \\
    --output_dir ./runs/maskrcnn_mvtec \\
    --epochs 150
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.mvtec   import MVTecDataset
from datasets.kolektor import KolektorSDD2Dataset

try:
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn   import MaskRCNNPredictor
    _WEIGHTS = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
except ImportError:
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    _WEIGHTS = None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset adapter
# ─────────────────────────────────────────────────────────────────────────────

class _MaskRCNNAdapter(Dataset):
    """
    Wraps MVTecDataset / KolektorSDD2Dataset and returns torchvision-compatible
    targets:  boxes [N,4], labels [N], masks [N,H,W], area [N], iscrowd [N].

    Images are kept in normalized float form.  The model's internal transform
    is patched to identity-normalization so it receives our ImageNet-normed
    tensors unchanged.
    """

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        img, tgt = self.base[idx]                    # [3,H,W], target dict
        _, H, W  = img.shape

        masks_flat = tgt["masks"].bool()             # [N, H*W]
        labels     = tgt["labels"]                   # [N]

        if len(masks_flat) == 0:
            return img, {
                "boxes":    torch.zeros(0, 4, dtype=torch.float32),
                "labels":   torch.zeros(0, dtype=torch.long),
                "masks":    torch.zeros(0, H, W, dtype=torch.bool),
                "image_id": torch.tensor([idx]),
                "area":     torch.zeros(0, dtype=torch.float32),
                "iscrowd":  torch.zeros(0, dtype=torch.long),
            }

        masks_2d = masks_flat.view(-1, H, W)         # [N, H, W]

        boxes = []
        for m in masks_2d:
            rows = m.any(dim=1)
            cols = m.any(dim=0)
            if not rows.any():
                boxes.append([0.0, 0.0, 1.0, 1.0])
                continue
            r0, r1 = rows.nonzero(as_tuple=False)[[0, -1], 0]
            c0, c1 = cols.nonzero(as_tuple=False)[[0, -1], 0]
            boxes.append([float(c0), float(r0), float(c1 + 1), float(r1 + 1)])

        boxes  = torch.tensor(boxes, dtype=torch.float32)
        area   = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return img, {
            "boxes":    boxes,
            "labels":   labels + 1,      # 0 = background in torchvision
            "masks":    masks_2d,
            "image_id": torch.tensor([idx]),
            "area":     area,
            "iscrowd":  torch.zeros(len(labels), dtype=torch.long),
        }

    def __getattr__(self, name):
        return getattr(self.base, name)


def _collate(batch):
    return tuple(zip(*batch))


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    try:
        model = maskrcnn_resnet50_fpn_v2(weights=_WEIGHTS)
    except Exception:
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        model = maskrcnn_resnet50_fpn(pretrained=True)

    # Replace prediction heads
    in_feat   = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes + 1)

    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes + 1)

    # Disable internal normalization — our dataset already applies ImageNet stats
    model.transform.image_mean = [0.0, 0.0, 0.0]
    model.transform.image_std  = [1.0, 1.0, 1.0]

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, loader, device, epoch, log_every=50):
    model.train()
    total, n = 0.0, 0

    for step, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss      = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item()
        n     += 1

        if (step + 1) % log_every == 0:
            print(f"    step {step+1:4d}/{len(loader)}  loss={total/n:.4f}")

    return total / max(n, 1)


# ── Metric helpers (mirror eval.py) ──────────────────────────────────────────

_IOU_THRS = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05).tolist()]


def _mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / (union + 1e-6)


def _ap_at_thr(pred_masks, pred_scores, gt_masks, thr: float) -> float:
    if len(gt_masks) == 0:
        return float(len(pred_masks) == 0)
    if len(pred_masks) == 0:
        return 0.0

    order      = torch.argsort(pred_scores, descending=True)
    pred_masks = pred_masks[order]

    tp      = torch.zeros(len(pred_masks))
    fp      = torch.zeros(len(pred_masks))
    matched = torch.zeros(len(gt_masks), dtype=torch.bool)

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


@torch.no_grad()
def evaluate(model, loader, device, conf_thr: float = 0.5) -> dict:
    model.eval()
    aps   = {t: [] for t in _IOU_THRS}
    mious = []

    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds  = model(images)

        for pred, tgt in zip(preds, targets):
            H_orig = images[0].shape[-2]
            W_orig = images[0].shape[-1]

            keep = pred["scores"] > conf_thr
            pred_m = (pred["masks"][keep, 0] > 0.5).cpu().view(-1, H_orig * W_orig)
            pred_s = pred["scores"][keep].cpu()
            gt_m   = tgt["masks"].bool().cpu().view(-1, H_orig * W_orig)

            for thr in _IOU_THRS:
                aps[thr].append(_ap_at_thr(pred_m, pred_s, gt_m, thr))

            # mIoU
            if len(gt_m) == 0 and len(pred_m) == 0:
                mious.append(1.0)
            elif len(gt_m) == 0 or len(pred_m) == 0:
                mious.append(0.0)
            else:
                matched = torch.zeros(len(pred_m), dtype=torch.bool)
                row_ious = []
                for gm in gt_m:
                    best, bi = 0.0, -1
                    for pi, pm in enumerate(pred_m):
                        if matched[pi]:
                            continue
                        iou = _mask_iou(pm, gm)
                        if iou > best:
                            best, bi = iou, pi
                    if best >= 0.5 and bi >= 0:
                        matched[bi] = True
                    row_ious.append(best)
                mious.append(float(np.mean(row_ious)))

    AP   = float(np.mean([np.mean(v) for v in aps.values()]))
    AP50 = float(np.mean(aps[0.50]))
    AP75 = float(np.mean(aps[0.75]))
    mIoU = float(np.mean(mious))
    return {"AP": AP, "AP50": AP50, "AP75": AP75, "mIoU": mIoU}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  required=True)
    p.add_argument("--dataset",    default="mvtec", choices=["mvtec", "kolektor"])
    p.add_argument("--output_dir", default="./runs/maskrcnn")
    p.add_argument("--epochs",     type=int,   default=150)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--img_size",   type=int,   default=1024)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--warmup",     type=int,   default=500,
                   help="Warmup steps (linear, not epochs)")
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--conf_thr",   type=float, default=0.5)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ──────────────────────────────────────────────
    if args.dataset == "mvtec":
        train_base = MVTecDataset(args.data_root, split="train", img_size=args.img_size)
        test_base  = MVTecDataset(args.data_root, split="test",  img_size=args.img_size)
        num_classes = train_base.num_classes
    else:
        train_base  = KolektorSDD2Dataset(args.data_root, split="train", img_size=args.img_size)
        test_base   = KolektorSDD2Dataset(args.data_root, split="test",  img_size=args.img_size)
        num_classes = KolektorSDD2Dataset.NUM_CLASSES

    train_ds = _MaskRCNNAdapter(train_base)
    test_ds  = _MaskRCNNAdapter(test_base)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers,
                              collate_fn=_collate, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=2,
                              shuffle=False, num_workers=args.workers,
                              collate_fn=_collate)

    print(f"num_classes={num_classes}  train={len(train_ds)}  test={len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────
    model = build_model(num_classes).to(device)

    # ── Optimizer (backbone 10× lower LR) ────────────────────
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = AdamW(
        [
            {"params": [p for p in model.parameters() if id(p) in backbone_ids],
             "lr": args.lr * 0.1},
            {"params": [p for p in model.parameters() if id(p) not in backbone_ids],
             "lr": args.lr},
        ],
        weight_decay=1e-4,
    )

    # ── LR schedule: linear warmup → cosine ──────────────────
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ── Training loop ─────────────────────────────────────────
    best_loss  = float("inf")
    best_ckpt  = os.path.join(args.output_dir, "best.pth")
    global_step = 0
    t0          = time.time()

    for epoch in range(args.epochs):
        model.train()
        ep_loss, n = 0.0, 0

        for step, (images, targets) in enumerate(train_loader):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss      = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ep_loss     += loss.item()
            n           += 1
            global_step += 1

            if (step + 1) % 50 == 0:
                print(f"    ep {epoch} step {step+1}/{len(train_loader)} "
                      f"loss={ep_loss/n:.4f}")

        avg_loss = ep_loss / max(n, 1)
        elapsed  = (time.time() - t0) / 60
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"lr={optimizer.param_groups[1]['lr']:.2e}  {elapsed:.1f}min")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "num_classes": num_classes}, best_ckpt)

    # ── Final evaluation ──────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    metrics = evaluate(model, test_loader, device, conf_thr=args.conf_thr)

    total_min = (time.time() - t0) / 60
    print(f"\n{'='*50}")
    print(f"Mask R-CNN  |  AP={metrics['AP']*100:.1f}  "
          f"AP50={metrics['AP50']*100:.1f}  AP75={metrics['AP75']*100:.1f}  "
          f"mIoU={metrics['mIoU']*100:.1f}  ({total_min:.1f}min)")
    print(f"{'='*50}")

    import json
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({k: round(v * 100, 1) for k, v in metrics.items()}, f, indent=2)


if __name__ == "__main__":
    main()

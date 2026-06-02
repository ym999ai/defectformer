"""
DefectFormer evaluation script.

Computes AP (IoU 0.50:0.05:0.95), AP50, AP75 and mIoU.

Example:
  python eval.py \\
      --data_root /data/mvtec_anomaly_detection \\
      --dataset mvtec \\
      --checkpoint ./runs/defectformer_mvtec/checkpoint_e149.pth
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.defectformer import DefectFormer
from datasets.mvtec import MVTecDataset
from datasets.kolektor import KolektorSDD2Dataset


# ---------------------------------------------------------------------------
# Per-image AP helpers
# ---------------------------------------------------------------------------

def _mask_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Binary mask IoU.  pred, gt: [H*W] BoolTensor."""
    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    return inter / (union + 1e-6)


def _compute_ap_at_threshold(
    pred_masks: torch.Tensor,    # [N_pred, H*W] bool
    pred_scores: torch.Tensor,   # [N_pred]
    gt_masks: torch.Tensor,      # [N_gt,   H*W] bool
    iou_thr: float,
) -> float:
    if len(gt_masks) == 0:
        return float(len(pred_masks) == 0)
    if len(pred_masks) == 0:
        return 0.0

    # Sort predictions by confidence (descending)
    order = torch.argsort(pred_scores, descending=True)
    pred_masks = pred_masks[order]

    tp = torch.zeros(len(pred_masks))
    fp = torch.zeros(len(pred_masks))
    matched = torch.zeros(len(gt_masks), dtype=torch.bool)

    for pi, pm in enumerate(pred_masks):
        best_iou, best_gi = 0.0, -1
        for gi, gm in enumerate(gt_masks):
            if matched[gi]:
                continue
            iou = _mask_iou(pm, gm)
            if iou > best_iou:
                best_iou, best_gi = iou, gi

        if best_iou >= iou_thr:
            tp[pi] = 1
            matched[best_gi] = True
        else:
            fp[pi] = 1

    cum_tp = torch.cumsum(tp, 0)
    cum_fp = torch.cumsum(fp, 0)
    prec = cum_tp / (cum_tp + cum_fp + 1e-6)
    rec  = cum_tp / len(gt_masks)

    # Interpolated AP (trapezoidal)
    return float(torch.trapz(prec, rec))


def _compute_miou(
    pred_masks: torch.Tensor,    # [N_pred, H*W] bool
    gt_masks:   torch.Tensor,    # [N_gt,   H*W] bool
    iou_thr: float = 0.5,
) -> float:
    """Mean IoU over matched GT instances at iou_thr."""
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return 1.0
    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return 0.0

    ious = []
    matched = torch.zeros(len(pred_masks), dtype=torch.bool)
    for gm in gt_masks:
        best = 0.0
        for pi, pm in enumerate(pred_masks):
            if matched[pi]:
                continue
            iou = _mask_iou(pm, gm)
            if iou > best:
                best, best_pi = iou, pi
        if best >= iou_thr:
            matched[best_pi] = True
            ious.append(best)
        else:
            ious.append(0.0)
    return float(np.mean(ious))


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

IOU_THRESHOLDS = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05).tolist()]


@torch.no_grad()
def evaluate(model, loader, device, conf_thr=0.3):
    model.eval()
    aps   = {t: [] for t in IOU_THRESHOLDS}
    mious = []

    for images, targets in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        H, W   = images.shape[-2:]
        results = model.predict(images, conf_threshold=conf_thr)

        for result, target in zip(results, targets):
            pred_masks  = result["masks"].cpu().view(-1, H * W)    # [N_pred, H*W]
            pred_scores = result["scores"].cpu()

            n_gt = len(target["labels"])
            gt_masks = target["masks"].bool().cpu()                 # [N_gt, H*W]

            for thr in IOU_THRESHOLDS:
                ap = _compute_ap_at_threshold(pred_masks, pred_scores, gt_masks, thr)
                aps[thr].append(ap)

            mious.append(_compute_miou(pred_masks, gt_masks))

    AP    = float(np.mean([np.mean(v) for v in aps.values()]))
    AP50  = float(np.mean(aps[0.50]))
    AP75  = float(np.mean(aps[0.75]))
    mIoU  = float(np.mean(mious))

    return {"AP": AP, "AP50": AP50, "AP75": AP75, "mIoU": mIoU}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--dataset",    default="mvtec", choices=["mvtec", "kolektor"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img_size",   type=int,   default=1024)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--conf_thr",   type=float, default=0.3)
    parser.add_argument("--workers",    type=int,   default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "mvtec":
        ds = MVTecDataset(args.data_root, split="test", img_size=args.img_size)
        num_classes = ds.num_classes
    else:
        ds = KolektorSDD2Dataset(args.data_root, split="test", img_size=args.img_size)
        num_classes = KolektorSDD2Dataset.NUM_CLASSES

    loader = DataLoader(
        ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                               [x[1] for x in b]),
    )

    model = DefectFormer(
        num_classes=num_classes,
        d=256, nq=100, num_decoder_layers=6, num_heads=8,
        pretrained=False,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)

    metrics = evaluate(model, loader, device, conf_thr=args.conf_thr)
    print(f"AP:   {metrics['AP']  * 100:.1f}")
    print(f"AP50: {metrics['AP50']* 100:.1f}")
    print(f"AP75: {metrics['AP75']* 100:.1f}")
    print(f"mIoU: {metrics['mIoU']* 100:.1f}")


if __name__ == "__main__":
    main()

"""
Automated ablation experiment runner for DefectFormer.

Reproduces all 10 rows of Table III in the paper by sequentially
training and evaluating each configuration.

Usage
-----
# Full run (150 epochs each — use on a cluster)
python ablation/run_ablation.py --data_root /data/mvtec --epochs 150

# Quick sanity-check (30 epochs)
python ablation/run_ablation.py --data_root /data/mvtec --epochs 30 --quick

# Resume interrupted run
python ablation/run_ablation.py --data_root /data/mvtec --resume

# Run a specific subset
python ablation/run_ablation.py --data_root /data/mvtec \\
    --experiments msfe_full,qdid_full,full

# Dry-run: print configs without training
python ablation/run_ablation.py --data_root /data/mvtec --dry_run
"""

from __future__ import annotations

import argparse
import json
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

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.ablation import DefectFormerAblation
from models.losses import DefectFormerLoss
from datasets.mvtec import MVTecDataset
from ablation.configs import ABLATION_EXPERIMENTS, EXPERIMENT_MAP, ExperimentConfig
from ablation.synthetic_aug import SyntheticAnomalyAug


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

class _AugWrapper(Dataset):
    """Wraps a base dataset and applies synthetic augmentation per-item."""

    def __init__(self, base: Dataset, aug: SyntheticAnomalyAug):
        self.base = base
        self.aug  = aug

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, tgt = self.base[idx]
        return self.aug(img, tgt)

    def __getattr__(self, name):
        return getattr(self.base, name)


def _collate(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), list(targets)


def build_loaders(args, use_aug: bool):
    train_ds = MVTecDataset(args.data_root, split="train", img_size=args.img_size)
    test_ds  = MVTecDataset(args.data_root, split="test",  img_size=args.img_size)

    if use_aug:
        train_ds = _AugWrapper(train_ds, SyntheticAnomalyAug(p_apply=0.5))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers,
                              collate_fn=_collate, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              collate_fn=_collate, pin_memory=True)
    return train_loader, test_loader, train_ds.num_classes


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: ExperimentConfig, num_classes: int,
                pretrained: bool = True) -> DefectFormerAblation:
    return DefectFormerAblation(
        num_classes        = num_classes,
        d                  = 256,
        nq                 = 100,
        num_decoder_layers = 6,
        num_heads          = 8,
        backbone           = cfg.backbone,
        msfe_variant       = cfg.msfe_variant,
        use_prior_init     = cfg.use_prior_init,
        use_dmca_mask      = cfg.use_dmca_mask,
        use_dcsa_repulsion = cfg.use_dcsa_repulsion,
        pretrained         = pretrained,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(model: nn.Module, loader: DataLoader,
          criterion: DefectFormerLoss,
          optimizer: torch.optim.Optimizer,
          device: torch.device, epoch: int,
          log_every: int = 50) -> float:
    model.train()
    total, n = 0.0, 0

    for step, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        cls_list, mask_list = model(imgs)
        loss = criterion(cls_list, mask_list, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total += loss.item()
        n     += 1

        if (step + 1) % log_every == 0:
            print(f"    step {step+1:4d}/{len(loader)}  "
                  f"loss={total/n:.4f}")

    return total / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

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
        if best_iou >= thr and best_gi >= 0:
            tp[pi] = 1
            matched[best_gi] = True
        else:
            fp[pi] = 1

    cum_tp   = torch.cumsum(tp, 0)
    cum_fp   = torch.cumsum(fp, 0)
    prec     = cum_tp / (cum_tp + cum_fp + 1e-6)
    rec      = cum_tp / len(gt_masks)
    return float(torch.trapz(prec, rec))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device,
             conf_thr: float = 0.3,
             small_pixel_thr: int = 1024) -> dict:
    """
    Returns AP, AP50, AP75, AP_S (small defects), mIoU.
    small_pixel_thr: GT masks with area ≤ this are counted as 'small'.
    """
    model.eval()
    aps      = {t: [] for t in _IOU_THRS}
    aps_small = {t: [] for t in _IOU_THRS}
    mious    = []

    for imgs, targets in loader:
        imgs = imgs.to(device)
        H, W = imgs.shape[-2:]
        results = model.predict(imgs, conf_threshold=conf_thr)

        for result, target in zip(results, targets):
            pred_m  = result["masks"].cpu().view(-1, H * W)
            pred_s  = result["scores"].cpu()
            gt_m    = target["masks"].bool().cpu()

            for thr in _IOU_THRS:
                aps[thr].append(
                    _ap_at_thr(pred_m, pred_s, gt_m, thr)
                )

            # Small-defect AP
            small_idx = [i for i, m in enumerate(gt_m)
                         if m.sum().item() <= small_pixel_thr]
            if small_idx:
                gt_small = gt_m[small_idx]
                for thr in _IOU_THRS:
                    aps_small[thr].append(
                        _ap_at_thr(pred_m, pred_s, gt_small, thr)
                    )

            # mIoU
            if len(gt_m) == 0 and len(pred_m) == 0:
                mious.append(1.0)
            elif len(gt_m) == 0 or len(pred_m) == 0:
                mious.append(0.0)
            else:
                matched = torch.zeros(len(pred_m), dtype=torch.bool)
                ious    = []
                for gm in gt_m:
                    best = 0.0
                    bi   = -1
                    for pi, pm in enumerate(pred_m):
                        if matched[pi]:
                            continue
                        iou = _mask_iou(pm, gm)
                        if iou > best:
                            best, bi = iou, pi
                    if best >= 0.5 and bi >= 0:
                        matched[bi] = True
                    ious.append(best)
                mious.append(float(np.mean(ious)))

    AP    = float(np.mean([np.mean(v) for v in aps.values()]))
    AP50  = float(np.mean(aps[0.50]))
    AP75  = float(np.mean(aps[0.75]))
    APS   = float(np.mean([np.mean(v) for v in aps_small.values()])) \
            if any(aps_small[0.50]) else float("nan")
    mIoU  = float(np.mean(mious))

    return {"AP": AP, "AP50": AP50, "AP75": AP75, "AP_S": APS, "mIoU": mIoU}


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(cfg: ExperimentConfig,
                   args: argparse.Namespace,
                   exp_dir: str,
                   device: torch.device,
                   num_classes: int) -> dict:

    print(f"\n  backbone={cfg.backbone}  msfe={cfg.msfe_variant}  "
          f"prior_init={cfg.use_prior_init}  dmca={cfg.use_dmca_mask}  "
          f"dcsa={cfg.use_dcsa_repulsion}  syn_aug={cfg.use_synthetic_aug}")

    # ── Dataset ──────────────────────────────────────────────────────
    train_loader, test_loader, _ = build_loaders(args, cfg.use_synthetic_aug)

    # ── Model ────────────────────────────────────────────────────────
    model     = build_model(cfg, num_classes, pretrained=True).to(device)
    criterion = DefectFormerLoss(num_classes=num_classes,
                                  lambda_cls=2.0, lambda_mask=5.0, lambda_dice=5.0)

    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = AdamW(
        [
            {"params": [p for p in model.parameters() if id(p) in backbone_ids],
             "lr": args.lr * 0.1},
            {"params": [p for p in model.parameters() if id(p) not in backbone_ids],
             "lr": args.lr},
        ],
        weight_decay=0.05,
    )

    warmup = args.warmup_epochs
    total  = args.epochs

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ── Training loop ────────────────────────────────────────────────
    best_loss  = float("inf")
    best_ckpt  = os.path.join(exp_dir, "best.pth")
    t_start    = time.time()

    for epoch in range(total):
        loss = train(model, train_loader, criterion, optimizer,
                     device, epoch, log_every=args.log_every)
        scheduler.step()

        lr_now = optimizer.param_groups[1]["lr"]
        elapsed = (time.time() - t_start) / 60
        print(f"  epoch {epoch:3d}/{total}  loss={loss:.4f}  "
              f"lr={lr_now:.2e}  elapsed={elapsed:.1f}min")

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), best_ckpt)

        # Light val eval every val_every epochs (skip in quick mode to save time)
        if not args.quick and (epoch + 1) % args.val_every == 0:
            m = evaluate(model, test_loader, device, conf_thr=args.conf_thr)
            print(f"    [val] AP={m['AP']*100:.1f}  AP50={m['AP50']*100:.1f}  "
                  f"AP_S={m['AP_S']*100:.1f}  mIoU={m['mIoU']*100:.1f}")

    # ── Final evaluation ─────────────────────────────────────────────
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    metrics = evaluate(model, test_loader, device, conf_thr=args.conf_thr)

    total_min = (time.time() - t_start) / 60
    print(f"  → AP={metrics['AP']*100:.1f}  AP50={metrics['AP50']*100:.1f}  "
          f"AP_S={metrics['AP_S']*100:.1f}  mIoU={metrics['mIoU']*100:.1f}  "
          f"({total_min:.1f}min)")

    return {
        "label":      cfg.label,
        "AP":         round(metrics["AP"]   * 100, 1),
        "AP50":       round(metrics["AP50"] * 100, 1),
        "AP75":       round(metrics["AP75"] * 100, 1),
        "AP_S":       round(metrics["AP_S"] * 100, 1) if not math.isnan(metrics["AP_S"]) else None,
        "mIoU":       round(metrics["mIoU"] * 100, 1),
        "paper_AP":   cfg.paper_ap,
        "paper_AP_S": cfg.paper_ap_s,
        "train_min":  round(total_min, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: dict):
    print("\n" + "=" * 90)
    print(f"  {'Configuration':<42}  {'AP':>5}  {'AP50':>5}  {'AP75':>5}  "
          f"{'AP_S':>5}  {'mIoU':>5}  {'Paper AP':>8}")
    print("  " + "-" * 86)
    for name, r in results.items():
        ap_s_str  = f"{r['AP_S']:5.1f}" if r.get("AP_S") is not None else "  N/A"
        paper_str = f"{r['paper_AP']:8.1f}" if r.get("paper_AP") else "     ---"
        print(f"  {r['label']:<42}  {r['AP']:5.1f}  {r['AP50']:5.1f}  "
              f"{r['AP75']:5.1f}  {ap_s_str}  {r['mIoU']:5.1f}  {paper_str}")
    print("=" * 90)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DefectFormer ablation runner")
    p.add_argument("--data_root",   required=True,
                   help="Path to MVTec AD root directory")
    p.add_argument("--output_dir",  default="./ablation_results",
                   help="Directory for checkpoints and results.json")
    p.add_argument("--experiments", default="all",
                   help="Comma-separated list of experiment names, or 'all'. "
                        "Available: " + ", ".join(EXPERIMENT_MAP))
    p.add_argument("--epochs",      type=int,   default=150)
    p.add_argument("--warmup_epochs",type=int,  default=10)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--img_size",    type=int,   default=1024)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--conf_thr",    type=float, default=0.3)
    p.add_argument("--val_every",   type=int,   default=20,
                   help="Evaluate on test set every N epochs (ignored in --quick)")
    p.add_argument("--log_every",   type=int,   default=50,
                   help="Print training loss every N steps")
    p.add_argument("--quick",       action="store_true",
                   help="Skip per-epoch val eval; faster iteration")
    p.add_argument("--resume",      action="store_true",
                   help="Skip experiments that already have results in results.json")
    p.add_argument("--dry_run",     action="store_true",
                   help="Print configurations and exit without training")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Epochs : {args.epochs}  Batch: {args.batch_size}  "
          f"ImgSize: {args.img_size}")

    # ── Select experiments ────────────────────────────────────────────
    if args.experiments.strip().lower() == "all":
        exps = ABLATION_EXPERIMENTS
    else:
        names = [n.strip() for n in args.experiments.split(",")]
        unknown = [n for n in names if n not in EXPERIMENT_MAP]
        if unknown:
            print(f"ERROR: unknown experiment(s): {unknown}")
            print(f"       valid names: {list(EXPERIMENT_MAP)}")
            sys.exit(1)
        exps = [EXPERIMENT_MAP[n] for n in names]

    if args.dry_run:
        print(f"\n{'─'*70}")
        print(f"{'#':<3}  {'name':<20}  {'backbone':<10}  {'msfe':<12}  "
              f"{'prior':<6}  {'dmca':<6}  {'dcsa':<6}  {'aug':<4}")
        print(f"{'─'*70}")
        for i, e in enumerate(exps, 1):
            print(f"{i:<3}  {e.name:<20}  {e.backbone:<10}  {e.msfe_variant:<12}  "
                  f"{str(e.use_prior_init):<6}  {str(e.use_dmca_mask):<6}  "
                  f"{str(e.use_dcsa_repulsion):<6}  {str(e.use_synthetic_aug):<4}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "results.json")

    # ── Load existing results (resume mode) ──────────────────────────
    results: dict = {}
    if args.resume and os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {results_file}")

    # ── Build dataset once to get num_classes ────────────────────────
    tmp_ds      = MVTecDataset(args.data_root, split="train", img_size=args.img_size)
    num_classes = tmp_ds.num_classes
    print(f"Num classes: {num_classes}")
    del tmp_ds

    # ── Run experiments ───────────────────────────────────────────────
    for exp in exps:
        if args.resume and exp.name in results:
            print(f"\n[SKIP] {exp.label}  (already in results.json)")
            continue

        exp_dir = os.path.join(args.output_dir, exp.name)
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n{'━'*70}")
        print(f"[{exp.name}]  {exp.label}")
        print(f"{'━'*70}")

        try:
            r = run_experiment(exp, args, exp_dir, device, num_classes)
            results[exp.name] = r
        except Exception as ex:
            import traceback
            print(f"  ERROR in {exp.name}: {ex}")
            traceback.print_exc()
            results[exp.name] = {"label": exp.label, "error": str(ex)}

        # Save after each experiment so a crash doesn't lose work
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved → {results_file}")

    # ── Final table ───────────────────────────────────────────────────
    valid = {k: v for k, v in results.items() if "error" not in v}
    if valid:
        print_table(valid)

    print(f"\nAll results saved to {results_file}")


if __name__ == "__main__":
    main()

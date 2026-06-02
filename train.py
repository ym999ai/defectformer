"""
DefectFormer training script.

Example (MVTec AD, 4× A100):
  torchrun --nproc_per_node=4 train.py \\
      --data_root /data/mvtec_anomaly_detection \\
      --dataset mvtec \\
      --output_dir ./runs/defectformer_mvtec
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from models.defectformer import DefectFormer
from models.losses import DefectFormerLoss
from datasets.mvtec import MVTecDataset
from datasets.kolektor import KolektorSDD2Dataset


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, optimizer, criterion, loader, device, epoch, log_every=50):
    model.train()
    running = 0.0
    t0 = time.time()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        all_cls, all_masks = model(images)

        loss = criterion(all_cls, all_masks, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        running += loss.item()

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"  [epoch {epoch:3d}  step {step+1:4d}/{len(loader)}]  "
                  f"loss={running / (step + 1):.4f}  "
                  f"({elapsed:.1f}s)")

    return running / len(loader)


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_dataset(args):
    if args.dataset == "mvtec":
        train_ds = MVTecDataset(args.data_root, split="train",
                                img_size=args.img_size)
        val_ds   = MVTecDataset(args.data_root, split="val",
                                img_size=args.img_size)
        num_classes = train_ds.num_classes
    elif args.dataset == "kolektor":
        train_ds = KolektorSDD2Dataset(args.data_root, split="train",
                                       img_size=args.img_size)
        val_ds   = KolektorSDD2Dataset(args.data_root, split="test",
                                       img_size=args.img_size)
        num_classes = KolektorSDD2Dataset.NUM_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return train_ds, val_ds, num_classes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--dataset",     default="mvtec", choices=["mvtec", "kolektor"])
    parser.add_argument("--output_dir",  default="./runs/defectformer")
    parser.add_argument("--epochs",      type=int,   default=150)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--wd",          type=float, default=0.05)
    parser.add_argument("--img_size",    type=int,   default=1024)
    parser.add_argument("--nq",          type=int,   default=100)
    parser.add_argument("--warmup",      type=int,   default=10,
                        help="Warmup epochs (linear ramp)")
    parser.add_argument("--workers",     type=int,   default=4)
    parser.add_argument("--resume",      default=None)
    parser.add_argument("--save_every",  type=int,   default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    train_ds, val_ds, num_classes = build_dataset(args)
    print(f"Dataset: {args.dataset}  classes={num_classes}  "
          f"train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              collate_fn=collate_fn, pin_memory=True)

    # Model
    model = DefectFormer(
        num_classes=num_classes,
        d=256,
        nq=args.nq,
        num_decoder_layers=6,
        num_heads=8,
    ).to(device)

    # Loss
    criterion = DefectFormerLoss(
        num_classes=num_classes,
        lambda_cls=2.0, lambda_mask=5.0, lambda_dice=5.0,
    )

    # Optimizer — backbone gets 10× lower LR (fine-tuning)
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) in backbone_ids]
    other_params    = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": other_params,    "lr": args.lr},
        ],
        weight_decay=args.wd,
    )

    # LR schedule: linear warmup + cosine decay
    warmup_epochs = args.warmup
    total_epochs  = args.epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, total_epochs):
        loss = train_epoch(model, optimizer, criterion, train_loader,
                           device, epoch)
        scheduler.step()

        lr_now = optimizer.param_groups[1]["lr"]
        print(f"Epoch {epoch:3d}/{total_epochs}  "
              f"train_loss={loss:.4f}  lr={lr_now:.2e}")

        if (epoch + 1) % args.save_every == 0 or epoch == total_epochs - 1:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_e{epoch:03d}.pth")
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()

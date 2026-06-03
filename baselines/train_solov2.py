"""
SOLOv2 baseline — mmdetection v3.x.

Architecture : SOLOv2-R50-FPN  (ResNet-50, same backbone as Table III)
Schedule     : 150 epochs  (cosine, warmup 500 steps)
Optimizer    : SGD momentum=0.9  (standard SOLOv2 setting)

Usage
-----
# If COCO data not yet generated, the script auto-converts it.
python baselines/train_solov2.py \\
    --dataset   mvtec \\
    --data_root /data/mvtec_anomaly_detection \\
    --coco_dir  ./coco_data/mvtec \\
    --output_dir ./runs/solov2_mvtec \\
    --epochs 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines._mmdet_base import ensure_coco_data, write_cfg_file, run_mmdet_train


# ─────────────────────────────────────────────────────────────────────────────
# Config generator
# ─────────────────────────────────────────────────────────────────────────────

_CFG_TEMPLATE = '''
# Auto-generated SOLOv2 config for DefectFormer comparison
_base_ = ['mmdet::solov2/solov2_r50_fpn_3x_coco.py']

# ── Dataset ──────────────────────────────────────────────────────
num_classes   = {num_classes}
_classes      = {classes_repr}

metainfo = dict(classes=_classes)

train_dataloader = dict(
    batch_size={batch_size},
    num_workers={workers},
    dataset=dict(
        type='CocoDataset',
        ann_file='{train_ann}',
        data_prefix=dict(img=''),
        metainfo=dict(classes=_classes),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='Resize', scale=({img_size}, {img_size}), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs'),
        ],
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        ann_file='{test_ann}',
        data_prefix=dict(img=''),
        metainfo=dict(classes=_classes),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=({img_size}, {img_size}), keep_ratio=True),
            dict(type='PackDetInputs'),
        ],
    ),
)
test_dataloader = val_dataloader

val_evaluator  = dict(type='CocoMetric', ann_file='{test_ann}',
                      metric=['bbox', 'segm'])
test_evaluator = val_evaluator

# ── Model: only override num_classes ─────────────────────────────
model = dict(mask_head=dict(num_classes=num_classes))

# ── Schedule ─────────────────────────────────────────────────────
max_epochs = {epochs}
train_cfg  = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=10)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, end=max_epochs, T_max=max_epochs,
         by_epoch=True, convert_to_iter_based=True),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# ── Misc ─────────────────────────────────────────────────────────
work_dir          = '{work_dir}'
default_hooks     = dict(checkpoint=dict(type='CheckpointHook', interval=10,
                                         save_best='coco/segm_mAP'))
load_from         = None
'''


def build_config(meta: dict, coco_dir: str, output_dir: str,
                 epochs: int, batch_size: int, img_size: int, workers: int) -> str:
    train_ann = os.path.join(coco_dir, "train.json")
    test_ann  = os.path.join(coco_dir, "test.json")
    cfg_str   = _CFG_TEMPLATE.format(
        num_classes   = meta["num_classes"],
        classes_repr  = repr(meta["classes_tuple"]),
        train_ann     = train_ann,
        test_ann      = test_ann,
        img_size      = img_size,
        epochs        = epochs,
        batch_size    = batch_size,
        workers       = workers,
        work_dir      = output_dir,
    )
    cfg_path = os.path.join(output_dir, "solov2_config.py")
    return write_cfg_file(cfg_str, cfg_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="mvtec", choices=["mvtec", "kolektor"])
    p.add_argument("--data_root",  default=None,
                   help="Raw dataset root (only needed if COCO not yet generated)")
    p.add_argument("--coco_dir",   required=True,
                   help="Directory containing COCO JSON files (from convert_to_coco.py)")
    p.add_argument("--output_dir", default="./runs/solov2")
    p.add_argument("--epochs",     type=int, default=150)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--img_size",   type=int, default=1024)
    p.add_argument("--workers",    type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    meta     = ensure_coco_data(args.dataset, args.data_root,
                                args.coco_dir, args.img_size)
    cfg_path = build_config(meta, args.coco_dir, args.output_dir,
                            args.epochs, args.batch_size,
                            args.img_size, args.workers)

    print(f"Config written → {cfg_path}")
    print(f"Starting SOLOv2 training  ({args.epochs} epochs)...")
    run_mmdet_train(cfg_path)


if __name__ == "__main__":
    main()

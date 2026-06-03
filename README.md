# DefectFormer

**DefectFormer: A Query-Driven Transformer with Multi-Scale Feature Enhancement for Industrial Defect Instance Segmentation**

DefectFormer is an end-to-end transformer-based architecture for industrial defect instance segmentation. It produces per-instance binary masks, class labels, and confidence scores — without NMS or hand-designed post-processing.

---

## Architecture

```
Input Image
    │
    ▼
┌─────────────────────────────┐
│  Swin-T Backbone            │  → {F2, F3, F4, F5}  strides {4,8,16,32}
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  MSFE Module                │
│  ├── Top-down pass C5→C2    │  CSAG: cross-scale attention gate
│  └── Bottom-up pass C2→C5  │  LTEC: local texture enhancement conv
└─────────────────────────────┘  → {P2, P3, P4, P5}
    │
    ▼
┌─────────────────────────────┐
│  Pixel Decoder              │  → E ∈ R^{H/8 × W/8 × 256}
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  QDID (L=6 layers)          │
│  ├── Defect-Prior Init      │  saliency Top-K seeding
│  ├── DMCA                  │  defect-masked cross-attention
│  └── DCSA                  │  IoU-based spatial repulsion
└─────────────────────────────┘
    │
    ▼
{masks, class labels, scores}
```

### Key Components

| Module | Description |
|---|---|
| **CSAG** | Cross-Scale Attention Gate — dynamic content-dependent cross-scale fusion via softmax similarity matrix |
| **LTEC** | Local Texture Enhancement Convolution — two-block DW-Sep-Conv with residual, recovers boundary detail |
| **Defect-Prior Init** | Seeds queries from Top-K saliency positions of E rather than random initialization |
| **DMCA** | Defect-Masked Cross-Attention — constrains attention to predicted foreground regions |
| **DCSA** | Defect Context Self-Attention — IoU repulsion (γ=0.5) prevents duplicate instance predictions |

---

## Results

### MVTec AD

| Method | AP | AP50 | AP75 | mIoU |
|---|---|---|---|---|
| Mask R-CNN | 29.4 | 51.2 | 28.7 | 42.3 |
| SOLOv2 | 32.1 | 54.8 | 31.5 | 44.7 |
| CondInst | 35.6 | 58.3 | 34.8 | 47.1 |
| Mask2Former | 41.3 | 63.7 | 40.8 | 53.2 |
| **DefectFormer (Ours)** | **48.6** | **71.4** | **47.9** | **59.8** |

### KolektorSDD2

| Method | AP | mIoU |
|---|---|---|
| Mask2Former | 43.7 | 56.8 |
| **DefectFormer (Ours)** | **52.4** | **64.3** |

---

## Installation

**Requirements:** Python ≥ 3.10, CUDA ≥ 11.8

```bash
git clone https://github.com/ym999ai/defectformer.git
cd defectformer
pip install -r requirements.txt
```

`requirements.txt`:
```
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.12
scipy>=1.11.0
numpy>=1.24.0
Pillow>=10.0.0
tqdm>=4.65.0
```

---

## Dataset Preparation

### MVTec AD

Download from [mvtec.com/research-teaching/datasets/mvtec-ad](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).

Expected layout:
```
mvtec_anomaly_detection/
├── bottle/
│   ├── train/good/
│   ├── test/good/
│   ├── test/<defect_type>/
│   └── ground_truth/<defect_type>/*_mask.png
├── cable/
│   └── ...
└── ...
```

Instance annotations are automatically derived from connected-component analysis on the provided pixel masks (components < 16 pixels are discarded). The anomalous test images are split 60/20/20 (train/val/test) per category.

### KolektorSDD2

Download from [vicos.si/resources/kolektorsdd2](https://www.vicos.si/resources/kolektorsdd2).

Expected layout:
```
kolektorsdd2/
├── train/
│   ├── *.jpg
│   └── *_GT.png
└── test/
    ├── *.jpg
    └── *_GT.png
```

---

## Training

**Single GPU:**
```bash
python train.py \
    --data_root /path/to/mvtec_anomaly_detection \
    --dataset mvtec \
    --output_dir ./runs/defectformer_mvtec \
    --epochs 150 \
    --batch_size 2 \
    --img_size 1024
```

**Multi-GPU (4× A100, batch=8):**
```bash
torchrun --nproc_per_node=4 train.py \
    --data_root /path/to/mvtec_anomaly_detection \
    --dataset mvtec \
    --output_dir ./runs/defectformer_mvtec \
    --epochs 150 \
    --batch_size 8
```

**KolektorSDD2:**
```bash
python train.py \
    --data_root /path/to/kolektorsdd2 \
    --dataset kolektor \
    --epochs 100
```

### Training Settings

| Setting | Value |
|---|---|
| Backbone | Swin-T, ImageNet-22K pretrained |
| Feature channel *d* | 256 |
| Queries *N_q* | 100 |
| Decoder layers *L* | 6 |
| Input resolution | 1024×1024 |
| Optimizer | AdamW, lr=1e-4, wd=0.05 |
| Backbone lr | 1e-5 (10× lower) |
| LR schedule | Linear warmup (10 ep) + cosine decay |
| Loss weights | λ_cls=2, λ_mask=5, λ_dice=5 |

---

## Evaluation

```bash
python eval.py \
    --data_root /path/to/mvtec_anomaly_detection \
    --dataset mvtec \
    --checkpoint ./runs/defectformer_mvtec/checkpoint_e149.pth
```

Metrics reported: AP (IoU 0.50:0.05:0.95), AP50, AP75, mIoU.

---

## Project Structure

```
defectformer/
├── models/
│   ├── msfe.py           # LTEC, CSAG, MSFE
│   ├── pixel_decoder.py  # multi-scale fusion → E
│   ├── qdid.py           # DefectPriorInitializer, DMCA, DCSA, QDID
│   ├── defectformer.py   # full model
│   └── losses.py         # HungarianMatcher + compound loss
├── datasets/
│   ├── mvtec.py          # MVTec AD loader
│   └── kolektor.py       # KolektorSDD2 loader
├── train.py
├── eval.py
└── requirements.txt
```

---

## Citation

If you find this code useful, please cite:

```bibtex
@article{defectformer2024,
  title   = {DefectFormer: A Query-Driven Transformer with Multi-Scale
             Feature Enhancement for Industrial Defect Instance Segmentation},
  author  = {Author, First and Author, Second and Author, Third},
  journal = {Pattern Recognition},
  year    = {2024}
}
```

---

## License

This project is released under the MIT License.

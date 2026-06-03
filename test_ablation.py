"""
Smoke test: verifies all 10 ablation model variants can do a forward pass.
Uses tiny inputs (CPU, no real data needed).
"""

import sys
import torch

sys.path.insert(0, ".")

from ablation.configs import ABLATION_EXPERIMENTS
from models.ablation import DefectFormerAblation


IMG_SIZE    = 224    # small for speed
NUM_CLASSES = 5
NQ          = 10
LAYERS      = 2
HEADS       = 4
BATCH       = 1


def _make_input():
    return torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)


def test_variant(cfg):
    model = DefectFormerAblation(
        num_classes        = NUM_CLASSES,
        d                  = 64,
        nq                 = NQ,
        num_decoder_layers = LAYERS,
        num_heads          = HEADS,
        backbone           = cfg.backbone,
        msfe_variant       = cfg.msfe_variant,
        use_prior_init     = cfg.use_prior_init,
        use_dmca_mask      = cfg.use_dmca_mask,
        use_dcsa_repulsion = cfg.use_dcsa_repulsion,
        pretrained         = False,   # no download during test
    )
    model.eval()

    x = _make_input()
    with torch.no_grad():
        cls_list, mask_list = model(x)

    L = IMG_SIZE * IMG_SIZE
    assert len(cls_list)  == LAYERS, "wrong number of decoder layers"
    assert len(mask_list) == LAYERS
    assert cls_list[-1].shape  == (BATCH, NQ, NUM_CLASSES + 1), \
        f"cls shape mismatch: {cls_list[-1].shape}"
    assert mask_list[-1].shape == (BATCH, NQ, L), \
        f"mask shape mismatch: {mask_list[-1].shape}"

    # predict()
    results = model.predict(x, conf_threshold=0.01)
    assert len(results) == BATCH
    assert "labels" in results[0] and "masks" in results[0]


def test_synthetic_aug():
    from ablation.synthetic_aug import SyntheticAnomalyAug

    aug = SyntheticAnomalyAug(p_apply=1.0)  # always apply
    img = torch.randn(3, IMG_SIZE, IMG_SIZE)
    target = {
        "labels": torch.tensor([0], dtype=torch.long),
        "masks":  torch.zeros(1, IMG_SIZE * IMG_SIZE, dtype=torch.bool),
    }
    img_aug, tgt_aug = aug(img, target)

    assert img_aug.shape == img.shape
    assert len(tgt_aug["labels"]) == 2,  "synthetic instance not appended"
    assert tgt_aug["masks"].shape[0] == 2
    print("  SyntheticAnomalyAug OK")


if __name__ == "__main__":
    passed, failed = 0, []

    for cfg in ABLATION_EXPERIMENTS:
        print(f"\n[{cfg.name}]  {cfg.label}")
        try:
            test_variant(cfg)
            print(f"  OK — backbone={cfg.backbone}, msfe={cfg.msfe_variant}, "
                  f"prior={cfg.use_prior_init}, dmca={cfg.use_dmca_mask}, "
                  f"dcsa={cfg.use_dcsa_repulsion}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed.append(cfg.name)

    print("\n[SyntheticAug]")
    try:
        test_synthetic_aug()
        passed += 1
    except Exception as e:
        import traceback
        print(f"  FAILED: {e}")
        traceback.print_exc()
        failed.append("synthetic_aug")

    total = len(ABLATION_EXPERIMENTS) + 1
    print(f"\n{'='*55}")
    print(f"Results: {passed}/{total} passed")
    if failed:
        print(f"Failed:  {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All ablation variant tests passed!")

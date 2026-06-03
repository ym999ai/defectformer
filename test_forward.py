"""
Smoke test: verify a full forward pass and loss computation without real data.
Uses random tensors on CPU (no GPU required).
"""

import sys
import torch

sys.path.insert(0, ".")


def test_msfe():
    from models.msfe import MSFE
    in_channels = [96, 192, 384, 768]
    msfe = MSFE(in_channels, d=256)
    msfe.eval()

    # Simulate Swin-T outputs at 64×64 base (stride-4 feature)
    feats = [
        torch.randn(1, c, 64 // (2**i), 64 // (2**i))
        for i, c in enumerate(in_channels)
    ]
    out = msfe(feats)
    assert len(out) == 4
    for i, f in enumerate(out):
        assert f.shape[1] == 256, f"MSFE out[{i}] channel mismatch"
    print(f"  MSFE OK — output shapes: {[tuple(f.shape) for f in out]}")


def test_pixel_decoder():
    from models.pixel_decoder import PixelDecoder
    pd = PixelDecoder(d=256)
    pd.eval()

    feats = [
        torch.randn(1, 256, 64 // (2**i), 64 // (2**i))
        for i in range(4)
    ]
    E = pd(feats)
    assert E.shape == (1, 256, 32, 32), f"PixelDecoder shape: {E.shape}"
    print(f"  PixelDecoder OK — E shape: {tuple(E.shape)}")


def test_qdid():
    from models.qdid import QDID
    NUM_CLS = 5
    qdid = QDID(d=64, nq=10, num_layers=2, num_classes=NUM_CLS, num_heads=4)
    qdid.eval()

    E = torch.randn(1, 64, 16, 16)
    with torch.no_grad():
        cls_list, mask_list = qdid(E)

    assert len(cls_list) == 2
    assert len(mask_list) == 2
    assert cls_list[-1].shape == (1, 10, NUM_CLS + 1)
    assert mask_list[-1].shape == (1, 10, 16 * 16)
    print(f"  QDID OK — cls: {tuple(cls_list[-1].shape)}, mask: {tuple(mask_list[-1].shape)}")


def test_full_model():
    """Full forward pass through DefectFormer (CPU, small image)."""
    from models.defectformer import DefectFormer
    print("  Loading Swin-T backbone (first run downloads pretrained weights)…")
    model = DefectFormer(num_classes=5, d=256, nq=20, num_decoder_layers=2,
                         num_heads=8, pretrained=True)
    model.eval()

    x = torch.randn(1, 3, 224, 224)   # small input for speed
    with torch.no_grad():
        cls_list, mask_list = model(x)

    assert len(cls_list) == 2
    H, W = 224, 224
    assert cls_list[-1].shape == (1, 20, 6)          # 5 classes + 1 no-obj
    assert mask_list[-1].shape == (1, 20, H * W)
    print(f"  Full model OK — cls: {tuple(cls_list[-1].shape)}, "
          f"mask: {tuple(mask_list[-1].shape)}")


def test_loss():
    """Loss computation with synthetic targets."""
    from models.defectformer import DefectFormer
    from models.losses import DefectFormerLoss

    model = DefectFormer(num_classes=5, d=256, nq=20, num_decoder_layers=2,
                         num_heads=8, pretrained=False)
    model.train()

    criterion = DefectFormerLoss(num_classes=5, lambda_cls=2, lambda_mask=5, lambda_dice=5)

    H, W = 224, 224
    x = torch.randn(2, 3, H, W)

    # Two synthetic GT instances for image 0, one for image 1
    targets = [
        {
            "labels": torch.tensor([0, 2], dtype=torch.long),
            "masks":  torch.zeros(2, H * W, dtype=torch.bool).scatter_(
                1, torch.randint(0, H * W, (2, 500)), True
            ),
        },
        {
            "labels": torch.tensor([1], dtype=torch.long),
            "masks":  torch.zeros(1, H * W, dtype=torch.bool).scatter_(
                1, torch.randint(0, H * W, (1, 300)), True
            ),
        },
    ]

    cls_list, mask_list = model(x)
    loss = criterion(cls_list, mask_list, targets)

    assert loss.item() > 0
    loss.backward()
    print(f"  Loss OK — value: {loss.item():.4f}, backward passed")


def test_predict():
    """Inference predict() method."""
    from models.defectformer import DefectFormer
    model = DefectFormer(num_classes=5, d=256, nq=20, num_decoder_layers=2,
                         num_heads=8, pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    results = model.predict(x, conf_threshold=0.01)   # low threshold → some predictions

    assert len(results) == 1
    r = results[0]
    assert "labels" in r and "scores" in r and "masks" in r
    print(f"  predict() OK — {len(r['labels'])} instances detected "
          f"(conf>0.01), mask shape: {r['masks'].shape if len(r['masks']) else 'empty'}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("MSFE",        test_msfe),
        ("PixelDecoder", test_pixel_decoder),
        ("QDID",        test_qdid),
        ("Full model (pretrained backbone)", test_full_model),
        ("Loss + backward",                  test_loss),
        ("predict()",                        test_predict),
    ]

    passed, failed = 0, []
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed.append(name)

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed")
    if failed:
        print(f"Failed:  {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All tests passed!")

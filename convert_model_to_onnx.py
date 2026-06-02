#!/usr/bin/env python3
"""
One-time conversion script: Export the trained ResNet18 PyTorch model to ONNX format.

This only needs to run ONCE. After conversion, the web app uses onnxruntime
for inference, which imports in < 1 second (vs PyTorch's 477s on this machine).

Usage:
    .venv/bin/python convert_model_to_onnx.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

# Single-threaded to avoid macOS deadlocks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "data/preprocessing_experiments/original_enhanced/model/best_resnet18.pt"
ONNX_PATH = ROOT / "data/preprocessing_experiments/original_enhanced/model/best_resnet18.onnx"
CLASSES_PATH = ROOT / "data/preprocessing_experiments/original_enhanced/model/classes.json"

def main():
    print("=" * 60)
    print("ResNet18 PyTorch → ONNX Conversion")
    print("=" * 60)

    print("\n⏳ Importing PyTorch (this will take a while on this machine)...")
    t0 = time.time()
    import torch
    t1 = time.time()
    print(f"  ✓ PyTorch {torch.__version__} imported in {t1 - t0:.1f}s")

    from evaluate_wild_no_torchvision import ResNet18
    import json

    print(f"\n⏳ Loading model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]
    print(f"  ✓ Classes: {classes}")

    model = ResNet18(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"  ✓ Model loaded successfully")

    # Save classes to a separate JSON file for the web app to load
    with open(CLASSES_PATH, "w") as f:
        json.dump(classes, f)
    print(f"  ✓ Classes saved to: {CLASSES_PATH}")

    # Create dummy input for export
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    print(f"\n⏳ Exporting to ONNX: {ONNX_PATH}")
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    t2 = time.time()

    # Verify the exported model
    import onnx
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    file_size = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"  ✓ ONNX model exported successfully!")
    print(f"  ✓ File size: {file_size:.1f} MB")
    print(f"  ✓ Export took: {t2 - t1:.1f}s")

    # Quick verification with onnxruntime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(ONNX_PATH))
        import numpy as np
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {"input": test_input})
        print(f"\n✅ ONNX Runtime verification passed!")
        print(f"   Output shape: {outputs[0].shape}")
        print(f"   Predicted class index: {outputs[0].argmax()}")
    except ImportError:
        print(f"\n⚠️  onnxruntime not installed — install it with:")
        print(f"   .venv/bin/pip install onnxruntime")

    print(f"\n{'=' * 60}")
    print(f"✅ Conversion complete!")
    print(f"   ONNX model: {ONNX_PATH}")
    print(f"   Classes JSON: {CLASSES_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

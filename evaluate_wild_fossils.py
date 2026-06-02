#!/usr/bin/env python3
"""
Evaluate the exploratory fossil classifier on wild/background-shift images.

This script intentionally does not train. It loads a saved model and evaluates
images from folders such as:

  ../FuckingFossils/WildCorals
  ../FuckingFossils/WildShells

The wild folders are mapped back to the training class names:

  WildCorals -> FuckingCorals
  WildShells -> FuckingShells
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_SOURCE = Path("../FuckingFossils")
DEFAULT_MODEL = Path("data/exploratory/model/best_resnet18.pt")
DEFAULT_OUTPUT = Path("data/exploratory/wild_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the fossil classifier on WildCorals/WildShells only."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["original", "enhanced"],
        default=["original", "enhanced"],
        help="Evaluate original images, enhanced duplicates, or both.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda", "auto"],
        default="cpu",
        help="Device for inference. CPU is the default because it is stable and fast enough.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def class_from_wild_folder(folder_name: str) -> str | None:
    lowered = folder_name.lower()
    if "coral" in lowered:
        return "FuckingCorals"
    if "shell" in lowered:
        return "FuckingShells"
    return None


def variant_original(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def variant_enhanced(image: Image.Image) -> Image.Image:
    image = ImageOps.autocontrast(image.convert("RGB"), cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(1.25)
    image = ImageEnhance.Sharpness(image).enhance(1.15)
    return image


VARIANTS = {
    "original": variant_original,
    "enhanced": variant_enhanced,
}


def discover_wild_images(source: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for folder in sorted(path for path in source.iterdir() if path.is_dir()):
        true_class = class_from_wild_folder(folder.name)
        if true_class is None:
            continue
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((image_path, true_class))
    return items


def load_model(model_path: Path, device):
    import torch
    from torch import nn
    from torchvision import models

    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, classes


def evaluate_wild(
    source: Path,
    model_path: Path,
    output: Path,
    variants: list[str],
    device_name: str = "cpu",
) -> dict:
    print("Importing PyTorch for wild evaluation...", flush=True)
    import torch
    from torchvision import transforms

    output.mkdir(parents=True, exist_ok=True)

    if device_name == "auto":
        resolved_device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        resolved_device = device_name
    device = torch.device(resolved_device)
    print(f"Using device: {device}", flush=True)

    model, classes = load_model(model_path, device)
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    items = discover_wild_images(source)
    if not items:
        raise RuntimeError(f"No wild images found under {source}")

    rows: list[dict[str, str | float]] = []
    confusion = [[0 for _ in classes] for _ in classes]
    correct = 0
    total = 0

    with torch.no_grad():
        for image_path, true_class in items:
            if true_class not in class_to_idx:
                raise RuntimeError(
                    f"Wild label {true_class} is not in trained classes {classes}"
                )

            image = Image.open(image_path).convert("RGB")
            for variant_name in variants:
                variant = VARIANTS[variant_name](image)
                tensor = transform(variant).unsqueeze(0).to(device)
                probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()
                predicted_idx = int(probabilities.argmax().item())
                true_idx = class_to_idx[true_class]
                predicted_class = classes[predicted_idx]
                confidence = float(probabilities[predicted_idx].item())
                is_correct = predicted_class == true_class

                total += 1
                correct += int(is_correct)
                confusion[true_idx][predicted_idx] += 1

                rows.append(
                    {
                        "image": str(image_path),
                        "variant": variant_name,
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "correct": str(is_correct),
                    }
                )

                print(
                    f"{image_path.parent.name}/{image_path.name} [{variant_name}] "
                    f"true={true_class} predicted={predicted_class} "
                    f"confidence={confidence:.3f} correct={is_correct}"
                )

    accuracy = correct / total if total else 0.0

    predictions_path = output / "wild_predictions.csv"
    with predictions_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image",
                "variant",
                "true_class",
                "predicted_class",
                "confidence",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "classes": classes,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "variants": variants,
        "source": str(source),
        "model": str(model_path),
    }
    summary_path = output / "wild_summary.json"
    with summary_path.open("w") as json_file:
        json.dump(summary, json_file, indent=2)

    print("\nWild test summary")
    print(f"Images before variants: {len(items)}")
    print(f"Evaluated samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    evaluate_wild(
        source=resolve(args.source),
        model_path=resolve(args.model),
        output=resolve(args.output),
        variants=args.variants,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()

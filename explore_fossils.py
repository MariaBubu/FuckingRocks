#!/usr/bin/env python3
"""
Exploratory fossil classifier pipeline.

What it does:
1. Finds RAW/JPEG/PNG images in class folders such as:
   ../FuckingFossils/FuckingCorals/*.RAF
   ../FuckingFossils/FuckingShells/*.RAF
   ../FuckingFossils/FuckingLeftovers/*.RAF
2. Converts RAW .RAF files to JPEG while preserving the class-folder layout.
3. Optionally creates preprocessed duplicates for quick experiments.
4. Trains a small ResNet18 image classifier using torchvision ImageFolder.

This is intentionally a first-pass exploration script, not the final thesis model.
It treats each whole image as one class, which is useful for testing photo signal
quality but weaker than object detection if each photo contains several fossils.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageEnhance, ImageOps


RAW_EXTENSIONS = {".raf", ".RAF"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_SOURCE = Path("../FuckingFossils")
DEFAULT_OUTPUT = Path("data/exploratory")


@dataclass(frozen=True)
class ImageItem:
    path: Path
    class_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert sorted fossil RAW folders and train a quick classifier."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Folder containing class subfolders with RAW/images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output folder for converted images, variants, model, and reports.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["convert", "preprocess", "train", "all"],
        default=["all"],
        help="Pipeline stages to run.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["original", "enhanced", "grayscale", "clahe"],
        default=["original", "enhanced"],
        help="Image variants to create/use. 'clahe' requires opencv-python.",
    )
    parser.add_argument(
        "--include-classes",
        nargs="*",
        default=None,
        help="Optional class folder names to include, e.g. FuckingCorals FuckingShells.",
    )
    parser.add_argument(
        "--ignore-classes",
        nargs="*",
        default=[],
        help="Class folder names to ignore.",
    )
    parser.add_argument(
        "--long-side",
        type=int,
        default=1600,
        help="Resize converted images so their longest side is this many pixels.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not use pretrained ImageNet weights. Useful without internet/cache.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate converted/preprocessed outputs even if files already exist.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def discover_items(
    source: Path, include_classes: list[str] | None, ignore_classes: list[str]
) -> list[ImageItem]:
    if not source.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source}")

    ignored = set(ignore_classes)
    included = set(include_classes) if include_classes else None
    items: list[ImageItem] = []

    for class_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        class_name = class_dir.name
        if class_name in ignored:
            continue
        if included is not None and class_name not in included:
            continue

        for path in sorted(class_dir.iterdir()):
            if path.suffix in RAW_EXTENSIONS or path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append(ImageItem(path=path, class_name=class_name))

    return items


def resize_long_side(image: Image.Image, long_side: int) -> Image.Image:
    width, height = image.size
    current_long_side = max(width, height)
    if current_long_side <= long_side:
        return image

    scale = long_side / current_long_side
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def convert_raw(path: Path) -> Image.Image:
    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError(
            "rawpy is required to convert .RAF files. Install it with:\n"
            "  python3 -m pip install rawpy"
        ) from exc

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_bps=8,
        )
    return Image.fromarray(rgb)


def open_source_image(path: Path) -> Image.Image:
    if path.suffix in RAW_EXTENSIONS:
        return convert_raw(path)
    return Image.open(path).convert("RGB")


def convert_images(
    items: Iterable[ImageItem],
    converted_dir: Path,
    long_side: int,
    jpeg_quality: int,
    force: bool,
) -> list[ImageItem]:
    items = list(items)
    converted_items: list[ImageItem] = []
    manifest_rows: list[dict[str, str]] = []

    for index, item in enumerate(items, start=1):
        class_dir = converted_dir / item.class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        destination = class_dir / f"{item.path.stem}.jpg"

        if force or not destination.exists():
            print(
                f"[convert {index}/{len(items)}] {item.class_name}/{item.path.name} -> "
                f"{destination.name}",
                flush=True,
            )
            image = open_source_image(item.path)
            image = resize_long_side(image.convert("RGB"), long_side)
            image.save(destination, quality=jpeg_quality, optimize=True)
        else:
            print(
                f"[convert {index}/{len(items)}] skipping existing {destination.name}",
                flush=True,
            )

        converted_items.append(ImageItem(path=destination, class_name=item.class_name))
        manifest_rows.append(
            {
                "source": str(item.path),
                "converted": str(destination),
                "class": item.class_name,
            }
        )

    manifest_path = converted_dir.parent / "conversion_manifest.csv"
    with manifest_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["source", "converted", "class"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    return converted_items


def variant_original(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def variant_enhanced(image: Image.Image) -> Image.Image:
    image = ImageOps.autocontrast(image.convert("RGB"), cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(1.25)
    image = ImageEnhance.Sharpness(image).enhance(1.15)
    return image


def variant_grayscale(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    return ImageOps.autocontrast(gray, cutoff=1).convert("RGB")


def variant_clahe(image: Image.Image) -> Image.Image:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "The 'clahe' variant requires opencv-python and numpy. Install with:\n"
            "  python3 -m pip install opencv-python numpy"
        ) from exc

    rgb = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    merged = cv2.merge((lightness, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)


VARIANT_FUNCTIONS = {
    "original": variant_original,
    "enhanced": variant_enhanced,
    "grayscale": variant_grayscale,
    "clahe": variant_clahe,
}


def preprocess_images(
    converted_dir: Path,
    dataset_dir: Path,
    variants: list[str],
    jpeg_quality: int,
    force: bool,
) -> None:
    if dataset_dir.exists() and force:
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        image_path
        for class_dir in sorted(p for p in converted_dir.iterdir() if p.is_dir())
        for image_path in sorted(class_dir.glob("*.jpg"))
    ]

    total_steps = len(image_paths) * len(variants)
    step = 0

    for class_dir in sorted(p for p in converted_dir.iterdir() if p.is_dir()):
        output_class_dir = dataset_dir / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(class_dir.glob("*.jpg")):
            image = Image.open(image_path).convert("RGB")
            for variant in variants:
                step += 1
                destination = output_class_dir / f"{image_path.stem}__{variant}.jpg"
                if destination.exists() and not force:
                    print(
                        f"[preprocess {step}/{total_steps}] skipping existing "
                        f"{destination.name}",
                        flush=True,
                    )
                    continue
                print(
                    f"[preprocess {step}/{total_steps}] {image_path.parent.name}/"
                    f"{image_path.name} -> {destination.name}",
                    flush=True,
                )
                variant_image = VARIANT_FUNCTIONS[variant](image)
                variant_image.save(destination, quality=jpeg_quality, optimize=True)


def train_classifier(
    dataset_dir: Path,
    model_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_ratio: float,
    seed: int,
    no_pretrained: bool,
) -> None:
    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise RuntimeError(
            "Training requires torch and torchvision. Install with:\n"
            "  python3 -m pip install torch torchvision"
        ) from exc

    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    base_dataset = datasets.ImageFolder(dataset_dir)
    if len(base_dataset.classes) < 2:
        raise RuntimeError("Need at least two class folders to train a classifier.")

    groups_by_class: dict[int, dict[str, list[int]]] = {}
    for index, (sample_path, class_index) in enumerate(base_dataset.samples):
        path = Path(sample_path)
        original_stem = path.stem.split("__", 1)[0]
        group_key = f"{path.parent.name}/{original_stem}"
        groups_by_class.setdefault(class_index, {}).setdefault(group_key, []).append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_index, groups in sorted(groups_by_class.items()):
        group_items = list(groups.items())
        random.shuffle(group_items)
        if len(group_items) == 1:
            train_indices.extend(group_items[0][1])
            continue

        val_group_count = max(1, round(len(group_items) * val_ratio))
        val_group_count = min(val_group_count, len(group_items) - 1)
        val_groups = group_items[:val_group_count]
        train_groups = group_items[val_group_count:]

        for _, indices in train_groups:
            train_indices.extend(indices)
        for _, indices in val_groups:
            val_indices.extend(indices)

    if not train_indices or not val_indices:
        raise RuntimeError(
            "Could not create a train/validation split. Add more images per class "
            "or reduce the number of classes."
        )

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    train_dataset = datasets.ImageFolder(dataset_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(dataset_dir, transform=val_transform)
    train_loader = DataLoader(
        Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices), batch_size=batch_size, shuffle=False
    )

    if no_pretrained:
        model = models.resnet18(weights=None)
    else:
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except Exception as exc:
            print(
                "Could not load pretrained ResNet18 weights. Falling back to random "
                f"initialization. Original error: {exc}",
                file=sys.stderr,
            )
            model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, len(base_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_resnet18.pt"

    print(
        f"Training samples: {len(train_indices)} | "
        f"validation samples: {len(val_indices)} | classes: {base_dataset.classes}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += images.size(0)

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        confusion = torch.zeros(len(base_dataset.classes), len(base_dataset.classes), dtype=torch.int64)

        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                predictions = outputs.argmax(dim=1)

                val_loss += loss.item() * images.size(0)
                val_correct += (predictions == labels).sum().item()
                val_total += images.size(0)

                for true_label, predicted_label in zip(labels.cpu(), predictions.cpu()):
                    confusion[true_label, predicted_label] += 1

        train_accuracy = train_correct / max(train_total, 1)
        val_accuracy = val_correct / max(val_total, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_total, 1),
            "train_accuracy": train_accuracy,
            "val_loss": val_loss / max(val_total, 1),
            "val_accuracy": val_accuracy,
        }
        history.append(row)

        print(
            f"epoch {epoch:02d}/{epochs} "
            f"train_acc={train_accuracy:.3f} val_acc={val_accuracy:.3f} "
            f"train_loss={row['train_loss']:.3f} val_loss={row['val_loss']:.3f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": base_dataset.classes,
                    "class_to_idx": base_dataset.class_to_idx,
                    "val_accuracy": val_accuracy,
                    "epoch": epoch,
                },
                best_model_path,
            )

    with (model_dir / "training_history.json").open("w") as json_file:
        json.dump(history, json_file, indent=2)

    with (model_dir / "class_mapping.json").open("w") as json_file:
        json.dump(base_dataset.class_to_idx, json_file, indent=2)

    with (model_dir / "last_confusion_matrix.json").open("w") as json_file:
        json.dump(
            {
                "classes": base_dataset.classes,
                "matrix": confusion.tolist(),
            },
            json_file,
            indent=2,
        )

    print(f"\nClasses: {base_dataset.classes}")
    print(f"Device: {device}")
    print(f"Best validation accuracy: {best_accuracy:.3f}")
    print(f"Saved best model to: {best_model_path}")


def main() -> None:
    args = parse_args()
    stages = {"convert", "preprocess", "train"} if "all" in args.stages else set(args.stages)

    source = resolve_repo_path(args.source)
    output = resolve_repo_path(args.output)
    converted_dir = output / "converted_jpg"
    dataset_dir = output / "classifier_dataset"
    model_dir = output / "model"

    random.seed(args.seed)

    print(f"Source: {source}")
    print(f"Output: {output}")

    if "convert" in stages:
        items = discover_items(source, args.include_classes, args.ignore_classes)
        if not items:
            raise RuntimeError(f"No images found in class folders under {source}")

        counts: dict[str, int] = {}
        for item in items:
            counts[item.class_name] = counts.get(item.class_name, 0) + 1

        print("Discovered images:")
        for class_name, count in sorted(counts.items()):
            print(f"  {class_name}: {count}")

        convert_images(
            items=items,
            converted_dir=converted_dir,
            long_side=args.long_side,
            jpeg_quality=args.jpeg_quality,
            force=args.force,
        )

    if "preprocess" in stages:
        if not converted_dir.exists():
            raise RuntimeError(
                f"Converted images not found at {converted_dir}. Run --stages convert first."
            )
        print(f"Creating variants {args.variants} in {dataset_dir}")
        preprocess_images(
            converted_dir=converted_dir,
            dataset_dir=dataset_dir,
            variants=args.variants,
            jpeg_quality=args.jpeg_quality,
            force=args.force,
        )

    if "train" in stages:
        if not dataset_dir.exists():
            raise RuntimeError(
                f"Classifier dataset not found at {dataset_dir}. Run --stages preprocess first."
            )
        train_classifier(
            dataset_dir=dataset_dir,
            model_dir=model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_ratio=args.val_ratio,
            seed=args.seed,
            no_pretrained=args.no_pretrained,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class SourceImage:
    path: Path
    class_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a crop classifier with split-first, controlled augmentation, and a real test set."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/Users/mariabuliga/Desktop/Thesis/FuckingFossils/BoxingData/CroppedClassifierSorting"),
        help="Folder containing sorted crop class folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/crop_classifier_controlled_aug"),
        help="Output folder for prepared dataset, model, and review files.",
    )
    parser.add_argument("--include-classes", nargs="+", default=["Coral", "Shell"])
    parser.add_argument("--ignore-classes", nargs="*", default=["Bad", "Unsure", "Unsorted"])
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def discover(source: Path, include_classes: list[str], ignore_classes: list[str]) -> list[SourceImage]:
    ignored = set(ignore_classes)
    included = set(include_classes)
    images: list[SourceImage] = []
    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        if class_dir.name in ignored or class_dir.name not in included:
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(SourceImage(path=path, class_name=class_dir.name))
    return images


def split_by_class(
    images: list[SourceImage],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[SourceImage]]:
    rng = random.Random(seed)
    by_class: dict[str, list[SourceImage]] = {}
    for image in images:
        by_class.setdefault(image.class_name, []).append(image)

    splits = {"train": [], "val": [], "test": []}
    for class_name, class_images in sorted(by_class.items()):
        shuffled = class_images[:]
        rng.shuffle(shuffled)
        if len(shuffled) < 3:
            raise RuntimeError(f"Need at least 3 images for class {class_name}.")
        test_count = max(1, round(len(shuffled) * test_ratio))
        val_count = max(1, round(len(shuffled) * val_ratio))
        if test_count + val_count >= len(shuffled):
            val_count = 1
            test_count = 1
        splits["test"].extend(shuffled[:test_count])
        splits["val"].extend(shuffled[test_count:test_count + val_count])
        splits["train"].extend(shuffled[test_count + val_count:])

    for split_images in splits.values():
        rng.shuffle(split_images)
    return splits


def safe_name(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in path.stem)


def augment_once(image: Image.Image, rng: random.Random) -> tuple[str, Image.Image]:
    choice = rng.choice(["lower_contrast", "brighter", "rotate"])
    image = image.convert("RGB")
    if choice == "lower_contrast":
        factor = rng.uniform(0.62, 0.82)
        return f"lower_contrast_{factor:.2f}", ImageEnhance.Contrast(image).enhance(factor)
    if choice == "brighter":
        factor = rng.uniform(1.15, 1.35)
        return f"brighter_{factor:.2f}", ImageEnhance.Brightness(image).enhance(factor)
    angle = rng.choice([90, 180, 270])
    return f"rotate_{angle}", image.rotate(angle, expand=True)


def copy_image(src: Path, dst: Path, jpeg_quality: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    image.save(dst, quality=jpeg_quality, optimize=True)


def prepare_dataset(
    splits: dict[str, list[SourceImage]],
    dataset_dir: Path,
    split_manifest_path: Path,
    jpeg_quality: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []

    for split_name, images in splits.items():
        for image_item in images:
            original_name = f"{safe_name(image_item.path)}__original.jpg"
            destination = dataset_dir / split_name / image_item.class_name / original_name
            copy_image(image_item.path, destination, jpeg_quality)
            rows.append(
                {
                    "split": split_name,
                    "class": image_item.class_name,
                    "source": str(image_item.path),
                    "prepared": str(destination),
                    "variant": "original",
                }
            )

            if split_name == "train":
                image = ImageOps.exif_transpose(Image.open(image_item.path)).convert("RGB")
                variant_name, variant_image = augment_once(image, rng)
                augmented_name = f"{safe_name(image_item.path)}__{variant_name}.jpg"
                augmented_destination = dataset_dir / split_name / image_item.class_name / augmented_name
                augmented_destination.parent.mkdir(parents=True, exist_ok=True)
                variant_image.save(augmented_destination, quality=jpeg_quality, optimize=True)
                rows.append(
                    {
                        "split": split_name,
                        "class": image_item.class_name,
                        "source": str(image_item.path),
                        "prepared": str(augmented_destination),
                        "variant": variant_name,
                    }
                )

    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with split_manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["split", "class", "source", "prepared", "variant"])
        writer.writeheader()
        writer.writerows(rows)


def get_device():
    import torch

    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def train_and_evaluate(
    dataset_dir: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    no_pretrained: bool,
) -> None:
    try:
        import certifi
        import os

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise RuntimeError("Training requires torch and torchvision.") from exc

    random.seed(seed)
    torch.manual_seed(seed)
    device = get_device()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(dataset_dir / "train", transform=transform)
    val_dataset = datasets.ImageFolder(dataset_dir / "val", transform=transform)
    test_dataset = datasets.ImageFolder(dataset_dir / "test", transform=transform)
    classes = train_dataset.classes
    if val_dataset.classes != classes or test_dataset.classes != classes:
        raise RuntimeError("Train/val/test class folders do not match.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_counts = [0] * len(classes)
    for _, class_index in train_dataset.samples:
        class_counts[class_index] += 1
    weights = torch.tensor(
        [sum(class_counts) / max(count, 1) for count in class_counts],
        dtype=torch.float32,
        device=device,
    )

    if no_pretrained:
        print("Creating ResNet18 without pretrained weights", flush=True)
        model = models.resnet18(weights=None)
    else:
        try:
            print("Loading pretrained ResNet18 weights", flush=True)
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            print("Loaded pretrained ResNet18 weights", flush=True)
        except Exception as exc:
            print(f"Could not load pretrained weights, using random init: {exc}", file=sys.stderr, flush=True)
            model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    model_dir = output / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_resnet18.pt"
    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0

    print(
        f"Training samples: {len(train_dataset)} | validation samples: {len(val_dataset)} | "
        f"test samples: {len(test_dataset)} | classes: {classes}",
        flush=True,
    )
    print(f"Training class counts after augmentation: {dict(zip(classes, class_counts))}", flush=True)

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

        val_loss, val_accuracy, val_confusion, _ = evaluate(model, val_loader, classes, criterion, device)
        train_accuracy = train_correct / max(train_total, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_total, 1),
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{epochs} train_acc={train_accuracy:.3f} val_acc={val_accuracy:.3f} "
            f"train_loss={row['train_loss']:.3f} val_loss={val_loss:.3f}",
            flush=True,
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "class_to_idx": train_dataset.class_to_idx,
                    "val_accuracy": val_accuracy,
                    "epoch": epoch,
                },
                best_model_path,
            )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_accuracy, test_confusion, predictions = evaluate(
        model, test_loader, classes, criterion, device, include_predictions=True
    )

    with (model_dir / "training_history.json").open("w", encoding="utf-8") as json_file:
        json.dump(history, json_file, indent=2)
    with (model_dir / "class_mapping.json").open("w", encoding="utf-8") as json_file:
        json.dump(train_dataset.class_to_idx, json_file, indent=2)
    with (model_dir / "last_confusion_matrix.json").open("w", encoding="utf-8") as json_file:
        json.dump({"classes": classes, "matrix": val_confusion}, json_file, indent=2)
    with (model_dir / "test_metrics.json").open("w", encoding="utf-8") as json_file:
        json.dump(
            {
                "classes": classes,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_confusion_matrix": test_confusion,
                "best_val_accuracy": best_val_accuracy,
            },
            json_file,
            indent=2,
        )

    write_predictions(output, predictions)

    print(f"\nClasses: {classes}")
    print(f"Device: {device}")
    print(f"Best validation accuracy: {best_val_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved test review files to: {output / 'test_review'}")


def evaluate(model, loader, classes, criterion, device, include_predictions: bool = False):
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.int64)
    predictions: list[dict[str, str | float]] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            predicted = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct += (predicted == labels).sum().item()
            total += images.size(0)
            for true_label, predicted_label in zip(labels.cpu(), predicted.cpu()):
                confusion[true_label, predicted_label] += 1

            if include_predictions:
                start = len(predictions)
                batch_samples = loader.dataset.samples[start:start + images.size(0)]
                for sample, true_label, predicted_label, probability_row in zip(
                    batch_samples, labels.cpu(), predicted.cpu(), probs.cpu()
                ):
                    confidence = float(probability_row[predicted_label])
                    predictions.append(
                        {
                            "path": sample[0],
                            "true_class": classes[int(true_label)],
                            "predicted_class": classes[int(predicted_label)],
                            "confidence": confidence,
                            "correct": str(int(true_label) == int(predicted_label)),
                        }
                    )

    return total_loss / max(total, 1), correct / max(total, 1), confusion.tolist(), predictions


def write_predictions(output: Path, predictions: list[dict[str, str | float]]) -> None:
    review_dir = output / "test_review"
    correct_dir = review_dir / "correct"
    wrong_dir = review_dir / "wrong"
    correct_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = review_dir / "test_predictions.csv"
    with predictions_csv.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["path", "true_class", "predicted_class", "confidence", "correct"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)

    for row in predictions:
        source = Path(str(row["path"]))
        verdict = "correct" if row["correct"] == "True" else "wrong"
        destination = review_dir / verdict / f"true-{row['true_class']}__pred-{row['predicted_class']}__{source.name}"
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    source = resolve(args.source)
    output = resolve(args.output)
    dataset_dir = output / "dataset"
    if args.force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    images = discover(source, args.include_classes, args.ignore_classes)
    if not images:
        raise RuntimeError(f"No images found in {source}.")

    counts: dict[str, int] = {}
    for image in images:
        counts[image.class_name] = counts.get(image.class_name, 0) + 1
    print(f"Source: {source}")
    print(f"Output: {output}")
    print(f"Discovered images: {counts}")

    splits = split_by_class(images, args.val_ratio, args.test_ratio, args.seed)
    print("Split counts:")
    for split_name, split_images in splits.items():
        split_counts: dict[str, int] = {}
        for image in split_images:
            split_counts[image.class_name] = split_counts.get(image.class_name, 0) + 1
        print(f"  {split_name}: {len(split_images)} originals {split_counts}")

    prepare_dataset(
        splits=splits,
        dataset_dir=dataset_dir,
        split_manifest_path=output / "split_manifest.csv",
        jpeg_quality=args.jpeg_quality,
        seed=args.seed,
    )
    train_and_evaluate(
        dataset_dir=dataset_dir,
        output=output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        no_pretrained=args.no_pretrained,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate saved ResNet18 fossil models without importing torchvision.

This is a small fallback evaluator for cases where torchvision import is slow or
stuck. The architecture below matches torchvision.models.resnet18 closely enough
to load the saved state dict from explore_fossils.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("../FuckingFossils")
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/preprocessing_experiments/original_only/model/best_resnet18.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/preprocessing_experiments/original_only/wild_test_updated_no_torchvision"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def class_from_folder(folder_name: str) -> str | None:
    lowered = folder_name.lower()
    if "coral" in lowered:
        return "FuckingCorals"
    if "shell" in lowered:
        return "FuckingShells"
    return None


def discover_images(source: Path) -> list[tuple[Path, str]]:
    items = []
    for folder in sorted(path for path in source.iterdir() if path.is_dir()):
        true_class = class_from_folder(folder.name)
        if true_class is None or "wild" not in folder.name.lower():
            continue
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((image_path, true_class))
    return items


def image_to_tensor(image_path: Path) -> Tensor:
    image = Image.open(image_path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.view(224, 224, 3).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (data - mean) / std


def main() -> None:
    args = parse_args()
    source = resolve(args.source)
    model_path = resolve(args.model)
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(model_path, map_location="cpu")
    classes = checkpoint["classes"]
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}

    model = ResNet18(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows = []
    confusion = [[0 for _ in classes] for _ in classes]
    correct = 0
    total = 0

    with torch.no_grad():
        for image_path, true_class in discover_images(source):
            tensor = image_to_tensor(image_path).unsqueeze(0)
            probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
            predicted_idx = int(probabilities.argmax().item())
            predicted_class = classes[predicted_idx]
            confidence = float(probabilities[predicted_idx].item())
            true_idx = class_to_idx[true_class]
            is_correct = predicted_class == true_class

            total += 1
            correct += int(is_correct)
            confusion[true_idx][predicted_idx] += 1

            row = {
                "image": str(image_path),
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "correct": str(is_correct),
            }
            rows.append(row)
            print(
                f"{image_path.parent.name}/{image_path.name} "
                f"true={true_class} predicted={predicted_class} "
                f"confidence={confidence:.3f} correct={is_correct}",
                flush=True,
            )

    accuracy = correct / total if total else 0.0
    predictions_path = output / "wild_predictions.csv"
    with predictions_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image", "true_class", "predicted_class", "confidence", "correct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "classes": classes,
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "source": str(source),
        "model": str(model_path),
    }
    summary_path = output / "wild_summary.json"
    with summary_path.open("w") as json_file:
        json.dump(summary, json_file, indent=2)

    print("\nWild test summary")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Predictions: {predictions_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

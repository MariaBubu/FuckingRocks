#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
BOXING_ROOT = ROOT.parent / "FuckingFossils" / "BoxingData"
OUTPUT_ROOT = BOXING_ROOT / "CroppedClassifierSorting"
UNSORTED_DIR = OUTPUT_ROOT / "Unsorted"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PADDING_FRACTION = 0.08


@dataclass(frozen=True)
class SourceGroup:
    name: str
    image_dir: Path
    labels_dir: Path


SOURCE_GROUPS = [
    SourceGroup(
        name="approved",
        image_dir=BOXING_ROOT / "BoxingApproved",
        labels_dir=BOXING_ROOT / "boxingapproved_yolo_labels",
    ),
    SourceGroup(
        name="classifier_only",
        image_dir=BOXING_ROOT / "ClassifierOnly",
        labels_dir=BOXING_ROOT / "labels_still_thesis_2026-06-02-03-12-29",
    ),
]


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def images_by_stem(folder: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def yolo_to_pixel_box(line: str, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        _class_id, x_center, y_center, width, height = [float(part) for part in parts]
    except ValueError:
        return None

    box_width = width * image_width
    box_height = height * image_height
    xmin = (x_center * image_width) - (box_width / 2)
    ymin = (y_center * image_height) - (box_height / 2)
    xmax = xmin + box_width
    ymax = ymin + box_height

    pad_x = box_width * PADDING_FRACTION
    pad_y = box_height * PADDING_FRACTION
    xmin = max(0, round(xmin - pad_x))
    ymin = max(0, round(ymin - pad_y))
    xmax = min(image_width, round(xmax + pad_x))
    ymax = min(image_height, round(ymax + pad_y))

    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def prepare_output_dirs() -> None:
    shutil.rmtree(UNSORTED_DIR, ignore_errors=True)
    UNSORTED_DIR.mkdir(parents=True, exist_ok=True)
    for folder_name in ["Coral", "Shell", "Unsure", "Bad"]:
        (OUTPUT_ROOT / folder_name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    prepare_output_dirs()

    rows: list[dict[str, str | int | float]] = []
    skipped: list[dict[str, str]] = []
    crop_count = 0

    for group in SOURCE_GROUPS:
        if not group.image_dir.exists():
            raise FileNotFoundError(f"Missing image folder: {group.image_dir}")
        if not group.labels_dir.exists():
            raise FileNotFoundError(f"Missing label folder: {group.labels_dir}")

        image_lookup = images_by_stem(group.image_dir)
        for label_path in sorted(group.labels_dir.glob("*.txt")):
            image_path = image_lookup.get(label_path.stem)
            if image_path is None:
                skipped.append({
                    "group": group.name,
                    "label": str(label_path),
                    "reason": "No matching image with same stem",
                })
                continue

            try:
                with Image.open(image_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
            except Exception as exc:
                skipped.append({
                    "group": group.name,
                    "image": str(image_path),
                    "reason": f"Could not open image: {exc}",
                })
                continue

            label_lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for box_index, line in enumerate(label_lines, start=1):
                box = yolo_to_pixel_box(line, image.width, image.height)
                if box is None:
                    skipped.append({
                        "group": group.name,
                        "label": str(label_path),
                        "reason": f"Invalid label line {box_index}: {line}",
                    })
                    continue

                xmin, ymin, xmax, ymax = box
                crop = image.crop((xmin, ymin, xmax, ymax))
                crop_name = f"{group.name}__{safe_stem(image_path.stem)}__box{box_index:03d}.jpg"
                crop_path = UNSORTED_DIR / crop_name
                crop.save(crop_path, format="JPEG", quality=95, optimize=True)
                crop_count += 1

                rows.append({
                    "crop_file": crop_name,
                    "source_group": group.name,
                    "source_image": str(image_path),
                    "source_label": str(label_path),
                    "box_index": box_index,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "crop_width": xmax - xmin,
                    "crop_height": ymax - ymin,
                    "padding_fraction": PADDING_FRACTION,
                })

    manifest_path = OUTPUT_ROOT / "crop_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "crop_file",
            "source_group",
            "source_image",
            "source_label",
            "box_index",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "crop_width",
            "crop_height",
            "padding_fraction",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    skipped_path = OUTPUT_ROOT / "skipped_crops.csv"
    with skipped_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["group", "image", "label", "reason"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(skipped)

    print(f"Created {crop_count} crops")
    print(f"Unsorted crops: {UNSORTED_DIR}")
    print(f"Manifest: {manifest_path}")
    print(f"Skipped rows: {len(skipped)} ({skipped_path})")


if __name__ == "__main__":
    main()

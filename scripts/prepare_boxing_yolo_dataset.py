#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT.parent / "FuckingFossils" / "BoxingData"
APPROVED_DIR = SOURCE_ROOT / "BoxingApproved"
CLASSIFIER_ONLY_DIR = SOURCE_ROOT / "ClassifierOnly"
NOISE_DIR = SOURCE_ROOT / "TileNoise"
APPROVED_LABELS_DIR = SOURCE_ROOT / "boxingapproved_yolo_labels"
CLASSIFIER_ONLY_LABELS_DIR = SOURCE_ROOT / "labels_still_thesis_2026-06-02-03-12-29"
OUTPUT_DIR = ROOT / "data" / "boxing_yolo_combined"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
RANDOM_SEED = 42
SKIP_IMAGES = {
    "TileNoise/20240913_170152.jpg",
    "TileNoise/Screenshot 2026-06-01 at 18.14.24.png",
}


@dataclass(frozen=True)
class Item:
    source_image: Path
    source_label: Path | None
    kind: str
    source_group: str


def safe_name(path: Path, prefix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")
    return f"{prefix}_{stem}.jpg"


def image_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and p.relative_to(SOURCE_ROOT).as_posix() not in SKIP_IMAGES
    )


def validate_label_file(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{line_number} does not have 5 columns")
            continue
        try:
            values = [float(part) for part in parts]
        except ValueError:
            errors.append(f"{path.name}:{line_number} has a non-number value")
            continue
        if values[0] != 0:
            errors.append(f"{path.name}:{line_number} class id is not 0")
        if any(value < 0 or value > 1 for value in values[1:]):
            errors.append(f"{path.name}:{line_number} YOLO coordinate outside 0..1")
    return len(lines), errors


def split_items(items: list[Item]) -> dict[str, list[Item]]:
    shuffled = items[:]
    random.Random(RANDOM_SEED).shuffle(shuffled)
    n = len(shuffled)
    train_end = max(1, round(n * 0.70))
    val_end = train_end + max(1, round(n * 0.15))
    if val_end >= n:
        val_end = max(train_end, n - 1)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def convert_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial_target = target.with_suffix(target.suffix + ".partial")
    partial_target.unlink(missing_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.save(partial_target, format="JPEG", quality=95, optimize=True)
    partial_target.replace(target)


def convert_image_with_timeout(source: Path, target: Path) -> None:
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--convert-one", str(source), str(target)],
        check=True,
        timeout=20,
        capture_output=True,
        text=True,
    )


def copy_split(split_name: str, items: list[Item], manifest: list[dict], skipped: list[dict]) -> int:
    images_out = OUTPUT_DIR / "images" / split_name
    labels_out = OUTPUT_DIR / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    box_count = 0
    for item in items:
        prefix = f"pos_{item.source_group}" if item.kind == "fossil" else "neg_tile_noise"
        image_name = safe_name(item.source_image, prefix)
        label_name = f"{Path(image_name).stem}.txt"
        image_target = images_out / image_name
        label_target = labels_out / label_name

        print(f"  {split_name}: {image_name}", flush=True)
        try:
            convert_image_with_timeout(item.source_image, image_target)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            skipped.append({
                "split": split_name,
                "source_image": str(item.source_image),
                "reason": str(error),
            })
            image_target.unlink(missing_ok=True)
            image_target.with_suffix(image_target.suffix + ".partial").unlink(missing_ok=True)
            continue

        if item.source_label is None:
            label_target.write_text("", encoding="utf-8")
            boxes = 0
        else:
            label_text = item.source_label.read_text(encoding="utf-8")
            label_target.write_text(label_text, encoding="utf-8")
            boxes = len([line for line in label_text.splitlines() if line.strip()])

        box_count += boxes
        manifest.append({
            "split": split_name,
            "kind": item.kind,
            "source_group": item.source_group,
            "boxes": boxes,
            "source_image": str(item.source_image),
            "image": str(image_target.relative_to(OUTPUT_DIR)),
            "label": str(label_target.relative_to(OUTPUT_DIR)),
        })
    return box_count


def main() -> None:
    positive_groups = [
        ("approved", APPROVED_DIR, APPROVED_LABELS_DIR),
        ("classifier_only", CLASSIFIER_ONLY_DIR, CLASSIFIER_ONLY_LABELS_DIR),
    ]

    for group_name, image_dir, labels_dir in positive_groups:
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing {group_name} image folder: {image_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Missing {group_name} YOLO labels folder: {labels_dir}")
    if not NOISE_DIR.exists():
        raise FileNotFoundError(f"Missing tile-noise folder: {NOISE_DIR}")

    noise_images = image_files(NOISE_DIR)

    labeled_items: list[Item] = []
    positive_group_summaries: dict[str, dict] = {}
    total_boxes = 0
    validation_errors: list[str] = []

    for group_name, image_dir, labels_dir in positive_groups:
        positive_images = image_files(image_dir)
        label_by_stem = {path.stem: path for path in sorted(labels_dir.glob("*.txt"))}
        group_boxes = 0
        group_items = 0
        missing_or_empty_labels: list[str] = []

        for image in positive_images:
            label = label_by_stem.get(image.stem)
            if label is None:
                missing_or_empty_labels.append(image.name)
                continue
            boxes, errors = validate_label_file(label)
            group_boxes += boxes
            validation_errors.extend(errors)
            if boxes == 0:
                missing_or_empty_labels.append(image.name)
                continue
            labeled_items.append(
                Item(
                    source_image=image,
                    source_label=label,
                    kind="fossil",
                    source_group=group_name,
                )
            )
            group_items += 1

        total_boxes += group_boxes
        positive_group_summaries[group_name] = {
            "image_dir": str(image_dir),
            "labels_dir": str(labels_dir),
            "images": len(positive_images),
            "labeled_fossil_images_used": group_items,
            "total_fossil_boxes": group_boxes,
            "missing_or_empty_label_images": missing_or_empty_labels,
        }

    if validation_errors:
        raise ValueError("Invalid YOLO labels:\n" + "\n".join(validation_errors[:20]))

    noise_items = [
        Item(source_image=image, source_label=None, kind="tile_noise", source_group="tile_noise")
        for image in noise_images
    ]

    for split in ["train", "val", "test"]:
        shutil.rmtree(OUTPUT_DIR / "images" / split, ignore_errors=True)
        shutil.rmtree(OUTPUT_DIR / "labels" / split, ignore_errors=True)
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    labeled_splits = split_items(labeled_items)
    noise_splits = split_items(noise_items)

    manifest: list[dict] = []
    summary = {
        "source_root": str(SOURCE_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "positive_groups": positive_group_summaries,
        "labeled_fossil_images_used": len(labeled_items),
        "tile_noise_images_used": len(noise_items),
        "total_fossil_boxes": total_boxes,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        items = labeled_splits[split] + noise_splits[split]
        random.Random(RANDOM_SEED + len(split)).shuffle(items)
        skipped: list[dict] = []
        print(f"Copying {split}: {len(items)} images", flush=True)
        boxes = copy_split(split, items, manifest, skipped)
        skipped_sources = {skip["source_image"] for skip in skipped}
        summary["splits"][split] = {
            "images": len(items) - len(skipped),
            "fossil_images": sum(
                item.kind == "fossil" and str(item.source_image) not in skipped_sources
                for item in items
            ),
            "tile_noise_images": sum(
                item.kind == "tile_noise" and str(item.source_image) not in skipped_sources
                for item in items
            ),
            "boxes": boxes,
            "skipped_copy_failures": skipped,
        }

    (OUTPUT_DIR / "dataset.yaml").write_text(
        "\n".join([
            f"path: {OUTPUT_DIR}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: fossil",
            "",
        ]),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--convert-one":
        convert_image(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main()

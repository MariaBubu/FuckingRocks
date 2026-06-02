#!/usr/bin/env python3
"""
Run a small preprocessing shootout for the fossil classifier.

This script compares several saved preprocessing variants:

1. original
2. original + enhanced
3. original + grayscale
4. original + clahe
5. original + enhanced + grayscale + clahe

Each experiment trains on the same coral/shell source classes, then evaluates
the resulting model on the WildCorals/WildShells folders only. The wild images
are never used for training.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

from explore_fossils import discover_items, convert_images, preprocess_images, train_classifier


DEFAULT_BASE_OUTPUT = Path("data/exploratory")
DEFAULT_EXPERIMENT_ROOT = Path("data/preprocessing_experiments")
DEFAULT_SOURCE = Path("../FuckingFossils")
DEFAULT_CLASSES = ["FuckingCorals", "FuckingShells"]

EXPERIMENTS = [
    ("original_only", ["original"]),
    ("original_enhanced", ["original", "enhanced"]),
    ("original_grayscale", ["original", "grayscale"]),
    ("original_clahe", ["original", "clahe"]),
    ("all_variants", ["original", "enhanced", "grayscale", "clahe"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate multiple preprocessing variant combinations."
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--base-output", type=Path, default=DEFAULT_BASE_OUTPUT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--include-classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument(
        "--wild-variants",
        nargs="+",
        choices=["original", "enhanced"],
        default=["original"],
        help="Variants to use for wild testing. Default keeps the test set clean.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate each experiment output folder before running.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def ensure_base_converted(source: Path, base_output: Path, include_classes: list[str]) -> None:
    converted_dir = base_output / "converted_jpg"
    if converted_dir.exists() and any(converted_dir.glob("*/*.jpg")):
        print(f"Using existing converted JPEGs from {converted_dir}", flush=True)
        return

    print(f"Creating base converted JPEGs in {converted_dir}", flush=True)
    items = discover_items(source, include_classes, [])
    convert_images(
        items=items,
        converted_dir=converted_dir,
        long_side=1600,
        jpeg_quality=92,
        force=False,
    )


def copy_converted_images(base_output: Path, experiment_output: Path) -> None:
    source = base_output / "converted_jpg"
    destination = experiment_output / "converted_jpg"
    if destination.exists():
        if destination.is_symlink():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.symlink_to(source, target_is_directory=True)

    for _ in range(30):
        if destination.exists() and destination.is_dir():
            return
        time.sleep(0.2)
    raise RuntimeError(f"Converted image symlink was not ready: {destination}")


def read_best_validation(model_dir: Path) -> float:
    history_path = model_dir / "training_history.json"
    with history_path.open() as json_file:
        history = json.load(json_file)
    return max(row["val_accuracy"] for row in history)


def read_wild_summary(wild_dir: Path) -> dict:
    with (wild_dir / "wild_summary.json").open() as json_file:
        return json.load(json_file)


def main() -> None:
    args = parse_args()
    source = resolve(args.source)
    base_output = resolve(args.base_output)
    experiment_root = resolve(args.experiment_root)
    experiment_root.mkdir(parents=True, exist_ok=True)

    ensure_base_converted(source, base_output, args.include_classes)

    rows: list[dict[str, str | int | float]] = []

    for experiment_name, variants in EXPERIMENTS:
        print("\n" + "=" * 72, flush=True)
        print(f"Experiment: {experiment_name} | variants={variants}", flush=True)
        print("=" * 72, flush=True)

        experiment_output = experiment_root / experiment_name
        if args.force and experiment_output.exists():
            shutil.rmtree(experiment_output)
        experiment_output.mkdir(parents=True, exist_ok=True)

        copy_converted_images(base_output, experiment_output)

        print(f"Preprocessing variants into {experiment_output / 'classifier_dataset'}", flush=True)
        preprocess_images(
            converted_dir=experiment_output / "converted_jpg",
            dataset_dir=experiment_output / "classifier_dataset",
            variants=variants,
            jpeg_quality=92,
            force=True,
        )

        print(f"Training model for {experiment_name}", flush=True)
        train_classifier(
            dataset_dir=experiment_output / "classifier_dataset",
            model_dir=experiment_output / "model",
            epochs=args.epochs,
            batch_size=8,
            learning_rate=1e-3,
            val_ratio=0.2,
            seed=42,
            no_pretrained=False,
        )

        wild_output = experiment_output / "wild_test_original_only"
        from evaluate_wild_fossils import evaluate_wild

        wild_summary = evaluate_wild(
            source=source,
            model_path=experiment_output / "model" / "best_resnet18.pt",
            output=wild_output,
            variants=args.wild_variants,
        )

        best_validation = read_best_validation(experiment_output / "model")
        rows.append(
            {
                "experiment": experiment_name,
                "train_variants": "+".join(variants),
                "epochs": args.epochs,
                "best_validation_accuracy": best_validation,
                "wild_accuracy": wild_summary["accuracy"],
                "wild_correct": wild_summary["correct"],
                "wild_total": wild_summary["total"],
            }
        )

    summary_path = experiment_root / "preprocessing_experiment_summary.csv"
    with summary_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "experiment",
                "train_variants",
                "epochs",
                "best_validation_accuracy",
                "wild_accuracy",
                "wild_correct",
                "wild_total",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nPreprocessing experiment summary")
    for row in rows:
        print(
            f"{row['experiment']}: val={row['best_validation_accuracy']:.3f} "
            f"wild={row['wild_accuracy']:.3f} "
            f"({row['wild_correct']}/{row['wild_total']})"
        )
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()

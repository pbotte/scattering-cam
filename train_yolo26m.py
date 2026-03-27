#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO, settings


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO26m model from a YOLO-format dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Path to the YOLO dataset directory. Default: dataset",
    )
    parser.add_argument(
        "--model",
        default="yolo26m.pt",
        help="Model weights or model name to train. Default: yolo26m.pt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs. Default: 100",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size. Default: 640",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size. Use -1 for auto batch. Default: 16",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, e.g. cpu, 0, 0,1. Default: Ultralytics auto-select",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers. Default: 8",
    )
    parser.add_argument(
        "--project",
        default="runs/train",
        help="Output project directory for Ultralytics runs. Default: runs/train",
    )
    parser.add_argument(
        "--name",
        default="yolo26m-custom",
        help="Run name. Default: yolo26m-custom",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience in epochs. Default: 30",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio for datasets without predefined splits. Default: 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val split. Default: 42",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow reusing an existing output run directory.",
    )
    return parser.parse_args()


def load_class_names(dataset_dir: Path) -> list[str]:
    classes_path = dataset_dir / "classes.txt"
    if not classes_path.is_file():
        raise FileNotFoundError(f"Missing classes file: {classes_path}")

    class_names = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]
    if not class_names:
        raise ValueError(f"No class names found in {classes_path}")
    return class_names


def collect_images(dataset_dir: Path) -> list[Path]:
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    images = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No images found in {images_dir}")

    missing_labels = [path for path in images if not (labels_dir / f"{path.stem}.txt").is_file()]
    if missing_labels:
        missing_str = ", ".join(path.name for path in missing_labels[:5])
        raise FileNotFoundError(f"Missing label files for: {missing_str}")

    return images


def split_images(images: list[Path], val_split: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("--val-split must be between 0 and 1")
    if len(images) < 2:
        raise ValueError("Need at least 2 images to build train/val splits")

    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * val_split)))
    val_count = min(val_count, len(shuffled) - 1)

    val_images = sorted(shuffled[:val_count])
    train_images = sorted(shuffled[val_count:])
    return train_images, val_images


def write_image_list(file_path: Path, images: list[Path]) -> None:
    file_path.write_text("".join(f"{image.resolve()}\n" for image in images))


def write_data_yaml(
    file_path: Path,
    train_list_path: Path,
    val_list_path: Path,
    class_names: list[str],
) -> None:
    payload = {
        "path": str(file_path.parent.resolve()),
        "train": str(train_list_path.resolve()),
        "val": str(val_list_path.resolve()),
        "names": {index: name for index, name in enumerate(class_names)},
    }
    file_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()

    try:
        class_names = load_class_names(dataset_dir)
        images = collect_images(dataset_dir)
        train_images, val_images = split_images(images, args.val_split, args.seed)
    except Exception as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 1

    generated_dir = dataset_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    train_list_path = generated_dir / "train.txt"
    val_list_path = generated_dir / "val.txt"
    data_yaml_path = generated_dir / "data.yaml"

    write_image_list(train_list_path, train_images)
    write_image_list(val_list_path, val_images)
    write_data_yaml(data_yaml_path, train_list_path, val_list_path, class_names)

    print(f"Prepared dataset config: {data_yaml_path}")
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")

    try:
        settings.update({"sync": False})
        model = YOLO(args.model)
        train_kwargs = {
            "data": str(data_yaml_path),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "project": args.project,
            "name": args.name,
            "patience": args.patience,
            "exist_ok": args.exist_ok,
            "seed": args.seed,
        }
        if args.device is not None:
            train_kwargs["device"] = args.device

        results = model.train(**train_kwargs)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    save_dir = getattr(results, "save_dir", None)
    if save_dir is not None:
        print(f"Training finished. Results saved to: {save_dir}")
    else:
        print("Training finished.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

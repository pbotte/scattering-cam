#!/usr/bin/env python3
import argparse
import importlib.util
import os
import sys
from importlib import metadata
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a YOLO .pt model to TensorFlow.js using Ultralytics."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/detect/runs/train/yolo26m-custom/weights/best.pt"),
        help="Path to the trained YOLO .pt weights file.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export image size. Default: 640",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Export batch size. Default: 1",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device, e.g. cpu, 0, mps. Default: cpu",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export with FP16 where supported.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Export with INT8 quantization where supported.",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Include NMS in the exported model if supported.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=20,
        help="ONNX opset used by the Ultralytics TensorFlow export path. Default: 20",
    )
    return parser.parse_args()


def ensure_local_ultralytics_config(project_root: Path) -> None:
    config_dir = project_root / ".ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))


def require_module(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(install_hint)


def require_distribution(distribution_name: str, install_hint: str) -> None:
    try:
        metadata.version(distribution_name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(install_hint) from exc


def build_dependency_hint() -> str:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    lines = [
        "Missing export dependencies.",
        (
            "Required packages for the Ultralytics tfjs export path: ultralytics, tensorflow, tensorflowjs, "
            "onnx, onnx2tf, tf_keras, sng4onnx, onnx_graphsurgeon, ai-edge-litert, onnxslim, onnxruntime, protobuf."
        ),
    ]
    if sys.version_info >= (3, 13):
        lines.extend(
            [
                f"Current Python version: {version}",
                "The Python tensorflowjs export stack is commonly not resolvable on Python 3.13 yet.",
                "Use a dedicated Python 3.11 or 3.12 virtual environment for the export step.",
            ]
        )
    else:
        lines.append(
            "Install the missing packages in your current virtual environment and rerun the export."
        )
    return " ".join(lines)


def validate_python_version() -> None:
    version = sys.version_info
    current = f"{version.major}.{version.minor}.{version.micro}"
    if version < (3, 10):
        raise RuntimeError(
            "TensorFlow.js export is not supported in this script with Python "
            f"{current}. Use a dedicated Python 3.12 environment for the export step."
        )


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    ensure_local_ultralytics_config(project_root)

    model_path = args.model.resolve()
    if not model_path.is_file():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1

    try:
        validate_python_version()
    except Exception as exc:
        print(f"Environment check failed: {exc}", file=sys.stderr)
        return 1

    dependency_hint = build_dependency_hint()
    try:
        require_module("ultralytics", dependency_hint)
        require_module("tensorflow", dependency_hint)
        require_module("tensorflowjs", dependency_hint)
        require_distribution("onnx", dependency_hint)
        require_distribution("onnx2tf", dependency_hint)
        require_distribution("tf_keras", dependency_hint)
        require_distribution("sng4onnx", dependency_hint)
        require_distribution("onnx_graphsurgeon", dependency_hint)
        require_distribution("ai-edge-litert", dependency_hint)
        require_distribution("onnxslim", dependency_hint)
        require_distribution("onnxruntime", dependency_hint)
        require_distribution("protobuf", dependency_hint)
        from ultralytics import YOLO, settings
    except Exception as exc:
        print(f"Dependency check failed: {exc}", file=sys.stderr)
        return 1

    try:
        settings.update({"sync": False})
        model = YOLO(str(model_path))
        exported_path = model.export(
            format="tfjs",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            half=args.half,
            int8=args.int8,
            nms=args.nms,
            opset=args.opset,
        )
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1

    print("TensorFlow.js export completed.")
    print(f"Model: {model_path}")
    print(f"Export: {exported_path}")
    print("Open the generated *_web_model directory in your HTML app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

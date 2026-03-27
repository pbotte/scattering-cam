#!/usr/bin/env python3
import argparse
import sys
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLOE, settings


def parse_prompt_values(prompt_args: list[str]) -> list[str]:
    prompts: list[str] = []
    for raw_value in prompt_args:
        for part in raw_value.split(","):
            prompt = part.strip()
            if prompt:
                prompts.append(prompt)
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live camera detection with a YOLOE model and custom text prompts."
    )
    parser.add_argument(
        "--model",
        default="yoloe-11s-seg.pt",
        help="Path to the YOLOE weights file. Default: yoloe-11s-seg.pt",
    )
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        required=True,
        help=(
            "Text prompt to detect. Repeat the flag for multiple prompts or pass a comma-separated list, "
            'for example: --prompt person --prompt bus or --prompt "person,bus"'
        ),
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index. Default: 0",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections. Default: 0.25",
    )
    parser.add_argument(
        "--mode",
        choices=("predict", "track"),
        default="predict",
        help="Inference mode. Use 'track' for object tracking. Default: predict",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Tracker config for track mode. Default: bytetrack.yaml",
    )
    parser.add_argument(
        "--track-history-seconds",
        type=float,
        default=3.0,
        help="How many seconds of track history to draw in track mode. Default: 3.0",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. Default: 640",
    )
    return parser.parse_args()


def configure_text_prompt(model: YOLOE, prompts: list[str]) -> None:
    try:
        model.set_classes(prompts, model.get_text_pe(prompts))
    except Exception as exc:
        raise RuntimeError(
            "YOLOE text prompts could not be initialized. "
            "Make sure the required YOLOE text-model dependencies are installed. "
            "Ultralytics typically expects the GitHub 'clip' package rather than the PyPI package named 'clip'. "
            f"Original error: {exc}"
        ) from exc


def main() -> int:
    args = parse_args()
    prompts = parse_prompt_values(args.prompts)
    if not prompts:
        print("At least one non-empty text prompt is required.", file=sys.stderr)
        return 1

    track_history: dict[int, deque[tuple[float, tuple[int, int]]]] = defaultdict(deque)

    try:
        settings.update({"sync": False})
        model = YOLOE(args.model)
        configure_text_prompt(model, prompts)
    except Exception as exc:
        print(f"Model could not be loaded or configured: {exc}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(
            f"Camera {args.camera_index} could not be opened.",
            file=sys.stderr,
        )
        return 1

    window_name = f"YOLOE Live Detection [{', '.join(prompts)}]"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame could not be read from camera.", file=sys.stderr)
                return 1

            if args.mode == "track":
                result = model.track(
                    source=frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    tracker=args.tracker,
                    persist=True,
                    verbose=False,
                )[0]
            else:
                result = model.predict(
                    source=frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )[0]
            annotated = result.plot()

            if args.mode == "track" and result.boxes is not None and result.boxes.id is not None:
                now = time.monotonic()
                track_ids = result.boxes.id.int().cpu().tolist()
                centers = result.boxes.xywh[:, :2].int().cpu().tolist()

                for track_id, center in zip(track_ids, centers):
                    history = track_history[track_id]
                    history.append((now, (center[0], center[1])))
                    while history and now - history[0][0] > args.track_history_seconds:
                        history.popleft()

                stale_track_ids = []
                for track_id, history in track_history.items():
                    while history and now - history[0][0] > args.track_history_seconds:
                        history.popleft()
                    if not history:
                        stale_track_ids.append(track_id)
                        continue

                    points = [point for _, point in history]
                    if len(points) > 1:
                        cv2.polylines(
                            annotated,
                            [np.array(points, dtype=np.int32)],
                            isClosed=False,
                            color=(0, 255, 255),
                            thickness=2,
                        )

                for track_id in stale_track_ids:
                    del track_history[track_id]

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import sys
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO, settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live camera detection with a YOLO26m model and OpenCV input."
    )
    parser.add_argument(
        "--model",
        default="yolo26m.pt",
        help="Path to the YOLO26m weights file. Default: yolo26m.pt",
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


def main() -> int:
    args = parse_args()
    track_history: dict[int, deque[tuple[float, tuple[int, int]]]] = defaultdict(deque)

    try:
        settings.update({"sync": False})
        model = YOLO(args.model)
    except Exception as exc:
        print(f"Model could not be loaded: {exc}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(
            f"Camera {args.camera_index} could not be opened.",
            file=sys.stderr,
        )
        return 1

    window_name = "YOLO26m Live Detection"

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

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import cv2

from uniface.constants import YOLOv8FaceWeights
from uniface.detection import YOLOv8Face
from uniface.model_store import set_cache_dir


LANDMARK_NAMES = (
    "left_eye",
    "right_eye",
    "nose",
    "left_mouth",
    "right_mouth",
)


def open_camera(index: int) -> cv2.VideoCapture:
    """Open the webcam with a sensible backend for the current OS."""
    system = platform.system().lower()

    if system == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    elif system == "windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)

    return cap


def draw_face(frame, face) -> None:
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    confidence = float(face.confidence)

    # Face bounding box and confidence
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"face {confidence:.2f}",
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Five landmarks: left eye, right eye, nose, left mouth, right mouth
    if face.landmarks is not None:
        for name, point in zip(LANDMARK_NAMES, face.landmarks):
            px, py = int(point[0]), int(point[1])
            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                name,
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live webcam face detection with YOLOv8-Lite-S + 5 facial landmarks"
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="Minimum face confidence, 0..1 (default: 0.50)",
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Requested webcam width (default: 1280)"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Requested webcam height (default: 720)"
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not mirror the webcam preview",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not (0.0 <= args.confidence <= 1.0):
        print("Error: --confidence must be between 0 and 1.")
        return 2

    project_dir = Path(__file__).resolve().parent
    model_dir = project_dir / "models"
    model_dir.mkdir(exist_ok=True)

    # Keep the automatically downloaded ONNX model inside this project.
    set_cache_dir(str(model_dir))

    print("Loading YOLOv8-Lite-S face model...")
    print("On the first run, UniFace will download the ~7.4 MB model into ./models")

    detector = YOLOv8Face(
        model_name=YOLOv8FaceWeights.YOLOV8_LITE_S,
        confidence_threshold=args.confidence,
        nms_threshold=0.45,
        nms_mode="numpy",
        providers=["CPUExecutionProvider"],
    )

    cap = open_camera(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        if sys.platform == "darwin":
            print(
                "On macOS, allow camera access for Terminal/Python in "
                "System Settings > Privacy & Security > Camera."
            )
        print("You can also try another camera, for example: python app.py --camera 1")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "YOLOv8-Lite-S Face + 5 Landmarks | Q / ESC = quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    print("Webcam started. Press Q or ESC to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Failed to read a frame from the webcam.")
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            faces = detector.detect(frame)

            for face in faces:
                draw_face(frame, face)

            now = time.perf_counter()
            dt = max(now - previous_time, 1e-6)
            current_fps = 1.0 / dt
            previous_time = now
            smoothed_fps = current_fps if smoothed_fps == 0 else (0.9 * smoothed_fps + 0.1 * current_fps)

            cv2.putText(
                frame,
                f"Faces: {len(faces)} | FPS: {smoothed_fps:.1f}",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

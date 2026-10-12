"""Run with python -m squat_coach."""
import argparse
from .app import run


def main():
    parser = argparse.ArgumentParser(description="Real-time AI squat repetition coach")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--model", default=None, help="Optional path to MediaPipe .task model")
    args = parser.parse_args()
    run(camera_index=args.camera, model_path=args.model)


if __name__ == "__main__":
    main()

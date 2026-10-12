"""Download/load the MediaPipe pose model."""
from pathlib import Path
from urllib.request import urlretrieve
from .config import MODEL_PATH, MODEL_URL


def ensure_model(path=MODEL_PATH):
    path = Path(path).expanduser()
    if path.is_file() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading MediaPipe pose model...")
    try:
        urlretrieve(MODEL_URL, path)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Downloaded model was empty")
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError("Could not download pose model. Check your connection.") from exc
    return path


def create_landmarker(model_path):
    import mediapipe as mp
    return mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

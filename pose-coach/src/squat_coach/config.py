"""Configuration and MediaPipe landmark indices."""
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
MODEL_PATH = Path.home() / ".cache" / "ai-squat-pose-coach" / "pose_landmarker_lite.task"
UP_ANGLE = 160.0
DOWN_ANGLE = 100.0
LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27
RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28

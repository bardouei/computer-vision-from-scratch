from pathlib import Path

from uniface.constants import YOLOv8FaceWeights
from uniface.detection import YOLOv8Face
from uniface.model_store import set_cache_dir

project_dir = Path(__file__).resolve().parent
model_dir = project_dir / "models"
model_dir.mkdir(exist_ok=True)
set_cache_dir(str(model_dir))

print("Downloading/checking YOLOv8-Lite-S (~7.4 MB)...")
YOLOv8Face(
    model_name=YOLOv8FaceWeights.YOLOV8_LITE_S,
    nms_mode="numpy",
    providers=["CPUExecutionProvider"],
)
print(f"Done. Model cache: {model_dir}")

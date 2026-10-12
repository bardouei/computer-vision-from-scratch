"""Camera loop and application orchestration."""
import time
import cv2
import mediapipe as mp
from .config import LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
from .counter import SquatCounter
from .geometry import calculate_angle
from .model import create_landmarker, ensure_model
from .overlay import draw_info_panel, draw_pose


def run(camera_index=0, model_path=None):
    model_path = ensure_model() if model_path is None else ensure_model(model_path)
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open webcam. Check permissions and camera availability.")

    counter = SquatCounter()
    start_time = time.perf_counter()
    last_timestamp_ms = -1
    print("Pose Coach started. Stand sideways to camera. Press Q to quit.")
    try:
        with create_landmarker(model_path) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read frame from webcam.")
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(int((time.perf_counter() - start_time) * 1000),
                                   last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(image, timestamp_ms)
                knee_angle = None
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    hip, knee, ankle = (landmarks[i] for i in (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE))
                    knee_angle = calculate_angle((hip.x, hip.y), (knee.x, knee.y),
                                                 (ankle.x, ankle.y))
                    draw_pose(frame, hip, knee, ankle, knee_angle)
                counter.update(knee_angle)
                draw_info_panel(frame, counter, knee_angle)
                cv2.imshow("AI Squat Pose Coach", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

"""
AI Squat Pose Coach - Single File Version

Install dependencies:
    pip install opencv-python mediapipe numpy

Run:
    python squat_pose_coach.py

Press Q to quit.

The MediaPipe pose model is downloaded automatically on first run.
"""

import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# Configuration
# ============================================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

MODEL_PATH = "pose_landmarker_lite.task"

UP_ANGLE = 160
DOWN_ANGLE = 100


# ============================================================
# Utilities
# ============================================================

def download_model_if_needed():
    """Download MediaPipe pose model if it does not exist."""
    if os.path.exists(MODEL_PATH):
        return

    print("Pose model not found.")
    print("Downloading MediaPipe pose model...")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")
    except Exception as exc:
        raise RuntimeError(
            "Could not download the pose model.\n"
            "Check your internet connection and run the program again."
        ) from exc


def calculate_angle(a, b, c):
    """
    Calculate angle ABC in degrees.

    a, b, c are [x, y] coordinates.
    b is the vertex of the angle.
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)

    if ba_norm == 0 or bc_norm == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle_radians = np.arccos(cosine_angle)

    return float(np.degrees(angle_radians))


def to_pixel(landmark, width, height):
    """Convert normalized landmark coordinates to pixel coordinates."""
    return (
        int(landmark.x * width),
        int(landmark.y * height),
    )


def draw_joint(frame, point, color=(0, 255, 255), radius=8):
    cv2.circle(frame, point, radius, color, -1)


def draw_info_panel(frame, squat_count, state, feedback, knee_angle=None):
    """Draw simple status panel."""
    cv2.rectangle(
        frame,
        (20, 20),
        (390, 205),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        frame,
        f"Squats: {squat_count}",
        (40, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"State: {state}",
        (40, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    angle_text = "--" if knee_angle is None else str(int(knee_angle))

    cv2.putText(
        frame,
        f"Knee angle: {angle_text} deg",
        (40, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        feedback,
        (40, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )


# ============================================================
# Main program
# ============================================================

def main():
    download_model_if_needed()

    # MediaPipe Tasks API shortcuts
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH,
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the webcam. "
            "Make sure your camera is connected and not used by another app."
        )

    squat_count = 0
    state = "UP"
    reached_bottom = False

    start_time = time.perf_counter()
    last_timestamp_ms = -1

    # MediaPipe landmark indices
    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27

    print("Pose Coach started.")
    print("Stand sideways to the camera for better results.")
    print("Press Q to quit.")

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:

            while True:
                success, frame = cap.read()

                if not success:
                    print("Could not read frame from webcam.")
                    break

                # Mirror view
                frame = cv2.flip(frame, 1)

                height, width, _ = frame.shape

                # OpenCV -> RGB
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )

                # VIDEO mode requires monotonically increasing timestamps
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1

                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                feedback = "Stand sideways in the camera"
                knee_angle = None

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]

                    hip_lm = landmarks[LEFT_HIP]
                    knee_lm = landmarks[LEFT_KNEE]
                    ankle_lm = landmarks[LEFT_ANKLE]

                    hip = [hip_lm.x, hip_lm.y]
                    knee = [knee_lm.x, knee_lm.y]
                    ankle = [ankle_lm.x, ankle_lm.y]

                    knee_angle = calculate_angle(
                        hip,
                        knee,
                        ankle,
                    )

                    hip_px = to_pixel(
                        hip_lm,
                        width,
                        height,
                    )

                    knee_px = to_pixel(
                        knee_lm,
                        width,
                        height,
                    )

                    ankle_px = to_pixel(
                        ankle_lm,
                        width,
                        height,
                    )

                    # Draw leg skeleton
                    cv2.line(
                        frame,
                        hip_px,
                        knee_px,
                        (255, 255, 255),
                        4,
                    )

                    cv2.line(
                        frame,
                        knee_px,
                        ankle_px,
                        (255, 255, 255),
                        4,
                    )

                    draw_joint(
                        frame,
                        hip_px,
                        (0, 255, 255),
                    )

                    draw_joint(
                        frame,
                        knee_px,
                        (0, 255, 0),
                        10,
                    )

                    draw_joint(
                        frame,
                        ankle_px,
                        (0, 255, 255),
                    )

                    # ------------------------------------------------
                    # Squat state machine
                    # ------------------------------------------------

                    if knee_angle > UP_ANGLE:
                        state = "UP"

                        if reached_bottom:
                            squat_count += 1
                            reached_bottom = False

                        feedback = "Ready"

                    elif knee_angle < DOWN_ANGLE:
                        state = "DOWN"
                        reached_bottom = True
                        feedback = "Good depth!"

                    else:
                        state = "MOVING"

                        if reached_bottom:
                            feedback = "Come up"
                        else:
                            feedback = "Go lower"

                    # Draw angle next to knee
                    cv2.putText(
                        frame,
                        f"{int(knee_angle)} deg",
                        (
                            knee_px[0] + 20,
                            knee_px[1],
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                draw_info_panel(
                    frame,
                    squat_count,
                    state,
                    feedback,
                    knee_angle,
                )

                cv2.imshow(
                    "AI Squat Pose Coach",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

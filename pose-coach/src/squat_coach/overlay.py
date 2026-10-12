"""OpenCV overlay rendering."""
import cv2
from .geometry import to_pixel


def draw_joint(frame, point, color=(0, 255, 255), radius=8):
    cv2.circle(frame, point, radius, color, -1)


def draw_pose(frame, hip, knee, ankle, angle):
    height, width = frame.shape[:2]
    points = [to_pixel(p, width, height) for p in (hip, knee, ankle)]
    cv2.line(frame, points[0], points[1], (255, 255, 255), 4)
    cv2.line(frame, points[1], points[2], (255, 255, 255), 4)
    draw_joint(frame, points[0])
    draw_joint(frame, points[1], (0, 255, 0), 10)
    draw_joint(frame, points[2])
    if angle is not None:
        cv2.putText(frame, f"{int(angle)} deg", (points[1][0] + 20, points[1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def draw_info_panel(frame, counter, knee_angle):
    cv2.rectangle(frame, (20, 20), (390, 205), (20, 20, 20), -1)
    labels = [
        (f"Squats: {counter.count}", 65, 1.0),
        (f"State: {counter.state}", 105, 0.75),
        (f"Knee angle: {'--' if knee_angle is None else int(knee_angle)} deg", 145, 0.75),
        (counter.feedback, 185, 0.65),
    ]
    for label, y, scale in labels:
        color = (0, 255, 0) if y == 185 else (255, 255, 255)
        cv2.putText(frame, label, (40, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

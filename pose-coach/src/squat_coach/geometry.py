"""Pure geometry utilities."""
import math


def calculate_angle(a, b, c):
    """Return angle ABC (degrees), or None for a degenerate angle."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    denominator = math.hypot(*ba) * math.hypot(*bc)
    if denominator == 0:
        return None
    cosine = (ba[0] * bc[0] + ba[1] * bc[1]) / denominator
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def to_pixel(landmark, width, height):
    """Convert a normalized MediaPipe landmark into pixel coordinates."""
    return int(landmark.x * width), int(landmark.y * height)

"""Squat repetition state machine (independent of camera and ML)."""
from dataclasses import dataclass
from .config import UP_ANGLE, DOWN_ANGLE


@dataclass
class SquatCounter:
    up_angle: float = UP_ANGLE
    down_angle: float = DOWN_ANGLE
    count: int = 0
    state: str = "UP"
    reached_bottom: bool = False
    feedback: str = "Stand sideways in the camera"

    def __post_init__(self):
        if self.down_angle >= self.up_angle:
            raise ValueError("down_angle must be less than up_angle")

    def update(self, angle):
        """Count a rep when returning upright after reaching squat depth.

        Missing/invalid measurements do not modify the state.
        """
        if angle is None or not 0 <= angle <= 180:
            self.feedback = "Move into camera view"
        elif angle > self.up_angle:
            self.state = "UP"
            if self.reached_bottom:
                self.count += 1
                self.reached_bottom = False
            self.feedback = "Ready"
        elif angle < self.down_angle:
            self.state = "DOWN"
            self.reached_bottom = True
            self.feedback = "Good depth!"
        else:
            self.state = "MOVING"
            self.feedback = "Come up" if self.reached_bottom else "Go lower"
        return self

import unittest
from squat_coach.geometry import calculate_angle


class AngleTests(unittest.TestCase):
    def test_right_angle(self):
        self.assertAlmostEqual(calculate_angle((1, 0), (0, 0), (0, 1)), 90)

    def test_straight_angle(self):
        self.assertAlmostEqual(calculate_angle((-1, 0), (0, 0), (1, 0)), 180)

    def test_degenerate_angle(self):
        self.assertIsNone(calculate_angle((0, 0), (0, 0), (1, 0)))

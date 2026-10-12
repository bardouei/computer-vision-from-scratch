import unittest
from squat_coach.counter import SquatCounter


class CounterTests(unittest.TestCase):
    def test_complete_repetition(self):
        counter = SquatCounter()
        for angle in (170, 130, 90, 130, 170):
            counter.update(angle)
        self.assertEqual(counter.count, 1)
        self.assertEqual(counter.state, "UP")

    def test_no_count_without_bottom(self):
        counter = SquatCounter()
        for angle in (170, 125, 170):
            counter.update(angle)
        self.assertEqual(counter.count, 0)

    def test_no_double_count(self):
        counter = SquatCounter()
        for angle in (90, 170, 170, 170):
            counter.update(angle)
        self.assertEqual(counter.count, 1)

    def test_missing_detection_does_not_count(self):
        counter = SquatCounter()
        for angle in (90, None, 170):
            counter.update(angle)
        self.assertEqual(counter.count, 1)

    def test_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            SquatCounter(up_angle=90, down_angle=100)

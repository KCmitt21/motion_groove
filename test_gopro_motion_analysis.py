import unittest

import numpy as np

from motion_groove.gopro_motion_analysis import (
    beat_synchronisation,
    beats_from_bpm,
    robust_standardize,
    signed_forward_lean_deg,
)


class MotionMathTests(unittest.TestCase):
    def test_front_forward_lean_is_positive(self):
        hip = np.array([0.0, 0.0, 0.0])
        shoulder = np.array([0.0, -1.0, -1.0])
        self.assertAlmostEqual(
            signed_forward_lean_deg(shoulder, hip, "front"), 45.0
        )

    def test_side_view_direction(self):
        hip = np.array([0.0, 0.0, 0.0])
        shoulder = np.array([1.0, -1.0, 0.0])
        self.assertAlmostEqual(
            signed_forward_lean_deg(shoulder, hip, "side-right"), 45.0
        )
        self.assertAlmostEqual(
            signed_forward_lean_deg(shoulder, hip, "side-left"), -45.0
        )

    def test_perfect_beat_sync(self):
        beats = beats_from_bpm(120, 2.0, 0.0)
        result = beat_synchronisation(
            np.array([0.0, 0.5, 1.0, 1.5]), beats
        )
        self.assertAlmostEqual(result["beat_sync_score"], 1.0)
        self.assertAlmostEqual(result["mean_abs_beat_error_ms"], 0.0)
        self.assertAlmostEqual(result["on_beat_fraction_100ms"], 1.0)

    def test_low_motion_is_not_treated_as_an_event(self):
        energy = robust_standardize(np.array([0.0, 1.0, 1.0, 2.0]))
        self.assertEqual(energy[0], 0.0)
        self.assertGreater(energy[-1], 0.0)


if __name__ == "__main__":
    unittest.main()

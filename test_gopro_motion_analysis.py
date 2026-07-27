import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from gopro_motion_analysis import (
    assign_default_gopro_recording_path,
    beat_synchronisation,
    beats_from_bpm,
    build_parser,
    prepare_stream_url,
    robust_standardize,
    signed_forward_lean_deg,
    validate_args,
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

    def test_usb_cli_defaults_to_rtsp(self):
        args = build_parser().parse_args(["--gopro-usb", "--bpm", "120"])
        validate_args(args)
        self.assertTrue(args.gopro_usb)
        self.assertEqual(args.gopro_protocol, "RTSP")

    def test_default_outputs_are_under_out_directory(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.output, Path("out/motion_metrics.csv"))
        self.assertEqual(args.summary, Path("out/motion_summary.json"))

    def test_usb_run_gets_timestamped_recording_under_out_directory(self):
        args = build_parser().parse_args(["--gopro-usb"])
        path = assign_default_gopro_recording_path(
            args, datetime(2026, 7, 27, 12, 34, 56)
        )
        self.assertEqual(path, Path("out/gopro_capture_20260727_123456.mp4"))
        self.assertEqual(args.record_video, path)

    def test_explicit_recording_path_is_preserved(self):
        args = build_parser().parse_args(
            ["--gopro-usb", "--record-video", "out/my_session.mp4"]
        )
        path = assign_default_gopro_recording_path(
            args, datetime(2026, 7, 27, 12, 34, 56)
        )
        self.assertEqual(path, Path("out/my_session.mp4"))

    def test_udp_source_gets_gopro_receive_buffer_options(self):
        self.assertEqual(
            prepare_stream_url("udp://0.0.0.0:8554"),
            "udp://0.0.0.0:8554?overrun_nonfatal=1&fifo_size=50000000",
        )


if __name__ == "__main__":
    unittest.main()

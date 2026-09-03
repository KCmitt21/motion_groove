from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from musician_interaction.audio import AudioFeatures
from musician_interaction.face import FaceHeadPoseEstimator, infer_pose_guided_faces, pose_face_roi
from musician_interaction.features import lagged_correlation, positions_to_motion
from musician_interaction.pose import PerformerTracker
from musician_interaction.qc import estimate_sync_offsets
from musician_interaction.triangulation import reprojection_error, triangulate_dlt
from musician_interaction.types import HeadPose, PoseDetection
from musician_interaction.video import FPS, SynchronizedVideoReader, time_sec


class TimeAndVideoTests(unittest.TestCase):
    def test_exact_time_formula(self):
        self.assertEqual(time_sec(0), 0.0)
        self.assertAlmostEqual(time_sec(30000), 1001.0)

    def test_reader_stops_at_shortest(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name, count in (("a", 4), ("b", 3), ("c", 5)):
                path = Path(directory) / f"{name}.avi"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), float(FPS), (32, 24))
                for idx in range(count):
                    writer.write(np.full((24, 32, 3), idx, np.uint8))
                writer.release()
                paths[name] = path
            with SynchronizedVideoReader(paths) as reader:
                batches = list(reader.frames())
            self.assertEqual(len(batches), 3)
            self.assertEqual([item[0] for item in batches], [0, 1, 2])


class TrackingTests(unittest.TestCase):
    @staticmethod
    def detection(x: float) -> PoseDetection:
        points = np.tile([x, 50.0], (17, 1))
        return PoseDetection(points, np.ones(17))

    def test_initial_semantic_order_and_temporal_assignment(self):
        tracker = PerformerTracker(["bassist", "guitarist"], ["bassist", "guitarist"], .3)
        first = tracker.assign([self.detection(80), self.detection(20)])
        self.assertEqual(first["bassist"].center[0], 20)
        second = tracker.assign([self.detection(77), self.detection(23)])
        self.assertEqual(second["bassist"].center[0], 23)


class FaceRoiTests(unittest.TestCase):
    @staticmethod
    def face_detection() -> PoseDetection:
        points = np.full((133, 2), np.nan)
        scores = np.zeros(133)
        points[23:27] = np.array([[90, 40], [110, 40], [90, 60], [110, 60]], dtype=float)
        scores[23:27] = 1.0
        return PoseDetection(points, scores)

    def test_roi_uses_wholebody_face_points_and_padding(self):
        roi = pose_face_roi(self.face_detection(), (100, 200, 3), .3, scale=2.5, min_size_px=20)
        self.assertEqual(roi, (75, 25, 125, 75))

    def test_model_axis_correction_makes_frontal_pose_neutral(self):
        yaw, pitch, roll = FaceHeadPoseEstimator._rotation_to_euler(
            FaceHeadPoseEstimator.MODEL_TO_CAMERA_AXES @ FaceHeadPoseEstimator.MODEL_TO_CAMERA_AXES
        )
        np.testing.assert_allclose([yaw, pitch, roll], [0, 0, 0], atol=1e-10)

    def test_roi_inference_retries_and_preserves_identity(self):
        class FakeEstimator:
            def __init__(self):
                self.calls = []

            def infer_roi(self, frame, roi, output_size):
                self.calls.append((roi, output_size))
                if len(self.calls) == 1:
                    return []
                x0, y0, x1, y1 = roi
                return [HeadPose(center=np.array([(x0 + x1) / 2, (y0 + y1) / 2]))]

        estimator = FakeEstimator()
        state = {}
        result = infer_pose_guided_faces(
            estimator,
            np.zeros((100, 200, 3), dtype=np.uint8),
            {"guitarist": self.face_detection()},
            .3,
            {"enabled": True, "scale": 2.5, "retry_scales": [3.5], "min_size_px": 20,
             "output_size": 256, "reuse_frames": 5},
            state,
        )
        self.assertIsNotNone(result["guitarist"])
        self.assertEqual(len(estimator.calls), 2)
        self.assertEqual(estimator.calls[0][1], 256)
        self.assertIn("guitarist", state)


class FeatureTests(unittest.TestCase):
    def test_velocity_does_not_bridge_missing_frame(self):
        table = pd.DataFrame([
            {"camera": "c", "performer": "p", "feature": "head", "frame_idx": 0, "time_sec": 0,
             "x": 0., "y": 0., "frame_width": 100, "frame_height": 100},
            {"camera": "c", "performer": "p", "feature": "head", "frame_idx": 1, "time_sec": 1,
             "x": 1., "y": 0., "frame_width": 100, "frame_height": 100},
            {"camera": "c", "performer": "p", "feature": "head", "frame_idx": 3, "time_sec": 3,
             "x": 3., "y": 0., "frame_width": 100, "frame_height": 100},
        ])
        result = positions_to_motion(table, 10)
        self.assertAlmostEqual(result.iloc[1]["speed_px_s"], 10)
        self.assertTrue(np.isnan(result.iloc[2]["speed_px_s"]))

    def test_lag_sign(self):
        rng = np.random.default_rng(2)
        first = rng.normal(size=200)
        second = np.r_[np.full(3, np.nan), first[:-3]]
        result = lagged_correlation(first, second, 8)
        best = result.loc[result["correlation"].idxmax()]
        self.assertEqual(best["lag_frames"], 3)

    def test_sync_qc_handles_empty_audio(self):
        empty = np.array([], dtype=float)
        audio = AudioFeatures(empty, empty, empty, empty, 48000)
        result = estimate_sync_offsets({"a": audio, "b": audio}, "a")
        self.assertEqual(len(result), 2)
        self.assertTrue(result["audio_offset_sec"].isna().all())


class TriangulationTests(unittest.TestCase):
    def test_dlt_and_reprojection(self):
        projections = {
            "a": np.array([[100, 0, 50, 0], [0, 100, 50, 0], [0, 0, 1, 0]], float),
            "b": np.array([[100, 0, 50, -100], [0, 100, 50, 0], [0, 0, 1, 0]], float),
        }
        expected = np.array([0.2, -0.1, 3.0])
        observations = {}
        for camera, matrix in projections.items():
            projected = matrix @ np.r_[expected, 1]
            observations[camera] = projected[:2] / projected[2]
        actual = triangulate_dlt(observations, projections)
        np.testing.assert_allclose(actual, expected, atol=1e-8)
        self.assertLess(reprojection_error(actual, observations["a"], projections["a"]), 1e-8)


if __name__ == "__main__":
    unittest.main()

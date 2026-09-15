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
from musician_interaction.gaze import classify_gaze_detailed
from musician_interaction.head_direction import (
    head_direction_summary, head_direction_timeline, mutual_facing_timeline,
    partner_orientation_intervals, reclassify_reference_angles, unknown_reason_summary,
)
from musician_interaction.laeo import angle_degrees, laeo_metrics, smooth_laeo_scores
from musician_interaction.pose import PerformerTracker
from musician_interaction.qc import estimate_sync_offsets
from musician_interaction.triangulation import reprojection_error, triangulate_dlt
from musician_interaction.types import HeadPose, InstrumentPose, PoseDetection
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


class GazeClassificationTests(unittest.TestCase):
    @staticmethod
    def pose(x: float = 0.0) -> PoseDetection:
        return PoseDetection(np.tile([x, 0.0], (17, 1)), np.ones(17))

    def test_wide_partner_uses_performer_reference_angle(self):
        config = {
            "reference_angles": {
                "cam_wide": {
                    "bassist": {
                        "partner": {
                            "yaw_deg": -40.0, "pitch_deg": 5.0,
                            "yaw_tolerance_deg": 20.0, "pitch_tolerance_deg": 10.0,
                        }
                    }
                }
            }
        }
        # The partner is on the right, which would produce the opposite sign in the old image-bearing method.
        head = HeadPose(yaw=-42.0, pitch=6.0, roll=0.0, center=np.array([0.0, 0.0]))
        label, score, reason = classify_gaze_detailed(
            head, self.pose(0), self.pose(100), InstrumentPose(), config, "cam_wide", "bassist"
        )
        self.assertEqual(label, "partner")
        self.assertGreater(score, 0.0)
        self.assertIsNone(reason)

    def test_reference_failure_and_pose_outlier_have_explicit_reasons(self):
        reference_config = {"reference_angles": {"cam_wide": {"bassist": {
            "partner": {"yaw_deg": -40.0, "pitch_deg": 0.0,
                        "yaw_tolerance_deg": 10.0, "pitch_tolerance_deg": 10.0}
        }}}}
        label, _, reason = classify_gaze_detailed(
            HeadPose(yaw=35.0, pitch=0.0, roll=0.0), self.pose(), self.pose(100),
            InstrumentPose(), reference_config, "cam_wide", "bassist",
        )
        self.assertEqual((label, reason), ("unknown", "outside_reference_tolerance"))

        quality_config = {"head_pose_quality": {"max_abs_pitch_deg": 75.0}}
        label, _, reason = classify_gaze_detailed(
            HeadPose(yaw=0.0, pitch=90.0, roll=0.0), self.pose(), None,
            InstrumentPose(), quality_config, "cam_wide", "bassist",
        )
        self.assertEqual((label, reason), ("unknown", "head_pose_angle_outlier"))

    def test_closest_reference_state_wins(self):
        config = {"reference_angles": {"cam_wide": {"bassist": {
            "partner": {"yaw_deg": -40.0, "pitch_deg": 0.0,
                        "yaw_tolerance_deg": 30.0, "pitch_tolerance_deg": 20.0},
            "forward": {"yaw_deg": -10.0, "pitch_deg": 0.0,
                        "yaw_tolerance_deg": 30.0, "pitch_tolerance_deg": 20.0},
        }}}}
        label, _, reason = classify_gaze_detailed(
            HeadPose(yaw=-14.0, pitch=1.0, roll=0.0), self.pose(), self.pose(100),
            InstrumentPose(), config, "cam_wide", "bassist",
        )
        self.assertEqual(label, "forward")
        self.assertIsNone(reason)


class LaeoTests(unittest.TestCase):
    @staticmethod
    def head(position, direction) -> HeadPose:
        return HeadPose(
            position_3d=np.asarray(position, dtype=float),
            gaze_direction_3d=np.asarray(direction, dtype=float),
        )

    def test_angle_and_mutual_score_use_partner_directions(self):
        head_a = self.head([0, 0, 0], [1, 0, 0])
        head_b = self.head([10, 0, 0], [-1, 0, 0])
        result = laeo_metrics(head_a, head_b, sigma_deg=25)
        self.assertAlmostEqual(angle_degrees([1, 0, 0], [0, 1, 0]), 90.0)
        self.assertAlmostEqual(result["theta_A"], 0.0)
        self.assertAlmostEqual(result["theta_B"], 0.0)
        self.assertAlmostEqual(result["p_A"], 1.0)
        self.assertAlmostEqual(result["p_B"], 1.0)
        self.assertAlmostEqual(result["laeo_score"], 1.0)

    def test_gaussian_probability_and_missing_pose(self):
        direction_60_deg = [0.5, np.sqrt(3) / 2, 0]
        result = laeo_metrics(
            self.head([0, 0, 0], direction_60_deg),
            self.head([1, 0, 0], [-1, 0, 0]),
            sigma_deg=30,
        )
        self.assertAlmostEqual(result["theta_A"], 60.0)
        self.assertAlmostEqual(result["p_A"], np.exp(-2.0))
        self.assertAlmostEqual(result["laeo_score"], np.exp(-2.0))
        self.assertTrue(np.isnan(laeo_metrics(None, None, 30)["laeo_score"]))

    def test_centered_smoothing_preserves_missing_track_boundaries(self):
        table = pd.DataFrame({
            "frame_idx": [0, 1, 2, 3, 4],
            "camera": ["wide"] * 5,
            "performer_A": ["a"] * 5,
            "performer_B": ["b"] * 5,
            "laeo_score": [0.0, 0.0, np.nan, 1.0, 1.0],
        })
        result = smooth_laeo_scores(
            table, {"method": "moving_average", "radius_frames": 1}
        )
        np.testing.assert_allclose(
            result["laeo_score_smoothed"].to_numpy(),
            [0.0, 0.0, np.nan, 1.0, 1.0],
            equal_nan=True,
        )


class HeadDirectionReportTests(unittest.TestCase):
    @staticmethod
    def source_table() -> pd.DataFrame:
        rows = []
        directions = {
            0: ("partner", "partner"),
            1: ("partner", "forward"),
            2: ("unknown", "partner"),
        }
        for frame_idx, pair in directions.items():
            for performer, direction in zip(("bassist", "guitarist"), pair):
                detected = not (performer == "bassist" and frame_idx == 2)
                rows.append({
                    "frame_idx": frame_idx, "time_sec": frame_idx / 30,
                    "camera": "cam_wide", "performer": performer,
                    "yaw_deg": 0.0 if detected else np.nan,
                    "pitch_deg": 0.0 if detected else np.nan,
                    "roll_deg": 0.0 if detected else np.nan,
                    "face_score": 1.0 if detected else np.nan,
                    "gaze_target": direction, "gaze_score": 1.0 if detected else np.nan,
                })
        return pd.DataFrame(rows)

    def test_mutual_and_one_sided_partner_frames(self):
        timeline = head_direction_timeline(self.source_table())
        mutual = mutual_facing_timeline(timeline)
        self.assertEqual(mutual["mutual_facing"].tolist(), [True, False, False])
        self.assertEqual(mutual["one_sided_partner"].tolist(), [False, True, True])
        intervals = partner_orientation_intervals(mutual)
        self.assertEqual(intervals["orientation"].tolist(), ["mutual", "bassist_only", "guitarist_only"])

    def test_summary_contains_zero_count_direction_classes(self):
        timeline = head_direction_timeline(self.source_table())
        summary = head_direction_summary(timeline)
        bassist = summary[summary["performer"].eq("bassist")]
        self.assertEqual(set(bassist["head_direction"]), {
            "partner", "own_instrument", "forward", "downward", "unknown"
        })
        own = bassist[bassist["head_direction"].eq("own_instrument")].iloc[0]
        self.assertEqual(own["frames"], 0)

    def test_unknown_reason_is_preserved_and_summarized(self):
        source = self.source_table()
        source["unknown_reason"] = None
        source.loc[source["gaze_target"].eq("unknown"), "unknown_reason"] = "face_not_detected"
        timeline = head_direction_timeline(source)
        summary = unknown_reason_summary(timeline)
        selected = summary[
            summary["camera"].eq("cam_wide")
            & summary["performer"].eq("bassist")
            & summary["unknown_reason"].eq("face_not_detected")
        ].iloc[0]
        self.assertEqual(selected["frames"], 1)
        self.assertEqual(selected["percent_of_unknown_frames"], 100.0)

    def test_existing_head_table_can_be_reclassified_from_config(self):
        source = self.source_table()
        source.loc[
            source["performer"].eq("bassist") & source["frame_idx"].eq(1),
            ["yaw_deg", "pitch_deg", "roll_deg"],
        ] = [-42.0, 3.0, 0.0]
        config = {"reference_angles": {"cam_wide": {"bassist": {"partner": {
            "yaw_deg": -40.0, "pitch_deg": 2.0,
            "yaw_tolerance_deg": 10.0, "pitch_tolerance_deg": 10.0,
        }}}}}
        result = reclassify_reference_angles(source, config)
        selected = result[result["performer"].eq("bassist") & result["frame_idx"].eq(1)].iloc[0]
        self.assertEqual(selected["gaze_target"], "partner")
        self.assertTrue(pd.isna(selected["unknown_reason"]))


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

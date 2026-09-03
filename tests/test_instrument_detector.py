from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from instrument_detector.manage import (
    InstrumentPoseDetection, PoseSmoother, infer_video, init_dataset, select_best_poses,
)
from musician_interaction.features import instrument_motion_events, instrument_pose_motion
from musician_interaction.instruments import InstrumentKeypointStore
from musician_interaction.types import INSTRUMENT_POINTS


class InstrumentDetectorTests(unittest.TestCase):
    def test_selects_highest_confidence_pose(self):
        points = np.array([
            [[10, 20], [11, 20], [12, 20], [13, 20], [14, 20]],
            [[20, 30], [21, 30], [22, 30], [23, 30], [24, 30]],
            [[30, 40], [31, 40], [32, 40], [33, 40], [34, 40]],
        ], float)
        selected = select_best_poses(
            xyxy=np.array([[10, 20, 30, 60], [0, 0, 100, 80], [5, 5, 25, 15]], float),
            box_scores=np.array([.7, .9, .8]),
            class_ids=np.array([0, 0, 1]),
            keypoints_xy=points,
            keypoint_scores=np.full((3, 5), .75),
            class_names={0: "guitar", 1: "bass"},
            expected=("guitar", "bass"),
        )
        np.testing.assert_allclose(selected["guitar"].points[0], [20, 30])
        np.testing.assert_allclose(selected["bass"].points[0], [30, 40])

    def test_smoothing_resets_after_gap(self):
        smoother = PoseSmoother(alpha=.5, reset_after_frames=1)

        def detection(x: float, y: float) -> InstrumentPoseDetection:
            return InstrumentPoseDetection(
                np.tile([x, y], (5, 1)).astype(float), np.full(5, .8), .9, (0, 0, 1, 1)
            )

        first = smoother.apply(0, "guitar", detection(0, 0))
        second = smoother.apply(1, "guitar", detection(10, 20))
        after_gap = smoother.apply(4, "guitar", detection(20, 30))
        np.testing.assert_allclose(first.points[0], [0, 0])
        np.testing.assert_allclose(second.points[0], [5, 10])
        np.testing.assert_allclose(after_gap.points[0], [20, 30])

    def test_init_dataset_writes_absolute_two_class_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            path = init_dataset(root)
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(data["path"], str(root.resolve()))
            self.assertEqual(data["names"], {0: "guitar", 1: "bass"})
            self.assertEqual(data["kpt_shape"], [5, 3])
            self.assertTrue((root / "raw" / "images").is_dir())

    def test_infer_video_exports_five_keypoint_interchange_csv(self):
        class FakeModel:
            names = {0: "guitar", 1: "bass"}

            @staticmethod
            def predict(**kwargs):
                del kwargs
                yield SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=np.array([[10, 20, 30, 60], [100, 100, 140, 120]], float),
                        conf=np.array([.9, .8]), cls=np.array([0, 1]),
                    ),
                    keypoints=SimpleNamespace(
                        xy=np.array([
                            [[20, 40], [18, 42], [25, 38], [30, 35], [35, 32]],
                            [[120, 110], [118, 112], [125, 108], [130, 105], [135, 102]],
                        ], float),
                        conf=np.full((2, 5), .75),
                    ),
                    orig_img=np.zeros((100, 200, 3), dtype=np.uint8),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.mp4"
            video.touch()
            output = root / "centers.csv"
            counts = infer_video(
                FakeModel(), video, output, ("guitar", "bass"), .5, .5, .5, 640, "cpu", 1.0
            )
            table = pd.read_csv(output)
            self.assertEqual(counts, {"guitar": 1, "bass": 1})
            self.assertEqual(len(table), 10)
            self.assertEqual(set(table["keypoint"]), {
                "body_center", "bridge", "neck_joint", "nut", "head_tip",
            })
            self.assertEqual(table.loc[table["instrument"] == "guitar", "x"].iloc[0], 20)

    def test_body_center_only_csv_loads_and_missing_path_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "centers.csv"
            path.write_text(
                "frame_idx,instrument,keypoint,x,y,score\n0,guitar,body_center,20,40,0.9\n",
                encoding="utf-8",
            )
            pose = InstrumentKeypointStore(path, .5).get(0, "guitar")
            np.testing.assert_allclose(pose.points[0], [20, 40])
            self.assertTrue(np.isnan(pose.points[1:]).all())
            with self.assertRaises(FileNotFoundError):
                InstrumentKeypointStore(root / "missing.csv")

    def test_pose_motion_finds_rise_and_visual_oscillation(self):
        fps = 30.0
        rows = []
        for frame_idx in range(120):
            time_sec = frame_idx / fps
            rise = max(0, min(frame_idx - 90, 10)) * 2.0
            oscillation = 5.0 * np.sin(2 * np.pi * 5.0 * time_sec)
            coordinates = {
                "body_center": (100, 100 - rise), "bridge": (90, 102 - rise),
                "neck_joint": (120, 80 - rise), "nut": (140, 65 - rise + oscillation / 2),
                "head_tip": (160, 50 - rise + oscillation),
            }
            for point in INSTRUMENT_POINTS:
                x, y = coordinates[point]
                rows.append({"frame_idx": frame_idx, "time_sec": time_sec, "camera": "cam",
                             "instrument": "guitar", "keypoint": point, "x": x, "y": y, "score": .9,
                             "frame_width": 1000, "frame_height": 1000})
        motion = instrument_pose_motion(
            pd.DataFrame(rows), fps, oscillation_min_power_ratio=.4,
            oscillation_min_rms_deg=.2, rise_velocity_threshold_norm_s=.02,
        )
        events = instrument_motion_events(motion, fps, rise_min_duration_sec=.2,
                                          oscillation_min_duration_sec=.2, ending_window_sec=2.0)
        self.assertTrue(motion["axis_angle_deg"].notna().all())
        self.assertIn("instrument_rise", set(events["event_type"]))
        self.assertIn("visual_oscillation", set(events["event_type"]))
        self.assertTrue(events.loc[events["event_type"] == "instrument_rise", "near_end"].any())


if __name__ == "__main__":
    unittest.main()

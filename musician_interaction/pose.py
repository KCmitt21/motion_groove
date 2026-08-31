from __future__ import annotations

import itertools
from typing import Any, Protocol

import numpy as np

from .types import PoseDetection


class PoseEstimator(Protocol):
    def infer(self, frame: np.ndarray) -> list[PoseDetection]: ...


class MMPoseWholeBodyEstimator:
    """Thin adapter around MMPoseInferencer (RTMPose WholeBody + RTMDet)."""

    def __init__(self, pose2d: str, det_model: str, device: str = "cpu") -> None:
        try:
            from mmpose.apis import MMPoseInferencer
        except ImportError as exc:
            raise RuntimeError(
                "MMPose is not installed. Install requirements-pose.txt or use pose.backend=mock."
            ) from exc
        self.inferencer = MMPoseInferencer(pose2d=pose2d, det_model=det_model, device=device)

    def infer(self, frame: np.ndarray) -> list[PoseDetection]:
        result = next(self.inferencer(frame, return_vis=False, show=False))
        predictions = result.get("predictions", [])
        if predictions and isinstance(predictions[0], list):
            predictions = predictions[0]
        detections = []
        for item in predictions:
            points = np.asarray(item.get("keypoints", []), dtype=float)
            scores = np.asarray(item.get("keypoint_scores", np.ones(len(points))), dtype=float)
            if points.ndim != 2 or points.shape[1] < 2:
                continue
            bbox = np.asarray(item.get("bbox", []), dtype=float).reshape(-1)
            detections.append(
                PoseDetection(points[:, :2], scores, bbox if bbox.size >= 4 else None,
                              float(item.get("bbox_score", np.nan)))
            )
        return detections


class MockPoseEstimator:
    """Deterministic moving skeleton for wiring tests; never a scientific result."""

    def __init__(self, people: int = 2) -> None:
        self.people = people
        self.index = 0

    def infer(self, frame: np.ndarray) -> list[PoseDetection]:
        height, width = frame.shape[:2]
        results = []
        for person_idx in range(self.people):
            cx = width * (0.33 if person_idx == 0 else 0.67) + 8 * np.sin(self.index / 8 + person_idx)
            cy = height * 0.48 + 5 * np.cos(self.index / 11 + person_idx)
            points = np.tile([cx, cy], (133, 1)).astype(float)
            # COCO-WholeBody body indices: nose, shoulders, elbows, wrists, hips.
            offsets = {
                0: (0, -150), 5: (-45, -80), 6: (45, -80), 7: (-70, -20), 8: (70, -20),
                9: (-90, 50), 10: (90, 50), 11: (-35, 80), 12: (35, 80),
            }
            for idx, (dx, dy) in offsets.items():
                points[idx] += (dx, dy)
            results.append(PoseDetection(points, np.full(133, 0.99)))
        self.index += 1
        return results


class PerformerTracker:
    """Assign stable semantic IDs using configured initial x order and temporal continuity."""

    def __init__(self, visible_performers: list[str], initial_left_to_right: list[str], min_score: float) -> None:
        self.visible = visible_performers
        self.initial_order = initial_left_to_right
        self.min_score = min_score
        self.previous: dict[str, np.ndarray] = {}

    @staticmethod
    def _distance(a: PoseDetection, b_center: np.ndarray, min_score: float) -> float:
        body = min(17, len(a.scores))
        valid = (a.scores[:body] >= min_score) & np.isfinite(a.keypoints[:body]).all(axis=1)
        if not valid.any() or not np.isfinite(b_center).all():
            return 1e12
        return float(np.linalg.norm(np.nanmedian(a.keypoints[:body][valid], axis=0) - b_center))

    def assign(self, detections: list[PoseDetection]) -> dict[str, PoseDetection | None]:
        output: dict[str, PoseDetection | None] = {name: None for name in self.visible}
        usable = [d for d in detections if np.sum(d.scores >= self.min_score) >= 3]
        if not usable:
            return output
        if not self.previous:
            ordered = sorted(usable, key=lambda detection: detection.center[0])
            for name, detection in zip(self.initial_order, ordered):
                if name in output:
                    output[name] = detection
        else:
            names = [name for name in self.visible if name in self.previous]
            best: tuple[float, tuple[int, ...]] | None = None
            for indices in itertools.permutations(range(len(usable)), min(len(names), len(usable))):
                cost = sum(self._distance(usable[idx], self.previous[name], self.min_score)
                           for name, idx in zip(names, indices))
                if best is None or cost < best[0]:
                    best = (cost, indices)
            if best:
                for name, idx in zip(names, best[1]):
                    output[name] = usable[idx]
            leftovers = [d for d in usable if all(d is not item for item in output.values())]
            for name in self.visible:
                if output[name] is None and leftovers:
                    output[name] = leftovers.pop(0)
        for name, detection in output.items():
            if detection is not None:
                self.previous[name] = detection.center
        return output


def build_pose_estimator(config: dict[str, Any], people: int) -> PoseEstimator:
    backend = config.get("backend", "mmpose")
    if backend == "mock":
        return MockPoseEstimator(people)
    if backend != "mmpose":
        raise ValueError(f"Unknown pose backend: {backend}")
    return MMPoseWholeBodyEstimator(config["model"], config["detector"], config.get("device", "cpu"))

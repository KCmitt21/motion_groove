from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


PERFORMERS = ("guitarist", "bassist")
INSTRUMENT_POINTS = ("body_center", "bridge", "neck_joint", "nut", "head_tip")


@dataclass
class PoseDetection:
    keypoints: np.ndarray
    scores: np.ndarray
    bbox: np.ndarray | None = None
    instance_score: float = np.nan

    @property
    def center(self) -> np.ndarray:
        valid = np.isfinite(self.keypoints).all(axis=1) & (self.scores > 0)
        return np.nanmedian(self.keypoints[valid], axis=0) if valid.any() else np.array([np.nan, np.nan])


@dataclass
class HeadPose:
    yaw: float = np.nan
    pitch: float = np.nan
    roll: float = np.nan
    score: float = np.nan
    center: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))
    landmarks: np.ndarray | None = None
    # Camera coordinates from solvePnP.  The gaze vector is unit length and
    # points out through the front of the face.
    position_3d: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    gaze_direction_3d: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))


@dataclass
class InstrumentPose:
    points: np.ndarray = field(default_factory=lambda: np.full((5, 2), np.nan))
    scores: np.ndarray = field(default_factory=lambda: np.full(5, np.nan))

from __future__ import annotations

import numpy as np

from .types import HeadPose, InstrumentPose, PoseDetection


GAZE_CLASSES = ("partner", "own_instrument", "forward", "downward", "unknown")


def _bearing_degrees(origin: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    delta = target - origin
    # Image y is downward. These are image-plane proxies, not eye tracking angles.
    return float(np.degrees(np.arctan2(delta[0], max(abs(delta[1]), 1.0)))), float(
        np.degrees(np.arctan2(delta[1], max(abs(delta[0]), 1.0)))
    )


def classify_gaze(
    head: HeadPose | None,
    performer: PoseDetection | None,
    partner: PoseDetection | None,
    own_instrument: InstrumentPose | None,
    config: dict,
) -> tuple[str, float]:
    """Coarse head-orientation target; not a claim about ocular gaze."""
    if head is None or not np.isfinite([head.yaw, head.pitch]).all():
        return "unknown", np.nan
    down_pitch = float(config.get("down_pitch_deg", 18.0))
    forward_yaw = float(config.get("forward_yaw_deg", 15.0))
    forward_pitch = float(config.get("forward_pitch_deg", 15.0))
    target_tolerance = float(config.get("target_tolerance_deg", 25.0))
    if head.pitch > down_pitch:
        if own_instrument is not None and np.isfinite(own_instrument.points[0]).all():
            return "own_instrument", max(0.0, 1.0 - abs(head.pitch - down_pitch) / 60.0)
        return "downward", max(0.0, min(1.0, head.pitch / 45.0))
    if partner is not None and np.isfinite(partner.center).all():
        target_yaw, _ = _bearing_degrees(head.center, partner.center)
        error = abs(head.yaw - target_yaw)
        if error <= target_tolerance:
            return "partner", max(0.0, 1.0 - error / target_tolerance)
    if abs(head.yaw) <= forward_yaw and abs(head.pitch) <= forward_pitch:
        return "forward", 1.0 - max(abs(head.yaw) / forward_yaw, abs(head.pitch) / forward_pitch)
    return "unknown", 0.0


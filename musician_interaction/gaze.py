from __future__ import annotations

import numpy as np

from .types import HeadPose, InstrumentPose, PoseDetection


GAZE_CLASSES = ("partner", "own_instrument", "forward", "downward", "unknown")
UNKNOWN_REASONS = (
    "face_not_detected",
    "invalid_head_angles",
    "head_pose_angle_outlier",
    "reference_profile_missing",
    "outside_reference_tolerance",
    "partner_pose_missing_and_outside_thresholds",
    "outside_class_thresholds",
    "reason_not_recorded",
)


def _bearing_degrees(origin: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    delta = target - origin
    # Image y is downward. These are image-plane proxies, not eye tracking angles.
    return float(np.degrees(np.arctan2(delta[0], max(abs(delta[1]), 1.0)))), float(
        np.degrees(np.arctan2(delta[1], max(abs(delta[0]), 1.0)))
    )


def _reference_profile(config: dict, camera: str | None, performer_name: str | None) -> tuple[dict | None, bool]:
    """Return a performer profile and whether reference-angle mode applies to this camera."""
    if camera is None or performer_name is None:
        return None, False
    cameras = config.get("reference_angles", {})
    if not isinstance(cameras, dict) or camera not in cameras:
        return None, False
    performers = cameras[camera]
    if not isinstance(performers, dict):
        return None, True
    profile = performers.get(performer_name)
    return (profile if isinstance(profile, dict) else None), True


def _reference_angle_classification(head: HeadPose, profile: dict) -> tuple[str, float] | None:
    """Choose the closest in-tolerance empirical yaw/pitch prototype."""
    candidates: list[tuple[float, str]] = []
    for label in GAZE_CLASSES:
        if label == "unknown" or label not in profile:
            continue
        reference = profile[label]
        if not isinstance(reference, dict):
            continue
        try:
            reference_yaw = float(reference["yaw_deg"])
            reference_pitch = float(reference["pitch_deg"])
            yaw_tolerance = float(reference.get("yaw_tolerance_deg", 20.0))
            pitch_tolerance = float(reference.get("pitch_tolerance_deg", 20.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite([reference_yaw, reference_pitch, yaw_tolerance, pitch_tolerance]).all():
            continue
        if yaw_tolerance <= 0 or pitch_tolerance <= 0:
            continue
        # Elliptical distance lets yaw and pitch tolerances be tuned independently.
        yaw_error = (head.yaw - reference_yaw + 180.0) % 360.0 - 180.0
        distance = float(np.hypot(
            yaw_error / yaw_tolerance, (head.pitch - reference_pitch) / pitch_tolerance
        ))
        candidates.append((distance, label))
    if not candidates:
        return None
    distance, label = min(candidates)
    return (label, max(0.0, 1.0 - distance)) if distance <= 1.0 else None


def _head_pose_is_outlier(head: HeadPose, config: dict) -> bool:
    quality = config.get("head_pose_quality", {})
    if not isinstance(quality, dict):
        return False
    for value, key in (
        (head.yaw, "max_abs_yaw_deg"),
        (head.pitch, "max_abs_pitch_deg"),
        (head.roll, "max_abs_roll_deg"),
    ):
        limit = quality.get(key)
        if limit is not None and (not np.isfinite(value) or abs(value) > float(limit)):
            return True
    return False


def classify_gaze_detailed(
    head: HeadPose | None,
    performer: PoseDetection | None,
    partner: PoseDetection | None,
    own_instrument: InstrumentPose | None,
    config: dict,
    camera: str | None = None,
    performer_name: str | None = None,
) -> tuple[str, float, str | None]:
    """Classify coarse head orientation and explain every unknown result."""
    if head is None:
        return "unknown", np.nan, "face_not_detected"
    if not np.isfinite([head.yaw, head.pitch]).all():
        return "unknown", np.nan, "invalid_head_angles"
    if _head_pose_is_outlier(head, config):
        return "unknown", 0.0, "head_pose_angle_outlier"

    profile, reference_mode = _reference_profile(config, camera, performer_name)
    if reference_mode and profile is not None:
        reference_result = _reference_angle_classification(head, profile)
        if reference_result is not None:
            label, score = reference_result
            return label, score, None

    down_pitch = float(config.get("down_pitch_deg", 18.0))
    forward_yaw = float(config.get("forward_yaw_deg", 15.0))
    forward_pitch = float(config.get("forward_pitch_deg", 15.0))
    target_tolerance = float(config.get("target_tolerance_deg", 25.0))
    if head.pitch > down_pitch:
        if own_instrument is not None and np.isfinite(own_instrument.points[0]).all():
            return "own_instrument", max(0.0, 1.0 - abs(head.pitch - down_pitch) / 60.0), None
        return "downward", max(0.0, min(1.0, head.pitch / 45.0)), None
    if not reference_mode and partner is not None and np.isfinite(partner.center).all():
        target_yaw, _ = _bearing_degrees(head.center, partner.center)
        error = abs(head.yaw - target_yaw)
        if error <= target_tolerance:
            return "partner", max(0.0, 1.0 - error / target_tolerance), None
    if abs(head.yaw) <= forward_yaw and abs(head.pitch) <= forward_pitch:
        return "forward", 1.0 - max(abs(head.yaw) / forward_yaw, abs(head.pitch) / forward_pitch), None
    if reference_mode:
        reason = "reference_profile_missing" if profile is None else "outside_reference_tolerance"
    elif partner is None or not np.isfinite(partner.center).all():
        reason = "partner_pose_missing_and_outside_thresholds"
    else:
        reason = "outside_class_thresholds"
    return "unknown", 0.0, reason


def classify_gaze(
    head: HeadPose | None,
    performer: PoseDetection | None,
    partner: PoseDetection | None,
    own_instrument: InstrumentPose | None,
    config: dict,
) -> tuple[str, float]:
    """Backward-compatible classifier without the unknown-reason field."""
    label, score, _ = classify_gaze_detailed(head, performer, partner, own_instrument, config)
    return label, score

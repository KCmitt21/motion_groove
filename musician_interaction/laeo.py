from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .types import HeadPose


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """Return the unsigned angle between two 3D vectors in degrees."""
    first = np.asarray(first, dtype=float).reshape(-1)
    second = np.asarray(second, dtype=float).reshape(-1)
    if first.size != 3 or second.size != 3 or not np.isfinite(first).all() or not np.isfinite(second).all():
        return np.nan
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= np.finfo(float).eps:
        return np.nan
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def laeo_metrics(
    head_a: HeadPose | None,
    head_b: HeadPose | None,
    sigma_deg: float,
) -> dict[str, float]:
    """Compute continuous looking-at-each-other evidence for one frame."""
    sigma_deg = float(sigma_deg)
    if not np.isfinite(sigma_deg) or sigma_deg <= 0:
        raise ValueError(f"laeo.sigma_deg must be positive and finite, got {sigma_deg}")

    missing = {
        "theta_A": np.nan,
        "theta_B": np.nan,
        "p_A": np.nan,
        "p_B": np.nan,
        "laeo_score": np.nan,
    }
    if head_a is None or head_b is None:
        return missing

    position_a = np.asarray(head_a.position_3d, dtype=float).reshape(-1)
    position_b = np.asarray(head_b.position_3d, dtype=float).reshape(-1)
    if position_a.size != 3 or position_b.size != 3:
        return missing
    a_to_b = position_b - position_a
    b_to_a = -a_to_b
    theta_a = angle_degrees(head_a.gaze_direction_3d, a_to_b)
    theta_b = angle_degrees(head_b.gaze_direction_3d, b_to_a)
    p_a = float(np.exp(-(theta_a ** 2) / (2 * sigma_deg ** 2))) if np.isfinite(theta_a) else np.nan
    p_b = float(np.exp(-(theta_b ** 2) / (2 * sigma_deg ** 2))) if np.isfinite(theta_b) else np.nan
    score = float(p_a * p_b) if np.isfinite([p_a, p_b]).all() else np.nan
    return {
        "theta_A": theta_a,
        "theta_B": theta_b,
        "p_A": p_a,
        "p_B": p_b,
        "laeo_score": score,
    }


def laeo_frame_row(
    frame_idx: int,
    timestamp: float,
    camera: str,
    performer_a: str,
    performer_b: str,
    head_a: HeadPose | None,
    head_b: HeadPose | None,
    sigma_deg: float,
) -> dict[str, Any]:
    """Build an auditable LAEO output row for one camera frame."""
    return {
        "frame_idx": frame_idx,
        "timestamp": timestamp,
        "time_sec": timestamp,
        "camera": camera,
        "performer_A": performer_a,
        "performer_B": performer_b,
        **laeo_metrics(head_a, head_b, sigma_deg),
    }


def _kernel(method: str, radius: int, gaussian_sigma_frames: float) -> np.ndarray:
    if radius < 0:
        raise ValueError(f"laeo.smoothing.radius_frames must be non-negative, got {radius}")
    if method == "moving_average":
        return np.ones(2 * radius + 1, dtype=float)
    if method == "gaussian":
        if not np.isfinite(gaussian_sigma_frames) or gaussian_sigma_frames <= 0:
            raise ValueError(
                "laeo.smoothing.gaussian_sigma_frames must be positive and finite, "
                f"got {gaussian_sigma_frames}"
            )
        offsets = np.arange(-radius, radius + 1, dtype=float)
        return np.exp(-(offsets ** 2) / (2 * gaussian_sigma_frames ** 2))
    raise ValueError(
        f"Unsupported laeo.smoothing.method={method!r}; use 'gaussian' or 'moving_average'"
    )


def _smooth_valid_runs(frame_indices: np.ndarray, scores: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Smooth contiguous valid tracks without filling or bridging missing frames."""
    result = np.full(len(scores), np.nan, dtype=float)
    valid = np.isfinite(scores)
    starts = np.flatnonzero(valid & np.r_[True, (~valid[:-1]) | (np.diff(frame_indices) != 1)])
    for start in starts:
        end = start + 1
        while end < len(scores) and valid[end] and frame_indices[end] == frame_indices[end - 1] + 1:
            end += 1
        values = scores[start:end]
        radius = len(kernel) // 2
        padded_values = np.pad(values, (radius, radius), mode="constant")
        padded_weights = np.pad(np.ones(len(values)), (radius, radius), mode="constant")
        numerator = np.convolve(padded_values, kernel, mode="valid")
        denominator = np.convolve(padded_weights, kernel, mode="valid")
        result[start:end] = numerator / denominator
    return result


def smooth_laeo_scores(table: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Add a centered temporal score while retaining the unmodified frame score."""
    config = config or {}
    method = str(config.get("method", "gaussian"))
    radius = int(config.get("radius_frames", 10))
    gaussian_sigma_frames = float(config.get("gaussian_sigma_frames", 4.0))
    kernel = _kernel(method, radius, gaussian_sigma_frames)

    result = table.copy()
    result["laeo_score_smoothed"] = np.nan
    if result.empty:
        return result
    required = {"frame_idx", "camera", "performer_A", "performer_B", "laeo_score"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"LAEO table is missing columns: {sorted(missing)}")
    groups = result.groupby(["camera", "performer_A", "performer_B"], sort=False, dropna=False)
    for _, indices in groups.groups.items():
        ordered = result.loc[indices].sort_values("frame_idx")
        smoothed = _smooth_valid_runs(
            ordered["frame_idx"].to_numpy(dtype=int),
            ordered["laeo_score"].to_numpy(dtype=float),
            kernel,
        )
        result.loc[ordered.index, "laeo_score_smoothed"] = smoothed
    return result.sort_values(["frame_idx", "camera", "performer_A", "performer_B"]).reset_index(drop=True)

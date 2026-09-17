from __future__ import annotations

import numpy as np
import pandas as pd


def keypoint_quality(keypoints: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if keypoints.empty:
        return pd.DataFrame()
    table = keypoints.assign(
        missing=~(np.isfinite(keypoints["x"]) & np.isfinite(keypoints["y"])),
        low_confidence=keypoints["score"].fillna(0).lt(threshold),
    )
    return table.groupby(["camera", "performer", "keypoint"], as_index=False).agg(
        samples=("frame_idx", "size"), missing_fraction=("missing", "mean"),
        low_confidence_fraction=("low_confidence", "mean"), mean_score=("score", "mean")
    )


def estimate_sync_offsets(audio_by_camera: dict[str, object], reference: str, max_lag_sec: float = 1.0) -> pd.DataFrame:
    def best_offset(a: np.ndarray, b: np.ndarray, dt: float, max_lag: int) -> tuple[float, float]:
        a = a - np.nanmean(a)
        b = b - np.nanmean(b)
        correlations = np.correlate(a, b, mode="full")
        lags = np.arange(-len(a) + 1, len(a))
        keep = np.abs(lags) <= max_lag
        lag = int(lags[keep][np.argmax(correlations[keep])])
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        peak = float(correlations[keep].max() / denom) if denom else np.nan
        return lag * dt, peak

    rows = []
    ref = audio_by_camera[reference]
    for camera, features in audio_by_camera.items():
        if len(ref.times) < 2 or len(features.times) < 2:
            rows.append({"camera": camera, "reference_camera": reference, "audio_offset_sec": np.nan,
                         "early_offset_sec": np.nan, "late_offset_sec": np.nan,
                         "estimated_drift_sec": np.nan, "normalized_peak": np.nan,
                         "sign_convention": "positive means camera audio is earlier than reference"})
            continue
        sample_dt = float(np.nanmedian(np.diff(ref.times)))
        max_lag = int(round(max_lag_sec / sample_dt))
        common_start = max(ref.times[0], features.times[0])
        common_end = min(ref.times[-1], features.times[-1])
        grid = np.arange(common_start, common_end, sample_dt)
        if len(grid) < max(10, 2 * max_lag + 1):
            rows.append({"camera": camera, "reference_camera": reference, "audio_offset_sec": np.nan,
                         "early_offset_sec": np.nan, "late_offset_sec": np.nan,
                         "estimated_drift_sec": np.nan, "normalized_peak": np.nan,
                         "sign_convention": "positive means camera audio is earlier than reference"})
            continue
        a = np.interp(grid, ref.times, ref.spectral_flux)
        b = np.interp(grid, features.times, features.spectral_flux)
        n = len(grid)
        offset, peak = best_offset(a, b, sample_dt, max_lag)
        window = max(10, n // 4)
        early_offset, _ = best_offset(a[:window], b[:window], sample_dt, max_lag)
        late_offset, _ = best_offset(a[-window:], b[-window:], sample_dt, max_lag)
        rows.append({"camera": camera, "reference_camera": reference, "audio_offset_sec": offset,
                     "early_offset_sec": early_offset, "late_offset_sec": late_offset,
                     "estimated_drift_sec": late_offset - early_offset, "normalized_peak": peak,
                     "sign_convention": "positive means camera audio is earlier than reference"})
    return pd.DataFrame(rows)


def instrument_quality(keypoints: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if keypoints.empty:
        return pd.DataFrame()
    table = keypoints.assign(
        missing=~(np.isfinite(keypoints["x"]) & np.isfinite(keypoints["y"])),
        low_confidence=keypoints["score"].fillna(0).lt(threshold),
    )
    return table.groupby(["camera", "instrument", "keypoint"], as_index=False).agg(
        samples=("frame_idx", "size"), missing_fraction=("missing", "mean"),
        low_confidence_fraction=("low_confidence", "mean"), mean_score=("score", "mean")
    )


def reprojection_quality(reprojection: pd.DataFrame, warn_px: float) -> pd.DataFrame:
    if reprojection.empty:
        return pd.DataFrame()
    return reprojection.assign(warning=reprojection["reprojection_error_px"] > warn_px)

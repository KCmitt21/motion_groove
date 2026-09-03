from __future__ import annotations

import numpy as np
import pandas as pd


BODY_INDEX = {
    "head": [0],
    "left_wrist": [9],
    "right_wrist": [10],
    "upper_body": [5, 6, 11, 12],
}


def positions_to_motion(positions: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Create framewise 2D speeds without bridging missing or nonconsecutive frames."""
    if positions.empty:
        return positions.copy()
    output = positions.sort_values(["camera", "performer", "feature", "frame_idx"]).copy()
    groups = output.groupby(["camera", "performer", "feature"], sort=False)
    dx = groups["x"].diff()
    dy = groups["y"].diff()
    frame_step = groups["frame_idx"].diff()
    valid = frame_step.eq(1) & output["x"].notna() & output["y"].notna()
    output["speed_px_s"] = np.where(valid, np.hypot(dx, dy) * fps, np.nan)
    diagonal = np.hypot(output["frame_width"], output["frame_height"])
    output["speed_norm_s"] = output["speed_px_s"] / diagonal
    return output


def lagged_correlation(
    first: np.ndarray, second: np.ndarray, max_lag_frames: int, min_samples: int = 20
) -> pd.DataFrame:
    rows = []
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag < 0:
            a, b = first[-lag:], second[:lag]
        elif lag > 0:
            a, b = first[:-lag], second[lag:]
        else:
            a, b = first, second
        valid = np.isfinite(a) & np.isfinite(b)
        corr = np.corrcoef(a[valid], b[valid])[0, 1] if valid.sum() >= min_samples else np.nan
        rows.append({"lag_frames": lag, "correlation": corr, "n_samples": int(valid.sum())})
    return pd.DataFrame(rows)


def performer_cross_correlations(motion: pd.DataFrame, fps: float, max_lag_sec: float) -> pd.DataFrame:
    rows = []
    subset = motion[motion["feature"] == "upper_body"]
    for camera, camera_data in subset.groupby("camera"):
        pivot = camera_data.pivot_table(index="frame_idx", columns="performer", values="speed_norm_s")
        if not {"guitarist", "bassist"}.issubset(pivot.columns):
            continue
        full_index = np.arange(int(pivot.index.min()), int(pivot.index.max()) + 1)
        pivot = pivot.reindex(full_index)
        result = lagged_correlation(
            pivot["guitarist"].to_numpy(), pivot["bassist"].to_numpy(), int(round(max_lag_sec * fps))
        )
        result.insert(0, "camera", camera)
        result["lag_sec"] = result["lag_frames"] / fps
        result["positive_lag_meaning"] = "bassist follows guitarist"
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["camera", "lag_frames", "lag_sec", "correlation", "n_samples", "positive_lag_meaning"]
    )


def audio_motion_correlations(
    motion: pd.DataFrame, audio_framewise: pd.DataFrame, fps: float, max_lag_sec: float
) -> pd.DataFrame:
    rows = []
    audio = audio_framewise.set_index("frame_idx")["spectral_flux"]
    for keys, data in motion.groupby(["camera", "performer", "feature"]):
        signal = data.set_index("frame_idx")["speed_norm_s"]
        index = audio.index.union(signal.index)
        result = lagged_correlation(
            audio.reindex(index).to_numpy(), signal.reindex(index).to_numpy(), int(round(max_lag_sec * fps))
        )
        result[["camera", "performer", "feature"]] = keys
        result["lag_sec"] = result["lag_frames"] / fps
        result["positive_lag_meaning"] = "motion follows audio"
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def event_triggered_average(
    motion: pd.DataFrame, onset_times: np.ndarray, fps: float, window_sec: float
) -> pd.DataFrame:
    radius = int(round(window_sec * fps))
    offsets = np.arange(-radius, radius + 1)
    rows = []
    for keys, data in motion.groupby(["camera", "performer", "feature"]):
        series = data.set_index("frame_idx")["speed_norm_s"]
        event_frames = np.rint(onset_times * fps).astype(int)
        samples = np.full((len(event_frames), len(offsets)), np.nan)
        for event_idx, center in enumerate(event_frames):
            samples[event_idx] = series.reindex(center + offsets).to_numpy()
        count = np.sum(np.isfinite(samples), axis=0)
        totals = np.nansum(samples, axis=0)
        mean = np.divide(totals, count, out=np.full(len(offsets), np.nan), where=count > 0)
        for offset, value, n in zip(offsets, mean, count):
            rows.append({"camera": keys[0], "performer": keys[1], "feature": keys[2],
                         "offset_sec": offset / fps, "mean_speed_norm_s": value,
                         "n_events": int(n)})
    return pd.DataFrame(rows)


def _orientation_degrees(x0: pd.Series, y0: pd.Series, x1: pd.Series, y1: pd.Series) -> pd.Series:
    """Image-plane orientation with positive angles pointing upward."""
    return pd.Series(np.degrees(np.arctan2(-(y1 - y0), x1 - x0)), index=x0.index)


def _rolling_oscillation(
    values: np.ndarray, fps: float, window_frames: int, min_hz: float, max_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dominant = np.full(len(values), np.nan)
    power_ratio = np.full(len(values), np.nan)
    rms = np.full(len(values), np.nan)
    if window_frames < 4 or len(values) < window_frames:
        return dominant, power_ratio, rms
    half = window_frames // 2
    for center in range(half, len(values) - (window_frames - half) + 1):
        start = center - half
        segment = values[start:start + window_frames]
        valid = np.isfinite(segment)
        if valid.sum() < max(4, int(np.ceil(window_frames * .8))):
            continue
        sample_index = np.arange(window_frames)
        filled = np.interp(sample_index, sample_index[valid], segment[valid])
        unwrapped = np.degrees(np.unwrap(np.radians(filled)))
        trend = np.polyval(np.polyfit(sample_index, unwrapped, 1), sample_index)
        centered = unwrapped - trend
        rms[center] = float(np.sqrt(np.mean(centered ** 2)))
        frequencies = np.fft.rfftfreq(window_frames, d=1.0 / fps)
        power = np.abs(np.fft.rfft(centered)) ** 2
        positive = frequencies > 0
        band = (frequencies >= min_hz) & (frequencies <= max_hz)
        total = float(np.sum(power[positive]))
        if total <= 0 or not band.any():
            continue
        band_power = power[band]
        power_ratio[center] = float(np.sum(band_power) / total)
        dominant[center] = float(frequencies[band][int(np.argmax(band_power))])
    return dominant, power_ratio, rms


def instrument_pose_motion(
    instruments: pd.DataFrame,
    fps: float,
    oscillation_window_sec: float = 1.0,
    oscillation_min_hz: float = 3.0,
    oscillation_max_hz: float = 9.0,
    oscillation_min_power_ratio: float = .5,
    oscillation_min_rms_deg: float = .5,
    rise_velocity_threshold_norm_s: float = .02,
    rise_smoothing_sec: float = .2,
) -> pd.DataFrame:
    """Convert five instrument points into orientation, height and visual oscillation signals."""
    columns = [
        "camera", "instrument", "frame_idx", "time_sec", "frame_width", "frame_height",
        "body_center_x", "body_center_y", "head_tip_x", "head_tip_y",
        "axis_angle_deg", "neck_angle_deg", "length_norm", "head_tip_height_norm",
        "body_center_height_norm", "head_tip_relative_height_norm",
        "body_center_vertical_speed_norm_s", "head_tip_vertical_speed_norm_s",
        "head_tip_vertical_speed_smoothed_norm_s",
        "axis_angular_velocity_deg_s", "neck_angular_velocity_deg_s",
        "oscillation_frequency_hz", "oscillation_band_power_ratio", "oscillation_rms_deg",
        "rise_candidate", "visual_vibrato_candidate",
    ]
    if instruments.empty:
        return pd.DataFrame(columns=columns)
    required_dimensions = {"frame_width", "frame_height"}
    if not required_dimensions.issubset(instruments.columns):
        raise ValueError("Instrument table needs frame_width and frame_height for normalized motion")
    rows = []
    point_names = ("body_center", "bridge", "neck_joint", "nut", "head_tip")
    for (camera, instrument), data in instruments.groupby(["camera", "instrument"], sort=False):
        index_columns = ["frame_idx", "time_sec", "frame_width", "frame_height"]
        output = data[index_columns].drop_duplicates("frame_idx").sort_values("frame_idx").copy()
        output["camera"] = camera
        output["instrument"] = instrument
        for point in point_names:
            point_data = data[data["keypoint"] == point].drop_duplicates("frame_idx").set_index("frame_idx")
            for value in ("x", "y", "score"):
                output[f"{point}_{value}"] = output["frame_idx"].map(point_data[value])
        output["axis_angle_deg"] = _orientation_degrees(
            output["body_center_x"], output["body_center_y"], output["head_tip_x"], output["head_tip_y"]
        )
        output["neck_angle_deg"] = _orientation_degrees(
            output["neck_joint_x"], output["neck_joint_y"], output["head_tip_x"], output["head_tip_y"]
        )
        diagonal = np.hypot(output["frame_width"], output["frame_height"])
        output["length_norm"] = np.hypot(
            output["head_tip_x"] - output["body_center_x"],
            output["head_tip_y"] - output["body_center_y"],
        ) / diagonal
        output["head_tip_height_norm"] = 1.0 - output["head_tip_y"] / output["frame_height"]
        output["body_center_height_norm"] = 1.0 - output["body_center_y"] / output["frame_height"]
        output["head_tip_relative_height_norm"] = (
            output["body_center_y"] - output["head_tip_y"]
        ) / diagonal
        frame_step = output["frame_idx"].diff()
        consecutive = frame_step.eq(1)
        for point in ("body_center", "head_tip"):
            velocity = -output[f"{point}_y"].diff() * fps / diagonal
            output[f"{point}_vertical_speed_norm_s"] = np.where(consecutive, velocity, np.nan)
        for angle in ("axis", "neck"):
            delta = output[f"{angle}_angle_deg"].diff()
            wrapped = (delta + 180.0) % 360.0 - 180.0
            output[f"{angle}_angular_velocity_deg_s"] = np.where(consecutive, wrapped * fps, np.nan)
        frequency, ratio, rms = _rolling_oscillation(
            output["neck_angle_deg"].to_numpy(), fps,
            max(4, int(round(oscillation_window_sec * fps))), oscillation_min_hz, oscillation_max_hz,
        )
        output["oscillation_frequency_hz"] = frequency
        output["oscillation_band_power_ratio"] = ratio
        output["oscillation_rms_deg"] = rms
        rise_window = max(1, int(round(rise_smoothing_sec * fps)))
        output["head_tip_vertical_speed_smoothed_norm_s"] = output[
            "head_tip_vertical_speed_norm_s"
        ].rolling(rise_window, center=True, min_periods=rise_window).mean()
        output["rise_candidate"] = (
            output["head_tip_vertical_speed_smoothed_norm_s"] >= rise_velocity_threshold_norm_s
        )
        output["visual_vibrato_candidate"] = (
            (output["oscillation_band_power_ratio"] >= oscillation_min_power_ratio)
            & (output["oscillation_rms_deg"] >= oscillation_min_rms_deg)
        )
        rows.append(output)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    return result


def _true_runs(frames: np.ndarray, mask: np.ndarray, minimum_frames: int) -> list[tuple[int, int]]:
    runs = []
    start = None
    previous = None
    for index, (frame, active) in enumerate(zip(frames, mask)):
        if active and (start is None or previous is None or frame != previous + 1):
            if start is not None and index - start >= minimum_frames:
                runs.append((start, index - 1))
            start = index
        elif not active and start is not None:
            if index - start >= minimum_frames:
                runs.append((start, index - 1))
            start = None
        previous = frame
    if start is not None and len(frames) - start >= minimum_frames:
        runs.append((start, len(frames) - 1))
    return runs


def instrument_motion_events(
    motion: pd.DataFrame,
    fps: float,
    rise_min_duration_sec: float = .2,
    oscillation_min_duration_sec: float = .3,
    ending_window_sec: float = 5.0,
) -> pd.DataFrame:
    """Consolidate framewise rise and visual-oscillation candidates into events."""
    event_columns = [
        "camera", "instrument", "event_type", "start_frame", "end_frame", "start_sec", "end_sec",
        "duration_sec", "near_end", "peak_upward_speed_norm_s", "dominant_frequency_hz",
        "mean_band_power_ratio", "peak_oscillation_rms_deg",
    ]
    if motion.empty:
        return pd.DataFrame(columns=event_columns)
    rows = []
    definitions = (
        ("instrument_rise", "rise_candidate", max(1, int(round(rise_min_duration_sec * fps)))),
        ("visual_oscillation", "visual_vibrato_candidate",
         max(1, int(round(oscillation_min_duration_sec * fps)))),
    )
    for (camera, instrument), data in motion.groupby(["camera", "instrument"], sort=False):
        data = data.sort_values("frame_idx").reset_index(drop=True)
        last_time = float(data["time_sec"].max())
        for event_type, candidate_column, minimum_frames in definitions:
            mask = data[candidate_column].fillna(False).to_numpy(dtype=bool)
            for start, end in _true_runs(data["frame_idx"].to_numpy(), mask, minimum_frames):
                event = data.iloc[start:end + 1]
                end_sec = float(event["time_sec"].iloc[-1])
                rows.append({
                    "camera": camera, "instrument": instrument, "event_type": event_type,
                    "start_frame": int(event["frame_idx"].iloc[0]),
                    "end_frame": int(event["frame_idx"].iloc[-1]),
                    "start_sec": float(event["time_sec"].iloc[0]), "end_sec": end_sec,
                    "duration_sec": float((event["frame_idx"].iloc[-1] - event["frame_idx"].iloc[0] + 1) / fps),
                    "near_end": bool(last_time - end_sec <= ending_window_sec),
                    "peak_upward_speed_norm_s": float(event["head_tip_vertical_speed_norm_s"].max()),
                    "dominant_frequency_hz": float(event["oscillation_frequency_hz"].median()),
                    "mean_band_power_ratio": float(event["oscillation_band_power_ratio"].mean()),
                    "peak_oscillation_rms_deg": float(event["oscillation_rms_deg"].max()),
                })
    return pd.DataFrame(rows, columns=event_columns)

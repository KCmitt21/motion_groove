from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .gaze import UNKNOWN_REASONS, classify_gaze_detailed
from .outputs import write_table
from .types import HeadPose
from .video import FPS


HEAD_DIRECTIONS = ("partner", "own_instrument", "forward", "downward", "unknown")


def reclassify_reference_angles(heads: pd.DataFrame, gaze_config: dict) -> pd.DataFrame:
    """Reapply configured reference angles to an existing head-pose table."""
    reference_cameras = gaze_config.get("reference_angles", {})
    if not isinstance(reference_cameras, dict) or not reference_cameras:
        return heads.copy()
    result = heads.copy()
    if "unknown_reason" not in result:
        result["unknown_reason"] = None
    selected = result["camera"].isin(reference_cameras)
    for idx, row in result[selected].iterrows():
        angles = np.asarray([row["yaw_deg"], row["pitch_deg"]], dtype=float)
        head = None if not np.isfinite(angles).all() else HeadPose(
            yaw=float(row["yaw_deg"]), pitch=float(row["pitch_deg"]),
            roll=float(row["roll_deg"]), score=float(row["face_score"]),
        )
        label, score, reason = classify_gaze_detailed(
            head, None, None, None, gaze_config, str(row["camera"]), str(row["performer"])
        )
        # Retain existing instrument evidence because this table does not contain instrument keypoints.
        if label == "downward" and row["gaze_target"] == "own_instrument":
            label, reason = "own_instrument", None
        result.at[idx, "gaze_target"] = label
        result.at[idx, "gaze_score"] = score
        result.at[idx, "unknown_reason"] = reason
    return result


def head_direction_timeline(heads: pd.DataFrame) -> pd.DataFrame:
    """Normalize pipeline head-pose rows into an explicit direction timeline."""
    required = {
        "frame_idx", "time_sec", "camera", "performer", "yaw_deg", "pitch_deg",
        "roll_deg", "face_score", "gaze_target", "gaze_score",
    }
    missing = required - set(heads.columns)
    if missing:
        raise ValueError(f"Head-pose table is missing columns: {sorted(missing)}")
    timeline = heads.copy()
    timeline["head_direction"] = timeline["gaze_target"].where(
        timeline["gaze_target"].isin(HEAD_DIRECTIONS), "unknown"
    )
    timeline["face_detected"] = timeline[["yaw_deg", "pitch_deg"]].notna().all(axis=1)
    is_unknown = timeline["head_direction"].eq("unknown")
    if "unknown_reason" not in timeline:
        timeline["unknown_reason"] = np.where(
            is_unknown & ~timeline["face_detected"], "face_not_detected",
            np.where(is_unknown, "reason_not_recorded", None),
        )
    else:
        missing_reason = timeline["unknown_reason"].isna() | timeline["unknown_reason"].eq("")
        timeline.loc[is_unknown & missing_reason & ~timeline["face_detected"], "unknown_reason"] = "face_not_detected"
        timeline.loc[is_unknown & missing_reason & timeline["face_detected"], "unknown_reason"] = "reason_not_recorded"
        timeline.loc[~is_unknown, "unknown_reason"] = None
    columns = [
        "frame_idx", "time_sec", "camera", "performer", "face_detected",
        "head_direction", "gaze_score", "unknown_reason", "yaw_deg", "pitch_deg", "roll_deg", "face_score",
    ]
    return timeline[columns].sort_values(["frame_idx", "camera", "performer"]).reset_index(drop=True)


def head_direction_summary(timeline: pd.DataFrame) -> pd.DataFrame:
    """Count every direction, including zero-count classes, for each camera and performer."""
    rows: list[dict[str, Any]] = []
    frame_period = float(1 / FPS)
    for (camera, performer), group in timeline.groupby(["camera", "performer"], sort=False):
        total = len(group)
        detected_total = int(group["face_detected"].sum())
        for direction in HEAD_DIRECTIONS:
            selected = group["head_direction"].eq(direction)
            count = int(selected.sum())
            detected_count = int((selected & group["face_detected"]).sum())
            rows.append({
                "camera": camera,
                "performer": performer,
                "head_direction": direction,
                "frames": count,
                "duration_sec": count * frame_period,
                "percent_of_all_frames": 100 * count / total if total else np.nan,
                "frames_with_face_detected": detected_count,
                "percent_when_face_detected": (
                    100 * detected_count / detected_total if detected_total else np.nan
                ),
                "total_frames": total,
                "face_detected_frames": detected_total,
            })
    return pd.DataFrame(rows)


def unknown_reason_summary(timeline: pd.DataFrame) -> pd.DataFrame:
    """Count why frames remained unknown, including known zero-count reason classes."""
    rows: list[dict[str, Any]] = []
    frame_period = float(1 / FPS)
    for (camera, performer), group in timeline.groupby(["camera", "performer"], sort=False):
        unknown = group[group["head_direction"].eq("unknown")]
        total = len(group)
        unknown_total = len(unknown)
        observed = [str(item) for item in unknown["unknown_reason"].dropna().unique()]
        reasons = list(dict.fromkeys([*UNKNOWN_REASONS, *observed]))
        for reason in reasons:
            count = int(unknown["unknown_reason"].eq(reason).sum())
            rows.append({
                "camera": camera,
                "performer": performer,
                "unknown_reason": reason,
                "frames": count,
                "duration_sec": count * frame_period,
                "percent_of_all_frames": 100 * count / total if total else np.nan,
                "percent_of_unknown_frames": 100 * count / unknown_total if unknown_total else np.nan,
                "total_frames": total,
                "unknown_frames": unknown_total,
            })
    return pd.DataFrame(rows)


def mutual_facing_timeline(
    timeline: pd.DataFrame, camera: str = "cam_wide"
) -> pd.DataFrame:
    """Put both performers on one row and mark simultaneous partner orientation."""
    selected = timeline[timeline["camera"].eq(camera)]
    base = selected[["frame_idx", "time_sec"]].drop_duplicates().sort_values("frame_idx")
    result = base.copy()
    for performer in ("bassist", "guitarist"):
        person = selected[selected["performer"].eq(performer)].set_index("frame_idx")
        direction = person["head_direction"] if "head_direction" in person else pd.Series(dtype=object)
        detected = person["face_detected"] if "face_detected" in person else pd.Series(dtype=bool)
        result[f"{performer}_direction"] = result["frame_idx"].map(direction).fillna("unknown")
        result[f"{performer}_face_detected"] = (
            result["frame_idx"].map(detected).fillna(False).astype(bool)
        )
    bassist_partner = result["bassist_direction"].eq("partner")
    guitarist_partner = result["guitarist_direction"].eq("partner")
    result["mutual_facing"] = bassist_partner & guitarist_partner
    result["one_sided_partner"] = bassist_partner ^ guitarist_partner
    result["partner_orientation"] = np.select(
        [result["mutual_facing"], bassist_partner, guitarist_partner],
        ["mutual", "bassist_only", "guitarist_only"],
        default="none",
    )
    return result.reset_index(drop=True)


def _boolean_intervals(timeline: pd.DataFrame, column: str) -> pd.DataFrame:
    selected = timeline[timeline[column]].copy()
    columns = ["start_frame", "end_frame", "start_sec", "end_sec", "frames", "duration_sec"]
    if selected.empty:
        return pd.DataFrame(columns=columns)
    selected["interval"] = selected["frame_idx"].diff().ne(1).cumsum()
    intervals = selected.groupby("interval", as_index=False).agg(
        start_frame=("frame_idx", "min"),
        end_frame=("frame_idx", "max"),
        start_sec=("time_sec", "min"),
        end_sec=("time_sec", "max"),
        frames=("frame_idx", "size"),
    )
    intervals["duration_sec"] = intervals["frames"] / float(FPS)
    return intervals[columns]


def partner_orientation_intervals(timeline: pd.DataFrame) -> pd.DataFrame:
    """Return contiguous mutual or one-sided partner-direction runs."""
    selected = timeline[timeline["partner_orientation"].ne("none")].copy()
    columns = [
        "orientation", "start_frame", "end_frame", "start_sec", "end_sec", "frames", "duration_sec"
    ]
    if selected.empty:
        return pd.DataFrame(columns=columns)
    selected["interval"] = (
        selected["frame_idx"].diff().ne(1)
        | selected["partner_orientation"].ne(selected["partner_orientation"].shift())
    ).cumsum()
    intervals = selected.groupby("interval", as_index=False).agg(
        orientation=("partner_orientation", "first"),
        start_frame=("frame_idx", "min"),
        end_frame=("frame_idx", "max"),
        start_sec=("time_sec", "min"),
        end_sec=("time_sec", "max"),
        frames=("frame_idx", "size"),
    )
    intervals["duration_sec"] = intervals["frames"] / float(FPS)
    return intervals[columns]


def mutual_facing_summary(timeline: pd.DataFrame, camera: str) -> pd.DataFrame:
    total = len(timeline)
    both_detected = timeline["bassist_face_detected"] & timeline["guitarist_face_detected"]
    mutual = timeline["mutual_facing"]
    one_sided = timeline["one_sided_partner"]
    frame_period = float(1 / FPS)
    return pd.DataFrame([{
        "camera": camera,
        "definition": "both performers are classified as partner in the same frame",
        "total_frames": total,
        "both_faces_detected_frames": int(both_detected.sum()),
        "mutual_facing_frames": int(mutual.sum()),
        "mutual_facing_duration_sec": float(mutual.sum()) * frame_period,
        "mutual_facing_percent_of_all_frames": 100 * float(mutual.mean()) if total else np.nan,
        "mutual_facing_percent_when_both_faces_detected": (
            100 * float(mutual[both_detected].mean()) if both_detected.any() else np.nan
        ),
        "one_sided_partner_frames": int(one_sided.sum()),
        "one_sided_partner_duration_sec": float(one_sided.sum()) * frame_period,
    }])


def write_head_direction_report(
    heads: pd.DataFrame,
    output_directory: str | Path,
    camera: str = "cam_wide",
    parquet: bool = True,
    gaze_config: dict | None = None,
) -> dict[str, Any]:
    output = Path(output_directory)
    if gaze_config is not None:
        heads = reclassify_reference_angles(heads, gaze_config)
    timeline = head_direction_timeline(heads)
    summary = head_direction_summary(timeline)
    reason_summary = unknown_reason_summary(timeline)
    mutual_timeline = mutual_facing_timeline(timeline, camera)
    mutual_intervals = _boolean_intervals(mutual_timeline, "mutual_facing")
    orientation_intervals = partner_orientation_intervals(mutual_timeline)
    facing_summary = mutual_facing_summary(mutual_timeline, camera)
    tables = {
        "head_direction_by_time": timeline,
        "head_direction_summary": summary,
        "unknown_reason_summary": reason_summary,
        "mutual_facing_by_time": mutual_timeline,
        "mutual_facing_intervals": mutual_intervals,
        "partner_orientation_intervals": orientation_intervals,
        "mutual_facing_summary": facing_summary,
    }
    written = {
        name: [str(path) for path in write_table(table, output / name, parquet)]
        for name, table in tables.items()
    }
    return {
        "output_directory": str(output.resolve()),
        "camera_for_mutual_facing": camera,
        "rows": {name: len(table) for name, table in tables.items()},
        "mutual_facing_frames": int(mutual_timeline["mutual_facing"].sum()),
        "one_sided_partner_frames": int(mutual_timeline["one_sided_partner"].sum()),
        "files": written,
    }

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import cv2
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "musician_interaction_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .types import INSTRUMENT_POINTS, InstrumentPose, PoseDetection


SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)]
COLORS = {"guitarist": (0, 210, 255), "bassist": (255, 150, 0)}


def write_table(table: pd.DataFrame, base: Path, parquet: bool) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    csv_path = base.with_suffix(".csv")
    table.to_csv(csv_path, index=False)
    paths.append(csv_path)
    if parquet:
        try:
            parquet_path = base.with_suffix(".parquet")
            table.to_parquet(parquet_path, index=False)
            paths.append(parquet_path)
        except (ImportError, ValueError):
            pass
    return paths


def draw_overlay(frame: np.ndarray, assigned: dict[str, PoseDetection | None], instruments: dict[str, InstrumentPose],
                 heads: dict[str, object | None], gaze: dict[str, str], score_threshold: float) -> np.ndarray:
    canvas = frame.copy()
    for performer, detection in assigned.items():
        if detection is None:
            continue
        color = COLORS.get(performer, (0, 255, 0))
        for a, b in SKELETON:
            if max(a, b) < len(detection.scores) and min(detection.scores[a], detection.scores[b]) >= score_threshold:
                cv2.line(canvas, tuple(np.rint(detection.keypoints[a]).astype(int)),
                         tuple(np.rint(detection.keypoints[b]).astype(int)), color, 2)
        anchor = detection.center.astype(int)
        cv2.putText(canvas, f"{performer} gaze={gaze.get(performer, 'unknown')}", tuple(anchor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    for instrument, pose in instruments.items():
        for idx, point in enumerate(pose.points):
            if np.isfinite(point).all():
                cv2.circle(canvas, tuple(np.rint(point).astype(int)), 5, (255, 0, 255), -1)
                cv2.putText(canvas, f"{instrument}:{INSTRUMENT_POINTS[idx]}", tuple(np.rint(point).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
    return canvas


class OverlayWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float, scale: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (int(round(width * scale)), int(round(height * scale)))
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create overlay video: {path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA))

    def close(self) -> None:
        self.writer.release()


def save_graphs(motion: pd.DataFrame, audio: pd.DataFrame, cross: pd.DataFrame, event: pd.DataFrame,
                directory: Path, instrument_motion: pd.DataFrame | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(audio["time_sec"], audio["rms"], label="RMS")
    axes[0].plot(audio["time_sec"], audio["spectral_flux"], alpha=.7, label="spectral flux")
    axes[0].legend(); axes[0].set_ylabel("audio feature")
    upper = motion[motion["feature"] == "upper_body"]
    for keys, data in upper.groupby(["camera", "performer"]):
        axes[1].plot(data["time_sec"], data["speed_norm_s"], alpha=.7, label=f"{keys[0]} {keys[1]}")
    axes[1].legend(ncol=2, fontsize=8); axes[1].set_ylabel("normalized speed / s"); axes[1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(directory / "audio_and_motion.png", dpi=150); plt.close(fig)
    if not cross.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for camera, data in cross.groupby("camera"):
            ax.plot(data["lag_sec"], data["correlation"], label=camera)
        ax.axvline(0, color="black", lw=.8); ax.legend(); ax.set(xlabel="lag (s)", ylabel="correlation")
        fig.tight_layout(); fig.savefig(directory / "performer_cross_correlation.png", dpi=150); plt.close(fig)
    if not event.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for keys, data in event[event["feature"] == "upper_body"].groupby(["camera", "performer"]):
            ax.plot(data["offset_sec"], data["mean_speed_norm_s"], label=f"{keys[0]} {keys[1]}")
        ax.axvline(0, color="black", lw=.8); ax.legend(fontsize=8); ax.set(xlabel="time from onset (s)", ylabel="event mean speed")
        fig.tight_layout(); fig.savefig(directory / "onset_triggered_average.png", dpi=150); plt.close(fig)
    if instrument_motion is not None and not instrument_motion.empty:
        usable = instrument_motion[instrument_motion["axis_angle_deg"].notna()]
        if not usable.empty:
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            for keys, data in usable.groupby(["camera", "instrument"]):
                label = f"{keys[0]} {keys[1]}"
                axes[0].plot(data["time_sec"], data["axis_angle_deg"], alpha=.7, label=label)
                axes[1].plot(data["time_sec"], data["head_tip_height_norm"], alpha=.7, label=label)
            axes[0].set_ylabel("instrument axis angle (deg)"); axes[0].legend(fontsize=8, ncol=2)
            axes[1].set_ylabel("head-tip height (normalized)"); axes[1].set_xlabel("time (s)")
            fig.tight_layout(); fig.savefig(directory / "instrument_orientation_and_height.png", dpi=150); plt.close(fig)


def write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

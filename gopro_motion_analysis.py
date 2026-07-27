#!/usr/bin/env python3
"""GoPro RGB video + MediaPipe Pose Landmarker movement analysis.

Collected values:
  * head sway relative to the shoulder centre (normalised by shoulder width)
  * signed forward lean of the upper body
  * synchronisation between motion peaks and musical beats

Press Q or Esc to stop a live capture.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# MediaPipe Pose landmark indices.
NOSE = 0
LEFT_EYE = 2
RIGHT_EYE = 5
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_HIP = 23
RIGHT_HIP = 24

UPPER_BODY_CONNECTIONS = (
    (LEFT_EAR, LEFT_EYE),
    (LEFT_EYE, NOSE),
    (NOSE, RIGHT_EYE),
    (RIGHT_EYE, RIGHT_EAR),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
)

CSV_FIELDS = (
    "frame",
    "time_s",
    "pose_detected",
    "head_x_shoulder_width",
    "head_y_shoulder_width",
    "head_sway_from_baseline",
    "head_frame_displacement",
    "forward_lean_deg",
    "total_torso_tilt_deg",
    "is_forward_leaning",
    "motion_energy",
    "motion_peak",
    "nearest_beat_error_ms",
    "on_beat",
)


@dataclass
class AnalysisConfig:
    view: str
    visibility_threshold: float
    forward_lean_threshold_deg: float
    baseline_seconds: float
    smoothing_seconds: float


def midpoint(a: Any, b: Any) -> np.ndarray:
    """Return the midpoint of two MediaPipe landmarks."""
    return np.array(
        [(a.x + b.x) / 2.0, (a.y + b.y) / 2.0, (a.z + b.z) / 2.0],
        dtype=float,
    )


def landmark_visible(landmark: Any, threshold: float) -> bool:
    """Tasks landmarks may expose visibility/presence as None."""
    visibility = 1.0 if landmark.visibility is None else landmark.visibility
    presence = 1.0 if landmark.presence is None else landmark.presence
    return visibility >= threshold and presence >= threshold


def mean_visible_landmarks(
    landmarks: Sequence[Any], indices: Sequence[int], threshold: float
) -> np.ndarray | None:
    points = [
        np.array([landmarks[i].x, landmarks[i].y, landmarks[i].z], dtype=float)
        for i in indices
        if landmark_visible(landmarks[i], threshold)
    ]
    return np.mean(points, axis=0) if points else None


def signed_forward_lean_deg(
    shoulder: np.ndarray, hip: np.ndarray, view: str
) -> float:
    """Calculate signed torso pitch; positive means leaning forward.

    ``front`` assumes the subject faces the camera. For side views, the name
    indicates the direction in which the subject faces in the image.
    """
    torso = shoulder - hip
    vertical = max(-float(torso[1]), 1e-9)
    if view == "front":
        forward = -float(torso[2])  # MediaPipe -z points towards camera.
    elif view == "side-right":
        forward = float(torso[0])
    elif view == "side-left":
        forward = -float(torso[0])
    else:
        raise ValueError(f"Unknown view: {view}")
    return math.degrees(math.atan2(forward, vertical))


def total_torso_tilt_deg(shoulder: np.ndarray, hip: np.ndarray) -> float:
    torso = shoulder - hip
    vertical = max(-float(torso[1]), 1e-9)
    horizontal = math.hypot(float(torso[0]), float(torso[2]))
    return math.degrees(math.atan2(horizontal, vertical))


class MotionAnalyzer:
    """Stateful per-frame motion metric calculator."""

    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self.smoothed_head: np.ndarray | None = None
        self.smoothed_lean: float | None = None
        self.previous_time: float | None = None
        self.previous_head: np.ndarray | None = None
        self.baseline_samples: list[np.ndarray] = []
        self.baseline: np.ndarray | None = None

    def _ema_alpha(self, time_s: float) -> float:
        if self.previous_time is None or self.config.smoothing_seconds <= 0:
            return 1.0
        dt = max(time_s - self.previous_time, 1e-6)
        return 1.0 - math.exp(-dt / self.config.smoothing_seconds)

    def process(
        self,
        image_landmarks: Sequence[Any],
        world_landmarks: Sequence[Any],
        time_s: float,
    ) -> dict[str, float | int]:
        threshold = self.config.visibility_threshold
        required = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
        if not all(landmark_visible(image_landmarks[i], threshold) for i in required):
            raise ValueError("Required torso landmarks are not sufficiently visible")

        head = mean_visible_landmarks(
            image_landmarks,
            (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR),
            threshold,
        )
        if head is None:
            raise ValueError("Head landmarks are not sufficiently visible")

        shoulder_image = midpoint(
            image_landmarks[LEFT_SHOULDER], image_landmarks[RIGHT_SHOULDER]
        )
        shoulder_width = math.dist(
            (image_landmarks[LEFT_SHOULDER].x, image_landmarks[LEFT_SHOULDER].y),
            (image_landmarks[RIGHT_SHOULDER].x, image_landmarks[RIGHT_SHOULDER].y),
        )
        if shoulder_width < 1e-4:
            raise ValueError("Shoulder width is too small")

        # Translation and distance-to-camera changes mostly cancel here.
        relative_head = (head[:2] - shoulder_image[:2]) / shoulder_width
        alpha = self._ema_alpha(time_s)
        if self.smoothed_head is None:
            self.smoothed_head = relative_head
        else:
            self.smoothed_head += alpha * (relative_head - self.smoothed_head)

        shoulder_world = midpoint(
            world_landmarks[LEFT_SHOULDER], world_landmarks[RIGHT_SHOULDER]
        )
        hip_world = midpoint(world_landmarks[LEFT_HIP], world_landmarks[RIGHT_HIP])
        lean = signed_forward_lean_deg(
            shoulder_world, hip_world, self.config.view
        )
        tilt = total_torso_tilt_deg(shoulder_world, hip_world)
        if self.smoothed_lean is None:
            self.smoothed_lean = lean
        else:
            self.smoothed_lean += alpha * (lean - self.smoothed_lean)

        if time_s <= self.config.baseline_seconds or not self.baseline_samples:
            self.baseline_samples.append(self.smoothed_head.copy())
        if self.baseline is None and time_s >= self.config.baseline_seconds:
            self.baseline = np.median(np.array(self.baseline_samples), axis=0)
        baseline = (
            self.baseline
            if self.baseline is not None
            else np.median(np.array(self.baseline_samples), axis=0)
        )

        displacement = (
            0.0
            if self.previous_head is None
            else float(np.linalg.norm(self.smoothed_head - self.previous_head))
        )
        sway = float(np.linalg.norm(self.smoothed_head - baseline))

        self.previous_head = self.smoothed_head.copy()
        self.previous_time = time_s
        return {
            "head_x_shoulder_width": float(self.smoothed_head[0]),
            "head_y_shoulder_width": float(self.smoothed_head[1]),
            "head_sway_from_baseline": sway,
            "head_frame_displacement": displacement,
            "forward_lean_deg": float(self.smoothed_lean),
            "total_torso_tilt_deg": tilt,
            "is_forward_leaning": int(
                self.smoothed_lean >= self.config.forward_lean_threshold_deg
            ),
        }


def beats_from_bpm(bpm: float, duration_s: float, offset_s: float) -> np.ndarray:
    if bpm <= 0:
        raise ValueError("BPM must be greater than zero")
    interval = 60.0 / bpm
    return np.arange(offset_s, duration_s + interval, interval, dtype=float)


def load_beat_times(path: Path) -> np.ndarray:
    """Load one beat time in seconds per line (a header is tolerated)."""
    beats: list[float] = []
    with path.open(encoding="utf-8-sig") as file:
        for row in csv.reader(file):
            if not row:
                continue
            try:
                beats.append(float(row[0]))
            except ValueError:
                continue
    if not beats:
        raise ValueError(f"No beat times found in {path}")
    return np.array(sorted(set(beats)), dtype=float)


def detect_audio_beats(path: Path) -> tuple[np.ndarray, float]:
    """Detect beats with librosa. GoPro MP4 decoding may require ffmpeg."""
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "--music requires librosa. Install requirements.txt first."
        ) from exc
    audio, sample_rate = librosa.load(str(path), sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
    tempo_value = float(np.asarray(tempo).reshape(-1)[0])
    if len(beat_times) < 2:
        raise RuntimeError("Could not detect enough beats from the music")
    return np.asarray(beat_times, dtype=float), tempo_value


def robust_standardize(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.zeros_like(values, dtype=float)
    if not finite.any():
        return result
    sample = values[finite]
    centre = np.median(sample)
    scale = np.median(np.abs(sample - centre)) * 1.4826
    if scale < 1e-9:
        scale = np.std(sample)
    if scale >= 1e-9:
        # These inputs are non-negative motion magnitudes. Values below the
        # resting median are not motion events and therefore contribute zero.
        result[finite] = np.maximum((sample - centre) / scale, 0.0)
    return result


def smooth_signal(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def find_motion_peaks(
    times: np.ndarray, energy: np.ndarray, beat_times: np.ndarray
) -> np.ndarray:
    if len(energy) < 3:
        return np.array([], dtype=int)
    finite_dt = np.diff(times)
    finite_dt = finite_dt[np.isfinite(finite_dt) & (finite_dt > 0)]
    fps = 30.0 if not len(finite_dt) else 1.0 / np.median(finite_dt)
    if len(beat_times) >= 2:
        minimum_seconds = max(0.15, 0.45 * np.median(np.diff(beat_times)))
    else:
        minimum_seconds = 0.25
    minimum_frames = max(1, int(round(minimum_seconds * fps)))
    threshold = np.median(energy) + 0.5 * np.std(energy)
    candidates = np.flatnonzero(
        (energy[1:-1] > energy[:-2])
        & (energy[1:-1] >= energy[2:])
        & (energy[1:-1] >= threshold)
    ) + 1

    selected: list[int] = []
    for index in candidates[np.argsort(energy[candidates])[::-1]]:
        if all(abs(int(index) - old) >= minimum_frames for old in selected):
            selected.append(int(index))
    return np.array(sorted(selected), dtype=int)


def beat_synchronisation(
    motion_peak_times: np.ndarray, beat_times: np.ndarray
) -> dict[str, float | int | None]:
    if len(motion_peak_times) == 0 or len(beat_times) < 2:
        return {
            "motion_peak_count": int(len(motion_peak_times)),
            "beat_count": int(len(beat_times)),
            "mean_abs_beat_error_ms": None,
            "on_beat_fraction_100ms": None,
            "beat_sync_score": None,
        }
    errors = np.array(
        [np.min(np.abs(beat_times - peak)) for peak in motion_peak_times],
        dtype=float,
    )
    half_beat = max(float(np.median(np.diff(beat_times))) / 2.0, 1e-6)
    score = 1.0 - float(np.mean(np.minimum(errors / half_beat, 1.0)))
    return {
        "motion_peak_count": int(len(motion_peak_times)),
        "beat_count": int(len(beat_times)),
        "mean_abs_beat_error_ms": float(np.mean(errors) * 1000.0),
        "on_beat_fraction_100ms": float(np.mean(errors <= 0.1)),
        "beat_sync_score": score,
    }


def add_motion_and_beat_metrics(
    rows: list[dict[str, Any]], beat_times: np.ndarray
) -> dict[str, float | int | None]:
    valid_indices = np.array(
        [i for i, row in enumerate(rows) if row["pose_detected"]], dtype=int
    )
    if not len(valid_indices):
        return beat_synchronisation(np.array([]), beat_times)
    times = np.array([rows[i]["time_s"] for i in valid_indices], dtype=float)
    head_motion = np.array(
        [rows[i]["head_frame_displacement"] for i in valid_indices], dtype=float
    )
    lean = np.array(
        [rows[i]["forward_lean_deg"] for i in valid_indices], dtype=float
    )
    lean_velocity = np.r_[0.0, np.abs(np.diff(lean))]
    energy = robust_standardize(head_motion) + robust_standardize(lean_velocity)
    if len(times) > 1:
        fps = 1.0 / max(np.median(np.diff(times)), 1e-6)
    else:
        fps = 30.0
    energy = smooth_signal(energy, max(1, int(round(fps * 0.10))))
    peaks = find_motion_peaks(times, energy, beat_times)

    for local_index, row_index in enumerate(valid_indices):
        rows[row_index]["motion_energy"] = float(energy[local_index])
        rows[row_index]["motion_peak"] = 0
    for local_index in peaks:
        row = rows[int(valid_indices[local_index])]
        row["motion_peak"] = 1
        if len(beat_times):
            error_s = float(np.min(np.abs(beat_times - row["time_s"])))
            row["nearest_beat_error_ms"] = error_s * 1000.0
            row["on_beat"] = int(error_s <= 0.1)
    return beat_synchronisation(times[peaks], beat_times)


def summarise(
    rows: list[dict[str, Any]],
    sync: dict[str, float | int | None],
    detected_tempo: float | None,
) -> dict[str, Any]:
    valid = [row for row in rows if row["pose_detected"]]
    duration = float(rows[-1]["time_s"]) if rows else 0.0
    summary: dict[str, Any] = {
        "duration_s": duration,
        "total_frames": len(rows),
        "pose_detected_frames": len(valid),
        "pose_detection_fraction": len(valid) / len(rows) if rows else 0.0,
        "detected_music_tempo_bpm": detected_tempo,
        **sync,
    }
    if not valid:
        return summary
    sway = np.array([row["head_sway_from_baseline"] for row in valid])
    displacement = np.array([row["head_frame_displacement"] for row in valid])
    lean = np.array([row["forward_lean_deg"] for row in valid])
    forward = np.array([row["is_forward_leaning"] for row in valid])
    summary.update(
        {
            "head_sway_rms_shoulder_width": float(np.sqrt(np.mean(sway**2))),
            "head_sway_p95_shoulder_width": float(np.percentile(sway, 95)),
            "head_path_length_shoulder_width": float(np.sum(displacement)),
            "forward_lean_mean_deg": float(np.mean(lean)),
            "forward_lean_max_deg": float(np.max(lean)),
            "forward_lean_fraction": float(np.mean(forward)),
            "forward_lean_event_count": int(
                np.sum((forward[1:] == 1) & (forward[:-1] == 0))
                + int(forward[0] == 1)
            ),
        }
    )
    return summary


def ensure_model(path: Path, allow_download: bool) -> None:
    if path.exists():
        return
    if not allow_download:
        raise FileNotFoundError(
            f"Model not found: {path}. Remove --no-download or pass --model."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"MediaPipe modelをダウンロードしています: {MODEL_URL}", file=sys.stderr)
    try:
        urllib.request.urlretrieve(MODEL_URL, partial)
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def parse_source(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def source_is_live(source: int | str) -> bool:
    if isinstance(source, int):
        return True
    return "://" in source


def prepare_stream_url(source: str) -> str:
    """Add the receive buffer options recommended by the Open GoPro demo."""
    if source.startswith("udp://") and "overrun_nonfatal=" not in source:
        separator = "&" if "?" in source else "?"
        return (
            f"{source}{separator}"
            "overrun_nonfatal=1&fifo_size=50000000"
        )
    return source


class UsbGoProStream:
    """Start and keep an Open GoPro wired webcam stream alive.

    The SDK is asynchronous, while OpenCV/MediaPipe processing is synchronous.
    A small background thread owns the SDK event loop for the whole capture.
    """

    def __init__(
        self,
        identifier: str | None = None,
        protocol: str = "RTSP",
        startup_timeout_s: float = 30.0,
    ) -> None:
        self.identifier = identifier
        self.protocol = protocol.upper()
        self.startup_timeout_s = startup_timeout_s
        self.source: str | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._stopped = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    async def _serve(self) -> None:
        try:
            try:
                from open_gopro import WiredGoPro
                from open_gopro.models.streaming import (
                    StreamType,
                    WebcamProtocol,
                    WebcamStreamOptions,
                )
                from returns.pipeline import is_successful
            except ImportError as exc:
                raise RuntimeError(
                    "USB自動接続にはopen-goproが必要です。"
                    "`python -m pip install -r requirements-motion.txt`を実行してください。"
                ) from exc

            protocol = (
                WebcamProtocol.RTSP
                if self.protocol == "RTSP"
                else WebcamProtocol.TS
            )
            async with WiredGoPro(self.identifier) as gopro:
                result = await gopro.streaming.start_stream(
                    stream_type=StreamType.WEBCAM,
                    options=WebcamStreamOptions(protocol=protocol),
                )
                if not is_successful(result):
                    raise RuntimeError(
                        f"GoPro Webcamストリームを開始できません: {result.failure()}"
                    )
                if not gopro.streaming.url:
                    raise RuntimeError("GoProからストリームURLが返されませんでした")
                self.source = str(gopro.streaming.url)
                self._ready.set()
                while not self._stop.is_set():
                    await asyncio.sleep(0.1)
                await gopro.streaming.stop_active_stream()
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            self._stopped.set()

    def _thread_main(self) -> None:
        asyncio.run(self._serve())

    def __enter__(self) -> str:
        self._thread = threading.Thread(
            target=self._thread_main,
            name="open-gopro-usb",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(self.startup_timeout_s):
            self._stop.set()
            raise RuntimeError(
                "GoProのUSB検出がタイムアウトしました。電源、USBケーブル、"
                "対応機種、macOSのネットワーク許可を確認してください。"
            )
        if self._error is not None:
            raise RuntimeError(str(self._error)) from self._error
        if self.source is None:
            raise RuntimeError("GoPro USBストリームを取得できませんでした")
        print(
            f"GoPro USB Webcam開始 ({self.protocol}): {self.source}",
            file=sys.stderr,
        )
        return self.source

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        self._stopped.wait(15.0)
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if exc_type is None and self._error is not None:
            raise RuntimeError(str(self._error)) from self._error


def draw_overlay(
    cv2: Any,
    frame: np.ndarray,
    image_landmarks: Sequence[Any] | None,
    metrics: dict[str, Any],
    threshold: float,
) -> None:
    height, width = frame.shape[:2]
    if image_landmarks is not None:
        for start, end in UPPER_BODY_CONNECTIONS:
            if landmark_visible(image_landmarks[start], threshold) and landmark_visible(
                image_landmarks[end], threshold
            ):
                a = image_landmarks[start]
                b = image_landmarks[end]
                cv2.line(
                    frame,
                    (round(a.x * width), round(a.y * height)),
                    (round(b.x * width), round(b.y * height)),
                    (60, 220, 60),
                    2,
                )
    if metrics:
        lines = (
            f"head sway: {metrics.get('head_sway_from_baseline', 0):.3f} shoulder",
            f"forward lean: {metrics.get('forward_lean_deg', 0):.1f} deg",
            "Q / Esc: stop",
        )
    else:
        lines = ("Pose not detected", "Q / Esc: stop")
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (15, 30 + 28 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (15, 30 + 28 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GoPro映像から頭揺れ・上体前傾・拍同期を収集します。"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="OpenCV入力: カメラ番号、MP4パス、RTSP/UDP URL (default: 0)",
    )
    parser.add_argument(
        "--gopro-usb",
        action="store_true",
        help="Open GoPro SDKでUSB接続したGoPro Webcamを自動開始",
    )
    parser.add_argument(
        "--gopro-id",
        help="複数台接続時のGoPro識別子（通常は省略）",
    )
    parser.add_argument(
        "--gopro-protocol",
        choices=("TS", "RTSP"),
        default="RTSP",
        help="USB Webcamプロトコル (default: RTSP)",
    )
    parser.add_argument(
        "--view",
        choices=("front", "side-right", "side-left"),
        default="front",
        help="front=正面、side-right/left=被写体が画面内で向く方向",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/motion_metrics.csv"),
        help="フレーム別CSVの保存先 (default: out/motion_metrics.csv)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help=(
            "集計JSONの保存先。省略時は"
            "out/motion_summary_YYYYMMDD_HHMMSS.json"
        ),
    )
    parser.add_argument(
        "--record-video",
        type=Path,
        help=(
            "解析と同時に受信映像をMP4へ保存（音声なし）。"
            "--gopro-usbでは省略時もout/へ自動保存"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pose_landmarker_full.task"),
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--music", type=Path, help="拍検出する音声またはGoPro MP4")
    parser.add_argument("--beat-times", type=Path, help="拍時刻(秒)を1列で持つCSV")
    parser.add_argument("--bpm", type=float, help="ライブ時などの既知BPM")
    parser.add_argument("--beat-offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, help="収集秒数（省略時は末尾/Qまで）")
    parser.add_argument("--forward-threshold", type=float, default=15.0)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--baseline-seconds", type=float, default=2.0)
    parser.add_argument("--smoothing-seconds", type=float, default=0.12)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--no-preview", action="store_true")
    return parser


def assign_default_session_paths(
    args: argparse.Namespace,
    started_at: datetime | None = None,
) -> tuple[Path, Path | None]:
    """Assign timestamped JSON/video paths without overwriting old sessions."""
    timestamp = (started_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("out")
    needs_summary = args.summary is None
    needs_recording = args.gopro_usb and args.record_video is None
    sequence = 2

    suffix = timestamp
    while True:
        summary_candidate = output_dir / f"motion_summary_{suffix}.json"
        recording_candidate = output_dir / f"gopro_capture_{suffix}.mp4"
        summary_collision = needs_summary and summary_candidate.exists()
        recording_collision = needs_recording and recording_candidate.exists()
        if not summary_collision and not recording_collision:
            break
        suffix = f"{timestamp}_{sequence}"
        sequence += 1

    if needs_summary:
        args.summary = summary_candidate
    if needs_recording:
        args.record_video = recording_candidate
    return args.summary, args.record_video


def validate_args(args: argparse.Namespace) -> None:
    beat_sources = sum(
        value is not None for value in (args.music, args.beat_times, args.bpm)
    )
    if beat_sources > 1:
        raise ValueError("--music, --beat-times, --bpm は1つだけ指定してください")
    if args.duration is not None and args.duration <= 0:
        raise ValueError("--duration は0より大きくしてください")
    if not 0 <= args.visibility_threshold <= 1:
        raise ValueError("--visibility-threshold は0〜1で指定してください")
    if args.gopro_id and not args.gopro_usb:
        raise ValueError("--gopro-id は --gopro-usb と一緒に指定してください")


def run_capture(args: argparse.Namespace, source: int | str) -> dict[str, Any]:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV/MediaPipeがありません。Python 3.11環境で "
            "`python -m pip install -r requirements-motion.txt` を実行してください。"
        ) from exc

    ensure_model(args.model, not args.no_download)
    live = source_is_live(source)
    if isinstance(source, str) and "://" in source:
        capture_source = prepare_stream_url(source)
        capture = cv2.VideoCapture(
            capture_source,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                15_000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                5_000,
            ],
        )
    else:
        capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    if not capture.isOpened():
        raise RuntimeError(f"映像入力を開けません: {source}")

    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(actual_fps) or actual_fps <= 1:
        actual_fps = args.fps
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    estimated_duration = (
        args.duration
        if args.duration is not None
        else total_frames / actual_fps if not live and total_frames > 0 else 3600.0
    )

    detected_tempo: float | None = None
    if args.music:
        beat_times, detected_tempo = detect_audio_beats(args.music)
    elif args.beat_times:
        beat_times = load_beat_times(args.beat_times)
    elif args.bpm:
        beat_times = beats_from_bpm(args.bpm, estimated_duration, args.beat_offset)
        detected_tempo = args.bpm
    else:
        beat_times = np.array([], dtype=float)

    config = AnalysisConfig(
        view=args.view,
        visibility_threshold=args.visibility_threshold,
        forward_lean_threshold_deg=args.forward_threshold,
        baseline_seconds=args.baseline_seconds,
        smoothing_seconds=args.smoothing_seconds,
    )
    analyzer = MotionAnalyzer(config)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model.resolve())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows: list[dict[str, Any]] = []
    video_writer = None
    frame_index = 0
    last_timestamp_ms = -1
    started = time.monotonic()
    try:
        try:
            with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if args.record_video is not None:
                        if video_writer is None:
                            args.record_video.parent.mkdir(parents=True, exist_ok=True)
                            frame_height, frame_width = frame.shape[:2]
                            video_writer = cv2.VideoWriter(
                                str(args.record_video),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                actual_fps,
                                (frame_width, frame_height),
                            )
                            if not video_writer.isOpened():
                                raise RuntimeError(
                                    f"録画ファイルを作成できません: {args.record_video}"
                                )
                        video_writer.write(frame)
                    if live:
                        time_s = time.monotonic() - started
                    else:
                        time_s = frame_index / actual_fps
                    if args.duration is not None and time_s > args.duration:
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = max(
                        last_timestamp_ms + 1, int(round(time_s * 1000))
                    )
                    last_timestamp_ms = timestamp_ms
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    row: dict[str, Any] = {
                        "frame": frame_index,
                        "time_s": time_s,
                        "pose_detected": 0,
                    }
                    displayed_landmarks = None
                    if result.pose_landmarks and result.pose_world_landmarks:
                        displayed_landmarks = result.pose_landmarks[0]
                        try:
                            row.update(
                                analyzer.process(
                                    result.pose_landmarks[0],
                                    result.pose_world_landmarks[0],
                                    time_s,
                                )
                            )
                            row["pose_detected"] = 1
                        except ValueError:
                            pass
                    rows.append(row)

                    if not args.no_preview:
                        draw_overlay(
                            cv2,
                            frame,
                            displayed_landmarks,
                            row if row["pose_detected"] else {},
                            args.visibility_threshold,
                        )
                        cv2.imshow("GoPro MediaPipe Motion Analysis", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                    frame_index += 1
        except KeyboardInterrupt:
            print(
                "\n終了要求を受信しました。結果を保存しています...",
                file=sys.stderr,
            )
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    sync = add_motion_and_beat_metrics(rows, beat_times)
    summary = summarise(rows, sync, detected_tempo)
    summary.update(
        {
            "source": str(source),
            "camera_view": args.view,
            "forward_lean_threshold_deg": args.forward_threshold,
            "recorded_video": (
                str(args.record_video) if args.record_video is not None else None
            ),
            "beat_method": (
                "audio"
                if args.music
                else "beat_times"
                if args.beat_times
                else "bpm"
                if args.bpm
                else "none"
            ),
        }
    )
    write_csv(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    assign_default_session_paths(args)
    if args.gopro_usb:
        # Fail before starting the camera, and download the model before the
        # stream starts so the first USB session is not left waiting.
        try:
            import cv2  # noqa: F401
            import mediapipe  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV/MediaPipeがありません。Python 3.11環境で "
                "`python -m pip install -r requirements-motion.txt` "
                "を実行してください。"
            ) from exc
        ensure_model(args.model, not args.no_download)
        with UsbGoProStream(
            identifier=args.gopro_id,
            protocol=args.gopro_protocol,
        ) as stream_source:
            return run_capture(args, stream_source)
    return run_capture(args, parse_source(args.source))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        summary = run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nCSV: {args.output}\n集計: {args.summary}")
    if args.record_video is not None:
        print(f"映像: {args.record_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

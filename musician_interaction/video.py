from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import subprocess
from typing import Iterator

import cv2


FPS = Fraction(30000, 1001)


@dataclass(frozen=True)
class VideoInfo:
    camera: str
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int


def time_sec(frame_idx: int) -> float:
    return frame_idx * 1001.0 / 30000.0


def probe_timecodes(path: Path) -> set[str]:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream_tags=timecode:format_tags=timecode",
         "-of", "json", str(path)], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    values = set()
    for stream in payload.get("streams", []):
        if stream.get("tags", {}).get("timecode"):
            values.add(stream["tags"]["timecode"])
    if payload.get("format", {}).get("tags", {}).get("timecode"):
        values.add(payload["format"]["tags"]["timecode"])
    return values


class SynchronizedVideoReader:
    """Read all cameras once per index and stop on the first failed stream."""

    def __init__(self, camera_paths: dict[str, Path], expected_fps: float = float(FPS)) -> None:
        if len(camera_paths) < 1:
            raise ValueError("At least one camera is required")
        self.captures: dict[str, cv2.VideoCapture] = {}
        self.infos: dict[str, VideoInfo] = {}
        for camera, path in camera_paths.items():
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                self.close()
                raise FileNotFoundError(f"Cannot open {camera}: {path}")
            info = VideoInfo(
                camera=camera,
                path=path,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=float(cap.get(cv2.CAP_PROP_FPS)),
                frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            )
            if abs(info.fps - expected_fps) > 0.01:
                cap.release()
                self.close()
                raise ValueError(f"{camera} fps={info.fps:.8f}, expected {expected_fps:.8f}")
            self.captures[camera] = cap
            self.infos[camera] = info
        self.shortest_frame_count = min(info.frame_count for info in self.infos.values())

    def __enter__(self) -> "SynchronizedVideoReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        for cap in self.captures.values():
            cap.release()

    def frames(self, max_frames: int | None = None) -> Iterator[tuple[int, float, dict[str, object]]]:
        limit = self.shortest_frame_count if max_frames is None else min(max_frames, self.shortest_frame_count)
        for frame_idx in range(limit):
            batch = {}
            ok_all = True
            for camera, cap in self.captures.items():
                ok, frame = cap.read()
                ok_all &= ok
                if ok:
                    batch[camera] = frame
            if not ok_all:
                break
            yield frame_idx, time_sec(frame_idx), batch

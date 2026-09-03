from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .types import INSTRUMENT_POINTS, InstrumentPose


class InstrumentKeypointStore:
    """Frame-indexed DLC interchange format, independent from a DLC runtime."""

    def __init__(self, csv_path: Path | None, score_threshold: float = 0.5) -> None:
        self.score_threshold = score_threshold
        self.rows: dict[tuple[int, str], InstrumentPose] = {}
        if csv_path is not None:
            if not csv_path.exists():
                raise FileNotFoundError(f"Instrument keypoint CSV not found: {csv_path}")
            self._load(csv_path)

    def _load(self, path: Path) -> None:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"frame_idx", "instrument", "keypoint", "x", "y", "score"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"DLC interchange CSV lacks columns {sorted(required)}: {path}")
            temporary: dict[tuple[int, str], dict[str, tuple[float, float, float]]] = {}
            for row in reader:
                key = (int(row["frame_idx"]), row["instrument"])
                temporary.setdefault(key, {})[row["keypoint"]] = (
                    float(row["x"]), float(row["y"]), float(row["score"])
                )
        for key, points in temporary.items():
            xy = np.full((len(INSTRUMENT_POINTS), 2), np.nan)
            score = np.full(len(INSTRUMENT_POINTS), np.nan)
            for idx, name in enumerate(INSTRUMENT_POINTS):
                if name in points:
                    x, y, confidence = points[name]
                    score[idx] = confidence
                    if confidence >= self.score_threshold:
                        xy[idx] = (x, y)
            self.rows[key] = InstrumentPose(xy, score)

    def get(self, frame_idx: int, instrument: str) -> InstrumentPose:
        return self.rows.get((frame_idx, instrument), InstrumentPose())

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_projection_matrices(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    projections = {}
    for camera, params in data["cameras"].items():
        intrinsic = np.asarray(params["camera_matrix"], float)
        rotation = np.asarray(params["rotation"], float)
        translation = np.asarray(params["translation"], float).reshape(3, 1)
        projections[camera] = intrinsic @ np.hstack([rotation, translation])
    return projections


def triangulate_dlt(observations: dict[str, np.ndarray], projections: dict[str, np.ndarray]) -> np.ndarray:
    rows = []
    for camera, point in observations.items():
        projection = projections[camera]
        rows.extend([point[0] * projection[2] - projection[0], point[1] * projection[2] - projection[1]])
    _, _, vh = np.linalg.svd(np.asarray(rows))
    homogeneous = vh[-1]
    return homogeneous[:3] / homogeneous[3] if abs(homogeneous[3]) > 1e-12 else np.full(3, np.nan)


def reprojection_error(point: np.ndarray, observation: np.ndarray, projection: np.ndarray) -> float:
    homogeneous = np.r_[point, 1.0]
    projected = projection @ homogeneous
    projected = projected[:2] / projected[2]
    return float(np.linalg.norm(projected - observation))


def triangulate_table(keypoints: pd.DataFrame, calibration_path: Path, min_score: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    projections = load_projection_matrices(calibration_path)
    points_rows, error_rows = [], []
    group_columns = ["frame_idx", "time_sec", "performer", "keypoint"]
    for keys, data in keypoints.groupby(group_columns, dropna=False):
        observations = {
            row.camera: np.array([row.x, row.y], float) for row in data.itertuples()
            if row.camera in projections and row.score >= min_score and np.isfinite([row.x, row.y]).all()
        }
        if len(observations) < 2:
            continue
        point = triangulate_dlt(observations, projections)
        points_rows.append(dict(zip(group_columns, keys)) | {"x_m": point[0], "y_m": point[1], "z_m": point[2],
                                                               "n_views": len(observations)})
        for camera, observation in observations.items():
            error_rows.append(dict(zip(group_columns, keys)) | {"camera": camera,
                              "reprojection_error_px": reprojection_error(point, observation, projections[camera])})
    return pd.DataFrame(points_rows), pd.DataFrame(error_rows)

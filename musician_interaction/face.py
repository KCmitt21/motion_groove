from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .types import HeadPose, PoseDetection


class FaceHeadPoseEstimator:
    """MediaPipe Face Landmarker plus solvePnP head Euler angles."""

    # Nose, chin, left/right eye outer corner, left/right mouth corner.
    FACE_INDICES = np.array([1, 152, 33, 263, 61, 291])
    MODEL_POINTS = np.array(
        [(0, 0, 0), (0, -63.6, -12.5), (-43.3, 32.7, -26), (43.3, 32.7, -26),
         (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)], dtype=np.float64
    )

    def __init__(self, model_path: Path, min_score: float = 0.5, max_faces: int = 2) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found: {model_path}. See README setup instructions."
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError("MediaPipe is not installed; install requirements-core.txt") from exc
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_score,
            min_face_presence_confidence=min_score,
            min_tracking_confidence=min_score,
            output_facial_transformation_matrixes=False,
        )
        self.mp = mp
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self.landmarker.close()

    @staticmethod
    def _rotation_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
        sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
        if sy > 1e-6:
            x = np.arctan2(rotation[2, 1], rotation[2, 2])
            y = np.arctan2(-rotation[2, 0], sy)
            z = np.arctan2(rotation[1, 0], rotation[0, 0])
        else:
            x = np.arctan2(-rotation[1, 2], rotation[1, 1])
            y = np.arctan2(-rotation[2, 0], sy)
            z = 0.0
        # Conventional reporting: yaw about y, pitch about x, roll about z.
        return tuple(float(np.degrees(value)) for value in (y, x, z))

    def infer(self, frame: np.ndarray) -> list[HeadPose]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(image)
        output = []
        focal = float(max(width, height))
        camera_matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=float)
        for face in result.face_landmarks:
            xy = np.asarray([(point.x * width, point.y * height) for point in face], dtype=float)
            image_points = xy[self.FACE_INDICES]
            ok, rotation_vector, _ = cv2.solvePnP(
                self.MODEL_POINTS, image_points, camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                continue
            rotation, _ = cv2.Rodrigues(rotation_vector)
            yaw, pitch, roll = self._rotation_to_euler(rotation)
            output.append(HeadPose(yaw, pitch, roll, 1.0, np.nanmean(xy, axis=0), xy))
        return output


class NullHeadPoseEstimator:
    def close(self) -> None:
        pass

    def infer(self, frame: np.ndarray) -> list[HeadPose]:
        return []


def assign_faces(
    faces: list[HeadPose], performers: dict[str, PoseDetection | None]
) -> dict[str, HeadPose | None]:
    assigned: dict[str, HeadPose | None] = {name: None for name in performers}
    candidates = list(faces)
    for name, detection in performers.items():
        if detection is None or not candidates:
            continue
        anchor = detection.keypoints[0] if len(detection.keypoints) else detection.center
        distances = [np.linalg.norm(face.center - anchor) for face in candidates]
        idx = int(np.argmin(distances))
        assigned[name] = candidates.pop(idx)
    return assigned


def build_face_estimator(config: dict[str, Any], resolve: Any) -> FaceHeadPoseEstimator | NullHeadPoseEstimator:
    if not config.get("enabled", True):
        return NullHeadPoseEstimator()
    model = resolve(config["model_path"])
    return FaceHeadPoseEstimator(model, float(config.get("min_score", 0.5)), int(config.get("max_faces", 2)))


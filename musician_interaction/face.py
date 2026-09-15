from __future__ import annotations

from pathlib import Path
from typing import Any
import multiprocessing as multiprocessing
import math

import cv2
import numpy as np

from .types import HeadPose, PoseDetection


FaceRoi = tuple[int, int, int, int]
FaceRoiState = dict[str, tuple[FaceRoi, int]]


def _prepare_face_roi(
    frame: np.ndarray, roi: FaceRoi, output_size: int
) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[int, int]]:
    """Crop and enlarge an ROI while retaining its full-frame coordinate mapping."""
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = roi
    x0, x1 = max(0, int(x0)), min(width, int(x1))
    y0, y1 = max(0, int(y0)), min(height, int(y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid face ROI after clipping: {(x0, y0, x1, y1)}")
    crop = frame[y0:y1, x0:x1]
    if output_size > 0 and crop.shape[:2] != (output_size, output_size):
        crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    return crop, (float(x0), float(y0), float(x1), float(y1)), (width, height)


def _square_roi(center: np.ndarray, size: float, frame_shape: tuple[int, ...]) -> FaceRoi | None:
    height, width = frame_shape[:2]
    if not np.isfinite(center).all() or not np.isfinite(size) or size <= 1:
        return None
    half = size / 2
    x0, y0 = max(0, math.floor(center[0] - half)), max(0, math.floor(center[1] - half))
    x1, y1 = min(width, math.ceil(center[0] + half)), min(height, math.ceil(center[1] + half))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def pose_face_roi(
    detection: PoseDetection | None,
    frame_shape: tuple[int, ...],
    min_score: float,
    scale: float = 2.5,
    min_size_px: float = 96.0,
) -> FaceRoi | None:
    """Build a padded face ROI from COCO-WholeBody face or body landmarks."""
    if detection is None:
        return None
    points = np.asarray(detection.keypoints, dtype=float)
    scores = np.asarray(detection.scores, dtype=float)

    # COCO-WholeBody indices 23..90 are the 68 detailed facial landmarks.
    stop = min(91, len(points), len(scores))
    if stop > 23:
        face_points = points[23:stop]
        valid = (scores[23:stop] >= min_score) & np.isfinite(face_points).all(axis=1)
        if np.count_nonzero(valid) >= 3:
            usable = face_points[valid]
            center = (np.nanmin(usable, axis=0) + np.nanmax(usable, axis=0)) / 2
            extent = np.ptp(usable, axis=0)
            return _square_roi(center, max(min_size_px, float(np.max(extent)) * scale), frame_shape)

    # Fall back to COCO body nose/eyes/ears, using shoulder width when necessary.
    head_stop = min(5, len(points), len(scores))
    head = points[:head_stop]
    valid_head = (scores[:head_stop] >= min_score) & np.isfinite(head).all(axis=1)
    usable_head = head[valid_head]
    center = np.nanmedian(usable_head, axis=0) if len(usable_head) else np.full(2, np.nan)
    extent = float(np.max(np.ptp(usable_head, axis=0))) if len(usable_head) >= 2 else 0.0

    if len(points) > 6 and len(scores) > 6:
        shoulders = points[[5, 6]]
        valid_shoulders = (scores[[5, 6]] >= min_score) & np.isfinite(shoulders).all(axis=1)
        if valid_shoulders.all():
            shoulder_width = float(np.linalg.norm(shoulders[1] - shoulders[0]))
            extent = max(extent, shoulder_width * 0.45)
            if not np.isfinite(center).all():
                shoulder_mid = np.mean(shoulders, axis=0)
                center = shoulder_mid + np.array([0.0, -shoulder_width * 0.65])
    if not np.isfinite(center).all() or extent <= 0:
        return None
    return _square_roi(center, max(min_size_px, extent * scale), frame_shape)


def _scaled_roi(roi: FaceRoi, factor: float, frame_shape: tuple[int, ...]) -> FaceRoi | None:
    x0, y0, x1, y1 = roi
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=float)
    return _square_roi(center, max(x1 - x0, y1 - y0) * factor, frame_shape)


def _nearest_face(faces: list[HeadPose], target: np.ndarray) -> HeadPose | None:
    if not faces:
        return None
    distances = [np.linalg.norm(face.center - target) for face in faces]
    return faces[int(np.argmin(distances))]


class FaceHeadPoseEstimator:
    """MediaPipe Face Landmarker plus solvePnP head Euler angles."""

    # Nose, chin, left/right eye outer corner, left/right mouth corner.
    FACE_INDICES = np.array([1, 152, 33, 263, 61, 291])
    MODEL_POINTS = np.array(
        [(0, 0, 0), (0, -63.6, -12.5), (-43.3, 32.7, -26), (43.3, 32.7, -26),
         (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)], dtype=np.float64
    )
    # MODEL_POINTS uses y-up/z-back while the OpenCV camera frame uses y-down/z-forward.
    MODEL_TO_CAMERA_AXES = np.diag([1.0, -1.0, -1.0])

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
            base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU),
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

    def infer_mapped(
        self,
        image_data: np.ndarray,
        coordinate_bounds: tuple[float, float, float, float],
        camera_size: tuple[int, int],
    ) -> list[HeadPose]:
        """Infer on an image and report landmarks in the original frame coordinates."""
        camera_width, camera_height = camera_size
        x0, y0, x1, y1 = coordinate_bounds
        rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(image)
        output = []
        focal = float(max(camera_width, camera_height))
        camera_matrix = np.array(
            [[focal, 0, camera_width / 2], [0, focal, camera_height / 2], [0, 0, 1]], dtype=float
        )
        for face in result.face_landmarks:
            xy = np.asarray(
                [(x0 + point.x * (x1 - x0), y0 + point.y * (y1 - y0)) for point in face], dtype=float
            )
            image_points = xy[self.FACE_INDICES]
            ok, rotation_vector, translation_vector = cv2.solvePnP(
                self.MODEL_POINTS, image_points, camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                continue
            raw_rotation, _ = cv2.Rodrigues(rotation_vector)
            rotation = raw_rotation @ self.MODEL_TO_CAMERA_AXES
            yaw, pitch, roll = self._rotation_to_euler(rotation)
            # The model's +z axis exits through the face.  Equivalently this is
            # -rotation[:, 2] after converting to the reporting coordinate axes.
            gaze_direction = raw_rotation[:, 2].astype(float)
            gaze_direction /= np.linalg.norm(gaze_direction)
            output.append(HeadPose(
                yaw, pitch, roll, 1.0, np.nanmean(xy, axis=0), xy,
                translation_vector.reshape(3).astype(float), gaze_direction,
            ))
        return output

    def infer(self, frame: np.ndarray) -> list[HeadPose]:
        height, width = frame.shape[:2]
        return self.infer_mapped(frame, (0.0, 0.0, float(width), float(height)), (width, height))

    def infer_roi(self, frame: np.ndarray, roi: FaceRoi, output_size: int = 256) -> list[HeadPose]:
        crop, bounds, camera_size = _prepare_face_roi(frame, roi, output_size)
        return self.infer_mapped(crop, bounds, camera_size)


class NullHeadPoseEstimator:
    def close(self) -> None:
        pass

    def infer(self, frame: np.ndarray) -> list[HeadPose]:
        return []

    def infer_roi(self, frame: np.ndarray, roi: FaceRoi, output_size: int = 256) -> list[HeadPose]:
        return []


def _face_worker(connection: Any, model_path: str, min_score: float, max_faces: int) -> None:
    try:
        estimator = FaceHeadPoseEstimator(Path(model_path), min_score, max_faces)
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
        connection.close()
        return
    connection.send(("ready", None))
    try:
        while True:
            payload = connection.recv()
            if payload is None:
                break
            image_data, coordinate_bounds, camera_size = payload
            connection.send(("result", estimator.infer_mapped(image_data, coordinate_bounds, camera_size)))
    finally:
        estimator.close()
        connection.close()


class IsolatedFaceHeadPoseEstimator:
    """Keep native MediaPipe failures from terminating the main analysis process."""

    def __init__(self, model_path: Path, min_score: float, max_faces: int, timeout_sec: float = 30.0) -> None:
        context = multiprocessing.get_context("spawn")
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(
            target=_face_worker, args=(child_connection, str(model_path), min_score, max_faces), daemon=True
        )
        self.timeout_sec = timeout_sec
        self.process.start()
        child_connection.close()
        if not self.connection.poll(timeout_sec):
            self.close()
            raise RuntimeError("MediaPipe Face Landmarker worker did not initialize")
        try:
            message, payload = self.connection.recv()
        except EOFError as exc:
            raise RuntimeError(
                "MediaPipe Face Landmarker worker crashed during initialization; check the MediaPipe/macOS version"
            ) from exc
        if message == "error":
            self.close()
            raise RuntimeError(f"MediaPipe Face Landmarker initialization failed: {payload}")
        if message != "ready":
            self.close()
            raise RuntimeError(f"Unexpected Face Landmarker worker response: {message}: {payload}")

    def _request(
        self,
        image_data: np.ndarray,
        coordinate_bounds: tuple[float, float, float, float],
        camera_size: tuple[int, int],
    ) -> list[HeadPose]:
        if not self.process.is_alive():
            raise RuntimeError("MediaPipe Face Landmarker worker exited unexpectedly")
        self.connection.send((image_data, coordinate_bounds, camera_size))
        if not self.connection.poll(self.timeout_sec):
            raise RuntimeError("MediaPipe Face Landmarker inference timed out")
        try:
            message, payload = self.connection.recv()
        except EOFError as exc:
            raise RuntimeError("MediaPipe Face Landmarker worker crashed during inference") from exc
        if message != "result":
            raise RuntimeError(f"Unexpected Face Landmarker worker response: {message}")
        return payload

    def infer(self, frame: np.ndarray) -> list[HeadPose]:
        height, width = frame.shape[:2]
        return self._request(frame, (0.0, 0.0, float(width), float(height)), (width, height))

    def infer_roi(self, frame: np.ndarray, roi: FaceRoi, output_size: int = 256) -> list[HeadPose]:
        crop, bounds, camera_size = _prepare_face_roi(frame, roi, output_size)
        return self._request(crop, bounds, camera_size)

    def close(self) -> None:
        if getattr(self, "process", None) is None:
            return
        if self.process.is_alive():
            try:
                self.connection.send(None)
                self.process.join(timeout=3)
            except (BrokenPipeError, EOFError):
                pass
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=3)
        self.connection.close()


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


def infer_pose_guided_faces(
    estimator: Any,
    frame: np.ndarray,
    performers: dict[str, PoseDetection | None],
    pose_score_threshold: float,
    roi_config: dict[str, Any],
    state: FaceRoiState,
) -> dict[str, HeadPose | None]:
    """Infer one face per pose-guided ROI and preserve semantic performer identity."""
    assigned: dict[str, HeadPose | None] = {name: None for name in performers}
    if not roi_config.get("enabled", True):
        return assign_faces(estimator.infer(frame), performers)

    scale = float(roi_config.get("scale", 2.5))
    if scale <= 0:
        raise ValueError(f"face.roi.scale must be positive, got {scale}")
    requested_scales = [scale, *(float(item) for item in roi_config.get("retry_scales", [3.5]))]
    scales = list(dict.fromkeys(item for item in requested_scales if item > 0))
    min_size_px = float(roi_config.get("min_size_px", 96.0))
    output_size = int(roi_config.get("output_size", 256))
    reuse_frames = max(0, int(roi_config.get("reuse_frames", 5)))

    for name, detection in performers.items():
        primary = pose_face_roi(detection, frame.shape, pose_score_threshold, scale, min_size_px)
        if primary is not None:
            state[name] = (primary, 0)
        elif name in state and state[name][1] < reuse_frames:
            previous, age = state[name]
            primary = previous
            state[name] = (previous, age + 1)
        else:
            state.pop(name, None)
            continue

        target = np.array([(primary[0] + primary[2]) / 2, (primary[1] + primary[3]) / 2], dtype=float)
        for trial_scale in scales:
            if detection is not None:
                trial = pose_face_roi(
                    detection, frame.shape, pose_score_threshold, trial_scale, min_size_px
                )
            else:
                trial = _scaled_roi(primary, trial_scale / scale, frame.shape)
            if trial is None:
                continue
            face = _nearest_face(estimator.infer_roi(frame, trial, output_size), target)
            if face is not None:
                assigned[name] = face
                break
    return assigned


def build_face_estimator(config: dict[str, Any], resolve: Any) -> Any:
    if not config.get("enabled", True):
        return NullHeadPoseEstimator()
    model = resolve(config["model_path"])
    min_score = float(config.get("min_score", 0.5))
    max_faces = int(config.get("max_faces", 2))
    if config.get("isolated_process", True):
        return IsolatedFaceHeadPoseEstimator(model, min_score, max_faces, float(config.get("timeout_sec", 30)))
    return FaceHeadPoseEstimator(model, min_score, max_faces)

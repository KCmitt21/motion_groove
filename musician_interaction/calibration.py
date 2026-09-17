from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def create_charuco_board(config: dict):
    dictionary_id = getattr(cv2.aruco, config.get("dictionary", "DICT_5X5_100"))
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (int(config["squares_x"]), int(config["squares_y"])),
        float(config["square_length_m"]), float(config["marker_length_m"]), dictionary,
    )
    return board, dictionary


def _detect(path: Path, board, dictionary):
    image = cv2.imread(str(path))
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None:
        return None
    count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if count is None or count < 6:
        return None
    return gray.shape[::-1], charuco_corners, charuco_ids


def calibrate_intrinsic(image_paths: list[Path], board, dictionary) -> dict:
    detections = [item for path in image_paths if (item := _detect(path, board, dictionary)) is not None]
    if len(detections) < 8:
        raise ValueError(f"Need at least 8 usable ChArUco frames, got {len(detections)}")
    image_size = detections[0][0]
    rms, matrix, distortion, rotations, translations = cv2.aruco.calibrateCameraCharuco(
        [item[1] for item in detections], [item[2] for item in detections], board, image_size, None, None
    )
    return {"image_size": image_size, "camera_matrix": matrix, "distortion": distortion,
            "intrinsic_rms_px": float(rms), "usable_frames": len(detections)}


def calibrate_charuco(camera_images: dict[str, list[Path]], config: dict, output: Path) -> dict:
    board, dictionary = create_charuco_board(config)
    intrinsics = {camera: calibrate_intrinsic(paths, board, dictionary)
                  for camera, paths in camera_images.items()}
    reference = config.get("reference_camera", next(iter(camera_images)))
    extrinsics = {reference: {"rotation": np.eye(3), "translation": np.zeros((3, 1))}}
    ref_by_name = {path.name: path for path in camera_images[reference]}
    object_points_all = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    for camera, paths in camera_images.items():
        if camera == reference:
            continue
        cam_by_name = {path.name: path for path in paths}
        object_sets, ref_sets, cam_sets = [], [], []
        for name in sorted(ref_by_name.keys() & cam_by_name.keys()):
            ref_det = _detect(ref_by_name[name], board, dictionary)
            cam_det = _detect(cam_by_name[name], board, dictionary)
            if ref_det is None or cam_det is None:
                continue
            ref_map = {int(idx): point for idx, point in zip(ref_det[2].ravel(), ref_det[1].reshape(-1, 2))}
            cam_map = {int(idx): point for idx, point in zip(cam_det[2].ravel(), cam_det[1].reshape(-1, 2))}
            common = sorted(ref_map.keys() & cam_map.keys())
            if len(common) >= 6:
                object_sets.append(object_points_all[common])
                ref_sets.append(np.asarray([ref_map[idx] for idx in common], np.float32))
                cam_sets.append(np.asarray([cam_map[idx] for idx in common], np.float32))
        if len(object_sets) < 5:
            raise ValueError(f"Need at least 5 paired ChArUco frames for {reference}-{camera}")
        ref_cal, cam_cal = intrinsics[reference], intrinsics[camera]
        rms, _, _, _, _, rotation, translation, _, _ = cv2.stereoCalibrate(
            object_sets, ref_sets, cam_sets, ref_cal["camera_matrix"], ref_cal["distortion"],
            cam_cal["camera_matrix"], cam_cal["distortion"], tuple(ref_cal["image_size"]),
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        extrinsics[camera] = {"rotation": rotation, "translation": translation, "stereo_rms_px": float(rms)}
    serializable = {"reference_camera": reference, "cameras": {}}
    for camera in intrinsics:
        serializable["cameras"][camera] = {
            "image_size": list(intrinsics[camera]["image_size"]),
            "camera_matrix": intrinsics[camera]["camera_matrix"].tolist(),
            "distortion": intrinsics[camera]["distortion"].tolist(),
            "intrinsic_rms_px": intrinsics[camera]["intrinsic_rms_px"],
            "usable_frames": intrinsics[camera]["usable_frames"],
            "rotation": extrinsics[camera]["rotation"].tolist(),
            "translation": extrinsics[camera]["translation"].tolist(),
            "stereo_rms_px": extrinsics[camera].get("stereo_rms_px", 0.0),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return serializable


def calibration_image_sets(root: Path, cameras: list[str], glob_pattern: str) -> dict[str, list[Path]]:
    return {camera: sorted((root / camera).glob(glob_pattern)) for camera in cameras}


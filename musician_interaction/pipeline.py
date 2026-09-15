from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .audio import analyze_audio, extract_audio, probe_audio_start
from .config import AppConfig
from .face import FaceRoiState, build_face_estimator, infer_pose_guided_faces
from .features import (
    BODY_INDEX, audio_motion_correlations, event_triggered_average,
    instrument_motion_events, instrument_pose_motion, performer_cross_correlations, positions_to_motion,
)
from .gaze import classify_gaze_detailed
from .instruments import InstrumentKeypointStore
from .laeo import laeo_frame_row, smooth_laeo_scores
from .outputs import OverlayWriter, draw_overlay, save_graphs, write_manifest, write_table
from .pose import PerformerTracker, build_pose_estimator
from .qc import estimate_sync_offsets, instrument_quality, keypoint_quality, reprojection_quality
from .triangulation import triangulate_table
from .types import INSTRUMENT_POINTS, PERFORMERS, InstrumentPose
from .video import FPS, SynchronizedVideoReader, probe_timecodes


BODY_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def keypoint_name(index: int) -> str:
    return BODY_NAMES[index] if index < len(BODY_NAMES) else f"wholebody_{index:03d}"


def inspect_inputs(config: AppConfig) -> dict[str, Any]:
    video_config = config.data["video"]
    camera_paths = {name: config.resolve(item["path"]) for name, item in video_config["cameras"].items()}
    expected_timecode = video_config.get("start_timecode")
    timecodes = {camera: sorted(probe_timecodes(path)) for camera, path in camera_paths.items()}
    if expected_timecode and any(expected_timecode not in values for values in timecodes.values()):
        raise ValueError(f"Start timecode mismatch; expected {expected_timecode}, got {timecodes}")
    with SynchronizedVideoReader(camera_paths) as reader:
        return {
            "fps_required": str(FPS), "shortest_frame_count": reader.shortest_frame_count,
            "cameras": {name: {"path": str(info.path), "width": info.width, "height": info.height,
                                "fps": info.fps, "frame_count": info.frame_count, "timecodes": timecodes[name]}
                        for name, info in reader.infos.items()},
        }


def _feature_position(detection, indices: list[int], threshold: float) -> tuple[float, float, float]:
    if detection is None:
        return np.nan, np.nan, np.nan
    usable = [idx for idx in indices if idx < len(detection.scores) and detection.scores[idx] >= threshold
              and np.isfinite(detection.keypoints[idx]).all()]
    if not usable:
        return np.nan, np.nan, np.nan
    return (*np.mean(detection.keypoints[usable], axis=0), float(np.mean(detection.scores[usable])))


def _add_keypoint_rows(rows: list[dict], camera: str, frame_idx: int, timestamp: float,
                       performer: str, detection, expected: int = 133) -> None:
    for idx in range(expected):
        valid = detection is not None and idx < len(detection.scores)
        point = detection.keypoints[idx] if valid else (np.nan, np.nan)
        score = detection.scores[idx] if valid else np.nan
        rows.append({"frame_idx": frame_idx, "time_sec": timestamp, "camera": camera,
                     "performer": performer, "keypoint": keypoint_name(idx), "keypoint_idx": idx,
                     "x": float(point[0]), "y": float(point[1]), "score": float(score)})


def _head_angular_velocity(heads: pd.DataFrame, fps: float) -> pd.DataFrame:
    if heads.empty:
        return heads
    heads = heads.sort_values(["camera", "performer", "frame_idx"]).copy()
    group = heads.groupby(["camera", "performer"], sort=False)
    frame_step = group["frame_idx"].diff()
    for angle in ("yaw_deg", "pitch_deg", "roll_deg"):
        heads[f"{angle[:-4]}_velocity_deg_s"] = np.where(frame_step.eq(1), group[angle].diff() * fps, np.nan)
    heads["angular_speed_deg_s"] = np.sqrt(sum(heads[f"{angle}_velocity_deg_s"] ** 2
                                                for angle in ("yaw", "pitch", "roll")))
    return heads


def run_pipeline(config: AppConfig, max_seconds: float | None = None, max_frames: int | None = None) -> dict[str, Any]:
    data = config.data
    input_metadata = inspect_inputs(config)
    output_dir = config.resolve(data["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_configs = data["video"]["cameras"]
    camera_paths = {name: config.resolve(item["path"]) for name, item in camera_configs.items()}
    fps = float(FPS)
    configured_seconds = data["video"].get("max_seconds", 60.0)
    selected_seconds = max_seconds if max_seconds is not None else configured_seconds
    seconds_limit = None if selected_seconds is None else float(selected_seconds)
    frame_limit = max_frames
    if seconds_limit is not None:
        seconds_frames = int(math.ceil(seconds_limit * fps))
        frame_limit = seconds_frames if frame_limit is None else min(frame_limit, seconds_frames)

    audio_config = data.get("audio", {})
    sample_rate = int(audio_config.get("sample_rate", 48000))
    audio_by_camera = {}
    external_wav = audio_config.get("wav_path")
    if external_wav:
        primary_path = config.resolve(external_wav)
        wav_offset = float(audio_config.get("wav_offset_sec", 0.0))
        wav_duration = None if seconds_limit is None else max(0.0, seconds_limit - wav_offset)
        primary_audio = analyze_audio(primary_path, sample_rate, int(audio_config.get("hop_length", 512)),
                                      wav_offset, wav_duration)
    else:
        for camera, source in camera_paths.items():
            wav = output_dir / "audio" / f"{camera}.wav"
            start_offset = probe_audio_start(source)
            extract_audio(source, wav, sample_rate)
            audio_duration = None if seconds_limit is None else max(0.0, seconds_limit - start_offset)
            audio_by_camera[camera] = analyze_audio(
                wav, sample_rate, int(audio_config.get("hop_length", 512)), start_offset, audio_duration
            )
        primary_camera = audio_config.get("source_camera", next(iter(camera_paths)))
        primary_audio = audio_by_camera[primary_camera]

    pose_config = data["pose"]
    backend = pose_config.get("backend", "mmpose")
    if backend == "mmpose":
        shared_pose = build_pose_estimator(pose_config, 2)
        estimators = {camera: shared_pose for camera in camera_paths}
    else:
        estimators = {camera: build_pose_estimator(pose_config, len(item["visible_performers"]))
                      for camera, item in camera_configs.items()}
    trackers = {
        camera: PerformerTracker(item["visible_performers"], item["initial_left_to_right"],
                                 float(pose_config.get("keypoint_score_threshold", 0.3)))
        for camera, item in camera_configs.items()
    }
    face_config = data.get("face", {"enabled": False})
    face_estimators = {camera: build_face_estimator(face_config, config.resolve)
                       for camera in camera_paths}
    face_roi_states: dict[str, FaceRoiState] = {camera: {} for camera in camera_paths}
    face_roi_config = face_config.get("roi", {"enabled": False})
    laeo_config = data.get("laeo", {})
    laeo_enabled = bool(laeo_config.get("enabled", True))
    laeo_performers = tuple(laeo_config.get("performers", ("bassist", "guitarist")))
    if len(laeo_performers) != 2 or laeo_performers[0] == laeo_performers[1]:
        raise ValueError("laeo.performers must contain two distinct performer names")
    laeo_sigma_deg = float(laeo_config.get("sigma_deg", 25.0))
    if not np.isfinite(laeo_sigma_deg) or laeo_sigma_deg <= 0:
        raise ValueError(f"laeo.sigma_deg must be positive and finite, got {laeo_sigma_deg}")
    configured_laeo_cameras = laeo_config.get("cameras")
    laeo_cameras = None if configured_laeo_cameras is None else set(configured_laeo_cameras)
    instrument_config = data.get("instruments", {})
    dlc_csv = instrument_config.get("keypoint_csv", {})
    instrument_stores = {
        camera: InstrumentKeypointStore(config.resolve(dlc_csv[camera]) if camera in dlc_csv else None,
                                        float(instrument_config.get("score_threshold", 0.5)))
        for camera in camera_paths
    }
    keypoint_rows: list[dict] = []
    head_rows: list[dict] = []
    laeo_rows: list[dict] = []
    instrument_rows: list[dict] = []
    position_rows: list[dict] = []
    frame_rows: list[dict] = []
    score_threshold = float(pose_config.get("keypoint_score_threshold", 0.3))
    overlay_enabled = bool(data["output"].get("overlay_video", True))
    writers: dict[str, OverlayWriter] = {}
    processed = 0
    termination = "frame_limit"
    try:
        with SynchronizedVideoReader(camera_paths) as reader:
            if overlay_enabled:
                writers = {camera: OverlayWriter(output_dir / "overlays" / f"{camera}.mp4", info.width,
                                                  info.height, fps, float(data["output"].get("overlay_scale", 0.5)))
                           for camera, info in reader.infos.items()}
            for frame_idx, timestamp, frames in reader.frames(frame_limit):
                processed += 1
                frame_rows.append({"frame_idx": frame_idx, "time_sec": timestamp, "all_cameras_read": True})
                for camera, frame in frames.items():
                    detections = estimators[camera].infer(frame)
                    assigned = trackers[camera].assign(detections)
                    heads = infer_pose_guided_faces(
                        face_estimators[camera], frame, assigned, score_threshold,
                        face_roi_config, face_roi_states[camera],
                    )
                    instruments = {
                        "guitar": instrument_stores[camera].get(frame_idx, "guitar"),
                        "bass": instrument_stores[camera].get(frame_idx, "bass"),
                    }
                    gaze_labels = {}
                    for performer in camera_configs[camera]["visible_performers"]:
                        detection = assigned.get(performer)
                        _add_keypoint_rows(keypoint_rows, camera, frame_idx, timestamp, performer, detection)
                        partner_name = "bassist" if performer == "guitarist" else "guitarist"
                        own_name = "guitar" if performer == "guitarist" else "bass"
                        label, gaze_score, unknown_reason = classify_gaze_detailed(
                            heads.get(performer), detection, assigned.get(partner_name), instruments[own_name],
                            data.get("gaze", {}), camera, performer,
                        )
                        gaze_labels[performer] = label
                        head = heads.get(performer)
                        position_3d = head.position_3d if head is not None else np.full(3, np.nan)
                        gaze_direction_3d = head.gaze_direction_3d if head is not None else np.full(3, np.nan)
                        head_rows.append({"frame_idx": frame_idx, "time_sec": timestamp, "camera": camera,
                                          "performer": performer, "yaw_deg": head.yaw if head else np.nan,
                                          "pitch_deg": head.pitch if head else np.nan, "roll_deg": head.roll if head else np.nan,
                                          "head_position_x": position_3d[0], "head_position_y": position_3d[1],
                                          "head_position_z": position_3d[2],
                                          "gaze_direction_x": gaze_direction_3d[0],
                                          "gaze_direction_y": gaze_direction_3d[1],
                                          "gaze_direction_z": gaze_direction_3d[2],
                                          "face_score": head.score if head else np.nan, "gaze_target": label,
                                          "gaze_score": gaze_score, "unknown_reason": unknown_reason})
                        for feature, indices in BODY_INDEX.items():
                            if feature == "head" and head is not None:
                                x, y, score = *head.center, head.score
                            else:
                                x, y, score = _feature_position(detection, indices, score_threshold)
                            position_rows.append({"frame_idx": frame_idx, "time_sec": timestamp, "camera": camera,
                                                  "performer": performer, "feature": feature, "x": x, "y": y,
                                                  "score": score, "frame_width": frame.shape[1],
                                                  "frame_height": frame.shape[0]})
                        instrument_pose = instruments[own_name]
                        position_rows.append({"frame_idx": frame_idx, "time_sec": timestamp, "camera": camera,
                                              "performer": performer, "feature": "instrument", "x": instrument_pose.points[0, 0],
                                              "y": instrument_pose.points[0, 1], "score": instrument_pose.scores[0],
                                              "frame_width": frame.shape[1], "frame_height": frame.shape[0]})
                    visible = set(camera_configs[camera]["visible_performers"])
                    if (
                        laeo_enabled
                        and (laeo_cameras is None or camera in laeo_cameras)
                        and set(laeo_performers).issubset(visible)
                    ):
                        performer_a, performer_b = laeo_performers
                        laeo_rows.append(laeo_frame_row(
                            frame_idx, timestamp, camera, performer_a, performer_b,
                            heads.get(performer_a), heads.get(performer_b), laeo_sigma_deg,
                        ))
                    for instrument, pose in instruments.items():
                        for idx, name in enumerate(INSTRUMENT_POINTS):
                            instrument_rows.append({"frame_idx": frame_idx, "time_sec": timestamp, "camera": camera,
                                                    "instrument": instrument, "keypoint": name,
                                                    "x": pose.points[idx, 0], "y": pose.points[idx, 1],
                                                    "score": pose.scores[idx], "frame_width": frame.shape[1],
                                                    "frame_height": frame.shape[0]})
                    if overlay_enabled:
                        writers[camera].write(draw_overlay(frame, assigned, instruments, heads, gaze_labels, score_threshold))
            if processed >= reader.shortest_frame_count:
                termination = "shortest_video_end"
    finally:
        for writer in writers.values():
            writer.close()
        for estimator in face_estimators.values():
            estimator.close()

    keypoints = pd.DataFrame(keypoint_rows)
    heads = _head_angular_velocity(pd.DataFrame(head_rows), fps)
    laeo_columns = [
        "frame_idx", "timestamp", "time_sec", "camera", "performer_A", "performer_B",
        "theta_A", "theta_B", "p_A", "p_B", "laeo_score",
    ]
    laeo = smooth_laeo_scores(
        pd.DataFrame(laeo_rows, columns=laeo_columns), laeo_config.get("smoothing", {})
    )
    instruments = pd.DataFrame(instrument_rows)
    motion = positions_to_motion(pd.DataFrame(position_rows), fps)
    frames = pd.DataFrame(frame_rows)
    video_times = frames["time_sec"].to_numpy()
    rms, flux = primary_audio.at_video_times(video_times)
    audio_framewise = pd.DataFrame({"frame_idx": frames["frame_idx"], "time_sec": video_times,
                                    "rms": rms, "spectral_flux": flux})
    onsets = pd.DataFrame({"onset_idx": np.arange(len(primary_audio.onset_times)),
                           "time_sec": primary_audio.onset_times,
                           "frame_idx": np.rint(primary_audio.onset_times * fps).astype(int)})
    analysis_config = data.get("analysis", {})
    instrument_analysis = analysis_config.get("instrument_motion", {})
    instrument_motion = instrument_pose_motion(
        instruments, fps,
        float(instrument_analysis.get("oscillation_window_sec", 1.0)),
        float(instrument_analysis.get("oscillation_min_hz", 3.0)),
        float(instrument_analysis.get("oscillation_max_hz", 9.0)),
        float(instrument_analysis.get("oscillation_min_power_ratio", .5)),
        float(instrument_analysis.get("oscillation_min_rms_deg", .5)),
        float(instrument_analysis.get("rise_velocity_threshold_norm_s", .02)),
        float(instrument_analysis.get("rise_smoothing_sec", .2)),
    )
    instrument_events = instrument_motion_events(
        instrument_motion, fps,
        float(instrument_analysis.get("rise_min_duration_sec", .2)),
        float(instrument_analysis.get("oscillation_min_duration_sec", .3)),
        float(instrument_analysis.get("ending_window_sec", 5.0)),
    )
    performer_cross = performer_cross_correlations(motion, fps, float(analysis_config.get("max_lag_sec", 2.0)))
    audio_motion = audio_motion_correlations(motion, audio_framewise, fps,
                                              float(analysis_config.get("max_lag_sec", 2.0)))
    event_average = event_triggered_average(motion, primary_audio.onset_times, fps,
                                             float(analysis_config.get("event_window_sec", 1.0)))
    keypoint_qc = keypoint_quality(keypoints, score_threshold)
    instrument_qc = instrument_quality(instruments, float(instrument_config.get("score_threshold", 0.5)))
    face_qc = heads.groupby(["camera", "performer"], as_index=False).agg(
        samples=("frame_idx", "size"), missing_fraction=("yaw_deg", lambda values: float(values.isna().mean())),
        mean_face_score=("face_score", "mean")) if not heads.empty else pd.DataFrame()
    sync_qc = estimate_sync_offsets(audio_by_camera, audio_config.get("source_camera", next(iter(camera_paths))),
                                    float(data.get("qc", {}).get("max_sync_lag_sec", 1.0))) if audio_by_camera else pd.DataFrame()

    parquet = bool(data["output"].get("parquet", True))
    tables = {"frames": frames, "pose_keypoints_2d": keypoints, "head_pose_gaze": heads,
              "laeo_by_time": laeo,
              "instrument_keypoints_2d": instruments, "motion_features": motion,
              "instrument_motion_features": instrument_motion, "instrument_motion_events": instrument_events,
              "audio_features": audio_framewise, "audio_onsets": onsets,
              "performer_cross_correlation": performer_cross, "audio_motion_lag": audio_motion,
              "onset_triggered_average": event_average, "qc_keypoints": keypoint_qc,
              "qc_instruments": instrument_qc, "qc_faces": face_qc, "qc_sync": sync_qc}
    for name, table in tables.items():
        write_table(table, output_dir / "tables" / name, parquet)

    calibration_config = data.get("calibration", {})
    calibration_path = config.resolve(calibration_config.get("output_file", "calibration/calibration.json"))
    triangulation_status = "skipped: calibration file not found"
    if calibration_config.get("triangulate_if_available", True) and calibration_path.exists():
        points_3d, reprojection = triangulate_table(keypoints, calibration_path, score_threshold)
        reprojection = reprojection_quality(reprojection, float(data.get("qc", {}).get("reprojection_warn_px", 5.0)))
        write_table(points_3d, output_dir / "tables" / "keypoints_3d", parquet)
        write_table(reprojection, output_dir / "tables" / "qc_reprojection", parquet)
        triangulation_status = f"completed: {len(points_3d)} points"
    save_graphs(motion, audio_framewise, performer_cross, event_average, output_dir / "graphs", instrument_motion)
    manifest = {
        "status": "completed", "fps": "30000/1001", "time_formula": "frame_idx * 1001 / 30000",
        "start_timecode": data["video"].get("start_timecode"),
        "input_shortest_frame_count": input_metadata["shortest_frame_count"],
        "frames_processed": processed, "termination": termination,
        "last_frame_idx": processed - 1 if processed else None,
        "last_time_sec": (processed - 1) * 1001 / 30000 if processed else None,
        "triangulation": triangulation_status, "pose_backend": backend,
        "laeo": {
            "enabled": laeo_enabled,
            "performers": list(laeo_performers),
            "sigma_deg": laeo_sigma_deg,
            "smoothing": laeo_config.get("smoothing", {}),
            "rows": len(laeo),
        },
        "scientific_warning": "mock pose output is synthetic and must not be used for research" if backend == "mock" else None,
    }
    write_manifest(output_dir / "manifest.json", manifest)
    return manifest

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import cv2

from .calibration import calibrate_charuco, calibration_image_sets
from .config import load_config
from .face import build_face_estimator
from .head_direction import write_head_direction_report
from .pipeline import inspect_inputs, run_pipeline
from .triangulation import triangulate_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronized musician interaction analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        item = sub.add_parser(name)
        item.add_argument("--config", default="configs/mvp.yaml")
        if name == "run":
            item.add_argument("--max-seconds", type=float)
            item.add_argument("--max-frames", type=int)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--config", default="configs/mvp.yaml")
    face_check = sub.add_parser("face-check")
    face_check.add_argument("--config", default="configs/mvp.yaml")
    face_check.add_argument("--image", required=True)
    triangulate = sub.add_parser("triangulate")
    triangulate.add_argument("--config", default="configs/mvp.yaml")
    triangulate.add_argument("--keypoints", required=True)
    triangulate.add_argument("--output", required=True)
    head_summary = sub.add_parser("head-summary")
    head_summary.add_argument("--config", default="configs/mvp.yaml")
    head_summary.add_argument("--input", default="out/mvp/tables/head_pose_gaze.csv")
    head_summary.add_argument("--output-dir", default="output/tables")
    head_summary.add_argument("--camera", default="cam_wide")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "validate":
        print(json.dumps(inspect_inputs(config), ensure_ascii=False, indent=2))
    elif args.command == "run":
        print(json.dumps(run_pipeline(config, args.max_seconds, args.max_frames), ensure_ascii=False, indent=2))
    elif args.command == "calibrate":
        section = config.data["calibration"]
        cameras = list(config.data["video"]["cameras"])
        images = calibration_image_sets(config.resolve(section["images_directory"]), cameras,
                                        section.get("image_glob", "*.png"))
        result = calibrate_charuco(images, section["charuco"], config.resolve(section["output_file"]))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "face-check":
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(args.image)
        estimator = build_face_estimator(config.data["face"], config.resolve)
        try:
            faces = estimator.infer(frame)
            print(json.dumps([{
                "yaw_deg": item.yaw,
                "pitch_deg": item.pitch,
                "roll_deg": item.roll,
                "head_position_3d": item.position_3d.tolist(),
                "gaze_direction_3d": item.gaze_direction_3d.tolist(),
            } for item in faces], indent=2))
        finally:
            estimator.close()
    elif args.command == "triangulate":
        section = config.data["calibration"]
        keypoints = pd.read_csv(args.keypoints)
        points, errors = triangulate_table(keypoints, config.resolve(section["output_file"]),
                                           float(config.data["pose"].get("keypoint_score_threshold", .3)))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        points.to_csv(output, index=False)
        errors.to_csv(output.with_name(output.stem + "_reprojection.csv"), index=False)
    elif args.command == "head-summary":
        heads = pd.read_csv(config.resolve(args.input))
        result = write_head_direction_report(
            heads, config.resolve(args.output_dir), args.camera,
            gaze_config=config.data.get("gaze", {}),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

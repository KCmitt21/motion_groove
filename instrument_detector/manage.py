from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import shutil
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

from musician_interaction.types import INSTRUMENT_POINTS


CLASS_NAMES = ("guitar", "bass")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SKELETON = ((0, 1), (0, 2), (2, 3), (3, 4))


@dataclass
class InstrumentPoseDetection:
    points: np.ndarray
    point_scores: np.ndarray
    box_score: float
    xyxy: tuple[float, float, float, float]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def select_best_poses(
    xyxy: Any,
    box_scores: Any,
    class_ids: Any,
    keypoints_xy: Any,
    keypoint_scores: Any,
    class_names: dict[int, str] | list[str] | tuple[str, ...],
    expected: Iterable[str],
) -> dict[str, InstrumentPoseDetection]:
    """Select the highest-confidence valid pose instance for each expected instrument."""
    allowed = set(expected)
    names = dict(enumerate(class_names)) if not isinstance(class_names, dict) else class_names
    boxes = _as_numpy(xyxy).reshape(-1, 4)
    confidences = _as_numpy(box_scores).reshape(-1)
    classes = _as_numpy(class_ids).reshape(-1)
    points = _as_numpy(keypoints_xy)
    point_confidences = _as_numpy(keypoint_scores)
    if points.ndim != 3 or points.shape[1] != len(INSTRUMENT_POINTS) or points.shape[2] != 2:
        raise ValueError(f"Expected keypoints shape (N, {len(INSTRUMENT_POINTS)}, 2), got {points.shape}")
    if point_confidences.shape != points.shape[:2]:
        raise ValueError(f"Expected keypoint scores shape {points.shape[:2]}, got {point_confidences.shape}")
    selected: dict[str, InstrumentPoseDetection] = {}
    for box, score, class_id, pose_points, pose_scores in zip(
        boxes, confidences, classes, points, point_confidences
    ):
        name = str(names.get(int(class_id), ""))
        x1, y1, x2, y2 = (float(value) for value in box)
        confidence = float(score)
        if name not in allowed or not np.isfinite([x1, y1, x2, y2, confidence]).all():
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        current = selected.get(name)
        if current is None or confidence > current.box_score:
            selected[name] = InstrumentPoseDetection(
                points=np.asarray(pose_points, dtype=float).copy(),
                point_scores=np.asarray(pose_scores, dtype=float).copy(),
                box_score=confidence,
                xyxy=(x1, y1, x2, y2),
            )
    return selected


class PoseSmoother:
    """Optional per-keypoint EMA smoothing that resets after a detection gap."""

    def __init__(self, alpha: float = 1.0, reset_after_frames: int = 1) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("EMA alpha must be in (0, 1]")
        if reset_after_frames < 0:
            raise ValueError("reset_after_frames must be non-negative")
        self.alpha = alpha
        self.reset_after_frames = reset_after_frames
        self.previous: dict[str, tuple[int, np.ndarray]] = {}

    def apply(
        self, frame_idx: int, instrument: str, detection: InstrumentPoseDetection
    ) -> InstrumentPoseDetection:
        points = detection.points.copy()
        previous = self.previous.get(instrument)
        if previous is not None and frame_idx - previous[0] <= self.reset_after_frames + 1:
            valid = np.isfinite(points).all(axis=1) & np.isfinite(previous[1]).all(axis=1)
            points[valid] = self.alpha * points[valid] + (1.0 - self.alpha) * previous[1][valid]
        self.previous[instrument] = (frame_idx, points)
        return InstrumentPoseDetection(points, detection.point_scores.copy(), detection.box_score, detection.xyxy)


def init_dataset(root: Path, force: bool = False) -> Path:
    root = root.resolve()
    for relative in ("raw/images", "raw/labels", "images/train", "images/val", "labels/train", "labels/val"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    data_file = root / "dataset.yaml"
    if data_file.exists() and not force:
        raise FileExistsError(f"Dataset config already exists; use --force to replace it: {data_file}")
    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "guitar", 1: "bass"},
        "kpt_shape": [len(INSTRUMENT_POINTS), 3],
        "flip_idx": list(range(len(INSTRUMENT_POINTS))),
        "kpt_names": {0: list(INSTRUMENT_POINTS), 1: list(INSTRUMENT_POINTS)},
    }
    data_file.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_file


def _parse_video_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Video must be CAMERA=PATH")
    camera, raw_path = value.split("=", 1)
    if not camera or not raw_path:
        raise argparse.ArgumentTypeError("Video must be CAMERA=PATH")
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Video not found: {path}")
    return camera, path


def extract_frames(
    videos: Iterable[tuple[str, Path]], output: Path, interval_sec: float, max_images_per_video: int | None
) -> int:
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")
    output.mkdir(parents=True, exist_ok=True)
    total = 0
    for camera, path in videos:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            capture.release()
            raise RuntimeError(f"Invalid video FPS: {path}: {fps}")
        stride = max(1, int(round(interval_sec * fps)))
        frame_idx = 0
        saved = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                destination = output / f"{camera}_frame_{frame_idx:06d}.jpg"
                if not cv2.imwrite(str(destination), frame):
                    capture.release()
                    raise RuntimeError(f"Cannot write image: {destination}")
                saved += 1
                total += 1
                if max_images_per_video is not None and saved >= max_images_per_video:
                    break
            frame_idx += 1
        capture.release()
        print(f"{camera}: extracted {saved} images from {path}")
    return total


def split_dataset(root: Path, validation_fraction: float, seed: int) -> tuple[int, int]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    root = root.resolve()
    source_images = root / "raw" / "images"
    source_labels = root / "raw" / "labels"
    images = sorted(path for path in source_images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) < 2:
        raise ValueError(f"At least two annotated images are required in {source_images}")
    missing_labels = [image.name for image in images if not (source_labels / f"{image.stem}.txt").exists()]
    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        raise ValueError(f"Every image needs a YOLO label file, including empty negative labels; missing: {preview}")
    destinations = [root / kind / split for kind in ("images", "labels") for split in ("train", "val")]
    if any(any(directory.iterdir()) for directory in destinations):
        raise FileExistsError("Train/validation directories are not empty; use a fresh dataset or clear them explicitly")
    shuffled = images.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation = {path.name for path in shuffled[:validation_count]}
    counts = {"train": 0, "val": 0}
    for image in images:
        split = "val" if image.name in validation else "train"
        shutil.copy2(image, root / "images" / split / image.name)
        label = source_labels / f"{image.stem}.txt"
        shutil.copy2(label, root / "labels" / split / label.name)
        counts[split] += 1
    return counts["train"], counts["val"]


def _load_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Create the detector environment and install requirements-instruments.txt"
        ) from exc
    return YOLO


def train_model(args: argparse.Namespace) -> None:
    data = Path(args.data).expanduser().resolve()
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    YOLO = _load_yolo()
    model = YOLO(args.model)
    result = model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        workers=args.workers, project=str(Path(args.project).expanduser().resolve()), name=args.name,
    )
    print(f"Training output: {result.save_dir}")
    print(f"Best weights: {Path(result.save_dir) / 'weights' / 'best.pt'}")


def validate_model(args: argparse.Namespace) -> None:
    YOLO = _load_yolo()
    model = YOLO(str(Path(args.model).expanduser().resolve()))
    metrics = model.val(data=str(Path(args.data).expanduser().resolve()), device=args.device, imgsz=args.imgsz)
    print(f"mAP50-95: {float(metrics.box.map):.6f}")
    print(f"mAP50: {float(metrics.box.map50):.6f}")
    if getattr(metrics, "pose", None) is not None:
        print(f"pose mAP50-95: {float(metrics.pose.map):.6f}")
        print(f"pose mAP50: {float(metrics.pose.map50):.6f}")


def infer_video(
    model: Any,
    video: Path,
    output: Path,
    expected: tuple[str, ...],
    confidence: float,
    keypoint_confidence: float,
    iou: float,
    image_size: int,
    device: str,
    ema_alpha: float,
    preview: Path | None = None,
) -> dict[str, int]:
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    model_names = model.names
    available = set(model_names.values() if isinstance(model_names, dict) else model_names)
    missing_classes = set(expected) - available
    if missing_classes:
        raise ValueError(f"Model lacks expected classes {sorted(missing_classes)}; available={sorted(available)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    counts = {name: 0 for name in expected}
    smoother = PoseSmoother(ema_alpha)
    writer = None
    if preview is not None:
        capture = cv2.VideoCapture(str(video))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        preview.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(preview), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create preview video: {preview}")
    try:
        results = model.predict(
            source=str(video), stream=True, conf=confidence, iou=iou, imgsz=image_size,
            device=device, vid_stride=1, verbose=False,
        )
        for frame_idx, result in enumerate(results):
            boxes = result.boxes
            keypoints = result.keypoints
            selected = {} if boxes is None or keypoints is None else select_best_poses(
                boxes.xyxy, boxes.conf, boxes.cls, keypoints.xy, keypoints.conf,
                model_names, expected,
            )
            smoothed_selected = {}
            for instrument, raw_detection in selected.items():
                low_confidence = raw_detection.point_scores < keypoint_confidence
                raw_detection.points[low_confidence] = np.nan
                detection = smoother.apply(frame_idx, instrument, raw_detection)
                smoothed_selected[instrument] = detection
                for point_idx, point_name in enumerate(INSTRUMENT_POINTS):
                    rows.append({"frame_idx": frame_idx, "instrument": instrument, "keypoint": point_name,
                                 "x": detection.points[point_idx, 0], "y": detection.points[point_idx, 1],
                                 "score": detection.point_scores[point_idx]})
                counts[instrument] += 1
            if writer is not None:
                canvas = result.orig_img.copy()
                for instrument, detection in smoothed_selected.items():
                    x1, y1, x2, y2 = np.rint(detection.xyxy).astype(int)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    for first, second in SKELETON:
                        if np.isfinite(detection.points[[first, second]]).all():
                            start, end = np.rint(detection.points[[first, second]]).astype(int)
                            cv2.line(canvas, tuple(start), tuple(end), (0, 255, 255), 2)
                    for point in detection.points:
                        if np.isfinite(point).all():
                            cv2.circle(canvas, tuple(np.rint(point).astype(int)), 5, (0, 255, 255), -1)
                    cv2.putText(canvas, f"{instrument} {detection.box_score:.2f}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 0, 255), 2, cv2.LINE_AA)
                writer.write(canvas)
    finally:
        if writer is not None:
            writer.release()
    never_detected = [name for name, count in counts.items() if count == 0]
    if never_detected:
        raise RuntimeError(f"No detections for {never_detected} in {video}; CSV was not written")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=("frame_idx", "instrument", "keypoint", "x", "y", "score"))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    print(f"Wrote {len(rows)} keypoints to {output}: detected frames={counts}")
    return counts


def run_inference(args: argparse.Namespace) -> None:
    YOLO = _load_yolo()
    model = YOLO(str(Path(args.model).expanduser().resolve()))
    infer_video(
        model=model,
        video=Path(args.video).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(),
        expected=tuple(args.instruments),
        confidence=args.confidence,
        keypoint_confidence=args.keypoint_confidence,
        iou=args.iou,
        image_size=args.imgsz,
        device=args.device,
        ema_alpha=args.ema_alpha,
        preview=Path(args.preview).expanduser().resolve() if args.preview else None,
    )


def run_config_inference(args: argparse.Namespace) -> None:
    from musician_interaction.config import load_config

    config = load_config(args.config)
    YOLO = _load_yolo()
    model = YOLO(str(Path(args.model).expanduser().resolve()))
    output_dir = config.resolve(args.output_dir)
    preview_dir = config.resolve(args.preview_dir) if args.preview_dir else None
    performer_to_instrument = {"guitarist": "guitar", "bassist": "bass"}
    for camera, camera_config in config.data["video"]["cameras"].items():
        expected = tuple(dict.fromkeys(
            performer_to_instrument[name] for name in camera_config["visible_performers"]
            if name in performer_to_instrument
        ))
        infer_video(
            model=model,
            video=config.resolve(camera_config["path"]),
            output=output_dir / f"{camera}.csv",
            expected=expected,
            confidence=args.confidence,
            keypoint_confidence=args.keypoint_confidence,
            iou=args.iou,
            image_size=args.imgsz,
            device=args.device,
            ema_alpha=args.ema_alpha,
            preview=(preview_dir / f"{camera}.mp4") if preview_dir else None,
        )
    print("Add these paths to instruments.keypoint_csv:")
    for camera in config.data["video"]["cameras"]:
        print(f"  {camera}: {args.output_dir}/{camera}.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run a five-keypoint guitar/bass pose model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-dataset", help="Create the YOLO dataset directory structure")
    init.add_argument("--output", default="datasets/instruments_pose")
    init.add_argument("--force", action="store_true")

    extract = subparsers.add_parser("extract", help="Extract frames to annotate")
    extract.add_argument("--video", action="append", required=True, type=_parse_video_spec,
                         metavar="CAMERA=PATH")
    extract.add_argument("--output", default="datasets/instruments_pose/raw/images")
    extract.add_argument("--interval-sec", type=float, default=1.0)
    extract.add_argument("--max-images-per-video", type=int)

    split = subparsers.add_parser("split", help="Split annotated raw images and labels into train/val")
    split.add_argument("--dataset", default="datasets/instruments_pose")
    split.add_argument("--validation-fraction", type=float, default=.2)
    split.add_argument("--seed", type=int, default=29)

    train = subparsers.add_parser("train", help="Fine-tune an Ultralytics detector")
    train.add_argument("--data", default="datasets/instruments_pose/dataset.yaml")
    train.add_argument("--model", default="yolo26n-pose.pt")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--device", default="mps")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--project", default="models/instrument_detector")
    train.add_argument("--name", default="yolo26n_pose")

    validate = subparsers.add_parser("validate", help="Evaluate trained weights")
    validate.add_argument("--model", required=True)
    validate.add_argument("--data", default="datasets/instruments_pose/dataset.yaml")
    validate.add_argument("--imgsz", type=int, default=640)
    validate.add_argument("--device", default="mps")

    infer = subparsers.add_parser("infer", help="Export one video to the five-keypoint CSV schema")
    infer.add_argument("--model", required=True)
    infer.add_argument("--video", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--instruments", nargs="+", choices=CLASS_NAMES, required=True)
    infer.add_argument("--confidence", type=float, default=.5)
    infer.add_argument("--keypoint-confidence", type=float, default=.5)
    infer.add_argument("--iou", type=float, default=.5)
    infer.add_argument("--imgsz", type=int, default=640)
    infer.add_argument("--device", default="mps")
    infer.add_argument("--ema-alpha", type=float, default=1.0)
    infer.add_argument("--preview")

    infer_config = subparsers.add_parser("infer-config", help="Export every camera from an MVP config")
    infer_config.add_argument("--config", default="configs/mvp.yaml")
    infer_config.add_argument("--model", required=True)
    infer_config.add_argument("--output-dir", default="out/dlc")
    infer_config.add_argument("--preview-dir", default="out/dlc/previews")
    infer_config.add_argument("--confidence", type=float, default=.5)
    infer_config.add_argument("--keypoint-confidence", type=float, default=.5)
    infer_config.add_argument("--iou", type=float, default=.5)
    infer_config.add_argument("--imgsz", type=int, default=640)
    infer_config.add_argument("--device", default="mps")
    infer_config.add_argument("--ema-alpha", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "init-dataset":
        print(f"Created dataset config: {init_dataset(Path(args.output), args.force)}")
    elif args.command == "extract":
        count = extract_frames(args.video, Path(args.output), args.interval_sec, args.max_images_per_video)
        print(f"Extracted {count} images")
    elif args.command == "split":
        train_count, validation_count = split_dataset(Path(args.dataset), args.validation_fraction, args.seed)
        print(f"Split dataset: train={train_count}, val={validation_count}")
    elif args.command == "train":
        train_model(args)
    elif args.command == "validate":
        validate_model(args)
    elif args.command == "infer":
        run_inference(args)
    elif args.command == "infer-config":
        run_config_inference(args)


if __name__ == "__main__":
    main()

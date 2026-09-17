"""Minimal DeepLabCut project lifecycle; run in the separate DLC environment."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--videos", nargs="+", required=True)
    create.add_argument("--workdir", default="dlc/projects")
    create.add_argument("--experimenter", default="researcher")
    for command in ("extract", "label", "train", "evaluate", "infer"):
        item = sub.add_parser(command)
        item.add_argument("--config", required=True)
        if command == "infer":
            item.add_argument("--videos", nargs="+", required=True)
    args = parser.parse_args()
    import deeplabcut
    if args.command == "create":
        path = deeplabcut.create_new_project(
            "musician_instruments", args.experimenter, [str(Path(v).resolve()) for v in args.videos],
            working_directory=str(Path(args.workdir).resolve()), copy_videos=False, multianimal=True,
        )
        print(f"Created: {path}\nApply bodyparts/individuals from dlc/project_template.yaml before labeling.")
    elif args.command == "extract":
        deeplabcut.extract_frames(args.config, mode="automatic", algo="kmeans", userfeedback=False)
    elif args.command == "label":
        deeplabcut.label_frames(args.config)
    elif args.command == "train":
        deeplabcut.create_training_dataset(args.config)
        deeplabcut.train_network(args.config)
    elif args.command == "evaluate":
        deeplabcut.evaluate_network(args.config, plotting=True)
    elif args.command == "infer":
        deeplabcut.analyze_videos(args.config, args.videos, videotype=".mov", save_as_csv=True)


if __name__ == "__main__":
    main()


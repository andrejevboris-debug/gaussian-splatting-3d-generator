#!/usr/bin/env python3
"""
Project-level CLI entry point for the Gaussian Splatting pipeline.

Usage examples:
  python run.py video-to-frames --video data/raw_capture.mp4 --output data/input_images
  python run.py colmap --images data/input_images
  python run.py train --config training/configs/default.yaml --max-steps 1000
  python run.py all --video data/raw_capture.mp4 --config training/configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from scripts.extract_frames import ExtractionSettings, extract_frames
from scripts.run_colmap import ColmapSettings, run_colmap
from scripts.train_3dgs import main as train_main

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FRAMES_DIR = PROJECT_ROOT / "data" / "input_images"
DEFAULT_CONFIG = PROJECT_ROOT / "training" / "configs" / "default.yaml"
DEFAULT_DATABASE = PROJECT_ROOT / "colmap" / "database" / "colmap.db"
DEFAULT_SPARSE = PROJECT_ROOT / "colmap" / "sparse"
DEFAULT_DENSE = PROJECT_ROOT / "colmap" / "dense"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gaussian Splatting pipeline orchestrator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # video-to-frames command
    video_parser = subparsers.add_parser("video-to-frames", help="Convert video into image sequence.")
    video_parser.add_argument("--video", required=True, type=Path, help="Input video file.")
    video_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FRAMES_DIR,
        help=f"Output directory for frames (default: {DEFAULT_FRAMES_DIR})",
    )
    video_parser.add_argument("--fps", type=float, default=None, help="Target frames per second.")
    video_parser.add_argument(
        "--frame-step",
        type=int,
        default=None,
        help="Take every Nth frame instead of specifying fps.",
    )
    video_parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        help="Image file extension (default: png).",
    )
    video_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frames.")
    video_parser.add_argument("--start-time", type=float, default=None, help="Start time in seconds.")
    video_parser.add_argument("--end-time", type=float, default=None, help="End time in seconds.")
    video_parser.add_argument("--dry-run", action="store_true", help="Print command without running.")

    # colmap command
    colmap_parser = subparsers.add_parser("colmap", help="Run COLMAP SfM + MVS pipeline.")
    colmap_parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_FRAMES_DIR,
        help=f"Directory with input images (default: {DEFAULT_FRAMES_DIR}).",
    )
    colmap_parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"COLMAP database path (default: {DEFAULT_DATABASE}).",
    )
    colmap_parser.add_argument(
        "--sparse",
        type=Path,
        default=DEFAULT_SPARSE,
        help=f"Sparse output directory (default: {DEFAULT_SPARSE}).",
    )
    colmap_parser.add_argument(
        "--dense",
        type=Path,
        default=DEFAULT_DENSE,
        help=f"Dense output directory (default: {DEFAULT_DENSE}).",
    )
    colmap_parser.add_argument(
        "--matcher",
        choices=("exhaustive", "sequential"),
        default="exhaustive",
        help="Feature matcher type.",
    )
    colmap_parser.add_argument(
        "--colmap-cmd",
        type=str,
        default="colmap",
        help="Path to the COLMAP executable.",
    )
    colmap_parser.add_argument("--overwrite", action="store_true", help="Overwrite database if exists.")
    colmap_parser.add_argument(
        "--skip-mvs",
        action="store_true",
        help="Skip dense stereo and fusion stages.",
    )
    colmap_parser.add_argument("--dry-run", action="store_true", help="Print commands only.")

    # train command
    train_parser = subparsers.add_parser("train", help="Train Gaussian Splatting model.")
    train_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Training configuration file (default: {DEFAULT_CONFIG}).",
    )
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override number of training steps (for quick iterations).",
    )
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without running training loop.",
    )
    train_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity for the training script.",
    )
    train_parser.add_argument(
        "--backend",
        type=str,
        choices=("auto", "nerfstudio", "gsplat"),
        default="auto",
        help="Training backend to use.",
    )
    train_parser.add_argument(
        "--backend-cmd",
        type=str,
        default=None,
        help="Executable path for the training backend.",
    )
    train_parser.add_argument(
        "--backend-arg",
        action="append",
        dest="backend_args",
        default=[],
        help="Additional backend argument (repeatable).",
    )

    # all command
    all_parser = subparsers.add_parser("all", help="Run the complete pipeline end-to-end.")
    all_parser.add_argument("--video", required=True, type=Path, help="Input video file.")
    all_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Training configuration file (default: {DEFAULT_CONFIG}).",
    )
    all_parser.add_argument(
        "--frames-dir",
        type=Path,
        default=DEFAULT_FRAMES_DIR,
        help=f"Where to store intermediate frames (default: {DEFAULT_FRAMES_DIR}).",
    )
    all_parser.add_argument("--fps", type=float, default=None, help="Frames-per-second for extraction.")
    all_parser.add_argument(
        "--frame-step",
        type=int,
        default=None,
        help="Take every Nth frame instead of specifying fps.",
    )
    all_parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        help="Image file extension for extracted frames.",
    )
    all_parser.add_argument("--start-time", type=float, default=None, help="Start time in seconds.")
    all_parser.add_argument("--end-time", type=float, default=None, help="End time in seconds.")
    all_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite intermediate artifacts (frames + database).",
    )
    all_parser.add_argument(
        "--skip-mvs",
        action="store_true",
        help="Skip COLMAP dense stereo and fusion.",
    )
    all_parser.add_argument(
        "--matcher",
        choices=("exhaustive", "sequential"),
        default="exhaustive",
        help="Feature matcher type for COLMAP.",
    )
    all_parser.add_argument(
        "--colmap-cmd",
        type=str,
        default="colmap",
        help="Path to the COLMAP executable.",
    )
    all_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training steps.",
    )
    all_parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    all_parser.add_argument(
        "--backend",
        type=str,
        choices=("auto", "nerfstudio", "gsplat"),
        default="auto",
        help="Training backend to use.",
    )
    all_parser.add_argument(
        "--backend-cmd",
        type=str,
        default=None,
        help="Executable path for the training backend.",
    )
    all_parser.add_argument(
        "--backend-arg",
        action="append",
        dest="backend_args",
        default=[],
        help="Additional backend argument (repeatable).",
    )
    all_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity for the training script.",
    )

    return parser


def run_video_to_frames(args: argparse.Namespace) -> None:
    settings = ExtractionSettings(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        frame_step=args.frame_step,
        image_format=args.image_format,
        overwrite=args.overwrite,
        start_time=args.start_time,
        end_time=args.end_time,
        dry_run=args.dry_run,
    )
    frame_paths = list(extract_frames(settings))
    if args.dry_run:
        print(f"[pipeline] Dry run complete. Existing frames detected: {len(frame_paths)}")
    else:
        print(f"[pipeline] Extracted {len(frame_paths)} frames to {settings.output_dir}")


def run_colmap_command(args: argparse.Namespace) -> None:
    settings = ColmapSettings(
        images_dir=args.images,
        database_path=args.database,
        sparse_dir=args.sparse,
        dense_dir=args.dense,
        colmap_cmd=args.colmap_cmd,
        matcher=args.matcher,
        overwrite=args.overwrite,
        run_mvs=not args.skip_mvs,
        dry_run=args.dry_run,
    )
    run_colmap(settings)
    if args.dry_run:
        print("[pipeline] COLMAP dry run finished.")
    else:
        print(f"[pipeline] COLMAP outputs available in {args.sparse} and {args.dense}")


def build_train_args(args: argparse.Namespace) -> List[str]:
    train_args: List[str] = ["--config", str(args.config)]
    if getattr(args, "max_steps", None) is not None:
        train_args += ["--max-steps", str(args.max_steps)]
    if getattr(args, "dry_run", False):
        train_args.append("--dry-run")
    log_level = getattr(args, "log_level", None)
    if log_level:
        train_args += ["--log-level", log_level]
    backend = getattr(args, "backend", None)
    if backend:
        train_args += ["--backend", backend]
    backend_cmd = getattr(args, "backend_cmd", None)
    if backend_cmd:
        train_args += ["--backend-cmd", backend_cmd]
    for extra in getattr(args, "backend_args", []) or []:
        train_args += ["--backend-arg", extra]
    return train_args


def run_train_command(args: argparse.Namespace) -> None:
    train_main(build_train_args(args))
    print("[pipeline] Training routine completed.")


def run_all_command(args: argparse.Namespace) -> None:
    # 1. Extract frames
    extract_settings = ExtractionSettings(
        video_path=args.video,
        output_dir=args.frames_dir,
        fps=args.fps,
        frame_step=args.frame_step,
        image_format=args.image_format,
        overwrite=args.overwrite,
        start_time=args.start_time,
        end_time=args.end_time,
        dry_run=args.dry_run,
    )
    frame_paths = list(extract_frames(extract_settings))
    if args.dry_run:
        print(f"[pipeline] Dry run: would extract frames to {args.frames_dir} (found {len(frame_paths)}).")
    else:
        print(f"[pipeline] Extracted {len(frame_paths)} frames to {args.frames_dir}")

    # 2. Run COLMAP
    colmap_settings = ColmapSettings(
        images_dir=args.frames_dir,
        database_path=DEFAULT_DATABASE,
        sparse_dir=DEFAULT_SPARSE,
        dense_dir=DEFAULT_DENSE,
        colmap_cmd=args.colmap_cmd,
        matcher=args.matcher,
        overwrite=args.overwrite,
        run_mvs=not args.skip_mvs,
        dry_run=args.dry_run,
    )
    run_colmap(colmap_settings)
    if args.dry_run:
        print("[pipeline] COLMAP dry run finished.")
    else:
        print(f"[pipeline] COLMAP outputs ready at {DEFAULT_SPARSE} (sparse) and {DEFAULT_DENSE} (dense).")

    # 3. Train 3DGS
    train_main(build_train_args(args))
    print("[pipeline] Training routine completed.")


def dispatch(argv: Optional[List[str]] = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "video-to-frames":
        run_video_to_frames(args)
    elif args.command == "colmap":
        run_colmap_command(args)
    elif args.command == "train":
        run_train_command(args)
    elif args.command == "all":
        run_all_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    dispatch()

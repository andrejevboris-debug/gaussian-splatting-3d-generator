#!/usr/bin/env python3
"""
Extract frames from a video file using ffmpeg.

This script provides both a callable function (`extract_frames`) and a CLI entry
point. It wraps ffmpeg in a small amount of ergonomics so callers can easily
convert a captured video into a folder of images ready for COLMAP.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


class FrameExtractionError(RuntimeError):
    """Raised when the ffmpeg extraction pipeline fails."""


@dataclass
class ExtractionSettings:
    video_path: Path
    output_dir: Path
    fps: Optional[float] = None
    frame_step: Optional[int] = None
    image_format: str = "png"
    overwrite: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    dry_run: bool = False

    def validate(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Input video not found: {self.video_path}")
        if self.fps is not None and self.frame_step is not None:
            raise ValueError("Specify either --fps or --frame-step, not both.")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("FPS must be greater than zero.")
        if self.frame_step is not None and self.frame_step <= 0:
            raise ValueError("Frame step must be greater than zero.")
        if not self.image_format:
            raise ValueError("Image format cannot be empty.")


def build_ffmpeg_command(settings: ExtractionSettings, output_pattern: Path) -> List[str]:
    command: List[str] = [
        "ffmpeg",
        "-y" if settings.overwrite else "-n",
        "-loglevel",
        "error",
    ]

    if settings.start_time is not None:
        command.extend(["-ss", str(settings.start_time)])
    command.extend(["-i", str(settings.video_path)])
    if settings.end_time is not None:
        command.extend(["-to", str(settings.end_time)])

    filters: List[str] = []
    if settings.fps is not None:
        filters.append(f"fps={settings.fps}")
    elif settings.frame_step is not None:
        filters.append(f"select=not(mod(n\\,{settings.frame_step}))")
        filters.append("setpts=N/TB")

    if filters:
        command.extend(["-vf", ",".join(filters)])

    command.extend(["-vsync", "vfr", str(output_pattern)])
    return command


def extract_frames(settings: ExtractionSettings) -> Iterable[Path]:
    """
    Extract frames and return a generator over the produced image paths.
    """
    settings.validate()

    if shutil.which("ffmpeg") is None:
        raise FrameExtractionError("ffmpeg executable not found in PATH.")

    settings.output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = settings.output_dir / f"frame_%06d.{settings.image_format}"

    command = build_ffmpeg_command(settings, output_pattern)
    if settings.dry_run:
        print("[extract-frames] Dry run:", " ".join(command))
        return sorted(settings.output_dir.glob(f"*.{settings.image_format}"))

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise FrameExtractionError("ffmpeg failed to extract frames.") from exc

    return sorted(settings.output_dir.glob(f"*.{settings.image_format}"))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a video using ffmpeg.")
    parser.add_argument("--video", required=True, type=Path, help="Path to the input video file.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where the extracted frames should be written.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output frame rate. Mutually exclusive with --frame-step.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=None,
        help="Keep every Nth frame. Mutually exclusive with --fps.",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        help="Image format/extension to use for saved frames (default: png).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frames if the output directory is not empty.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Optional start time in seconds.",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Optional end time in seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ffmpeg command without executing it.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
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
        print(
            f"[extract-frames] Dry run complete. Existing frames counted: {len(frame_paths)}"
        )
    else:
        print(f"[extract-frames] Extracted {len(frame_paths)} frames into {settings.output_dir}")


if __name__ == "__main__":
    main()

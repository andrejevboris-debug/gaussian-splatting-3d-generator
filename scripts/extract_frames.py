#!/usr/bin/env python3
"""
Extract frames from a video file using ffmpeg-python.

This module exposes a high-level `extract_frames` function alongside a CLI entry
point so it can be reused programmatically (e.g. from `run.py`) or invoked
directly. It supports sampling by FPS or frame step, partial extraction, and dry
run previews. The output naming scheme matches COLMAP expectations.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import ffmpeg  # type: ignore
except ImportError as exc:  # pragma: no cover - guidance for first-time users
    raise SystemExit(
        "The `ffmpeg-python` package is required. Install it with `pip install ffmpeg-python`."
    ) from exc


log = logging.getLogger(__name__)


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
        if self.end_time is not None and self.start_time is not None:
            if self.end_time <= self.start_time:
                raise ValueError("--end-time must be greater than --start-time.")


def _build_ffmpeg_stream(settings: ExtractionSettings, output_pattern: Path) -> ffmpeg.nodes.OutputStream:
    input_kwargs = {}
    if settings.start_time is not None:
        input_kwargs["ss"] = settings.start_time

    stream = ffmpeg.input(str(settings.video_path), **input_kwargs)

    if settings.fps is not None:
        stream = stream.filter("fps", fps=settings.fps)
    elif settings.frame_step is not None:
        stream = stream.filter("select", f"not(mod(n\\,{settings.frame_step}))")
        stream = stream.filter("setpts", "N/TB")

    output_kwargs = {"vsync": "vfr"}
    if settings.end_time is not None:
        if settings.start_time is not None:
            duration = settings.end_time - settings.start_time
            output_kwargs["t"] = duration
        else:
            output_kwargs["to"] = settings.end_time

    return ffmpeg.output(stream, str(output_pattern), **output_kwargs)


def extract_frames(settings: ExtractionSettings) -> Iterable[Path]:
    """
    Extract frames and return an iterable of produced image paths.
    """
    settings.validate()
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    extension = settings.image_format.lstrip(".")
    output_pattern = settings.output_dir / f"frame_%06d.{extension}"

    stream = _build_ffmpeg_stream(settings, output_pattern)
    command: List[str] = ffmpeg.compile(stream, overwrite_output=settings.overwrite)

    if settings.dry_run:
        print("[extract-frames] Dry run:", " ".join(command))
        return sorted(settings.output_dir.glob(f"*.{extension}"))

    try:
        ffmpeg.run(stream, overwrite_output=settings.overwrite, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if getattr(exc, "stderr", None) else ""
        raise FrameExtractionError(f"ffmpeg failed to extract frames: {stderr}") from exc

    produced = sorted(settings.output_dir.glob(f"*.{extension}"))
    log.debug("Extracted %d frames into %s", len(produced), settings.output_dir)
    return produced


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a video using ffmpeg-python.")
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
        print(f"[extract-frames] Dry run complete. Existing frames counted: {len(frame_paths)}")
    else:
        print(f"[extract-frames] Extracted {len(frame_paths)} frames into {settings.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
High-level COLMAP pipeline runner.

This module provides a Python interface for executing the typical COLMAP SfM +
MVS workflow:
  1. Feature extraction
  2. Feature matching (exhaustive or sequential)
  3. Sparse mapping
  4. Sparse model export (TXT)
  5. Undistortion + dense stereo + fusion (optional)

The functions are structured so that the pipeline can be consumed from a CLI or
programmatically (via `run.py`). Each stage prints the COLMAP command being
executed, and a `--dry-run` flag can be used to inspect the pipeline without
running it.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


log = logging.getLogger(__name__)


class ColmapError(RuntimeError):
    """Raised when the COLMAP pipeline fails."""


@dataclass
class ColmapSettings:
    images_dir: Path
    database_path: Path
    sparse_dir: Path
    dense_dir: Path
    colmap_cmd: str = "colmap"
    matcher: str = "exhaustive"
    overwrite: bool = False
    run_mvs: bool = True
    dry_run: bool = False

    def validate(self) -> None:
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if shutil.which(self.colmap_cmd) is None:
            raise ColmapError(f"COLMAP executable '{self.colmap_cmd}' not found in PATH.")


def _log_command(command: List[str]) -> None:
    log.info("[colmap] %s", " ".join(command))
    print("[colmap]", " ".join(command))


def _run(command: List[str], dry_run: bool) -> None:
    _log_command(command)
    if dry_run:
        return
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise ColmapError(f"COLMAP command failed: {' '.join(command)}") from exc


def _clean_path(path: Path, overwrite: bool, is_dir: bool = True) -> None:
    if overwrite and path.exists():
        if is_dir:
            shutil.rmtree(path)
        else:
            path.unlink()
    if is_dir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)


def _sparse_models(sparse_dir: Path) -> Iterable[Path]:
    if not sparse_dir.exists():
        return []
    return sorted(p for p in sparse_dir.iterdir() if p.is_dir())


def run_colmap(settings: ColmapSettings) -> None:
    """
    Execute the COLMAP SfM + optional MVS pipeline.
    """
    settings.validate()

    _clean_path(settings.database_path, settings.overwrite, is_dir=False)
    _clean_path(settings.sparse_dir, settings.overwrite, is_dir=True)
    if settings.run_mvs:
        _clean_path(settings.dense_dir, settings.overwrite, is_dir=True)
    else:
        settings.dense_dir.mkdir(parents=True, exist_ok=True)

    # Feature extraction is implicitly creating the database if needed.
    feature_cmd = [
        settings.colmap_cmd,
        "feature_extractor",
        "--database_path",
        str(settings.database_path),
        "--image_path",
        str(settings.images_dir),
    ]
    _run(feature_cmd, settings.dry_run)

    if settings.matcher == "exhaustive":
        matcher_cmd = [
            settings.colmap_cmd,
            "exhaustive_matcher",
            "--database_path",
            str(settings.database_path),
        ]
    elif settings.matcher == "sequential":
        matcher_cmd = [
            settings.colmap_cmd,
            "sequential_matcher",
            "--database_path",
            str(settings.database_path),
            "--SiftMatching.guided_matching",
            "1",
        ]
    else:
        raise ValueError(f"Unsupported matcher: {settings.matcher}")
    _run(matcher_cmd, settings.dry_run)

    mapper_cmd = [
        settings.colmap_cmd,
        "mapper",
        "--database_path",
        str(settings.database_path),
        "--image_path",
        str(settings.images_dir),
        "--output_path",
        str(settings.sparse_dir),
    ]
    _run(mapper_cmd, settings.dry_run)

    model_dirs = list(_sparse_models(settings.sparse_dir))
    if not model_dirs and settings.dry_run:
        print("[colmap] Dry run: mapper output assumed at sparse/0.")
        model_dirs = [settings.sparse_dir / "0"]
    elif not model_dirs:
        raise ColmapError("No sparse models produced; cannot convert or run dense reconstruction.")

    for model_dir in model_dirs:
        text_dir = model_dir.with_name(f"{model_dir.name}_text")
        text_dir.mkdir(parents=True, exist_ok=True)
        converter_cmd = [
            settings.colmap_cmd,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(text_dir),
            "--output_type",
            "TXT",
        ]
        _run(converter_cmd, settings.dry_run)

    if not settings.run_mvs:
        print("[colmap] Skipping dense reconstruction.")
        return

    reference_model = model_dirs[0]
    undistort_cmd = [
        settings.colmap_cmd,
        "image_undistorter",
        "--image_path",
        str(settings.images_dir),
        "--input_path",
        str(reference_model),
        "--output_path",
        str(settings.dense_dir),
        "--output_type",
        "COLMAP",
    ]
    _run(undistort_cmd, settings.dry_run)

    stereo_cmd = [
        settings.colmap_cmd,
        "patch_match_stereo",
        "--workspace_path",
        str(settings.dense_dir),
        "--workspace_format",
        "COLMAP",
        "--PatchMatchStereo.geom_consistency",
        "true",
    ]
    _run(stereo_cmd, settings.dry_run)

    fusion_cmd = [
        settings.colmap_cmd,
        "stereo_fusion",
        "--workspace_path",
        str(settings.dense_dir),
        "--workspace_format",
        "COLMAP",
        "--input_type",
        "geometric",
        "--output_path",
        str(settings.dense_dir / "fused.ply"),
    ]
    _run(fusion_cmd, settings.dry_run)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the COLMAP SfM + MVS pipeline.")
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/input_images"),
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("colmap/database/colmap.db"),
        help="Path to the COLMAP database.",
    )
    parser.add_argument(
        "--sparse",
        type=Path,
        default=Path("colmap/sparse"),
        help="Output directory for sparse reconstructions.",
    )
    parser.add_argument(
        "--dense",
        type=Path,
        default=Path("colmap/dense"),
        help="Output directory for dense reconstructions.",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        choices=("exhaustive", "sequential"),
        default="exhaustive",
        help="Feature matcher to use.",
    )
    parser.add_argument(
        "--colmap-cmd",
        type=str,
        default="colmap",
        help="Name or path of the COLMAP executable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing COLMAP artifacts before running.",
    )
    parser.add_argument(
        "--skip-mvs",
        action="store_true",
        help="Skip the dense stereo and fusion stages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

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


if __name__ == "__main__":
    main()

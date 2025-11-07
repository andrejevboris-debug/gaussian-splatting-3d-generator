#!/usr/bin/env python3
"""
Python wrapper for executing a COLMAP SfM + MVS pipeline.

The script mirrors the steps typically performed in the bash helper:
  1. Create a COLMAP database and extract image features.
  2. Match features (default: exhaustive matcher).
  3. Run sparse reconstruction (mapper).
  4. Convert sparse models to TXT for downstream consumption.
  5. Undistort images and run dense stereo + fusion (optional).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


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


def run_command(command: List[str], dry_run: bool) -> None:
    print("[colmap]", " ".join(command))
    if dry_run:
        return
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise ColmapError(f"COLMAP command failed: {' '.join(command)}") from exc


def get_sparse_model_dirs(sparse_dir: Path) -> Iterable[Path]:
    return sorted(p for p in sparse_dir.iterdir() if p.is_dir())


def run_colmap(settings: ColmapSettings) -> None:
    settings.validate()

    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    settings.sparse_dir.mkdir(parents=True, exist_ok=True)
    settings.dense_dir.mkdir(parents=True, exist_ok=True)

    if settings.overwrite and settings.database_path.exists():
        settings.database_path.unlink()

    database_cmd = [
        settings.colmap_cmd,
        "database_creator",
        "--database_path",
        str(settings.database_path),
    ]
    run_command(database_cmd, settings.dry_run)

    feature_cmd = [
        settings.colmap_cmd,
        "feature_extractor",
        "--database_path",
        str(settings.database_path),
        "--image_path",
        str(settings.images_dir),
    ]
    run_command(feature_cmd, settings.dry_run)

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
    run_command(matcher_cmd, settings.dry_run)

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
    run_command(mapper_cmd, settings.dry_run)

    model_dirs = list(get_sparse_model_dirs(settings.sparse_dir))
    if not model_dirs and settings.dry_run:
        print(
            "[colmap] Dry run: no sparse models available to convert. Skipping text export preview."
        )
    else:
        for model_dir in model_dirs:
            text_dir = model_dir.with_name(model_dir.name + "_text")
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
            run_command(converter_cmd, settings.dry_run)

    if not settings.run_mvs:
        print("[colmap] Skipping dense reconstruction.")
        return

    model_dirs = list(get_sparse_model_dirs(settings.sparse_dir))
    if not model_dirs:
        if settings.dry_run:
            print(
                "[colmap] Dry run: assuming mapper output at sparse/0 for dense reconstruction preview."
            )
            reference_model = settings.sparse_dir / "0"
        else:
            raise ColmapError("No sparse models produced. Cannot continue to dense stage.")
    else:
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
        "--max_image_size",
        "2000",
    ]
    run_command(undistort_cmd, settings.dry_run)

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
    run_command(stereo_cmd, settings.dry_run)

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
    run_command(fusion_cmd, settings.dry_run)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COLMAP SfM + MVS pipeline.")
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
        help="Overwrite existing COLMAP database before starting.",
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

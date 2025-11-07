#!/usr/bin/env python3
"""
Launch Gaussian Splatting training via gsplat or Nerfstudio.

This script translates project configuration files into concrete training
commands for either backend. It validates COLMAP outputs, prepares logging and
checkpoint directories, and finally shells out to the requested training CLI.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover - first run guidance
    raise SystemExit("PyYAML is required. Install with `pip install pyyaml`.") from exc


log = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    """Raised when the training pipeline encounters an unrecoverable error."""


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        if config_path.suffix == ".json":
            return json.load(handle)
        raise ValueError(f"Unsupported config extension: {config_path.suffix}")


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


@dataclass
class TrainingPaths:
    config_path: Path
    scene_image_dir: Path
    sparse_dir: Path
    dense_dir: Path
    log_dir: Path
    checkpoint_dir: Path
    model_dir: Path


def prepare_training_paths(config: Dict[str, Any], config_path: Path) -> TrainingPaths:
    base = config_path.parent
    try:
        scene_cfg = config["scene"]
        logging_cfg = config.get("logging", {})
        output_cfg = config.get("output", {})
    except KeyError as exc:
        raise TrainingError(f"Missing configuration key: {exc}") from exc

    scene_image_dir = resolve_path(base, scene_cfg["image_dir"])
    sparse_dir = resolve_path(base, scene_cfg["colmap_sparse_dir"])
    dense_dir = resolve_path(base, scene_cfg["colmap_dense_dir"])
    log_dir = resolve_path(base, logging_cfg.get("output_dir", "training/logs"))
    checkpoint_dir = resolve_path(base, output_cfg.get("checkpoint_dir", "training/checkpoints"))
    model_dir = resolve_path(base, output_cfg.get("model_dir", "outputs/3dgs_models"))

    return TrainingPaths(
        config_path=config_path.resolve(),
        scene_image_dir=scene_image_dir,
        sparse_dir=sparse_dir,
        dense_dir=dense_dir,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        model_dir=model_dir,
    )


@dataclass
class TrainingSettings:
    config_path: Path
    max_steps: Optional[int]
    dry_run: bool
    log_level: str
    backend: str
    backend_cmd: Optional[str]
    extra_args: List[str]


def ensure_inputs(paths: TrainingPaths) -> None:
    if not paths.scene_image_dir.exists():
        raise TrainingError(f"Input image directory not found: {paths.scene_image_dir}")
    if not paths.sparse_dir.exists():
        raise TrainingError(f"Sparse COLMAP outputs not found: {paths.sparse_dir}")
    if not paths.dense_dir.exists():
        log.warning("Dense COLMAP outputs not found at %s. Continuing without dense supervision.", paths.dense_dir)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)


def discover_backend(settings: TrainingSettings) -> str:
    if settings.backend != "auto":
        return settings.backend

    if settings.backend_cmd:
        cmd = Path(settings.backend_cmd).name.lower()
        if "ns-train" in cmd or "nerfstudio" in cmd:
            return "nerfstudio"
        if "gsplat" in cmd:
            return "gsplat"

    if shutil.which("gsplat"):
        return "gsplat"
    if shutil.which("ns-train"):
        return "nerfstudio"

    raise TrainingError(
        "Could not automatically determine training backend. Install gsplat or Nerfstudio, "
        "or specify --backend together with --backend-cmd."
    )


def _coerce_steps(config: Dict[str, Any], override: Optional[int]) -> Optional[int]:
    try:
        default_steps = int(config.get("training", {}).get("steps", 0))
    except (TypeError, ValueError):
        default_steps = 0
    if override is not None:
        return override
    return default_steps or None


def build_nerfstudio_command(
    settings: TrainingSettings, paths: TrainingPaths, config: Dict[str, Any]
) -> List[str]:
    executable = settings.backend_cmd or "ns-train"
    if shutil.which(executable) is None:
        raise TrainingError(f"Nerfstudio executable not found: {executable}")

    scene_name = config.get("scene", {}).get("name", paths.scene_image_dir.name)
    steps = _coerce_steps(config, settings.max_steps)

    command: List[str] = [
        executable,
        "splatfacto",
        "--data",
        str(paths.scene_image_dir),
        "--load-colmap",
        str(paths.sparse_dir),
        "--output-dir",
        str(paths.model_dir),
        "--experiment-name",
        scene_name,
        "--viewer.quit-on-train-completion",
        "True",
    ]

    if paths.dense_dir.exists():
        command += ["--pipeline.datamanager.dataparser.include_mono_prior", "False"]

    if steps is not None:
        command += ["--max-num-iterations", str(steps)]

    log_interval = config.get("logging", {}).get("log_interval")
    if isinstance(log_interval, int) and log_interval > 0:
        command += ["--logging.interval", str(log_interval)]

    command.extend(settings.extra_args)
    return command


def build_gsplat_command(settings: TrainingSettings, paths: TrainingPaths, config: Dict[str, Any]) -> List[str]:
    executable = settings.backend_cmd or "gsplat"
    if shutil.which(executable) is None:
        raise TrainingError(f"gsplat executable not found: {executable}")

    steps = _coerce_steps(config, settings.max_steps)
    scene_name = config.get("scene", {}).get("name", paths.scene_image_dir.name)

    command: List[str] = [
        executable,
        "train",
        "--config",
        str(settings.config_path),
        "--data",
        str(paths.scene_image_dir),
        "--colmap",
        str(paths.sparse_dir),
        "--dense",
        str(paths.dense_dir),
        "--output",
        str(paths.model_dir),
        "--scene-name",
        scene_name,
        "--checkpoint-dir",
        str(paths.checkpoint_dir),
    ]

    if steps is not None:
        command += ["--max-steps", str(steps)]

    command.extend(settings.extra_args)
    return command


def run_training(settings: TrainingSettings) -> None:
    config = load_config(settings.config_path)
    paths = prepare_training_paths(config, settings.config_path)
    ensure_inputs(paths)

    backend = discover_backend(settings)
    log.info("Using training backend: %s", backend)

    if backend == "nerfstudio":
        command = build_nerfstudio_command(settings, paths, config)
    elif backend == "gsplat":
        command = build_gsplat_command(settings, paths, config)
    else:
        raise TrainingError(f"Unsupported backend: {backend}")

    log.info("[train] %s", " ".join(command))
    print("[train]", " ".join(command))

    if settings.dry_run:
        log.info("Dry run enabled. Training command not executed.")
        return

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise TrainingError(f"Training command failed: {' '.join(command)}") from exc


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Gaussian Splatting model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/configs/default.yaml"),
        help="Path to the training configuration file (YAML or JSON).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the number of training steps.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=("auto", "nerfstudio", "gsplat"),
        help="Training backend to use. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--backend-cmd",
        type=str,
        default=None,
        help="Executable to invoke for the training backend (e.g. path to ns-train or gsplat).",
    )
    parser.add_argument(
        "--backend-arg",
        action="append",
        dest="extra_args",
        default=[],
        help="Additional argument to forward to the backend. Can be specified multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show the command without executing it.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    settings = TrainingSettings(
        config_path=config_path,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
        log_level=args.log_level,
        backend=args.backend,
        backend_cmd=args.backend_cmd,
        extra_args=list(args.extra_args),
    )

    run_training(settings)


if __name__ == "__main__":
    main()

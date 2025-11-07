#!/usr/bin/env python3
"""
Minimal entry point for training a Gaussian Splatting model.

This script loads a YAML configuration file, initializes placeholder components,
and outlines the steps required to integrate an actual 3DGS implementation.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover - guidance for first use
    raise SystemExit(
        "PyYAML is required to parse configuration files. Install with `pip install pyyaml`."
    ) from exc


def load_config(config_path: pathlib.Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        if config_path.suffix == ".json":
            return json.load(handle)
        raise ValueError(f"Unsupported config extension: {config_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Gaussian Splatting model.")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("training/configs/default.yaml"),
        help="Path to the training configuration file (YAML or JSON).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print("[3DGS] Loaded configuration:")
    print(json.dumps(config, indent=2))

    print("[3DGS] TODO: Initialize dataset using COLMAP outputs located at:")
    print(f"        Sparse: {config['scene']['colmap_sparse_dir']}")
    print(f"        Dense:  {config['scene']['colmap_dense_dir']}")
    print("[3DGS] TODO: Build Gaussian Splatting model with hyperparameters from `model` section.")
    print("[3DGS] TODO: Implement training loop respecting optimizer settings in `optimization`.")
    print("[3DGS] TODO: Save checkpoints to", config["output"]["checkpoint_dir"])
    print("[3DGS] TODO: Export final model to", config["output"]["model_dir"])


if __name__ == "__main__":
    main()

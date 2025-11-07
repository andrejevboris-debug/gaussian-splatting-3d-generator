# Gaussian Splatting Pipeline

This repository contains a minimal scaffold for running a Gaussian Splatting (3DGS) reconstruction pipeline. It separates data ingestion, COLMAP reconstruction, neural training, and export of final 3D Gaussian models.

## Directory Layout

- `data/input_images` – Raw posed or unposed RGB images captured from the scene.
- `colmap/database` – COLMAP database (`.db`) and feature cache produced during SfM.
- `colmap/sparse` – Sparse reconstruction results (`cameras.txt`, `images.txt`, `points3D.txt`).
- `colmap/dense` – Optional dense reconstructions (depth maps, meshes).
- `training/configs` – YAML/JSON configs describing training hyperparameters.
- `training/checkpoints` – Intermediate checkpoints produced while training the 3DGS model.
- `training/logs` – Training logs and metrics.
- `outputs/3dgs_models` – Exported 3D Gaussian Splatting models ready for rendering or evaluation.
- `scripts` – Convenience scripts for COLMAP preprocessing and 3DGS training.

All empty directories contain `.gitkeep` files so that the structure persists in version control.

## Quick Start

1. **Install dependencies**
   - Install COLMAP (>= 3.9) and CUDA-enabled PyTorch.
   - Install Python requirements with `pip install -r requirements.txt` (update as needed).
2. **Prepare inputs**
   - Copy your scene images into `data/input_images`.
3. **Extract frames (if starting from video)**
   - `python run.py video-to-frames --video path/to/capture.mp4 --output data/input_images`
4. **Run Structure-from-Motion**
   - `python run.py colmap --images data/input_images`
5. **Configure training**
   - Edit `training/configs/default.yaml` to match your scene and hardware.
6. **Train Gaussian Splatting**
   - `python run.py train --config training/configs/default.yaml --max-steps 1000`
7. **Export models**
   - The trained Gaussian point cloud will be written to `outputs/3dgs_models`.

## Pipeline CLI

- `python run.py video-to-frames …` – convert a captured video into a frame directory using ffmpeg.
- `python run.py colmap …` – run SfM + MVS via COLMAP and store outputs in `colmap/`.
- `python run.py train …` – launch the (stub) Gaussian Splatting trainer with the chosen config.
- `python run.py all …` – end-to-end pipeline: extract frames, run COLMAP, then train.

See `python run.py --help` or each sub-command’s `--help` flag for the complete set of options.

## Next Steps

- Tune `scripts/run_colmap.py` with matcher/mapper parameters suited to your capture hardware.
- Replace the stubbed logic inside `scripts/train_3dgs.py` with a real Gaussian Splatting trainer.
- Extend `training/configs/default.yaml` with scene-specific hyperparameters and optimizer settings.

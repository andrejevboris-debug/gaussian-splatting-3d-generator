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
3. **Run Structure-from-Motion**
   - Adapt and execute `scripts/run_colmap.sh` to generate sparse/dense reconstructions in `colmap`.
4. **Configure training**
   - Edit `training/configs/default.yaml` to match your scene and hardware.
5. **Train Gaussian Splatting**
   - Run `python scripts/train_3dgs.py --config training/configs/default.yaml`.
6. **Export models**
   - The trained Gaussian point cloud will be written to `outputs/3dgs_models`.

## Next Steps

- Flesh out `scripts/run_colmap.sh` with COLMAP commands suited to your capture rig.
- Implement the training loop inside `scripts/train_3dgs.py`, integrating your preferred 3DGS library.
- Extend `training/configs/default.yaml` with scene-specific hyperparameters and optimizer settings.

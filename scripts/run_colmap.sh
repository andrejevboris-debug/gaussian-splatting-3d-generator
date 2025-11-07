#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGES_DIR="${ROOT_DIR}/data/input_images"
COLMAP_DIR="${ROOT_DIR}/colmap"
DATABASE_PATH="${COLMAP_DIR}/database/colmap.db"
SPARSE_DIR="${COLMAP_DIR}/sparse"
DENSE_DIR="${COLMAP_DIR}/dense"

mkdir -p "${COLMAP_DIR}/database" "${SPARSE_DIR}" "${DENSE_DIR}"
rm -f "${DATABASE_PATH}"

echo "[COLMAP] Creating new database at ${DATABASE_PATH}"
colmap database_creator --database_path "${DATABASE_PATH}"

echo "[COLMAP] Extracting features from ${IMAGES_DIR}"
colmap feature_extractor \
  --database_path "${DATABASE_PATH}" \
  --image_path "${IMAGES_DIR}"

echo "[COLMAP] Matching features"
colmap exhaustive_matcher \
  --database_path "${DATABASE_PATH}"

echo "[COLMAP] Running sparse reconstruction"
mkdir -p "${SPARSE_DIR}/0"
colmap mapper \
  --database_path "${DATABASE_PATH}" \
  --image_path "${IMAGES_DIR}" \
  --output_path "${SPARSE_DIR}"

echo "[COLMAP] Converting sparse models to text"
for MODEL_DIR in "${SPARSE_DIR}"/*; do
  if [ -d "${MODEL_DIR}" ]; then
    colmap model_converter \
      --input_path "${MODEL_DIR}" \
      --output_path "${MODEL_DIR}_text" \
      --output_type TXT
  fi
done

echo "[COLMAP] Undistorting images"
colmap image_undistorter \
  --image_path "${IMAGES_DIR}" \
  --input_path "${SPARSE_DIR}/0" \
  --output_path "${DENSE_DIR}" \
  --output_type COLMAP \
  --max_image_size 2000

echo "[COLMAP] Dense stereo"
colmap patch_match_stereo \
  --workspace_path "${DENSE_DIR}" \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true

echo "[COLMAP] Fusing depth maps"
colmap stereo_fusion \
  --workspace_path "${DENSE_DIR}" \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path "${DENSE_DIR}/fused.ply"

echo "[COLMAP] Done"

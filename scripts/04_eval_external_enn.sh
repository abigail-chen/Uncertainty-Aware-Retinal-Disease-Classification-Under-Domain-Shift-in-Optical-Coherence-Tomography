#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${ENN_CKPT:?Set ENN_CKPT to the trained ENN best.pt path.}"

OCTDL_ROOT="${OCTDL_ROOT:-/path/to/OCTDL}"
OCTID_ROOT="${OCTID_ROOT:-/path/to/OCTID}"
SRIN_ROOT="${SRIN_ROOT:-/path/to/Srinivasan}"

mkdir -p results/enn_external

python src/evaluation/eval_external_enn.py \
  --ckpt "$ENN_CKPT" \
  --external_csv src/data/external_OCTDL/labels_for_eval_4class_MAPPED.csv \
  --out_dir results/enn_external/octdl \
  --root "$OCTDL_ROOT" \
  --arch resnet18 \
  --img_size 224 \
  --batch_size 64 \
  --num_workers 4 \
  --ece_bins 15 \
  --drop_unknown 1

python src/evaluation/eval_external_enn.py \
  --ckpt "$ENN_CKPT" \
  --external_csv src/data/external_OCTID/labels_for_eval_4class_MAPPED.csv \
  --out_dir results/enn_external/octid \
  --root "$OCTID_ROOT" \
  --arch resnet18 \
  --img_size 224 \
  --batch_size 64 \
  --num_workers 4 \
  --ece_bins 15 \
  --drop_unknown 1

python src/evaluation/eval_external_enn.py \
  --ckpt "$ENN_CKPT" \
  --external_csv src/data/external_Srinivasan/labels_for_eval_4class_MAPPED.csv \
  --out_dir results/enn_external/srinivasan \
  --root "$SRIN_ROOT" \
  --arch resnet18 \
  --img_size 224 \
  --batch_size 64 \
  --num_workers 4 \
  --ece_bins 15 \
  --drop_unknown 1

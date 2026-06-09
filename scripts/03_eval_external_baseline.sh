#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${BASELINE_CKPT:?Set BASELINE_CKPT to the trained baseline best.pt path.}"

OCTDL_ROOT="${OCTDL_ROOT:-/path/to/OCTDL}"
OCTID_ROOT="${OCTID_ROOT:-/path/to/OCTID}"
SRIN_ROOT="${SRIN_ROOT:-/path/to/Srinivasan}"

mkdir -p results/baseline_external

python src/evaluation/eval_external_baseline.py \
  --ckpt "$BASELINE_CKPT" \
  --data_root "$OCTDL_ROOT" \
  --labels_csv src/data/external_OCTDL/labels_for_eval_4class_MAPPED.csv \
  --arch resnet18 \
  --img_size 224 \
  --batch_size 64 \
  --num_workers 4 \
  --use_split all \
  --ece_bins 15 \
  --out_json results/baseline_external/octdl_baseline_metrics.json

python src/evaluation/eval_external_baseline.py \
  --ckpt "$BASELINE_CKPT" \
  --data_root "$OCTID_ROOT" \
  --labels_csv src/data/external_OCTID/labels_for_eval_4class_MAPPED.csv \
  --arch resnet18 \
  --img_size 224 \
  --batch_size 64 \
  --num_workers 4 \
  --use_split all \
  --ece_bins 15 \
  --out_json results/baseline_external/octid_baseline_metrics.json

python src/evaluation/eval_external_baseline.py \
  --ckpt "$BASELINE_CKPT" \
  --data_root "$SRIN_ROOT" \
  --labels_csv src/data/external_Srinivasan/labels_for_eval_4class_MAPPED.csv \
  --arch resnet18 \
  --img_size 224 \
  --batch_size 64 \
  --num_workers 4 \
  --use_split all \
  --ece_bins 15 \
  --out_json results/baseline_external/srinivasan_baseline_metrics.json

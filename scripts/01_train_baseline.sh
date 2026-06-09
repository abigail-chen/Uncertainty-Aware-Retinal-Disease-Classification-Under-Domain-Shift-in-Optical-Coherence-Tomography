#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

KERMANY_DATA_ROOT="${KERMANY_DATA_ROOT:-/path/to/kermany_clean2}"
KERMANY_LABELS_CSV="${KERMANY_LABELS_CSV:-/path/to/kermany_clean2/labels.csv}"

python src/training/train_baseline_Kermany.py \
  --project_root "$REPO_ROOT" \
  --data_root "$KERMANY_DATA_ROOT" \
  --labels_csv "$KERMANY_LABELS_CSV" \
  --arch resnet18 \
  --pretrained 1 \
  --img_size 224 \
  --batch_size 64 \
  --epochs 50 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --val_frac 0.1 \
  --num_workers 4 \
  --seed 42 \
  --ece_bins 15 \
  --balance fixed_per_class \
  --per_class 5000 \
  --scheduler none \
  --early_stop 0 \
  --runs_dir runs/baseline_kermany

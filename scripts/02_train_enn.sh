#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_CSV="${TRAIN_CSV:-src/data/Kermany/train_subset.csv}"
VAL_CSV="${VAL_CSV:-src/data/Kermany/val.csv}"
TEST_CSV="${TEST_CSV:-src/data/Kermany/test_official.csv}"

python src/training/train_enn_kermany.py \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --test_csv "$TEST_CSV" \
  --arch resnet18 \
  --pretrained 1 \
  --img_size 224 \
  --batch_size 64 \
  --epochs 50 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --seed 42 \
  --ece_bins 15 \
  --lam 0.7 \
  --early_stop 0 \
  --num_workers 4 \
  --runs_dir runs/enn_kermany

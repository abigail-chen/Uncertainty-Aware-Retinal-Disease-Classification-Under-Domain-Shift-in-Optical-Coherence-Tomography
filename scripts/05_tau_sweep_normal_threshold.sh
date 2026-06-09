#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${INTERNAL_ENN_CSV:?Set INTERNAL_ENN_CSV to the internal ENN prediction CSV with prob_* columns.}"
: "${SRIN_ENN_CSV:?Set SRIN_ENN_CSV to the Srinivasan ENN prediction CSV with prob_* columns.}"
: "${OCTDL_ENN_CSV:?Set OCTDL_ENN_CSV to the OCTDL ENN prediction CSV with prob_* columns.}"
: "${OCTID_ENN_CSV:?Set OCTID_ENN_CSV to the OCTID ENN prediction CSV with prob_* columns.}"

mkdir -p results/tau_sweep

TAU_CSV="results/tau_sweep/tau_values.csv"

cat > "$TAU_CSV" <<EOF
tau
0.236
0.267
0.327
0.404
0.532
0.724
0.838
0.869
0.883
EOF

python src/evaluation/tau_sweep_normal_threshold.py \
  --internal_csv "$INTERNAL_ENN_CSV" \
  --srin_csv "$SRIN_ENN_CSV" \
  --octdl_csv "$OCTDL_ENN_CSV" \
  --octid_csv "$OCTID_ENN_CSV" \
  --tau_csv "$TAU_CSV" \
  --out_csv results/tau_sweep/tau_sweep_summary.csv \
  --criterion safety \
  --acc_floor_drop 0.01

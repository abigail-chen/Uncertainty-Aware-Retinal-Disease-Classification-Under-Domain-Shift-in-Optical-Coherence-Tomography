#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${INTERNAL_ENN_CSV:?Set INTERNAL_ENN_CSV to the internal ENN prediction CSV with u_mass and prob_* columns.}"
: "${SRIN_ENN_CSV:?Set SRIN_ENN_CSV to the Srinivasan ENN prediction CSV with u_mass and prob_* columns.}"
: "${OCTDL_ENN_CSV:?Set OCTDL_ENN_CSV to the OCTDL ENN prediction CSV with u_mass and prob_* columns.}"
: "${OCTID_ENN_CSV:?Set OCTID_ENN_CSV to the OCTID ENN prediction CSV with u_mass and prob_* columns.}"

mkdir -p results/umass_thresholding

UMASS_TAU_CSV="results/umass_thresholding/umass_tau_values.csv"

cat > "$UMASS_TAU_CSV" <<EOF
tau
0.70
0.75
0.80
0.85
0.90
0.95
EOF

python src/evaluation/umass_thresholding.py \
  --internal_csv "$INTERNAL_ENN_CSV" \
  --srin_csv "$SRIN_ENN_CSV" \
  --octdl_csv "$OCTDL_ENN_CSV" \
  --octid_csv "$OCTID_ENN_CSV" \
  --tau_csv "$UMASS_TAU_CSV" \
  --out_csv results/umass_thresholding/umass_thresholding_summary.csv \
  --criterion safety \
  --acc_floor_drop 0.01

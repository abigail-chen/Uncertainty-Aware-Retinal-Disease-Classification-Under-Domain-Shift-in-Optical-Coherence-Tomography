#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${INTERNAL_ENN_CSV:?Set INTERNAL_ENN_CSV to the internal ENN prediction CSV with u_mass.}"
: "${SRIN_ENN_CSV:?Set SRIN_ENN_CSV to the Srinivasan ENN prediction CSV with u_mass.}"
: "${OCTDL_ENN_CSV:?Set OCTDL_ENN_CSV to the OCTDL ENN prediction CSV with u_mass.}"
: "${OCTID_ENN_CSV:?Set OCTID_ENN_CSV to the OCTID ENN prediction CSV with u_mass.}"

mkdir -p results/selective_coverage

python src/evaluation/selective_coverage_accuracy_umass.py \
  --internal_csv "$INTERNAL_ENN_CSV" \
  --srin_csv "$SRIN_ENN_CSV" \
  --octdl_csv "$OCTDL_ENN_CSV" \
  --octid_csv "$OCTID_ENN_CSV" \
  --min_coverage 50 \
  --max_coverage 100 \
  --step 1 \
  --out_csv results/selective_coverage/selective_coverage_accuracy_umass.csv

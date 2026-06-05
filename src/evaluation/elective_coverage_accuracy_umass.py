from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_prediction_csv(path: str) -> pd.DataFrame:
    """
    Load one ENN prediction CSV for selective prediction based on u_mass.

    Required:
        u_mass

    Accuracy is computed from one of:
        1. correct
        2. y_true and y_pred
        3. true_label and pred_label
    """
    df = pd.read_csv(path)

    if "u_mass" not in df.columns:
        raise ValueError(f"{path} is missing required column: u_mass")

    if "correct" in df.columns:
        df["correct_for_curve"] = df["correct"].astype(float)

    elif {"y_true", "y_pred"}.issubset(df.columns):
        df["correct_for_curve"] = (
            df["y_true"].astype(str) == df["y_pred"].astype(str)
        ).astype(float)

    elif {"true_label", "pred_label"}.issubset(df.columns):
        df["correct_for_curve"] = (
            df["true_label"].astype(str).str.upper()
            == df["pred_label"].astype(str).str.upper()
        ).astype(float)

    else:
        raise ValueError(
            f"{path} must contain either 'correct', "
            f"or 'y_true'/'y_pred', or 'true_label'/'pred_label'."
        )

    df = df.dropna(subset=["u_mass", "correct_for_curve"]).copy()
    return df


def retained_accuracy_by_umass(
    df: pd.DataFrame,
    coverage_percent: float,
) -> tuple[float, int, float]:
    """
    Selective prediction using u_mass.

    Lower u_mass = lower uncertainty.
    Therefore, retained cases are selected by sorting u_mass from low to high.
    """
    if not (0 < coverage_percent <= 100):
        raise ValueError("coverage_percent must be in (0, 100].")

    n_total = len(df)
    n_keep = int(np.ceil(n_total * coverage_percent / 100.0))
    n_keep = max(1, min(n_keep, n_total))

    retained = df.sort_values("u_mass", ascending=True).head(n_keep)

    acc = float(retained["correct_for_curve"].mean())
    u_threshold = float(retained["u_mass"].max())

    return acc, n_keep, u_threshold


def make_curve(
    df: pd.DataFrame,
    coverages: list[float],
) -> tuple[list[float], list[int], list[float]]:
    accs = []
    ns = []
    thresholds = []

    for cov in coverages:
        acc, n_keep, u_thr = retained_accuracy_by_umass(df, cov)
        accs.append(acc)
        ns.append(n_keep)
        thresholds.append(u_thr)

    return accs, ns, thresholds


def main():
    parser = argparse.ArgumentParser(
        description="Create CSV table for selective coverage-vs-accuracy using ENN u_mass."
    )

    parser.add_argument("--internal_csv", required=True)
    parser.add_argument("--srin_csv", required=True)
    parser.add_argument("--octdl_csv", required=True)
    parser.add_argument("--octid_csv", required=True)

    parser.add_argument("--min_coverage", type=int, default=50)
    parser.add_argument("--max_coverage", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)

    parser.add_argument("--out_csv", required=True)

    args = parser.parse_args()

    df_internal = load_prediction_csv(args.internal_csv)
    df_srin = load_prediction_csv(args.srin_csv)
    df_octdl = load_prediction_csv(args.octdl_csv)
    df_octid = load_prediction_csv(args.octid_csv)

    coverages = list(
        range(args.max_coverage, args.min_coverage - 1, -args.step)
    )

    internal_acc, internal_n, internal_u_thr = make_curve(df_internal, coverages)
    srin_acc, srin_n, srin_u_thr = make_curve(df_srin, coverages)
    octdl_acc, octdl_n, octdl_u_thr = make_curve(df_octdl, coverages)
    octid_acc, octid_n, octid_u_thr = make_curve(df_octid, coverages)

    avg_external_acc = [
        float(np.nanmean([s, d, i]))
        for s, d, i in zip(srin_acc, octdl_acc, octid_acc)
    ]

    out = pd.DataFrame(
        {
            "coverage_retained_percent": coverages,
            "internal_acc": internal_acc,
            "avg_external_acc": avg_external_acc,
            "srinivasan_acc": srin_acc,
            "octdl_acc": octdl_acc,
            "octid_acc": octid_acc,
            "internal_n_retained": internal_n,
            "srinivasan_n_retained": srin_n,
            "octdl_n_retained": octdl_n,
            "octid_n_retained": octid_n,
            "internal_umass_threshold": internal_u_thr,
            "srinivasan_umass_threshold": srin_u_thr,
            "octdl_umass_threshold": octdl_u_thr,
            "octid_umass_threshold": octid_u_thr,
        }
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[Saved] {out_path}")

    print("\nKey coverage levels:")
    key = out[out["coverage_retained_percent"].isin([100, 90, 80, 70, 60, 50])]
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CLASSES_DEFAULT = ["CNV", "DME", "DRUSEN", "NORMAL"]


def find_prob_cols(df: pd.DataFrame):
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        raise ValueError("No prob_* columns found.")
    classes = [c.replace("prob_", "") for c in prob_cols]
    return prob_cols, classes


def load_probs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "y_true" not in df.columns:
        raise ValueError(f"{path} missing required column y_true")
    df["y_true"] = df["y_true"].astype(str).str.upper()
    return df


def apply_tau_rule_prob_normal(df: pd.DataFrame, prob_cols, tau: float) -> np.ndarray:
    """
    Tau-only rule (NO uncertainty):

      if prob_NORMAL >= tau:
          predict NORMAL
      else:
          predict argmax among DISEASE classes (exclude NORMAL)

    Assumes df has prob_NORMAL and prob_{CNV,DME,DRUSEN}.
    """
    if "prob_NORMAL" not in df.columns:
        raise ValueError("Need prob_NORMAL column for tau rule.")

    # disease argmax
    disease_prob_cols = [c for c in prob_cols if c != "prob_NORMAL"]
    if not disease_prob_cols:
        raise ValueError("No disease prob cols found (expected prob_CNV/prob_DME/prob_DRUSEN).")

    disease_probs = df[disease_prob_cols].to_numpy()
    disease_idx = disease_probs.argmax(axis=1)
    disease_classes = np.array([c.replace("prob_", "") for c in disease_prob_cols])
    disease_pred = disease_classes[disease_idx]

    # tau gate on NORMAL
    pN = df["prob_NORMAL"].to_numpy()
    pred = disease_pred.copy()
    pred[pN >= tau] = "NORMAL"
    return pred


def accuracy(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float((y_true == y_pred).mean())


def confusion_counts(y_true, y_pred, classes):
    # returns dict[class] -> TP,FP,FN,TN
    out = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        tn = np.sum((y_true != c) & (y_pred != c))
        out[c] = (tp, fp, fn, tn)
    return out


def macro_fn_fp_rates(y_true, y_pred, classes):
    """
    Macro FN rate (OvR): mean over classes of FN/(FN+TP)
    Macro FP rate (OvR): mean over classes of FP/(FP+TN)
    """
    cc = confusion_counts(y_true, y_pred, classes)
    fn_rates = []
    fp_rates = []
    for c in classes:
        tp, fp, fn, tn = cc[c]
        fn_denom = fn + tp
        fp_denom = fp + tn
        fn_rate = (fn / fn_denom) if fn_denom > 0 else np.nan
        fp_rate = (fp / fp_denom) if fp_denom > 0 else np.nan
        fn_rates.append(fn_rate)
        fp_rates.append(fp_rate)
    return float(np.nanmean(fn_rates)), float(np.nanmean(fp_rates))


def disease_vs_normal_auc(df: pd.DataFrame) -> float:
    """
    Binary Disease-vs-Normal AUC with score = 1 - p(NORMAL).
    y_bin = 1 for Disease (CNV/DME/DRUSEN), 0 for NORMAL.
    """
    if "prob_NORMAL" not in df.columns:
        raise ValueError("Need prob_NORMAL column to compute Disease-vs-Normal AUC.")

    y_true = df["y_true"].astype(str).str.upper().to_numpy()
    y_bin = (y_true != "NORMAL").astype(int)
    score = 1.0 - df["prob_NORMAL"].to_numpy()

    # roc_auc_score fails if only one class present
    if len(np.unique(y_bin)) < 2:
        return np.nan
    return float(roc_auc_score(y_bin, score))


def summarize_one(df: pd.DataFrame, pred: np.ndarray, classes):
    acc = accuracy(df["y_true"], pred)
    auc = disease_vs_normal_auc(df)
    fn, fp = macro_fn_fp_rates(df["y_true"], pred, classes)
    return acc, auc, fn, fp


def choose_best(df_summary: pd.DataFrame, mode: str, acc_floor_drop: float = 0.01):
    """
    mode:
      - accuracy: max avg_external_acc, tie-break lower avg_external_fn, then lower avg_external_fp
      - balanced: min (avg_external_fn + avg_external_fp), tie-break higher avg_external_acc
      - safety: among rows within acc_floor_drop of best avg_external_acc, minimize avg_external_fn, then avg_external_fp
    """
    d = df_summary.copy()
    d["J_bal"] = d["avg_external_macro_fn_rate"] + d["avg_external_macro_fp_rate"]

    if mode == "accuracy":
        best = d.sort_values(
            ["avg_external_acc", "avg_external_macro_fn_rate", "avg_external_macro_fp_rate"],
            ascending=[False, True, True]
        ).iloc[0]
        return best

    if mode == "balanced":
        best = d.sort_values(
            ["J_bal", "avg_external_acc"],
            ascending=[True, False]
        ).iloc[0]
        return best

    if mode == "safety":
        best_acc = d["avg_external_acc"].max()
        pool = d[d["avg_external_acc"] >= (best_acc - acc_floor_drop)].copy()
        best = pool.sort_values(
            ["avg_external_macro_fn_rate", "avg_external_macro_fp_rate", "avg_external_acc"],
            ascending=[True, True, False]
        ).iloc[0]
        return best

    raise ValueError("mode must be one of: accuracy, balanced, safety")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--internal_csv", required=True)
    ap.add_argument("--srin_csv", required=True)
    ap.add_argument("--octdl_csv", required=True)
    ap.add_argument("--octid_csv", required=True)

    ap.add_argument("--tau_csv", required=True, help="CSV containing tau values tried (must include a 'tau' column)")
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--criterion", default="safety", choices=["accuracy", "balanced", "safety"])
    ap.add_argument(
        "--acc_floor_drop",
        type=float,
        default=0.01,
        help="Only for safety: keep avg_external_acc within this absolute drop from best (e.g. 0.01=1%)"
    )

    args = ap.parse_args()

    # Load probs
    df_int = load_probs(args.internal_csv)
    df_srin = load_probs(args.srin_csv)
    df_octdl = load_probs(args.octdl_csv)
    df_octid = load_probs(args.octid_csv)

    # Determine classes (prefer prob cols if they look right)
    prob_cols_int, classes_int = find_prob_cols(df_int)
    classes_int = [c.upper() for c in classes_int]
    classes = classes_int if set(CLASSES_DEFAULT).issubset(set(classes_int)) else CLASSES_DEFAULT

    # Load tau list
    tau_df = pd.read_csv(args.tau_csv)
    if "tau" not in tau_df.columns:
        raise ValueError(f"{args.tau_csv} must contain a 'tau' column; got: {tau_df.columns.tolist()}")
    taus = sorted(tau_df["tau"].dropna().unique().tolist())

    rows = []
    for tau in taus:
        tau = float(tau)

        # prob cols should match, but compute per-dataset safely
        prob_cols_srin, _ = find_prob_cols(df_srin)
        prob_cols_octdl, _ = find_prob_cols(df_octdl)
        prob_cols_octid, _ = find_prob_cols(df_octid)

        pred_int = apply_tau_rule_prob_normal(df_int, prob_cols_int, tau)
        pred_srin = apply_tau_rule_prob_normal(df_srin, prob_cols_srin, tau)
        pred_octdl = apply_tau_rule_prob_normal(df_octdl, prob_cols_octdl, tau)
        pred_octid = apply_tau_rule_prob_normal(df_octid, prob_cols_octid, tau)

        # metrics
        int_acc, int_auc, int_fn, int_fp = summarize_one(df_int, pred_int, classes)
        s_acc, s_auc, s_fn, s_fp = summarize_one(df_srin, pred_srin, classes)
        d_acc, d_auc, d_fn, d_fp = summarize_one(df_octdl, pred_octdl, classes)
        i_acc, i_auc, i_fn, i_fp = summarize_one(df_octid, pred_octid, classes)

        avg_ext_acc = float(np.nanmean([s_acc, d_acc, i_acc]))
        avg_ext_auc = float(np.nanmean([s_auc, d_auc, i_auc]))
        avg_ext_fn = float(np.nanmean([s_fn, d_fn, i_fn]))
        avg_ext_fp = float(np.nanmean([s_fp, d_fp, i_fp]))

        rows.append({
            "tau": tau,

            "internal_acc": int_acc,
            "avg_external_acc": avg_ext_acc,

            "internal_auc_d_vs_n": int_auc,
            "avg_external_auc_d_vs_n": avg_ext_auc,

            "internal_macro_fn_rate": int_fn,
            "avg_external_macro_fn_rate": avg_ext_fn,

            "internal_macro_fp_rate": int_fp,
            "avg_external_macro_fp_rate": avg_ext_fp,

            # optional breakdown
            "srin_acc": s_acc, "octdl_acc": d_acc, "octid_acc": i_acc,
            "srin_auc": s_auc, "octdl_auc": d_auc, "octid_auc": i_auc,
            "srin_fn": s_fn, "octdl_fn": d_fn, "octid_fn": i_fn,
            "srin_fp": s_fp, "octdl_fp": d_fp, "octid_fp": i_fp,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

    print("WROTE:", args.out_csv)
    best = choose_best(out, args.criterion, acc_floor_drop=args.acc_floor_drop)
    print(f"BEST TAU (criterion={args.criterion}):")
    show_cols = [
        "tau",
        "avg_external_acc",
        "avg_external_auc_d_vs_n",
        "avg_external_macro_fn_rate",
        "avg_external_macro_fp_rate",
        "internal_acc",
        "internal_auc_d_vs_n",
        "internal_macro_fn_rate",
        "internal_macro_fp_rate",
    ]
    print(best[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()


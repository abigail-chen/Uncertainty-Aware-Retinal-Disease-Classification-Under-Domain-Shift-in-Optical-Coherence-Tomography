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

def load_umass_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["y_true", "u_mass"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing required column: {col}")
    df["y_true"] = df["y_true"].astype(str).str.upper()
    return df

def argmax_pred(df: pd.DataFrame, prob_cols):
    probs = df[prob_cols].to_numpy()
    idx = probs.argmax(axis=1)
    classes = np.array([c.replace("prob_", "") for c in prob_cols])
    return classes[idx]

def apply_umass_tau_rule(df: pd.DataFrame, prob_cols, tau: float):
    """
    u_mass tau rule (NO abstention):

    base_pred = argmax over all 4 classes
    if base_pred == NORMAL and u_mass >= tau:
        override to argmax among DISEASE classes only (exclude NORMAL)
    else:
        keep base_pred
    """
    if "u_mass" not in df.columns:
        raise ValueError("Need u_mass column.")
    if "prob_NORMAL" not in df.columns:
        raise ValueError("Need prob_NORMAL column.")

    base_pred = argmax_pred(df, prob_cols)
    u = df["u_mass"].to_numpy()

    disease_prob_cols = [c for c in prob_cols if c != "prob_NORMAL"]
    disease_probs = df[disease_prob_cols].to_numpy()
    disease_idx = disease_probs.argmax(axis=1)
    disease_classes = np.array([c.replace("prob_", "") for c in disease_prob_cols])
    disease_pred = disease_classes[disease_idx]

    pred = base_pred.copy()
    mask = (base_pred == "NORMAL") & (u >= tau)
    pred[mask] = disease_pred[mask]
    return pred

def accuracy(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float((y_true == y_pred).mean())

def confusion_counts(y_true, y_pred, classes):
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
    cc = confusion_counts(y_true, y_pred, classes)
    fn_rates, fp_rates = [], []
    for c in classes:
        tp, fp, fn, tn = cc[c]
        fn_denom = fn + tp
        fp_denom = fp + tn
        fn_rates.append((fn / fn_denom) if fn_denom > 0 else np.nan)
        fp_rates.append((fp / fp_denom) if fp_denom > 0 else np.nan)
    return float(np.nanmean(fn_rates)), float(np.nanmean(fp_rates))

def disease_vs_normal_auc(df: pd.DataFrame) -> float:
    """
    Binary Disease-vs-Normal AUC with score = 1 - p(NORMAL).
    """
    y_true = df["y_true"].astype(str).str.upper().to_numpy()
    y_bin = (y_true != "NORMAL").astype(int)
    score = 1.0 - df["prob_NORMAL"].to_numpy()
    if len(np.unique(y_bin)) < 2:
        return np.nan
    return float(roc_auc_score(y_bin, score))

def summarize_one(df: pd.DataFrame, pred: np.ndarray, classes):
    acc = accuracy(df["y_true"], pred)
    auc = disease_vs_normal_auc(df)
    fn, fp = macro_fn_fp_rates(df["y_true"], pred, classes)
    return acc, auc, fn, fp

def choose_best(df_summary: pd.DataFrame, mode: str, acc_floor_drop: float = 0.01):
    d = df_summary.copy()
    d["J_bal"] = d["avg_external_macro_fn_rate"] + d["avg_external_macro_fp_rate"]

    if mode == "accuracy":
        return d.sort_values(
            ["avg_external_acc", "avg_external_macro_fn_rate", "avg_external_macro_fp_rate"],
            ascending=[False, True, True]
        ).iloc[0]

    if mode == "balanced":
        return d.sort_values(["J_bal", "avg_external_acc"], ascending=[True, False]).iloc[0]

    if mode == "safety":
        best_acc = d["avg_external_acc"].max()
        pool = d[d["avg_external_acc"] >= (best_acc - acc_floor_drop)].copy()
        return pool.sort_values(
            ["avg_external_macro_fn_rate", "avg_external_macro_fp_rate", "avg_external_acc"],
            ascending=[True, True, False]
        ).iloc[0]

    raise ValueError("mode must be one of: accuracy, balanced, safety")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal_csv", required=True)
    ap.add_argument("--srin_csv", required=True)
    ap.add_argument("--octdl_csv", required=True)
    ap.add_argument("--octid_csv", required=True)
    ap.add_argument("--tau_csv", required=True, help="CSV with a 'tau' column")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--criterion", default="safety", choices=["accuracy", "balanced", "safety"])
    ap.add_argument("--acc_floor_drop", type=float, default=0.01)
    args = ap.parse_args()

    df_int = load_umass_csv(args.internal_csv)
    df_srin = load_umass_csv(args.srin_csv)
    df_octdl = load_umass_csv(args.octdl_csv)
    df_octid = load_umass_csv(args.octid_csv)

    prob_cols_int, classes_int = find_prob_cols(df_int)
    classes_int = [c.upper() for c in classes_int]
    classes = classes_int if set(CLASSES_DEFAULT).issubset(set(classes_int)) else CLASSES_DEFAULT

    tau_df = pd.read_csv(args.tau_csv)
    if "tau" not in tau_df.columns:
        raise ValueError(f"{args.tau_csv} must contain a 'tau' column; got {tau_df.columns.tolist()}")
    taus = sorted(tau_df["tau"].dropna().unique().tolist())

    rows = []
    for tau in taus:
        tau = float(tau)

        prob_cols_srin, _ = find_prob_cols(df_srin)
        prob_cols_octdl, _ = find_prob_cols(df_octdl)
        prob_cols_octid, _ = find_prob_cols(df_octid)

        pred_int  = apply_umass_tau_rule(df_int,  prob_cols_int,  tau)
        pred_srin = apply_umass_tau_rule(df_srin, prob_cols_srin, tau)
        pred_octdl= apply_umass_tau_rule(df_octdl,prob_cols_octdl,tau)
        pred_octid= apply_umass_tau_rule(df_octid,prob_cols_octid,tau)

        int_acc, int_auc, int_fn, int_fp = summarize_one(df_int, pred_int, classes)
        s_acc, s_auc, s_fn, s_fp = summarize_one(df_srin, pred_srin, classes)
        d_acc, d_auc, d_fn, d_fp = summarize_one(df_octdl, pred_octdl, classes)
        i_acc, i_auc, i_fn, i_fp = summarize_one(df_octid, pred_octid, classes)

        avg_ext_acc = float(np.nanmean([s_acc, d_acc, i_acc]))
        avg_ext_auc = float(np.nanmean([s_auc, d_auc, i_auc]))
        avg_ext_fn  = float(np.nanmean([s_fn,  d_fn,  i_fn ]))
        avg_ext_fp  = float(np.nanmean([s_fp,  d_fp,  i_fp ]))

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
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print("WROTE:", args.out_csv)

    best = choose_best(out, args.criterion, acc_floor_drop=args.acc_floor_drop)
    print(f"BEST TAU (criterion={args.criterion}):")
    print(best.to_string(index=False))

if __name__ == "__main__":
    main()

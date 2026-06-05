# src/eval_external_enn.py
# ------------------------------------------------------------
# Evaluate a trained ENN checkpoint (best.pt) on an external CSV.
# Saves:
#   <out_dir>/external_metrics.json
#   <out_dir>/external_uncertainty.csv
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, roc_auc_score


# -----------------------------
# Model (same as training)
# -----------------------------
def build_backbone(arch: str, pretrained: bool):
    arch = arch.lower()
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    elif arch == "resnet101":
        m = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError("arch must be one of: resnet18, resnet50, resnet101")
    in_feats = m.fc.in_features
    m.fc = nn.Identity()
    return m, in_feats


class EvidentialNet(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool):
        super().__init__()
        self.backbone, feat_dim = build_backbone(arch, pretrained)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.head(z)
        evidence = F.softplus(logits)   # >= 0
        alpha = evidence + 1.0
        return alpha


# -----------------------------
# Data
# -----------------------------
class ExternalFromCSV(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict[str, int], tfm, root: str | None = None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.tfm = tfm
        self.root = Path(root) if root else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        if "dst_path" in row and isinstance(row["dst_path"], str) and len(row["dst_path"]) > 0:
            img_path = Path(row["dst_path"])
        elif "relpath" in row and self.root is not None:
            img_path = self.root / str(row["relpath"])
        else:
            raise ValueError("CSV must contain dst_path OR (relpath + --root)")

        label_str = str(row["label"])
        img = Image.open(str(img_path)).convert("RGB")
        x = self.tfm(img)

        y = self.class_to_idx[label_str]
        return x, y, str(img_path)


def get_transforms(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == labels).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


@torch.no_grad()
def brier_score_multiclass(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    y = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    y[np.arange(labels.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True, help="Path to best.pt")
    ap.add_argument("--external_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--root", type=str, default=None, help="Used only if CSV has relpath instead of dst_path")

    ap.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50", "resnet101"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--ece_bins", type=int, default=15)

    ap.add_argument("--label_map_json", type=str, default=None,
                    help="JSON dict mapping external labels -> Kermany labels")
    ap.add_argument("--drop_unknown", type=int, default=1,
                    help="If 1: drop rows whose mapped label not in Kermany classes; if 0: error")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    # Load checkpoint (gets Kermany class order)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    class_names = ckpt["classes"]
    class_to_idx = ckpt["class_to_idx"]
    num_classes = len(class_names)
    print(f"[Info] Kermany classes ({num_classes}): {class_names}")

    # Load external CSV
    df = pd.read_csv(args.external_csv)

    # Apply label mapping if provided
    if args.label_map_json is not None:
        mapping = json.loads(Path(args.label_map_json).read_text())
        df["label_orig"] = df["label"].astype(str)
        df["label"] = df["label_orig"].map(lambda x: mapping.get(x, x))

    # Filter / validate labels
    ok = df["label"].astype(str).isin(class_to_idx.keys())
    if not ok.all():
        bad = df.loc[~ok, "label"].astype(str).value_counts().to_dict()
        msg = f"[Warn] Found labels not in Kermany classes: {bad}"
        if args.drop_unknown:
            print(msg + " -> dropping these rows")
            df = df.loc[ok].reset_index(drop=True)
        else:
            raise ValueError(msg + " (set --drop_unknown 1 to drop them)")

    print(f"[Info] external rows after mapping/filter: {len(df)}")
    if len(df) == 0:
        raise RuntimeError("No external samples left after label filtering. Check label mapping.")

    # DataLoader
    tfm = get_transforms(args.img_size)
    ds = ExternalFromCSV(df, class_to_idx, tfm, root=args.root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # Model
    model = EvidentialNet(args.arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Run inference
    all_probs, all_labels, all_paths = [], [], []
    all_strength, all_umass, all_pmax, all_entropy = [], [], [], []
    all_preds = []

    K = num_classes

    with torch.no_grad():
        for x, y, paths in dl:
            x = x.to(device, non_blocking=True)
            alpha = model(x)                 # [B,K]
            S = alpha.sum(dim=1)             # [B]
            probs = (alpha / S.unsqueeze(1)) # [B,K]

            p_max, pred = probs.max(dim=1)
            umass = K / torch.clamp(S, min=1e-8)

            probs_np = probs.cpu().numpy()
            y_np = y.numpy()
            pred_np = pred.cpu().numpy()

            ent_np = predictive_entropy(probs_np)

            all_probs.append(probs_np)
            all_labels.append(y_np)
            all_paths.extend(list(paths))

            all_strength.append(S.cpu().numpy())
            all_umass.append(umass.cpu().numpy())
            all_pmax.append(p_max.cpu().numpy())
            all_entropy.append(ent_np)
            all_preds.append(pred_np)

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    strength = np.concatenate(all_strength)
    u_mass = np.concatenate(all_umass)
    p_max = np.concatenate(all_pmax)
    entropy = np.concatenate(all_entropy)
    correct = (preds == labels)

    # Metrics
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    ece = expected_calibration_error(probs, labels, n_bins=args.ece_bins)
    brier = brier_score_multiclass(probs, labels, num_classes=num_classes)

    metrics = {
        "n": int(len(labels)),
        "acc": float(acc),
        "macro_auc_ovr": float(auc),
        "ece": float(ece),
        "brier": float(brier),
        "p_max_mean": float(p_max.mean()),
        "entropy_mean": float(entropy.mean()),
        "strength_mean": float(strength.mean()),
        "u_mass_mean": float(u_mass.mean()),
        "p_max_mean_correct": float(p_max[correct].mean()) if correct.any() else float("nan"),
        "p_max_mean_incorrect": float(p_max[~correct].mean()) if (~correct).any() else float("nan"),
        "entropy_mean_correct": float(entropy[correct].mean()) if correct.any() else float("nan"),
        "entropy_mean_incorrect": float(entropy[~correct].mean()) if (~correct).any() else float("nan"),
        "strength_mean_correct": float(strength[correct].mean()) if correct.any() else float("nan"),
        "strength_mean_incorrect": float(strength[~correct].mean()) if (~correct).any() else float("nan"),
        "u_mass_mean_correct": float(u_mass[correct].mean()) if correct.any() else float("nan"),
        "u_mass_mean_incorrect": float(u_mass[~correct].mean()) if (~correct).any() else float("nan"),
    }

    # Save files
    (out_dir / "external_metrics.json").write_text(json.dumps(metrics, indent=2))

    df_out = pd.DataFrame({
        "path": np.array(all_paths, dtype=object),
        "y_true": labels,
        "y_pred": preds,
        "p_max": p_max,
        "entropy": entropy,
        "strength": strength,
        "u_mass": u_mass,
        "correct": correct.astype(int),
    })
    # keep original labels if present
    if "label_orig" in df.columns:
        df_out["label_orig"] = df["label_orig"].astype(str).values
    df_out["label_mapped"] = df["label"].astype(str).values

    df_out.to_csv(out_dir / "external_uncertainty.csv", index=False)

    print("\n[External metrics]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"\n[Done] Wrote:\n  {out_dir / 'external_metrics.json'}\n  {out_dir / 'external_uncertainty.csv'}")


if __name__ == "__main__":
    main()

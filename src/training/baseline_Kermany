# src/baseline_Kermany.py
# ------------------------------------------------------------
# Baseline Softmax ResNet on Kermany using labels.csv (cleaned layout)
#   data/kermany_clean2/
#     train/<LABEL>/<PATIENT_ID>/*.jpeg
#     test/<LABEL>/<PATIENT_ID>/*.jpeg
#     labels.csv
#
# Features:
# - patient-level val split (no leakage)
# - balance options: none / class_weight / fixed_per_class / sampler
# - optional LR scheduler: none / cosine / step
# - metrics: acc, macro AUROC (ovr), ECE, Brier
# - uncertainty: predictive entropy + per-image CSV
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms, models
from sklearn.metrics import accuracy_score, roc_auc_score


# =========================
# Defaults for server
# =========================
# Your server path looks like: ~/projects/ROP_Project
PROJECT_ROOT_DEFAULT = Path.home() / "projects" / "ROP_Project"
DATA_ROOT_DEFAULT = PROJECT_ROOT_DEFAULT / "data" / "kermany_clean2"
LABELS_CSV_DEFAULT = DATA_ROOT_DEFAULT / "labels.csv"


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True is faster but can introduce minor nondeterminism across GPUs;
    # keep it True for speed unless you need strict determinism.
    torch.backends.cudnn.benchmark = True


# -----------------------------
# Dataset
# -----------------------------
class KermanyFromCSV(Dataset):
    """Returns (img_tensor, y_int, img_path_str)."""

    def __init__(self, df: pd.DataFrame, class_to_idx: dict[str, int], tfm):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img_path = str(row["dst_path"])
        label_str = row["label"]

        img = Image.open(img_path).convert("RGB")
        img = self.tfm(img)

        y = self.class_to_idx[label_str]
        return img, y, img_path


# -----------------------------
# Transforms
# -----------------------------
def get_transforms(img_size: int):
    # mild aug only (OCT structure-sensitive)
    train_tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    test_tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_tfm, test_tfm


# -----------------------------
# Model
# -----------------------------
def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        m = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
    elif arch == "resnet50":
        m = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
    elif arch == "resnet101":
        m = models.resnet101(
            weights=models.ResNet101_Weights.DEFAULT if pretrained else None
        )
    else:
        raise ValueError("arch must be one of: resnet18, resnet50, resnet101")

    in_feats = m.fc.in_features
    m.fc = nn.Linear(in_feats, num_classes)
    return m


# -----------------------------
# Calibration metrics
# -----------------------------
@torch.no_grad()
def expected_calibration_error(
        probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
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
def brier_score_multiclass(
        probs: np.ndarray, labels: np.ndarray, num_classes: int
) -> float:
    y = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    y[np.arange(labels.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


# -----------------------------
# Uncertainty: predictive entropy
# -----------------------------
def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Entropy in nats: -sum p log p."""
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def normalized_entropy(ent: np.ndarray, num_classes: int) -> np.ndarray:
    """Normalize entropy to [0,1] by dividing by log(K)."""
    return ent / math.log(num_classes)


# -----------------------------
# Train / eval
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y, _paths in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    all_probs, all_labels, all_paths = [], [], []
    for x, y, paths in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
        all_paths.extend(list(paths))
    return np.concatenate(all_probs), np.concatenate(all_labels), np.array(all_paths, dtype=object)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, ece_bins: int, return_details: bool = False):
    probs, labels, paths = predict_probs(model, loader, device)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(labels, preds)

    # AUC can be nan if some classes are missing in this split
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    ece = expected_calibration_error(probs, labels, n_bins=ece_bins)
    brier = brier_score_multiclass(probs, labels, num_classes=num_classes)

    ent = predictive_entropy(probs)
    ent_n = normalized_entropy(ent, num_classes)

    correct = (preds == labels)
    out = {
        "acc": float(acc),
        "macro_auc_ovr": float(auc),
        "ece": float(ece),
        "brier": float(brier),
        "entropy_mean": float(ent.mean()),
        "entropy_norm_mean": float(ent_n.mean()),
        "entropy_mean_correct": float(ent[correct].mean()) if correct.any() else float("nan"),
        "entropy_mean_incorrect": float(ent[~correct].mean()) if (~correct).any() else float("nan"),
        "entropy_norm_mean_correct": float(ent_n[correct].mean()) if correct.any() else float("nan"),
        "entropy_norm_mean_incorrect": float(ent_n[~correct].mean()) if (~correct).any() else float("nan"),
    }

    if return_details:
        details = {
            "paths": paths,
            "labels": labels,
            "preds": preds,
            "max_prob": probs.max(axis=1),
            "entropy": ent,
            "entropy_norm": ent_n,
        }
        return out, details

    return out


# -----------------------------
# Patient-level split
# -----------------------------
def make_patient_val_split(df_train: pd.DataFrame, val_frac: float, seed: int):
    """
    Split by patient_id to avoid leakage.
    For Kermany, patient folder is typically single label, so stratify by patient label.
    """
    rng = np.random.RandomState(seed)

    patient_label = (
        df_train.groupby("patient_id")["label"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
    )

    patients = patient_label["patient_id"].values
    labels = patient_label["label"].values

    unique_labels = np.unique(labels)
    val_patients = []
    for lab in unique_labels:
        p_lab = patients[labels == lab]
        n_val = max(1, int(round(len(p_lab) * val_frac)))
        chosen = rng.choice(p_lab, size=n_val, replace=False)
        val_patients.extend(chosen.tolist())

    val_patients = set(val_patients)
    df_val = df_train[df_train["patient_id"].isin(val_patients)].copy()
    df_tr = df_train[~df_train["patient_id"].isin(val_patients)].copy()
    return df_tr, df_val


# -----------------------------
# Balance helper: fixed N per class
# -----------------------------
def make_fixed_per_class_df(df_tr: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """
    Makes training balanced with exactly `per_class` rows per label.
    If a class has < per_class, it upsamples with replacement (repeats rows).
    Deterministic given seed + df_tr.
    """
    rng = np.random.RandomState(seed)
    parts = []
    for lab, g in df_tr.groupby("label"):
        g = g.sample(frac=1.0, random_state=seed)  # shuffle deterministically
        if len(g) >= per_class:
            parts.append(g.iloc[:per_class].copy())
        else:
            extra = g.sample(n=per_class - len(g), replace=True, random_state=seed)
            parts.append(pd.concat([g, extra], ignore_index=True))

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # final shuffle
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=str(PROJECT_ROOT_DEFAULT))
    ap.add_argument("--data_root", type=str, default=str(DATA_ROOT_DEFAULT))
    ap.add_argument("--labels_csv", type=str, default=str(LABELS_CSV_DEFAULT))

    ap.add_argument("--arch", type=str, default="resnet50", choices=["resnet18", "resnet50", "resnet101"])
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ece_bins", type=int, default=15)

    # balancing
    ap.add_argument("--balance", type=str, default="class_weight",
                    choices=["none", "class_weight", "fixed_per_class", "sampler"])
    ap.add_argument("--per_class", type=int, default=5000)

    # scheduler
    ap.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step"])
    ap.add_argument("--step_size", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    ap.add_argument("--runs_dir", type=str, default="")  # if empty -> <project_root>/runs/baseline_kermany

    ap.add_argument("--early_stop", type=int, default=1)      # 1=on, 0=off
    ap.add_argument("--patience", type=int, default=8)        # common: 5-10
    ap.add_argument("--min_delta", type=float, default=1e-3)  # 0.001 = 0.1% acc


    args = ap.parse_args()

    set_seed(args.seed)

    project_root = Path(args.project_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    labels_csv = Path(args.labels_csv).expanduser().resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")
    print(f"[Info] project_root = {project_root}")
    print(f"[Info] data_root = {data_root}")
    print(f"[Info] labels_csv = {labels_csv}")

    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_csv}")

    df = pd.read_csv(labels_csv)
    df["dst_path"] = df["dst_path"].astype(str)

    # fixed label order
    class_names = sorted(df["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    num_classes = len(class_names)

    # Split from CSV
    df_train = df[df["split"] == "train"].copy()
    df_test = df[df["split"] == "test"].copy()

    # Patient-level val split from train
    df_tr, df_val = make_patient_val_split(df_train, val_frac=args.val_frac, seed=args.seed)

    # Apply balancing to TRAIN ONLY
    if args.balance == "fixed_per_class":
        df_tr = make_fixed_per_class_df(df_tr, per_class=args.per_class, seed=args.seed)

    print(f"[Info] classes ({num_classes}): {class_names}")
    print(f"[Info] train images: {len(df_tr)}, val images: {len(df_val)}, test images: {len(df_test)}")

    train_tfm, test_tfm = get_transforms(args.img_size)

    ds_tr = KermanyFromCSV(df_tr, class_to_idx, train_tfm)
    ds_val = KermanyFromCSV(df_val, class_to_idx, test_tfm)
    ds_te = KermanyFromCSV(df_test, class_to_idx, test_tfm)

    # Dataloader: support sampler mode
    train_sampler = None
    shuffle = True
    if args.balance == "sampler":
        # inverse-frequency sample weights (based on current df_tr)
        lab_counts = df_tr["label"].value_counts()
        sample_w = df_tr["label"].map(lambda x: 1.0 / float(lab_counts[x])).values
        sample_w = torch.tensor(sample_w, dtype=torch.double)

        train_sampler = WeightedRandomSampler(
            weights=sample_w,
            num_samples=len(sample_w),
            replacement=True,
        )
        shuffle = False

    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.arch, num_classes=num_classes, pretrained=bool(args.pretrained)).to(device)

    # Loss choice
    counts = df_tr["label"].value_counts().reindex(class_names).fillna(0).astype(float).values
    counts[counts == 0] = 1.0
    print("[Info] class counts (train):", dict(zip(class_names, counts.astype(int))))

    if args.balance == "class_weight":
        weights = (counts.sum() / (num_classes * counts))
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
        print("[Info] class weights:", weights_t.detach().cpu().tolist())
        criterion = nn.CrossEntropyLoss(weight=weights_t)
    else:
        print("[Info] using unweighted CrossEntropyLoss()")
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler (optional)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
        print("[Info] scheduler = CosineAnnealingLR")
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
        print("[Info] scheduler = StepLR")
    else:
        print("[Info] scheduler = none")

    # Run folder
    runs_dir = Path(args.runs_dir) if args.runs_dir else (project_root / "runs" / "baseline_kermany")
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_name = (
        f"kermany_{args.arch}_pre{args.pretrained}_bal{args.balance}"
        f"_bs{args.batch_size}_ep{args.epochs}_lr{args.lr}_seed{args.seed}"
    )
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    best_path = run_dir / "best.pt"
    history = []
    best_val = -1.0
    bad_epochs = 0


    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, dl_tr, optimizer, criterion, device)
        val_metrics = evaluate(model, dl_val, device, num_classes=num_classes, ece_bins=args.ece_bins)
        dt = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": float(loss),
            "lr": float(cur_lr),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "sec": dt,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"lr={cur_lr:.6g} | "
            f"loss={loss:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_auc={val_metrics['macro_auc_ovr']:.4f} | "
            f"val_ece={val_metrics['ece']:.4f} | "
            f"val_brier={val_metrics['brier']:.4f} | "
            f"val_entN={val_metrics['entropy_norm_mean']:.4f} | "
            f"{dt:.1f}s"
        )

        val_acc = float(val_metrics["acc"])

        if val_acc > best_val_acc  + args.min_delta:
            best_val_acc  = val_acc
            bad_epochs = 0
            torch.save(
        {
            "model_state": model.state_dict(),
            "classes": class_names,
            "class_to_idx": class_to_idx,
            "args": vars(args),
            "best_val_acc": best_val_acc,
            "best_epoch": epoch,
        },
        best_path,
    )
        else:
            bad_epochs += 1

        if args.early_stop and bad_epochs >= args.patience:
            print(f"[EarlyStopping] stop at epoch {epoch} | best_val_acc={best_val_acc:.4f}")
            break


    # Test with best checkpoint
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    test_metrics, test_details = evaluate(
        model, dl_te, device, num_classes=num_classes, ece_bins=args.ece_bins, return_details=True
    )

    print("\n[Best checkpoint]")
    print(f"  path: {best_path}")
    print(f"  best_val_acc: {best_val_acc:.4f}")
    print("[Test metrics]")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")

    # Save artifacts
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    df_u = pd.DataFrame(
        {
            "path": test_details["paths"],
            "y_true": test_details["labels"],
            "y_pred": test_details["preds"],
            "max_prob": test_details["max_prob"],
            "entropy": test_details["entropy"],
            "entropy_norm": test_details["entropy_norm"],
            "correct": (test_details["preds"] == test_details["labels"]).astype(int),
        }
    )
    df_u.to_csv(run_dir / "test_uncertainty.csv", index=False)

    print(f"\n[Done] Saved to: {run_dir}")
    print(f"[Done] Wrote: {run_dir / 'test_uncertainty.csv'}")


if __name__ == "__main__":
    main()

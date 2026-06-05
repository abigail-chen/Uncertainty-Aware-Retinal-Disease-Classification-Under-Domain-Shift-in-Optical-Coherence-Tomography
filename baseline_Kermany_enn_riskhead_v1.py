# src/baseline_Kermany_enn.py
# ------------------------------------------------------------
# Evidential Neural Network (ENN) on Kermany using fixed CSV splits.
#
# You MUST pass the split CSVs produced by your no-leak pipeline:
#   --train_csv splits/.../train_subset.csv   (5000 per class)
#   --val_csv   splits/.../val.csv            (patient-disjoint from train)
#   --test_csv  splits/.../test_official.csv  (official Kermany test)
#
# Outputs:
#   runs/enn_kermany/<run_name>/
#     best.pt
#     history.json
#     test_metrics.json
#     test_uncertainty.csv
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
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
from sklearn.metrics import accuracy_score, roc_auc_score


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
        label_str = str(row["label"])

        img = Image.open(img_path).convert("RGB")
        img = self.tfm(img)

        y = self.class_to_idx[label_str]
        return img, y, img_path


# -----------------------------
# Transforms
# -----------------------------
def get_transforms(img_size: int):
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
# Model: ENN head (evidence)
# -----------------------------
def build_backbone(arch: str, pretrained: bool) -> nn.Module:
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

    # remove final fc, keep features
    in_feats = m.fc.in_features
    m.fc = nn.Identity()
    return m, in_feats


class EvidentialNet(nn.Module):
    """
    Outputs evidence e >= 0 for each class.
    Dirichlet alpha = e + 1
    Predictive prob = alpha / sum(alpha)
    """

    def __init__(self, arch: str, num_classes: int, pretrained: bool):
        super().__init__()
        self.backbone, feat_dim = build_backbone(arch, pretrained)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        z = self.backbone(x)              # [B, feat_dim]
        logits = self.head(z)             # [B, K]
        evidence = F.softplus(logits)     # >= 0
        alpha = evidence + 1.0
        return alpha


# -----------------------------
# ENN loss (Dirichlet)
# -----------------------------
def _one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(y, num_classes=num_classes).float()


def edl_mse_loss(alpha: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    A common EDL loss: MSE between Dirichlet mean and one-hot + variance regularizer.
    mean = alpha / S
    var term encourages calibrated uncertainty.
    """
    S = torch.sum(alpha, dim=1, keepdim=True)             # [B,1]
    m = alpha / S                                         # [B,K]
    y_oh = _one_hot(y, num_classes)                        # [B,K]

    mse = torch.sum((y_oh - m) ** 2, dim=1)               # [B]
    var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1.0)), dim=1)  # [B]
    return torch.mean(mse + var)


def kl_dirichlet(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    KL( Dir(alpha) || Dir(1) ) : prior is uniform Dirichlet(1).
    """
    K = num_classes
    beta = torch.ones((1, K), device=alpha.device)

    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    sum_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB_alpha = torch.lgamma(sum_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_beta = torch.lgamma(sum_beta) - torch.sum(torch.lgamma(beta), dim=1, keepdim=True)

    digamma_sum = torch.digamma(sum_alpha)
    digamma_alpha = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (digamma_alpha - digamma_sum), dim=1, keepdim=True) + lnB_alpha - lnB_beta
    return torch.mean(kl)


def edl_loss(alpha: torch.Tensor, y: torch.Tensor, num_classes: int, lam: float) -> torch.Tensor:
    """
    EDL loss with 'evidence removal' trick:
    KL is applied to (alpha_tilde) where the true class evidence is removed,
    so correct predictions are encouraged to have high evidence,
    wrong predictions are pushed toward low evidence.
    """
    y_oh = _one_hot(y, num_classes)  # [B,K]

    # remove evidence on the true class for KL term
    alpha_tilde = (alpha - 1.0) * (1.0 - y_oh) + 1.0

    return edl_mse_loss(alpha, y, num_classes) + lam * kl_dirichlet(alpha_tilde, num_classes)

# -----------------------------
# Metrics + uncertainty
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


def normalized_entropy(ent: np.ndarray, num_classes: int) -> np.ndarray:
    return ent / math.log(num_classes)
@torch.no_grad()
def predict_probs_and_strength(model, loader, device, return_alpha: bool = False):
    model.eval()

    all_probs = []
    all_labels = []
    all_paths = []
    all_strength = []
    all_evidence_sum = []
    all_evidence_max = []
    all_alphas = []  # NEW

    for x, y, paths in loader:
        x = x.to(device, non_blocking=True)

        out = model(x)
        # Support both (logits, alpha) and alpha-only models
        if isinstance(out, (tuple, list)) and len(out) == 2:
            _logits, alpha = out
        else:
            alpha = out

        S = torch.sum(alpha, dim=1)                 # [B]
        probs = alpha / S.unsqueeze(1)              # [B, K]

        evidence = alpha - 1.0                      # [B, K]
        evidence_sum = torch.sum(evidence, 1)       # [B]
        evidence_max = torch.max(evidence, 1).values  # [B]

        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())
        all_paths.extend(list(paths))
        all_strength.append(S.cpu().numpy())
        all_evidence_sum.append(evidence_sum.cpu().numpy())
        all_evidence_max.append(evidence_max.cpu().numpy())

        if return_alpha:
            all_alphas.append(alpha.cpu().numpy())  # NEW

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    paths = np.array(all_paths, dtype=object)
    strength = np.concatenate(all_strength)
    evidence_sum = np.concatenate(all_evidence_sum)
    evidence_max = np.concatenate(all_evidence_max)

    if return_alpha:
        alpha_full = np.concatenate(all_alphas)
        return probs, labels, paths, strength, evidence_sum, evidence_max, alpha_full

    return probs, labels, paths, strength, evidence_sum, evidence_max


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, ece_bins: int, return_details: bool = False):
    if return_details:
        probs, labels, paths, strength, evidence_sum, evidence_max, alpha_full = \
            predict_probs_and_strength(model, loader, device, return_alpha=True)
    else:
        probs, labels, paths, strength, evidence_sum, evidence_max = \
            predict_probs_and_strength(model, loader, device, return_alpha=False)

    preds = probs.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    ece = expected_calibration_error(probs, labels, n_bins=ece_bins)
    brier = brier_score_multiclass(probs, labels, num_classes=num_classes)

    ent = predictive_entropy(probs)
    ent_n = normalized_entropy(ent, num_classes)
    correct = (preds == labels)

    # ENN uncertainty extras:
    # - strength S = sum(alpha): higher S -> more confident
    # - epistemic proxy: K / S (roughly, lower is more confident)
    K = num_classes
    inv_strength = K / np.clip(strength, 1e-8, None)

    out = {
        "acc": float(acc),
        "macro_auc_ovr": float(auc),
        "ece": float(ece),
        "brier": float(brier),
        "entropy_mean": float(ent.mean()),
        "entropy_norm_mean": float(ent_n.mean()),
        "entropy_mean_correct": float(ent[correct].mean()) if correct.any() else float("nan"),
        "entropy_mean_incorrect": float(ent[~correct].mean()) if (~correct).any() else float("nan"),
        "strength_mean": float(strength.mean()),
        "inv_strength_mean": float(inv_strength.mean()),
    }


    if return_details:
        details = {
        "paths": paths,
        "labels": labels,
        "preds": preds,
        "p_max": probs.max(axis=1),        # same value, clearer name
        "entropy": ent,
        "entropy_norm": ent_n,
        "strength": strength,
        "inv_strength": inv_strength,
        "u_mass": inv_strength,            # u_mass = K/S (same as inv_strength)
        "evidence_sum": evidence_sum,
        "evidence_max": evidence_max, 
        "probs": probs,
        "alpha_full": alpha_full, 
 }
        return out, details

    return out

def _enn_quantities_from_logits(logits_t: torch.Tensor):
    """
    logits_t: torch tensor (B, K) = ENN head output (before evidence)
    Returns: probs (B,K), evidence (B,K), alpha (B,K), strength S (B,), u_mass (B,), p_max (B,), y_pred (B,)
    """
    evidence = F.softplus(logits_t)      # (B,K) >= 0
    alpha = evidence + 1.0              # (B,K)
    S = alpha.sum(dim=1)                # (B,)
    probs = alpha / S.unsqueeze(1)      # (B,K) predictive mean probs
    K = logits_t.size(1)
    u_mass = K / S                      # (B,)
    p_max, y_pred = probs.max(dim=1)    # (B,), (B,)
    return probs, evidence, alpha, S, u_mass, p_max, y_pred
# -----------------------------
# Train loop
# -----------------------------
def train_one_epoch(model, loader, optimizer, device, num_classes: int, lam: float):
    model.train()
    total_loss, n = 0.0, 0
    for x, y, _paths in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        alpha = model(x)
        mse = edl_mse_loss(alpha, y, num_classes)
        y_oh = F.one_hot(y, num_classes=num_classes).float()
        alpha_tilde = (alpha - 1.0) * (1.0 - y_oh) + 1.0
        kl = kl_dirichlet(alpha_tilde, num_classes)
        loss = mse + lam * kl

        if n == 0:  # first batch of epoch
            print(f"[loss parts] mse={mse.item():.4f}  kl={kl.item():.4f}  lam*kl={(lam*kl).item():.4f}")

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)

    ap.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet50", "resnet101"])
    ap.add_argument("--pretrained", type=int, default=1)

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ece_bins", type=int, default=15)

    # ENN-specific: KL weight
    ap.add_argument("--lam", type=float, default=1.0)

    # early stopping
    ap.add_argument("--early_stop", type=int, default=1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min_delta", type=float, default=1e-3)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--runs_dir", type=str, default="runs/enn_kermany")
    ap.add_argument("--eval_only", type=int, default=0,
                help="1 = skip training and only evaluate using existing best.pt")

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    # Load CSVs
    tr = pd.read_csv(args.train_csv)
    va = pd.read_csv(args.val_csv)
    te = pd.read_csv(args.test_csv)

    # fixed label order from TRAIN CSV
    class_names = sorted(tr["label"].astype(str).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    num_classes = len(class_names)

    # Basic sanity
    print(f"[Info] classes ({num_classes}): {class_names}")
    print(f"[Info] train rows: {len(tr)} | val rows: {len(va)} | test rows: {len(te)}")

    # Dataloaders
    train_tfm, test_tfm = get_transforms(args.img_size)
    ds_tr = KermanyFromCSV(tr, class_to_idx, train_tfm)
    ds_va = KermanyFromCSV(va, class_to_idx, test_tfm)
    ds_te = KermanyFromCSV(te, class_to_idx, test_tfm)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    # Model
    model = EvidentialNet(args.arch, num_classes=num_classes, pretrained=bool(args.pretrained)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Run folder
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_name = (
        f"kermany_{args.arch}_pre{args.pretrained}"
        f"_enn_lam{args.lam}"
        f"_bs{args.batch_size}_ep{args.epochs}_lr{args.lr}_seed{args.seed}"
    )
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "best.pt"
    history = []
    best_val_acc = -1.0
    bad_epochs = 0


    # -----------------------------
# Eval-only mode (no training)
# -----------------------------
    if args.eval_only:
        if not best_path.exists():
            raise FileNotFoundError(f"Eval-only requested but checkpoint not found: {best_path}")

        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.to(device)

        test_metrics, details = evaluate(
              model, dl_te, device, num_classes=num_classes, ece_bins=args.ece_bins, return_details=True
    )

        print("\n[Eval-only] Loaded checkpoint")
        print(f"  path: {best_path}")
        print("[Test metrics]")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.6f}")

    # --- Save artifacts (same as below) ---
        with open(run_dir / "test_metrics_evalonly.json", "w") as f:
             json.dump(test_metrics, f, indent=2)

        df_u = pd.DataFrame(
        {
            "path": details["paths"],
            "y_true": details["labels"],
            "y_pred": details["preds"],
            "entropy": details["entropy"],
            "entropy_norm": details["entropy_norm"],
            "strength": details["strength"],
            "inv_strength": details["inv_strength"],
            "correct": (details["preds"] == details["labels"]).astype(int),
            "u_mass": details["u_mass"],
            "p_max": details["p_max"],
            "evidence_sum": details["evidence_sum"],
            "evidence_max": details["evidence_max"],
        }
    )

        P = np.asarray(details["probs"])
        A = np.asarray(details["alpha_full"])
        for c in range(P.shape[1]):
            df_u[f"p_{c}"] = P[:, c]
        for c in range(A.shape[1]):
            df_u[f"alpha_{c}"] = A[:, c]

        out_csv = run_dir / "test_uncertainty_with_probs.csv"
        df_u.to_csv(out_csv, index=False)

        print(f"\n[Done] Wrote: {out_csv}")
        return



    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # simple annealing: warm up KL over first 10 epochs
        lam_eff = args.lam * min(1.0, epoch / 10.0)
        train_loss = train_one_epoch(model, dl_tr, optimizer, device, num_classes=num_classes, lam=lam_eff)
        val_metrics = evaluate(model, dl_va, device, num_classes=num_classes, ece_bins=args.ece_bins)
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "sec": float(dt),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_auc={val_metrics['macro_auc_ovr']:.4f} | "
            f"val_ece={val_metrics['ece']:.4f} | "
            f"val_brier={val_metrics['brier']:.4f} | "
            f"val_strength={val_metrics['strength_mean']:.2f} | "
            f"{dt:.1f}s"
        )

        val_acc = float(val_metrics["acc"])
        if val_acc > best_val_acc + args.min_delta:
            best_val_acc = val_acc
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

    test_metrics, details = evaluate(
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
            "path": details["paths"],
            "y_true": details["labels"],
            "y_pred": details["preds"],
            "entropy": details["entropy"],
            "entropy_norm": details["entropy_norm"],
            "strength": details["strength"],
            "inv_strength": details["inv_strength"],
            "correct": (details["preds"] == details["labels"]).astype(int),
            "u_mass": details["u_mass"],
            "p_max": details["p_max"],
            "evidence_sum": details["evidence_sum"],
            "evidence_max": details["evidence_max"],
        }
    )
    # Add per-class probs + alpha
    P = np.asarray(details["probs"])          # [N, K]
    A = np.asarray(details["alpha_full"])     # [N, K]

    for c in range(P.shape[1]):
         df_u[f"p_{c}"] = P[:, c]
    for c in range(A.shape[1]):
        df_u[f"alpha_{c}"] = A[:, c]
    df_u.to_csv(run_dir / "test_uncertainty_with_probs.csv", index=False)

    print(f"\n[Done] Saved to: {run_dir}")
    print(f"[Done] Wrote: {run_dir / 'test_uncertainty.csv'}")


if __name__ == "__main__":
    main()

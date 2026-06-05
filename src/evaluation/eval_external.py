from __future__ import annotations
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision import transforms

# ----------------------------
# Utils
# ----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

KERMANY_CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_TO_IDX = {c:i for i,c in enumerate(KERMANY_CLASSES)}

def parse_label_map(pairs: list[str]) -> dict[str, str]:
    m = {}
    for s in pairs:
        if "=" not in s:
            raise ValueError(f"Bad --label_map entry '{s}'. Use like CNVM=CNV")
        a,b = s.split("=", 1)
        m[a.strip()] = b.strip()
    return m

def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def ece_score(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    # probs: (N,C), y_true: (N,)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = acc[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)

def brier_score_multiclass(probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> float:
    y_onehot = np.zeros((len(y_true), n_classes), dtype=np.float32)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

def macro_ovr_auc(probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> float | None:
    # Uses sklearn if available; otherwise returns None
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None
    y_onehot = np.zeros((len(y_true), n_classes), dtype=np.int32)
    y_onehot[np.arange(len(y_true)), y_true] = 1
    try:
        return float(roc_auc_score(y_onehot, probs, average="macro", multi_class="ovr"))
    except Exception:
        return None

def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        m = torchvision.models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "resnet50":
        m = torchvision.models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"Unsupported arch: {arch}")

def load_checkpoint_into_model(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device)

    # handle different save formats
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "model_state", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    # strip "module." if saved from DataParallel
    if isinstance(ckpt, dict):
        new = {}
        for k,v in ckpt.items():
            if k.startswith("module."):
                new[k[len("module."):]] = v
            else:
                new[k] = v
        ckpt = new

    model.load_state_dict(ckpt, strict=True)

# ----------------------------
# Dataset
# ----------------------------
class LabelsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tfm):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["dst_path"]
        y = int(row["y"])
        img = Image.open(path).convert("RGB")
        img = self.tfm(img)
        return img, y

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--use_split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--ece_bins", type=int, default=15)
    ap.add_argument("--label_map", nargs="*", default=["CNVM=CNV"])  # IMPORTANT: no PCV mapping
    ap.add_argument("--out_json", default="runs/eval_external_results.json")
    parser.add_argument("--out_csv", default=None, help="Optional: save per-image predictions to CSV")

    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    labels_csv = Path(args.labels_csv).expanduser().resolve()
    ckpt_path  = Path(args.ckpt).expanduser().resolve()
    out_json   = Path(args.out_json).expanduser().resolve()

    if not labels_csv.exists():
        raise FileNotFoundError(f"labels_csv not found: {labels_csv}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    df = pd.read_csv(labels_csv)

    # accept either dst_path already absolute, or build from data_root + relpath/dst_path
    if "dst_path" not in df.columns:
        if "relpath" in df.columns:
            df["dst_path"] = df["relpath"].astype(str).apply(lambda x: str(data_root / x))
        else:
            raise SystemExit("labels.csv missing dst_path (or relpath).")

    # filter split
    if args.use_split != "all":
        if "split" not in df.columns:
            raise SystemExit("labels.csv missing 'split' column but --use_split != all.")
        df = df[df["split"] == args.use_split].copy()

    # map labels (CNVM->CNV), DO NOT FORCE PCV
    label_map = parse_label_map(args.label_map)
    df["label_mapped"] = df["label"].astype(str).map(lambda x: label_map.get(x, x))

    # keep only overlapping classes
    before = len(df)
    df = df[df["label_mapped"].isin(KERMANY_CLASSES)].copy()
    dropped = before - len(df)

    # build y
    df["y"] = df["label_mapped"].map(CLASS_TO_IDX).astype(int)

    # verify file exists (sample a few)
    missing = df[~df["dst_path"].astype(str).map(lambda p: Path(p).exists())]
    if len(missing) > 0:
        ex = missing.iloc[0]["dst_path"]
        raise FileNotFoundError(f"Some dst_path files are missing. Example:\n  {ex}")

    print(f"[Info] External data_root: {data_root}")
    print(f"[Info] labels_csv: {labels_csv}")
    print(f"[Info] ckpt: {ckpt_path}")
    print(f"[Info] Using split: {args.use_split}")
    print(f"[Info] Label map: {label_map} (PCV is NOT mapped; will be dropped if present)")
    print(f"[Info] Kept {len(df)} rows; dropped {dropped} non-overlap rows.")
    print("[Info] Class counts (mapped):")
    print(df["label_mapped"].value_counts().to_string())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    ds = LabelsDataset(df, tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = build_model(args.arch, num_classes=len(KERMANY_CLASSES))
    load_checkpoint_into_model(model, ckpt_path, device)
    model.to(device)
    model.eval()

    all_logits = []
    all_y = []

    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            logits = model(x).detach().cpu().numpy()
            all_logits.append(logits)
            all_y.append(y.numpy())

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)

    probs = softmax_np(logits)
    y_pred = probs.argmax(axis=1)

    acc = float((y_pred == y_true).mean())
    ece = ece_score(probs, y_true, n_bins=args.ece_bins)
    brier = brier_score_multiclass(probs, y_true, n_classes=len(KERMANY_CLASSES))
    auc = macro_ovr_auc(probs, y_true, n_classes=len(KERMANY_CLASSES))

    results = {
        "ckpt": str(ckpt_path),
        "data_root": str(data_root),
        "labels_csv": str(labels_csv),
        "use_split": args.use_split,
        "kept_rows": int(len(df)),
        "dropped_nonoverlap_rows": int(dropped),
        "class_counts": df["label_mapped"].value_counts().to_dict(),
        "metrics": {
            "acc": acc,
            "macro_auc_ovr": auc,
            "ece": ece,
            "brier": brier,
        }
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print("\n[Test metrics on external dataset]")
    print(f"acc: {acc:.6f}")
    if auc is None:
        print("macro_auc_ovr: (skipped; sklearn not available)")
    else:
        print(f"macro_auc_ovr: {auc:.6f}")
    print(f"ece: {ece:.6f}")
    print(f"brier: {brier:.6f}")
    print(f"\n[Saved] {out_json}")

if __name__ == "__main__":
    main()

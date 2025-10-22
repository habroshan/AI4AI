#!/usr/bin/env python3
import argparse, os, torch, numpy as np, h5py
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# MNIST model from your repo so keys match the MNIST checkpoint
try:
    from tm_trainer import SimpleCNN as MNISTModel, flip_labels as mnist_flip_labels
except Exception:
    MNISTModel = None
    mnist_flip_labels = None

"""
BMM feature extractor with per-dataset target models and robust ChestXray14 support.

- --dataset {mnist,cifar10,chestxray14}
- --mode {clean,poison}                  (clean excludes flipped indices; poison = ONLY flipped indices)
- --feature-mode {bb,gb,bbgb}            (BB=outputs-only; GB=penultimate; BB+GB=concat)
Writes HDF5 with dataset 'X' and attrs {'feature_mode','subset','dataset'}.
"""

# ---------------- Utilities ----------------
def _find_last_linear(model: nn.Module) -> nn.Linear:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("No nn.Linear layer found for penultimate hook.")
    return last

class _PenultimateHook:
    def __init__(self, model: nn.Module):
        self.model = model
        self.buffer = None
        self.handle = None
    def __enter__(self):
        layer = _find_last_linear(self.model)
        def hook_fn(mod, inp, outp):
            self.buffer = inp[0].detach().cpu()   # (B, D) feeding final linear
        self.handle = layer.register_forward_hook(hook_fn)
        return self
    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
    def get_np(self) -> np.ndarray:
        if self.buffer is None:
            raise RuntimeError("Penultimate buffer is empty — run a forward pass first.")
        return self.buffer.numpy().astype("float32")
    def clear(self): self.buffer = None

# -------------- ChestXray14 dataset --------------
class ChestXrayDataset(Dataset):
    """
    Binary subset (No Finding=0, Pneumonia=1).
    CSV accepted:
      - <root>/Data_Entry_2017_v2020.csv  OR
      - <root>/Data_Entry_2017.csv
    Images: anywhere under <root>/ (we recursively scan all subfolders and map filenames)
    """
    def __init__(self, root: str, transform, mode='train', split_ratio=0.8):
        import pandas as pd

        # CSV
        cand_csv = [
            os.path.join(root, "Data_Entry_2017_v2020.csv"),
            os.path.join(root, "Data_Entry_2017.csv"),
        ]
        csv_path = next((p for p in cand_csv if os.path.isfile(p)), None)
        if csv_path is None:
            raise FileNotFoundError(
                "ChestXray14 CSV not found. Looked for:\n  " + "\n  ".join(cand_csv)
            )
        df = pd.read_csv(csv_path)
        if "Finding Labels" not in df.columns or "Image Index" not in df.columns:
            raise RuntimeError(f"Unexpected CSV schema in {csv_path}")

        # Two-class subset
        df = df[df["Finding Labels"].isin(["No Finding", "Pneumonia"])].reset_index(drop=True)
        df["label"] = df["Finding Labels"].map({"No Finding": 0, "Pneumonia": 1})

        # Train/test split
        cut = int(len(df) * split_ratio)
        self.df = df.iloc[:cut].reset_index(drop=True) if mode == 'train' else df.iloc[cut:].reset_index(drop=True)

        # Build mapping only for the filenames we actually need
        needed = set(self.df["Image Index"].astype(str).tolist())
        # also accept alternative extension swap (.png <-> .jpg) if present
        needed_alt = set()
        for name in list(needed):
            stem, ext = os.path.splitext(name)
            if ext.lower() == ".png":
                needed_alt.add(stem + ".jpg")
            elif ext.lower() == ".jpg":
                needed_alt.add(stem + ".png")
        needed_all = needed | needed_alt

        # Recursively scan <root> once and map basenames to full paths
        exts = {".png", ".jpg", ".jpeg"}
        path_map = {}
        lower_needed = {n.lower() for n in needed_all}
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                lf = fname.lower()
                if not any(lf.endswith(e) for e in exts):
                    continue
                if lf in lower_needed:
                    path_map[fname] = os.path.join(dirpath, fname)

        # Keep only rows whose images we found (with alt-ext fallback and case-insensitive match)
        keep_rows = []
        miss = 0
        for _, row in self.df.iterrows():
            name = str(row["Image Index"])
            fp = path_map.get(name)
            if fp is None:
                # try alt extension
                stem, ext = os.path.splitext(name)
                alt = stem + (".jpg" if ext.lower() == ".png" else ".png")
                fp = path_map.get(alt)
            if fp is None:
                # try case-insensitive lookup
                target_lower = name.lower()
                candidates = [k for k in path_map.keys() if k.lower() == target_lower]
                if candidates:
                    fp = path_map[candidates[0]]
                else:
                    miss += 1
                    continue
            row = row.copy()
            row["__path"] = fp
            keep_rows.append(row)

        if miss:
            print(f"[ChestXray14] Warning: {miss} listed images not found under {root}; skipped.")

        import pandas as pd
        self.df = pd.DataFrame(keep_rows).reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError(
                f"After scanning {root}, none of the required ChestXray14 images were found. "
                "Please verify your folder layout."
            )

        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image
        row = self.df.iloc[idx]
        fp = row["__path"]
        img = Image.open(fp).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, int(row['label'])

# -------------- Model builders --------------
def _safe_load_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # torch>=2.4
    except TypeError:
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        return sd

def build_model(dataset: str, chkpt_path: str) -> nn.Module:
    if not os.path.isfile(chkpt_path):
        raise FileNotFoundError(f"Model file not found: {chkpt_path}")
    sd = _safe_load_state(chkpt_path)

    if dataset == "mnist":
        if MNISTModel is None:
            raise RuntimeError("tm_trainer.SimpleCNN not found; place tm_trainer.py next to this script.")
        model = MNISTModel()
        model.load_state_dict(sd, strict=True)
        return model

    if dataset == "cifar10":
        model = models.resnet18(weights=None)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, 10)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[warn] CIFAR10 unexpected keys: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")
        if missing:
            print(f"[warn] CIFAR10 missing keys: {missing[:8]}{'...' if len(missing)>8 else ''}")
        return model

    if dataset == "chestxray14":
        model = models.resnet50(weights=None)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, 2)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[warn] ChestXray14 unexpected keys: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")
        if missing:
            print(f"[warn] ChestXray14 missing keys: {missing[:8]}{'...' if len(missing)>8 else ''}")
        return model

    raise ValueError(f"Unknown dataset {dataset}")

# -------------- Data builders --------------
def build_dataset(dataset: str, data_dir: str, split: str):
    if dataset == "mnist":
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return datasets.MNIST(data_dir, train=(split=="train"), download=True, transform=tfm)

    if dataset == "cifar10":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        from torchvision.datasets import CIFAR10
        return CIFAR10(data_dir, train=(split=="train"), download=True, transform=tfm)

    if dataset == "chestxray14":
        tfm = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        return ChestXrayDataset(data_dir, tfm, mode=split, split_ratio=0.8)

    raise ValueError(dataset)

def select_flip_indices(labels: np.ndarray, src: int, fraction: float, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    idx = np.where(labels == src)[0]
    k = int(len(idx) * max(0.0, min(1.0, fraction)))
    if k <= 0: return np.array([], dtype=np.int64)
    sel = rng.choice(idx, size=k, replace=False)
    return sel

# -------------- Feature builder --------------
class BMMExtractor:
    def __init__(self, model, device, feature_mode: str):
        self.model = model.to(device)
        self.device = device
        self.feature_mode = feature_mode  # 'bb'|'gb'|'bbgb'

    @staticmethod
    def _entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        p = np.clip(p, eps, 1 - eps)
        return -(p * np.log(p)).sum(axis=1, keepdims=True)

    @staticmethod
    def _top2_margin(p: np.ndarray) -> np.ndarray:
        s = -np.sort(-p, axis=1)
        return s[:, [0]] - s[:, [1]]

    def _bb_feats(self, logits: torch.Tensor) -> np.ndarray:
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        H = self._entropy(probs)
        M = self._top2_margin(probs)
        return np.concatenate([probs, H, M], axis=1).astype("float32")

    def _fallback_penultimate(self, x_batch: torch.Tensor) -> np.ndarray:
        if hasattr(self.model, "features"):
            with torch.no_grad():
                pen = self.model.features(x_batch)
                if isinstance(pen, (list, tuple)): pen = pen[0]
                pen = torch.nn.functional.adaptive_avg_pool2d(torch.relu(pen), (1,1)).view(x_batch.size(0), -1)
                return pen.detach().cpu().numpy().astype("float32")
        raise RuntimeError("Penultimate hook failed and no fallback available for this model.")

    def extract(self, loader: DataLoader, out_h5: str):
        self.model.eval()
        feats_all = []
        with _PenultimateHook(self.model) as hook, torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)  # forward triggers hook
                if self.feature_mode == "bb":
                    feats = self._bb_feats(logits)
                elif self.feature_mode == "gb":
                    try: feats = hook.get_np()
                    except RuntimeError: feats = self._fallback_penultimate(xb)
                else:  # "bbgb"
                    bb = self._bb_feats(logits)
                    try: pen = hook.get_np()
                    except RuntimeError: pen = self._fallback_penultimate(xb)
                    feats = np.concatenate([bb, pen], axis=1)
                feats_all.append(feats)
                hook.clear()
        X = np.vstack(feats_all)
        os.makedirs(os.path.dirname(out_h5), exist_ok=True)
        with h5py.File(out_h5, "w") as f:
            f.create_dataset("X", data=X, compression="gzip")
            f.attrs["feature_mode"] = self.feature_mode
        print(f"✅ Wrote features: {X.shape} -> {out_h5}")

# ------------------- CLI -------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract BMM features (BB/GB/BB+GB)")
    ap.add_argument("--dataset", choices=["mnist","cifar10","chestxray14"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-h5", required=True)
    ap.add_argument("--split", choices=["train","test"], default="train")
    ap.add_argument("--mode", choices=["clean","poison"], default="clean")
    ap.add_argument("--feature-mode", choices=["bb","gb","bbgb"], default="bbgb")
    ap.add_argument("--poison-fraction", type=float, default=0.0)
    ap.add_argument("--flip-src", type=int, default=0)  # MNIST default in runner is 1->7; Chest: 0->1
    ap.add_argument("--flip-dst", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", choices=["cpu","cuda"], default=None)
    args = ap.parse_args()

    # Build full dataset (then we subset to clean/poison)
    ds_full = build_dataset(args.dataset, args.data_dir, args.split)

    # Collect labels
    all_labels = []
    for i in range(len(ds_full)):
        _, y = ds_full[i]
        all_labels.append(int(y))
    all_labels = np.asarray(all_labels, dtype=np.int64)

    # Choose flip indices once (seed=0) and create clean/poison subsets
    flip_idx = select_flip_indices(all_labels, args.flip_src, args.poison_fraction, seed=0)
    keep_mask = np.ones(len(all_labels), dtype=bool)
    keep_mask[flip_idx] = False
    clean_indices  = np.where(keep_mask)[0]
    poison_indices = flip_idx

    class _Subset(Dataset):
        def __init__(self, base, indices):
            self.base = base; self.indices = np.asarray(indices, dtype=np.int64)
        def __len__(self): return len(self.indices)
        def __getitem__(self, i): return self.base[int(self.indices[i])]

    sub = _Subset(ds_full, clean_indices if args.mode=="clean" else poison_indices)

    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(args.device=="cuda"))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[extractor] dataset={args.dataset} split={args.split} mode={args.mode} device={device} | clean={len(clean_indices)} poison={len(poison_indices)}")

    model = build_model(args.dataset, args.model_path)
    extractor = BMMExtractor(model, device, feature_mode=args.feature_mode)
    extractor.extract(loader, args.output_h5)

    with h5py.File(args.output_h5, "a") as f:
        f.attrs["subset"] = args.mode
        f.attrs["dataset"] = args.dataset



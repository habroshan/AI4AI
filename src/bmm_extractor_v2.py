import argparse
import os
import torch
import numpy as np
import h5py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

"""
BMM feature extractor (v2)
- Keeps dataset subset control: --mode {clean, poison}
- Adds --feature-mode {bb, gb, bbgb}
- Optional: --autotrain-if-missing and --train-epochs (quickly trains a small model if --model-path is missing)
- Writes HDF5 with dataset 'X' and attrs: feature_mode, subset

This script does NOT write labels; the runner merges clean+poison into one H5 with y∈{0,1}.
"""

# -------- Safe fallbacks if project-local helpers are absent --------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.fc = nn.Linear(9216, 10)
    def forward(self, x):
        x = self.features(x)
        return self.fc(x)

def flip_labels_inplace(ds, frac, src, dst):
    import numpy as np
    if frac <= 0 or src is None or dst is None:
        return ds
    targets = np.array(ds.targets)
    idx = np.where(targets == src)[0]
    if len(idx) == 0:
        return ds
    m = int(len(idx) * frac)
    if m == 0:
        return ds
    sel = np.random.choice(idx, size=m, replace=False)
    targets[sel] = dst
    ds.targets = targets.tolist()
    return ds

# -------- Penultimate hook utilities --------

def find_last_linear(model: nn.Module) -> nn.Linear:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("Could not find a Linear layer to hook for penultimate features.")
    return last

class PenultimateHook:
    def __init__(self, model: nn.Module):
        self.model = model
        self.buffer = None
        self.handle = None
    def __enter__(self):
        layer = find_last_linear(self.model)
        def hook_fn(mod, inp, outp):
            self.buffer = inp[0].detach().cpu()  # activation feeding the final linear
        self.handle = layer.register_forward_hook(hook_fn)
        return self
    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
    def get(self) -> torch.Tensor:
        if self.buffer is None:
            raise RuntimeError("Penultimate buffer is empty — run a forward pass first.")
        return self.buffer
    def clear(self):
        self.buffer = None

# -------- Extractor --------
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
        return np.concatenate([probs, H, M], axis=1).astype('float32')
    def extract(self, loader: DataLoader, out_h5: str):
        self.model.eval()
        feats_all = []
        with PenultimateHook(self.model) as hook:
            with torch.no_grad():
                for xb, _ in loader:
                    xb = xb.to(self.device)
                    logits = self.model(xb)
                    if self.feature_mode == 'bb':
                        feats = self._bb_feats(logits)
                    elif self.feature_mode == 'gb':
                        pen = hook.get().numpy().astype('float32')
                        feats = pen
                    else:  # 'bbgb'
                        pen = hook.get().numpy().astype('float32')
                        feats = np.concatenate([self._bb_feats(logits), pen], axis=1)
                    feats_all.append(feats)
                    hook.clear()
        X = np.vstack(feats_all)
        os.makedirs(os.path.dirname(out_h5), exist_ok=True)
        with h5py.File(out_h5, 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.attrs['feature_mode'] = self.feature_mode
        print(f"✅ Saved features: {X.shape} -> {out_h5}")

# -------- Main --------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Extract BMM features (BB/GB/BB+GB) for MNIST-like data')
    ap.add_argument('--data-dir', default='./data')
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--output-h5', required=True)
    ap.add_argument('--split', choices=['train','test'], default='train')
    ap.add_argument('--mode', choices=['clean','poison'], default='clean', help='dataset subset')
    ap.add_argument('--feature-mode', choices=['bb','gb','bbgb'], default='bbgb')
    ap.add_argument('--poison-fraction', type=float, default=0.0)
    ap.add_argument('--flip-src', type=int, default=None)
    ap.add_argument('--flip-dst', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--device', choices=['cpu','cuda'], default=None)
    ap.add_argument('--autotrain-if-missing', action='store_true')
    ap.add_argument('--train-epochs', type=int, default=1)
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Load or auto-train model
    model = SimpleCNN()
    if os.path.isfile(args.model_path):
        try:
            state = torch.load(args.model_path, map_location='cpu')
        except TypeError:
            state = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state)
    elif args.autotrain_if_missing:
        print(f"[info] Model not found at {args.model_path}. Auto-training for {args.train_epochs} epoch(s)...")
        tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tr)
        loader_tr = DataLoader(train_ds, batch_size=256, shuffle=True)
        model.to(device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(args.train_epochs):
            for xb, yb in loader_tr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); logits = model(xb)
                loss = nn.CrossEntropyLoss()(logits, yb)
                loss.backward(); opt.step()
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        torch.save(model.state_dict(), args.model_path)
        model.to('cpu')
    else:
        raise FileNotFoundError(f"Model file not found: {args.model_path}. Provide a valid file or add --autotrain-if-missing.")

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    ds = datasets.MNIST(args.data_dir, train=(args.split=='train'), download=True, transform=transform)
    if args.mode == 'poison' and args.poison_fraction > 0:
        ds = flip_labels_inplace(ds, args.poison_fraction, args.flip_src, args.flip_dst)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Extract
    extractor = BMMExtractor(model, device, args.feature_mode)
    extractor.extract(loader, args.output_h5)

    # Tag subset attr
    with h5py.File(args.output_h5, 'a') as f:
        f.attrs['subset'] = args.mode

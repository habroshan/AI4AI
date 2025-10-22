#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import h5py
except ImportError:
    h5py = None

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "ablation_out"
PY = sys.executable

# EDIT THESE PATHS
DATA_CFG: Dict[str, Dict[str, str]] = {
    "mnist": {"data_dir": "./data/mnist", "model_path": "./models/mnist_cnn.pt"},
}

CMD_TEMPLATES = {
    "extract": (
        PY,
        str(ROOT / "bmm_extractor_v2.py"),
        "--data-dir", "{data_dir}",
        "--model-path", "{model_path}",
        "--output-h5", "{out}",
        "--split", "{split}",
        "--mode", "{dp_mode}",
        "--feature-mode", "{feature_mode}",
        "--batch-size", "{batch_size}",
        "--device", "{device}",
        "--poison-fraction", "{poison_fraction}",
        "--autotrain-if-missing",
        "--train-epochs", "{train_epochs}",
    ),
    "train": (
        PY,
        str(ROOT / "dm_trainer.py"),
        "--train_h5", "{train_h5}",
        "--test_h5", "{test_h5}",
        "--detector", "{det}",
        "--seed", "{seed}",
    ),
}

JSON_METRICS_REGEX = re.compile(r"METRICS\s+(\{.*\})")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd: Tuple[str, ...], cwd: Optional[Path] = None) -> Tuple[int,str,str]:
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    so, se = proc.communicate()
    return proc.returncode, so, se

def merge_h5(clean_h5: Path, poison_h5: Path, out_h5: Path, feature_mode: str) -> Path:
    if h5py is None:
        raise RuntimeError("h5py required: pip install h5py")
    with h5py.File(clean_h5, 'r') as fc, h5py.File(poison_h5, 'r') as fp:
        Xc = fc['X'][()]
        Xp = fp['X'][()]
    X = np.concatenate([Xc, Xp], axis=0)
    y = np.concatenate([np.zeros(len(Xc), dtype=np.int64), np.ones(len(Xp), dtype=np.int64)], axis=0)
    ensure_dir(out_h5.parent)
    with h5py.File(out_h5, 'w') as f:
        f.create_dataset('X', data=X, compression='gzip')
        f.create_dataset('y', data=y)
        f.attrs['feature_mode'] = feature_mode
    return out_h5

def parse_metrics(so: str, se: str) -> Optional[Dict[str,float]]:
    for blob in (so, se):
        m = JSON_METRICS_REGEX.search(blob)
        if m:
            try:
                d = json.loads(m.group(1))
                return {k: float(v) for k, v in d.items()}
            except Exception:
                pass
    return None

def extract_split(dataset: str, split: str, feature_mode: str, seed: int, out_dir: Path, *, device: str, batch_size: int, poison_fraction: float, train_epochs: int) -> Path:
    cfg = DATA_CFG.get(dataset)
    if not cfg:
        raise RuntimeError(f"Please configure DATA_CFG for dataset {dataset}")
    data_dir = cfg['data_dir']
    model_path = cfg['model_path']

    clean_h5 = out_dir / f"{split}_clean.h5"
    poison_h5 = out_dir / f"{split}_poison.h5"
    merged_h5 = out_dir / f"{split}.h5"

    for dp_mode, out_h5 in (("clean", clean_h5), ("poison", poison_h5)):
        if out_h5.exists():
            continue
        cmd = [tok.format(data_dir=data_dir, model_path=model_path, out=str(out_h5), split=split, dp_mode=dp_mode, feature_mode=feature_mode, batch_size=str(batch_size), device=device, poison_fraction=str(poison_fraction), train_epochs=str(train_epochs)) for tok in CMD_TEMPLATES['extract']]
        rc, so, se = run_cmd(tuple(cmd))
        if rc != 0:
            print('[extract ERROR] stdout:\n' + so)
            print('[extract ERROR] stderr:\n' + se)
            raise RuntimeError(f"Extractor failed: {dataset}/{split}/{dp_mode}/{feature_mode}")
    return merge_h5(clean_h5, poison_h5, merged_h5, feature_mode)

def train_and_eval(det: str, train_h5: Path, test_h5: Path, seed: int) -> Dict[str,float]:
    cmd = [tok.format(train_h5=str(train_h5), test_h5=str(test_h5), det=det, seed=str(seed)) for tok in CMD_TEMPLATES['train']]
    rc, so, se = run_cmd(tuple(cmd))
    if rc != 0:
        print('[train ERROR] stdout:\n' + so)
        print('[train ERROR] stderr:\n' + se)
        raise RuntimeError(f"Trainer failed for {det}")
    m = parse_metrics(so, se)
    if m is None:
        raise RuntimeError('Trainer did not emit METRICS JSON')
    return m

def ci95(vals: List[float]) -> Tuple[float,float]:
    arr = np.asarray(vals, float)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / max(len(arr),1)**0.5) if len(arr) > 1 else 0.0
    return mean, 1.96*se

def summarise(rows: List[Dict[str,float]], dataset: str, out_dir: Path) -> None:
    if pd is None:
        return
    df = pd.DataFrame(rows)
    agg_cols = ["AUC","AUCPR","Accuracy","Precision","Recall","F1"]
    def agg_ci(g):
        out = {}
        for c in agg_cols:
            mean, ci = ci95(g[c].tolist())
            out[c], out[c+"_CI95"] = mean, ci
        return pd.Series(out)
    sdf = df.groupby(["det","mode"], as_index=False).apply(agg_ci)
    rows2 = []
    for det in sorted(sdf.det.unique()):
        base = sdf[(sdf.det==det) & (sdf.mode=='bb')]
        for mode in ['bb','gb','bbgb']:
            cur = sdf[(sdf.det==det) & (sdf.mode==mode)]
            if cur.empty: continue
            rec = {"det": det, "mode": mode}
            for c in agg_cols:
                rec[c] = float(cur[c].values[0]); rec[c+"_CI95"] = float(cur[c+"_CI95"].values[0])
            if not base.empty:
                rec["ΔAUC_vs_BB"] = rec["AUC"] - float(base["AUC"].values[0])
                rec["ΔAUCPR_vs_BB"] = rec["AUCPR"] - float(base["AUCPR"].values[0])
            rows2.append(rec)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows2).to_csv(out_dir / f"table_{dataset}_ablation.csv", index=False)
    df.to_csv(out_dir / f"raw_runs_{dataset}.csv", index=False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['mnist'])
    ap.add_argument('--modes', nargs='+', default=['bb','gb','bbgb'])
    ap.add_argument('--detectors', nargs='+', default=['xgb','mlp','if','maha'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--poison-fraction', type=float, default=0.1)
    ap.add_argument('--train-epochs', type=int, default=1)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str,float]] = []

    for dataset in args.datasets:
        print(f"=== DATASET: {dataset} ===")
        ds_root = OUT_ROOT / dataset
        ds_root.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            for mode in args.modes:
                run_root = ds_root / f'seed_{seed}' / mode
                (run_root / 'train').mkdir(parents=True, exist_ok=True)
                (run_root / 'test').mkdir(parents=True, exist_ok=True)
                train_h5 = extract_split(dataset, 'train', mode, seed, run_root / 'train', device=args.device, batch_size=args.batch_size, poison_fraction=args.poison_fraction, train_epochs=args.train_epochs)
                test_h5  = extract_split(dataset, 'test',  mode, seed, run_root / 'test',  device=args.device, batch_size=args.batch_size, poison_fraction=args.poison_fraction, train_epochs=args.train_epochs)
                for det in args.detectors:
                    m = train_and_eval(det, train_h5, test_h5, seed)
                    rec = {"dataset": dataset, "seed": seed, "mode": mode, "det": det, **m}
                    all_rows.append(rec)
        if pd is not None and all_rows:
            summarise([r for r in all_rows if r['dataset']==dataset], dataset, OUT_ROOT)
    if pd is not None and all_rows:
        pd.DataFrame(all_rows).to_csv(OUT_ROOT / 'raw_runs_ALL.csv', index=False)
    print("All done. See ./ablation_out")

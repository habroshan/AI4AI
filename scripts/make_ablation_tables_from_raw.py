#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("ablation_out")
DATASETS = ["mnist", "cifar10", "chestxray14"]
METRIC_LIST = ["AUC","AUCPR","Accuracy","Precision","Recall","F1"]

def ci95(vals):
    arr = np.asarray(vals, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()) if arr.size else float("nan"), 0.0
    se = arr.std(ddof=1) / np.sqrt(arr.size)
    return float(arr.mean()), float(1.96 * se)

def find_col(df, *candidates):
    lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    return None

def normalize_modes(s):
    # Clean up whitespace/case; map any synonyms if needed
    m = (str(s).strip().lower()
         .replace("bb+gb", "bbgb")
         .replace("bb-gb", "bbgb")
         .replace("bb + gb", "bbgb"))
    return m

def load_and_normalize(path):
    df = pd.read_csv(path)
    # Column name normalization
    det_col  = find_col(df, "det", "detector", "clf", "model")
    mode_col = find_col(df, "mode", "feature_mode", "features")
    seed_col = find_col(df, "seed", "run", "trial")

    if det_col is None or mode_col is None:
        raise ValueError(f"{path.name}: missing detector/mode columns. Found: {df.columns.tolist()}")

    # Ensure required metric columns exist
    present_metrics = []
    metric_map = {}
    for m in METRIC_LIST:
        col = find_col(df, m)
        if col is None:
            # allow lowercase alternatives, but skip if truly absent
            continue
        present_metrics.append(m)
        metric_map[m] = col
    if not present_metrics:
        raise ValueError(f"{path.name}: no metrics found among {METRIC_LIST}")

    # Build a clean frame with canonical names
    out = pd.DataFrame({
        "det":  df[det_col].astype(str),
        "mode": df[mode_col].map(normalize_modes),
    })
    if seed_col is not None:
        out["seed"] = df[seed_col]
    for m in present_metrics:
        out[m] = df[metric_map[m]]

    # Keep only known modes (bb, gb, bbgb) if present
    out = out[out["mode"].isin(["bb","gb","bbgb"])]
    return out, present_metrics

def build_ablation(df, dsname, metrics):
    # Aggregate mean ± 95% CI per detector × mode
    rows = []
    for (det, mode), g in df.groupby(["det","mode"]):
        rec = {"det": det, "mode": mode}
        for m in metrics:
            mean, ci = ci95(g[m].values)
            rec[m] = mean
            rec[m+"_CI95"] = ci
        rows.append(rec)
    if not rows:
        return None

    tab = pd.DataFrame(rows)

    # To compute deltas vs BB (difference of means)
    tab["ΔAUC_vs_BB"] = np.nan
    tab["ΔAUCPR_vs_BB"] = np.nan
    for det in tab["det"].unique():
        base = tab[(tab["det"]==det) & (tab["mode"]=="bb")]
        if base.empty:
            continue
        base_auc   = float(base["AUC"].iloc[0])   if "AUC" in tab.columns else np.nan
        base_aupr  = float(base["AUCPR"].iloc[0]) if "AUCPR" in tab.columns else np.nan
        for mode in ["gb","bbgb"]:
            idx = (tab["det"]==det) & (tab["mode"]==mode)
            if not idx.any():
                continue
            if "AUC" in tab.columns:
                tab.loc[idx, "ΔAUC_vs_BB"] = tab.loc[idx, "AUC"] - base_auc
            if "AUCPR" in tab.columns:
                tab.loc[idx, "ΔAUCPR_vs_BB"] = tab.loc[idx, "AUCPR"] - base_aupr

    # Pretty columns with mean±CI strings
    pretty = tab.copy()
    for m in metrics:
        pretty[m] = tab.apply(lambda r: f"{r[m]:.3f}$\\pm${r[m+'_CI95']:.3f}", axis=1)

    def fmt_delta(x):
        return "--" if pd.isna(x) else f"{x:+.3f}"

    pretty["ΔAUC_vs_BB"]   = pretty["ΔAUC_vs_BB"].map(fmt_delta)
    pretty["ΔAUCPR_vs_BB"] = pretty["ΔAUCPR_vs_BB"].map(fmt_delta)

    # Reorder columns
    cols = ["det","mode"] + [m for m in METRIC_LIST if m in metrics] + ["ΔAUC_vs_BB","ΔAUCPR_vs_BB"]
    pretty = pretty[cols]

    # Sort for stable display
    pretty = (pretty
              .sort_values(by=["det","mode"],
                           key=lambda s: s.map({"bb":0,"gb":1,"bbgb":2}).fillna(99) if s.name=="mode" else s))

    # Save
    out_csv = BASE / f"table_{dsname}_ablation.csv"
    pretty.to_csv(out_csv, index=False)

    # Also emit LaTeX table rows
    def row_to_tex(r):
        parts = [str(r["det"]).upper(), str(r["mode"]).lower()]
        for m in [m for m in METRIC_LIST if m in metrics]:
            parts.append(r[m])
        parts += [r["ΔAUC_vs_BB"], r["ΔAUCPR_vs_BB"]]
        return " & ".join(parts) + r" \\"

    tex_lines = [row_to_tex(r) for _, r in pretty.iterrows()]
    (BASE / f"{dsname}_ablation_table_rows.tex").write_text("\n".join(tex_lines))

    print(f"[ok] {dsname}: wrote {out_csv} and {dsname}_ablation_table_rows.tex")
    return pretty

def main():
    for ds in DATASETS:
        p = BASE / f"raw_runs_{ds}.csv"
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        try:
            df, metrics = load_and_normalize(p)
            if df.empty:
                print(f"[warn] {ds}: no usable rows after normalisation")
                continue
            build_ablation(df, ds, metrics)
        except Exception as e:
            print(f"[error] {ds}: {e}")

if __name__ == "__main__":
    main()

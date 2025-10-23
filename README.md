# AI to Protect AI: A Modular Pipeline for Detecting Label‑Flipping Poisoning

This repo contains the code used for the paper **“AI to Protect AI: A Modular Pipeline for Detecting Label‑Flipping Poisoning Attacks.”** It implements a two‑stage, modular defence:

* **Behaviour Monitoring Module (BMM):** trains a standard classifier (the “target model”) and extracts behaviour features from it (probabilities, entropy, top‑2 margin, and optionally penultimate activations).
* **Detection Module (DM):** trains a separate detector (XGBoost / MLP or unsupervised IF / Mahalanobis) on those features to flag poisoned vs. clean samples.

The pipeline is dataset‑ and model‑agnostic. We evaluate on **MNIST**, **CIFAR‑10**, and **ChestXray14** with 10% and 20% label‑flip rates.

> If you are reading the journal paper: you can reproduce the main results with the commands in **Reproduce the paper** below.

---

## Repo structure

```
AI-to-Protect-AI/
├── README.md                      # this file
├── LICENSE                        # license (MIT)
├── env/                           # environment files
│   ├── requirements.txt           # pip env (CPU)
│   └── environment.yml            # conda env (GPU optional. We ran the solution on both GPU+CPU and CPU-only machines; however, we recommend using GPUs for faster results.)
├── src/
│   ├── bmm_extractor.py           # BMM (MNIST/CIFAR + ChestXray14 support)
│   ├── bmm_extractor_v2.py        # variant used for ablations (feature modes)
│   ├── tm_trainer.py              # MNIST target model (SimpleCNN)
│   ├── tm_trainer_cifar.py        # CIFAR-10 target model
│   ├── run_chestxray14_pipeline.py# ChestXray14 end‑to‑end pipeline (ResNet18 TM)
│   ├── data_loaders.py            # dataset loaders / transforms
│   ├── dm_trainer.py              # Detection Module training & metrics
│   ├── detectors.py               # XGB/MLP/IF/Mahalanobis helpers
│   ├── metrics_utils.py           # ROC/PR, AUCPR, confusion matrices
│   ├── plots.py                   # plotting helpers
│   └── diagnose_embeddings.py     # debugging utilities
├── scripts/                       # thin CLI wrappers to run pipelines
│   ├── run_all.py                 # quick MNIST demo (train TM, extract, detect)
│   ├── run_evaluation.py          # paper‑style eval for MNIST/CIFAR
│   ├── run_ablation.py            # access‑mode ablations BB/GB/BB+GB
│   ├── run_ablation_v2.py         # as above, refactored
│   ├── run_all_pipelines.py       # batch MNIST/CIFAR jobs
│   ├── run_all_pipelines_extended.py
│   └── make_ablation_tables_from_raw.py
├── configs/                       # optional JSON/YAML configs (create as needed)
├── data/                          # datasets (not uploaded to the repository, but we have included guidelines)
│   ├── mnist/                     # auto‑downloaded by torchvision
│   ├── cifar10/                   # auto‑downloaded by torchvision
│   └── chestxray14/               # NIH images + Data_Entry_2017.csv (it's a large dataset so please use the instruction to get and use the dataset)
├── models/                        # trained target models (git‑ignored)
├── features/                      # extracted BMM features (HDF5, git‑ignored)
├── detectors/                     # trained DM artefacts (.joblib, git‑ignored)
├── results/                       # metrics JSON/CSVs, confusion matrices, PR/ROC
└── figures/                       # plots (which we have included in the paper)
```

**Files present in the upload and how they map:**

* `run_all.py`, `run_evaluation.py`, `run_ablation*.py`, `run_all_pipelines*.py`, `tm_trainer.py`, `tm_trainer_cifar.py`, `bmm_extractor*.py`, `dm_trainer.py`, `detectors.py`, `data_loaders.py`, `plots.py`, `metrics_utils.py`, `run_chestxray14_pipeline.py`, `diagnose_embeddings.py`, `make_ablation_tables_from_raw.py` → moved under `src/` and `scripts/` as above.
* Convenience / exploration scripts like `replot_aggregate_curves*.py`, `test.py` can stay under `scripts/` (optional for reproduction).
* `.DS_Store` and `__MACOSX` were created by macOS and should **not** be committed.

## Environment

Tested with Python 3.10 (Linux), PyTorch 2.x.

### Conda (recommended)

```
conda env create -f env/environment.yml
conda activate a2pa
```

### Pip

```
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt
```

**Dependencies (summary):**

* torch, torchvision, torchaudio (CPU or CUDA build)
* numpy, scipy, scikit‑learn, xgboost, joblib
* h5py, pandas, matplotlib, pillow

## Data

* **MNIST** and **CIFAR‑10** will be downloaded automatically by `torchvision` into `data/`.
* **ChestXray14** must be supplied by the user:

  * Place the NIH images under `data/chestxray14/images/` (subfolders allowed) and `Data_Entry_2017.csv` at `data/chestxray14/Data_Entry_2017.csv`.
  * The default pipeline uses the binary task *pneumonia vs. other*.

> Tip: Fast local SSD is strongly recommended for ChestXray14.

## Reproduce the paper

Below are minimal commands to reproduce the three datasets and the access‑mode ablations reported in the paper. The defaults implement random class‑pair flips; you can control the poisoning rate and flip pair via flags.

### 1) MNIST (10% label‑flip 1→7)

Train target model (TM), extract BMM features, and train DM (XGBoost):

```
python scripts/run_all.py --dataset mnist --poison-frac 0.10 --flip-src 1 --flip-dst 7 --feature-mode bb --outdir results/mnist_10p_bb
```

Ablate GB and BB+GB feature modes:

```
python scripts/run_ablation.py --dataset mnist --poison-frac 0.10 --feature-mode gb
python scripts/run_ablation.py --dataset mnist --poison-frac 0.10 --feature-mode bbgb
```

Expected: ROC–AUC ≳ 0.95 for supervised DMs; AUCPR increases slightly with BB+GB.

### 2) CIFAR‑10 (10% label‑flip 0→1)

```
python scripts/run_all.py --dataset cifar10 --poison-frac 0.10 --flip-src 0 --flip-dst 1 --feature-mode bb --outdir results/cifar10_10p_bb
```

Then repeat for `--feature-mode gb` and `bbgb` as above.

### 3) ChestXray14 (10% pneumonia→other)

This script wraps the full pipeline (ResNet18 TM → BMM → DM) and expects the data to be present as above:

```
python src/run_chestxray14_pipeline.py --data-root data/chestxray14 --poison-frac 0.10 --feature-mode bbgb --outdir results/chx14_10p_bbgb
```

### 20% poisoning

Increase `--poison-frac` to `0.20` in the commands above.

### Determinism

We report mean ±95% CI over 5 seeds in the paper. To match that:

```
for s in 0 1 2 3 4; do \
  python scripts/run_all.py --dataset mnist --poison-frac 0.10 --seed $s --outdir results/mnist_10p_bb/seed_$s; \
  done
```

Aggregate with `scripts/make_ablation_tables_from_raw.py`.

## What each stage does (short version)

1. **Train TM (target model)** per dataset.
2. **Extract BMM features**:

   * **BB:** posterior probabilities, Shannon entropy, top‑2 margin.
   * **GB:** penultimate activations.
   * **BB+GB:** concatenation.
3. **Train DM** on those features:

   * **Supervised:** XGBoost (default) or MLP.
   * **Unsupervised:** Isolation Forest or Mahalanobis.
4. **Evaluate** with ROC–AUC and AUCPR (plus Accuracy/Precision/Recall/F1 at a chosen threshold); save plots and JSON.

## Results at a glance (sanity checks)

* **MNIST:** supervised DMs typically AUC ≳ 0.95 at 10–20% flip; unsupervised lower but usable.
* **CIFAR‑10:** supervised DMs remain strong (AUC around low–mid 0.90s); unsupervised degrade.
* **ChestXray14:** hardest; AUC ~0.85 for supervised, unsupervised struggle. Prefer BB+GB here.

(Exact numbers and discussion are in the paper; see the **Discussion/Limitations** there.)

## Command reference (most common flags)

```
--dataset {mnist,cifar10,chestxray14}
--poison-frac FLOAT            # 0.0–1.0
--flip-src INT --flip-dst INT  # class ids (MNIST/CIFAR‑10)
--feature-mode {bb,gb,bbgb}
--batch-size INT --epochs INT
--seed INT
--outdir PATH
```

ChestXray14 extras:

```
--data-root PATH               # folder containing images/ and Data_Entry_2017.csv
--arch {resnet18,resnet34}     # TM backbone
```

## What to commit vs. ignore

**Commit**

* `src/`, `scripts/`, `env/`, `configs/`, `README.md`, `LICENSE`.

**Do NOT commit**

* `data/` (datasets), `models/` (TM checkpoints), `features/` (HDF5), `detectors/` (joblib files), `results/` heavy artefacts, and any `__MACOSX`/`.DS_Store`.

Suggested `.gitignore` entries:

```
# datasets & artefacts
/data/
/models/
/features/
/detectors/
/results/
/figures/*.png

# OS/editor
.DS_Store
__MACOSX/
*.swp
.venv/
```

## What was *not* used for the final figures

The codebase includes a few convenience scripts kept from development:

* `replot_aggregate_curves*.py` and `test.py` were used during figure iteration / debugging; they are safe to keep under `scripts/` but are **not required** to reproduce the main tables.

## Citing the paper

If you find this useful, please cite:

```
Abroshan, H. (2025). AI to Protect AI: A Modular Pipeline for Detecting Label‑Flipping Poisoning Attacks. Machine Learning with Applications, 100768.

```

## Troubleshooting

* **XGBoost GPU not found**: the code falls back to CPU automatically; you can also pass `tree_method=hist` via a config or edit `src/dm_trainer.py`.
* **ChestXray14 path errors**: ensure `Data_Entry_2017.csv` exists and image filenames match (the loader is case‑insensitive and accepts `.png`/`.jpg` variants).
* **Low AUCPR with IF/Mahalanobis** on ChestXray14 is expected; prefer supervised DMs or BB+GB.

## License

Pick a permissive license (MIT or BSD‑3‑Clause). If you add NIH data handling code, follow the dataset terms.

---

### Maintainer
Dr Hossein Abroshan - Anglia Ruskin University (ARU), Cambridge, UK

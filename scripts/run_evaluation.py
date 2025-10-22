#!/usr/bin/env python3
"""
Comprehensive evaluation pipeline with cross-validation and hold-out testing
for MNIST poisoning detection using BMM features and multiple detectors.
Generates CV metrics, final hold-out performance, and saves ROC curves and confusion matrices.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, classification_report, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt

# ------------------ Config ------------------
DATASET = 'MNIST'  # or 'CIFAR10'
POISON_FRAC = 0.1
SRC_CLASS = 1
DST_CLASS = 7
BATCH_SIZE = 256
CV_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ Target Model ------------------
from tm_trainer import SimpleCNN  # for MNIST
# from tm_trainer_cifar import CIFARResNet  # for CIFAR10 if needed

def load_target_model():
    tm = SimpleCNN().to(DEVICE)
    state = torch.load('models/tm_clean/tm_final.pt', map_location='cpu', weights_only=True)
    tm.load_state_dict(state)
    tm.eval()
    return tm

# ------------------ BMMNet Feature Extractor ------------------
class BMMNet(nn.Module):
    def __init__(self, tm: nn.Module):
        super().__init__()
        self.tm = tm
    def forward(self, x, y_true=None):
        B = x.size(0)
        acts, hooks = [], []
        for module in self.tm.modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(
                    lambda m, inp, outp: acts.append(
                        torch.cat([
                            outp.view(B, outp.size(1), -1).mean(2),
                            outp.view(B, outp.size(1), -1).std(2),
                            outp.view(B, outp.size(1), -1).quantile(0.1,2),
                            outp.view(B, outp.size(1), -1).quantile(0.9,2)
                        ], dim=1)
                    )
                ))
        out = self.tm(x)
        for h in hooks: h.remove()
        feats = []
        if y_true is not None:
            loss_raw = F.cross_entropy(out, y_true, reduction='none')
            feats.append(loss_raw.detach().unsqueeze(1))
            probs = F.softmax(out, dim=1)
            feats.append(probs.gather(1, y_true.unsqueeze(1)).detach())
        feats.append(torch.cat(acts, dim=1).detach())
        return torch.cat(feats, dim=1)

# ------------------ Data & Embedding Extraction ------------------

def get_dataloaders():
    if DATASET == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        full = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    else:
        raise ValueError('Unsupported dataset')
    tr_len = int(0.9 * len(full))
    val_len = len(full) - tr_len
    train_ds, val_ds = random_split(full, [tr_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def extract_embeddings(loader, bmm, flip_frac=0.0):
    feats, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_in = y.clone()
            label_flags = np.zeros(y.size(0), dtype=int)
            if flip_frac > 0:
                y_np = y.cpu().numpy(); idx = np.where(y_np == SRC_CLASS)[0]
                flip = np.random.choice(idx, int(flip_frac*len(idx)), replace=False)
                y_np[flip] = DST_CLASS
                label_flags[flip] = 1
                y_in = torch.from_numpy(y_np).to(DEVICE)
            emb = bmm(x, y_in).cpu().numpy()
            feats.append(emb); labs.append(label_flags)
    return np.vstack(feats), np.concatenate(labs)

# ------------------ Cross‑Validation ------------------
def cross_validate(clf_class, X, y, clf_kwargs, name):
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    aucs, f1s = [], []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        clf = clf_class(**clf_kwargs)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_va)[:,1]
        auc = roc_auc_score(y_va, probs)
        t = tune_threshold(probs, y_va)
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_va, preds)
        aucs.append(auc); f1s.append(f1)
    print(f"{name} CV — AUC: {np.mean(aucs):.4f}±{np.std(aucs):.4f}, F1: {np.mean(f1s):.4f}±{np.std(f1s):.4f}")
    return

# ------------------ Threshold Tuning ------------------
def tune_threshold(scores, labels, recall_min=None):
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0,1,101):
        preds = (scores >= t).astype(int)
        rec = recall_score(labels, preds)
        if recall_min and rec < recall_min: continue
        f1 = f1_score(labels, preds)
        if f1 > best_f1: best_f1, best_t = f1, t
    return best_t

# ------------------ Plot and Save ------------------
def save_roc_cm(labels, scores, name, is_distance=False):
    # ROC
    fpr, tpr, _ = roc_curve(labels, scores if not is_distance else -scores)
    plt.figure(); plt.plot(fpr, tpr, label=f'{name} ROC')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'{name} ROC'); plt.legend()
    plt.savefig(f'{OUTPUT_DIR}/roc_{name}.png'); plt.close()
    # CM at  threshold=0.5 (or adjust later)
    preds = (scores >= 0.5).astype(int) if not is_distance else (scores <= np.percentile(scores, POISON_FRAC*100)).astype(int)
    cm = confusion_matrix(labels, preds)
    plt.figure(); plt.matshow(cm, cmap='Blues'); plt.title(f'{name} Confusion Matrix')
    plt.colorbar(); plt.savefig(f'{OUTPUT_DIR}/cm_{name}.png'); plt.close()

# ------------------ Main Pipeline ------------------
if __name__ == '__main__':
    # Load models and data
    tm = load_target_model()
    bmm = BMMNet(tm).to(DEVICE)
    train_loader, val_loader, test_loader = get_dataloaders()

    # 1. Extract train embeddings for detector CV / training
    X_tr, y_tr = extract_embeddings(train_loader, bmm, flip_frac=POISON_FRAC)
    # 2. Cross-validate XGBoost
    print('Cross-validating XGBoost...')
    cross_validate(xgb.XGBClassifier, X_tr, y_tr,
                   {'tree_method':'gpu_hist','predictor':'gpu_predictor','use_label_encoder':False,'max_depth':4,'subsample':0.8,'colsample_bytree':0.8,'reg_lambda':1.0}, 'XGBoost')
    # 3. Cross-validate MLP
    print('Cross-validating MLPClassifier...')
    cross_validate(MLPClassifier, X_tr, y_tr, {'hidden_layer_sizes':(64,),'max_iter':50}, 'MLP')

    # 4. Train final detectors on full train
    xgb_clf = xgb.XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', use_label_encoder=False, max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
    xgb_clf.fit(X_tr, y_tr)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=50)
    mlp_clf.fit(X_tr, y_tr)
    if_clf = IsolationForest(contamination=POISON_FRAC)
    if_clf.fit(X_tr[y_tr == 0])
    cov = EmpiricalCovariance().fit(*extract_embeddings(train_loader, bmm, flip_frac=0.0))

    # 5. Prepare hold-out test: initial test split -> val2 + holdout
    X_test_full, y_test_full = extract_embeddings(test_loader, bmm, flip_frac=POISON_FRAC)
    X_val2, X_hold, y_val2, y_hold = train_test_split(X_test_full, y_test_full, test_size=0.2, stratify=y_test_full, random_state=42)

    # 6. Compute validation scores & tune thresholds
    s_xgb_val = xgb_clf.predict_proba(X_val2)[:,1]
    s_mlp_val = mlp_clf.predict_proba(X_val2)[:,1]
    raw_if_val = -if_clf.decision_function(X_val2)
    s_if_val = (raw_if_val - raw_if_val.min())/(raw_if_val.max()-raw_if_val.min()+1e-8)
    D_val = cov.mahalanobis(X_val2)
    s_mah_val = (D_val - D_val.min())/(D_val.max()-D_val.min()+1e-8)
    s_ens_val = (s_xgb_val + s_mlp_val + s_if_val + s_mah_val)/4
    t_xgb = tune_threshold(s_xgb_val, y_val2)
    t_mlp = tune_threshold(s_mlp_val, y_val2)
    t_if  = tune_threshold(s_if_val,  y_val2)
    t_mah = tune_threshold(s_mah_val, y_val2)
    t_ens = tune_threshold(s_ens_val, y_val2)
    print(f'Tuned thresholds -> XGB: {t_xgb:.2f}, MLP: {t_mlp:.2f}, IF: {t_if:.2f}, Maha: {t_mah:.2f}, Ens: {t_ens:.2f}')

    # 7. Evaluate on hold-out
    s_xgb_hold = xgb_clf.predict_proba(X_hold)[:,1]
    s_mlp_hold = mlp_clf.predict_proba(X_hold)[:,1]
    raw_if_hold = -if_clf.decision_function(X_hold)
    s_if_hold = (raw_if_hold - raw_if_hold.min())/(raw_if_hold.max()-raw_if_hold.min()+1e-8)
    D_hold = cov.mahalanobis(X_hold)
    s_mah_hold = (D_hold - D_hold.min())/(D_hold.max()-D_hold.min()+1e-8)
    s_ens_hold = (s_xgb_hold + s_mlp_hold + s_if_hold + s_mah_hold)/4

    # 7. Evaluate on hold-out with consistent threshold logic
    detectors = [
        ('XGB', s_xgb_hold, t_xgb),
        ('MLP', s_mlp_hold, t_mlp),
        ('IF',  s_if_hold,  t_if),
        ('Maha',s_mah_hold, t_mah),  # treat like other scores: higher means more anomalous
        ('Ens', s_ens_hold, t_ens)
    ]
    for name, scores, thr in detectors:
        auc = roc_auc_score(y_hold, scores)
        preds = (scores >= thr).astype(int)
        acc = accuracy_score(y_hold, preds)
        print(f"-- {name} on hold-out (thr={thr:.2f}) -> AUC={auc:.4f}, Acc={acc:.4f}")
        print(classification_report(y_hold, preds, target_names=['clean','poison'], digits=4))
        # Save ROC curve and CM
        save_roc_cm(y_hold, scores, name)

    print('Evaluation complete.')

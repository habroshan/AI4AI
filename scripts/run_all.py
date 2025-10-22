#!/usr/bin/env python3
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
from sklearn.metrics import roc_auc_score, classification_report, recall_score, f1_score
import joblib

from tm_trainer import SimpleCNN

# --- BMMNet: extract rich stats (loss, conf, activations) ---
class BMMNet(nn.Module):
    def __init__(self, tm: nn.Module):
        super().__init__()
        self.tm = tm

    def forward(self, x, y_true=None):
        B = x.size(0)
        acts, hooks = [], []
        for module in self.tm.modules():
            if isinstance(module, nn.Conv2d):
                def hook_fn(mod, inp, outp):
                    flat = outp.view(B, outp.size(1), -1)
                    m_ = flat.mean(2)
                    s_ = flat.std(2)
                    q10 = flat.quantile(0.1, 2)
                    q90 = flat.quantile(0.9, 2)
                    acts.append(torch.cat([m_, s_, q10, q90], dim=1))
                hooks.append(module.register_forward_hook(hook_fn))
        out = self.tm(x)
        for h in hooks: h.remove()
        feats = []
        if y_true is not None:
            loss_raw = F.cross_entropy(out, y_true, reduction='none')
            feats.append(loss_raw.detach().unsqueeze(1))
            probs = F.softmax(out, dim=1)
            feats.append(probs.gather(1, y_true.unsqueeze(1)).detach())
        act_feats = torch.cat(acts, dim=1).detach()
        feats.append(act_feats)
        return torch.cat(feats, dim=1)

# --- extract embeddings and binary labels (0 clean, 1 poison) ---
def extract_embeddings(loader, bmm, device, flip_frac=0.0, src=1, dst=7):
    bmm.eval()
    feats_list, labs_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_in = y.clone()
            labels = np.zeros(y.size(0), dtype=int)
            if flip_frac > 0:
                y_np = y.cpu().numpy().copy()
                idx = np.where(y_np == src)[0]
                num = int(flip_frac * len(idx))
                flip = np.random.choice(idx, num, replace=False)
                y_np[flip] = dst
                y_in = torch.from_numpy(y_np).to(device)
                labels[flip] = 1
            emb = bmm(x, y_in).cpu().numpy()
            feats_list.append(emb)
            labs_list.append(labels)
    return np.vstack(feats_list), np.concatenate(labs_list)

# --- threshold tuning by maximizing F1 (or recall >= target) ---
def tune_threshold(scores, labels, recall_min=None):
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0, 1, 101):
        preds = (scores >= t).astype(int)
        rec = recall_score(labels, preds)
        if recall_min is not None and rec < recall_min:
            continue
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    poison_frac = 0.1

    # prepare data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    tr_size = int(0.9 * len(full_train)); val_size = len(full_train) - tr_size
    train_ds, val_ds = random_split(full_train, [tr_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_ds      = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # 1) Train TM clean
    print("\n== Training Target Model (clean) ==")
    os.system("python3 tm_trainer.py --poison-fraction 0.0 --output-dir models/tm_clean")

    # load TM & build BMM
    tm = SimpleCNN().to(device)
    tm.load_state_dict(torch.load('models/tm_clean/tm_final.pt', map_location='cpu', weights_only=True))
    bmm = BMMNet(tm).to(device)

    # 2) Extract embeddings
    print("\n== Extracting Embeddings ==")
    X_tr_clean, _ = extract_embeddings(train_loader, bmm, device, flip_frac=0.0)
    X_val_clean, _= extract_embeddings(val_loader, bmm, device, flip_frac=0.0)
    X_tr, y_tr    = extract_embeddings(train_loader, bmm, device, flip_frac=poison_frac)
    X_val, y_val  = extract_embeddings(val_loader, bmm, device, flip_frac=poison_frac)

    # 3) Train detectors
    print("\n== Training Detectors ==")
    xgb_clf = xgb.XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', use_label_encoder=False)
    xgb_clf.fit(X_tr, y_tr)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=50)
    mlp_clf.fit(X_tr, y_tr)
    if_clf = IsolationForest(contamination=poison_frac)
    if_clf.fit(X_tr[y_tr == 0])
    cov = EmpiricalCovariance().fit(X_tr_clean)

    # 4) Tune thresholds on validation
    print("\n== Tuning thresholds on validation ==")
    s_xgb_val = xgb_clf.predict_proba(X_val)[:,1]
    s_mlp_val = mlp_clf.predict_proba(X_val)[:,1]
    raw_if_val= -if_clf.decision_function(X_val)
    s_if_val  = (raw_if_val - raw_if_val.min())/(raw_if_val.max()-raw_if_val.min()+1e-8)
    D_val     = cov.mahalanobis(X_val)
    s_mah_val= (D_val - D_val.min())/(D_val.max()-D_val.min()+1e-8)
    s_ens_val= (s_xgb_val + s_mlp_val + s_if_val + s_mah_val)/4
    t_xgb = tune_threshold(s_xgb_val, y_val)
    t_mlp = tune_threshold(s_mlp_val, y_val)
    t_if  = tune_threshold(s_if_val,  y_val)
    t_mah = tune_threshold(s_mah_val, y_val)
    t_ens = tune_threshold(s_ens_val, y_val)
    print(f"Thresholds -> XGB:{t_xgb:.2f}, MLP:{t_mlp:.2f}, IF:{t_if:.2f}, Maha:{t_mah:.2f}, Ens:{t_ens:.2f}")

    # 5) Evaluate on test
    print("\n== Evaluating on TEST ==")
    X_test, y_test = extract_embeddings(test_loader, bmm, device, flip_frac=poison_frac)
    s_xgb_test = xgb_clf.predict_proba(X_test)[:,1]
    s_mlp_test = mlp_clf.predict_proba(X_test)[:,1]
    raw_if_test= -if_clf.decision_function(X_test)
    s_if_test  = (raw_if_test - raw_if_test.min())/(raw_if_test.max()-raw_if_test.min()+1e-8)
    D_test     = cov.mahalanobis(X_test)
    s_mah_test= (D_test - D_test.min())/(D_test.max()-D_test.min()+1e-8)
    s_ens_test= (s_xgb_test + s_mlp_test + s_if_test + s_mah_test)/4
    # report each
    for name, scores, thresh in [
        ('XGB', s_xgb_test, t_xgb),
        ('MLP', s_mlp_test, t_mlp),
        ('IF',  s_if_test,  t_if),
        ('Maha',s_mah_test, t_mah),
        ('Ens', s_ens_test, t_ens)
    ]:
        auc = roc_auc_score(y_test, scores)
        preds = (scores >= thresh).astype(int)
        print(f"\n-- {name} Detector (thr={thresh:.2f}) -- AUC: {auc:.4f}")
        print(classification_report(y_test, preds, target_names=['clean','poison'], digits=4))

    print("\n== Pipeline complete ==")

# dm_trainer.py (updated for BB/GB/BB+GB & AUCPR)

import argparse
import json
import os
import joblib
import numpy as np
import h5py
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
)


def load_h5(path):
    with h5py.File(path, 'r') as f:
        X = f['X'][()]
        y = f['y'][()] if 'y' in f else None
        mode = f.attrs.get('feature_mode', 'unknown')
    return X, y, mode


def build_model(method, use_gpu=False):
    if method == 'xgb':
        params = {
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'tree_method': 'gpu_hist' if use_gpu else 'hist',
            'predictor': 'gpu_predictor' if use_gpu else 'cpu_predictor',
            'device': 'cuda' if use_gpu else 'cpu'
        }
        return XGBClassifier(**params)
    elif method == 'mlp':
        if use_gpu:
            print("⚠️  MLPClassifier has no GPU support; using CPU.")
        return MLPClassifier(max_iter=300)
    elif method == 'if':
        if use_gpu:
            print("⚠️  IsolationForest has no GPU support; using CPU.")
        return IsolationForest(contamination=0.1)
    else:
        raise ValueError('Unknown method: ' + method)


def mahalanobis_scores(X_tr, y_tr, X_te):
    from sklearn.covariance import EmpiricalCovariance
    Xc = X_tr[y_tr == 0]
    cov = EmpiricalCovariance().fit(Xc)
    D = cov.mahalanobis(X_te)
    return D


def evaluate_scores(y_true, scores):
    # threshold-free
    roc = float(roc_auc_score(y_true, scores))
    pr, rc, th = precision_recall_curve(y_true, scores)
    aucpr = float(auc(rc, pr))
    # threshold by max-F1 (for reporting)
    if len(th) > 0:
        f1s = (2 * pr * rc) / (pr + rc + 1e-12)
        t_idx = int(np.argmax(f1s))
        thr = float(th[t_idx])
    else:
        thr = 0.5
    y_pred = (scores >= thr).astype(int)
    return {
        'AUC': roc,
        'AUCPR': aucpr,
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
        'thr': thr,
    }


def fit_and_score(method, X_tr, y_tr, X_te, y_te, use_gpu=False):
    # Standardize (fit on train only)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Optional PCA for high-D unsupervised methods
    if method in ('if',):
        pca = PCA(n_components=0.95, svd_solver='full').fit(X_tr_s)
        X_tr_s = pca.transform(X_tr_s)
        X_te_s = pca.transform(X_te_s)

    if method in ('xgb', 'mlp'):
        clf = build_model(method, use_gpu)
        if method == 'xgb':
            # rebalance via scale_pos_weight
            pos = max((y_tr == 1).sum(), 1)
            neg = max((y_tr == 0).sum(), 1)
            spw = float(neg) / float(pos)
            clf.set_params(scale_pos_weight=spw)
        clf.fit(X_tr_s, y_tr)
        if hasattr(clf, 'predict_proba'):
            scores = clf.predict_proba(X_te_s)[:, 1]
        else:
            scores = clf.decision_function(X_te_s)
    elif method == 'if':
        iso = build_model('if')
        iso.fit(X_tr_s[y_tr == 0])
        scores = -iso.score_samples(X_te_s)  # higher = more anomalous
    elif method == 'maha':
        scores = mahalanobis_scores(X_tr_s, y_tr, X_te_s)
    else:
        raise ValueError('Unknown method: ' + method)

    return evaluate_scores(y_te, scores)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train/Evaluate detectors on merged H5 (X,y) and report AUC/AUCPR')
    p.add_argument('--train_h5', required=True)
    p.add_argument('--test_h5',  required=True)
    p.add_argument('--detector', choices=['xgb','mlp','if','maha'], default='xgb')
    p.add_argument('--use-gpu',  action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dump_scores', type=str, default=None, help='(optional) JSONL file of {y_true, score}')
    p.add_argument('--save_model', type=str, default=None, help='(optional) path to save fitted model')
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)

    X_tr, y_tr, mode_tr = load_h5(args.train_h5)
    X_te, y_te, mode_te = load_h5(args.test_h5)
    assert y_tr is not None and y_te is not None, "train/test H5 must contain 'y' labels (0=clean,1=poison)"

    metrics = fit_and_score(args.detector, X_tr, y_tr, X_te, y_te, use_gpu=args.use_gpu)

    # Optional score dump unsupported in this minimal path; runner can compute metrics itself if needed
    # Save model if requested (supervised only)
    if args.save_model and args.detector in ('xgb','mlp'):
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        joblib.dump(metrics, args.save_model)

    print('METRICS', json.dumps(metrics))


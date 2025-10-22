import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from data_loaders import get_mnist_loaders, get_cifar_loaders, get_chestxray14_loaders
from tm_trainer import SimpleCNN
from torchvision import models
import torch.nn as nn

from run_comprehensive_evaluation import BMMNet, extract_emb as extract_mnist_emb
from run_chestxray14_pipeline import (
    ChestTM, ChestBMM, train_tm as train_chest_tm,
    extract_emb as extract_chest_emb, train_ld, test_ld
)

from detectors import train_xgb, train_mlp, train_if, train_maha, get_detector_predictions
from metrics_utils import plot_confusion_matrix, plot_multi_roc, save_metrics_list


def train_mnist_tm(poison_frac, device):
    os.makedirs('models/MNIST', exist_ok=True)
    tm = SimpleCNN().to(device)
    tm_path = 'models/MNIST/tm_final.pt'
    if os.path.exists(tm_path):
        tm.load_state_dict(torch.load(tm_path, map_location=device))
    else:
        trn_loader, _, _ = get_mnist_loaders()
        opt = torch.optim.Adam(tm.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        for ep in range(10):
            tm.train()
            total = 0
            for imgs, labels in trn_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                out = tm(imgs)
                loss = crit(out, labels)
                loss.backward()
                opt.step()
                total += loss.item()
            print(f'MNIST TM Epoch {ep+1}/10: loss={total/len(trn_loader):.4f}')
        torch.save(tm.state_dict(), tm_path)
    tm.eval()
    return tm


def train_cifar_tm(poison_frac, device, batch_size=64):
    os.makedirs('models/CIFAR10', exist_ok=True)
    trn_loader, tst_loader = get_cifar_loaders(batch_size)
    tm = models.resnet18(pretrained=False, num_classes=10).to(device)
    tm_path = 'models/CIFAR10/tm_final.pt'
    if os.path.exists(tm_path):
        tm.load_state_dict(torch.load(tm_path, map_location=device))
    else:
        opt = torch.optim.Adam(tm.parameters(), lr=1e-4)
        crit = nn.CrossEntropyLoss()
        for ep in range(10):
            tm.train()
            total = 0
            for imgs, labels in trn_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                mask = (labels == 1)
                idxs = mask.nonzero(as_tuple=True)[0]
                k = int(len(idxs) * poison_frac)
                if k > 0:
                    perm = torch.randperm(len(idxs))
                    flip = idxs[perm[:k]]
                    labels[flip] = 7
                opt.zero_grad()
                out = tm(imgs)
                loss = crit(out, labels)
                loss.backward()
                opt.step()
                total += loss.item()
            print(f'CIFAR10 TM Epoch {ep+1}/10: loss={total/len(trn_loader):.4f}')
        torch.save(tm.state_dict(), tm_path)
    tm.eval()
    return tm, trn_loader, tst_loader


def run_pipeline(dataset, poison_frac, device):
    results = []
    if dataset == 'MNIST':
        outdir = f'extended_output_MNIST'
        os.makedirs(outdir, exist_ok=True)
        tm = train_mnist_tm(poison_frac, device)
        bmm = BMMNet(tm).to(device)
        trn_loader, _, tst_loader = get_mnist_loaders()
        Xtr, ytr = extract_mnist_emb(trn_loader, bmm, 1, 7, poison_frac)
        Xte, yte = extract_mnist_emb(tst_loader, bmm, 1, 7, poison_frac)
    elif dataset == 'CIFAR10':
        outdir = f'extended_output_CIFAR10'
        os.makedirs(outdir, exist_ok=True)
        tm, trn_loader, tst_loader = train_cifar_tm(poison_frac, device)
        bmm = BMMNet(tm).to(device)
        Xtr, ytr = extract_mnist_emb(trn_loader, bmm, 1, 7, poison_frac)
        Xte, yte = extract_mnist_emb(tst_loader, bmm, 1, 7, poison_frac)
    elif dataset == 'ChestXray14':
        outdir = f'extended_output_ChestXray14'
        os.makedirs(outdir, exist_ok=True)
        tm = ChestTM().to(device)
        tm_path = 'models/ChestXray14/tm_final.pt'
        if os.path.exists(tm_path):
            tm.load_state_dict(torch.load(tm_path, map_location=device))
        else:
            tm = train_chest_tm(poison_frac, tm_path)
        tm.eval()
        bmm = ChestBMM(tm).to(device)
        Xtr, ytr = extract_chest_emb(train_ld, bmm, poison_frac)
        Xte, yte = extract_chest_emb(test_ld, bmm, poison_frac)
    else:
        raise ValueError('Unknown dataset')

    # train detectors
    dets = {
        'XGB': train_xgb(Xtr, ytr),
        'MLP': train_mlp(Xtr, ytr, hidden_layer_sizes=(64,)),
        'IF': train_if(Xtr, contamination=poison_frac),
        'Maha': train_maha(Xtr, ytr)
    }

    # evaluate
    roc_data = {}
    for name, model in dets.items():
        y_pred, scores = get_detector_predictions(model, Xte, name, frac=poison_frac)
        # confusion matrix
        plot_confusion_matrix(yte, y_pred, labels=[0,1], output_path=os.path.join(outdir, f'cm_{name}.png'), title=f'{dataset} CM {name} ({int(poison_frac*100)}%)')
        fpr, tpr, _ = roc_curve(yte, scores)
        auc_val = auc(fpr, tpr)
        roc_data[name] = (fpr, tpr, auc_val)
        results.append({
            'Dataset': dataset,
            'Detector': name,
            'PoisonFrac': poison_frac,
            'TP': int(((yte == 1) & (y_pred == 1)).sum()),
            'FP': int(((yte == 0) & (y_pred == 1)).sum()),
            'TN': int(((yte == 0) & (y_pred == 0)).sum()),
            'FN': int(((yte == 1) & (y_pred == 0)).sum()),
            'Precision': precision_score(yte, y_pred),
            'Recall': recall_score(yte, y_pred),
            'F1': f1_score(yte, y_pred),
            'AUC': auc_val
        })

    plot_multi_roc(roc_data, output_path=os.path.join(outdir, f'roc_all_{int(poison_frac*100)}pct.png'), title=f'{dataset} ROCs ({int(poison_frac*100)}%)')
    save_metrics_list(results, os.path.join(outdir, 'metrics_summary.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['MNIST','CIFAR10','ChestXray14','all'], default='all')
    parser.add_argument('--poison_frac', type=float, nargs='+', default=[0.1, 0.2])
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = ['MNIST','CIFAR10','ChestXray14'] if args.dataset == 'all' else [args.dataset]
    for ds in datasets:
        for pf in args.poison_frac:
            run_pipeline(ds, pf, device)
np.save(os.path.join(outdir, "y_true.npy"), y_true)

# Save scores and predictions for each detector
np.save(os.path.join(outdir, f"{detector_name}_scores.npy"), scores)
np.save(os.path.join(outdir, f"{detector_name}_preds.npy"), y_pred)

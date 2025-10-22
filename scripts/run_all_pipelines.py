#!/usr/bin/env python3
"""
Unified pipeline runner for MNIST, CIFAR-10, and ChestXray14 datasets.
For each dataset, trains/loads the Target Model (TM), extracts BMM features,
trains detectors (XGB, MLP, IF, Mahalanobis, Ensemble),
and saves ROC and CM plots plus metrics CSVs under output_<dataset>/.

Usage:
  # run all with 10% poison. We sometimes used nohup in our project as it could take a while and we didn't want to lose the results.
  nohup python3 run_all_pipelines.py --poison_frac 0.1 &

  # run only CIFAR-10
  python3 run_all_pipelines.py --dataset CIFAR10 --poison_frac 0.05
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance

# ------------------ Utility ------------------

def save_metrics_and_plots(y_true, scores, name, outdir, prefix):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr)
    plt.title(f'{name} ROC ({prefix})'); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.savefig(os.path.join(outdir, f'roc_{name}_{prefix}.png')); plt.close()

    thresh = np.linspace(0,1,101)
    f1s = [f1_score(y_true, (scores>=t).astype(int)) for t in thresh]
    best_t = thresh[np.argmax(f1s)]
    preds = (scores>=best_t).astype(int)
    cm = confusion_matrix(y_true, preds)
    plt.figure(); plt.matshow(cm, cmap='Blues')
    plt.title(f'{name} CM ({prefix})'); plt.colorbar()
    plt.savefig(os.path.join(outdir, f'cm_{name}_{prefix}.png')); plt.close()

    acc = accuracy_score(y_true, preds)
    f1v = f1_score(y_true, preds)
    return roc_auc, acc, f1v

# ------------------ MNIST ------------------

def run_mnist(poison_frac, device):
    from tm_trainer import SimpleCNN
    from run_comprehensive_evaluation import BMMNet, extract_emb, get_loaders

    print('== MNIST pipeline ==')
    os.makedirs('models/MNIST', exist_ok=True)
    os.makedirs('output_MNIST', exist_ok=True)

    # Train or load TM
    tm = SimpleCNN().to(device)
    tm_path = 'models/MNIST/tm_final.pt'
    if os.path.exists(tm_path):
        tm.load_state_dict(torch.load(tm_path, map_location=device))
    else:
        trn, _, _ = get_loaders()
        opt = torch.optim.Adam(tm.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        for ep in range(10):
            tm.train(); tot=0
            for imgs, labels in trn:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad(); out=tm(imgs)
                loss=crit(out,labels); loss.backward(); opt.step()
                tot+=loss.item()
            print(f'MNIST TM Epoch {ep+1}/10: loss={tot/len(trn):.4f}')
        torch.save(tm.state_dict(), tm_path)
    tm.eval()

    # Extract BMM features
    bmm = BMMNet(tm).to(device)
    trn, _, tst = get_loaders()
    Xtr, ytr = extract_emb(trn, bmm, 1,7, poison_frac)
    Xte, yte = extract_emb(tst, bmm, 1,7, poison_frac)

    # Fit detectors
    dets = {
        'XGB': xgb.XGBClassifier(tree_method='hist', max_depth=4, subsample=0.8, colsample_bytree=0.8),
        'MLP': MLPClassifier(hidden_layer_sizes=(64,), max_iter=100),
        'IF':  IsolationForest(contamination=poison_frac)
    }
    dets['Maha'] = EmpiricalCovariance().fit(Xtr[ytr==0])

    metrics=[]; prefix=f'MNIST_{int(poison_frac*100)}pct'
    for name, clf in dets.items():
        print(f'MNIST: Eval {name}')
        if name=='IF': clf.fit(Xtr); scores = -clf.decision_function(Xte)
        elif name=='Maha': scores = -clf.mahalanobis(Xte)
        else: clf.fit(Xtr,ytr); scores = clf.predict_proba(Xte)[:,1]
        roc, acc, f1v = save_metrics_and_plots(yte, scores, name, 'output_MNIST', prefix)
        metrics.append((name, roc, acc, f1v))

    # Ensemble
    arr = np.vstack([
        dets['XGB'].predict_proba(Xte)[:,1],
        dets['MLP'].predict_proba(Xte)[:,1],
        -dets['IF'].decision_function(Xte),
        -dets['Maha'].mahalanobis(Xte)
    ]).T
    ens = (arr-arr.min(0))/(arr.ptp(0)+1e-8)
    es = ens.mean(1)
    roc, acc, f1v = save_metrics_and_plots(yte, es, 'Ens', 'output_MNIST', prefix)
    metrics.append(('Ens', roc, acc, f1v))

    with open('output_MNIST/metrics_summary.csv','w') as f:
        f.write('Detector,ROC_AUC,Acc,F1\n')
        for n,r,a,ff in metrics: f.write(f'{n},{r:.4f},{a:.4f},{ff:.4f}\n')

# ------------------ CIFAR-10 ------------------

def run_cifar10(poison_frac, device):
    print('== CIFAR-10 pipeline ==')
    os.makedirs('models/CIFAR10', exist_ok=True)
    os.makedirs('output_CIFAR10', exist_ok=True)

    from torchvision.datasets import CIFAR10
    # transforms: resize to 224 for resnet18, normalize imagenet stats
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    trn_ds = CIFAR10('data',train=True,download=True,transform=tf)
    tst_ds = CIFAR10('data',train=False,download=True,transform=tf)
    trn_ld = DataLoader(trn_ds,batch_size=64,shuffle=True)
    tst_ld = DataLoader(tst_ds,batch_size=64,shuffle=False)

    # Train or load ResNet18 TM
    tm = models.resnet18(pretrained=False, num_classes=10).to(device)
    tm_path = 'models/CIFAR10/tm_final.pt'
    if os.path.exists(tm_path):
        tm.load_state_dict(torch.load(tm_path,map_location=device))
    else:
        opt = torch.optim.Adam(tm.parameters(),lr=1e-4)
        crit = nn.CrossEntropyLoss()
        for ep in range(10):
            tm.train(); tot=0
            for imgs, labels in trn_ld:
                imgs, labels = imgs.to(device), labels.to(device)
                # label flipping: 1->7
                mask = (labels==1)
                idxs = mask.nonzero(as_tuple=True)[0]
                k = int(idxs.size(0)*poison_frac)
                if k>0:
                    perm = torch.randperm(idxs.size(0))
                    flip = idxs[perm[:k]]
                    labels[flip] = 7
                opt.zero_grad(); out=tm(imgs)
                loss=crit(out,labels); loss.backward(); opt.step()
                tot+=loss.item()
            print(f'CIFAR10 TM Epoch {ep+1}/10: loss={tot/len(trn_ld):.4f}')
        torch.save(tm.state_dict(),tm_path)
    tm.eval()

    # BMM features via generic BMMNet
    from run_comprehensive_evaluation import BMMNet, extract_emb
    bmm = BMMNet(tm).to(device)
    Xtr, ytr = extract_emb(trn_ld, bmm, 1,7, poison_frac)
    Xte, yte = extract_emb(tst_ld, bmm, 1,7, poison_frac)

    dets = {
        'XGB': xgb.XGBClassifier(tree_method='hist', max_depth=4, subsample=0.8, colsample_bytree=0.8),
        'MLP': MLPClassifier(hidden_layer_sizes=(64,), max_iter=100),
        'IF':  IsolationForest(contamination=poison_frac)
    }
    dets['Maha'] = EmpiricalCovariance().fit(Xtr[ytr==0])

    metrics=[]; prefix=f'CIFAR10_{int(poison_frac*100)}pct'
    for name, clf in dets.items():
        print(f'CIFAR10: Eval {name}')
        if name=='IF': clf.fit(Xtr); scores=-clf.decision_function(Xte)
        elif name=='Maha': scores=-clf.mahalanobis(Xte)
        else: clf.fit(Xtr,ytr); scores=clf.predict_proba(Xte)[:,1]
        roc, acc, f1v = save_metrics_and_plots(yte, scores, name, 'output_CIFAR10',prefix)
        metrics.append((name,roc,acc,f1v))

    arr = np.vstack([
        dets['XGB'].predict_proba(Xte)[:,1],
        dets['MLP'].predict_proba(Xte)[:,1],
        -dets['IF'].decision_function(Xte),
        -dets['Maha'].mahalanobis(Xte)
    ]).T
    ens = (arr-arr.min(0))/(arr.ptp(0)+1e-8)
    es = ens.mean(1)
    roc, acc, f1v = save_metrics_and_plots(yte, es, 'Ens', 'output_CIFAR10', prefix)
    metrics.append(('Ens',roc,acc,f1v))

    with open('output_CIFAR10/metrics_summary.csv','w') as f:
        f.write('Detector,ROC_AUC,Acc,F1\n')
        for n,r,a,ff in metrics: f.write(f'{n},{r:.4f},{a:.4f},{ff:.4f}\n')

# ------------------ ChestXray14 ------------------
def run_chest(poison_frac, device):
    print('== ChestXray14 pipeline ==')
    os.makedirs('models/ChestXray14', exist_ok=True)
    os.makedirs('output_ChestXray14', exist_ok=True)
    from run_chestxray14_pipeline import ChestTM, ChestBMM, train_tm, extract_emb, train_ld, test_ld
    tm = ChestTM().to(device)
    tm_path='models/ChestXray14/tm_final.pt'
    if os.path.exists(tm_path): tm.load_state_dict(torch.load(tm_path,map_location=device))
    else: tm=train_tm(poison_frac,tm_path)
    tm.eval()
    bmm=ChestBMM(tm).to(device)
    Xtr,ytr=extract_emb(train_ld,bmm,poison_frac)
    Xte,yte=extract_emb(test_ld,bmm,poison_frac)
    dets={'XGB':xgb.XGBClassifier(tree_method='hist',max_depth=4,subsample=0.8,colsample_bytree=0.8),
          'MLP':MLPClassifier(hidden_layer_sizes=(64,),max_iter=100),
          'IF':IsolationForest(contamination=poison_frac)}
    dets['Maha']=EmpiricalCovariance().fit(Xtr[ytr==0])
    metrics=[]; prefix=f'Chest_{int(poison_frac*100)}pct'
    for name,clf in dets.items():
        print(f'Chest: Eval {name}')
        if name=='IF': clf.fit(Xtr); scores=-clf.decision_function(Xte)
        elif name=='Maha': scores=-clf.mahalanobis(Xte)
        else: clf.fit(Xtr,ytr); scores=clf.predict_proba(Xte)[:,1]
        roc, acc, f1v = save_metrics_and_plots(yte, scores, name, 'output_ChestXray14', prefix)
        metrics.append((name,roc,acc,f1v))
    arr=np.vstack([dets['XGB'].predict_proba(Xte)[:,1],dets['MLP'].predict_proba(Xte)[:,1],
                   -dets['IF'].decision_function(Xte),-dets['Maha'].mahalanobis(Xte)]).T
    ens=(arr-arr.min(0))/(arr.ptp(0)+1e-8); es=ens.mean(1)
    roc, acc, f1v = save_metrics_and_plots(yte, es, 'Ens', 'output_ChestXray14', prefix)
    metrics.append(('Ens',roc,acc,f1v))
    with open('output_ChestXray14/metrics_summary.csv','w') as f:
        f.write('Detector,ROC_AUC,Acc,F1\n')
        for n,r,a,ff in metrics: f.write(f'{n},{r:.4f},{a:.4f},{ff:.4f}\n')

# --------------- Main ---------------
if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['MNIST','CIFAR10','ChestXray14','all'], default='all', help='Datasets to run')
    p.add_argument('--poison_frac', type=float, default=0.1, help='Poison fraction')
    args=p.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset in ('MNIST','all'): run_mnist(args.poison_frac, dev)
    if args.dataset in ('CIFAR10','all'): run_cifar10(args.poison_frac, dev)
    if args.dataset in ('ChestXray14','all'): run_chest(args.poison_frac, dev)
    print('All requested pipelines complete!')

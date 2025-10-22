#!/usr/bin/env python3
"""
Pipeline for ChestXray14 poisoning detection: TM definition, loader setup, BMM extraction, detectors.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

# ----------------------- Dataset -----------------------
class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform, mode='train', split_ratio=0.8):
        df = pd.read_csv(csv_file)
        df = df[df['Finding Labels'].isin(['No Finding','Pneumonia'])].reset_index(drop=True)
        df['label'] = df['Finding Labels'].map({'No Finding':0,'Pneumonia':1})
        cut = int(len(df)*split_ratio)
        self.df = df.iloc[:cut] if mode=='train' else df.iloc[cut:]
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['Image Index'])).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, row['label']

# ----------------------- Transforms & Loaders -----------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

csv_path = 'data/ChestXray14/Data_Entry_2017_v2020.csv'
img_dir  = 'data/ChestXray14/images'

def get_chest_loaders(batch_size=32, split_ratio=0.8):
    ds_train = ChestXrayDataset(csv_path, img_dir, transform, 'train', split_ratio)
    ds_test  = ChestXrayDataset(csv_path, img_dir, transform, 'test',  split_ratio)
    return DataLoader(ds_train, batch_size=batch_size, shuffle=True), DataLoader(ds_test, batch_size=batch_size, shuffle=False)

train_ld, test_ld = get_chest_loaders()

# ----------------------- TM Definition -----------------------
class ChestTM(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(pretrained=True)
        base.fc = nn.Linear(base.fc.in_features, 2)
        self.model = base
    def forward(self, x): return self.model(x)

# ----------------------- BMM Feature Extractor -----------------------
class ChestBMM(nn.Module):
    def __init__(self, tm):
        super().__init__(); self.tm = tm
    def forward(self, x, y=None):
        B = x.size(0)
        acts, hooks = [], []
        for m in self.tm.modules():
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(lambda _,__,o: acts.append(o.view(B,o.size(1),-1).mean(2))))
        out = self.tm(x)
        for h in hooks: h.remove()
        feats = [torch.cat(acts, 1).detach()]
        if y is not None:
            loss = F.cross_entropy(out, y, reduction='none')
            feats.insert(0, loss.unsqueeze(1).detach())
            conf = F.softmax(out,1).gather(1, y.unsqueeze(1)).detach()
            feats.insert(1, conf)
        return torch.cat(feats, 1)

# ----------------------- TM Training -----------------------
def train_tm(poison_frac, output_path, src=0, dst=1, epochs=5, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tm = ChestTM().to(device)
    optim = torch.optim.Adam(tm.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        tm.train(); total=0
        for imgs, labels in train_ld:
            imgs, labels = imgs.to(device), labels.to(device)
            mask = (labels==src)
            idxs = mask.nonzero(as_tuple=True)[0]
            k = int(idxs.size(0) * poison_frac)
            if k>0:
                perm = torch.randperm(idxs.size(0))
                flip = idxs[perm[:k]]
                labels[flip] = dst
            optim.zero_grad()
            out = tm(imgs)
            loss = crit(out, labels)
            loss.backward(); optim.step()
            total += loss.item()
        print(f'Epoch {ep}/{epochs-1}: loss={total/len(train_ld):.4f}')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(tm.state_dict(), output_path)
    return tm

# ----------------------- Embedding Extraction -----------------------
def extract_emb(dataloader, bmm_model, poison_frac, src=0, dst=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = [], []
    bmm_model.eval()
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        mask = (labels==src)
        idxs = mask.nonzero(as_tuple=True)[0]
        k = int(idxs.size(0) * poison_frac)
        det = torch.zeros_like(labels)
        if k>0:
            perm = torch.randperm(idxs.size(0))
            flip = idxs[perm[:k]]
            labels[flip] = dst
            det[flip] = 1
        emb = bmm_model(imgs, labels).cpu().numpy()
        X.append(emb)
        y.append(det.cpu().numpy())
    return np.vstack(X), np.concatenate(y)

# ----------------------- Example Main -----------------------
if __name__=='__main__':
    tm = train_tm(0.1, 'models/ChestXray14/tm_final.pt')
    print('TM trained')

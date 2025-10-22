import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

from tm_trainer import SimpleCNN
from meta_trainer import BMMNet, attack_simulator
from torchvision import datasets, transforms

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256

# Load MNIST and prepare subset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
full_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
_, val_subset = random_split(full_ds, [len(full_ds)-5000, 5000])
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Instantiate BMMNet and load checkpoint
tm = SimpleCNN().to(device)
ckpt = torch.load('models/meta/checkpoint.pth', map_location='cpu', weights_only=True)
tm.load_state_dict(torch.load('models/tm_clean/tm_final.pt', map_location='cpu', weights_only=True))
bmm = BMMNet(tm).to(device)

# Dummy forward pass to initialize fc layers
dummy_x, dummy_y = next(iter(val_loader))
dummy_x, dummy_y = dummy_x.to(device), dummy_y.to(device)
dx_p, dy_p = attack_simulator(dummy_x, dummy_y)
dummy_xall = torch.cat([dummy_x, dx_p], dim=0)
dummy_ycls = torch.cat([dummy_y, dy_p], dim=0).to(device)
_ = bmm(dummy_xall, dummy_ycls)

# Load BMMNet weights
bmm.load_state_dict(ckpt['bmm'], strict=False)

# Load XGBoost
xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model('models/meta/xgb.json')
# Load sklearn models
mlp_clf = joblib.load('models/meta/mlp.pkl')
if_clf  = joblib.load('models/meta/if.pkl')

# Function to extract embeddings and true detection labels
def extract_embeddings(loader):
    bmm.eval()
    feats, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_p, y_p = attack_simulator(x, y)
            y_det = torch.cat([torch.zeros_like(y), torch.ones_like(y_p)], dim=0).cpu().numpy()
            x_all = torch.cat([x, x_p], dim=0)
            y_cls = torch.cat([y, y_p], dim=0).to(device)
            emb = bmm(x_all, y_cls)
            feats.append(emb.cpu().numpy())
            labs.append(y_det)
    return np.vstack(feats), np.concatenate(labs)

# Extract validation embeddings
emb_val, label_val = extract_embeddings(val_loader)

# 1) t-SNE
print("Running t-SNE...")
tsne = TSNE(n_components=2, init='pca', learning_rate='auto')
tsne_vis = tsne.fit_transform(emb_val)
plt.figure(figsize=(8,8))
for cls, name in [(0,'clean'), (1,'poison')]:
    idx = label_val == cls
    plt.scatter(tsne_vis[idx,0], tsne_vis[idx,1], label=name, alpha=0.5, s=5)
plt.legend(); plt.title('t-SNE of BMMNet embeddings')
plt.savefig('tsne_val_embeddings.png')
print("Saved tsne_val_embeddings.png")

# 2) PCA
pca = PCA(n_components=2)
pca_vis = pca.fit_transform(emb_val)
plt.figure(figsize=(8,8))
for cls,name in [(0,'clean'), (1,'poison')]:
    idx = label_val==cls
    plt.scatter(pca_vis[idx,0], pca_vis[idx,1], label=name, alpha=0.5, s=5)
plt.legend(); plt.title('PCA of BMMNet embeddings')
plt.savefig('pca_val_embeddings.png')
print("Saved pca_val_embeddings.png")

# 3) XGBoost feature importances
gain = xgb_clf.get_booster().get_score(importance_type='gain')
top10 = sorted(gain.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 XGBoost Importances:")
for feat, score in top10:
    print(f"Feature {feat}: {score:.4f}")

print("Diagnostics complete.")

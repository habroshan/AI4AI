import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*5*5,128), nn.ReLU(), nn.Linear(128,10)
        )
    def forward(self,x):
        return self.classifier(self.features(x))

def flip_labels(dataset,fraction,src=None,dst=None):
    import numpy as np
    targets = np.array(dataset.targets)
    n = len(targets)
    num = int(fraction*n)
    if src is not None and dst is not None:
        idx = np.where(targets==src)[0]
        flip = np.random.choice(idx, min(num,len(idx)), replace=False)
        targets[flip]=dst
    else:
        flip = np.random.choice(n,num,replace=False)
        targets[flip] = np.random.randint(0,10,size=num)
    dataset.targets = targets.tolist()
    return dataset

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--output-dir', default='./models/tm_clean')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--poison-fraction', type=float, default=0.0)
    p.add_argument('--flip-src', type=int, default=None)
    p.add_argument('--flip-dst', type=int, default=None)
    p.add_argument('--device', choices=['cpu','cuda'], default=None)
    args=p.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    if args.poison_fraction>0:
        ds = flip_labels(ds,args.poison_fraction,args.flip_src,args.flip_dst)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = SimpleCNN().to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.output_dir,exist_ok=True)
    for ep in range(1,args.epochs+1):
        total=0
        model.train()
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad()
            out=model(x)
            loss=crit(out,y)
            loss.backward()
            opt.step()
            total+=loss.item()*x.size(0)
        print(f"Epoch {ep}/{args.epochs} avg_loss={total/len(ds):.4f}")
    torch.save(model.state_dict(), os.path.join(args.output_dir,'tm_final.pt'))
    print("TM training complete.")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import argparse

class CIFARResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=False)
        # Modify for CIFAR-10 input (3x32x32)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        # Adjust final FC
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def train_cifar(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    # Datasets
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_ds  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = CIFARResNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss, total_correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            total_correct += (preds == y).sum().item()
            total += x.size(0)
        avg_loss = total_loss/total
        acc = total_correct/total
        print(f"Epoch {epoch}/{args.epochs} loss={avg_loss:.4f} acc={acc:.4f}")

    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'cifar_resnet.pt'))
    print(f"Model saved to {args.output_dir}/cifar_resnet.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', type=str, default='models/tm_cifar')
    args = parser.parse_args()
    train_cifar(args)

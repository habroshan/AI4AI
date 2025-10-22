import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms

# MNIST loader
def get_mnist_loaders(batch_size=64, val_split=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full = MNIST(root='data', train=True, download=True, transform=transform)
    val_size = int(len(full) * val_split)
    train_size = len(full) - val_size
    train_ds, val_ds = random_split(full, [train_size, val_size])
    test_ds = MNIST(root='data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# CIFAR-10 loader
def get_cifar_loaders(batch_size=64):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trn_ds = CIFAR10('data', train=True, download=True, transform=tf)
    tst_ds = CIFAR10('data', train=False, download=True, transform=tf)
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=False)
    return trn_loader, tst_loader

# ChestXray14 loader
def get_chestxray14_loaders(batch_size=32, split_ratio=0.8):
    from run_chestxray14_pipeline import get_chest_loaders
    return get_chest_loaders(batch_size=batch_size, split_ratio=split_ratio)

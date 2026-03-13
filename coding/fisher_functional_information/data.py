"""
Data loading for the Fisher Fundamental Theorem experiment.

Setting 1: MNIST (for MLP)
Setting 2: CIFAR-10 (for CNN)
Setting 3: AG News subset (for Transformer) — 4-class text classification
"""

import ssl
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

DATA_ROOT = "/tmp/data"


def get_mnist(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = torchvision.datasets.MNIST(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_cifar10(batch_size=128):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform_test
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_synthetic_text(batch_size=128, num_train=10000, num_test=2000,
                       vocab_size=10000, seq_len=64, num_classes=4):
    """
    Synthetic text classification: class is determined by the *combination*
    of two marker tokens appearing in the sequence — a pattern that requires
    attending to multiple positions.

    Made harder: markers are drawn from a shared pool so single-token
    detection is insufficient; the class depends on *which pair* co-occurs.
    """
    rng = np.random.default_rng(42)

    # Shared marker pool: 40 special tokens
    n_markers = 40
    marker_start = 2
    marker_pool = list(range(marker_start, marker_start + n_markers))
    noise_start = marker_start + n_markers

    # Class c is identified by the co-occurrence of markers from group c
    # Each group has 10 markers; class is signalled by having exactly 1 marker
    # from each of two specific groups.
    group_size = n_markers // num_classes
    groups = [marker_pool[c * group_size:(c + 1) * group_size] for c in range(num_classes)]
    # Class c uses markers from groups c and (c+1) % num_classes
    class_groups = [(c, (c + 1) % num_classes) for c in range(num_classes)]

    def make_sample(label):
        seq = rng.integers(noise_start, vocab_size, size=seq_len)
        g1, g2 = class_groups[label]
        pos = rng.choice(seq_len, size=2, replace=False)
        seq[pos[0]] = rng.choice(groups[g1])
        seq[pos[1]] = rng.choice(groups[g2])
        # Add a small amount of distractor markers from other groups
        n_distractors = rng.integers(0, 3)
        other_groups = [g for g in range(num_classes) if g not in (g1, g2)]
        for _ in range(n_distractors):
            dpos = rng.integers(0, seq_len)
            dgroup = rng.choice(other_groups)
            seq[dpos] = rng.choice(groups[dgroup])
        return seq

    def make_dataset(n):
        labels = rng.integers(0, num_classes, size=n)
        seqs = np.stack([make_sample(int(l)) for l in labels])
        return TensorDataset(
            torch.tensor(seqs, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    train_ds = make_dataset(num_train)
    test_ds = make_dataset(num_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def randomize_labels(loader, num_classes=10, seed=0):
    """Return a new DataLoader with randomly shuffled labels (Zhang et al. control)."""
    rng = np.random.default_rng(seed)
    dataset = loader.dataset

    if isinstance(dataset, TensorDataset):
        x = dataset.tensors[0]
        new_labels = torch.tensor(
            rng.integers(0, num_classes, size=len(x)), dtype=torch.long
        )
        new_ds = TensorDataset(x, new_labels)
    elif isinstance(dataset, Subset):
        indices = dataset.indices
        new_labels = torch.tensor(
            rng.integers(0, num_classes, size=len(indices)), dtype=torch.long
        )
        xs = torch.stack([dataset[i][0] for i in range(len(dataset))])
        new_ds = TensorDataset(xs, new_labels)
    else:
        xs = []
        for i in range(len(dataset)):
            xs.append(dataset[i][0])
        xs = torch.stack(xs)
        new_labels = torch.tensor(
            rng.integers(0, num_classes, size=len(dataset)), dtype=torch.long
        )
        new_ds = TensorDataset(xs, new_labels)

    return DataLoader(new_ds, batch_size=loader.batch_size, shuffle=True)

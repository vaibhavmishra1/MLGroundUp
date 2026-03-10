import torch
from torchvision import datasets
import numpy as np

class MNISTCustom(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)

    def __len__(self):
        return len(self.data)  # self.data is the tensor set by parent MNIST class

    def __getitem__(self, idx):
        img = self.data[idx]               # tensor of shape (28, 28), uint8
        img = img.numpy().astype(np.float32)
        img = img.flatten()                # shape (784,)
        img = img / 255.0                  # normalize to [0, 1]

        label = int(self.targets[idx])
        return img, label

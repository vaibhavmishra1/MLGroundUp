from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch

class SudokuExtremeDataset(Dataset):
    def __init__(self, data_path: str, split: str, subsample = 1000):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.subsample = subsample
        self.data, self.data_len = self.load_dataset(data_path, split)
    
    def load_dataset(self, data_path: str, split: str):
        input_path = os.path.join(data_path, split, "all__inputs.npy")
        label_path = os.path.join(data_path, split, "all__labels.npy")
        data = {
            'inputs': np.load(input_path).astype(np.int64),
            'labels': np.load(label_path).astype(np.int64)
        }
        
        return data, len(data['inputs'])
    
    def __len__(self):
        return self.subsample

    def __getitem__(self, index):
        input = self.data['inputs'][index]
        label = self.data['labels'][index]
        return input, label


def main():
    dataset = SudokuExtremeDataset(data_path="data/sudoku-extreme-1k-aug-1000", split="train")
    train_loader = DataLoader(dataset, batch_size = 1, shuffle=True)
    
    for batch in train_loader:
        inputs, labels = batch
        print(inputs.shape, labels.shape)
        print(inputs, labels)
        break

if __name__ == "__main__":
    main()
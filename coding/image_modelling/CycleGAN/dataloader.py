import os
import random
import zipfile
import urllib.request
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
# Standard CycleGAN transforms: resize to 256x256, normalize to [-1, 1]
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),                          # [0, 255] -> [0.0, 1.0], shape (C, H, W)
    transforms.Normalize(mean=[0.5, 0.5, 0.5],     # [0.0, 1.0] -> [-1.0, 1.0]
                         std=[0.5, 0.5, 0.5])
])

DATA_URL = "https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip"
DATA_DIR = "data/horse2zebra"

def download_horse2zebra(data_dir=DATA_DIR):
    """Download and extract the horse2zebra dataset from the official CycleGAN source."""
    if os.path.exists(data_dir):
        return  # already downloaded
    os.makedirs("data", exist_ok=True)
    zip_path = "data/horse2zebra.zip"
    print(f"Downloading horse2zebra dataset...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as f:
        f.extractall("data")
    os.remove(zip_path)
    print(f"Dataset ready at {data_dir}")

def load_images(folder):
    """Load all image paths from a folder."""
    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    ]

class Horse2ZebraDataset(Dataset):
    """
    Loads the horse2zebra dataset downloaded from efrosgans.eecs.berkeley.edu.
    Returns unpaired (horse_img, zebra_img) tensors of shape (3, 256, 256) in [-1, 1].
    split: 'train' or 'test'
    """
    def __init__(self, split='train'):
        #download_horse2zebra()
        self.horses = load_images(os.path.join(DATA_DIR, f'train{"A" if split == "train" else "A"}'))
        self.zebras = load_images(os.path.join(DATA_DIR, f'train{"B" if split == "train" else "B"}'))
        if split == 'test':
            self.horses = load_images(os.path.join(DATA_DIR, 'testA'))
            self.zebras = load_images(os.path.join(DATA_DIR, 'testB'))

    def __len__(self):
        return max(len(self.horses), len(self.zebras))

    def __getitem__(self, idx):
        horse_img = Image.open(self.horses[idx % len(self.horses)]).convert('RGB')
        zebra_img = Image.open(self.zebras[random.randint(0, len(self.zebras) - 1)]).convert('RGB')
        return transform(horse_img), transform(zebra_img)


if __name__ == '__main__':
    
    dataset = Horse2ZebraDataset(split='train')
    print(f"Horses: {len(dataset.horses)}, Zebras: {len(dataset.zebras)}")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    horses, zebras = next(iter(loader))
    print(horses.shape)
    print(zebras.shape)
    print(f"Batch shapes — horses: {horses.shape}, zebras: {zebras.shape}")

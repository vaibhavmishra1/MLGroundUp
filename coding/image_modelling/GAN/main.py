import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from dataloader import MNISTCustom
from torch.utils.data import DataLoader
from model import Discriminator, Generator

img_dim = 784
latent_dim = 50   # small latent space — G maps 100-dim noise → 784-dim images

def sample_noise(batch_size):
    return torch.randn(batch_size, latent_dim)

def main():
    dataset = MNISTCustom(root='data', train=True, download=True)
    batch_size = 512
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    discriminator = Discriminator(img_dim)
    generator = Generator(latent_dim, img_dim)

    # BCELoss is numerically stable — no NaN from manual log(0)
    criterion = nn.BCELoss()

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))

    real_label = 0.9   # label smoothing for real
    fake_label = 0.0

    dataloader_iter = iter(dataloader)

    for j in range(10000):
        # ====== Train Discriminator ======
        try:
            inputs, _ = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            inputs, _ = next(dataloader_iter)

        inputs = inputs.float()
        current_batch_size = inputs.size(0)

        # Real images
        optimizer_D.zero_grad()
        labels_real = torch.full((current_batch_size, 1), real_label)
        D_out_real = discriminator(inputs)
        D_loss_real = criterion(D_out_real, labels_real)

        # Fake images
        z = sample_noise(current_batch_size)
        fake_images = generator(z).detach()       # detach: don't update G here
        labels_fake = torch.full((current_batch_size, 1), fake_label)
        D_out_fake = discriminator(fake_images)
        D_loss_fake = criterion(D_out_fake, labels_fake)

        D_loss = D_loss_real + D_loss_fake
        D_loss.backward()
        optimizer_D.step()

        # ====== Train Generator ======
        optimizer_G.zero_grad()
        z = sample_noise(current_batch_size)
        fake_images = generator(z)
        labels_gen = torch.full((current_batch_size, 1), 1.0)  # G wants D to think these are real
        D_out_gen = discriminator(fake_images)
        G_loss = criterion(D_out_gen, labels_gen)
        G_loss.backward()
        optimizer_G.step()

        # ====== Monitoring ======
        if (j + 1) % 100 == 0:
            with torch.no_grad():
                D_real_mean = D_out_real.mean().item()
                D_fake_mean = D_out_fake.mean().item()
            print(f"iter {j+1}: D_loss={D_loss.item():.3f} | G_loss={G_loss.item():.3f} | D(real)={D_real_mean:.3f} | D(fake)={D_fake_mean:.3f}")

    # ====== Generate 10 samples after training ======
    os.makedirs("results", exist_ok=True)
    generator.eval()
    with torch.no_grad():
        z = sample_noise(10)
        samples = generator(z).numpy()
        for i, sample in enumerate(samples):
            img_array = (sample.reshape(28, 28) * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            img.save(f"results/sample_{i+1}.png")
            print(f"Saved results/sample_{i+1}.png")

main()

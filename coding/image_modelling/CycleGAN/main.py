import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from model import Discriminator, Generator
from dataloader import Horse2ZebraDataset

lambda_cycle = 10

def cycle_loss(real_images, gen_images):
    return torch.mean(torch.abs(real_images - gen_images))

def save_image(tensor, path):
    """Convert tensor from [-1, 1] to PIL Image and save."""
    # tensor shape: (C, H, W) or (B, C, H, W)
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image if batch
    
    # Convert from [-1, 1] to [0, 1]
    img = (tensor * 0.5 + 0.5).clamp(0, 1)
    
    # Convert to numpy and scale to [0, 255]
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # Save as PIL Image
    Image.fromarray(img_np, mode='RGB').save(path)
def main():
    dataset = Horse2ZebraDataset(split='train')
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G_horse = Generator(n_residual_blocks=9)  # Horse -> Zebra
    G_zebra = Generator(n_residual_blocks=9)  # Zebra -> Horse
    D_horse = Discriminator()                  # Discriminator for horses
    D_zebra = Discriminator()  
    # BCEWithLogitsLoss handles raw logits (no sigmoid needed) - standard for PatchGAN
    criterion = nn.BCEWithLogitsLoss()

    optimizer_D_horse = torch.optim.Adam(D_horse.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_zebra = torch.optim.Adam(D_zebra.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_G_horse = torch.optim.Adam(G_horse.parameters(), lr=0.0005, betas=(0.5, 0.999))
    optimizer_G_zebra = torch.optim.Adam(G_zebra.parameters(), lr=0.0005, betas=(0.5, 0.999))

    real_label = 0.9   # label smoothing for real
    fake_label = 0.0

    # Create outputs folder
    os.makedirs("outputs", exist_ok=True)

    dataloader_iter = iter(dataloader)

    for j in range(1000):
        # ====== Train Discriminator ======
        try:
            inputs_horse, inputs_zebra = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            inputs_horse, inputs_zebra = next(dataloader_iter)

        inputs_horse, inputs_zebra = inputs_horse.float(), inputs_zebra.float()
        current_batch_size = inputs_horse.size(0)
        ########################################################
        ## horse discriminator 
        ########################################################
        optimizer_D_horse.zero_grad()
        D_out_real_horse = D_horse(inputs_horse)  # (B, 1, 16, 16) patch output
        labels_real_horse = torch.full_like(D_out_real_horse, real_label)  # Match patch shape
        D_loss_real_horse = criterion(D_out_real_horse, labels_real_horse)
        # Fake images
        fake_images_horse = G_zebra(inputs_zebra).detach()       # detach: don't update G here
        D_out_fake_horse = D_horse(fake_images_horse)  # (B, 1, 16, 16) patch output
        labels_fake_horse = torch.full_like(D_out_fake_horse, fake_label)  # Match patch shape
        D_loss_fake_horse = criterion(D_out_fake_horse, labels_fake_horse)

        D_loss_horse = D_loss_real_horse + D_loss_fake_horse
        D_loss_horse.backward()
        optimizer_D_horse.step()
        
        ########################################################
        ## zebra discriminator 
        ########################################################
        optimizer_D_zebra.zero_grad()
        D_out_real_zebra = D_zebra(inputs_zebra)  # (B, 1, 16, 16) patch output
        labels_real_zebra = torch.full_like(D_out_real_zebra, real_label)  # Match patch shape
        D_loss_real_zebra = criterion(D_out_real_zebra, labels_real_zebra)
        # Fake image
        fake_images_zebra = G_horse(inputs_horse).detach()       # detach: don't update G here
        D_out_fake_zebra = D_zebra(fake_images_zebra)  # (B, 1, 16, 16) patch output
        labels_fake_zebra = torch.full_like(D_out_fake_zebra, fake_label)  # Match patch shape
        D_loss_fake_zebra = criterion(D_out_fake_zebra, labels_fake_zebra)
        D_loss_zebra = D_loss_real_zebra + D_loss_fake_zebra
        D_loss_zebra.backward()
        optimizer_D_zebra.step()
        
        # ====== Train Generator ======
        optimizer_G_horse.zero_grad()
        optimizer_G_zebra.zero_grad()
        
        fake_zebra = G_horse(inputs_horse)
        gen_horse = G_zebra(fake_zebra)

        fake_horse = G_zebra(inputs_zebra)
        gen_zebra = G_horse(fake_horse)

        D_out_gen_zebra = D_zebra(fake_zebra)  # (B, 1, 16, 16) patch output
        D_out_gen_horse = D_horse(fake_horse)  # (B, 1, 16, 16) patch output

        labels_gen_horse = torch.full_like(D_out_gen_horse, 1.0)  # G wants D to think these are real (match patch shape)
        labels_gen_zebra = torch.full_like(D_out_gen_zebra, 1.0)  # G wants D to think these are real (match patch shape)

        loss_horse_gen = criterion(D_out_gen_horse, labels_gen_horse)
        loss_zebra_gen = criterion(D_out_gen_zebra, labels_gen_zebra)
        
        cycle_loss_horse = cycle_loss(inputs_horse, gen_horse)
        cycle_loss_zebra = cycle_loss(inputs_zebra, gen_zebra)

        total_loss = loss_horse_gen + loss_zebra_gen + lambda_cycle * cycle_loss_horse + lambda_cycle * cycle_loss_zebra
        total_loss.backward()
        optimizer_G_horse.step()
        optimizer_G_zebra.step()

        # ====== Monitoring ======
        if (j + 1) % 1 == 0:
            with torch.no_grad():
                # Convert logits to probabilities for easier interpretation
                D_real_prob_horse = torch.sigmoid(D_out_real_horse).mean().item()
                D_fake_prob_horse = torch.sigmoid(D_out_fake_horse).mean().item()
                D_real_prob_zebra = torch.sigmoid(D_out_real_zebra).mean().item()
                D_fake_prob_zebra = torch.sigmoid(D_out_fake_zebra).mean().item()
            print(f"iter {j+1}: D_loss_horse={D_loss_horse.item():.3f} | D_loss_zebra={D_loss_zebra.item():.3f} | G_loss_horse={loss_horse_gen.item():.3f} | G_loss_zebra={loss_zebra_gen.item():.3f} | cycle_loss_horse={cycle_loss_horse.item():.3f} | cycle_loss_zebra={cycle_loss_zebra.item():.3f} | D(real_horse)={D_real_prob_horse:.3f} | D(fake_horse)={D_fake_prob_horse:.3f} | D(real_zebra)={D_real_prob_zebra:.3f} | D(fake_zebra)={D_fake_prob_zebra:.3f}")

        # ====== Save sample images periodically ======
        if (j + 1) % 10 == 0:
            G_horse.eval()
            G_zebra.eval()
            with torch.no_grad():
                # Get a batch for visualization
                try:
                    vis_horse, vis_zebra = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    vis_horse, vis_zebra = next(dataloader_iter)
                
                vis_horse, vis_zebra = vis_horse.float(), vis_zebra.float()
                vis_batch_size = vis_horse.size(0)
                
                # Generate samples
                fake_zebra_samples = G_horse(vis_horse)
                fake_horse_samples = G_zebra(vis_zebra)
                
                # Save original and generated images
                for idx in range(min(4, vis_batch_size)):
                    save_image(vis_horse[idx], f"outputs/iter_{j+1}_original_horse_{idx}.png")
                    save_image(fake_zebra_samples[idx], f"outputs/iter_{j+1}_generated_zebra_{idx}.png")
                    save_image(vis_zebra[idx], f"outputs/iter_{j+1}_original_zebra_{idx}.png")
                    save_image(fake_horse_samples[idx], f"outputs/iter_{j+1}_generated_horse_{idx}.png")
                
                print(f"Saved sample images to outputs/ folder at iteration {j+1}")
            G_horse.train()
            G_zebra.train()

    # ====== Save final results after training ======
    print("\nTraining complete! Saving final results...")
    G_horse.eval()
    G_zebra.eval()
    with torch.no_grad():
        try:
            final_horse, final_zebra = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            final_horse, final_zebra = next(dataloader_iter)
        
        final_horse, final_zebra = final_horse.float(), final_zebra.float()
        
        # Generate final samples
        final_fake_zebra = G_horse(final_horse)
        final_fake_horse = G_zebra(final_zebra)
        
        # Save final results
        for idx in range(min(10, final_horse.size(0))):
            save_image(final_horse[idx], f"outputs/final_original_horse_{idx}.png")
            save_image(final_fake_zebra[idx], f"outputs/final_generated_zebra_{idx}.png")
            save_image(final_zebra[idx], f"outputs/final_original_zebra_{idx}.png")
            save_image(final_fake_horse[idx], f"outputs/final_generated_horse_{idx}.png")
    
    print("Final results saved to outputs/ folder!")

    
main()

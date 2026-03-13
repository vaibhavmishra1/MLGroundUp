"""
train.py — Train DDPM on MNIST
================================
Trains the U-Net noise prediction network on MNIST digits.

Dataset:
  - MNIST 28×28 greyscale images, padded/resized to 32×32
  - Normalised to [−1, 1]  (so that x₀ and x_T are on compatible scales)

Training loop:
  1. For each batch x₀, sample random timesteps t ~ Uniform{0, …, T−1}
  2. Compute xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε  via closed-form forward process
  3. Predict noise ε̂θ(xₜ, t) with the U-Net
  4. Gradient descent on MSE loss  ‖ε − ε̂θ‖²

Usage:
  python train.py

  Checkpoints are saved to ./checkpoints/ every SAVE_EVERY epochs.
  A training-loss curve is saved at the end.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")   # headless backend (no display required)
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import UNet
from diffusion import DDPM


# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
T           = 1000      # Number of diffusion timesteps
BETA_START  = 1e-4      # Start of linear noise schedule
BETA_END    = 0.02      # End of linear noise schedule

IMG_SIZE    = 32        # Spatial resolution (MNIST 28×28 → padded to 32×32)
CHANNELS    = 1         # Greyscale
BASE_CH     = 32        # U-Net base channel width
TIME_EMB_DIM = 128      # Dimensionality of the time embedding

BATCH_SIZE  = 128
EPOCHS      = 100
LR          = 2e-4      # AdamW learning rate
GRAD_CLIP   = 1.0       # Gradient clipping norm

SAVE_EVERY  = 10        # Save checkpoint every N epochs
SAVE_DIR    = "checkpoints"

# Device selection: CUDA → MPS (Apple Silicon) → CPU
DEVICE = (
    "cuda"  if torch.cuda.is_available() else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
def get_dataloader() -> DataLoader:
    """
    Load MNIST, resize to IMG_SIZE×IMG_SIZE, normalise to [−1, 1].

    Normalisation to [−1, 1]:
      - The noise ε added during the forward process comes from N(0, I)
      - At t=T, xT ≈ N(0, I) regardless of x₀, because ᾱT ≈ 0
      - To be on the same scale, x₀ should also live in [−1, 1]
        (std ≈ 0.3 for MNIST, comfortably within this range)
    """
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # [0,1] → [−1,1]
    ])

    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE != "cpu"),
        drop_last=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Model & diffusion schedule ──────────────────────────────────────
    model = UNet(
        in_channels=CHANNELS,
        base_ch=BASE_CH,
        time_emb_dim=TIME_EMB_DIM,
    ).to(DEVICE)

    diffusion = DDPM(T=T, beta_start=BETA_START, beta_end=BETA_END).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model  : U-Net  ({n_params:,} parameters)")
    print(f"Device : {DEVICE}")
    print(f"Epochs : {EPOCHS}   |   Batch: {BATCH_SIZE}   |   LR: {LR}")

    # ── Optimiser ────────────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Cosine annealing: smoothly decays LR to near zero by end of training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Data ──────────────────────────────────────────────────────────────
    loader = get_dataloader()
    print(f"Dataset: MNIST  ({len(loader.dataset):,} samples, {len(loader)} batches/epoch)\n")

    # ── Training loop ─────────────────────────────────────────────────────
    all_losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch:>3}/{EPOCHS}", leave=False)
        for images, _ in pbar:
            images = images.to(DEVICE)

            # ── Core DDPM training step ─────────────────────────────────
            # 1. Sample random t ∈ {0, …, T−1} for each sample in batch
            # 2. Add noise: xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε
            # 3. Predict noise ε̂θ(xₜ, t)
            # 4. MSE(ε, ε̂θ)
            optimizer.zero_grad()
            loss = diffusion.compute_loss(model, images)
            loss.backward()

            # Gradient clipping prevents instability from exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        all_losses.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:>3}/{EPOCHS}  |  Loss: {avg_loss:.4f}  |  LR: {current_lr:.6f}")

        # ── Save checkpoint ───────────────────────────────────────────────
        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"ddpm_epoch_{epoch:04d}.pt")
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss":                 avg_loss,
                    "hparams": {
                        "T": T, "base_ch": BASE_CH,
                        "time_emb_dim": TIME_EMB_DIM, "img_size": IMG_SIZE,
                    },
                },
                ckpt_path,
            )
            print(f"  → Checkpoint saved: {ckpt_path}")

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(SAVE_DIR, "ddpm_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved: {final_path}")

    # ── Loss curve ────────────────────────────────────────────────────────
    _save_loss_curve(all_losses)

    print("\nTraining complete!")
    return model, all_losses


def _save_loss_curve(losses: list[float]):
    path = os.path.join(SAVE_DIR, "training_loss.png")
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(losses) + 1), losses, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("DDPM Training Loss on MNIST")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Loss curve saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()

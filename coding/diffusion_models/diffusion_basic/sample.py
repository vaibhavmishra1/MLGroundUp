"""
sample.py — Generate Samples from a Trained DDPM
===================================================
Loads a trained U-Net checkpoint and runs the full reverse diffusion process
to generate new MNIST digit images.

Two visualisations are produced:
  1. A grid of 16 generated images          → samples/generated_samples.png
  2. A denoising trajectory (every 100 steps) → samples/denoising_trajectory.png
     Shows how xT (pure noise) is gradually transformed into x₀ (a digit)

Usage:
  python sample.py                           # uses default checkpoint
  python sample.py --checkpoint path/to.pt  # custom checkpoint
  python sample.py --n_samples 64           # generate more samples
"""

import os
import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import UNet
from diffusion import DDPM


# ─────────────────────────────────────────────────────────────────────────────
# Config (must match train.py)
# ─────────────────────────────────────────────────────────────────────────────
T            = 1000
IMG_SIZE     = 32
CHANNELS     = 1
BASE_CH      = 32
TIME_EMB_DIM = 128

DEVICE = (
    "cuda"  if torch.cuda.is_available() else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> UNet:
    """Load U-Net weights from a checkpoint file."""
    model = UNet(
        in_channels=CHANNELS,
        base_ch=BASE_CH,
        time_emb_dim=TIME_EMB_DIM,
    ).to(DEVICE)

    state = torch.load(checkpoint_path, map_location=DEVICE)

    # Support both raw state_dict and full checkpoint dict
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        epoch = state.get("epoch", "?")
        print(f"Loaded checkpoint from epoch {epoch}  ({checkpoint_path})")
    else:
        model.load_state_dict(state)
        print(f"Loaded model weights from {checkpoint_path}")

    model.eval()
    return model


def denorm(x: torch.Tensor) -> np.ndarray:
    """Convert from [−1, 1] tensor to [0, 1] numpy array for plotting."""
    return ((x.clamp(-1, 1) + 1) / 2).cpu().numpy()


def save_image_grid(
    images: np.ndarray,
    path: str,
    title: str = "",
    ncols: int = 4,
):
    """
    Save a grid of greyscale images.

    Args:
        images: (N, 1, H, W) numpy array in [0, 1]
        path:   output file path
        title:  figure title
        ncols:  number of columns in the grid
    """
    n = len(images)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_samples(model: UNet, n_samples: int, save_dir: str):
    """Generate n_samples images and save a grid."""
    os.makedirs(save_dir, exist_ok=True)
    diffusion = DDPM(T=T).to(DEVICE)

    print(f"Generating {n_samples} samples on {DEVICE} …")
    samples = diffusion.p_sample_loop(
        model=model,
        shape=(n_samples, CHANNELS, IMG_SIZE, IMG_SIZE),
        device=DEVICE,
    )

    images = denorm(samples)  # (N, 1, H, W) in [0,1]
    path = os.path.join(save_dir, "generated_samples.png")
    save_image_grid(images, path, title=f"DDPM Generated Samples (MNIST) — {n_samples} images")
    return images


def generate_denoising_trajectory(model: UNet, save_dir: str):
    """
    Visualise the denoising trajectory for a single sample.

    Shows x at every 100 timesteps as the model transforms
    pure Gaussian noise xT into a digit x₀.
    This is one of the most intuitive demonstrations of how
    diffusion models work.
    """
    os.makedirs(save_dir, exist_ok=True)
    diffusion = DDPM(T=T).to(DEVICE)

    print("Generating denoising trajectory …")
    trajectory = diffusion.p_sample_loop(
        model=model,
        shape=(1, CHANNELS, IMG_SIZE, IMG_SIZE),
        device=DEVICE,
        return_trajectory=True,
    )

    # trajectory is a list of tensors saved every 100 steps (reversed)
    # Index 0 = xT (pure noise), last = x₀ (generated image)
    # Because we iterate reversed(range(T)), step 0 is t=999, step 1 is t=899, ...
    n_frames = len(trajectory)
    images = [denorm(frame) for frame in trajectory]   # list of (1,1,H,W) arrays

    # Timestep labels (reversed: 999, 899, …, 99, 0 (approx))
    step_labels = [f"t={999 - i * 100}" for i in range(n_frames)]
    if len(step_labels) > 0:
        step_labels[-1] = "t=0  (x₀)"

    fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 2.5, 3))
    if n_frames == 1:
        axes = [axes]

    for ax, img, label in zip(axes, images, step_labels):
        ax.imshow(img[0, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    fig.suptitle("Reverse Diffusion Denoising Process", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "denoising_trajectory.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def visualise_forward_process(save_dir: str):
    """
    Visualise the forward (noising) process on a real MNIST image.
    Shows x₀ → x₂₅₀ → x₅₀₀ → x₇₅₀ → x₉₉₉ for a single digit.
    Requires torchvision.
    """
    os.makedirs(save_dir, exist_ok=True)
    try:
        from torchvision import datasets, transforms
    except ImportError:
        print("torchvision not available — skipping forward process visualisation.")
        return

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    x0, label = dataset[0]                          # single image (1,32,32)
    x0 = x0.unsqueeze(0).to(DEVICE)                 # (1,1,32,32)

    diffusion = DDPM(T=T).to(DEVICE)

    timesteps = [0, 100, 250, 500, 750, 999]
    titles    = [f"x₀  (digit {label})"] + [f"x_t = {t}" for t in timesteps[1:]]
    titles[-1] += "  ≈ N(0,I)"

    fig, axes = plt.subplots(1, len(timesteps), figsize=(len(timesteps) * 2.5, 3))

    for ax, t_val, title in zip(axes, timesteps, titles):
        t_tensor = torch.tensor([t_val], device=DEVICE)
        x_t, _   = diffusion.q_sample(x0, t_tensor)
        ax.imshow(denorm(x_t)[0, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle("Forward Diffusion Process (Adding Noise)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "forward_process.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Sample from a trained DDPM")
    p.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/ddpm_final.pt",
        help="Path to model checkpoint (default: checkpoints/ddpm_final.pt)",
    )
    p.add_argument(
        "--n_samples", type=int, default=16,
        help="Number of samples to generate (default: 16)",
    )
    p.add_argument(
        "--save_dir", type=str, default="samples",
        help="Directory to save output images (default: samples/)",
    )
    p.add_argument(
        "--forward_only", action="store_true",
        help="Only visualise the forward (noising) process; skip generation",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Always visualise the forward process (no model needed)
    visualise_forward_process(args.save_dir)

    if not args.forward_only:
        if not os.path.exists(args.checkpoint):
            print(
                f"\nCheckpoint not found: {args.checkpoint}\n"
                "Run `python train.py` first to train the model.\n"
                "You can still view the forward_process.png visualisation."
            )
        else:
            model = load_model(args.checkpoint)
            generate_samples(model, args.n_samples, args.save_dir)
            generate_denoising_trajectory(model, args.save_dir)

    print(f"\nAll outputs saved to ./{args.save_dir}/")

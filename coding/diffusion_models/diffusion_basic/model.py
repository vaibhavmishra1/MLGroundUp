"""
model.py — U-Net Noise Prediction Network for DDPM
====================================================
The backbone of a diffusion model is a neural network ε_θ(x_t, t) that takes
a noisy image x_t and timestep t, and predicts the noise ε that was added.

Architecture: U-Net
  - Sinusoidal time embeddings encode the noise level t
  - Encoder path: 3 stages of ResBlocks + strided convolution downsampling
  - Bottleneck: ResBlock + Self-Attention + ResBlock
  - Decoder path: 3 stages of ResBlocks + transposed convolution upsampling
  - Skip connections from encoder to decoder (preserving spatial detail)
  - Output: 1×1 conv → predicted noise (same shape as input)

All residual blocks are conditioned on the time embedding, so the network
"knows" what noise level it is operating at.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Time Embedding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encode a scalar timestep t into a d-dimensional vector using sinusoidal
    positional encodings (Vaswani et al., 2017 "Attention Is All You Need").

    For position t, the i-th component is:
        PE(t, 2i)   = sin( t / 10000^(2i/d) )
        PE(t, 2i+1) = cos( t / 10000^(2i/d) )

    This gives the network a continuous, smooth encoding of the noise level
    that generalises to unseen timesteps and captures both fine-grained and
    coarse-grained distinctions between timesteps.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer timestep indices
        Returns:
            embedding: (B, dim) sinusoidal embeddings
        """
        device = t.device
        half_dim = self.dim // 2

        # Frequency bands: 10000^(2i/d) for i = 0 … half_dim-1
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / (half_dim - 1)
        )

        # Outer product: (B, half_dim)
        args = t[:, None].float() * freqs[None, :]

        # Interleave sin and cos → (B, dim)
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Residual block with time conditioning via addition.

    Data flow:
        x  →  GroupNorm → SiLU → Conv2d
           →  + time_projection(t_emb)        ← inject noise level
           →  GroupNorm → SiLU → Conv2d
           →  + residual_connection(x)

    GroupNorm is used instead of BatchNorm because it is stable at small
    batch sizes (which diffusion models often use) and independent of batch
    statistics at inference time.

    SiLU (Sigmoid Linear Unit, aka Swish) is used as the activation function
    because it is smooth and has been empirically shown to work well in
    diffusion models.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Project time embedding to match number of output channels
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.act = nn.SiLU()

        # Match channels for residual if needed
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, in_channels, H, W)
            t_emb: (B, time_emb_dim) time embedding
        Returns:
            (B, out_channels, H, W)
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding: (B, out_channels) → (B, out_channels, 1, 1)
        h = h + self.act(self.time_proj(t_emb))[:, :, None, None]

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """
    Self-attention over spatial positions.

    At the bottleneck (smallest spatial resolution), the receptive field of
    convolutions is limited. Self-attention lets every spatial location attend
    to every other location, capturing long-range dependencies.

    Data flow:
        x → GroupNorm → reshape to (B, H*W, C)
          → MultiheadAttention(Q=K=V)
          → reshape to (B, C, H, W)
          → + residual connection

    This is the same attention used in Transformers, applied here to image
    feature maps treated as sequences of spatial patches.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Normalise and flatten spatial dims → sequence of tokens
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)

        # Self-attention (Q = K = V)
        h, _ = self.attn(h, h, h)

        # Restore spatial shape and add residual
        return x + h.transpose(1, 2).view(B, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net noise prediction network: ε_θ(x_t, t).

    Given a noisy image x_t and the current timestep t, predicts the noise
    ε ∼ N(0, I) that was added to the clean image x_0.

    Input shape:  (B, in_channels, H, W)   — noisy image
    Output shape: (B, in_channels, H, W)   — predicted noise

    Spatial flow (for 32×32 input, base_ch=32):
    ┌─────────────────────────────────────────────────────────────────┐
    │ init_conv      (1 ,32,32) → (32,32,32)                          │
    │                                                                 │
    │ enc1 [ResBlock]  (32,32,32) → skip1=(32,32,32)                  │
    │ down1 [stride2]  (32,32,32) → (32,16,16)                        │
    │                                                                 │
    │ enc2 [ResBlock]  (32,16,16) → skip2=(64,16,16)                  │
    │ down2 [stride2]  (64,16,16) → (64, 8, 8)                        │
    │                                                                 │
    │ enc3 [ResBlock]  (64, 8, 8) → skip3=(128, 8, 8)                 │
    │ down3 [stride2] (128, 8, 8) → (128, 4, 4)                       │
    │                                                                 │
    │ Bottleneck: ResBlock → SelfAttn → ResBlock  (128,4,4)           │
    │                                                                 │
    │ up3   [ConvTranspose] (128, 4, 4) → (128, 8, 8)                 │
    │ cat(up3, skip3)       → (256, 8, 8)                             │
    │ dec3  [ResBlock]      (256, 8, 8) → ( 64, 8, 8)                 │
    │                                                                 │
    │ up2   [ConvTranspose] ( 64, 8, 8) → ( 64,16,16)                 │
    │ cat(up2, skip2)       → (128,16,16)                             │
    │ dec2  [ResBlock]      (128,16,16) → ( 32,16,16)                 │
    │                                                                 │
    │ up1   [ConvTranspose] ( 32,16,16) → ( 32,32,32)                 │
    │ cat(up1, skip1)       → ( 64,32,32)                             │
    │ dec1  [ResBlock]      ( 64,32,32) → ( 32,32,32)                 │
    │                                                                 │
    │ out_conv [1×1 Conv]   ( 32,32,32) → (  1,32,32) ← predicted ε  │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_ch: int = 32,
        time_emb_dim: int = 128,
    ):
        super().__init__()

        c = base_ch  # shorthand

        # ── Time embedding ────────────────────────────────────────────────
        # Sinusoidal → linear → SiLU → linear (doubles then compresses)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # ── Initial projection ────────────────────────────────────────────
        self.init_conv = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)

        # ── Encoder ──────────────────────────────────────────────────────
        # enc1: c→c  (32×32) | skip1 has c channels
        self.enc1  = ResBlock(c,   c,   time_emb_dim)
        self.down1 = nn.Conv2d(c,   c,   kernel_size=4, stride=2, padding=1)

        # enc2: c→2c  (16×16) | skip2 has 2c channels
        self.enc2  = ResBlock(c,   c*2, time_emb_dim)
        self.down2 = nn.Conv2d(c*2, c*2, kernel_size=4, stride=2, padding=1)

        # enc3: 2c→4c  (8×8) | skip3 has 4c channels
        self.enc3  = ResBlock(c*2, c*4, time_emb_dim)
        self.down3 = nn.Conv2d(c*4, c*4, kernel_size=4, stride=2, padding=1)

        # ── Bottleneck (4×4) ──────────────────────────────────────────────
        self.mid1     = ResBlock(c*4, c*4, time_emb_dim)
        self.mid_attn = AttentionBlock(c*4)
        self.mid2     = ResBlock(c*4, c*4, time_emb_dim)

        # ── Decoder ──────────────────────────────────────────────────────
        # up3 + cat(skip3=4c) → dec3: 8c→2c  (8×8)
        self.up3  = nn.ConvTranspose2d(c*4, c*4, kernel_size=4, stride=2, padding=1)
        self.dec3 = ResBlock(c*4 + c*4, c*2, time_emb_dim)

        # up2 + cat(skip2=2c) → dec2: 4c→c  (16×16)
        self.up2  = nn.ConvTranspose2d(c*2, c*2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock(c*2 + c*2, c,   time_emb_dim)

        # up1 + cat(skip1=c) → dec1: 2c→c  (32×32)
        self.up1  = nn.ConvTranspose2d(c,   c,   kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock(c + c,     c,   time_emb_dim)

        # ── Output ────────────────────────────────────────────────────────
        self.norm_out = nn.GroupNorm(8, c)
        self.act_out  = nn.SiLU()
        self.out_conv = nn.Conv2d(c, in_channels, kernel_size=1)

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        """Zero-initialise the output projection so training starts smoothly."""
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) — noisy image x_t
            t: (B,)                   — integer timestep indices
        Returns:
            (B, in_channels, H, W)    — predicted noise ε̂
        """
        # 1. Embed timestep t → dense vector
        t_emb = self.time_mlp(t)              # (B, time_emb_dim)

        # 2. Initial projection
        x = self.init_conv(x)                 # (B, c,   H,   W)

        # 3. Encoder (save skip connections)
        x1 = self.enc1(x,  t_emb)            # (B, c,   H,   W)  ← skip1
        x  = self.down1(x1)                   # (B, c,   H/2, W/2)

        x2 = self.enc2(x,  t_emb)            # (B, 2c,  H/2, W/2) ← skip2
        x  = self.down2(x2)                   # (B, 2c,  H/4, W/4)

        x3 = self.enc3(x,  t_emb)            # (B, 4c,  H/4, W/4) ← skip3
        x  = self.down3(x3)                   # (B, 4c,  H/8, W/8)

        # 4. Bottleneck
        x = self.mid1(x, t_emb)              # (B, 4c,  H/8, W/8)
        x = self.mid_attn(x)                  # self-attention
        x = self.mid2(x, t_emb)              # (B, 4c,  H/8, W/8)

        # 5. Decoder (concatenate skip connections)
        x = self.up3(x)                       # (B, 4c,  H/4, W/4)
        x = torch.cat([x, x3], dim=1)        # (B, 8c,  H/4, W/4)
        x = self.dec3(x, t_emb)              # (B, 2c,  H/4, W/4)

        x = self.up2(x)                       # (B, 2c,  H/2, W/2)
        x = torch.cat([x, x2], dim=1)        # (B, 4c,  H/2, W/2)
        x = self.dec2(x, t_emb)              # (B, c,   H/2, W/2)

        x = self.up1(x)                       # (B, c,   H,   W)
        x = torch.cat([x, x1], dim=1)        # (B, 2c,  H,   W)
        x = self.dec1(x, t_emb)              # (B, c,   H,   W)

        # 6. Output
        x = self.act_out(self.norm_out(x))
        return self.out_conv(x)               # (B, in_channels, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = UNet(in_channels=1, base_ch=32, time_emb_dim=128)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet parameters: {n_params:,}")

    # Dummy forward pass: batch of 4 noisy 32×32 grayscale images
    x = torch.randn(4, 1, 32, 32)
    t = torch.randint(0, 1000, (4,))
    out = model(x, t)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # Should be (4, 1, 32, 32)
    assert out.shape == x.shape, "Output shape mismatch!"
    print("Sanity check passed ✓")

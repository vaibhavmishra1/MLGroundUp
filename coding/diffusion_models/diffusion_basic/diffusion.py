"""
diffusion.py — DDPM Forward and Reverse Diffusion Process
===========================================================
Implements the core mathematics of Denoising Diffusion Probabilistic Models
(Ho et al., 2020, https://arxiv.org/abs/2006.11239).

────────────────────────────────────────────────────────────────────────────
THEORY RECAP
────────────────────────────────────────────────────────────────────────────

FORWARD PROCESS  q(x₁:T | x₀)
  Gradually corrupts x₀ with Gaussian noise over T steps.
  Each step:
      q(xₜ | xₜ₋₁) = N(xₜ; √(1−βₜ)·xₜ₋₁, βₜ·I)

  Using the "nice property" of Gaussians we can jump directly to step t:
      q(xₜ | x₀) = N(xₜ; √ᾱₜ·x₀, (1−ᾱₜ)·I)

  Sampling x_t in one shot:
      xₜ = √ᾱₜ · x₀  +  √(1−ᾱₜ) · ε,     ε ~ N(0, I)

  where:
      αₜ  = 1 − βₜ
      ᾱₜ  = ∏ₛ₌₁ᵗ αₛ   (cumulative product)

REVERSE PROCESS  pθ(x₀:T)
  Learned Gaussian approximation to the true reverse:
      pθ(xₜ₋₁ | xₜ) = N(xₜ₋₁; μθ(xₜ, t), σₜ²·I)

  The optimal mean (when predicting noise ε̂θ) is:
      μₜ = (1/√αₜ) · (xₜ − βₜ/√(1−ᾱₜ) · ε̂θ(xₜ, t))

  The variance is fixed to:
      σₜ² = β̃ₜ = (1−ᾱₜ₋₁)/(1−ᾱₜ) · βₜ    (posterior variance)

TRAINING OBJECTIVE
  Simplified noise-prediction loss (Eq. 14 in Ho et al.):
      L_simple = E_{t,x₀,ε} [ ‖ε − ε̂θ(√ᾱₜ·x₀ + √(1−ᾱₜ)·ε, t)‖² ]

SAMPLING ALGORITHM  (DDPM, Algorithm 2)
  1. Sample xT ~ N(0, I)
  2. For t = T, T−1, …, 1:
       z ~ N(0, I)  if t > 1,  else z = 0
       xₜ₋₁ = (1/√αₜ)·(xₜ − βₜ/√(1−ᾱₜ)·ε̂θ(xₜ, t)) + √β̃ₜ·z
  3. Return x₀
────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


class DDPM:
    """
    Denoising Diffusion Probabilistic Model utilities.

    Holds the noise schedule and provides:
        q_sample       — forward process (add noise to x₀ → xₜ)
        p_sample       — single reverse denoising step (xₜ → xₜ₋₁)
        p_sample_loop  — full generation loop (xT → x₀)
        compute_loss   — training loss
    """

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        """
        Args:
            T:          Number of diffusion timesteps.
            beta_start: Starting variance of the noise schedule.
            beta_end:   Ending variance of the noise schedule.
        """
        self.T = T

        # ── Linear noise schedule ────────────────────────────────────────
        # βₜ increases linearly from beta_start to beta_end
        betas = torch.linspace(beta_start, beta_end, T)        # (T,)

        # ── Derived quantities ───────────────────────────────────────────
        alphas      = 1.0 - betas                               # (T,) αₜ
        alpha_bars  = torch.cumprod(alphas, dim=0)              # (T,) ᾱₜ

        # ᾱₜ₋₁: shift alpha_bars right, pad the first element with 1.0
        # (ᾱ₀ is defined as 1 — before any noise is added)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)  # (T,)

        # Posterior variance  β̃ₜ = (1−ᾱₜ₋₁)/(1−ᾱₜ)·βₜ
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        # Register all tensors (will be moved to device by .to())
        self.betas                      = betas
        self.alphas                     = alphas
        self.alpha_bars                 = alpha_bars
        self.sqrt_alpha_bars            = alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars  = (1.0 - alpha_bars).sqrt()
        self.posterior_variance         = posterior_variance

    def to(self, device):
        """Move all schedule tensors to the given device."""
        self.betas                     = self.betas.to(device)
        self.alphas                    = self.alphas.to(device)
        self.alpha_bars                = self.alpha_bars.to(device)
        self.sqrt_alpha_bars           = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        self.posterior_variance        = self.posterior_variance.to(device)
        return self

    # ─────────────────────────────────────────────────────────────────────
    # Forward Process
    # ─────────────────────────────────────────────────────────────────────

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample xₜ from x₀ using the closed-form forward process.

            xₜ = √ᾱₜ · x₀  +  √(1−ᾱₜ) · ε,   ε ~ N(0, I)

        Args:
            x0:    (B, C, H, W) clean images, normalised to [−1, 1]
            t:     (B,) integer timestep indices in [0, T−1]
            noise: (B, C, H, W) optional pre-sampled noise; sampled if None

        Returns:
            x_t:   (B, C, H, W) noisy image at timestep t
            noise: (B, C, H, W) the noise that was added (needed for loss)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Gather schedule values and broadcast over spatial dims
        sqrt_ab     = self.sqrt_alpha_bars[t][:, None, None, None]           # √ᾱₜ
        sqrt_1m_ab  = self.sqrt_one_minus_alpha_bars[t][:, None, None, None] # √(1−ᾱₜ)

        x_t = sqrt_ab * x0 + sqrt_1m_ab * noise
        return x_t, noise

    # ─────────────────────────────────────────────────────────────────────
    # Reverse Process
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        Single reverse denoising step: sample xₜ₋₁ given xₜ.

            μₜ = (1/√αₜ) · (xₜ − βₜ/√(1−ᾱₜ) · ε̂θ(xₜ, t))
            xₜ₋₁ = μₜ + √β̃ₜ · z      (z=0 at t=0)

        Args:
            model: noise prediction network ε̂θ
            x_t:   (B, C, H, W) noisy image at timestep t
            t:     integer scalar timestep

        Returns:
            x_prev: (B, C, H, W) slightly denoised image at timestep t−1
        """
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)

        # ── 1. Predict noise ──────────────────────────────────────────────
        eps_pred = model(x_t, t_tensor)          # ε̂θ(xₜ, t)

        # ── 2. Compute denoised mean μₜ ──────────────────────────────────
        alpha_t      = self.alphas[t]             # αₜ
        alpha_bar_t  = self.alpha_bars[t]         # ᾱₜ
        beta_t       = self.betas[t]              # βₜ

        # μₜ = (1/√αₜ) · (xₜ − βₜ/√(1−ᾱₜ) · ε̂θ)
        coeff  = beta_t / (1.0 - alpha_bar_t).sqrt()
        mean   = (1.0 / alpha_t.sqrt()) * (x_t - coeff * eps_pred)

        # ── 3. Add stochastic noise (except at final step t=0) ────────────
        if t == 0:
            return mean
        else:
            z = torch.randn_like(x_t)
            sigma_t = self.posterior_variance[t].sqrt()   # √β̃ₜ
            return mean + sigma_t * z

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: torch.nn.Module,
        shape: tuple,
        device: torch.device | str,
        return_trajectory: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Full reverse diffusion: generate images from pure Gaussian noise.

            xT ~ N(0, I) → xT₋₁ → … → x₁ → x₀

        Args:
            model:             noise prediction network
            shape:             (B, C, H, W) shape of samples to generate
            device:            device to generate on
            return_trajectory: if True, return list of xₜ at every 100 steps
                               (useful for visualising the denoising process)

        Returns:
            x0: (B, C, H, W) generated images in [−1, 1]
            or list of intermediate images if return_trajectory=True
        """
        model.eval()
        x = torch.randn(shape, device=device)   # xT ~ N(0, I)

        trajectory = []

        for t in tqdm(reversed(range(self.T)), desc="Sampling", total=self.T, leave=False):
            x = self.p_sample(model, x, t)

            if return_trajectory and t % 100 == 0:
                trajectory.append(x.clone())

        if return_trajectory:
            return trajectory
        return x

    # ─────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the simplified DDPM training loss (Ho et al., Eq. 14):

            L_simple = E_{t,x₀,ε} [ ‖ε − ε̂θ(xₜ, t)‖² ]

        Training procedure:
            1. Sample random timesteps t ~ Uniform{0, …, T−1}
            2. Sample noise ε ~ N(0, I)
            3. Compute xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε  (forward process)
            4. Predict ε̂θ(xₜ, t) with the model
            5. Return MSE loss between true ε and predicted ε̂θ

        Args:
            model: noise prediction network ε̂θ(xₜ, t)
            x0:    (B, C, H, W) clean images in [−1, 1]

        Returns:
            scalar MSE loss
        """
        B      = x0.shape[0]
        device = x0.device

        # Step 1: random timesteps
        t = torch.randint(0, self.T, (B,), device=device)    # (B,)

        # Steps 2 & 3: sample xₜ via forward process
        x_t, noise = self.q_sample(x0, t)                    # both (B,C,H,W)

        # Step 4: predict noise
        noise_pred = model(x_t, t)                           # (B,C,H,W)

        # Step 5: MSE between actual and predicted noise
        return F.mse_loss(noise_pred, noise)


# ─────────────────────────────────────────────────────────────────────────────
# Quick demonstration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    ddpm = DDPM(T=1000)

    # Visualise how ᾱₜ decays → image progressively loses signal
    t_range = np.arange(1000)
    alpha_bars = ddpm.alpha_bars.numpy()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(t_range, alpha_bars)
    plt.xlabel("Timestep t")
    plt.ylabel("ᾱₜ  (signal strength)")
    plt.title("Cumulative product of (1−βₜ)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t_range, ddpm.sqrt_alpha_bars.numpy(), label="√ᾱₜ  (signal)")
    plt.plot(t_range, ddpm.sqrt_one_minus_alpha_bars.numpy(), label="√(1−ᾱₜ)  (noise)")
    plt.xlabel("Timestep t")
    plt.title("Signal vs. Noise Coefficients")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("noise_schedule.png", dpi=120)
    plt.show()
    print("Noise schedule plot saved to noise_schedule.png")

    # Verify forward process works on a dummy image
    x0   = torch.randn(4, 1, 32, 32)              # fake batch
    t    = torch.tensor([0, 250, 500, 999])
    x_t, noise = ddpm.q_sample(x0, t)
    print(f"\nForward process demo:")
    print(f"  x0    mean/std: {x0.mean():.3f} / {x0.std():.3f}")
    print(f"  x_t   mean/std: {x_t.mean():.3f} / {x_t.std():.3f}")
    print(f"  noise mean/std: {noise.mean():.3f} / {noise.std():.3f}")
    print("  Expected x_T ≈ N(0,1) ✓" if abs(x_t[-1].std().item() - 1.0) < 0.2 else "")

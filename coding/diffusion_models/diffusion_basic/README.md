# Denoising Diffusion Probabilistic Models (DDPM)
### A Ground-Up Implementation and Learning Guide

> **Paper:** [Ho et al., 2020 — Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)  
> **Implementation:** DDPM trained on MNIST using a time-conditioned U-Net

---

## Table of Contents
1. [Intuition](#1-intuition)
2. [The Forward Process — Adding Noise](#2-the-forward-process--adding-noise)
3. [The Reverse Process — Removing Noise](#3-the-reverse-process--removing-noise)
4. [The Training Objective — Where Does the Loss Come From?](#4-the-training-objective--where-does-the-loss-come-from)
5. [Sampling Algorithm](#5-sampling-algorithm)
6. [Network Architecture — The U-Net](#6-network-architecture--the-u-net)
7. [Code Walkthrough](#7-code-walkthrough)
8. [Running the Code](#8-running-the-code)
9. [What to Expect](#9-what-to-expect)
10. [Extensions and Further Reading](#10-extensions-and-further-reading)
11. [Resources](#11-resources)

---

## 1. Intuition

Diffusion models are inspired by **non-equilibrium thermodynamics**.  
Imagine taking a photo and slowly dissolving it in fog until nothing is recognizable — that is the **forward process**. Now imagine learning to reverse that process: given a foggy blob, step-by-step reconstruct a clear image.

If we can learn to reverse the corruption, we can generate new images by starting from **pure random noise** and running the reverse process.

```
Forward process (destroy):   x₀  →  x₁  →  x₂  → … →  xT  ≈ N(0,I)
                              clean          noisier       pure noise

Reverse process (create):    xT  →  xT₋₁ → … →  x₁  →  x₀
                              noise                        generated image
```

The key insight: we **don't** try to learn the entire reverse directly. Instead, we train a neural network to predict **the noise that was added** at each step, and use that to reverse the process one small step at a time.

---

## 2. The Forward Process — Adding Noise

### 2.1 One Step at a Time

The forward process `q` is a **Markov chain** that gradually adds Gaussian noise. At each step `t`, a small amount of noise is added:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\; \sqrt{1-\beta_t}\;\mathbf{x}_{t-1},\; \beta_t\,\mathbf{I}\right)$$

- `βₜ` is the **noise schedule** — a small positive number that controls how much noise is added at step `t`
- The mean `√(1−βₜ)·xₜ₋₁` slightly shrinks the previous image (signal decay)
- The variance `βₜ·I` adds fresh Gaussian noise

After `T` steps, when `ᾱT ≈ 0`, we get `xT ≈ N(0, I)` — pure noise.

### 2.2 The Noise Schedule

We use a **linear schedule** (original DDPM):

$$\beta_t = \beta_{\text{start}} + \frac{t-1}{T-1}\left(\beta_{\text{end}} - \beta_{\text{start}}\right)$$

with `β_start = 1e-4`, `β_end = 0.02`, `T = 1000`.

> **Alternative:** The cosine schedule (Nichol & Dhariwal, 2021) avoids abrupt transitions at the ends and tends to work better in practice.

### 2.3 The "Nice Property" — Sampling xₜ Directly

Computing `xₜ` step-by-step for `t = 1000` during training would be extremely slow. Fortunately, since the forward process is a chain of Gaussians, we can **collapse the entire chain** and jump directly from `x₀` to `xₜ` in one shot.

**Define:**

$$\alpha_t = 1 - \beta_t \qquad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$$

`ᾱₜ` is the **cumulative product** of all `αₛ` up to step `t`. It tells us how much of the original signal survives at time `t`:
- `ᾱ₁ ≈ 1` → almost no noise, signal mostly intact
- `ᾱT ≈ 0` → almost all noise, signal nearly gone

**Key result** (by repeatedly applying the Gaussian composition rule):

$$\boxed{q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\; (1-\bar{\alpha}_t)\,\mathbf{I}\right)}$$

So we can **sample xₜ in one step**:

$$\mathbf{x}_t = \underbrace{\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0}_{\text{scaled signal}} + \underbrace{\sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}}_{\text{scaled noise}}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

This is the **reparameterisation trick** — we express a random variable as a deterministic function of a standard normal.

**Why this matters for training:**  
At each training step we can instantly corrupt `x₀` to any noise level `t` without iterating through all `t` steps. This makes training efficient.

### 2.4 Signal-to-Noise Ratio

As `t` increases:
- Signal coefficient `√ᾱₜ` → 0
- Noise coefficient `√(1−ᾱₜ)` → 1

The image transitions smoothly from pure data to pure noise.

```
t=0    t=100  t=250  t=500  t=750  t=999
 [8]    [8̃]    [~]    [▒]    [░]    [noise]
```

---

## 3. The Reverse Process — Removing Noise

### 3.1 The True Reverse is Intractable

The true reverse distribution `q(xₜ₋₁ | xₜ)` requires integrating over the entire data distribution — this is exactly what we don't know. We want to learn it.

We approximate the reverse with a **learned Gaussian**:

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\; \sigma_t^2\,\mathbf{I}\right)$$

The variance `σₜ²` is fixed (not learned) and the mean `μθ` is predicted by the neural network.

### 3.2 The Tractable Posterior

Although `q(xₜ₋₁ | xₜ)` is intractable, the **conditional** posterior given the clean image `x₀` is tractable by Bayes' rule:

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0),\; \tilde{\beta}_t\,\mathbf{I}\right)$$

**Posterior mean:**

$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}\,\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,\mathbf{x}_t$$

**Posterior variance:**

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$$

This tells us: "if we knew `x₀`, here is the optimal one-step denoised distribution." The network's job is to predict `x₀` (or equivalently the noise `ε`).

### 3.3 Predicting Noise Instead of the Image

Ho et al. showed empirically that predicting the **noise `ε`** gives better results than directly predicting `x₀` or the mean `μ`. Given a noise prediction `ε̂θ`, we recover the predicted mean via:

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)\right)$$

**Intuition:** The network sees a noisy image `xₜ` and the current noise level `t`, and learns to estimate the noise that was added. Subtracting this estimate from `xₜ` gives a (slightly) cleaner image `xₜ₋₁`.

---

## 4. The Training Objective — Where Does the Loss Come From?

### 4.1 ELBO Derivation (Summary)

Diffusion models are trained by maximising the **Evidence Lower Bound (ELBO)** on the log-likelihood `log p(x₀)`:

$$\log p(\mathbf{x}_0) \geq \mathbb{E}_q\!\left[\log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1) - \sum_{t=2}^{T} D_{\text{KL}}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \;\|\; p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)) - D_{\text{KL}}(q(\mathbf{x}_T \mid \mathbf{x}_0) \;\|\; p(\mathbf{x}_T))\right]$$

Each KL divergence term `Lₜ` compares the learned reverse step to the tractable posterior. Since both are Gaussian:

$$L_{t-1} = \mathbb{E}_q\!\left[\frac{1}{2\sigma_t^2}\left\|\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t)\right\|^2\right]$$

### 4.2 Simplified Loss

Substituting the noise-parameterisation for `μθ` into `Lₜ` and absorbing all constants, Ho et al. showed the objective simplifies to:

$$\boxed{\mathcal{L}_{\text{simple}} = \mathbb{E}_{t \sim \mathcal{U}[1,T],\; \mathbf{x}_0,\; \boldsymbol{\epsilon}}\!\left[\left\|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_\theta\!\left(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},\; t\right)\right\|^2\right]}$$

This is just **MSE between the true noise `ε` and the predicted noise `ε̂θ`**.

### 4.3 Why This Loss Makes Sense

Think of it as a form of **score matching**: the network is learning to point toward the clean data distribution from any noise level. The gradient of the log-density is exactly the direction to remove noise — and that's precisely what `−ε̂θ` approximates.

At **high noise levels** (large `t`): the task is coarse — remove big blobs of noise.  
At **low noise levels** (small `t`): the task is fine — sharpen edges and details.

By sampling `t` uniformly during training, the network learns to denoise across all noise levels simultaneously.

---

## 5. Sampling Algorithm

Once trained, generating a new image requires running the **reverse process** from `T` to `0`:

### Algorithm (DDPM Sampler)

```
Input:  trained model ε̂θ, noise schedule {βₜ}, number of steps T

1.  Sample xT ~ N(0, I)                        ← start from pure noise

2.  For t = T, T−1, …, 1:
      a. Sample z ~ N(0, I)  if t > 1, else z = 0
      b. Predict noise:  ε̂ = ε̂θ(xₜ, t)
      c. Compute mean:
             μₜ = (1/√αₜ) · (xₜ − βₜ/√(1−ᾱₜ) · ε̂)
      d. Sample:  xₜ₋₁ = μₜ + √β̃ₜ · z

3.  Return x₀
```

At the final step `t=1 → t=0` we set `z=0` (no noise injection) to avoid random fluctuations in the output.

### Computational Cost

DDPM requires **T = 1000 forward passes** through the network per sample. This is slow but exact. Faster samplers (DDIM, DPM-Solver) can generate high-quality samples in 20–50 steps by making the sampler deterministic or using better ODE solvers.

---

## 6. Network Architecture — The U-Net

The noise predictor `ε̂θ(xₜ, t)` is a **U-Net** — an encoder–decoder CNN with skip connections, originally developed for biomedical image segmentation.

### Why U-Net?

- **Skip connections** preserve spatial information that gets compressed in the encoder, helping the decoder reconstruct fine detail.
- **Multi-scale processing** — different levels of the U-Net handle structures at different scales (global composition at the bottleneck, texture at the top levels).
- The encoder down-samples to build context; the decoder up-samples to predict pixel-level noise.

### 6.1 Time Conditioning

The network must know **which noise level** it is operating at. We encode the scalar timestep `t` as a **sinusoidal embedding** (adapted from Transformer positional encodings):

$$\text{SinEmb}(t)_i = \begin{cases} \sin\!\left(\dfrac{t}{10000^{2i/d}}\right) & i < d/2 \\[6pt] \cos\!\left(\dfrac{t}{10000^{(2i-d)/d}}\right) & i \geq d/2 \end{cases}$$

This `d`-dimensional vector is passed through a small MLP and then **added** into each residual block after the first convolution. This lets the network condition all of its intermediate computations on the current noise level.

### 6.2 Residual Blocks

```
x ──► GroupNorm ──► SiLU ──► Conv2d ──────────────────────────────► +──► out
                                  ▲                               ▲  │
                              time_proj(t_emb)               res_conv(x)
                                  │                               │
                              GroupNorm ──► SiLU ──► Conv2d ──────┘
```

Key design choices:
- **GroupNorm** instead of BatchNorm: stable with small batch sizes, no batch statistics at test time
- **SiLU** (Swish) activation: smooth, non-monotone, empirically superior for diffusion models
- **Residual connection**: standard ResNet trick; helps gradient flow during training

### 6.3 Self-Attention at the Bottleneck

At the smallest spatial resolution (4×4), a **self-attention** block is inserted. Every spatial location attends to every other:

```
x ──► GroupNorm ──► Reshape to (B, H·W, C) ──► MultiheadAttention ──► Reshape ──► + x
```

This is essential for coherent global structure — without it, left and right halves of the image can be independent. Attention makes the model aware of long-range spatial relationships.

### 6.4 Full Architecture (for 32×32 input, base_ch=32)

```
Input  (B, 1, 32, 32)  — noisy image xₜ
  + t  (B,)            — timestep
  ↓
 ┌──────────────────────────────────────────────────────────────────┐
 │ time_mlp: SinEmb(128) → Linear(512) → SiLU → Linear(128)        │
 └──────────────────────────────────────────────────────────────────┘
  ↓ t_emb (B,128)

init_conv  [Conv 3×3]            (B,  1,32,32) → (B, 32,32,32)

── ENCODER ──────────────────────────────────────────────────────────────
enc1  [ResBlock 32→32]           (B, 32,32,32) → skip1=(B,32,32,32)
down1 [Conv 4×4 stride-2]        (B, 32,32,32) → (B,32,16,16)

enc2  [ResBlock 32→64]           (B, 32,16,16) → skip2=(B,64,16,16)
down2 [Conv 4×4 stride-2]        (B, 64,16,16) → (B,64, 8, 8)

enc3  [ResBlock 64→128]          (B, 64, 8, 8) → skip3=(B,128,8,8)
down3 [Conv 4×4 stride-2]        (B,128, 8, 8) → (B,128,4, 4)

── BOTTLENECK ───────────────────────────────────────────────────────────
mid1  [ResBlock 128→128]         (B,128, 4, 4) → (B,128,4,4)
attn  [Self-Attention 128]       (B,128, 4, 4) → (B,128,4,4)
mid2  [ResBlock 128→128]         (B,128, 4, 4) → (B,128,4,4)

── DECODER (skip connections from encoder) ──────────────────────────────
up3   [ConvTranspose 4×4 s=2]    (B,128, 4, 4) → (B,128, 8, 8)
cat3  [cat with skip3]           → (B,256, 8, 8)
dec3  [ResBlock 256→64]          (B,256, 8, 8) → (B, 64, 8, 8)

up2   [ConvTranspose 4×4 s=2]    (B, 64, 8, 8) → (B, 64,16,16)
cat2  [cat with skip2]           → (B,128,16,16)
dec2  [ResBlock 128→32]          (B,128,16,16) → (B, 32,16,16)

up1   [ConvTranspose 4×4 s=2]    (B, 32,16,16) → (B, 32,32,32)
cat1  [cat with skip1]           → (B, 64,32,32)
dec1  [ResBlock 64→32]           (B, 64,32,32) → (B, 32,32,32)

── OUTPUT ────────────────────────────────────────────────────────────────
norm  [GroupNorm] + act [SiLU]
out   [Conv 1×1]                 (B, 32,32,32) → (B,  1,32,32)  ← ε̂
```

Total parameters: ~1.6M (with base_ch=32, time_emb_dim=128)

---

## 7. Code Walkthrough

### File Structure
```
diffusion_basic/
├── model.py        ← U-Net architecture (ε̂θ network)
├── diffusion.py    ← DDPM: noise schedule, forward/reverse process, loss
├── train.py        ← Training loop on MNIST
├── sample.py       ← Generate samples + visualise denoising
├── requirements.txt
└── README.md       ← You are here
```

### `diffusion.py` — Core Math

```python
# Forward process: sample xₜ from x₀ in one step
def q_sample(self, x0, t, noise=None):
    sqrt_ab    = self.sqrt_alpha_bars[t][:, None, None, None]         # √ᾱₜ
    sqrt_1m_ab = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]  # √(1−ᾱₜ)
    return sqrt_ab * x0 + sqrt_1m_ab * noise, noise

# Training loss: MSE between true noise and predicted noise
def compute_loss(self, model, x0):
    t        = torch.randint(0, self.T, (B,))      # random timestep
    x_t, eps = self.q_sample(x0, t)                # corrupt image
    eps_pred = model(x_t, t)                       # predict noise
    return F.mse_loss(eps_pred, eps)               # L_simple

# Reverse step: xₜ → xₜ₋₁
def p_sample(self, model, x_t, t):
    eps_pred = model(x_t, t_tensor)                # ε̂θ(xₜ, t)
    coeff    = beta_t / (1 - alpha_bar_t).sqrt()
    mean     = (1 / alpha_t.sqrt()) * (x_t - coeff * eps_pred)
    return mean + posterior_std * z                # + stochastic noise
```

### `model.py` — Time Conditioning

```python
# In ResBlock.forward():
h = self.conv1(h)
h = h + self.act(self.time_proj(t_emb))[:, :, None, None]  # inject t
h = self.conv2(h)
```

The time embedding is added after the first convolution, "steering" the second convolution with knowledge of the current noise level.

---

## 8. Running the Code

### Setup

```bash
cd coding/diffusion_models/diffusion_basic
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

- Downloads MNIST automatically to `./data/`
- Saves checkpoints every 10 epochs to `./checkpoints/`
- Saves training loss curve to `./checkpoints/training_loss.png`
- Expected training time: ~2–3 min/epoch on GPU, ~15–20 min/epoch on CPU

**Tip:** For a quick test, reduce `EPOCHS = 10` in `train.py` — you'll see blurry but digit-like samples.

### Generating Samples

```bash
# Visualise forward process (no checkpoint needed)
python sample.py --forward_only

# Generate 16 samples from a trained model
python sample.py --checkpoint checkpoints/ddpm_final.pt

# Generate 64 samples
python sample.py --n_samples 64
```

Output saved to `./samples/`:
- `forward_process.png`       — shows how a real MNIST digit is progressively corrupted
- `generated_samples.png`     — grid of model-generated digits
- `denoising_trajectory.png`  — single sample denoised step-by-step

### Sanity checks

```bash
# Test model architecture (no data needed)
python model.py

# Test noise schedule and forward process
python diffusion.py
```

---

## 9. What to Expect

| Epoch | Expected Loss | Sample Quality |
|-------|--------------|----------------|
| 10    | ~0.09        | Blurry blobs, barely digit-like |
| 30    | ~0.06        | Recognisable digit shapes |
| 50    | ~0.05        | Clear digits, some artifacts |
| 100   | ~0.04        | Crisp MNIST-quality digits |

Loss decreasing below 0.05 typically produces recognisable digits. Diffusion models are slow to train but produce diverse, high-quality samples without mode collapse (unlike GANs).

---

## 10. Extensions and Further Reading

Once you understand the basics, here is the natural progression:

### 10.1 Improved Sampling Speed (DDIM)
The original DDPM sampler requires `T = 1000` forward passes. **DDIM** (Song et al., 2020) reformulates the reverse as a deterministic ODE, enabling high-quality samples in ~50 steps:
- Replace stochastic sampling with deterministic integration
- No retraining needed — applies to any DDPM checkpoint

### 10.2 Better Training (Improved DDPM)
Nichol & Dhariwal (2021) introduced:
- **Cosine noise schedule**: avoids too-small gradients at early/late timesteps
- **Learned variance**: instead of fixing `σₜ`, learn it by predicting a mixture coefficient
- These together improve log-likelihood and sample quality

### 10.3 Classifier Guidance
Dhariwal & Nichol (2021): train a classifier `p(y | xₜ)` on noisy images, then use its gradient to steer sampling toward a desired class:
```
ε̂guided = ε̂θ(xₜ, t) − s · ∇_{xₜ} log p(y | xₜ)
```
Enabled diffusion models to surpass GANs on ImageNet for the first time.

### 10.4 Classifier-Free Guidance (CFG)
Ho & Salimans (2022): train a single conditional model that also works unconditionally:
```
ε̂CFG = ε̂θ(xₜ, t, c) + s · (ε̂θ(xₜ, t, c) − ε̂θ(xₜ, t, ∅))
```
This is the dominant guidance method in Stable Diffusion, DALL-E 3, Imagen, etc.

### 10.5 Latent Diffusion Models (Stable Diffusion)
Rombach et al. (2022): instead of running diffusion in pixel space, first encode images to a compact **latent space** with a VAE, run diffusion there, then decode back. This reduces computational cost by ~16–64× and enables high-resolution generation.

### 10.6 Score-Based Perspective
Song & Ermon (2019, 2020) show that diffusion models are equivalent to learning the **score function** (gradient of the log data density):
$$\nabla_{\mathbf{x}} \log p(\mathbf{x}) \approx -\frac{\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)}{\sqrt{1-\bar{\alpha}_t}}$$
This unifies diffusion models with score matching and SDEs, enabling continuous-time formulations.

---

## 11. Resources

### 📄 Must-Read Papers

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| [DDPM — Ho et al.](https://arxiv.org/abs/2006.11239) | 2020 | Original DDPM formulation |
| [DDIM — Song et al.](https://arxiv.org/abs/2010.02502) | 2020 | Deterministic fast sampling |
| [Score-Based SDEs — Song et al.](https://arxiv.org/abs/2011.13456) | 2020 | Continuous-time unification |
| [Improved DDPM — Nichol & Dhariwal](https://arxiv.org/abs/2102.09672) | 2021 | Learned variance + cosine schedule |
| [ADM — Dhariwal & Nichol](https://arxiv.org/abs/2105.05233) | 2021 | Classifier guidance, beats GANs |
| [Classifier-Free Guidance — Ho & Salimans](https://arxiv.org/abs/2207.12598) | 2022 | CFG, used in all modern systems |
| [LDM — Rombach et al.](https://arxiv.org/abs/2112.10752) | 2022 | Latent diffusion (Stable Diffusion) |
| [DALL-E 2 — Ramesh et al.](https://arxiv.org/abs/2204.06125) | 2022 | CLIP + diffusion text-to-image |
| [Imagen — Saharia et al.](https://arxiv.org/abs/2205.11487) | 2022 | Large-scale text-to-image |
| [DPM-Solver — Lu et al.](https://arxiv.org/abs/2206.00927) | 2022 | ODE solver, 10-20 step sampling |

### 📝 Blogs & Tutorials

| Resource | Description |
|----------|-------------|
| [Lilian Weng — What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | ⭐ Best mathematical overview; covers DDPM → score matching → guidance |
| [Hugging Face — The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) | Line-by-line code walkthrough using PyTorch |
| [Hugging Face — The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) | Visual guide to latent diffusion and CLIP |
| [AssemblyAI — Diffusion Models from Scratch](https://www.assemblyai.com/blog/diffusion-models-for-audio-classification-from-scratch/) | Clean PyTorch tutorial |
| [Sander Dieleman — Perspectives on Diffusion](https://sander.ai/2022/01/31/diffusion.html) | Deep dive from a researcher at DeepMind |
| [Calvin Luo — Understanding Diffusion Models](https://arxiv.org/abs/2208.11970) | Arxiv tutorial paper: ELBO derivation in full detail |

### 🎥 Videos & Lectures

| Resource | Description |
|----------|-------------|
| [Ari Seff — What are Diffusion Models?](https://www.youtube.com/watch?v=fbLgFrlTnGU) | Excellent visual intuition (12 min) |
| [Yannic Kilcher — DDPM Paper Explained](https://www.youtube.com/watch?v=W-O7AZNzbzQ) | Paper walkthrough with commentary |
| [Stanford CS236 — Deep Generative Models](https://deepgenerativemodels.github.io/) | Full course; diffusion covered in later lectures |
| [MIT 6.S191 — Diffusion Models](https://www.youtube.com/watch?v=rTAwesyNLeE) | Accessible introduction |
| [Diffusion Models from Scratch (labml.ai)](https://nn.labml.ai/diffusion/ddpm/index.html) | Annotated, runnable implementation |

### 🔧 Code References

| Repository | Description |
|-----------|-------------|
| [openai/improved-diffusion](https://github.com/openai/improved-diffusion) | Official Improved DDPM code |
| [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) | Official Stable Diffusion (LDM) code |
| [huggingface/diffusers](https://github.com/huggingface/diffusers) | Production-ready library for all diffusion variants |
| [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) | Clean minimal PyTorch DDPM |
| [pesser/pytorch_diffusion](https://github.com/pesser/pytorch_diffusion) | Educational PyTorch implementation |

---

## Key Equations Summary

| Symbol | Meaning |
|--------|---------|
| `x₀` | Clean data sample |
| `xₜ` | Noisy data at timestep `t` |
| `βₜ` | Noise variance at step `t` (noise schedule) |
| `αₜ = 1 − βₜ` | Signal retention at step `t` |
| `ᾱₜ = ∏αₛ` | Cumulative signal retention |
| `ε ~ N(0,I)` | Standard Gaussian noise |
| `ε̂θ(xₜ, t)` | Network's predicted noise |
| `q(xₜ\|x₀)` | Forward process distribution |
| `pθ(xₜ₋₁\|xₜ)` | Learned reverse distribution |
| `L_simple` | MSE training loss |

**Forward:**  
&nbsp;&nbsp;`xₜ = √ᾱₜ · x₀ + √(1−ᾱₜ) · ε`

**Reverse step:**  
&nbsp;&nbsp;`xₜ₋₁ = (1/√αₜ) · (xₜ − βₜ/√(1−ᾱₜ) · ε̂θ(xₜ,t)) + √β̃ₜ · z`

**Loss:**  
&nbsp;&nbsp;`L = E[ ‖ε − ε̂θ(xₜ, t)‖² ]`

---

*Happy diffusing! 🌊*

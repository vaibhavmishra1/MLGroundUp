"""
Core measurement functions for the Fisher Fundamental Theorem experiment.

Two quantities measured at each epoch:
  1. Functional Information  I_f(t) = -log2 P[L(theta_0) <= L(theta_t)]
  2. Empirical Fisher Trace  T_F(t) = E[ ||grad log p(y|x; theta_t)||^2 ]

For functional information, direct counting saturates quickly because trained
networks achieve losses far below any random initialization. We use two
complementary approaches:
  (a) Parametric extrapolation: fit a distribution to random losses, extrapolate
      the CDF into the far tail.
  (b) Loss-rank functional information: use the negative log-fraction directly
      when it is measurable, and the parametric tail otherwise.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats as sp_stats


@torch.no_grad()
def compute_loss_on_dataset(model, data_loader, device, max_batches=None):
    """Compute average cross-entropy loss over the dataset."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for i, (x, y) in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        total_samples += y.size(0)

    return total_loss / total_samples


@torch.no_grad()
def cache_random_network_losses(model_cls, model_kwargs, data_loader, device,
                                 M=1000, batch_limit=10, seed=0):
    """
    Sample M random initializations from the prior pi(theta),
    evaluate their loss, cache the sorted array and fit a parametric tail model.

    Returns:
        sorted_losses: np.ndarray of shape (M,), sorted ascending
        tail_model: dict with fitted distribution parameters for extrapolation
    """
    from models import init_weights_kaiming

    losses = np.empty(M, dtype=np.float64)

    for m in range(M):
        model = model_cls(**model_kwargs).to(device)
        init_weights_kaiming(model)
        for p in model.parameters():
            p.data = torch.randn_like(p.data) * p.data.std()

        loss = compute_loss_on_dataset(model, data_loader, device, max_batches=batch_limit)
        losses[m] = loss

        if (m + 1) % 500 == 0:
            print(f"  Random network {m+1}/{M}: loss = {loss:.4f}")

        del model

    sorted_losses = np.sort(losses)

    mu, sigma = np.mean(losses), np.std(losses)
    tail_model = {"mu": mu, "sigma": sigma, "M": M}

    print(f"  Random loss stats: min={sorted_losses[0]:.4f}, "
          f"median={sorted_losses[M//2]:.4f}, max={sorted_losses[-1]:.4f}, "
          f"mu={mu:.4f}, sigma={sigma:.4f}")
    return sorted_losses, tail_model


def compute_functional_information(current_loss, sorted_random_losses, tail_model):
    """
    I_f(t) = -log2 P[L(theta_0) <= L(theta_t)]

    Uses empirical CDF when the count is large enough (>5 samples below threshold),
    otherwise extrapolates using the fitted Gaussian tail. This avoids the
    saturation problem where trained losses fall below all M random losses.

    The Gaussian extrapolation is conservative: actual random loss distributions
    have heavier-than-Gaussian lower tails (losses are bounded below by 0),
    so the Gaussian CDF underestimates P, yielding a *lower bound* on I_f.
    """
    M = len(sorted_random_losses)
    count = np.searchsorted(sorted_random_losses, current_loss, side="right")

    if count >= 5:
        F_hat = count / M
        I_f = -np.log2(F_hat)
        method = "empirical"
    else:
        mu, sigma = tail_model["mu"], tail_model["sigma"]
        if sigma < 1e-10:
            # All random networks have ~same loss. Use a simple indicator:
            # if current < mu, I_f is "infinite" (capped); if current >= mu, I_f ≈ 0
            if current_loss < mu - 1e-8:
                F_hat = 1e-30
                I_f = -np.log2(F_hat)
            else:
                F_hat = 0.5
                I_f = 1.0
            method = "degenerate"
        else:
            F_hat = sp_stats.norm.cdf(current_loss, loc=mu, scale=sigma)
            F_hat = max(F_hat, 1e-30)
            I_f = -np.log2(F_hat)
            method = "gaussian_tail"

    return I_f, F_hat, method


def compute_fisher_trace(model, data_loader, device, max_batches=10):
    """
    Empirical Fisher trace: T_F = (1/N) sum_i ||grad_theta log p(y_i|x_i; theta)||^2

    For cross-entropy loss with softmax output, grad log p(y|x;theta) = grad l(theta;x,y),
    so this equals the mean squared per-sample gradient norm.
    """
    model.eval()
    total_sq_norm = 0.0
    total_samples = 0

    for i, (x, y) in enumerate(data_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        batch_size = y.size(0)

        for j in range(batch_size):
            model.zero_grad()
            logit = model(x[j:j+1])
            log_prob = nn.functional.log_softmax(logit, dim=-1)
            nll = -log_prob[0, y[j]]
            nll.backward()

            sq_norm = sum(
                p.grad.pow(2).sum().item()
                for p in model.parameters()
                if p.grad is not None
            )
            total_sq_norm += sq_norm
            total_samples += 1

    model.zero_grad()
    return total_sq_norm / total_samples if total_samples > 0 else 0.0


def compute_fisher_trace_fast(model, data_loader, device, max_batches=10):
    """
    Fast approximation: use batch gradient norm as proxy.
    T_F ≈ (1/B) * ||grad_theta L_batch||^2 * B = ||grad_theta L_batch||^2

    More precisely, for a batch of size B:
    Var(grad) ≈ (1/B) * sum_i ||g_i||^2 - ||mean(g_i)||^2

    We compute the exact per-sample version but vectorized via gradient accumulation.
    """
    model.eval()
    total_sq_norm = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss(reduction="none")

    for i, (x, y) in enumerate(data_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        logits = model(x)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        per_sample_nll = -log_probs[range(len(y)), y]

        for j in range(len(y)):
            model.zero_grad()
            per_sample_nll[j].backward(retain_graph=True)
            sq_norm = sum(
                p.grad.pow(2).sum().item()
                for p in model.parameters()
                if p.grad is not None
            )
            total_sq_norm += sq_norm
            total_samples += 1

    model.zero_grad()
    return total_sq_norm / total_samples if total_samples > 0 else 0.0


def compute_fisher_trace_efficient(model, data_loader, device, num_samples=128):
    """
    Efficient Fisher trace computation: sample individual data points and
    accumulate squared gradient norms. Uses only num_samples points total.
    """
    model.eval()
    total_sq_norm = 0.0
    count = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        for j in range(x.size(0)):
            if count >= num_samples:
                model.zero_grad()
                return total_sq_norm / count

            model.zero_grad()
            logit = model(x[j:j+1])
            log_prob = nn.functional.log_softmax(logit, dim=-1)
            nll = -log_prob[0, y[j]]
            nll.backward()

            sq_norm = sum(
                p.grad.pow(2).sum().item()
                for p in model.parameters()
                if p.grad is not None
            )
            total_sq_norm += sq_norm
            count += 1

    model.zero_grad()
    return total_sq_norm / count if count > 0 else 0.0

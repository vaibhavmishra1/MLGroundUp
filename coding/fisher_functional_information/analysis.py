"""
Analysis and plotting for the Fisher Fundamental Theorem experiment.

Produces:
  1. I_f(t) vs epoch  — tests monotonicity (Part A)
  2. dI_f/dt vs T_F(t) — tests rate law (Part E)
  3. Summary statistics: R², Spearman rank correlation, monotonicity fraction
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def load_results(tag):
    path = os.path.join(RESULTS_DIR, f"{tag}.json")
    with open(path) as f:
        return json.load(f)


def extract_timeseries(results):
    """Extract arrays from results dict."""
    epochs_data = results["epochs"]
    epochs = np.array([d["epoch"] for d in epochs_data])
    loss = np.array([d["train_loss"] for d in epochs_data])
    I_f = np.array([d["functional_info"] for d in epochs_data])
    F_hat = np.array([d["F_hat"] for d in epochs_data])
    T_F = np.array([d["fisher_trace"] for d in epochs_data])
    methods = [d.get("fi_method", "unknown") for d in epochs_data]
    return epochs, loss, I_f, F_hat, T_F, methods


def test_monotonicity(I_f):
    """Test whether I_f is non-decreasing. Returns fraction of non-decreasing steps."""
    diffs = np.diff(I_f)
    n_nondecreasing = np.sum(diffs >= -1e-10)
    fraction = n_nondecreasing / len(diffs)
    return fraction, diffs


def test_rate_law(I_f, T_F, epochs):
    """
    Test: dI_f/dt ∝ T_F(t)

    Compute finite differences of I_f, pair with T_F at the midpoint,
    and test correlation.
    """
    dI_f = np.diff(I_f)
    dt = np.diff(epochs)
    rate = dI_f / dt

    T_F_mid = (T_F[:-1] + T_F[1:]) / 2

    mask = np.isfinite(rate) & np.isfinite(T_F_mid)
    rate_clean = rate[mask]
    T_F_clean = T_F_mid[mask]

    if len(rate_clean) < 3:
        return {
            "r_squared": np.nan, "spearman_r": np.nan,
            "spearman_p": np.nan, "alpha_fit": np.nan,
            "rate": rate, "T_F_mid": T_F_mid,
        }

    slope, intercept, r_value, p_value, std_err = stats.linregress(T_F_clean, rate_clean)
    spearman_r, spearman_p = stats.spearmanr(T_F_clean, rate_clean)

    return {
        "r_squared": r_value**2,
        "slope": slope,
        "intercept": intercept,
        "p_value": p_value,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "rate": rate,
        "T_F_mid": T_F_mid,
    }


def analyze_and_plot(tag, results=None):
    """Full analysis pipeline for one experiment run."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if results is None:
        results = load_results(tag)

    epochs, loss, I_f, F_hat, T_F, methods = extract_timeseries(results)
    setting = results.get("setting", tag)
    opt = results.get("optimizer", "?")
    rl = results.get("random_labels", False)
    title_suffix = f" [{opt}]" + (" [random labels]" if rl else "")

    # Part A: Monotonicity of I_f
    mono_frac_If, _ = test_monotonicity(I_f)

    # Loss should decrease monotonically (the direct "fitness" analog)
    neg_loss = -loss
    mono_frac_loss, _ = test_monotonicity(neg_loss)

    # Part E: Rate law on I_f
    rate_If = test_rate_law(I_f, T_F, epochs)

    # Direct Fisher theorem analog: -dL/dt vs T_F
    # (negative because we want rate of fitness *increase* = rate of loss *decrease*)
    rate_loss = test_rate_law(neg_loss, T_F, epochs)

    print(f"\n{'='*60}")
    print(f"Analysis: {tag}")
    print(f"{'='*60}")
    print(f"  FI methods used: {set(methods)}")
    print(f"  --- Part A: Monotonicity ---")
    print(f"  I_f monotonicity:   {mono_frac_If:.1%}")
    print(f"  Loss decrease mono: {mono_frac_loss:.1%}")
    print(f"  --- Part E: Rate Law (dI_f/dt ~ T_F) ---")
    print(f"  R² = {rate_If['r_squared']:.4f},  "
          f"Spearman ρ = {rate_If['spearman_r']:.4f} (p={rate_If['spearman_p']:.2e})")
    print(f"  --- Direct Fisher Theorem (-dL/dt ~ T_F) ---")
    print(f"  R² = {rate_loss['r_squared']:.4f},  "
          f"Spearman ρ = {rate_loss['spearman_r']:.4f} (p={rate_loss['spearman_p']:.2e})")

    # ---- Figure 1: Six-panel overview ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Fisher Fundamental Theorem — {setting}{title_suffix}", fontsize=14)

    # 1a: Loss curve
    ax = axes[0, 0]
    ax.plot(epochs, loss, "b-o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss L(θ_t)")
    ax.grid(True, alpha=0.3)

    # 1b: Functional information
    ax = axes[0, 1]
    ax.plot(epochs, I_f, "r-o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("I_f (bits)")
    ax.set_title(f"Functional Information (mono={mono_frac_If:.0%})")
    ax.grid(True, alpha=0.3)

    # 1c: Fisher trace
    ax = axes[0, 2]
    ax.plot(epochs, T_F, "g-o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Tr(F)")
    ax.set_title("Empirical Fisher Trace T_F(t)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # 1d: Rate law scatter (I_f)
    ax = axes[1, 0]
    rate = rate_If["rate"]
    T_F_mid = rate_If["T_F_mid"]
    mask = np.isfinite(rate) & np.isfinite(T_F_mid)
    if mask.any():
        ax.scatter(T_F_mid[mask], rate[mask], c=epochs[1:][mask], cmap="viridis",
                   edgecolors="k", linewidth=0.5, s=40, zorder=5)
        if 'slope' in rate_If and np.isfinite(rate_If['slope']):
            x_fit = np.linspace(T_F_mid[mask].min(), T_F_mid[mask].max(), 100)
            ax.plot(x_fit, rate_If['slope'] * x_fit + rate_If['intercept'],
                    'r--', label=f"R²={rate_If['r_squared']:.3f}")
            ax.legend()
    ax.set_xlabel("Fisher Trace T_F(t)")
    ax.set_ylabel("dI_f/dt")
    ax.set_title("Rate Law: dI_f/dt vs T_F(t)")
    ax.grid(True, alpha=0.3)

    # 1e: Direct Fisher theorem (-dL/dt vs T_F)
    ax = axes[1, 1]
    rate_l = rate_loss["rate"]
    T_F_mid_l = rate_loss["T_F_mid"]
    mask_l = np.isfinite(rate_l) & np.isfinite(T_F_mid_l)
    if mask_l.any():
        sc = ax.scatter(T_F_mid_l[mask_l], rate_l[mask_l], c=epochs[1:][mask_l],
                        cmap="viridis", edgecolors="k", linewidth=0.5, s=40, zorder=5)
        if 'slope' in rate_loss and np.isfinite(rate_loss['slope']):
            x_fit = np.linspace(T_F_mid_l[mask_l].min(), T_F_mid_l[mask_l].max(), 100)
            ax.plot(x_fit, rate_loss['slope'] * x_fit + rate_loss['intercept'],
                    'r--', label=f"R²={rate_loss['r_squared']:.3f}")
            ax.legend()
    ax.set_xlabel("Fisher Trace T_F(t)")
    ax.set_ylabel("-dL/dt (fitness increase rate)")
    ax.set_title("Fisher Theorem: -dL/dt vs T_F(t)")
    ax.grid(True, alpha=0.3)

    # 1f: Log-log of -dL/dt vs T_F
    ax = axes[1, 2]
    if mask_l.any():
        pos_mask = mask_l & (rate_l > 0) & (T_F_mid_l > 0)
        if pos_mask.any():
            ax.scatter(np.log10(T_F_mid_l[pos_mask]), np.log10(rate_l[pos_mask]),
                       c=epochs[1:][pos_mask], cmap="viridis",
                       edgecolors="k", linewidth=0.5, s=40, zorder=5)
            log_T = np.log10(T_F_mid_l[pos_mask])
            log_r = np.log10(rate_l[pos_mask])
            if len(log_T) >= 3:
                sl, ic, rv, pv, se = stats.linregress(log_T, log_r)
                x_fit = np.linspace(log_T.min(), log_T.max(), 100)
                ax.plot(x_fit, sl * x_fit + ic, 'r--',
                        label=f"slope={sl:.2f}, R²={rv**2:.3f}")
                ax.legend()
    ax.set_xlabel("log₁₀ T_F")
    ax.set_ylabel("log₁₀ (-dL/dt)")
    ax.set_title("Log-Log: Power Law?")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{tag}_overview.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Figure 2: Dual-axis I_f and T_F ----
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1, color2 = "#d62728", "#2ca02c"

    ax1.plot(epochs, I_f, color=color1, marker="o", markersize=3, label="I_f(t)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Functional Information I_f (bits)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(epochs, T_F, color=color2, marker="s", markersize=3, label="T_F(t)")
    ax2.set_ylabel("Fisher Trace T_F(t)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_yscale("log")

    fig.suptitle(f"I_f and T_F Co-evolution — {setting}{title_suffix}", fontsize=13)
    fig.tight_layout()
    path2 = os.path.join(PLOTS_DIR, f"{tag}_dual.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path2}")

    return {
        "tag": tag,
        "mono_If": mono_frac_If,
        "mono_loss": mono_frac_loss,
        "If_r2": rate_If["r_squared"],
        "If_spearman": rate_If["spearman_r"],
        "loss_r2": rate_loss["r_squared"],
        "loss_spearman": rate_loss["spearman_r"],
        "loss_spearman_p": rate_loss["spearman_p"],
    }


def compare_runs(tags, title="Comparison"):
    """Compare multiple experiment runs side by side."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for tag in tags:
        results = load_results(tag)
        epochs, loss, I_f, F_hat, T_F, methods = extract_timeseries(results)
        label = tag.replace("_", " ")

        axes[0].plot(epochs, loss, "-o", markersize=2, label=label)
        axes[1].plot(epochs, I_f, "-o", markersize=2, label=label)
        axes[2].plot(epochs, T_F, "-o", markersize=2, label=label)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Functional Information")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("I_f (bits)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Fisher Trace")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Tr(F)")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"comparison_{'_'.join(tags[:3])}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison: {path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for tag in sys.argv[1:]:
            analyze_and_plot(tag)
    else:
        result_files = [f.replace(".json", "") for f in os.listdir(RESULTS_DIR)
                        if f.endswith(".json")]
        summaries = []
        for tag in sorted(result_files):
            s = analyze_and_plot(tag)
            summaries.append(s)

        if summaries:
            print(f"\n{'='*80}")
            print("SUMMARY TABLE")
            print(f"{'='*80}")
            print(f"{'Tag':<35} {'If_mono':>7} {'L_mono':>7} "
                  f"{'If_R²':>7} {'L_R²':>7} {'L_ρ':>7} {'L_p':>10}")
            print("-" * 82)
            for s in summaries:
                print(f"{s['tag']:<35} {s['mono_If']:>6.0%} {s['mono_loss']:>6.0%} "
                      f"{s['If_r2']:>7.3f} {s['loss_r2']:>7.3f} "
                      f"{s['loss_spearman']:>7.3f} {s['loss_spearman_p']:>10.2e}")

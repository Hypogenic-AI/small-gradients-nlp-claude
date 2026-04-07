"""
Visualization script for Small Gradients research.
Generates all figures from experiment results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path("/workspaces/small-gradients-nlp-claude/results")
FIGURES_DIR = Path("/workspaces/small-gradients-nlp-claude/figures")

sns.set_theme(style="whitegrid", font_scale=1.2)
COLORS = {"human": "#2196F3", "synthetic": "#FF5722"}


def plot_perplexity_distributions(ppl_results):
    """Figure 1: Perplexity distributions for human vs synthetic data."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    h_vals = np.array(ppl_results["human"]["values"])
    s_vals = np.array(ppl_results["synthetic"]["values"])
    # Clip extreme outliers for visualization
    h_clip = h_vals[h_vals < np.percentile(h_vals, 99)]
    s_clip = s_vals[s_vals < np.percentile(s_vals, 99)]

    ax.hist(h_clip, bins=50, alpha=0.6, color=COLORS["human"], label="Human", density=True)
    ax.hist(s_clip, bins=50, alpha=0.6, color=COLORS["synthetic"], label="Synthetic", density=True)
    ax.set_xlabel("Perplexity")
    ax.set_ylabel("Density")
    ax.set_title("Perplexity Distribution")
    ax.legend()

    # Box plot
    ax = axes[1]
    data = [h_clip, s_clip]
    bp = ax.boxplot(data, labels=["Human", "Synthetic"], patch_artist=True,
                     widths=0.5, showfliers=False)
    bp['boxes'][0].set_facecolor(COLORS["human"])
    bp['boxes'][1].set_facecolor(COLORS["synthetic"])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity Comparison")

    # Add stats text
    stats_text = (f"Human: μ={ppl_results['human']['mean']:.1f}, σ={ppl_results['human']['std']:.1f}\n"
                  f"Synthetic: μ={ppl_results['synthetic']['mean']:.1f}, σ={ppl_results['synthetic']['std']:.1f}\n"
                  f"p={ppl_results['test']['p_value']:.2e}, d={ppl_results['test']['cohens_d']:.2f}")
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_perplexity_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig1_perplexity_distributions.png")


def plot_gradient_norms(grad_results):
    """Figure 2: Gradient norm comparison over training steps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    h_norms = np.array(grad_results["human"]["values"])
    s_norms = np.array(grad_results["synthetic"]["values"])

    # Time series
    ax = axes[0]
    ax.plot(h_norms, color=COLORS["human"], alpha=0.7, label="Human", linewidth=0.8)
    ax.plot(s_norms, color=COLORS["synthetic"], alpha=0.7, label="Synthetic", linewidth=0.8)
    # Smoothed
    window = 10
    h_smooth = np.convolve(h_norms, np.ones(window)/window, mode='valid')
    s_smooth = np.convolve(s_norms, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(h_norms)), h_smooth, color=COLORS["human"], linewidth=2.5, label="Human (smoothed)")
    ax.plot(range(window-1, len(s_norms)), s_smooth, color=COLORS["synthetic"], linewidth=2.5, label="Synthetic (smoothed)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient L2 Norm")
    ax.set_title("Gradient Norms Over Training")
    ax.legend(fontsize=9)

    # Distribution
    ax = axes[1]
    ax.hist(h_norms, bins=40, alpha=0.6, color=COLORS["human"], label="Human", density=True)
    ax.hist(s_norms, bins=40, alpha=0.6, color=COLORS["synthetic"], label="Synthetic", density=True)
    ax.set_xlabel("Gradient L2 Norm")
    ax.set_ylabel("Density")
    ax.set_title("Gradient Norm Distribution")
    ax.legend()
    ratio = grad_results["ratio"]
    ax.text(0.98, 0.98, f"Ratio (H/S): {ratio:.2f}x\np={grad_results['test']['p_value']:.2e}",
            transform=ax.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Loss curves
    ax = axes[2]
    h_losses = grad_results["human_losses"]
    s_losses = grad_results["synthetic_losses"]
    ax.plot(h_losses, color=COLORS["human"], alpha=0.5, linewidth=0.8)
    ax.plot(s_losses, color=COLORS["synthetic"], alpha=0.5, linewidth=0.8)
    h_loss_smooth = np.convolve(h_losses, np.ones(window)/window, mode='valid')
    s_loss_smooth = np.convolve(s_losses, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(h_losses)), h_loss_smooth, color=COLORS["human"], linewidth=2.5, label="Human")
    ax.plot(range(window-1, len(s_losses)), s_loss_smooth, color=COLORS["synthetic"], linewidth=2.5, label="Synthetic")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_gradient_norms.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig2_gradient_norms.png")


def plot_gradient_directions(direction_results):
    """Figure 3: Gradient direction analysis (PCA variance, similarity)."""
    n_layers = len(direction_results)
    if n_layers == 0:
        print("No direction results to plot")
        return

    fig, axes = plt.subplots(1, min(n_layers, 4), figsize=(5 * min(n_layers, 4), 5))
    if min(n_layers, 4) == 1:
        axes = [axes]

    for idx, (layer, data) in enumerate(list(direction_results.items())[:4]):
        ax = axes[idx]
        h_var = np.cumsum(data["human_explained_var"])
        s_var = np.cumsum(data["synthetic_explained_var"])
        components = range(1, len(h_var) + 1)

        ax.plot(components, h_var, color=COLORS["human"], linewidth=2, label="Human")
        ax.plot(components, s_var, color=COLORS["synthetic"], linewidth=2, label="Synthetic")
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label="90% threshold")
        ax.set_xlabel("# PCA Components")
        ax.set_ylabel("Cumulative Variance Explained")
        short_name = layer.split(".")[-2] + "." + layer.split(".")[-1]
        ax.set_title(f"{short_name}")
        ax.legend(fontsize=8)

        info = (f"H rank-90%: {data['human_rank90']}\n"
                f"S rank-90%: {data['synthetic_rank90']}\n"
                f"H cos-sim: {data['human_internal_cosine_sim']:.3f}\n"
                f"S cos-sim: {data['synthetic_internal_cosine_sim']:.3f}")
        ax.text(0.98, 0.02, info, transform=ax.transAxes, fontsize=8,
                va='bottom', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_gradient_directions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig3_gradient_directions.png")


def plot_gradient_direction_summary(direction_results):
    """Figure 3b: Summary bar chart of rank and similarity."""
    if not direction_results:
        return

    layers = list(direction_results.keys())
    short_names = [".".join(l.split(".")[-2:]) for l in layers]

    h_ranks = [direction_results[l]["human_rank90"] for l in layers]
    s_ranks = [direction_results[l]["synthetic_rank90"] for l in layers]
    h_sims = [direction_results[l]["human_internal_cosine_sim"] for l in layers]
    s_sims = [direction_results[l]["synthetic_internal_cosine_sim"] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(layers))
    width = 0.35

    ax = axes[0]
    ax.bar(x - width/2, h_ranks, width, color=COLORS["human"], alpha=0.8, label="Human")
    ax.bar(x + width/2, s_ranks, width, color=COLORS["synthetic"], alpha=0.8, label="Synthetic")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Components for 90% Variance")
    ax.set_title("Gradient Subspace Dimensionality")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.legend()

    ax = axes[1]
    ax.bar(x - width/2, h_sims, width, color=COLORS["human"], alpha=0.8, label="Human")
    ax.bar(x + width/2, s_sims, width, color=COLORS["synthetic"], alpha=0.8, label="Synthetic")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Internal Cosine Similarity")
    ax.set_title("Gradient Self-Similarity (Higher = More Concentrated)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3b_direction_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig3b_direction_summary.png")


def plot_learning_efficiency(eff_results):
    """Figure 4: Learning efficiency comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    fractions = sorted(eff_results["human"].keys(), key=float)
    h_ppls = [eff_results["human"][f]["val_ppl_mean"] for f in fractions]
    s_ppls = [eff_results["synthetic"][f]["val_ppl_mean"] for f in fractions]
    h_n = [eff_results["human"][f]["n_sequences"] for f in fractions]
    s_n = [eff_results["synthetic"][f]["n_sequences"] for f in fractions]

    ax.plot([float(f) * 100 for f in fractions], h_ppls, 'o-', color=COLORS["human"],
            linewidth=2, markersize=8, label="Human Data")
    ax.plot([float(f) * 100 for f in fractions], s_ppls, 's-', color=COLORS["synthetic"],
            linewidth=2, markersize=8, label="Synthetic Data")
    ax.set_xlabel("Fraction of Training Data (%)")
    ax.set_ylabel("Validation Perplexity")
    ax.set_title("Learning Efficiency: Human vs Synthetic Data")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_learning_efficiency.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig4_learning_efficiency.png")


def plot_layer_gradient_ratios(grad_results):
    """Figure 5: Per-layer gradient norm ratios."""
    layer_comp = grad_results.get("layer_comparison", {})
    if not layer_comp:
        return

    # Sort by layer depth
    items = sorted(layer_comp.items())
    # Keep only a manageable number
    if len(items) > 20:
        items = items[::len(items)//20]

    names = [".".join(n.split(".")[-2:]) for n, _ in items]
    ratios = [d["ratio"] for _, d in items]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(names)), ratios, color=[COLORS["human"] if r > 1 else COLORS["synthetic"] for r in ratios], alpha=0.8)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Gradient Norm Ratio (Human / Synthetic)")
    ax.set_title("Per-Layer Gradient Norm Ratio")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_layer_ratios.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig5_layer_ratios.png")


def main():
    print("Loading results...")
    with open(RESULTS_DIR / "all_results.json") as f:
        results = json.load(f)

    print("\nGenerating visualizations...")
    plot_perplexity_distributions(results["exp1_perplexity"])
    plot_gradient_norms(results["exp2_gradient_norms"])
    plot_gradient_directions(results["exp3_gradient_directions"])
    plot_gradient_direction_summary(results["exp3_gradient_directions"])
    plot_learning_efficiency(results["exp4_learning_efficiency"])
    plot_layer_gradient_ratios(results["exp2_gradient_norms"])

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

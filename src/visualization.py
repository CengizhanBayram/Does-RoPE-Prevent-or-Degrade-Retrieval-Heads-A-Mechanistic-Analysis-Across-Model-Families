"""
Visualization functions for all figures in the paper.

All figures are saved under results/{layer}/figures/ as PDF (default) or PNG.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Consistent style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {
    "llama3": "#1f77b4",
    "llama2": "#ff7f0e",
    "qwen": "#2ca02c",
    "olmo": "#d62728",
    "retrieval": "#d62728",
    "non_retrieval": "#1f77b4",
}
MARKERS = {"llama3": "o", "llama2": "s", "qwen": "^", "olmo": "D"}


def _save(fig: plt.Figure, path: str | Path, dpi: int = 300) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved to %s", path)


# ---------------------------------------------------------------------------
# Figure 1 — Retrieval score vs. dimension utility scatter
# ---------------------------------------------------------------------------

def plot_retrieval_vs_utility_scatter(
    results: dict[str, dict],
    save_path: str | Path,
    dpi: int = 300,
) -> None:
    """
    Scatter plot: retrieval score (x) vs. mean dimension utility (y).

    Each point is a (layer, head) pair. Retrieval heads are outlined in red.
    A regression line and Pearson r/p are shown in the legend.

    Args:
        results: Dict keyed by model name. Each value must contain:
            - "retrieval_scores": nested list (n_layers × n_heads)
            - "dimension_utility.per_head_scalar": nested list (n_layers × n_heads)
            - "retrieval_heads": list of [layer, head] pairs
            - "statistical_tests.pearson_r": float
            - "statistical_tests.pearson_p": float
        save_path: Output file path.
        dpi: Output resolution.
    """
    from scipy import stats as sp_stats

    fig, ax = plt.subplots(figsize=(8, 6))

    all_x, all_y = [], []

    for model_name, res in results.items():
        scores = np.array(res["retrieval_scores"])
        utility = np.array(res["dimension_utility"]["per_head_scalar"])
        retrieval_set = {(r[0], r[1]) for r in res["retrieval_heads"]}

        n_layers, n_heads = scores.shape
        layer_colors = np.arange(n_layers)

        x_vals = scores.flatten()
        y_vals = utility.flatten()
        all_x.extend(x_vals.tolist())
        all_y.extend(y_vals.tolist())

        # Color by layer depth
        layer_indices = np.repeat(np.arange(n_layers), n_heads)
        sc = ax.scatter(
            x_vals,
            y_vals,
            c=layer_indices,
            cmap="viridis",
            alpha=0.5,
            marker=MARKERS.get(model_name, "o"),
            s=30,
            label=model_name,
        )

        # Red outline for retrieval heads
        ret_x, ret_y = [], []
        for l in range(n_layers):
            for h in range(n_heads):
                if (l, h) in retrieval_set:
                    ret_x.append(scores[l, h])
                    ret_y.append(utility[l, h])
        if ret_x:
            ax.scatter(
                ret_x,
                ret_y,
                facecolors="none",
                edgecolors="red",
                linewidths=1.5,
                s=60,
                zorder=5,
            )

    # Regression line over all models combined
    if len(all_x) > 2:
        slope, intercept, r_val, p_val, _ = sp_stats.linregress(all_x, all_y)
        x_line = np.linspace(min(all_x), max(all_x), 200)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="black",
            linewidth=1.5,
            linestyle="--",
            label=f"OLS  r={r_val:.3f}, p={p_val:.3e}",
        )

    plt.colorbar(sc, ax=ax, label="Layer depth")
    ax.set_xlabel("Retrieval score (NIAH hit rate)", fontsize=12)
    ax.set_ylabel("Mean dimension utility (L1 norm)", fontsize=12)
    ax.set_title("Figure 1 — Retrieval Score vs. Dimension Utility", fontsize=13)
    ax.legend(fontsize=9)

    _save(fig, save_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 2 — Dimension utility profile by RoPE frequency
# ---------------------------------------------------------------------------

def plot_dimension_utility_profile(
    results: dict[str, dict],
    save_path: str | Path,
    dpi: int = 300,
) -> None:
    """
    Line plot of per-dimension utility sorted by ascending RoPE frequency.

    One subplot per model. Retrieval heads in red, non-retrieval in blue,
    with ± 1 std shaded bands.

    Args:
        results: Dict keyed by model name. Each value must contain
            "dimension_utility.frequency_profile" with keys:
            retrieval_mean, retrieval_std, non_retrieval_mean, non_retrieval_std.
        save_path: Output file path.
        dpi: Output resolution.
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        prof = results[model_name]["dimension_utility"]["frequency_profile"]
        ret_mean = np.array(prof["retrieval_mean"])
        ret_std = np.array(prof["retrieval_std"])
        non_mean = np.array(prof["non_retrieval_mean"])
        non_std = np.array(prof["non_retrieval_std"])
        x = np.arange(len(ret_mean))

        ax.plot(x, non_mean, color=PALETTE["non_retrieval"], label="Non-retrieval", linewidth=1.5)
        ax.fill_between(x, non_mean - non_std, non_mean + non_std, alpha=0.2, color=PALETTE["non_retrieval"])

        ax.plot(x, ret_mean, color=PALETTE["retrieval"], label="Retrieval", linewidth=1.5)
        ax.fill_between(x, ret_mean - ret_std, ret_mean + ret_std, alpha=0.2, color=PALETTE["retrieval"])

        theta = prof.get("theta", "?")
        ax.set_title(f"{model_name}\n(θ={theta:.0f})", fontsize=11)
        ax.set_xlabel("Dimension index (low → high RoPE freq.)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Normalized L1 norm", fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("Figure 2 — Dimension Utility Profile by RoPE Frequency", fontsize=13, y=1.02)
    _save(fig, save_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 3 — OLMo training dynamics
# ---------------------------------------------------------------------------

def plot_olmo_training_dynamics(
    checkpoint_results: list[dict],
    save_path: str | Path,
    dpi: int = 300,
) -> None:
    """
    Dual-axis line plot of retrieval head count and dimension utility over training.

    Args:
        checkpoint_results: List of per-checkpoint result dicts, each with:
            - "step": int
            - "summary.n_retrieval_heads": int
            - "dimension_utility.per_head_scalar": nested list (n_layers × n_heads)
                whose top-20 values are averaged for the right axis.
        save_path: Output file path.
        dpi: Output resolution.
    """
    steps = [r["step"] for r in checkpoint_results]
    n_heads_list = [r["summary"]["n_retrieval_heads"] for r in checkpoint_results]

    # Top-20 mean utility
    top20_utility = []
    for r in checkpoint_results:
        scalar_grid = np.array(r["dimension_utility"]["per_head_scalar"]).flatten()
        top20 = np.sort(scalar_grid)[-20:]
        top20_utility.append(float(top20.mean()))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    color_ret = PALETTE["retrieval"]
    color_util = "#1f77b4"

    ax1.plot(steps, n_heads_list, color=color_ret, linewidth=2, marker="o", markersize=4, label="# Retrieval heads")
    ax2.plot(steps, top20_utility, color=color_util, linewidth=2, marker="s", markersize=4, linestyle="--", label="Top-20 mean utility")

    # Annotate crystallization step: max discrete derivative
    if len(n_heads_list) > 2:
        diffs = np.diff(n_heads_list)
        cryst_idx = int(np.argmax(diffs)) + 1
        cryst_step = steps[cryst_idx]
        ax1.axvline(cryst_step, color="gray", linestyle=":", linewidth=1.5)
        ax1.annotate(
            f"Crystallization\nstep={cryst_step:,}",
            xy=(cryst_step, n_heads_list[cryst_idx]),
            xytext=(cryst_step + (steps[-1] - steps[0]) * 0.03, max(n_heads_list) * 0.8),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=8,
        )

        # Compute lag: at crystallization step, what is the utility trend?
        util_at_cryst = top20_utility[cryst_idx]
        util_before = top20_utility[max(0, cryst_idx - 1)]
        direction = "decreasing" if util_at_cryst < util_before else "increasing/stable"
        ax1.set_title(
            f"Figure 3 — OLMo-2 Training Dynamics\nUtility at crystallization: {direction}",
            fontsize=12,
        )

    ax1.set_xlabel("Training step", fontsize=11)
    ax1.set_ylabel("# Retrieval heads", color=color_ret, fontsize=11)
    ax2.set_ylabel("Top-20 mean dimension utility", color=color_util, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_ret)
    ax2.tick_params(axis="y", labelcolor=color_util)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    _save(fig, save_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 4 — θ comparison heatmaps
# ---------------------------------------------------------------------------

def plot_theta_comparison_heatmaps(
    llama2_results: dict,
    llama3_results: dict,
    save_path: str | Path,
    dpi: int = 300,
) -> None:
    """
    2×2 heatmap grid: (LLaMA-2, LLaMA-3) × (retrieval scores, dimension utility).

    Args:
        llama2_results: Result dict for LLaMA-2.
        llama3_results: Result dict for LLaMA-3.
        save_path: Output file path.
        dpi: Output resolution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    datasets = [
        ("LLaMA-2 (θ=10K)", llama2_results),
        ("LLaMA-3.1 (θ=500K)", llama3_results),
    ]
    metrics = ["retrieval_scores", "per_head_utility"]

    # Compute global color ranges for fair comparison
    all_ret = [
        np.array(r["retrieval_scores"]) for _, r in datasets
    ]
    all_util = [
        np.array(r["dimension_utility"]["per_head_scalar"]) for _, r in datasets
    ]
    vmin_ret, vmax_ret = min(a.min() for a in all_ret), max(a.max() for a in all_ret)
    vmin_util, vmax_util = min(a.min() for a in all_util), max(a.max() for a in all_util)

    cmaps = ["YlOrRd", "viridis"]
    vmins = [vmin_ret, vmin_util]
    vmaxs = [vmax_ret, vmax_util]
    titles = ["Retrieval Score", "Dimension Utility (mean L1 norm)"]

    for col, (model_label, res) in enumerate(datasets):
        for row, (metric_title, data_key, cmap, vmin, vmax) in enumerate(
            zip(titles, metrics, cmaps, vmins, vmaxs)
        ):
            ax = axes[row, col]
            if data_key == "retrieval_scores":
                mat = np.array(res["retrieval_scores"])
            else:
                mat = np.array(res["dimension_utility"]["per_head_scalar"])

            im = ax.imshow(
                mat.T,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel("Layer", fontsize=9)
            ax.set_ylabel("Head index", fontsize=9)
            ax.set_title(f"{model_label}\n{metric_title}", fontsize=10)

    fig.suptitle("Figure 4 — θ Comparison: LLaMA-2 vs LLaMA-3.1", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, save_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 5 — Activation patching results
# ---------------------------------------------------------------------------

def plot_activation_patching_results(
    patching_results: dict,
    save_path: str | Path,
    dpi: int = 300,
) -> None:
    """
    Bar chart comparing NIAH accuracy across four patching conditions.

    Bars are averaged over all retrieval heads; error bars show ± 1 std.

    Args:
        patching_results: Dict mapping (layer, head) → condition dict, as
            returned by ActivationPatcher.run_patching_experiment().
        save_path: Output file path.
        dpi: Output resolution.
    """
    conditions = ["baseline", "low_utility", "random", "high_utility"]
    condition_labels = ["Baseline", "Low-utility\nzeroed", "Random\nzeroed", "High-utility\nzeroed"]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#dd8452"]

    acc_per_cond: dict[str, list[float]] = {c: [] for c in conditions}
    for data in patching_results.values():
        for c in conditions:
            if c in data:
                acc_per_cond[c].append(data[c])

    means = [np.mean(acc_per_cond[c]) if acc_per_cond[c] else 0.0 for c in conditions]
    stds = [np.std(acc_per_cond[c]) if acc_per_cond[c] else 0.0 for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, fontsize=10)
    ax.set_ylabel("NIAH Accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 5 — Activation Patching: Dimension Importance for Retrieval", fontsize=12)

    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Causal effect annotation
    if acc_per_cond["baseline"] and acc_per_cond["low_utility"] and acc_per_cond["random"]:
        ce = np.mean(acc_per_cond["random"]) - np.mean(acc_per_cond["low_utility"])
        direction = "H2 (low-util unused)" if ce < 0 else "unexpected (low-util load-bearing)"
        ax.text(
            0.98, 0.95,
            f"Causal effect = {ce:.3f}\n→ {direction}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
        )

    _save(fig, save_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 6 — NIAH comparison heatmaps
# ---------------------------------------------------------------------------

def plot_niah_comparison(
    normal_acc: np.ndarray,
    masked_acc: np.ndarray,
    context_lengths: list[int],
    needle_positions: list[float],
    save_path: str | Path,
    dpi: int = 300,
) -> None:
    """
    Side-by-side NIAH heatmaps: normal vs. retrieval-heads-masked.

    An optional difference map (masked - normal) is also shown.

    Args:
        normal_acc: (n_lengths, n_positions) accuracy matrix without masking.
        masked_acc: (n_lengths, n_positions) accuracy matrix with masking.
        context_lengths: Row labels.
        needle_positions: Column labels.
        save_path: Output file path.
        dpi: Output resolution.
    """
    diff = masked_acc - normal_acc

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Normal (no mask)", "Retrieval heads masked", "Difference (masked − normal)"]
    mats = [normal_acc, masked_acc, diff]
    cmaps = ["RdYlGn", "RdYlGn", "RdBu_r"]
    vmins = [0, 0, -1]
    vmaxs = [1, 1, 1]

    x_labels = [f"{p:.0%}" for p in needle_positions]
    y_labels = [str(cl) for cl in context_lengths]

    for ax, title, mat, cmap, vmin, vmax in zip(axes, titles, mats, cmaps, vmins, vmaxs):
        im = sns.heatmap(
            mat,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            xticklabels=x_labels,
            yticklabels=y_labels,
            linewidths=0.3,
            linecolor="gray",
            cbar=True,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Needle position", fontsize=9)
        ax.set_ylabel("Context length (tokens)", fontsize=9)

    fig.suptitle("Figure 6 — NIAH Accuracy: Normal vs. Retrieval Heads Masked", fontsize=13)
    plt.tight_layout()
    _save(fig, save_path, dpi=dpi)

"""
Visualization functions for all figures in the paper.

All figures are saved under results/{layer}/figures/ as PDF (default) or PNG.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
# Use non-interactive Agg backend before pyplot is imported.
# If pyplot was already imported elsewhere, this call is a no-op but harmless.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.1)

PALETTE = {
    "llama3": "#1f77b4",
    "llama2": "#ff7f0e",
    "qwen": "#2ca02c",
    "olmo": "#d62728",
    "retrieval": "#d62728",
    "non_retrieval": "#1f77b4",
}
# FIX #9: removed unused `layer_colors` local variable that appeared in the
# original scatter function. Markers dict kept here for reference.
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
        save_path: Output file path.
        dpi: Output resolution.
    """
    from scipy import stats as sp_stats

    fig, ax = plt.subplots(figsize=(8, 6))

    all_x: list[float] = []
    all_y: list[float] = []
    last_sc = None  # keep reference for colorbar

    for model_name, res in results.items():
        scores = np.array(res["retrieval_scores"])
        # P3: prefer the layer-wise z-scored utility (removes the per-layer scale
        # effect that causes vertical banding); fall back to raw L1 if absent.
        du = res["dimension_utility"]
        if du.get("per_head_scalar_zscore") is not None:
            utility = np.array(du["per_head_scalar_zscore"])
        else:
            utility = np.array(du["per_head_scalar"])
        retrieval_set = {(r[0], r[1]) for r in res["retrieval_heads"]}

        n_layers, n_heads = scores.shape
        x_vals = scores.flatten()
        y_vals = utility.flatten()
        all_x.extend(x_vals.tolist())
        all_y.extend(y_vals.tolist())

        # FIX #9: layer_indices replaces the unused layer_colors variable.
        layer_indices = np.repeat(np.arange(n_layers), n_heads)
        last_sc = ax.scatter(
            x_vals,
            y_vals,
            c=layer_indices,
            cmap="viridis",
            alpha=0.5,
            marker=MARKERS.get(model_name, "o"),
            s=30,
            label=model_name,
        )

        ret_x = [scores[l, h] for l, h in retrieval_set if l < n_layers and h < n_heads]
        ret_y = [utility[l, h] for l, h in retrieval_set if l < n_layers and h < n_heads]
        if ret_x:
            ax.scatter(
                ret_x, ret_y,
                facecolors="none",
                edgecolors="red",
                linewidths=1.5,
                s=60,
                zorder=5,
            )

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

    if last_sc is not None:
        plt.colorbar(last_sc, ax=ax, label="Layer depth")

    # P4: annotate the layer-controlled partial correlation (the trustworthy
    # statistic) when available, since the raw OLS r is layer-confounded.
    partials = [res["statistical_tests"].get("partial_spearman_r")
                for res in results.values()
                if res.get("statistical_tests", {}).get("partial_spearman_r") is not None]
    if partials:
        ax.text(
            0.98, 0.02,
            "layer-partial ρ: " + ", ".join(
                f"{m}={res['statistical_tests']['partial_spearman_r']:.3f}"
                for m, res in results.items()
                if res.get("statistical_tests", {}).get("partial_spearman_r") is not None),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
        )

    ax.set_xlabel("Retrieval score (NIAH hit rate)", fontsize=12)
    ax.set_ylabel("Dimension utility (layer-wise z-score)", fontsize=12)
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
            dimension_utility.frequency_profile with keys:
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
        theta_str = f"{theta:.0f}" if isinstance(theta, (int, float)) else str(theta)
        ax.set_title(f"{model_name}\n(θ={theta_str})", fontsize=11)
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
    Dual-axis line plot of retrieval head count and mean dimension utility over training.

    Args:
        checkpoint_results: List of per-checkpoint result dicts, each with:
            - "step": int
            - "summary.n_retrieval_heads": int
            - "dimension_utility.per_head_scalar": nested list (n_layers × n_heads).
        save_path: Output file path.
        dpi: Output resolution.
    """
    import re

    # P14: x-axis is tokens SEEN (B), parsed from the revision string
    # ("stage1-step40000-tokens168B"); fall back to step if unavailable.
    def _tokens_b(r):
        m = re.search(r"tokens(\d+)B", r.get("revision", ""))
        return int(m.group(1)) if m else None

    tokens = [_tokens_b(r) for r in checkpoint_results]
    if all(t is not None for t in tokens):
        x_vals, x_label = tokens, "Tokens seen (B)"
    else:
        x_vals, x_label = [r["step"] for r in checkpoint_results], "Training step"
    steps = x_vals  # kept for the crystallization annotation below
    n_heads_list = [r["summary"]["n_retrieval_heads"] for r in checkpoint_results]

    top20_utility = []
    for r in checkpoint_results:
        scalar_grid = np.array(r["dimension_utility"]["per_head_scalar"]).flatten()
        top20 = np.sort(scalar_grid)[-20:]
        top20_utility.append(float(top20.mean()))

    # P9: lead-lag — do retrieval-head changes PRECEDE utility changes?
    lead_lag_note = ""
    if len(n_heads_list) >= 4:
        from src.stats_utils import lead_lag
        ll = lead_lag(n_heads_list, top20_utility)
        if ll["best_lag"] == ll["best_lag"]:  # not NaN
            rel = ("heads LEAD utility" if ll["best_lag"] > 0
                   else "utility LEADS heads" if ll["best_lag"] < 0
                   else "synchronous")
            lead_lag_note = (f"lead-lag={ll['best_lag']:+d} ({rel}), "
                             f"peak r={ll['peak_corr']:.2f}, p={ll['p_value']:.3f}")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    color_ret = PALETTE["retrieval"]
    color_util = "#1f77b4"

    ax1.plot(steps, n_heads_list, color=color_ret, linewidth=2, marker="o", markersize=4, label="# Retrieval heads")
    ax2.plot(steps, top20_utility, color=color_util, linewidth=2, marker="s", markersize=4, linestyle="--", label="Top-20 mean utility")

    if len(n_heads_list) > 2:
        # P7 FIX: crystallization = ONSET of the sustained rise, not the single
        # largest first-difference. argmax(diff) is fooled by a later dip→recovery
        # spike (e.g. the ~3050B outlier), mislabelling the onset. Instead take
        # the FIRST checkpoint whose (smoothed) head count crosses the midpoint
        # between the early baseline and the peak — robust to later dips.
        from src.stats_utils import crystallization_onset
        cryst_idx = crystallization_onset(n_heads_list)
        cryst_x = steps[cryst_idx]
        unit = "B tokens" if x_label.startswith("Tokens") else "step"
        ax1.axvline(cryst_x, color="gray", linestyle=":", linewidth=1.5)
        ax1.annotate(
            f"Crystallization onset\n~{cryst_x:,} {unit}",
            xy=(cryst_x, n_heads_list[cryst_idx]),
            xytext=(cryst_x + (steps[-1] - steps[0]) * 0.03, max(n_heads_list) * 0.8),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=8,
        )

        util_at_cryst = top20_utility[cryst_idx]
        util_before = top20_utility[max(0, cryst_idx - 1)]
        direction = "decreasing" if util_at_cryst < util_before else "increasing/stable"
        subtitle = f"Utility at crystallization: {direction}"
        if lead_lag_note:
            subtitle += f"  |  {lead_lag_note}"
        ax1.set_title(f"Figure 3 — OLMo-2 Training Dynamics\n{subtitle}", fontsize=11)
    else:
        ax1.set_title("Figure 3 — OLMo-2 Training Dynamics", fontsize=12)

    ax1.set_xlabel(x_label, fontsize=11)
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

    model_pairs = [
        ("LLaMA-2 (θ=10K)", llama2_results),
        ("LLaMA-3.1 (θ=500K)", llama3_results),
    ]
    # FIX #10: removed unused `metrics` variable; data_keys are now inline.
    row_specs = [
        ("Retrieval Score",                "retrieval_scores",      "YlOrRd"),
        ("Dimension Utility (mean L1 norm)", "per_head_utility",   "viridis"),
    ]

    all_ret = [np.array(r["retrieval_scores"]) for _, r in model_pairs]
    all_util = [np.array(r["dimension_utility"]["per_head_scalar"]) for _, r in model_pairs]
    vmin_ret, vmax_ret = min(a.min() for a in all_ret), max(a.max() for a in all_ret)
    vmin_util, vmax_util = min(a.min() for a in all_util), max(a.max() for a in all_util)
    vmins = [vmin_ret, vmin_util]
    vmaxs = [vmax_ret, vmax_util]

    for col, (model_label, res) in enumerate(model_pairs):
        for row, (metric_title, data_key, cmap) in enumerate(row_specs):
            ax = axes[row, col]
            mat = (
                np.array(res["retrieval_scores"])
                if data_key == "retrieval_scores"
                else np.array(res["dimension_utility"]["per_head_scalar"])
            )
            im = ax.imshow(
                mat.T,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                vmin=vmins[row],
                vmax=vmaxs[row],
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

    # Frequency-axis conditions (paper §6) are shown only when present.
    has_freq = any("low_freq" in d and "high_freq" in d for d in patching_results.values())
    if has_freq:
        conditions += ["low_freq", "high_freq"]
        condition_labels += ["Low-freq\nzeroed", "High-freq\nzeroed"]
        colors += ["#8172b3", "#937860"]

    acc_per_cond: dict[str, list[float]] = {c: [] for c in conditions}
    for data in patching_results.values():
        for c in conditions:
            if c in data:
                acc_per_cond[c].append(float(data[c]))

    means = [np.mean(acc_per_cond[c]) if acc_per_cond[c] else 0.0 for c in conditions]
    stds = [np.std(acc_per_cond[c]) if acc_per_cond[c] else 0.0 for c in conditions]

    fig, ax = plt.subplots(figsize=(11, 5) if has_freq else (8, 5))
    x = np.arange(len(conditions))
    bars = ax.bar(
        x, means, yerr=stds, capsize=5,
        color=colors, alpha=0.85, edgecolor="black", linewidth=0.8,
    )
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
            ha="center", va="bottom", fontsize=9,
        )

    annotations = []
    if acc_per_cond["baseline"] and acc_per_cond["low_utility"] and acc_per_cond["random"]:
        ce = np.mean(acc_per_cond["random"]) - np.mean(acc_per_cond["low_utility"])
        direction = "H2 (low-util unused)" if ce < 0 else "unexpected (low-util load-bearing)"
        annotations.append(f"Utility effect = {ce:.3f}\n→ {direction}")
    if has_freq and acc_per_cond["low_freq"] and acc_per_cond["high_freq"]:
        fe = np.mean(acc_per_cond["low_freq"]) - np.mean(acc_per_cond["high_freq"])
        fdir = ("frequency-specific\n(high-freq load-bearing)" if fe > 0
                else "frequency-specific\n(low-freq load-bearing)" if fe < 0
                else "no frequency specificity")
        annotations.append(f"Frequency effect = {fe:.3f}\n→ {fdir}")
    if annotations:
        ax.text(
            0.98, 0.95,
            "\n\n".join(annotations),
            transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
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
    Side-by-side NIAH heatmaps: normal vs. retrieval-heads-masked, plus diff map.

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
        sns.heatmap(
            mat,
            ax=ax,
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            annot=True, fmt=".2f",
            xticklabels=x_labels,
            yticklabels=y_labels,
            linewidths=0.3, linecolor="gray",
            cbar=True,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Needle position", fontsize=9)
        ax.set_ylabel("Context length (tokens)", fontsize=9)

    fig.suptitle("Figure 6 — NIAH Accuracy: Normal vs. Retrieval Heads Masked", fontsize=13)
    plt.tight_layout()
    _save(fig, save_path, dpi=dpi)

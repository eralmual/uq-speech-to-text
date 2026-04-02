import argparse
import os
import glob
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def load_results(results_dir="results"):
    """Load all CSV result files from the results directory."""
    results = {}
    for csv_path in glob.glob(os.path.join(results_dir, "*", "*.csv")):
        name = os.path.basename(os.path.dirname(csv_path))
        results[name] = pd.read_csv(csv_path)
    return results


def plot_summary(results: dict[str, pd.DataFrame], metric: str = "R"):
    """Plot mean, min, and max of a metric across all models for each experiment.

    Uses a box/candlestick style: a bar for the mean with error bars spanning
    min to max.  When the raw mean of a metric is negative the absolute value
    is plotted and an asterisk (*) is appended to the experiment label in the
    legend / x-axis.

    Args:
        results: dict mapping experiment name to its DataFrame.
        metric: column name to summarise (default "R").
    """
    names = sorted(results.keys())

    raw_means = [results[n][metric].mean() for n in names]
    raw_mins = [results[n][metric].min() for n in names]
    raw_maxs = [results[n][metric].max() for n in names]

    # Use absolute values; track which ones were negative
    means = [abs(v) for v in raw_means]
    mins = [abs(v) for v in raw_mins]
    maxs = [abs(v) for v in raw_maxs]
    is_neg = [v < 0 for v in raw_means]

    # For each experiment ensure min <= mean <= max after abs()
    for i in range(len(names)):
        lo = min(mins[i], means[i], maxs[i])
        hi = max(mins[i], means[i], maxs[i])
        mins[i], maxs[i] = lo, hi

    labels = [f"{n} *" if neg else n for n, neg in zip(names, is_neg)]

    x = np.arange(len(names))
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(names))
    colors = [cmap(i) for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Error bars from min to max (candlestick style)
    err_low = [m - lo for m, lo in zip(means, mins)]
    err_high = [hi - m for m, hi in zip(means, maxs)]

    bars = ax.bar(
        x, means, width=0.45, color=colors, alpha=0.85,
        yerr=[err_low, err_high],
        error_kw=dict(ecolor="black", capsize=8, capthick=2, elinewidth=1.5),
    )

    # Value annotations
    for i, bar in enumerate(bars):
        # Mean label
        ax.annotate(
            f"{means[i]:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, means[i]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
        # Min label
        ax.annotate(
            f"{mins[i]:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, mins[i]),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center", va="top", fontsize=7, fontweight="bold",
        )
        # Max label
        ax.annotate(
            f"{maxs[i]:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, maxs[i]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=7, fontweight="bold",
        )

    # Legend: one entry per experiment
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=0.85, label=labels[i]) for i in range(len(names))]
    ax.legend(handles=legend_elements)

    use_abs = any(is_neg)
    ylabel = f"|{metric}|" if use_abs else metric

    ax.set_xlabel("Experiment")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric} Summary (Mean with Min–Max range)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experiment result summaries.")
    parser.add_argument("-r",
        "--results_dir",
        nargs="?",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Path to the results folder (default: ./results next to this script)",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        default=["R"],
        help="Metric column(s) to plot (default: R)",
    )
    args = parser.parse_args()

    results = load_results(args.results_dir)
    for m in args.metric:
        plot_summary(results, metric=m)

"""Bar-plot visualiser for CMA-ES optimization progress.

Reads an ``evolution_history.csv`` produced by :mod:`optimize` and renders one
bar per evaluated sample.  Bars are laid out in evaluation order and grouped per
generation, with blank gaps, dashed separators and shaded background bands so the
generation boundaries are obvious.  Bars are coloured by the optimized parameter
(dropout rate / temperature) via a continuous colourbar.

Usage examples::

    python src/plot/plot_progress.py -i results/optim/mcd/evolution_history.csv
    python src/plot/plot_progress.py -i results/optim/mcd/evolution_history.csv \
        -o results/optim/mcd/progress.pdf
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, PowerNorm

# --- Publication-friendly style (mirrors src/plot/plot_avg_results.py) --------
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["cmr10", "Computer Modern Roman", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["axes.formatter.use_mathtext"] = True
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 11

# Required columns plus the value we plot on the y-axis.
_REQUIRED_COLUMNS = {"generation", "individual", "eval"}
_VALUE_COLUMN = "mean_R"
# Candidate names for the optimised hyper-parameter, in preference order.
_PARAM_CANDIDATES = ("dropout_rate", "temperature")


def load_history(csv_path: str) -> tuple[pd.DataFrame, str]:
    """Load and validate an ``evolution_history.csv``.

    Args:
        csv_path: path to the CSV written by ``optimize.py``.

    Returns:
        A ``(dataframe, param_column)`` tuple where ``param_column`` is the name
        of the optimised hyper-parameter detected in the file.

    Raises:
        FileNotFoundError: if ``csv_path`` does not exist.
        ValueError: if mandatory columns or a parameter column are missing.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No such file: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required column(s): {sorted(missing)}"
        )
    if _VALUE_COLUMN not in df.columns:
        raise ValueError(f"{csv_path} is missing the value column '{_VALUE_COLUMN}'")

    param_column = next((c for c in _PARAM_CANDIDATES if c in df.columns), None)
    if param_column is None:
        raise ValueError(
            f"{csv_path} has no optimized-parameter column "
            f"(looked for {_PARAM_CANDIDATES})"
        )

    df = df.sort_values(["generation", "individual", "eval"]).reset_index(drop=True)
    return df, param_column


def plot_individual(df: pd.DataFrame, param_column: str,
                    title: str | None) -> plt.Figure:
    """Bar plot in evaluation order, grouped and separated per generation.

    Each evaluated sample is a bar whose height is the R score.  A blank gap, a
    dashed separator and an alternating shaded band mark the boundary between
    successive generations.  Bars are coloured by the optimized parameter via a
    continuous colourbar.
    """
    values, ylabel = df[_VALUE_COLUMN], r"$R$"

    gap = 1.0  # blank space (in bar slots) inserted between generations
    positions: list[float] = []
    gen_spans: dict[int, list[float]] = {}
    cursor = 0.0
    prev_gen = None
    for gen in df["generation"]:
        if prev_gen is not None and gen != prev_gen:
            cursor += gap
        positions.append(cursor)
        gen_spans.setdefault(int(gen), []).append(cursor)
        cursor += 1.0
        prev_gen = gen

    positions = np.asarray(positions)

    # Colour bars by the optimised parameter value.  Dropout values cluster at
    # the low end with a few large outliers, so a power-law norm (gamma < 1)
    # stretches the dense low region for finer colour granularity.  Temperature
    # spans its range more evenly, so a plain linear scale is clearer there.
    param_values = df[param_column].to_numpy()
    is_temperature = param_column == "temperature"
    vmin, vmax = param_values.min(), param_values.max()
    if is_temperature:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = PowerNorm(gamma=0.35, vmin=vmin, vmax=vmax)
    cmap = sns.color_palette("viridis", as_cmap=True)
    colors = cmap(norm(param_values))

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(positions) + 2), 6))

    # Alternating background bands per generation, plus dashed separators.
    ordered_gens = sorted(gen_spans)
    for band_idx, gen in enumerate(ordered_gens):
        xs = gen_spans[gen]
        left = min(xs) - 0.5
        right = max(xs) + 0.5
        if band_idx % 2 == 1:
            ax.axvspan(left, right, color="0.92", zorder=0)
        if band_idx > 0:
            sep = left - gap / 2
            ax.axvline(sep, color="0.6", linestyle="--", linewidth=1, zorder=1)

    bars = ax.bar(
        positions, values, width=0.9, color=colors,
        edgecolor="black", linewidth=0.4, zorder=2,
    )

    # Annotate each bar with its individual index and score.
    for pos, bar, (_, row), val in zip(positions, bars, df.iterrows(), values):
        offset = 3 if val >= 0 else -3
        va = "bottom" if val >= 0 else "top"
        ax.annotate(
            f"{val:.3f}",
            xy=(pos, val), xytext=(0, offset), textcoords="offset points",
            ha="center", va=va, fontsize=7,
        )

    # Generation-centred labels beneath the axis.
    gen_centers = [np.mean(gen_spans[g]) for g in ordered_gens]
    ax.set_xticks(gen_centers)
    ax.set_xticklabels([f"Gen {g}" for g in ordered_gens])
    ax.tick_params(axis="x", length=0)

    # Per-bar (individual) minor ticks for finer reference.
    ax.set_xticks(positions, minor=True)
    ax.set_xticklabels(
        [f"{i}" for i in df["individual"]], minor=True, fontsize=7
    )
    ax.tick_params(axis="x", which="minor", labelsize=7, pad=14)

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Generation (individual index below each bar)")
    ax.set_title(title or "Optimization progress per generation")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.margins(x=0.01)

    # Colourbar for the optimized parameter.
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, pad=0.02)
    if is_temperature:
        # Evenly spaced round ticks across the linear temperature range.
        candidate_ticks = np.arange(0.0, vmax + 0.2, 0.2)
        ticks = candidate_ticks[(candidate_ticks >= vmin - 1e-9) & (candidate_ticks <= vmax + 1e-9)]
    else:
        # Dense round ticks in the low range, sparser round ticks above 0.4,
        # matching the power-law scale's extra resolution at the low end.
        candidate_ticks = np.concatenate([
            np.arange(0.0, 0.41, 0.1),  # 0.0, 0.1, ..., 0.4
            np.arange(0.5, vmax + 1e-9, 0.1),  # 0.5, 0.6, ...
        ])
        ticks = candidate_ticks[(candidate_ticks >= vmin) & (candidate_ticks <= vmax)]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
    cbar.set_label(param_column.replace("_", " "))

    fig.tight_layout()
    return fig


# python src/postprocess/plot_progress.py -i results/optim/lmcd/evolution_history.csv -o out.png
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bar-plot CMA-ES optimization progress from evolution_history.csv."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to evolution_history.csv",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Path to save the figure. If omitted, the plot is shown interactively.",
    )
    parser.add_argument("--title", default=None, help="Custom plot title.")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", font="serif")
    # Re-apply LaTeX-safe rcParams that seaborn's theme overrides.
    matplotlib.rcParams["font.serif"] = ["cmr10", "Computer Modern Roman", "DejaVu Serif"]
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["axes.unicode_minus"] = False

    df, param_column = load_history(args.input)

    title = args.title
    if title is None:
        method = os.path.basename(os.path.dirname(os.path.abspath(args.input)))
        title = "Optimization progress per generation"
        if method:
            title += f" for {method.upper()}"

    fig = plot_individual(df, param_column, title)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        fig.savefig(args.output, bbox_inches="tight", dpi=300)
        print(f"Saved figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

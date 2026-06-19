import argparse
import os
import glob
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from wilcoxon import pairwise_tests, significance_stars

# Shared with wilcoxon.py so colors stay consistent across plots.
# Colors taken from seaborn's default "deep" palette.
_DEEP = sns.color_palette("deep")
METHOD_COLORS = {
    "lmcd": _DEEP[2],  # green
    "mcd": _DEEP[0],   # blue
    "ts": _DEEP[8],    # yellow
}


def _format_label(class_name: str, exp_name: str) -> str:
    """Method name in caps, e.g. 'LMCD'.

    The method name is taken from the leading token of ``class_name`` (the text
    before the first ``-``).  This handles both the nested ``optim`` layout,
    where each class folder is already the method name (e.g. ``lmcd``), and the
    flat ``experiments`` layout, where the class folder is a full experiment
    name (e.g. ``lmcd-iterations_10-dropout_0.037-...``).
    """
    return class_name.split("-")[0].upper()


def _method_color(label: str):
    """Color for a label, keyed on the leading method name (case-insensitive)."""
    return METHOD_COLORS.get(label.split()[0].lower(), plt.colormaps["tab10"](7))

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Match the font used in written/optim_paper/main.tex (default LaTeX
# article class -> Computer Modern serif at 11pt).
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['cmr10', 'Computer Modern Roman', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['axes.formatter.use_mathtext'] = True
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11


def _lower_is_better(metric: str) -> bool:
    """Return True when a *lower* value of ``metric`` is better.

    Metrics whose name references a word error rate or a generic error are
    minimised; everything else (e.g. ``R`` correlation) is maximised.
    """
    name = metric.lower()
    return "wer" in name or "error" in name


def load_classes(parent_dir: str, metric: str = "R"):
    """Select the best experiment per class under ``parent_dir``.

    Each immediate subfolder of ``parent_dir`` is treated as a *class*.  Within
    a class every ``<experiment>/<experiment>.csv`` file is read and ranked by
    the mean of ``metric``; the best experiment is kept.  Whether "best" means
    the highest or lowest mean is inferred from the metric name (see
    :func:`_lower_is_better`).  Classes without any CSV files are skipped.

    Args:
        parent_dir: directory containing one subfolder per class (e.g. ``optim``).
        metric: column used to rank experiments within each class.

    Returns:
        dict mapping ``"<class> (<experiment>)"`` to the chosen experiment's
        DataFrame.
    """
    lower_better = _lower_is_better(metric)
    chosen: dict[str, pd.DataFrame] = {}

    for class_name in sorted(os.listdir(parent_dir)):
        class_dir = os.path.join(parent_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        best_df = None
        best_exp = None
        best_score = None
        # Discover CSVs directly inside the class folder (flat ``experiments``
        # layout) as well as one level deeper (nested ``optim`` layout).
        csv_paths = glob.glob(os.path.join(class_dir, "*.csv")) + glob.glob(
            os.path.join(class_dir, "*", "*.csv")
        )
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            if metric not in df.columns:
                continue
            score = df[metric].mean()
            if pd.isna(score):
                continue
            if (
                best_score is None
                or (lower_better and score < best_score)
                or (not lower_better and score > best_score)
            ):
                best_score = score
                best_df = df
                best_exp = os.path.basename(os.path.dirname(csv_path))

        if best_df is not None:
            chosen[_format_label(class_name, best_exp)] = best_df

    return chosen


def plot_summary(results: dict[str, pd.DataFrame], metric: str = "R"):
    """Bar plot of each experiment's mean metric with standard-deviation bars.

    Bars show the mean across models with symmetric error bars spanning one
    standard deviation.  When the raw mean of a metric is negative its absolute
    value is plotted (the standard deviation is sign-invariant) and an asterisk
    (*) is appended to the experiment label.

    Args:
        results: dict mapping experiment name to its DataFrame.
        metric: column name to summarise (default "R").
    """
    # Order bars by their plotted height (|mean|), lowest to highest.
    def _bar_height(name: str) -> float:
        vals = results[name][metric].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        return abs(float(np.mean(vals))) if vals.size else float("nan")

    names = sorted(results.keys(), key=_bar_height)

    # Build long-form data. Flip the sign of negative-mean groups so the bar
    # height is |mean| while the (sign-invariant) standard deviation is kept.
    records = []
    is_neg = []
    for n in names:
        vals = results[n][metric].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        neg = bool(np.mean(vals) < 0) if vals.size else False
        is_neg.append(neg)
        if neg:
            vals = -vals
        for v in vals:
            records.append({"label": n, "value": v})

    labels = [f"{n} *" if neg else n for n, neg in zip(names, is_neg)]
    label_map = dict(zip(names, labels))
    data = pd.DataFrame(records)
    data["label"] = data["label"].map(label_map)

    order = [label_map[n] for n in names]
    palette = {label_map[n]: _method_color(n) for n in names}

    sns.set_theme(style="whitegrid", font="serif")
    # Re-apply LaTeX-safe rcParams that seaborn's theme overrides.
    matplotlib.rcParams["font.serif"] = ["cmr10", "Computer Modern Roman", "DejaVu Serif"]
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        data=data, x="label", y="value", order=order,
        hue="label", palette=palette,
        legend=False,
        errorbar="sd", capsize=0.15, width=0.65, alpha=0.9,
        edgecolor="black", linewidth=0.8,
        err_kws=dict(color="black", linewidth=1.3), ax=ax,
    )

    # Annotate each bar with its mean height.
    means = {label_map[n]: _bar_height(n) for n in names}
    for patch, lbl in zip(ax.patches, order):
        ax.annotate(
            f"{means[lbl]:.3f}",
            xy=(patch.get_x() + 2*patch.get_width() / 3, patch.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=10,
        )

    use_abs = any(is_neg)
    ylabel = f"|{metric}|" if use_abs else metric

    ax.set_xlabel("Method", labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_title(f"{metric} between uncertainty score and WER", pad=12)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylim(0, 1.19)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)

    # Pairwise paired-Wilcoxon significance brackets (R only, >=2 methods).
    if metric == "R" and len(names) >= 2:
        pair_df = pairwise_tests({n: results[n] for n in names}, alpha=0.05)
        name_to_x = {n: i for i, n in enumerate(names)}
        tops = []
        for n in names:
            vals = results[n][metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            tops.append(abs(float(np.mean(vals))) + float(np.std(vals)) if vals.size else 0.0)
        baseline = max(tops) if tops else 0.0
        rows = sorted(
            pair_df.iterrows(),
            key=lambda r: abs(name_to_x[r[1]["method_a"]] - name_to_x[r[1]["method_b"]]),
        )
        # Distribute the brackets within the headroom between the tallest
        # bar+std and the y-axis cap of 1.19 so nothing is clipped.
        n_levels = max(len(rows), 1)
        headroom = max(1.19 - baseline, 0.0)
        step = headroom / (n_levels + 1)
        bar_h = step * 0.2
        for level, (_, row) in enumerate(rows):
            x1, x2 = sorted((name_to_x[row["method_a"]], name_to_x[row["method_b"]]))
            y = baseline + step * (level + 1)
            ax.plot([x1, x1, x2, x2], [y, y + bar_h, y + bar_h, y],
                    color="black", linewidth=1.2, zorder=5)
            ax.text((x1 + x2) / 2, y + bar_h, significance_stars(row["p_value"]),
                    ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experiment result summaries.")
    parser.add_argument("-r",
        "--results_dir",
        nargs="?",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "optim"),
        help="Parent folder whose subfolders are classes (default: ./optim next to this script)",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        default=["R"],
        help="Metric column(s) to plot (default: R)",
    )
    args = parser.parse_args()

    for m in args.metric:
        results = load_classes(args.results_dir, metric=m)
        plot_summary(results, metric=m)

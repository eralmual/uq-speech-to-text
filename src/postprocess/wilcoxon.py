"""Validate and compare UQ method results with Wilcoxon tests.

Point this at an optimisation folder (e.g. ``results/optim``). Each subfolder is
a UQ method; its best run (highest mean ``R``, matching ``optimize.py``) is used.

    python src/postprocess/wilcoxon.py results/optim
    python src/postprocess/wilcoxon.py results/optim -o results/optim/wilcoxon
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

METRIC = "R"
ID_COLUMN = "Model ID"


def discover_methods(optim_dir: str) -> dict[str, pd.DataFrame]:
    """Map each method under ``optim_dir`` to its best run's DataFrame.

    Works for both the nested optim layout (``optim/<method>/<run>/file.csv``)
    and the flat experiments layout (``experiments/<run>/file.csv``). Each CSV's
    method label is the prefix before the first ``-`` of its first path
    component relative to ``optim_dir``; runs sharing a label are grouped and the
    one with the highest mean ``R`` is kept.
    """
    best: dict[str, tuple[pd.DataFrame, float]] = {}
    for root, _, files in os.walk(optim_dir):
        for name in files:
            if not name.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(root, name))
            if METRIC not in df.columns:
                continue
            df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")
            score = df[METRIC].mean()
            if not np.isfinite(score):
                continue
            rel = os.path.relpath(root, optim_dir)
            top = rel.split(os.sep)[0] if rel != os.curdir else name
            label = top.split("-")[0]
            if label not in best or score > best[label][1]:
                best[label] = (df, score)
    if not best:
        raise ValueError(f"No method results with an '{METRIC}' column under {optim_dir!r}.")
    return {label: df for label, (df, _) in sorted(best.items())}


def validity_tests(methods: dict[str, pd.DataFrame], alpha: float) -> pd.DataFrame:
    """One-sample Wilcoxon test that each method's R is greater than 0."""
    rows = []
    for label, df in methods.items():
        values = df[METRIC].to_numpy()
        values = values[np.isfinite(values)]
        if values.size and np.count_nonzero(values) > 0:
            _, p = stats.wilcoxon(values, alternative="greater")
        else:
            p = float("nan")
        rows.append({
            "method": label,
            "n": values.size,
            "median": float(np.median(values)) if values.size else float("nan"),
            "p_value": p,
            "verdict": "significant" if p < alpha else "not significant",
        })
    return pd.DataFrame(rows)


def pairwise_tests(methods: dict[str, pd.DataFrame], alpha: float) -> pd.DataFrame:
    """Paired two-sided Wilcoxon test for every pair of methods."""
    names = list(methods)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = methods[names[i]], methods[names[j]]
            merged = a.merge(b, on=ID_COLUMN, suffixes=("_a", "_b"))
            diffs = merged[f"{METRIC}_a"].to_numpy() - merged[f"{METRIC}_b"].to_numpy()
            if diffs.size and np.count_nonzero(diffs) > 0:
                _, p = stats.wilcoxon(diffs, alternative="two-sided")
            else:
                p = float("nan")
            rows.append({
                "method_a": names[i],
                "method_b": names[j],
                "n_pairs": diffs.size,
                "median_diff": float(np.median(diffs)) if diffs.size else float("nan"),
                "p_value": p,
                "verdict": "significant" if p < alpha else "not significant",
            })
    return pd.DataFrame(rows)


def significance_stars(p: float) -> str:
    """Star notation for a p-value (``NS`` when not significant or undefined)."""
    if not np.isfinite(p):
        return "NS"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "NS"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("optim_dir", help="Folder with one subfolder per method.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument("-o", "--output", default=None,
                        help="Prefix to save '<prefix>_validity.csv' and '<prefix>_pairwise.csv'.")
    args = parser.parse_args()

    methods = discover_methods(args.optim_dir)
    print("Selected best run per method (by mean R):", ", ".join(methods))

    validity_df = validity_tests(methods, args.alpha)
    pairwise_df = pairwise_tests(methods, args.alpha)

    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print("\n" + "=" * 72)
    print(f"Validity: Wilcoxon signed-rank of 'R' vs 0 (H1: greater, alpha={args.alpha})")
    print("-" * 72)
    print(validity_df.to_string(index=False))
    print("\n" + "=" * 72)
    print(f"Pairwise: paired Wilcoxon signed-rank of 'R' (H1: two-sided, alpha={args.alpha})")
    print("-" * 72)
    print(pairwise_df.to_string(index=False) if not pairwise_df.empty
          else "Only one method given; no pairwise comparison.")
    print("=" * 72)

    if args.output:
        prefix = os.path.splitext(args.output)[0]
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
        validity_df.to_csv(f"{prefix}_validity.csv", index=False)
        pairwise_df.to_csv(f"{prefix}_pairwise.csv", index=False)
        print(f"\nSaved: {prefix}_validity.csv and {prefix}_pairwise.csv")


if __name__ == "__main__":
    main()

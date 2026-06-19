# Results Provenance

This note maps the numeric claims in `paper.tex` to saved result artifacts. It is
intended as an internal camera-ready audit trail, not as manuscript text.

## Optimization Results on Calibration Split

The reported optimal hyperparameters and calibration mean Pearson `R` values are
the best observed rows in each `evolution_history.csv`, selected by maximum
`mean_R`.

| Method | Reported value in paper | Exact source value | Source |
|---|---:|---:|---|
| TS temperature | 1.126 | 1.125754261830254 | `results/optim/ts/evolution_history.csv`, generation 5, individual 2, eval 19 |
| TS calibration mean `R` | 0.539 | 0.5392746296305527 | `results/optim/ts/evolution_history.csv`, generation 5, individual 2, eval 19 |
| MCD dropout | 0.003 | 0.0033767203579313 | `results/optim/mcd/evolution_history.csv`, generation 1, individual 2, eval 3 |
| MCD calibration mean `R` | 0.464 | 0.463654354017311 | `results/optim/mcd/evolution_history.csv`, generation 1, individual 2, eval 3 |
| LMCD dropout | 0.037 | 0.0371378673157789 | `results/optim/lmcd/evolution_history.csv`, generation 2, individual 0, eval 5 |
| LMCD calibration mean `R` | 0.716 | 0.716078890803388 | `results/optim/lmcd/evolution_history.csv`, generation 2, individual 0, eval 5 |

## Held-out Test Results

The reported test mean Pearson `R` values are means across the 10 fold-level rows
in each final experiment CSV.

| Method | Reported test mean `R` | Exact source mean `R` | Test median `R` | Mean WER in source CSV | Source |
|---|---:|---:|---:|---:|---|
| TS | 0.439 | 0.4393356208344142 | 0.4262280488895974 | 0.45649040801913376 | `results/experiments/ts-temperature_1.126-dataset_test-folds_10/ts-temperature_1.126.csv` |
| MCD | 0.469 | 0.46901715700871816 | 0.4657641522086836 | 0.45649040801913376 | `results/experiments/mcd-iterations_10-dropout_0.003-dataset_test-folds_10/mcd-iterations_10-dropout_0.003.csv` |
| LMCD | 0.679 | 0.6788428370678947 | 0.788769381034713 | 0.7063214387559335 | `results/experiments/lmcd-iterations_10-dropout_0.037-dataset_test-folds_10/lmcd-iterations_10-dropout_0.037.csv` |

## Wilcoxon Validity Tests

Calibration validity values are stored under `results/optim`; test validity
values were generated from `results/experiments` with:

```powershell
& c:\Users\soulf\maestria\Scripts\python.exe src/postprocess/wilcoxon.py results/experiments -o results/experiments/wilcoxon
```

| Split | Method | Reported median `R` | Exact median `R` | Exact p-value | Source |
|---|---|---:|---:|---:|---|
| Calibration | LMCD | 0.7965 | 0.7965030961211133 | 0.0009765625 | `results/optim/wilcoxon_validity.csv` |
| Calibration | MCD | 0.5068 | 0.5067705861760285 | 0.0009765625 | `results/optim/wilcoxon_validity.csv` |
| Calibration | TS | 0.6634 | 0.6634037822931228 | 0.0009765625 | `results/optim/wilcoxon_validity.csv` |
| Test | LMCD | 0.7888 | 0.788769381034713 | 0.001953125 | `results/experiments/wilcoxon_validity.csv` |
| Test | MCD | 0.4658 | 0.4657641522086836 | 0.0009765625 | `results/experiments/wilcoxon_validity.csv` |
| Test | TS | 0.4262 | 0.4262280488895974 | 0.001953125 | `results/experiments/wilcoxon_validity.csv` |

## Wilcoxon Pairwise Tests

| Split | Comparison | Reported p-value | Exact p-value | Exact median difference | Source |
|---|---|---:|---:|---:|---|
| Calibration | LMCD vs MCD | 0.0020 | 0.001953125 | 0.24850264277187706 | `results/optim/wilcoxon_pairwise.csv` |
| Calibration | LMCD vs TS | 0.027 | 0.02734375 | 0.1484560920243233 | `results/optim/wilcoxon_pairwise.csv` |
| Calibration | MCD vs TS | 0.084 | 0.083984375 | -0.10004655074755375 | `results/optim/wilcoxon_pairwise.csv` |
| Test | LMCD vs MCD | 0.020 | 0.01953125 | 0.2540478338498888 | `results/experiments/wilcoxon_pairwise.csv` |
| Test | LMCD vs TS | 0.0059 | 0.005859375 | 0.25058221878758247 | `results/experiments/wilcoxon_pairwise.csv` |
| Test | MCD vs TS | 0.49 | 0.4921875 | 0.03278559176647261 | `results/experiments/wilcoxon_pairwise.csv` |

## Artifact Coverage and Caveats

- No `_samples.csv` files are currently present under `results/`, even though the
  current `experiment.py` writes them for new runs. Existing final CSVs are
  fold-level aggregate artifacts only.
- Sample-level analyses such as Spearman correlation, selective risk curves, or
  per-utterance error-bin analysis require rerunning the relevant experiments or
  using another source of per-sample outputs.
- TS and MCD test CSVs have identical mean WER values, but LMCD has a different
  mean WER. This matches the current implementation: MCD evaluates WER on a clean
  eval-mode transcript, while LMCD returns the stochastic medoid as its predicted
  transcript. This is a methodological caveat to resolve before strengthening
  direct uncertainty-method comparison claims.
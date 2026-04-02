import argparse
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so experiment plots don't block

import cma
import numpy as np
import pandas as pd
import torch

from experiment import run_experiment, ExperimentType


class UQObjective:
    """Callable objective for CMA-ES: maximize mean |R| across all folds.

    Decision variables (depend on experiment type):
      - TS:   [temperature, dummy]
      - MCD:  [dropout_rate, dummy]
      - SMCD: [temperature, dropout_rate]
    """

    def __init__(self, exp_type: ExperimentType, device: torch.device,
                 test_size: int = -1, output_dir: str = "optim"):
        self.exp_type = exp_type
        self.device = device
        self.test_size = test_size
        self.output_dir = output_dir
        self.eval_count = 0

    def __call__(self, x):
        self.eval_count += 1

        if self.exp_type == ExperimentType.TS:
            temperature = float(x[0])
            num_iterations = 10
            tag = f"ts-{temperature:.4f}t"
        elif self.exp_type == ExperimentType.MCD:
            dropout_rate = float(x[0])
            num_iterations = 10
            temperature = 0.75
            tag = f"mcd-{dropout_rate:.4f}d"
        else:  # SMCD
            temperature = float(x[0])
            dropout_rate = float(x[1])
            num_iterations = 10
            tag = f"smcd-{temperature:.4f}t-{dropout_rate:.4f}d"

        print(f"\n[eval {self.eval_count}] {tag}")

        run_experiment(
            tag, {}, {},
            exp_type=self.exp_type,
            device=self.device,
            temperature=temperature,
            num_iterations=num_iterations,
            dropout_rate=dropout_rate,
            test_size=self.test_size,
            end_fold=10,
            output_dir=self.output_dir
        )

        csv_path = os.path.join(self.output_dir, tag, tag + ".csv")
        df = pd.read_csv(csv_path)
        mean_r = df["R"].abs().mean()

        print(f"  -> mean |R| = {mean_r:.4f}")

        # CMA-ES minimises, so negate to maximise |R|
        return -mean_r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimise UQ hyperparameters with CMA-ES.")
    parser.add_argument("-e", "--exp_type", choices=["ts", "mcd", "smcd"], default="smcd",
                        help="Experiment type to optimise (default: smcd)")
    parser.add_argument("-p", "--pop_size", type=int, default=5,
                        help="CMA-ES population size (default: 5)")
    parser.add_argument("-n", "--n_gen", type=int, default=20,
                        help="Maximum number of generations (default: 20)")
    parser.add_argument("-t", "--test_size", type=int, default=-1,
                        help="Number of test samples per fold (-1 = all)")
    parser.add_argument("-o", "--output_dir", type=str, default="optim",
                        help="Output folder for results (default: optim)")
    args = parser.parse_args()

    exp_map = {"ts": ExperimentType.TS, "mcd": ExperimentType.MCD, "smcd": ExperimentType.SMCD}
    exp_type = exp_map[args.exp_type]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        dev = torch.device("xpu")
    print("Using", dev, "device")

    objective = UQObjective(exp_type, dev,
                            test_size=args.test_size,
                            output_dir=args.output_dir)

    if exp_type == ExperimentType.TS:
        x0 = [1.0, 0.05]
        bounds = [[0.01, 0.049], [2.0, 0.051]]
    elif exp_type == ExperimentType.MCD:
        x0 = [0.05, 0.05]
        bounds = [[0.001, 0.049], [0.5, 0.051]]
    else:  # SMCD
        x0 = [1.0, 0.05]
        bounds = [[0.01, 0.001], [2.0, 0.5]]

    opts = cma.CMAOptions()
    opts.set("popsize", args.pop_size)
    opts.set("maxiter", args.n_gen)
    opts.set("bounds", bounds)
    opts.set("seed", 42)

    es = cma.CMAEvolutionStrategy(x0, 0.5, opts)
    es.optimize(objective)
    res = es.result

    print("\n========== Optimisation complete ==========")
    if exp_type == ExperimentType.TS:
        print(f"Best temperature:      {res.xbest[0]:.4f}")
    elif exp_type == ExperimentType.MCD:
        print(f"Best dropout_rate:     {res.xbest[0]:.4f}")
    else:
        print(f"Best temperature:      {res.xbest[0]:.4f}")
        print(f"Best dropout_rate:     {res.xbest[1]:.4f}")
    print(f"Best mean |R|:         {-res.fbest:.4f}")
    print(f"Total evaluations:     {objective.eval_count}")


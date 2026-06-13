import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so experiment plots don't block

import cma
import pandas as pd
import torch

from experiment import run_experiment, ExperimentType, DatasetType


def setup_logging(output_dir: str) -> logging.Logger:
    """Configure a logger that writes the evolution trace to console and file."""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("optimize")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(os.path.join(output_dir, "evolution.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


def result_csv_path(exp_type: ExperimentType, dataset_type: DatasetType, k_folds: int,
                    temperature: float, dropout_rate: float, dropout_iterations: int,
                    output_dir: str) -> str:
    """Reproduce the store path that run_experiment writes its CSV to."""
    exp_name = exp_type.name.lower()
    if exp_type == ExperimentType.TS:
        exp_name += f"-temperature_{temperature}"
    else:
        exp_name += f"-iterations_{dropout_iterations}-dropout_{dropout_rate}"
    store_dir = os.path.join(output_dir, f"{exp_name}-dataset_{dataset_type.value}-folds_{k_folds}")
    return os.path.join(store_dir, exp_name + ".csv")


class UQObjective:
    """Callable objective for CMA-ES: maximize mean |R| across all folds.

    Decision variables (depend on experiment type):
      - TS:   [temperature, dummy]
      - MCD:  [dropout_rate, dummy]
      - LMCD: [dropout_rate, dummy]
    """

    def __init__(self, exp_type: ExperimentType, device: torch.device, logger: logging.Logger,
                 dataset_type: DatasetType = DatasetType.CALIBRATION,
                 k_folds: int = 10, dropout_iterations: int = 10,
                 sample_size: int = -1, output_dir: str = "optim"):
        self.exp_type = exp_type
        self.device = device
        self.logger = logger
        self.dataset_type = dataset_type
        self.k_folds = k_folds
        self.dropout_iterations = dropout_iterations
        self.sample_size = sample_size
        self.output_dir = output_dir
        self.eval_count = 0
        self.history = []

    def evaluate(self, x, generation: int, individual: int) -> float:
        self.eval_count += 1

        temperature = 0.75
        dropout_rate = 0.05
        if self.exp_type == ExperimentType.TS:
            temperature = float(x[0])
            param_name, param_value = "temperature", temperature
        else:  # MCD / LMCD
            dropout_rate = float(x[0])
            param_name, param_value = "dropout_rate", dropout_rate

        self.logger.info(
            f"[gen {generation:03d} | ind {individual:02d} | eval {self.eval_count:04d}] "
            f"start {self.exp_type.name} {param_name}={param_value:.4f}"
        )

        run_experiment(
            exp_type=self.exp_type,
            dataset_type=self.dataset_type,
            k_folds=self.k_folds,
            sample_size=self.sample_size,
            temperature=temperature,
            dropout_rate=dropout_rate,
            dropout_iterations=self.dropout_iterations,
            output_dir=self.output_dir,
            device=self.device,
        )

        csv_path = result_csv_path(
            self.exp_type, self.dataset_type, self.k_folds,
            temperature, dropout_rate, self.dropout_iterations, self.output_dir,
        )
        df = pd.read_csv(csv_path)
        mean_r = df["R"].abs().mean()

        self.logger.info(
            f"[gen {generation:03d} | ind {individual:02d} | eval {self.eval_count:04d}] "
            f"done  {param_name}={param_value:.4f} -> mean |R| = {mean_r:.4f}"
        )

        self.history.append({
            "generation": generation,
            "individual": individual,
            "eval": self.eval_count,
            param_name: param_value,
            "mean_abs_R": mean_r,
        })

        # CMA-ES minimises, so negate to maximise |R|
        return -mean_r


# python optimize.py -e lmcd -p 5 -n 3 -o optim/lmcd
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimise UQ hyperparameters with CMA-ES.")
    parser.add_argument("-e", "--exp_type", choices=["ts", "mcd", "lmcd"], default="lmcd",
                        help="Experiment type to optimise (default: lmcd)")
    parser.add_argument("-p", "--pop_size", type=int, default=5,
                        help="CMA-ES population size (default: 5)")
    parser.add_argument("-n", "--n_gen", type=int, default=20,
                        help="Maximum number of generations (default: 20)")
    parser.add_argument("-k", "--k_folds", type=int, default=10,
                        help="Number of folds per evaluation (default: 10)")
    parser.add_argument("-i", "--dropout_iterations", type=int, default=10,
                        help="Number of dropout iterations for MCD/LMCD (default: 10)")
    parser.add_argument("-s", "--sample_size", type=int, default=-1,
                        help="Number of samples per fold (-1 = all)")
    parser.add_argument("-o", "--output_dir", type=str, default="optim",
                        help="Output folder for results (default: optim)")
    args = parser.parse_args()

    exp_map = {"ts": ExperimentType.TS, "mcd": ExperimentType.MCD, "lmcd": ExperimentType.LMCD}
    exp_type = exp_map[args.exp_type]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        dev = torch.device("xpu")
    print("Using", dev, "device")

    logger = setup_logging(args.output_dir)
    logger.info(
        f"Starting optimisation: exp_type={args.exp_type} dataset={DatasetType.CALIBRATION.value} "
        f"pop_size={args.pop_size} max_gen={args.n_gen} k_folds={args.k_folds} device={dev}"
    )

    objective = UQObjective(
        exp_type, dev, logger,
        dataset_type=DatasetType.CALIBRATION,
        k_folds=args.k_folds,
        dropout_iterations=args.dropout_iterations,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
    )

    if exp_type == ExperimentType.TS:
        x0 = [1.0, 0.05]
        bounds = [[0.01, 0.049], [2.0, 0.051]]
    else:  # MCD / LMCD
        x0 = [0.05, 0.05]
        bounds = [[0.001, 0.049], [0.99, 0.051]]

    opts = cma.CMAOptions()
    opts.set("popsize", args.pop_size)
    opts.set("maxiter", args.n_gen)
    opts.set("bounds", bounds)
    opts.set("seed", 42)

    es = cma.CMAEvolutionStrategy(x0, 0.5, opts)

    generation = 0
    while not es.stop():
        generation += 1
        solutions = es.ask()
        fitnesses = []
        for individual, x in enumerate(solutions):
            fitnesses.append(objective.evaluate(x, generation, individual))
        es.tell(solutions, fitnesses)
        es.logger.add()
        es.disp()

    res = es.result

    # Persist the full evolution trace so any (generation, individual) can be related back.
    history_df = pd.DataFrame(objective.history)
    history_path = os.path.join(args.output_dir, "evolution_history.csv")
    history_df.to_csv(history_path, index=False)
    logger.info(f"Saved evolution history to {history_path}")

    logger.info("========== Optimisation complete ==========")
    if exp_type == ExperimentType.TS:
        logger.info(f"Best temperature:  {res.xbest[0]:.4f}")
    else:
        logger.info(f"Best dropout_rate: {res.xbest[0]:.4f}")
    logger.info(f"Best mean |R|:     {-res.fbest:.4f}")
    logger.info(f"Total evaluations: {objective.eval_count}")


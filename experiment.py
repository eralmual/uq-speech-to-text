import os
import torch

import numpy as np
import pandas as pd

from enum import Enum
from tqdm.auto import tqdm
from dataloader import Dataloader
from monte_carlo_dropout import MonteCarloDropout
from temperature_scaling import TemperatureScaling
from levenshtein_monte_carlo_dropout import LevenshteinMonteCarloDropout


class ExperimentType(Enum):
    FDE = 1
    MCD = 2
    TS = 3
    LMCD = 4


class DatasetType(Enum):
    TEST = "test"
    FINE_TUNE = "fine-tune"
    CALIBRATION = "calibration"


def build_uq_method(exp_type: ExperimentType, model_name: str, temperature: float,
                    dropout_rate: float, dropout_iterations: int, device: torch.device):
    """
    Factory that builds the UQ method matching the requested experiment type.
    """
    if exp_type == ExperimentType.TS:
        return TemperatureScaling(model_name, temperature=temperature, device=device)
    if exp_type == ExperimentType.MCD:
        return MonteCarloDropout(model_name, num_iterations=dropout_iterations,
                                 dropout_rate=dropout_rate, device=device)
    if exp_type == ExperimentType.LMCD:
        return LevenshteinMonteCarloDropout(model_name, num_iterations=dropout_iterations,
                                       dropout_rate=dropout_rate, device=device)
    raise NotImplementedError("Invalid experiment type: " + str(exp_type.name))

def run_experiment(exp_type: ExperimentType, dataset_type: DatasetType, k_folds: int = 10,  sample_size: int = -1,
                   temperature: float = 0.75, dropout_rate: float = 0.05, dropout_iterations: int = 10, 
                   output_dir: str = "results", device: torch.device="cpu") -> None:
    exp_name = f"{exp_type.name.lower()}"
    if(exp_type == ExperimentType.TS):
        exp_name += f"-temperature_{temperature}"
    else:
        exp_name += f"-iterations_{dropout_iterations}-dropout_{dropout_rate}"
    # Check if the store directory exists
    store_dir = os.path.join(output_dir, f"{exp_name}-dataset_{dataset_type.value}-folds_{k_folds}")
    os.makedirs(store_dir, exist_ok=True)

    mean_wers = []
    std_wers = []
    pearson_corrs = []
    ids = []
    sample_records = []

    # Load data
    dataset = Dataloader.load_uq_partitions(dataset_type.value, 1, k_folds + 1)

    # Run experiments for each fold, this is 1-indexed to match the model naming convention
    for id in tqdm(range(1, k_folds + 1), desc="Running experiments for each fold"):

        # Build the model name
        model_name = f"danrdz/whisper-finetuned-es-modelo_{id:02d}"

        # Get fold data
        dataset_fold = dataset[id - 1]
        # Optionally subsample the fold for quick tests
        if(sample_size > 0):
            dataset_fold = dataset_fold.select(range(min(sample_size, len(dataset_fold))))

        # Build the UQ method and run the evaluation
        uq_model = build_uq_method(exp_type, model_name, temperature,
                                   dropout_rate, dropout_iterations, device)
        wers, uq_scores, transcriptions, references = uq_model.evaluate(dataset_fold)

        # Calculate stats
        ids.append(id)
        wers = np.array(wers)
        uq_scores = np.array(uq_scores)
        if (np.std(uq_scores) == 0):
            print(f"Warning: UQ scores for model {model_name} have zero variance. Pearson correlation is not defined.")
            pearson_corr = 0.0
        elif(np.std(wers) == 0):
            print(f"Warning: WERs for model {model_name} have zero variance. Pearson correlation is not defined.")
            pearson_corr = 0.0
        else:
            pearson_corr = np.corrcoef(uq_scores, wers)[0, 1]

        mean_wer = np.mean(wers)
        std_wer = np.std(wers)

        # Store per-sample stats (UQ score, WER, transcription, reference) so
        # any plot or further test can be reproduced without re-running inference
        for sample_idx in range(len(wers)):
            sample_records.append({
                "Model ID": id,
                "Sample": sample_idx,
                "UQ Score": uq_scores[sample_idx],
                "WER": wers[sample_idx],
                "Transcription": transcriptions[sample_idx],
                "Reference": references[sample_idx],
            })

        # Store stats
        mean_wers.append(mean_wer)
        std_wers.append(std_wer)
        pearson_corrs.append(pearson_corr)     

    # Per-fold aggregate results
    res = pd.DataFrame({"Model ID": ids, "R": pearson_corrs, "Mean WER": mean_wers, "Std WER": std_wers})
    print(f"Mean $R$: {res.loc[:, 'R'].mean():.4f}, Mean WER: {res.loc[:, 'Mean WER'].mean():.4f}, Mean Std WER: {res.loc[:, 'Std WER'].mean():.4f}")
    print("=============== Results ===============\n", res)

    # Store per-fold aggregates (one row per model/fold)
    res.to_csv(os.path.join(store_dir, exp_name + ".csv"), index=False)

    # Store per-sample stats (one row per audio) for later plotting or analysis
    samples = pd.DataFrame(sample_records,
                           columns=["Model ID", "Sample", "UQ Score", "WER",
                                    "Transcription", "Reference"])
    samples.to_csv(os.path.join(store_dir, f"{exp_name}_samples.csv"), index=False)
    
import os
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from enum import Enum
from tqdm.auto import tqdm
from typing import Callable
from datasets import Audio
from dataloader import Dataloader
from whisper_wrapper import WhisperWrapper
from monte_carlo_dropout import MonteCarloDropout
from temperature_scaling import TemperatureScaling
from feature_density_estimator import FeatureDensityEstimator
from scaled_monte_carlo_dropout import ScaledMonteCarloDropout



class ExperimentType(Enum):
    FDE = 1
    MCD = 2
    TS = 3
    SMCD = 4

def run_feature_densities_experiment(finetune_audios: list, test_ds: list , test_audios: list, 
                                     train_size: int, top_k:int, model_wrapper: WhisperWrapper,
                                     aggregation_fn: Callable, reduction_fn: Callable, 
                                     gen_kwargs: dict, embedding_kwargs: dict):
        
        # Generate aux class for the method
        fde = FeatureDensityEstimator(model_wrapper)

        # Use the number of samples specified
        if(train_size > 0):
            finetune_audios = finetune_audios[0][:train_size]
        else:
            finetune_audios = finetune_audios[0]


        # Use FD training data to estimate feature densities
        histograms_and_buckets = fde.generate_feature_densities(finetune_audios, 
                                                                top_k, 
                                                                aggregation_fn,
                                                                reduction_fn,
                                                                gen_kwargs,
                                                                embedding_kwargs)
        
        # Compute the feature density scores
        uq_scores_test = fde.eval_likelihood(test_audios, histograms_and_buckets, 
                                             gen_kwargs, reduction_fn, aggregation_fn)
        # Compute transcription
        transcriptions_list, gt_list, _ = model_wrapper.transcribe_dataset(test_ds)
        
        # Fetch WERS
        wers = model_wrapper.compute_wers(transcriptions_list, gt_list)

        return wers, uq_scores_test


def run_experiment(exp_name: str, gen_kwargs: dict, embedding_kwargs: dict,
                   exp_type: ExperimentType, device: torch.device="cpu", top_k: int = 1,
                   aggregation_fn: Callable = lambda x: torch.cat(x, dim=1).squeeze(),
                   reduction_fn: Callable = lambda x: torch.flatten(x), 
                   temperature: float = 0.75, num_iterations: int = 10, dropout_rate: float = 0.05,
                   train_size: int = -1, test_size: int = -1) -> None:

    # Check if the store directory exists
    store_dir = f"results/{exp_name}"
    os.makedirs(store_dir, exist_ok=True)

    mean_wers = []
    std_wers = []
    pearson_corrs = []
    ids = []
    fig, axes = plt.subplots(2, 5, figsize=(16, 4), sharex=True, sharey=True)

    # Run experiments for each model
    for id in tqdm(range(1, 11), desc="Evaluating target models"):

        # Build the model objects
        model_name = f"danrdz/whisper-finetuned-es-modelo_{id:02d}"
                
        # Load data
        test_ds, test_audios = Dataloader.load_uq_partitions("test", id, id + 1)

        # Use the number of samples specified
        if(test_size > 0):
            test_audios = test_audios[0][:test_size]
            test_ds = test_ds[0].select(range(test_size))
        else:
            test_audios = test_audios[0]
            test_ds = test_ds[0]

        wers = []
        uq_scores = []
        if (exp_type == ExperimentType.FDE):
            model_wrapper = WhisperWrapper(model_name, device=device)
            # We need fine tune data for FDE
            _, finetune_audios = Dataloader.load_uq_partitions("fine-tune", id, id + 1)
            # Call the experiment
            wers, uq_scores = run_feature_densities_experiment( finetune_audios, test_ds, test_audios, 
                                                                train_size, top_k, model_wrapper,
                                                                aggregation_fn, reduction_fn, 
                                                                gen_kwargs, embedding_kwargs)
        elif (exp_type == ExperimentType.TS):
            # Build the TS model
            ts_model = TemperatureScaling(model_name, temperature=temperature, device=device)
            # Transcribe audios and get uncertainties
            transcriptions_list, gt_list, uq_scores = ts_model.transcribe_dataset(test_ds)
            # Fetch WERS
            wers = ts_model.compute_wers(transcriptions_list, gt_list)

        elif(exp_type == ExperimentType.MCD):
            # Build the MCD model
            mcd_model = MonteCarloDropout(model_name, num_iterations=num_iterations, dropout_rate=dropout_rate, device=device)
            # Transcribe audios and get uncertainties
            transcriptions_list, gt_list, uq_scores = mcd_model.transcribe_dataset(test_ds)
            # Fetch WERS
            wers = mcd_model.compute_wers(transcriptions_list, gt_list)
        elif(exp_type == ExperimentType.SMCD):
            # Build the SMCD model
            smcd_model = ScaledMonteCarloDropout(model_name, num_iterations=num_iterations, temperature=temperature, dropout_rate=dropout_rate, device=device)
            # Transcribe audios and get uncertainties
            transcriptions_list, gt_list, _ = smcd_model.transcribe_dataset(test_ds)
            total_dist, _, max_lenghts, _ = smcd_model.uncertainty_MCD(transcriptions_list)
            uq_scores = smcd_model.dividir_distancias(total_dist, max_lenghts)
            # Fetch WERS
            wers = smcd_model.compute_wers(transcriptions_list, gt_list)
            wers = [np.mean(wers[i:i+num_iterations]) for i in range(0, len(wers), num_iterations)]
        else:
            raise NotImplementedError("Invalid experiment type: " + str(exp_type.name))

        # Calculate stats
        ids.append(id)
        id = id - 1
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
        
        # Plot results
        i = id // 5
        j = id % 5
        axes[i][j].scatter(uq_scores, wers)
        axes[i][j].set_xlabel("UQ Score")
        axes[i][j].set_ylabel("WER")
        axes[i][j].set_title(f"Model {model_name.split('-')[-1][-2:]}")
        axes[i][j].grid()
        
        # Store stats
        mean_wers.append(mean_wer)
        std_wers.append(std_wer)
        pearson_corrs.append(pearson_corr)     

    # Print results
    res = pd.DataFrame({"Model ID": ids, "R": pearson_corrs, "Mean WER": mean_wers, "Std WER": std_wers})
    print(f"Mean R: {res.loc[:, 'R'].mean():.4f}, Mean WER: {res.loc[:, 'Mean WER'].mean():.4f}, Mean STD WER: {res.loc[:, 'Std WER'].mean():.4f}")
    fig.subplots_adjust(hspace=0.5)
    fig.show()
    print("=============== Results ===============\n", res)
    
    # Store results
    res.to_csv(os.path.join(store_dir,exp_name + ".csv"), index = False)
    fig.savefig(os.path.join(store_dir, f"{exp_name}_results.png"))
    
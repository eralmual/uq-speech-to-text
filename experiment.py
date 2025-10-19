import os
import copy
import math
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from typing import Callable
from dataloader import Dataloader
from whisper_wrapper import WhisperWrapper
from feature_density_estimator import FeatureDensityEstimator


def run_experiment(exp_name: str, gen_kwargs: dict, embedding_kwargs: dict, device: torch.device="cpu",
                   top_k: int = 1,
                   aggregation_fn: Callable = lambda x: torch.cat(x, dim=1).squeeze(),
                   reduction_fn: Callable = lambda x: torch.flatten(x),
                   ) -> None:

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
        model_wrapper = WhisperWrapper(model_name, device=device)
        fde = FeatureDensityEstimator(model_wrapper)

        # Load data
        _, finetune_audios = Dataloader.load_uq_partitions("fine-tune", id, id + 1)
        test_ds, test_audios = Dataloader.load_uq_partitions("test", id, id + 1)

        # Use FD training data to estimate feature densities
        histograms_and_buckets = fde.generate_feature_densities(
                                                                finetune_audios[0],
                                                                top_k, 
                                                                aggregation_fn,
                                                                reduction_fn,
                                                                gen_kwargs,
                                                                embedding_kwargs,
                                                                batch_size=100)        
        # Compute the feature density scores
        uq_scores_test = fde.eval_likelihood(test_audios[0], histograms_and_buckets, 
                                             gen_kwargs, reduction_fn, aggregation_fn)

        # Compute transcription
        transcriptions_list, gt_list = model_wrapper.transcribe_dataset(test_ds[0])

        # Fetch WERS
        wers = model_wrapper.compute_wers(transcriptions_list, gt_list)

        # Calculate stats
        ids.append(id)
        id = id - 1
        wers = np.array(wers)
        uq_scores_test = np.array(uq_scores_test)
        pearson_corr = np.corrcoef(uq_scores_test, wers)[0, 1]
        mean_wer = np.mean(wers)
        std_wer = np.std(wers)
        
        # Plot results
        i = id // 5
        j = id % 5
        axes[i][j].scatter(uq_scores_test, wers)
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
    

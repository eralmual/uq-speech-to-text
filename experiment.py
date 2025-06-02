import os
import copy
import math
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from model_evaluator import WhisperEvaluator
from model_wrapper import WhisperWrapper, IntermediateLayerGetter
from feature_density_estimator import FeatureDensityEstimator, clear_cache


def plot_uq(target_ds, uq_scores_test, evaluator, store_dir):

    # Compute transcription
    transcriptions_list, gt_list = evaluator.transcribe_dataset(target_ds)
    # Fetch WERS
    wers = evaluator.compute_wers(transcriptions_list, gt_list)

    # Plot results
    wers_np = np.array(wers)
    uq_scores_fd_np = np.array(uq_scores_test)

    plt.scatter(uq_scores_fd_np, wers_np)
    plt.xlabel("UQ Score")
    plt.ylabel("WER")
    plt.show()

    # Calculate stats
    pearson_corr = np.corrcoef(uq_scores_fd_np, wers_np)[0, 1]
    mean_wer = np.mean(wers_np)
    std_wer = np.std(wers_np)

    return mean_wer, std_wer, pearson_corr


def run_experiment( fde: FeatureDensityEstimator, model_evaluator: WhisperEvaluator, 
                    baseline_audios: list, test_ds: list, test_audios:list, 
                    exp_name: str, store_dir: str, **kwargs: dict) -> None:

    # Check if the store directory exists
    os.makedirs(store_dir, exist_ok=True)

    # Use FD training data to estimate feature densities
    histograms_and_buckets = fde.base_density_estimation(baseline_audios, **kwargs)
    
    # Collect data for each partition
    mean_wers = []
    std_wers = []
    pearson_corrs = []
    partitions = list(range(len(test_audios)))
    for i in tqdm(partitions, desc = "Processing partitions", leave=False):
        # Compute the feature density scores
        uq_scores_test = fde.eval_likelihood(test_audios[i], histograms_and_buckets, **kwargs)
        # Compute transcription
        transcriptions_list, gt_list = model_evaluator.transcribe_dataset(test_ds[i])
        # Fetch WERS
        wers = model_evaluator.compute_wers(transcriptions_list, gt_list)

        # Calculate stats
        wers = np.array(wers)
        uq_scores_test = np.array(uq_scores_test)
        pearson_corr = np.corrcoef(uq_scores_test, wers)[0, 1]
        mean_wer = np.mean(wers)
        std_wer = np.std(wers)
        print(f"Partition {i} - Mean WER: {mean_wer:.4f}, Std WER: {std_wer:.4f}, Pearson correlation coefficient: {pearson_corr:.4f}")

        # Plot results
        plt.scatter(uq_scores_test, wers)
        plt.xlabel("UQ Score")
        plt.ylabel("WER")
        plt.savefig(os.path.join(store_dir, f"{exp_name}_partition_{i}.png"))  
        plt.show()

        # Store stats
        mean_wers.append(mean_wer)
        std_wers.append(std_wer)
        pearson_corrs.append(pearson_corr)      

    # Print results
    res = pd.DataFrame({"Partition": partitions, "R": pearson_corrs, "Mean WER": mean_wers, "Std WER": std_wers})
    print("=============== Results ===============\n", res)
    print("=============== Mean results ===============\n", res.mean())
    res.to_csv(os.path.join(store_dir,exp_name + ".csv"), index = False)


def identify_influential_layers(model_wrapper: WhisperWrapper, target_layers: list, featured_audios: list) -> dict:
    # Hook the model on all intermediate results
    hooked_model = IntermediateLayerGetter(model_wrapper.model, target_layers, keep_output=False)
    model_wrapper.model.eval()
    dev = model_wrapper.device

    layer_outputs = {}
    for audio in tqdm(featured_audios, desc="Processing baseline audios"):
        
        # Process the audio
        input_features = model_wrapper.feature_extractor(audio, return_tensors="pt", sampling_rate = model_wrapper.sampling_rate).input_features.to(dev)
        decoder_input_ids = torch.tensor([[1, 1]], device=dev) * model_wrapper.model.config.decoder_start_token_id
        # Get the model output
        y = hooked_model(input_features, decoder_input_ids = decoder_input_ids)[0]
        y = IntermediateLayerGetter.calculate_block_influence(y)

        # Free the tensors so we dont run out of memory
        if(layer_outputs == {}):
            layer_outputs = copy.deepcopy(y)
            for k in list(y.keys()):
                del y[k]
        else:
            for k in list(y.keys()):
                layer_outputs[k] += y[k]
                del y[k]
        del y
        del input_features
        clear_cache(dev)

    # Average each channel and across all blocks, also remove empty entries
    layer_outputs = {k: ((v / len(featured_audios)).mean().item() if v != [] else -1) for k, v in layer_outputs.items() }
    # Remove Nans
    layer_outputs = {k: v for k, v in layer_outputs.items() if not math.isnan(v)}
    # Sort the results
    layer_outputs = dict(sorted(layer_outputs.items(), key=lambda item: item[1], reverse=True))

    return layer_outputs
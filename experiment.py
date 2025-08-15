import os
import copy
import math
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from dataloader import Dataloader
from model_evaluator import WhisperEvaluator
from model_wrapper import WhisperWrapper, IntermediateLayerGetter
from feature_density_estimator import FeatureDensityEstimator, clear_cache


def run_experiment(exp_name: str, store_dir: str, device: torch.device="cpu",  **kwargs: dict) -> None:

    # Check if the store directory exists
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
        model_evaluator = WhisperEvaluator(model = model_wrapper.model_cond_gen, processor = model_wrapper.processor)
        fde = FeatureDensityEstimator(model_wrapper)

        # Load data
        _, finetune_audios = Dataloader.load_uq_partitions("fine-tune", id, id + 1)
        test_ds, test_audios = Dataloader.load_uq_partitions("test", id, id + 1)

        # Use FD training data to estimate feature densities
        histograms_and_buckets = fde.base_density_estimation(finetune_audios[0], **kwargs)
        
        # Compute the feature density scores
        uq_scores_test = fde.eval_likelihood(test_audios[0], histograms_and_buckets, **kwargs)
        # Compute transcription
        transcriptions_list, gt_list = model_evaluator.transcribe_dataset(test_ds[0])
        
        # Fetch WERS
        wers = model_evaluator.compute_wers(transcriptions_list, gt_list)

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
    print(f"Mean R: {res.loc[:, "R"].mean():.4f}, Mean WER: {res.loc[:, "Mean WER"].mean():.4f}, Mean STD WER: {res.loc[:, "Std WER"].mean():.4f}")
    fig.subplots_adjust(hspace=0.5)
    fig.show()
    print("=============== Results ===============\n", res)
    
    # Store results
    res.to_csv(os.path.join(store_dir,exp_name + ".csv"), index = False)
    fig.savefig(os.path.join(store_dir, f"{exp_name}_results.png"))
    

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
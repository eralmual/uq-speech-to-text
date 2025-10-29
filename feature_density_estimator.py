import gc
import torch

import numpy as np

from tqdm.auto import tqdm
from typing import Callable
from whisper_wrapper import WhisperWrapper

def clear_cache(device: torch.device):
    """
    Clear the GPU cache.
    """
    if (device == "cuda"):
        torch.cuda.empty_cache()
    elif (device == "xpu"):
        torch.xpu.empty_cache()
    gc.collect()
    

class FeatureDensityEstimator:

    def __init__(self, model: WhisperWrapper):
        """
        Initialize the FeatureDensityEstimator with a model.
        """
        self.model = model
        self.model.model.eval()
        self.CosineSimilarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    @torch.no_grad()
    def _dataset_embedding_extraction(self, audios: list,
                                      aggregation_fn: Callable,
                                      gen_kwargs: dict,
                                      use_decoder: bool,
                                      use_encoder: bool ) -> dict:
        
        outputs = {"decoder_hidden_states": [], "encoder_hidden_states": []}

        # Extract embbedings for all audios
        for _, audio in tqdm(enumerate(audios), leave=False, desc="Extracting embeddings", total=len(audios)):
            output = self.model(audio, **gen_kwargs)
            if(use_decoder):
                outputs["decoder_hidden_states"].append(output.decoder_hidden_states)
            if(use_encoder):
                outputs["encoder_hidden_states"].append(output.encoder_hidden_states)

            clear_cache(self.model.model.device)

        # Check if the outputs have hidden states
        embeddings = {}
        if(use_decoder):
            # Generate embeddings per layer by concatenating the output for each token at the specific layer
            decoder_hidden_states = {}
            for layer in range(len(outputs["decoder_hidden_states"][0][0])):
                decoder_hidden_states[layer] = []
                for audio in range(len(outputs["decoder_hidden_states"])):
                    for token in range(len(outputs["decoder_hidden_states"][audio])):
                        decoder_hidden_states[layer].append(outputs["decoder_hidden_states"][audio][token][layer].detach().cpu())
                decoder_hidden_states[layer] = aggregation_fn(decoder_hidden_states[layer])
            # Store the embeddings
            embeddings["decoder_hidden_states"] = decoder_hidden_states


        if(use_encoder):
            encoder_hidden_states = {}
            for layer in range(len(outputs["encoder_hidden_states"][0])):
                encoder_hidden_states[layer] = []
                for audio in range(len(outputs["encoder_hidden_states"])):
                    encoder_hidden_states[layer].append(outputs["encoder_hidden_states"][audio][layer].detach().cpu())
                encoder_hidden_states[layer] = aggregation_fn(encoder_hidden_states[layer])
            # Store the embeddings
            embeddings["encoder_hidden_states"] = encoder_hidden_states

        return embeddings
    
    @torch.no_grad()
    def _block_influence_layer_selector(self, embeddings: dict, top_k: int):

         
        bi = {}
        selected_embeddings = {}
        # Calculate the BI metric for each hstate
        for hstate_name, layers in embeddings.items():
            bi[hstate_name] = {}
            # For each layer of each hstate
            for l in range(len(layers) - 1):
                bi[hstate_name][l] = torch.mean(1 - self.CosineSimilarity(layers[l], layers[l + 1]))

            # Once we have the BI, sort the dictionary
            bi[hstate_name] = dict(sorted(bi[hstate_name].items(), key=lambda item: item[1], reverse=True))
            # Select the top_k layers
            tk = top_k if (top_k > 0) else len(bi[hstate_name])
            selected_embeddings[hstate_name] = {}
            for k, _ in bi[hstate_name].items():
                if(tk == 0):
                    break
                selected_embeddings[hstate_name][k] = layers[k]
                tk -= 1

        return selected_embeddings 
    
    def _generate_histogram(self, embeddings: dict, reduction_fn: Callable, num_bins: int = 20):

        histograms_and_buckets = {}
        # Values for edge cases
        torch_zero = torch.tensor([0])

        # Calculate the histograms for each hstate module
        for hstate, layers in embeddings.items():
            histograms_and_buckets[hstate] = {}
            # Generate an histogram for each layer
            for layer, emb in layers.items():
                # Get a normalized histogram that fills the entire numeric range
                (hist, buckets) = torch.histogram(reduction_fn(emb), bins=num_bins, density=True)
                # Insert extra bins for edge cases 
                hist = torch.cat([torch_zero, hist, torch_zero], dim=0)
                # Store data, add epsilon to the histogram
                histograms_and_buckets[hstate][layer] = (hist + 1e-6, buckets)

        return histograms_and_buckets


    def generate_feature_densities( self, audios: list, 
                                    top_k: int,
                                    aggregation_fn: Callable,
                                    reduction_fn: Callable,
                                    gen_kwargs: dict, 
                                    embedding_kwargs: dict):
        """
        Estimate the base density of the data.
        """
        # Extract embeddings for the dataset
        embeddings = self._dataset_embedding_extraction(audios, aggregation_fn, gen_kwargs, **embedding_kwargs)
        # Free GPU memory
        clear_cache(self.model.model.device)

        # Use BI metric to select the top K layers for each hstate module
        embeddings = self._block_influence_layer_selector(embeddings, top_k)
        clear_cache(self.model.model.device)

        # Calculate histograms and buckets for the best (or selected) layers
        histograms_and_buckets = self._generate_histogram(embeddings, reduction_fn)
        # Free GPU memory
        clear_cache(self.model.model.device)
        
        return histograms_and_buckets


    @torch.no_grad()
    def eval_likelihood(self, audios: list, histograms_and_buckets: dict, gen_kwargs: dict,
                        reduction_fn: Callable, aggregation_fn: Callable):
        """
        Evaluate the ood score of a list of audios
        """
        ood_scores = []
        for audio in tqdm(audios, leave=False, desc="Evaluating likelihood"):
            # Base values
            sum_log_likelihoods = 0
            # Extract embeddings for the target audio
            outputs = self.model(audio, **gen_kwargs)
            # For each layer
            for hstate, layers in histograms_and_buckets.items():
                for layer, hist_and_buck in layers.items():
                    token_emb = []
                    histogram = hist_and_buck[0]
                    buckets = hist_and_buck[1]
                    # Encoder module does not generate tokens so indexing is different
                    if(hstate == "encoder_hidden_states"):
                        token_emb = reduction_fn(outputs[hstate][layer].cpu())
                    else:
                        # For decoder modules, aggregate all the tokens 
                        for token in range(len(outputs[hstate])):
                            token_emb.append(outputs[hstate][token][layer].cpu())
                        # Aggregate and reduce so that the embeddings can be processed
                        token_emb = reduction_fn(aggregation_fn(token_emb))
                        
                    # Find the closest bucket to the current feature value
                    bucket_idxs = torch.bucketize(token_emb, buckets)
                    # Fetch the density (estimated with histograms) of the current dimension
                    likelihood = torch.gather(histogram, 0, bucket_idxs)
                    # Accumulate the log likelihoods
                    sum_log_likelihoods += torch.log(likelihood).sum()

            # Add audio OOD score to the scores list
            ood_scores += [-sum_log_likelihoods.item()]

        return ood_scores

import gc
import torch
import math

import numpy as np

from tqdm.auto import tqdm
from typing import Callable
from whisper_wrapper import WhisperWrapper

def clear_cache(device: str):
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
            outputs["decoder_hidden_states"].append(output.decoder_hidden_states)
            outputs["encoder_hidden_states"].append(output.encoder_hidden_states)

        # Check if the outputs have hidden states
        embeddings = {}
        if((outputs["decoder_hidden_states"][0] is not None) and use_decoder):
            # Generate embeddings per layer by concatenating the output for each token at the specific layer
            decoder_hidden_states = {}
            for layer in range(len(outputs["decoder_hidden_states"][0][0])):
                decoder_hidden_states[layer] = []
                for audio in range(len(outputs["decoder_hidden_states"])):
                    for token in range(len(outputs["decoder_hidden_states"][audio])):
                        decoder_hidden_states[layer].append(outputs["decoder_hidden_states"][audio][token][layer].cpu())
                decoder_hidden_states[layer] = aggregation_fn(decoder_hidden_states[layer])
            # Store the embeddings
            embeddings["decoder_hidden_states"] = decoder_hidden_states


        if((outputs["encoder_hidden_states"] is not None) and use_encoder):
            encoder_hidden_states = {}
            for layer in range(len(outputs["encoder_hidden_states"][0])):
                encoder_hidden_states[layer] = []
                for audio in range(len(outputs["encoder_hidden_states"])):
                    encoder_hidden_states[layer].append(outputs["encoder_hidden_states"][audio][layer].cpu())
                encoder_hidden_states[layer] = aggregation_fn(encoder_hidden_states[layer])
            # Store the embeddings
            embeddings["encoder_hidden_states"] = encoder_hidden_states

        return embeddings
    
    @torch.no_grad()
    def _block_influence_layer_selector(self, embeddings: dict, top_k: int = 1):

         
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
            tk = top_k
            selected_embeddings[hstate_name] = {}
            for k, _ in bi[hstate_name].items():
                if(tk == 0):
                    break
                selected_embeddings[hstate_name][k] = layers[k]
                tk -= 1


        return selected_embeddings 

    def _generate_histogram(self, embeddings: dict, reduction_fn: Callable, num_bins: int = 20):

        histograms_and_buckets = {}

        # Calculate the histograms for each hstate module
        for hstate, layers in embeddings.items():
            histograms_and_buckets[hstate] = {}
            # Generate an histogram for each layer
            for layer, emb in layers.items():
                # Get a normalized histogram that fills the entire numeric range
                (hist, buckets) = np.histogram(reduction_fn(emb), bins=num_bins, density=True)
                # Instead of bin edges, get bin mean
                buckets = np.convolve(buckets, [0.5, 0.5], mode='valid')
                # Store data
                histograms_and_buckets[hstate][layer] = (torch.from_numpy(hist), torch.from_numpy(buckets))

        return histograms_and_buckets


    def generate_feature_densities(self, audios: np.array, 
                                    top_k: int,
                                    aggregation_fn: Callable,
                                    reduction_fn: Callable,
                                    gen_kwargs: dict, 
                                    embedding_kwargs: dict,
                                    batch_size: int = 100):
        """
        Estimate the base density of the data using multiple batches.
        Returns a LIST of histograms_and_buckets (one per batch).
        """
        all_histograms = []
        
        # Batch processing
        num_batches = math.ceil(len(audios) / batch_size)
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            # Get the current batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(audios))
            batch_audios = audios[start_idx:end_idx]
            
            # Extract embeddings for the batch
            embeddings = self._dataset_embedding_extraction(batch_audios, aggregation_fn, gen_kwargs, **embedding_kwargs)
            clear_cache(self.model.model.device)

            # Use BI metric to select the top K layers
            embeddings = self._block_influence_layer_selector(embeddings, top_k)
            clear_cache(self.model.model.device)

            # Calculate histograms for this batch
            histograms_and_buckets = self._generate_histogram(embeddings, reduction_fn)
            clear_cache(self.model.model.device)
            
            # Save the histogram for this batch
            all_histograms.append(histograms_and_buckets)
        
        return all_histograms
    

    def _find_closest_bucket(self, vals_feature_all_obs, buckets):
        """
        Finds the closest bucket position, according to a set of values (from features) received
        :param vals_feature_all_obs: values of features received, to map to the buckets
        :param buckets: buckets of the previously calculated histogram
        :return: returns the list of bucket numbers closest to the buckets received
        """

        # create repeated map to do a matrix substraction, unsqueezeing and transposing the feature values for all the observations
        vals_feature_all_obs = vals_feature_all_obs.unsqueeze(dim=0).transpose(0, 1)
        # rep mat
        repeated_vals_dim_obs = vals_feature_all_obs.repeat(1, buckets.shape[0])
        repeated_vals_dim_obs = repeated_vals_dim_obs.view(-1, buckets.shape[0])
        # do substraction
        substracted_all_obs = torch.abs(repeated_vals_dim_obs - buckets)
        # find the closest bin per observation (one observation per row)
        min_buckets_all_obs = torch.argmin(substracted_all_obs, 1)

        return min_buckets_all_obs


    @torch.no_grad()
    def eval_likelihood(self, audios: list, all_histograms: list, gen_kwargs: dict,
                        reduction_fn: Callable, aggregation_fn: Callable):
        """
        Evaluate the ood score of a list of audios against MULTIPLE histograms.
        all_histograms: lista de histogramas (uno por batch)
        """
        ood_scores = []
        
        for audio in tqdm(audios, leave=False, desc="Evaluating likelihood"):
            # Extract embeddings for the target audio (una sola vez)
            outputs = self.model(audio, **gen_kwargs)
            
            # List for storing likelihoods for each histogram
            audio_ood_scores_per_histogram = []
            
            # Evaluate against each histogram
            for histograms_and_buckets in all_histograms:
                sum_log_likelihoods = 0
                epsilon = 1e-6
                
                # For each layer in this histogram
                for hstate, layers in histograms_and_buckets.items():
                    for layer, hist_and_buck in layers.items():
                        token_emb = []
                        
                        # Encoder module does not generate tokens
                        if(hstate == "encoder_hidden_states"):
                            token_emb = reduction_fn(outputs[hstate][layer].cpu())
                        else:
                            # For decoder modules, aggregate all tokens 
                            for token in range(len(outputs[hstate])):
                                token_emb.append(outputs[hstate][token][layer].cpu())
                            token_emb = reduction_fn(aggregation_fn(token_emb))
                        
                        # Find closest bucket
                        closest_bucket_position = self._find_closest_bucket(token_emb, hist_and_buck[1])
                        # Fetch density
                        histogram_dim_d = hist_and_buck[0]
                        likelihood_dim_d = histogram_dim_d[closest_bucket_position] + epsilon
                        # Accumulate log likelihoods
                        sum_log_likelihoods += torch.log(likelihood_dim_d).sum()
                
                # Save the OOD score for this histogram
                audio_ood_scores_per_histogram.append(-sum_log_likelihoods.item())
            
            # Average the OOD scores of all histograms
            avg_ood_score = np.mean(audio_ood_scores_per_histogram)
            ood_scores.append(avg_ood_score)
        
        return ood_scores

import gc
import torch

import numpy as np

from tqdm.auto import tqdm
from typing import Callable
from model_wrapper import WhisperWrapper, IntermediateLayerGetter

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

    def __init__(self, model_wrapper: WhisperWrapper):
        """
        Initialize the FeatureDensityEstimator with a model.
        """
        self.model_wrapper = model_wrapper

    def _extract_embeddings( self, audio_tensor: torch.Tensor, target_layers: list,
                            embedding_filter: Callable = lambda x: x.cpu().detach()) -> torch.Tensor:
        """
        Fetches the embeddings for a given audio tensor.
        Sequence of hidden-states at the output of the last layer of the decoder of the model.
        @param audio_tensor: audio tensor to process
        @param target_layers: list with the names of the layers we want to extract
        @param embedding_filter: function to filter the embeddings, selects dimensions, apply transforms, etc
        @return: embeddings, usually a 1-D Tensor with the embeddings for the audio
        """
        # Encoded inputs are computed
        device = self.model_wrapper.device
        # Encode the inputs
        input_features = self.model_wrapper.feature_extractor(audio_tensor, return_tensors="pt", sampling_rate = self.model_wrapper.sampling_rate).input_features.to(device)
        decoder_input_ids = torch.tensor([[1, 1]], device=device) * self.model_wrapper.model.config.decoder_start_token_id
        # The ILG expects a dictionary with the layer name and a friendly name, we just use the original one in both cases
        target_layers = dict(zip(target_layers, target_layers))
        # Extract the intermediate layers
        intermediate_layer_getter = IntermediateLayerGetter(self.model_wrapper.model, target_layers)
        # Extract the embeddings
        embeddings_per_layer = intermediate_layer_getter(input_features, decoder_input_ids = decoder_input_ids)[0]
        # Filter the embeddings and generate a per-layer dictionary
        return {k: embedding_filter(v) for k, v in embeddings_per_layer.items()}
    
    def _dataset_embedding_extraction(self, audios: list, target_layers: list, **kwargs) -> dict:

        # Container for the embeddings
        embeddings_per_layer = dict(zip(target_layers, len(target_layers)*[[]]))

        # Extract embbedings for all audios and sort it per layer
        for i, audio in tqdm(enumerate(audios), leave=False, desc="Extracting embeddings", total=len(audios)):
            # Extract the 1-D embeddings and add to the matrix
            embedding = self._extract_embeddings(audio, target_layers=target_layers, **kwargs)
            for k, v in embedding.items():
                embeddings_per_layer[k].append(torch.squeeze(v).cpu().detach())

        # Stack the embbedings in N-D fashion where the first dimension is the number of audios
        for k, v in embeddings_per_layer.items():
            embeddings_per_layer[k] = torch.stack(v, dim=0)
            
        return embeddings_per_layer
    
    def _calculate_histogram_per_layer(self, embeddings: dict, num_bins: int = 20):
        """
        Calculates the feature densities histogram using a sample of feature arrays
        param embeddings: Dictionary indexed by layer names, containing, usually, a Tensor of shape (num_audios, num_features)
        param num_bins: number of bins to use
        return: histograms_all_features, buckets_all_features
        """
        histograms_and_buckets = {}

        # Calculate the histograms for each layer
        for k, v in embeddings.items():
            # Initialize the histograms and buckets for each channel
            num_features = v.shape[1]
            histograms_and_buckets[k] = (torch.empty((num_features, num_bins), dtype=torch.float32, device="cpu"), 
                                         torch.empty((num_features, num_bins), dtype=torch.float32, device="cpu"))

            for d in range(num_features):

                (hist, buckets) = np.histogram(v[:, d], bins=num_bins,
                                            range=None, density=False)
                
                # Normalize the histogram
                hist = hist / hist.sum()
                # Instead of bin edges, get bin mean
                buckets = np.convolve(buckets, [0.5, 0.5], mode='valid')
                # Store data
                histograms_and_buckets[k][0][d, :] = torch.from_numpy(hist)
                histograms_and_buckets[k][1][d, :] = torch.from_numpy(buckets)

        return histograms_and_buckets


    def base_density_estimation(self, audios: np.array, **kwargs):
        """
        Estimate the base density of the data.
        """
        embeddings_per_layer = self._dataset_embedding_extraction(audios, **kwargs)
        clear_cache(self.model_wrapper.device)
        histograms_and_buckets = self._calculate_histogram_per_layer(embeddings_per_layer)
        clear_cache(self.model_wrapper.device)
        return histograms_and_buckets
        



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


    def eval_likelihood(self, audios: list, histograms_and_buckets, **kwargs):
        """
        Evaluate the ood score of a list of audios
        """
        ood_scores = []
        for audio in tqdm(audios, leave=False, desc="Evaluating likelihood"):
            # Base values
            sum_log_likelihoods = 0
            epsilon = 1e-6
            # Extract embeddings for the target audio
            embeddings = self._extract_embeddings(audio, **kwargs)
            # For each layer
            for k, v in embeddings.items():
                # Get the layer's histograms and buckets
                base_histograms = histograms_and_buckets[k][0]
                base_buckets = histograms_and_buckets[k][1]
                # Go through all the dimensions
                for d in range(v.shape[0]):
                    # Find the closest bucket to the current feature value
                    closest_bucket_position = self._find_closest_bucket(v[d].unsqueeze(0), base_buckets[d, :])
                    # Fetch the density (estimated with histograms) of the current dimension
                    histogram_dim_d = base_histograms[d, :]
                    likelihood_dim_d = histogram_dim_d[closest_bucket_position] + epsilon
                    # Accumulate the log likelihoods
                    sum_log_likelihoods += torch.log(likelihood_dim_d)

                ood_scores += [-sum_log_likelihoods.item()]

        return ood_scores

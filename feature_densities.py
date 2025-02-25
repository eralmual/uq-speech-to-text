import gc
import torch

import numpy as np

from tqdm.auto import tqdm
from model_wrapper import WhisperWrapper

def extract_embeddings(audio_tensor: torch.Tensor, model_wrap: WhisperWrapper):
    """
    Fetches the embeddings for a given audio tensor.
    Sequence of hidden-states at the output of the last layer of the decoder of the model.
    @param audio_tensor: audio tensor, could be also an audio array
    @param model: model to use, usually WhisperModel
    @param input_encoder: input_encoder used to encode the audio tensor, usually AutoFeatureExtractor
    @param sampling_rate: sampling rate
    """
    # Encoded inputs are computed
    device = model_wrap.device
    # Encode the inputs
    input_features = model_wrap.feature_extractor(
        audio_tensor, return_tensors="pt", sampling_rate=model_wrap.sampling_rate
    ).input_features.to(device)

    # Pass inputs through the encoder to get its output
    encoder_outputs = model_wrap.model.encoder(input_features)

    # Prepare the decoder input (e.g., start token for autoregressive generation)
    decoder_input_ids = torch.tensor([[model_wrap.model.config.decoder_start_token_id]], device=device)

    # Pass both encoder outputs and decoder inputs into the decoder
    decoder_outputs = model_wrap.model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        return_dict=True,
    )

    # Access the cross-attention output (key hidden states in the decoder output)
    # This is the last hidden state of the decoder, which incorporates cross-attention.
    cross_attention_hidden_state = decoder_outputs.last_hidden_state

    # Optionally, select a specific token embedding or return the entire sequence
    selected_embedding = cross_attention_hidden_state[0, 0, :].cpu().detach()
    
    return selected_embedding

def extract_embeddings_all_audios(model_wrap: WhisperWrapper, audios: np.array) -> np.array:
    """
    Fetches the embeddings for a list of audio arrays
    Sequence of hidden-states at the output of the last layer of the decoder of the model.
    @param model_path: path to the model to use
    @param list_audios_array: list of audio arrays
    @param sampling_rate: sampling rate
    @param device: device to use
    return embedding_matrix, returns the set of embeddings to use
    """

    # Extract embbeding dimensions
    embbeding_0 = extract_embeddings(audios[0], model_wrap)

    # Init the embeddings matrix
    embeddings_matrix = np.zeros((len(audios), embbeding_0.shape[0]))
    embeddings_matrix[0, :] = embbeding_0

    # Extract embbedings for all audios
    for i, audio_array in tqdm(enumerate(audios[1:]), leave=False, desc="Extracting embeddings", total=len(audios[1:])):
        # Extract the embbedings and add to the matrix
        embedding = extract_embeddings(audio_array, model_wrap)
        embeddings_matrix[i, :] = embedding


    gc.collect()
    torch.cuda.empty_cache()
    return embeddings_matrix

def calculate_feature_densities_histogram(embeddings_matrix, num_bins = 20):
      """
      Calculates the feature densities histogram using a sample of feature arrays
      param embeddings_matrix: set of embeddings to use
      param num_bins: number of bins to use
      return: histograms_all_features, buckets_all_features
      """
      d_dimensions = embeddings_matrix.shape[1]
      histograms_all_features = np.zeros((d_dimensions, num_bins))
      buckets_all_features = np.zeros((d_dimensions, num_bins))
      for d in range(0, d_dimensions):
        # calculate the histograms
        data_dimension = embeddings_matrix[:, d]
    
        (hist1, bucks1) = np.histogram(data_dimension, bins=num_bins, range=None,
                                        density=False)
        # manual normalization, np doesnt work
        hist1 = hist1 / hist1.sum()
        # instead of bin edges, get bin mean
        bucks1 = np.convolve(bucks1, [0.5, 0.5], mode='valid')
        # normalize the histograms and move it to the gpu
        hist1 = torch.tensor(np.array(hist1))
        bucks1 = torch.tensor(bucks1)
        histograms_all_features[d, :] = hist1
        buckets_all_features[d, :] = bucks1
      return torch.tensor(histograms_all_features), torch.tensor(buckets_all_features)
    

def build_feature_densities_estimation(model_wrap, list_audios_array):
        """
        Build the feature densities estimation (training)
        """
        embeddings_matrix = extract_embeddings_all_audios(model_wrap, list_audios_array)
        histograms_all_features, buckets_all_features = calculate_feature_densities_histogram(embeddings_matrix)
        return histograms_all_features, buckets_all_features


def eval_likelihood(target_audio, model_wrap, base_histograms, base_buckets):

    # Extract embeddings for the target audio
    vector_embedding = extract_embeddings(target_audio, model_wrap)
    #print("vector_embedding demo ", vector_embedding.shape)
    #go through all the dimensions
    sum_log_likelihoods = 0
    epsilon = 1e-6
    for d in range(vector_embedding.shape[0]):
        
        # Find the closest bucket to the current feature value
        closest_bucket_position = find_closest_bucket_all_obs(vector_embedding[d].unsqueeze(0), base_buckets[d, :])
        #fetch the density (estimated with histograms) of the current dimension
        histogram_dim_d = base_histograms[d, :]
        #print("closest_bucket ", closest_bucket)
        likelihood_dim_d = histogram_dim_d[closest_bucket_position] + epsilon
        #print("likelihood_dim_d ", likelihood_dim_d)
        #accumulate the log likelihoods
        sum_log_likelihoods += torch.log(likelihood_dim_d)
    ood_score = -sum_log_likelihoods
    #print("ood_score ", ood_score)
    return ood_score
    
def find_closest_bucket_all_obs(vals_feature_all_obs, buckets):
    """
    Finds the closest bucket position, according to a set of values (from features) received
    :param vals_feature_all_obs: values of features received, to map to the buckets
    :param buckets: buckets of the previously calculated histogram
    :return: returns the list of bucket numbers closest to the buckets received
    """

    #print("vals_feature_all_obs ", vals_feature_all_obs)
    # create repeated map to do a matrix substraction, unsqueezeing and transposing the feature values for all the observations
    vals_feature_all_obs = vals_feature_all_obs.unsqueeze(dim=0).transpose(0, 1)
    #print("vals_feature_all_obs \n ", vals_feature_all_obs)
    # rep mat
    repeated_vals_dim_obs = vals_feature_all_obs.repeat(1, buckets.shape[0])
    repeated_vals_dim_obs = repeated_vals_dim_obs.view(-1, buckets.shape[0])
    #print("repeated_vals_dim_obs \n", repeated_vals_dim_obs)
    # do substraction
    substracted_all_obs = torch.abs(repeated_vals_dim_obs - buckets)
    #print("substracted_all_obs \n", substracted_all_obs)
    # find the closest bin per observation (one observation per row)
    min_buckets_all_obs = torch.argmin(substracted_all_obs, 1)
    #print("min_buckets_all_obs \n", min_buckets_all_obs)
    return min_buckets_all_obs
    
def eval_likelihood_all_audios(model_wrap, query_audios_list, histograms_all_features, buckets_all_features):
    """
    Evaluate the ood score of a list of audios
    """
    list_ood_scores = []
    for query_audio_array in tqdm(query_audios_list, leave=False, desc="Evaluating likelihood"):
        ood_score = eval_likelihood(query_audio_array, model_wrap, histograms_all_features, buckets_all_features)
        list_ood_scores += [ood_score.item()]
    return list_ood_scores
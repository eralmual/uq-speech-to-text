
import torch
import Levenshtein

import numpy as np

from uq_method import UQMethod


class LevenshteinMonteCarloDropout(UQMethod):
    
    def __init__(self, model_name: str, num_iterations: int, dropout_rate: float = 0.1, sampling_rate: int = 16000, device=torch.device("cpu")):
        super().__init__(model_name, dropout=dropout_rate, sampling_rate=sampling_rate, device=device)
        self.num_iterations = num_iterations

    def transcribe_audio(self, audio_array):
        # Enable dropout at inference time
        self.model.train()

        transcriptions = []
        uncertainties = []
        for _ in range(self.num_iterations):
            # Generate input features => output.sequences: token ids, output.scores: tuple of logits per step
            output = self(audio_array, return_dict_in_generate=True, 
                          output_hidden_states=False, output_attentions=False, output_scores=True)
            
            # Stack logits and compute softmax
            logits = torch.stack(output.scores, dim=1)
            probabilities = torch.softmax(logits, dim=-1)
            # Get the max probability per token
            max_probs = torch.max(probabilities, dim=2).values
            uncertainties.append(1 - max_probs.mean().item())

            # Fetch transcription
            decoded = self.processor.tokenizer.batch_decode(output.sequences, skip_special_tokens=True, normalize=True)
            transcriptions.append(self.tildar_oracion(decoded[0]))

        # lets calculate the unceretainty for the prediction 
        medoid = self.find_medoid_sequence(transcriptions)
        normalized_distances = self.calculate_levenshtein_distances(medoid, transcriptions)

        # Chose the medoid as the prediction and the average normalized distance as the uncertainty
        return medoid, sum(normalized_distances) / len(normalized_distances)
    

    def find_medoid_sequence(self, transcriptions: list[str]) -> str:
        medoid = ""
        min_total_distance = float('inf')

        # Iterate across all transcriptions looking for the one
        # that is closest to all the others 
        for t1 in transcriptions:
            total_distance = 0
            for t2 in transcriptions:
                total_distance += Levenshtein.distance(t1, t2)

            if total_distance < min_total_distance:
                min_total_distance = total_distance
                medoid = t1

        return medoid
    
    def calculate_levenshtein_distances(self, medoid: str, transcriptions: list[str]) -> list[int]:
        distances = []

        for t in transcriptions:
            # Get Levenshtein distance and the longest sequence lenght
            distance = Levenshtein.distance(medoid, t)
            max_len = max(len(medoid), len(t))

            # Normalize distance when possible
            if max_len > 0:
                distance = distance / max_len 
            else:
                distance = 0.0

            distances.append(distance)

        return distances
  
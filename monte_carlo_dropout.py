
import torch

from uq_method import UQMethod


class MonteCarloDropout(UQMethod):
    
    def __init__(self, model_name: str, num_iterations: int, dropout_rate: float = 0.1, sampling_rate: int = 16000, device=torch.device("cpu")):
        super().__init__(model_name, dropout=dropout_rate, sampling_rate=sampling_rate, device=device)
        self.num_iterations = num_iterations

    def transcribe_audio(self, audio_array):
        # Deterministic prediction: clean eval-mode pass (no dropout) for a fair
        # WER comparison against the other methods
        self.model.eval()
        clean_output = self(audio_array, return_dict_in_generate=True,
                            output_hidden_states=False, output_attentions=False, output_scores=False)
        transcriptions = self.processor.tokenizer.batch_decode(clean_output.sequences, skip_special_tokens=False, normalize=True)
        transcription = self.tildar_oracion(transcriptions[0])

        # Enable dropout at inference time to estimate uncertainty
        self.model.train()

        all_probabilities = []
        for _ in range(self.num_iterations):
            # Generate input features => output.sequences: token ids, output.scores: tuple of logits per step
            output = self(audio_array, return_dict_in_generate=True, 
                          output_hidden_states=False, output_attentions=False, output_scores=True)
            
            # Stack logits and compute softmax
            logits = torch.stack(output.scores, dim=1)
            probabilities = torch.softmax(logits, dim=-1)
            # Get the max probability per token
            max_probs = torch.max(probabilities, dim=2).values
            all_probabilities.append(max_probs)

        # Calculate uncertainty as variance across MC iterations
        # Pad to same length since dropout causes variable-length outputs
        max_len = max(p.shape[-1] for p in all_probabilities)
        padded = [torch.nn.functional.pad(p, (0, max_len - p.shape[-1]), value=0.0) for p in all_probabilities]
        stacked = torch.stack(padded, dim=0)  # (num_iterations, batch, max_len)
        uncertainty = stacked.var(dim=0).mean().item()

        self.model.eval()
        return transcription, uncertainty
    
    
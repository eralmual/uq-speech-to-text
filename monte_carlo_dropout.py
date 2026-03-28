
import torch

from datasets import Audio
from tqdm.auto import tqdm
from whisper_wrapper import WhisperWrapper


class MonteCarloDropout(WhisperWrapper):
    
    def __init__(self, model_name: str, num_iterations: int, dropout_rate: float = 0.1, sampling_rate: int = 16000, device=torch.device("cpu")):
        super().__init__(model_name, dropout=dropout_rate, sampling_rate=sampling_rate, device=device)
        self.num_iterations = num_iterations

    def transcribe_audio(self, audio_array):
        # Enable dropout at inference time
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

        # Fetch transcription from last forward pass
        transcriptions = self.processor.tokenizer.batch_decode(output.sequences, skip_special_tokens=False, normalize=True)
        transcription = self.tildar_oracion(transcriptions[0])

        # Calculate uncertainty as variance across MC iterations
        # Pad to same length since dropout causes variable-length outputs
        max_len = max(p.shape[-1] for p in all_probabilities)
        padded = [torch.nn.functional.pad(p, (0, max_len - p.shape[-1]), value=0.0) for p in all_probabilities]
        stacked = torch.stack(padded, dim=0)  # (num_iterations, batch, max_len)
        uncertainty = stacked.var(dim=0).mean().item()

        self.model.eval()
        return transcription, uncertainty
    
    def transcribe_dataset(self, dataset_audios: list):

        transcriptions_list = []
        gt_list = []
        uncertainties = []

        for audio in tqdm(dataset_audios, desc="Transcribing audio", leave=False):
            # Transcribe the audio
            transcription, uncertainty = self.transcribe_audio(audio["audio"]["array"])    
            transcriptions_list.append(transcription)   
            uncertainties.append(uncertainty)
            gt_list.append(audio["sentence"])
        
        return transcriptions_list, gt_list, uncertainties
    
    
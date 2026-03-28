import torch

from datasets import Audio
from tqdm.auto import tqdm
from whisper_wrapper import WhisperWrapper


class TemperatureScaling(WhisperWrapper):
    def __init__(self, model_name: str, temperature: float, sampling_rate: int = 16000, device=torch.device("cpu")):
        super().__init__(model_name, sampling_rate=sampling_rate, device=device)
        self.temperature = temperature

    def transcribe_audio(self, audio_array):
        # Generate input features => output.sequences: token ids, output.scores: tuple of logits per step
        output = self(audio_array, return_dict_in_generate=True, 
                      output_hidden_states=False, output_attentions=False, output_scores=True)
        
        # Fetch transcriptions in text
        transcriptions = self.processor.tokenizer.batch_decode(output.sequences, skip_special_tokens=False, normalize=True)
        # Adds accents
        transcription = self.tildar_oracion(transcriptions[0])

        # Get the temperature scaled softmax of the logits
        logits = torch.stack(output.scores, dim=1)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        # Get the selected token score for each token of the sequence
        probabilities = torch.max(probabilities, dim=2).values
        # Calculate the uncertainty as 1 - the mean probability of the tokens in the sequence
        uncertainty = 1 - probabilities.mean().item()

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

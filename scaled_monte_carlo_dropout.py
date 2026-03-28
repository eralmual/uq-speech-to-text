
import torch
import Levenshtein

from datasets import Audio
from tqdm.auto import tqdm
from whisper_wrapper import WhisperWrapper


class ScaledMonteCarloDropout(WhisperWrapper):
    
    def __init__(self, model_name: str, num_iterations: int, temperature: float, dropout_rate: float = 0.1, sampling_rate: int = 16000, device=torch.device("cpu")):
        super().__init__(model_name, dropout=dropout_rate, sampling_rate=sampling_rate, device=device)
        self.num_iterations = num_iterations
        self.temperature = temperature

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
            probabilities = torch.softmax(logits / self.temperature, dim=-1)
            # Get the max probability per token
            max_probs = torch.max(probabilities, dim=2).values
            uncertainties.append(1 - max_probs.mean().item())

            # Fetch transcription
            decoded = self.processor.tokenizer.batch_decode(output.sequences, skip_special_tokens=True, normalize=True)
            transcriptions.append(self.tildar_oracion(decoded[0]))

        return transcriptions, uncertainties
    
    def transcribe_dataset(self, dataset_audios: list):

        transcriptions_list = []
        gt_list = []
        uncertainties = []

        for audio in tqdm(dataset_audios, desc="Transcribing audio", leave=False):
            # Transcribe the audio
            transcriptions, certainties = self.transcribe_audio(audio["audio"]["array"])    
            transcriptions_list.extend(transcriptions)   
            uncertainties.extend(certainties)
            gt_list.extend(self.num_iterations * [audio["sentence"]])
        
        return transcriptions_list, gt_list, uncertainties
    
    def calcular_distancias_levensthein(self, oracion, transcripciones):
        distancias = []
        longitudes_maximas = []

        for t in transcripciones:
            distancia = Levenshtein.distance(oracion, t)
            distancias.append(distancia)
            longitud_maxima = max(len(oracion), len(t))
            longitudes_maximas.append(longitud_maxima)

        return distancias, longitudes_maximas

    def encontrar_medoid(self, transcripciones):
        medoid = None
        min_total_distance = float('inf')

        for t1 in transcripciones:
            total_distance = 0
            for t2 in transcripciones:
                total_distance += Levenshtein.distance(t1, t2)

            if total_distance < min_total_distance:
                min_total_distance = total_distance
                medoid = t1

        return medoid, min_total_distance

    def uncertainty_MCD(self, transcripciones):
        distancias_totales = []
        medoides = []
        max = []
        longitudes = []
        for i in range(0, len(transcripciones), self.num_iterations):
            transcripciones_grupo = transcripciones[i:i+self.num_iterations-1]
            medoide, distancia_minima = self.encontrar_medoid(transcripciones_grupo)
            distancias_grupo, longitudes_maximas = self.calcular_distancias_levensthein(medoide, transcripciones_grupo)
            max.append(longitudes_maximas)
            distancias_totales.append(distancias_grupo)
            medoides.append(medoide)
            longitudes = [len(medoide) for medoide in medoides]

        return distancias_totales, medoides, max, longitudes

    def dividir_distancias(self, distancias_totales, distancias_maximas):
        resultados = []
        promedios = []

        for i in range(len(distancias_totales)):
            distancia_total = distancias_totales[i]
            distancia_maxima = distancias_maximas[i]

            for j in range(len(distancia_total)):
                if distancia_maxima[j] == 0:
                    resultado = 0.0
                else:
                    resultado = distancia_total[j] / distancia_maxima[j]
                resultados.append(resultado)

        for i in range(0, len(resultados), self.num_iterations - 1):
            grupo = resultados[i:i + self.num_iterations - 1]
            promedio_grupo = sum(grupo) / len(grupo)
            promedios.append(promedio_grupo)

        return promedios
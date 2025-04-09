import jiwer
import torch

from datasets import Audio
from tqdm.auto import tqdm

# Diccionario de palabras sin tilde a palabras con tilde

class WhisperEvaluator():

  def __init__(self, model: torch.nn.Module, processor = None, sampling_rate:int = 16000):
    """
    Builds an instance of Whisper Evaluator
    param model: whisper model to evaluate
    param processor: whisper audio to feature representation (MCC based) converter
    param sampling_rate: use 16 Khz by default
    return instance of the class
    """
    self.sampling_rate = sampling_rate
    self.model = model
    self.processor = processor
    self.diccionario_tildes = {
        "proximas": "próximas",
        "tambien": "también",
        "pasaria": "pasaría",
        "raton": "ratón",
        "mas":"más",
        "autentico":"auténtico",
        "termino":"términos",
        "publico":"público",
        "si":"sí",
        "tu":"tú",
        "relacion":"relación",
        "dificil":"difícil",
        "biologica":"biológica",
        "comun":"común"
    }

  def tildar_oracion(self, oracion):
      """
      Puts accents to the spanish words needed
      param oracion: spanish sentence
      return oracion with accents
      """
      palabras = oracion.split()
      palabras_tildadas = [self.diccionario_tildes.get(palabra, palabra) for palabra in palabras]
      oracion_tildada = " ".join(palabras_tildadas)
      return oracion_tildada

  def transcribe_audio(self, audio_array):
    """Transform the audio to the input using the required representations
    param audio_array: audio array
    param model: model
    param processor: processor
    param sampling_rate: sampling rate
    return transcription 
    """
    # Generate input features
    inputs = self.processor.feature_extractor(audio_array, return_tensors="pt", sampling_rate = self.sampling_rate).input_features.to(self.model.device)

    forced_decoder_ids = self.processor.get_decoder_prompt_ids(language = "es", task = "transcribe")
    self.model.eval()
    with torch.no_grad():
        # Generation was failing for me so had to do this
        generated_ids = 0
        done = False
        while(not(done)):
            try:
                generated_ids = self.model.generate(
                    inputs,
                    forced_decoder_ids=forced_decoder_ids,
                    num_return_sequences=1
                )
                done = True
            except:
                pass

        #fetch transcriptions in text
        transcriptions = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens = False, normalize = True)
        #adds accents
        transcription = self.tildar_oracion(transcriptions[0])
    return transcription

  def transcribe_dataset(self, dataset_audios):
    """
    Transcribes an entire dataset with the format audio["audio"]["array"], audio["audio"]["sampling_rate"]
    and audio["sentence"]
    param dataset_audios: dataset of audios
    param model: model
    param processor: processor
    return transcriptions_list and the corresponding groundtruth 
    """
    #make sure the sampling rate is always 16 kHz
    dataset_audios = dataset_audios.cast_column("audio", Audio(sampling_rate= self.sampling_rate))
    transcriptions_list = []
    gt_list = []

    for audio in tqdm(dataset_audios, desc="Tanscribing audio", leave=False):
        #transcribe the audio
        transcription = self.transcribe_audio(audio["audio"]["array"])    
        transcriptions_list.append(transcription)   
        gt_list.append(audio["sentence"])
    
    return transcriptions_list, gt_list

  def compute_wers(self, transcriptions_all, gt_texts):
    """
    Compute the word error rate per audio 
    param transcriptions_all: list of transcriptions
    param gt_texts: list of groundtruth texts
    return wers in a list
    """
    wers = []
    #preprocessing before computing the WER
    transforms = jiwer.Compose(
        [
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords()

        ]
    )
    #go through each sentence
    for i in range(len(transcriptions_all)):
      gt_text = gt_texts[i]
      transcription_text = transcriptions_all[i]
    
      #compute a per sentence word error rate
      wer = jiwer.wer(
                      [gt_text],
                      [transcription_text],
                      truth_transform=transforms,
                      hypothesis_transform=transforms,
                  )
      wers.append(wer)
      #print(f"Word Error Rate (WER) :", wer, "of audio ", i)
    return wers

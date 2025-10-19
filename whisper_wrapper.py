import jiwer

from torch import device
from datasets import Audio
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class WhisperWrapper():

  def __init__(self, model_name: str, sampling_rate:int = 16000, device: device = device("cpu")):
    """
    Builds an instance of Whisper Evaluator
    param model: whisper model to evaluate
    param processor: whisper audio to feature representation (MCC based) converter
    param sampling_rate: use 16 Khz by default
    return instance of the class
    """
    # Store audio sampling rate 
    self.sampling_rate = sampling_rate
    # Store and configure the model
    self.model = WhisperForConditionalGeneration.from_pretrained(model_name, return_dict_in_generate=True).to(device)
    self.processor = WhisperProcessor.from_pretrained(model_name)
    #self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="es", task="transcribe")
    #self.model.generation_config.forced_decoder_ids = None
    self.model.generation_config.language = "spanish"
    self.model.generation_config.task = "transcribe"
        
    # Accents for generation
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

  def __call__(self, audio, return_dict_in_generate: bool = True,
               output_scores: bool = False, output_hidden_states: bool = True, 
               output_attentions: bool = True):

    inputs = self.processor.feature_extractor(audio, return_tensors="pt", sampling_rate=self.sampling_rate).input_features.to(self.model.device)
    output = self.model.generate(
        inputs,
        task="transcribe",
        language="es",
        return_dict_in_generate=return_dict_in_generate,
        output_scores=output_scores, 
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
        return_legacy_cache=False,
    )

    return output

  def transcribe_audio(self, audio_array):
    """Transform the audio to the input using the required representations
    param audio_array: audio array
    param model: model
    param processor: processor
    param sampling_rate: sampling rate
    return transcription 
    """
    # Generate input features
    generated_ids = self(audio_array, return_dict_in_generate=False, 
                         output_hidden_states = False, output_attentions = False)
    # Fetch transcriptions in text
    transcriptions = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens = False, normalize = True)
    # Adds accents
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
                      reference_transform=transforms,
                      hypothesis_transform=transforms,
                  )
      wers.append(wer)
      #print(f"Word Error Rate (WER) :", wer, "of audio ", i)
    return wers

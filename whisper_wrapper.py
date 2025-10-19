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
    self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="es", task="transcribe")
    #self.model.generation_config.forced_decoder_ids = None
    #self.model.generation_config.language = "spanish"
    #self.model.generation_config.task = "transcribe"
        
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
        forced_decoder_ids=self.forced_decoder_ids,
        return_dict_in_generate=return_dict_in_generate,
        output_scores=output_scores, 
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
        return_legacy_cache=True
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
      Calculate the WER per sentence between the transcripts and their reference texts.
      - transcriptions_all: list[str] or list[dict] (if dict, use the ‘text’ key)
      - gt_texts: list[str]
      Returns: list[float] with one WER per element.
      """
      import jiwer

      def _to_text(x):
          if isinstance(x, str):
              return x
          if isinstance(x, dict):
              # try common keys
              for k in ("text", "transcription", "pred", "prediction"):
                  if k in x:
                      return x[k]
          return str(x)

      # Transforms compatible with jiwer 2.x and 3.x; if not possible, use identity.
      try:
          Compose = jiwer.transformations.Compose
          RemoveEmptyStrings = jiwer.transformations.RemoveEmptyStrings
          Strip = jiwer.transformations.Strip
          ReduceToListOfListOfWords = jiwer.transformations.ReduceToListOfListOfWords
          _truth_transform = Compose([RemoveEmptyStrings(), Strip(), ReduceToListOfListOfWords()])
          _hyp_transform   = Compose([RemoveEmptyStrings(), Strip(), ReduceToListOfListOfWords()])
          _has_transforms = True
      except Exception:
          _truth_transform = None
          _hyp_transform = None
          _has_transforms = False

      wers = []
      for gt_text, trn in zip(gt_texts, transcriptions_all):
          transcription_text = _to_text(trn)
          gt_text = _to_text(gt_text)

          # First try API jiwer<=2.5 (wer with kwargs)
          try:
              if _has_transforms:
                  wer = jiwer.wer(
                      [gt_text],
                      [transcription_text],
                      truth_transform=_truth_transform,
                      hypothesis_transform=_hyp_transform
                  )
              else:
                  wer = jiwer.wer([gt_text], [transcription_text])
          except TypeError:
              # jiwer>=3: use compute_measures (try with transforms, if not, without them)
              try:
                  if _has_transforms:
                      measures = jiwer.compute_measures(
                          [gt_text], [transcription_text],
                          truth_transform=_truth_transform,
                          hypothesis_transform=_hyp_transform
                      )
                  else:
                      measures = jiwer.compute_measures([gt_text], [transcription_text])
                  wer = float(measures["wer"])
              except TypeError:
                  # Minimum fallback (without transforms)
                  wer = jiwer.wer([gt_text], [transcription_text])

          # Ensure simple float type
          try:
              wer = float(wer)
          except Exception:
              wer = float(getattr(wer, "item", lambda: 0.0)())

          wers.append(wer)

      return wers

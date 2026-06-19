import torch
import numpy as np

from tqdm.auto import tqdm
from datasets import Dataset
from abc import ABC, abstractmethod
from whisper_wrapper import WhisperWrapper


class UQMethod(WhisperWrapper, ABC):
    """
    Abstract base class for uncertainty quantification (UQ) methods.

    Subclasses implement the UQ-specific logic in transcribe_audio. The shared
    transcribe_dataset and evaluate template methods take care of iterating
    over the dataset and computing WERs so that every method exposes the same
    public interface.
    """

    @abstractmethod
    def transcribe_audio(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe a single audio and produce its uncertainty score.

        param audio_array: raw audio samples
        return tuple (transcription, uncertainty)
        """
        ...

    def transcribe_dataset(self, dataset: Dataset) -> tuple[list[str], list[str], list[float]]:
        """
        Transcribe an entire dataset by delegating to transcribe_audio.

        param dataset: dataset with audios and transcriptions
        return tuple (transcriptions_list, gt_list, uncertainties)
        """
        transcriptions_list = []
        gt_list = []
        uncertainties = []

        for sample in tqdm(dataset, desc="Transcribing audio", leave=False):
            transcription, uncertainty = self.transcribe_audio(sample["audio"])
            transcriptions_list.append(transcription)
            uncertainties.append(uncertainty)
            gt_list.append(sample["sentence"])

        return transcriptions_list, gt_list, uncertainties

    def evaluate(self, dataset: Dataset) -> tuple[list[float], list[float], list[str], list[str]]:
        """
        Transcribe a dataset and compute the per-audio WERs and UQ scores.

        Default template suitable for methods that yield one transcription and
        one uncertainty per audio. Subclasses with a different alignment (e.g.
        multiple passes per audio) should override this method.

        param dataset: dataset with audios and transcriptions
        return tuple (wers, uq_scores, transcriptions, references)
        """
        transcriptions_list, gt_list, uq_scores = self.transcribe_dataset(dataset)
        wers = self.compute_wers(transcriptions_list, gt_list)
        return wers, uq_scores, transcriptions_list, gt_list

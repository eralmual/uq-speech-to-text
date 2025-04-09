import os
import librosa

import numpy as np
import pandas as pd

from zipfile import ZipFile
from IPython.display import Audio as AudioDis
from huggingface_hub import snapshot_download
from datasets import load_dataset, Audio, Dataset


class Dataloader():

    @staticmethod
    def load_dataset_ciempies():
      """
      Load dataset from ciempiess dataset
      """
      dataset = load_dataset("ciempiess/ciempiess_test", split="test")
      dataset = dataset.cast_column("audio", Audio(sampling_rate = 16_000))
      #fetch audios
      audios = load_dataset("ciempiess/ciempiess_test")["test"]["audio"]
      #fetch texts from groundtruth
      texts = dataset["normalized_text"] #transcripciones
      #print("texts ", texts)
      return audios, texts
      
    @staticmethod
    def load_dataset_raw_uq(repo_id="saul1917/SpaTrans_UQ_Bench_raw", partition_type = "test", partition_id = "partition_0", audio_extension = ".wav"):
        """""
        Load dataset from hugging face repo
        param repo_id: id of the repo
        param partition_type: type of the partition (test, calibration or fine_tune)
        param partition_id: Partition name
        return audios and texts
        """
        # Donwload the data as is
        download_path = snapshot_download(repo_id=repo_id, repo_type="dataset")

        # Generate I/O paths
        zip_path = os.path.join(download_path, partition_id, partition_type + ".zip")
        csv_path = os.path.join(download_path, partition_id, partition_type + ".csv")
        unzip_path = os.path.join(download_path, partition_id)

        # This partition is compresed without a top dir
        if(partition_id == "fine_tuning_partition"):
           partition_path = os.path.join(download_path, partition_id)
        else:
            partition_path = os.path.join(download_path, partition_id, partition_type)

        # Read the metadata
        data_frame_metadata = pd.read_csv(csv_path)
        # Extract the files in the full path corresponding to the audios
        try:
            rar_file = ZipFile(zip_path, 'r')
            rar_file.extractall(path = unzip_path)
            rar_file.close()
        except:
            print("Error extracting audios, they might've been extracted already")

        # Build the dataset
        audios = []
        transcriptions = []
        list_dict_audios = []
        for _, row in data_frame_metadata.iterrows():

            # Build the target audio path
            audio_path = os.path.join(partition_path, row["audio_id"] + audio_extension)
            # Load the audio
            audio, sample_rate = librosa.load(audio_path)
            # Store the audio and transcription
            audios.append(audio)
            transcriptions.append(row["transcription"])
            # Build HF dataset
            list_dict_audios.append({"sentence": row["transcription"], 
                                     "audio":   {
                                                "array": audio, 
                                                "sampling_rate": sample_rate
                                                }
                                    })
            
        # Generate HF dataset
        hf_dataset = Dataset.from_list(list_dict_audios)

        return hf_dataset, audios
    
    @staticmethod
    def load_uq_partitions(partition: str, start_part: int = 1, last_part: int = 10) -> tuple[list, list]:
        df_list = []
        audios_list = []
        for i in range(start_part, last_part):
            # Load the fine tunning and test data
            df, audios = Dataloader.load_dataset_raw_uq(partition_type = partition, partition_id = f"partition_{i}")
            # Merge
            df_list.append(df)
            audios_list.append(audios)

        return df_list, audios_list
    
      
    @staticmethod
    def contaminate_audio_array(audio_array, noise_audios_dataset, weight_noise = 0.6, sampling_rate = 16000):
        """""
        Contaminate audio with noise
        param weight_noise: weight of noise in the audio
        param sampling_rate: sampling rate of the audio
        param audio_array: input audio array
        param noise_audios_dataset: noise dataset
        return audio_combined_array
        """
        sample_clean_data_array = audio_array

        #load noisy dataset

        train_dataset = noise_audios_dataset['train']
        noise_audios = train_dataset['audio']
        sample_noise = noise_audios[1]
        sample_noise_array = sample_noise["array"]

        #Make sample_noise_array audio have the same length of sample_clean_data_array
        min_length =len(sample_clean_data_array)
        if len(sample_noise_array)<min_length:
          sample_ruido_data_array=np.tile(sample_noise_array, int(np.round(min_length/len(sample_noise_array))+1))[:min_length]
        else:
          sample_ruido_data_array = sample_noise_array[:min_length]
        #weight clean and noisy audio contribution
        weight_clean =  1 - weight_noise
        audio_combined_array = (weight_clean * sample_clean_data_array) + (weight_noise * sample_ruido_data_array)
        # Debugging
        AudioDis(data = audio_combined_array, rate = sampling_rate)
        return audio_combined_array
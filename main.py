import torch

from experiment import run_experiment, ExperimentType, DatasetType

"""
gen_kwargs = {  "return_dict_in_generate": True,
                "output_scores": False, 
                "output_hidden_states": True,
                "output_attentions": False}

embedding_kwargs = {"use_decoder": False,
                    "use_encoder": True,}
"""
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev = torch.device("xpu" if torch.xpu.is_available() else dev)
print("Using", dev, "device")

folds = 10
sample_size = 2
num_iterations = 2
dropout_rate = 0.01
temperature = 1.35


run_experiment(ExperimentType.TS, DatasetType.CALIBRATION, k_folds=folds, sample_size=sample_size,
               temperature=temperature,output_dir="results/experiment_test", device=dev)

run_experiment(ExperimentType.MCD, DatasetType.CALIBRATION, k_folds=folds, sample_size=sample_size,
               dropout_rate=dropout_rate, dropout_iterations=num_iterations,
               output_dir="results/experiment_test", device=dev)

run_experiment(ExperimentType.LMCD, DatasetType.CALIBRATION, k_folds=folds, sample_size=sample_size,
               dropout_rate=dropout_rate, dropout_iterations=num_iterations,
               output_dir="results/experiment_test", device=dev)
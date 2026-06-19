import torch

from experiment import run_experiment, ExperimentType, DatasetType


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev = torch.device("xpu" if torch.xpu.is_available() else dev)
print("Using", dev, "device")


num_iterations = 10
dataset_type = DatasetType.TEST

#run_experiment(ExperimentType.TS, dataset_type,
#               temperature=1.126,
#               output_dir="results/experiments", device=dev)

run_experiment(ExperimentType.MCD, dataset_type,
               dropout_rate=0.003, dropout_iterations=num_iterations,
               output_dir="results/experiments", device=dev)

run_experiment(ExperimentType.LMCD, dataset_type,
               dropout_rate=0.037, dropout_iterations=num_iterations,
               output_dir="results/experiments", device=dev)
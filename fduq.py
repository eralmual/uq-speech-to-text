import torch

from experiment import run_experiment

gen_kwargs = {  "return_dict_in_generate": True,
                "output_scores": False, 
                "output_hidden_states": True,
                "output_attentions": False}

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev = torch.device("xpu" if torch.xpu.is_available() else dev)
print("Using", dev, "device")

run_experiment("cat-flatten-k1", gen_kwargs, dev, 1)
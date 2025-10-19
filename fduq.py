import torch

from experiment import run_experiment

gen_kwargs = {
    "return_dict_in_generate": True,
    "output_scores": False,
    "output_hidden_states": True,
    "output_attentions": False
}

embedding_kwargs = {
    "use_decoder": False,
    "use_encoder": True
}

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else dev)
print("Using", dev, "device")

for k in [1, 5]:
    run_experiment(f"cat-flatten-k{k}", gen_kwargs, embedding_kwargs, dev, k)
    run_experiment(f"enc_only-cat-flatten-k{k}", gen_kwargs, {"use_decoder": False, "use_encoder": True}, dev, k)

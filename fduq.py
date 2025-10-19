import torch
from experiment import run_experiment

gen_kwargs = {  "return_dict_in_generate": True,
                "output_scores": False, 
                "output_hidden_states": True,
                "output_attentions": False}

embedding_kwargs = {"use_decoder": False,
                    "use_encoder": True,}

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev = torch.device("xpu" if torch.xpu.is_available() else dev)
print("Using", dev, "device")

k = 1
# TBD use max
run_experiment(f"enc_only-cat-flatten-k{k}", gen_kwargs, embedding_kwargs, 
               device=dev, top_k=k)

k= 5
#run_experiment(f"cat-flatten-k{k}", gen_kwargs, dev, k)
#run_experiment(f"enc_only-cat-flatten-k{k}", gen_kwargs, embedding_kwargs, dev, k)

#hist = torch.tensor([1.0000e-06, 5.2083e-06, 7.9437e-06, 8.0840e-06, 7.7333e-06, 8.7387e-06, 2.3748e-05, 3.7547e-04, 8.5873e-04, 6.7730e-03, 4.1596e-01, 1.1086e-01, 3.2592e-03, 4.4224e-04, 5.8748e-05, 1.1194e-05, 5.9097e-06, 7.0319e-06, 6.9384e-06, 4.2264e-06, 2.6600e-06, 1.0000e-06])
#buckets = torch.tensor([-3.4028e+38, -1.7872e+01, -1.6015e+01, -1.4159e+01, -1.2302e+01, -1.0446e+01, -8.5893e+00, -6.7329e+00, -4.8764e+00, -3.0200e+00, -1.1636e+00,  6.9286e-01,  2.5493e+00,  4.4057e+00,  6.2622e+00, 8.1186e+00,  9.9750e+00,  1.1831e+01,  1.3688e+01,  1.5544e+01, 1.7401e+01,  1.9257e+01])
# 
# 
# 
# 

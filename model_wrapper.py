import os
import torch
import functools

from collections import OrderedDict
from transformers import WhisperForConditionalGeneration, WhisperModel, WhisperTokenizer, WhisperProcessor, WhisperFeatureExtractor


class WhisperWrapper():
    """
    Builds and trains a Whisper model.
    """
    def __init__(self, model_name: str = "openai/whisper-small", sampling_rate: int = 16000, device: torch.device = torch.device("cpu")):
        # Load the model
        self.model = WhisperModel.from_pretrained(model_name).to(device)
        self.model_cond_gen = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        # Config the conditional generation model
        self.model_cond_gen.generation_config.language = "spanish"
        self.model_cond_gen.generation_config.task = "transcribe"
        self.model_cond_gen.generation_config.forced_decoder_ids = None
        # Other attributes
        self.device = device
        self.sampling_rate = sampling_rate


# Code extracted from https://github.com/sebamenabar/Pytorch-IntermediateLayerGetter
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class IntermediateLayerGetter:

    CosineSimilarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def __init__(self, model: torch.nn.Module, return_layers: list, keep_output: bool = True):
        """Wraps a Pytorch module to get intermediate values
        
        Arguments:
            model {nn.module} -- The Pytorch module to call
            return_layers {list} -- List with the names of the selected submodules
            to return the output (format: {[current_module_name]: [desired_output_name]},
            current_module_name can be a nested submodule, e.g. submodule1.submodule2.submodule3)
        
        Keyword Arguments:
            keep_output {bool} -- If True model_output contains the final model's output
            in the other case model_output is None (default: {True})

        Returns:
            (mid_outputs {OrderedDict}, model_output {any}) -- mid_outputs keys are 
            your desired_output_name (s) and their values are the returned tensors
            of those submodules (OrderedDict([(desired_output_name,tensor(...)), ...).
            See keep_output argument for model_output description.
            In case a submodule is called more than one time, all it's outputs are 
            stored in a list.
        """
        self._model = model
        self.return_layers = dict(zip(return_layers, return_layers))
        self.keep_output = keep_output
        
    def __call__(self, *args, **kwargs):
        ret = OrderedDict()
        handles = []
        for name, new_name in self.return_layers.items():
            layer = rgetattr(self._model, name)
            
            def hook(module, input, output, new_name=new_name):
                if new_name in ret:
                    if type(ret[new_name]) is list:
                        ret[new_name].append(output.detach())
                    else:
                        ret[new_name] = [ret[new_name], output.detach()]
                else:
                    ret[new_name] = output.detach()
            try:
                h = layer.register_forward_hook(hook)
            except AttributeError as e:
                raise AttributeError(f'Module {name} not found')
            handles.append(h)
            
        if self.keep_output:
            output = self._model(*args, **kwargs)
        else:
            self._model(*args, **kwargs)
            output = None
            
        for h in handles:
            h.remove()
        
        return ret, output
    
    @staticmethod
    def get_layer_names(model, name = ""):
        children = model.named_children()
        if len(list(children)) > 0:
            layer_names = []
            for child_name, module in model.named_children():
                layer_names += IntermediateLayerGetter.get_layer_names(module, f"{name}.{child_name}")

            return layer_names
        
        else:
            return [name[1:]]
        

    @staticmethod
    def calculate_block_influence(layer_outputs: dict):
        """
        Calculate the influence of each block in the model on the final output.
        
        Arguments:
            later_outputs {dict} -- Dictionary with the outputs of the layers in the model.
        
        Returns:
            influence {dict} -- BI for each layer in the model.
        """
        layers = list(layer_outputs.keys())
        bi = dict(zip(layers, len(layers)*[[]]))

        for l in range(len(layers) - 1):
            if(layer_outputs[layers[l]].shape == layer_outputs[layers[l + 1]].shape):
                bi[layers[l]] = 1 - IntermediateLayerGetter.CosineSimilarity(layer_outputs[layers[l]].detach(), 
                                                                            layer_outputs[layers[l + 1]].detach()).detach()
                
        return bi
    
from typing import List, Tuple, Iterable
import argparse
import torch
import transformers
import torch.nn as nn

try:
    import utils
except ModuleNotFoundError:
    from . import utils
import copy
import numpy as np
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import itertools

class LoRALayerWrapper(nn.Module):
    def __init__(self, base_module: nn.Module, lora_rank: int):
        super().__init__()

        self.base_module = base_module

        ###
        ### Set up your LoRA-augmented layer here.
        ### You should initialize your parameters so that the residual matrix AB^T is zero,
        ###     but be careful how you do this (i.e., make sure you eventually get
        ###     non-zero gradients to both matrices during fine-tuning)!
        ### For randomly initializing the parameters, use torch.randn.
        ### Note: you should use nn.Parameter to wrap your parameters so that they are registered as
        ### learnable.
        ### Initialization hint: what do the gradients look like after 1 and 2 steps of fine-tuning
        ###     if you initialize both A and B to zero? What about if just one is zero?
        ###
        self.lora_A, self.lora_B = None, None
        ## YOUR CODE HERE, complete for Q2.2b
        mid=lora_rank//2
        A=torch.randn(self.base_module.weight.shape[0], lora_rank)
        A[:,mid:]=0
        B=torch.randn(self.base_module.weight.shape[1], lora_rank)
        B[:,:mid]=0
        self.lora_A = nn.Parameter(A)
        self.lora_B = nn.Parameter(B)
        #assert False, "Complete this for Q2.2b"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_module(x)  # The output of the pre-trained module.
        ### Perform the forward pass of your LoRA-augmented layer here.
        ### Note: you don't need to ever explicitly construct the matrix AB^T.
        ### Hint: matrix multiplication is associative.
        ###
        ## YOUR CODE HERE, complete for Q2.2b
        low_rank_weights = torch.matmul(self.lora_B, self.lora_A.T)
        x_reshaped = x.view(-1, x.shape[-1])
        loras_out = torch.matmul(x_reshaped,low_rank_weights.T)
        loras_out = loras_out.view(x.shape[0], x.shape[1], -1)
        return base_out+loras_out
        #assert False, "Complete this for Q2.2b"
        #pass


def parameters_to_fine_tune(model: nn.Module, mode: str) -> Iterable[nn.Parameter]:
    """
    Select the parameters in `model` that should be fine-tuned in mode `mode`.

    For every mode except "all", the model is going to be GPT-2 (transformers.GPT2LMHeadModel).
    We encourage you to print the model architecture (e.g. by placing a PDB breakpoint and doing
    `print(model)`), and identifying where the model layers are located.

    Note: this function only returns the list of parameters to fine tune. It doesn't change the
    `requires_grad` component of any parameters, and you should not touch that in this assignment!

    Args:
      model: the model we're fine-tuning
      mode: the fine-tuning mode we're using; may be 'all', 'last', 'first',
        'middle', or 'loraN' (where N is an integer)

    Returns:
      A list of nn.Parameters of `model` that should be fine-tuned in the given
        fine-tuning mode.
    """
    parameters_to_fine_tune: List[nn.Parameter] = None
    if mode == "all":
        # Every learnable parameter from `model` should be fine-tuned.
        # Complete this for Q0.1
        for param in model.parameters():
            if parameters_to_fine_tune is None:
                parameters_to_fine_tune=[param]
            else:
                parameters_to_fine_tune.append(param)
#        assert False, "Complete this for Q0.1"
    elif mode == "swin_unet_last":
        # Only fine tune the last 2 transformer blocks
        # Complete this for Q2.1
        flag='swin_unet.layers.2.blocks.2.mlp'
        anames = [n[0] for n in model.named_parameters()]
        pnames = [n[0] for n in model.named_parameters() if n[0].startswith(flag)]
        for k,v in model.named_parameters():
            if k in pnames:
                if parameters_to_fine_tune is None:
                    parameters_to_fine_tune=[v]
                else:
                    parameters_to_fine_tune.append(v)
       # assert False, "Complete this for Q2.1"
    elif mode == "swin_unet_decoder":
        # Only fine tune the last 2 transformer blocks
        # Complete this for Q2.1
        flag='swin_unet.layers.2'
     #   anames = [n[0] for n in model.named_parameters()]
        pnames = [n[0] for n in model.named_parameters() if n[0].startswith(flag)]
        for k,v in model.named_parameters():
            if k in pnames:
                if parameters_to_fine_tune is None:
                    parameters_to_fine_tune=[v]
                else:
                    parameters_to_fine_tune.append(v)
    elif mode == "segformer_decoder":
        # Only fine tune the last 2 transformer blocks
        # Complete this for Q2.1
        flag='decode'
        anames = [n[0] for n in model.named_parameters()]
        pnames = [n[0] for n in model.named_parameters() if n[0].startswith(flag)]
        for k,v in model.named_parameters():
            if k in pnames:
                if parameters_to_fine_tune is None:
                    parameters_to_fine_tune=[v]
                else:
                    parameters_to_fine_tune.append(v)
       # assert False, "Complete this for Q2.1"
    elif mode == "first":
        # Only fine tune the first 2 transformer blocks
        # Complete this for Q2.1
        names = [n[0] for n in model.named_parameters() if n[0].startswith('transformer.h.')]
        lnum = [int(n[14:17]) if n[14:17].isnumeric() else (int(n[14:16]) if n[14:16].isnumeric() else int(n[14])) for n
                in names]
        lnum = sorted(list(set(lnum)))
        sflag = ['transformer.h.' + str(d)+'.' for d in lnum[:2]]
        for k, v in model.named_parameters():
            if k.startswith(sflag[0]) or k.startswith(sflag[1]):
                if parameters_to_fine_tune is None:
                    parameters_to_fine_tune = [v]
                else:
                    parameters_to_fine_tune.append(v)
#        assert False, "Complete this for Q2.1"
    elif mode == "middle":
        # Only fine tune middle 2 transformer blocks
        # Complete this for Q2.1
        names = [n[0] for n in model.named_parameters() if n[0].startswith('transformer.h.')]
        lnum = [int(n[14:17]) if n[14:17].isnumeric() else (int(n[14:16]) if n[14:16].isnumeric() else int(n[14])) for n
                in names]
        lnum = sorted(list(set(lnum)))
        mid=len(lnum)//2
        sflag = ['transformer.h.' + str(d)+'.' for d in lnum[(mid-1):(mid+1)]]
        for k, v in model.named_parameters():
            if k.startswith(sflag[0]) or k.startswith(sflag[1]):
                if parameters_to_fine_tune is None:
                    parameters_to_fine_tune = [v]
                else:
                    parameters_to_fine_tune.append(v)
       # assert False, "Complete this for Q2.1"
    elif mode is not None and mode.startswith("lora"):
        # Only fine tune the rank decomposition matrices A and B from the LoRA layers.
        # Hint: consider using the `.modules()` function of nn.Module and checking for modules that
        # are an instance of LoRALayerWrapper.
        # Complete this for Q2.2c
        suffix=['lora_A','lora_B']
        for module in model.modules():
            if isinstance(module, LoRALayerWrapper):
                for k,v in module.named_parameters():
                    if k.endswith(suffix[0]) or k.endswith(suffix[1]):
                        if parameters_to_fine_tune is None:
                            parameters_to_fine_tune = [v]
                        else:
                            parameters_to_fine_tune.append(v)
#        assert False, "Complete this for Q2.2c"
    else:
        parameters_to_fine_tune= model.parameters()
    return parameters_to_fine_tune

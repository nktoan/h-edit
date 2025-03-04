
import os
from tqdm import tqdm
from typing import Union, List

import torch
import torch.nn.functional as F
from torch import autocast, inference_mode
from torch.optim.adam import Adam

from diffusers.utils.torch_utils import randn_tensor

def encode_text(model, prompts: Union[str, List[str]]):
    """
    Encode text prompts into embeddings using the model's tokenizer and text encoder.

    Parameters:
        model (object): Model containing a tokenizer and text encoder.
        prompts (str or List[str]): Input text prompt(s) to be encoded.

    Returns:
        torch.Tensor: Encoded text representation.
    """
        
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length, 
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
        
    return text_encoding

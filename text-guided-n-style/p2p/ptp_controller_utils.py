
from p2p.ptp_classes import LocalBlend, AttentionReplace, AttentionRefine, AttentionReweight
from p2p.ptp_utils import get_time_words_attention_alpha, get_word_inds

from typing import Optional, Union, Tuple, List, Callable, Dict

import difflib
import nltk
from nltk.tokenize import word_tokenize

import torch

def preprocessing(src_prompt: str, tar_prompt: str, is_global_edit: bool = True) -> Tuple:
    """
    Find local blend words and words_to_focus (increase attention weights)
    """
    def get_differences(src_prompt, trg_prompt):
        src_words = word_tokenize(src_prompt)
        trg_words = word_tokenize(trg_prompt)
        
        matcher = difflib.SequenceMatcher(None, src_words, trg_words)
        src_text = []
        trg_text = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                src_text.extend(src_words[i1:i2])
                trg_text.extend(trg_words[j1:j2])
            elif tag == 'insert':
                trg_text.extend(trg_words[j1:j2])
            elif tag == 'delete':
                src_text.extend(src_words[i1:i2])
        
        return ' '.join(src_text), ' '.join(trg_text)

    src_text, tar_text = get_differences(src_prompt, tar_prompt)

    if len(src_text) == 0 or len(tar_text) == 0:
        blend_word = None
    elif is_global_edit:
        blend_word = (((src_text,), (tar_text,))) #hard to choose what to focus, require human knowledge
    else:
        blend_word = None

    words_to_focus = tar_text.split() #intensify their attention values

    if len(words_to_focus) > 0:
        eq_params = {"words": tuple(words_to_focus), "values": tuple(1.5 for _ in words_to_focus)}
    else:
        eq_params = None
        
    return blend_word, eq_params 

def preprocessing_attn_focus(src_prompt: str, tar_prompt: str, is_global_edit: bool = True) -> Tuple:
    def get_differences(src_prompt, trg_prompt):
        src_words = word_tokenize(src_prompt)
        trg_words = word_tokenize(trg_prompt)
        
        matcher = difflib.SequenceMatcher(None, src_words, trg_words)
        src_text = []
        trg_text = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                src_text.extend(src_words[i1:i2])
                trg_text.extend(trg_words[j1:j2])
            elif tag == 'insert':
                trg_text.extend(trg_words[j1:j2])
            elif tag == 'delete':
                src_text.extend(src_words[i1:i2])
        
        return ' '.join(src_text), ' '.join(trg_text)

    src_text, tar_text = get_differences(src_prompt, tar_prompt)

    if len(src_text) == 0 or len(tar_text) == 0:
        blend_word = None
    elif is_global_edit:
        blend_word = (((src_text,), (tar_text,))) #hard to choose what to focus, require human knowledge
    else:
        blend_word = None

    words_to_focus = tar_text.split() #intensify their attention values

    if len(words_to_focus) > 0:
        eq_params = {"words": tuple(words_to_focus), "values": tuple(1.25 for _ in words_to_focus)}
    else:
        eq_params = None
        
    return blend_word, eq_params 

def get_equalizer(text: str,
                  word_select: Union[int, Tuple[int, ...]],
                  values: Union[List[float], Tuple[float, ...]],
                  tokenizer):
    
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def make_controller(prompts: List[str],
                    is_replace_controller: bool,
                    cross_replace_steps: Dict[str, float],
                    self_replace_steps: float,
                    blend_word=None,
                    equilizer_params=None,
                    num_steps=None,
                    tokenizer=None,
                    device=None):
    """
    Make controller for P2P
    """

    if blend_word is None:
        lb = None
    else:
        lb = LocalBlend(prompts, num_steps, blend_word, tokenizer=tokenizer, device=device)
    if is_replace_controller:
        controller = AttentionReplace(prompts, num_steps, cross_replace_steps=cross_replace_steps, 
                self_replace_steps=self_replace_steps, local_blend=lb, tokenizer=tokenizer, device=device)
    else:
        controller = AttentionRefine(prompts, num_steps, cross_replace_steps=cross_replace_steps,
                self_replace_steps=self_replace_steps, local_blend=lb, tokenizer=tokenizer, device=device)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"], tokenizer=tokenizer)
        controller = AttentionReweight(prompts, num_steps, cross_replace_steps=cross_replace_steps,
                self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller,
                tokenizer=tokenizer, device=device)
    return controller
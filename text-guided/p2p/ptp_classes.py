"""
This code was originally taken from
https://github.com/google/prompt-to-prompt
"""

LOW_RESOURCE = False #set to False for consistency
MAX_NUM_WORDS = 77

from typing import Optional, Union, Tuple, List, Callable, Dict
import p2p.ptp_utils as ptp_utils
import p2p.seq_aligner as seq_aligner
import torch
import torch.nn.functional as nnf
import abc
import numpy as np

class LocalBlend:
    def __init__(self, prompts: List[str], num_steps: int, words: List[List[str]], substruct_words=None, start_blend=0.2, th=(.3, .3), tokenizer=None, device=None):
        self.max_num_words = MAX_NUM_WORDS
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, self.max_num_words) #(2,1,1,1,1,77)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, self.max_num_words) #(0,1,1,1,1,77)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * num_steps)
        self.counter = 0 
        self.th=th

    def get_mask(self, maps, alpha, use_pool, x_t): 
        k = 1
        maps = (maps * alpha).sum(-1).mean(1) #([2, 40, 1, 16, 16, 77]) * (2,1,1,1,1,77)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    

    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3] #down_cross: 2,3; up_cross: 1,2
            #each map: shape: ([2, 8, 1, 16, 16, 77])

            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, self.max_num_words) for item in maps]
            maps = torch.cat(maps, dim=1) #maps: shape - ([2, 40, 1, 16, 16, 77])
            mask = self.get_mask(maps, self.alpha_layers, True, x_t)

            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False, x_t)
                mask = mask * maps_sub

            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

class AttentionControl(abc.ABC):

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str, save_attn: bool):
        raise NotImplementedError

    @property
    # revised for low resource
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    # revised for low resource
    def __call__(self, attn, is_cross: bool, place_in_unet: str, save_attn: bool):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet, save_attn)
            else:
                h = attn.shape[0]
                # only use text input, cut off unconditional attn maps
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet, save_attn)
        
        if save_attn == False:
            return attn
        
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers: ####TODO####
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0 

class EmptyControl(AttentionControl):
    def forward (self, attn, is_cross: bool, place_in_unet: str, save_attn: bool):
        return attn
    
class AttentionStore(AttentionControl):
    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str, save_attn: bool = True):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            if save_attn:
                self.step_store[key].append(attn)

        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()
    
    # average with step
    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

class AttentionControlEdit(AttentionStore, abc.ABC):

    def __init__(self, 
                 prompts, 
                 num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend],
                 tokenizer,
                 device):
        super(AttentionControlEdit, self).__init__()
        # add tokenizer and device here

        self.tokenizer = tokenizer
        self.device = device

        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, self.tokenizer).to(self.device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        # if att_replace.shape[2] <= 16 ** 2: # ???
        if att_replace.shape[2] <= 32 ** 2: #depends on the layers (maybe depend on the using model)
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    def forward(self, attn, is_cross: bool, place_in_unet: str, save_attn: bool):
        # attention store
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet, save_attn)
        # FIXME not replace correctly
        # replace attention map
        
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size) #self.batch_size = len(prompt) = 2 (src & edit)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            # print(f'attn.shape: {attn.shape}') #([2, 8, 4096, 77])

            attn_base, attn_replace = attn[0], attn[1:] #[8, 4096, 77] and [1, 8, 4096, 77]

            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                # if xa == 0.6 -> that means (0.6 * the number of steps) initial steps of editing goes with "self.replace_cross_attention(attn_base, attn_replace)" (which is the source prompt's cross attention map)
                # the rest goes with (attn_replace) - edit prompt's cross-attention maps
                attn_replace_new = self.replace_cross_attention(attn_base, attn_replace) * alpha_words + (1 - alpha_words) * attn_replace
                attn[1:] = attn_replace_new
            else:
                # if sa == 0.2 -> that means (0.2 * the number of steps) initial steps of editing goes with "self.replace_self_attention(attn_base, attn_replace)" (which is the source prompt's SELF attention map)
                # the rest goes with edit prompt's SELF-attention maps

                attn[1:] = self.replace_self_attention(attn_base, attn_replace)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

class AttentionReplace(AttentionControlEdit):
    def __init__(self,
                 prompts,
                 num_steps: int,
                 cross_replace_steps: float,
                 self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,
                 tokenizer=None,
                 device=None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer, device)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, self.tokenizer).to(self.device) #(1,77,77)

    def replace_cross_attention(self, attn_base, att_replace):
        d = torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
        return d
        

class AttentionRefine(AttentionControlEdit):
    def __init__(self, prompts,
                 num_steps: int,
                 cross_replace_steps: float,
                 self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,
                 tokenizer=None,
                 device=None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer, device)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, self.tokenizer)
        self.mapper, alphas = self.mapper.to(self.device), alphas.to(self.device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

class AttentionReweight(AttentionControlEdit):
    def __init__(self,
                 prompts,
                 num_steps: int,
                 cross_replace_steps: float,
                 self_replace_steps: float,
                 equalizer,
                 local_blend: Optional[LocalBlend] = None,
                 controller: Optional[AttentionControlEdit] = None,
                 tokenizer=None,
                 device=None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer, device)
        self.equalizer = equalizer.to(self.device)
        self.prev_controller = controller

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float], Tuple[float, ...]], tokenizer):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer

from PIL import Image

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompts=None):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0, prompts=None, tokenizer=None):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select, prompts)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    return(ptp_utils.view_images(np.stack(images, axis=0)))
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))

def run_and_display(model, prompts, controller, latent=None, run_baseline=False, generator=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(model, prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ld
    

def load_512(image_path, left=0, right=0, top=0, bottom=0, device=None):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(device)

    return image

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


def get_variance(model, timestep):
    """
    Compute the variance for a given timestep in DDIM sampling formula (\omega_{t,t-1}^2 in Eq. 3 of our paper)

    Parameters:
        model: Diffusion model with a scheduler providing diffusion parameters.
        timestep (int): Current timestep in the reverse diffusion process.

    Returns:
        torch.Tensor: Computed variance for the given timestep.
    """

    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance

def reverse_step(model, model_output, timestep, sample, eta = 0, variance_noise=None, return_pred_x0 = False, return_mu = False, is_ddim_inversion = False):
    """
    Perform a reverse diffusion step to compute the previous latent sample.

    Parameters:
        model: Diffusion model with a scheduler providing diffusion parameters.
        model_output (torch.Tensor): Predicted noise output.
        timestep (int): Current diffusion timestep.
        sample (torch.Tensor): Current latent sample.
        eta (float, optional, default=0): Stochasticity factor controlling variance. 
        #(eta = 0 -> DDIM sampling, eta = 1 -> DDPM sampling)
        variance_noise (torch.Tensor, optional, default=None): z_t for DDPM sampling (see Eq. 3 of our paper for more details) (used if eta > 0).
        return_pred_x0, return_mu: returning options

        is_ddim_inversion: THIS IS IMPORTANT ARGUMENT! We decide the function is currently performing DDIM or DDPM sampling.
        This happens since we set eta = 1 in DDIM sampling to account for u_t^orig in the computation of x_{t-1} (see Eq. 7 in our paper)
    
    Returns:
        Depends on cases to return tuple: 
        - (prev_sample, pred_original_sample) if we want to return pred_x0 
        - (prev_sample, mu_prev_sample) if we want to return the mean of p(xtm1 | xt)
        - only the previous sample prev_sample

    """
    
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    # 4. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)    
    # variance = self.scheduler._get_variance(timestep, prev_timestep)
    variance = get_variance(model, timestep) 
    
    # Take care of asymetric reverse process (asyrp)
    model_output_direction = model_output
    # 5. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
    if is_ddim_inversion:
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * model_output_direction
    else:
        pred_sample_direction = (1 - alpha_prod_t_prev - (eta ** 2) * variance) ** (0.5) * model_output_direction
        
    # 6. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    mu_prev_sample = prev_sample
    # 7. Add noice if eta > 0
    if eta > 0:
        if is_ddim_inversion:
            prev_sample = prev_sample + eta * variance_noise 
            sigma_z = eta * variance_noise             
        else:
            if variance_noise is None:
                variance_noise = torch.randn(model_output.shape, device=model.device)
            sigma_z =  eta * variance ** (0.5) * variance_noise
            prev_sample = prev_sample + sigma_z

    if (return_pred_x0 == True):
        return prev_sample, pred_original_sample
    elif (return_mu == True):
        return prev_sample, mu_prev_sample
    else:
        return prev_sample

def reverse_step_pred_x0(model, model_output, timestep, sample, eta = 0, variance_noise=None):
    """
    Similar to `reverse_step` function but only returns the predicted x0.
    """

    # 1. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t
    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    return pred_original_sample

def slerp(val, low, high):
    """ 
    taken from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/4
    """
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

def slerp_tensor(val, low, high):
    """ 
    the interpolation used in negtive prompt inversion
    """
    shape = low.shape
    res = slerp(val, low.flatten(1), high.flatten(1))
    return res.reshape(shape)

"""

Below are util functions for h-edit specifically:
- def compute_full_coeff()

"""

def compute_full_coeff(model, timestep, prev_timestep, eta, is_ddim_inversion = False):
    """
    Compute the coefficient sqrt(1 - alphabar_{t-1} - w_{t,t-1}^2) in Eq. 23 or Eq. 25 of our paper. 

    Parameters:
        model         : Model with a scheduler providing diffusion parameters.
        timestep      : The current time step 
        prev_timestep : The previou time step
        etas          : eta can be 0.0 or 1.0
        is_ddim_inversion: can be True or False depends on the editing method
    
    Returns:
        The coefficient 
    """

    alpha_bar = model.scheduler.alphas_cumprod
    sigma = (1-alpha_bar) ** 0.5
    a = alpha_bar ** 0.5
    
    omega_timestep_prev_timestep = eta * (sigma[prev_timestep] / (sigma[timestep] * a[prev_timestep])) * \
                        ((alpha_bar[prev_timestep] - alpha_bar[timestep]) ** 0.5)
                        
    if is_ddim_inversion:
        omega_timestep_prev_timestep = 0
                        
    new_coeff = (1 - alpha_bar[prev_timestep] - omega_timestep_prev_timestep ** 2) ** 0.5
    
    return new_coeff
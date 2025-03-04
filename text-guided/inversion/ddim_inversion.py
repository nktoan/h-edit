
from inversion.inversion_utils import encode_text
from typing import Union
import torch
import numpy as np
from tqdm import tqdm

def next_step(model, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
    """
    Compute the next sample in the inversion process. For details, see the DDIM inversion formula in our paper.

    Parameters:
      model         : Model with a scheduler providing diffusion parameters.
      model_output  : Predicted noise output.
      timestep      : Current timestep.
      sample        : Current sample.
    
    Returns:
      The updated sample.
    """
        
    timestep, next_timestep = min(timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_prod_t_next = model.scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample

def get_noise_pred(model, latent: Union[torch.FloatTensor, np.ndarray], t: int, context: Union[torch.FloatTensor, np.ndarray], cfg_scale: float):
    """
    Compute the predicted noise for a given latent state using classifier-free guidance (CFG).

    Parameters:
        model (object): Diffusion model with a U-Net denoiser.
        latent (torch.Tensor): Input latent representation.
        t (int): Current diffusion timestep.
        context (torch.Tensor): Conditioning information (e.g., text embeddings).
        cfg_scale (float): Classifier-free guidance scale.

    Returns:
        torch.Tensor: Predicted noise after applying CFG.
    """
        
    latents_input = torch.cat([latent] * 2) # Duplicate for conditional & unconditional paths

    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2) # Split into unconditional & conditional predictions

    noise_pred = noise_pred_uncond + cfg_scale * (noise_prediction_text - noise_pred_uncond)
    return noise_pred

@torch.no_grad()
def ddim_inversion(model, w0: Union[torch.FloatTensor, np.ndarray], prompt: str, cfg_scale: float):
    """
    Perform the DDIM inversion process to construct latent variables.

    Parameters:
        model (object): Diffusion model with a U-Net denoiser and a scheduler.
        w0 (torch.Tensor): Initial latent representation.
        prompt (str): Text prompt guiding the inversion process.
        cfg_scale (float): Classifier-free guidance scale.

    Returns:
        latent (torch.Tensor): Final latent representation after inversion.
        zs (torch.Tensor): Collection of noise corrections at each step.
        latents (List[torch.Tensor]): Sequence of latent states during the process.
    """
    
    # Encode text and unconditional embeddings
    text_embedding = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")
    context = torch.cat([uncond_embedding, text_embedding])

    # Initialize latents
    latent = w0.clone().detach()
    latents = []
    latents.append(latent) 

    # Forward diffusion through inference steps
    for i in tqdm(range(model.scheduler.num_inference_steps)):
        t = model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1] #t=0, 20, 40
        noise_pred = get_noise_pred(model, latent, t, context, cfg_scale) #1. e(x0, 0); 2. e(x0, 20); 3. e(x20, t=40)
        latent = next_step(model, noise_pred, t, latent) #x0, x20, x40
        latents.append(latent) #[x[0], x[0], x[20], x[40]]

    # Initialize zs array
    variance_noise_shape = (
        model.scheduler.num_inference_steps,
        model.unet.in_channels, 
        model.unet.sample_size,
        model.unet.sample_size)

    zs = torch.zeros(size=variance_noise_shape, device=model.device)
    
    # Mapping timesteps to indices
    t_to_idx = {int(v):k for k,v in enumerate(model.scheduler.timesteps)}
    op = tqdm(model.scheduler.timesteps)

    # Reverse process of DDIM inversion to "compute zs" (u_{t}^{orig} in our paper!)
    for t in op:
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-1
        xt = latents[idx+1]

        with torch.no_grad():
            noise_pred = get_noise_pred(model, xt, t, context, cfg_scale)

        xtm1 =  latents[idx]

        # Compute predicted original sample
        alpha_bar = model.scheduler.alphas_cumprod
        pred_original_sample = (xt - (1-alpha_bar[t])  ** 0.5 * noise_pred) / (alpha_bar[t] ** 0.5)

        # Compute next timestep parameters
        prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod

        # Compute predicted sample direction
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * noise_pred
        mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        # Compute noise correction
        z = (xtm1 - mu_xt)
        zs[idx] = z

        # Apply correction to avoid error accumulation
        xtm1 = mu_xt + z
        latents[idx] = xtm1

    return latent, zs, latents
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import autocast

from inversion.inversion_utils import encode_text, reverse_step, reverse_step_pred_x0

"""
Method. EF with P2P
"""

def ef_p2p(model, image_encoder, xT,  etas = 1.0, 
           prompts = "", cfg_scales = None, prog_bar = False, zs = None, controller=None, 
           weight_edit_clip = 1.5, is_ddim_inversion = False):
    
    """
    The implementation of EF editing method combined with P2P for Combined Text-guided and Style Editing.

    Parameters:
      model         : Model with a scheduler providing diffusion parameters.
      image_encoder : CLIP Model for style transfer.
      xT            : The last sample to start perform editing
      etas          : eta should be 1 for EF and PnP Inv (as we explain in Step 2.2.1 below)
      prompts       : source and target prompts
      cfg_scales    : classifier-free guidance strengths
      prog_bar      : whether to show prog_bar
      zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
      controller    : Attention Controller: Refine, Replace, etc. It depends on editing types (see P2P paper for more details)
      weight_edit_clip: used for editing the style of CLIP Gram Matrix Loss.
      is_ddim_inversion: can be False for EF, and True for PnP Inv
    
    Returns:
      The edited sample, the reconstructed sample

    """    
    # 1. Define coefficients, embeddings, etas, etc
    assert len(prompts) >= 2,  "for prompt-to-prompt, requires both source and target prompts"

    batch_size = len(prompts)
    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1,1,1,1).to(model.device)

    text_embeddings = encode_text(model, prompts)
    uncond_embedding = encode_text(model, [""] * batch_size)

    if etas is None: etas = 0
    if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    
    xt.requires_grad = True
    model.requires_grad = True
    
    op = list(timesteps[-zs.shape[0]:]) if prog_bar else list(timesteps[-zs.shape[0]:])

    t_to_idx = {int(v):k for k,v in enumerate(timesteps[-zs.shape[0]:])}

    src_prompt_embed, tar_prompt_embed = text_embeddings.chunk(2)
    cfg_scales_tensor_src, cfg_scales_tensor_tar = cfg_scales_tensor.chunk(2)
    
    # 2. Perform Editing

    for i, t in enumerate(tqdm(op)):
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-zs.shape[0]+1)    
        xt_input = torch.cat([xt] * 2) #expanding for doing classifier free guidance
        prompt_embeds_input_ = torch.cat([uncond_embedding, text_embeddings])

        with torch.no_grad():
            #2.1. This line will perform P2P
            noise_preds = model.unet(xt_input, t, encoder_hidden_states=prompt_embeds_input_).sample
        
        noise_pred_uncond, noise_pred_text = noise_preds.chunk(2)
        uncond_out_src, uncond_out_tar = noise_pred_uncond.chunk(2)
        cond_out_src, cond_out_tar = noise_pred_text.chunk(2)

        noise_pred_src = uncond_out_src + cfg_scales_tensor_src * (cond_out_src - uncond_out_src)
        noise_pred_tar = uncond_out_tar + cfg_scales_tensor_tar * (cond_out_tar - uncond_out_tar)

        z = zs[idx] if not zs is None else None

        # 2.2 compute less noisy image and set x_t -> x_t-1  
        xt_0_prev = reverse_step(model, noise_pred_src , t, xt[0], eta = etas[idx], variance_noise = z, is_ddim_inversion=is_ddim_inversion)
        if is_ddim_inversion: 
            xt_1_prev = reverse_step(model, noise_pred_tar, t, xt[1], eta = 0, variance_noise = z, is_ddim_inversion=is_ddim_inversion)
        else:
            xt_1_prev = reverse_step(model, noise_pred_tar, t, xt[1], eta = etas[idx], variance_noise = z, is_ddim_inversion=is_ddim_inversion)

        # 2.2 perform style editing

        with torch.enable_grad():
            xt_prev_opt_style = xt[1:].clone().detach().requires_grad_(True)
            
            # 2.2.1. Compute Noise 
            xt_input_tar = torch.cat([xt_prev_opt_style] * 2)  
            no_attention_kwargs = {'use_controller': False}
            
            prompt_embeds_input_ = torch.cat([uncond_embedding[1:], text_embeddings[1:]])
            noise_preds_tar_txt = model.unet(xt_input_tar, t, encoder_hidden_states=prompt_embeds_input_, cross_attention_kwargs = no_attention_kwargs).sample
        
            uncond_out_tar_txt, cond_out_tar_txt = noise_preds_tar_txt.chunk(2)
            noise_pred_tar_txt = uncond_out_tar_txt + cfg_scales_tensor_tar * (cond_out_tar_txt - uncond_out_tar_txt)

            # 2.2.2. Use Tweedie's formula to get prediction of x0

            pred_x0_given_xt = reverse_step_pred_x0(model, noise_pred_tar_txt, t, xt_prev_opt_style)
            correction = cond_out_tar_txt - uncond_out_tar_txt

            with autocast("cuda"):
                x0_dec = model.vae.decode(1 / 0.18215 * pred_x0_given_xt).sample
            
            # 2.2.3. Get Loss and Gradient

            residual = image_encoder.get_gram_matrix_residual(x0_dec)
            loss_from_clip = torch.linalg.norm(residual)

            norm_grad = torch.autograd.grad(outputs=loss_from_clip, inputs=xt_prev_opt_style)[0]
            rho = (correction * correction).mean().sqrt().item() 
            rho = rho / (norm_grad * norm_grad).mean().sqrt().item() * weight_edit_clip

        # 2.2.4. Update sample
        xt_1_prev = xt_1_prev - rho * norm_grad.detach()
            
        xt = torch.cat([xt_0_prev, xt_1_prev])
        
        # 2.3. Perform local blend
        if controller is not None:
            xt = controller.step_callback(xt)

    return xt[1].unsqueeze(0), xt[0].unsqueeze(0)
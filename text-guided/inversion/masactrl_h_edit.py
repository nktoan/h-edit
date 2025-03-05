import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from inversion.inversion_utils import encode_text, reverse_step, compute_full_coeff

"""

A. Methods: h-edit-R and h-edit-D in the IMPLICIT form with MasaCtrl

"""

def h_Edit_masactrl_implicit(model, xT,  eta = 0, prompts = "", cfg_scales = None,
                             prog_bar = False, zs = None,
                             optimization_steps=1, after_skip_steps=35, is_ddim_inversion = True):

    """
    This is the implementation of h-Edit-R and h-edit-D in the IMPLICIT form (see Eq. 25 in our paper) WITH MasaCtrl. 
    Here, the editing term is based on h(x_{t-1},t-1). The main idea is optimizing on x_{t-1} space.

    As we note in the MasaCtrl's documentation, `source prompt` in MasaCtrl will be `null text`.
    For ease, we only provide code with 1 step optimization here. 
    Those who interested in performing MOS with MasaCtrl can try modify the code based on our implementation in p2p_h_edit.py

    Parameters:
        model         : Model with a scheduler providing diffusion parameters.
        xT            : The last sample to start perform editing
        etas          : eta should be 1 for h-Edit-R and h-edit-D (to account for u_t^orig) (`is_ddim_inversion` will handle the rest)
        prompts       : source and target prompts
        cfg_scales    : classifier-free guidance strengths
        prog_bar      : whether to show prog_bar
        zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
        is_ddim_inversion: False for h-edit-R, True for h-edit-D

        optimization_steps: the # of multiple optimization implicit loops, by default set to 1
        
        after_skip_steps: the number of sampling (editing) steps after skipping some initial sampling steps
        (e.g., sampling 50 steps and skip 15 steps, after_skip_steps will be 35). 

    
    Returns:
        The edited sample, the reconstructed sample
    
    """
    # 1. Prepare embeddings, etas, timesteps, classifier-free guidance strengths
    batch_size = len(prompts)
    assert batch_size >= 2, "only support prompt editing"

    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1,1,1,1).to(model.device)
    text_embeddings = encode_text(model, prompts)
    uncond_embeddings = encode_text(model, [""] * batch_size) 

    if eta is None: etas = 0
    if type(eta) in [int, float]: etas = [eta]*model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps

    timesteps = model.scheduler.timesteps.to(model.device)

    if (xT.dim() < 4):
        xt = xT.unsqueeze(0) #do not need expand(batch_size, -1, -1, -1) as (bs=2, 4, 64, 64), expand when necessary
    else:
        xt = xT

    op = list(timesteps[-after_skip_steps:]) if prog_bar else list(timesteps[-after_skip_steps:])
    t_to_idx = {int(v):k for k,v in enumerate(timesteps[-after_skip_steps:])}
    
    # Prepare embeddings, coefficients

    src_prompt_embed, tar_prompt_embed = text_embeddings.chunk(2)
    cfg_src, cfg_src_edit, cfg_tar = cfg_scales_tensor.chunk(3)

    xt = torch.cat([xt] * batch_size)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
    sqrt_alpha_bar = alpha_bar ** 0.5
    
    # The flag to deactivate MasaCtrl, otherwise it is automatically activated
    attn_kwargs_no_attn = {'use_editor': False}

    # 2. Do editing and reconstruction
    for i, t in enumerate(tqdm(op)): 
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-after_skip_steps+1)
        z = zs[idx] if not zs is None else None

        # 2.1. Compute x_{t-1}^orig and x_{t-1}^base first!!!
        with torch.no_grad():
            xt_input = torch.cat([xt] * 2)
            prompt_embeds_input_ = torch.cat([uncond_embeddings[:1], uncond_embeddings[:1], text_embeddings[:1], text_embeddings[:1]])
            """
            `xt_input` contains [x_t^orig, x_t^edit, x_t^orig, x_t^edit]
            `prompt_embeds_input_` contains [null, null, src_prompt, src_prompt]

            We compute \eps(x_t^orig, t, null), \eps(x_t^edit, t, null), \eps(x_t^orig, t, src_prompt) and \eps(x_t^edit, t, src_prompt) 
            """

            # Do not use MasaCtrl here
            noise_preds = model.unet(xt_input, t, encoder_hidden_states=prompt_embeds_input_,cross_attention_kwargs = attn_kwargs_no_attn).sample
            
            uncond_out_src, cond_out_src = noise_preds.chunk(2)

            noise_pred_src_orig = uncond_out_src + cfg_src * (cond_out_src - uncond_out_src)

        # Compute x_{t-1}^orig and x_{t-1}^base
        xt_prev_initial = reverse_step(model, noise_pred_src_orig, t, xt, eta = etas[idx], 
                                       variance_noise=z, is_ddim_inversion=is_ddim_inversion)
        
        xt_prev_initial_src, xt_prev_initial_tar = xt_prev_initial.chunk(2)

        # Take the previous timestep tt of t
        
        if i < len(op) - 1:
            tt = op[i+1]
        else:
            tt = 0 

        xt_prev_opt = xt_prev_initial_tar.clone().detach().requires_grad_(True)

        # 2.2. Computing the editing term
        with torch.no_grad():
            for opt_step in range(optimization_steps): #fix = 1 optimization steps for ease first!

                # Compute \eps(x_{t-1},t-1,c^src)
                cond_out_src = model.unet(xt_prev_opt, tt, encoder_hidden_states=text_embeddings[:1],cross_attention_kwargs = attn_kwargs_no_attn).sample

                # Computing editing term, will apply MasaCtrl here!!!
                xt_prev_opt_input = torch.cat([xt_prev_initial_src, xt_prev_opt] * 2)
                prompt_embeds_input_ = torch.cat([uncond_embeddings, text_embeddings])
                
                # This line applies MasaCtrl
                noise_preds = model.unet(xt_prev_opt_input, tt, encoder_hidden_states=prompt_embeds_input_).sample
                noise_pred_uncond, noise_pred_text = noise_preds.chunk(2)

                _, uncond_out_tar = noise_pred_uncond.chunk(2)
                _, cond_out_tar = noise_pred_text.chunk(2)

                noise_pred_src_orig = uncond_out_tar + cfg_src * (cond_out_src - uncond_out_tar)
                noise_pred_src_edit = uncond_out_tar + cfg_src_edit * (cond_out_src - uncond_out_tar) #cfg_src_edit is hat{w}^orig in our paper
                noise_pred_tar = uncond_out_tar + cfg_tar * (cond_out_tar - uncond_out_tar)

                # Reconstruction term
                rec_term = xt_prev_opt

                # Editing coefficient and term
                ratio_alpha = sqrt_alpha_bar[tt] / sqrt_alpha_bar[t]
                coeff = compute_full_coeff(model, t, tt, etas[idx], is_ddim_inversion) - sqrt_one_minus_alpha_bar[t] * ratio_alpha
                
                edit_term = coeff * (noise_pred_tar - noise_pred_src_edit)

                # Update x_{t-1}^{k+1} = x_{t-1}^k + coeff * f(x_{t-1}^k, t-1) 
                xt_prev_opt = rec_term + edit_term

        xt = torch.cat([xt_prev_initial_src, xt_prev_opt.clone().detach()])

    return xt[1].unsqueeze(0), xt[0].unsqueeze(0) #only return edit version!

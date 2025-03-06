import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import autocast

from inversion.inversion_utils import encode_text, reverse_step, compute_full_coeff, reverse_step_pred_x0

"""
Method. h-Edit-R WITH P2P in Implicit form
"""

def h_Edit_p2p_implicit(model, image_encoder, xT, eta = 1.0,
                    prompts = "", cfg_scales = None, prog_bar = False, zs = None, controller=None,
                    weight_edit_clip=0.55, optimization_steps=1, 
                    after_skip_steps=100, is_ddim_inversion = False):
    
    """
    This is the implementation of h-Edit-R in the IMPLICIT form (see Eq. 25 in our paper) WITH P2P for Combined Text-guided and Style Editing
    Here, the editing term is based on h(x_{t-1},t-1). The main idea is optimizing on x_{t-1} space, offering multiple optimization steps
    P2P is applied when computing h(x_{t-1},t-1) of text-guided editing.

    Parameters:
        model         : Model with a scheduler providing diffusion parameters.
        image_encoder : CLIP Model for style transfer.
        xT            : The last sample to start perform editing
        eta           : Set default to 1.0.
        prompts       : source and target prompts
        cfg_scales    : classifier-free guidance strengths
        prog_bar      : whether to show prog_bar
        zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
        controller    : Attention Controller: Refine, Replace, etc. It depends on editing types (see P2P paper for more details)
        is_ddim_inversion: False for h-edit-R, by default!

        weight_edit_clip: used for editing the style of CLIP Gram Matrix Loss.
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
    
    # The flag to deactivate P2P, otherwise P2P is activated by default
    attn_kwargs_no_attn = {'use_controller': False}

    # 2. Perform Editing Prompts and Style
    for i, t in enumerate(tqdm(op)): 
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-after_skip_steps+1)
        z = zs[idx] if not zs is None else None

        # 2.1. Compute x_{t-1}^orig and x_{t-1}^base first
        with torch.no_grad():
            xt_input = torch.cat([xt] * 2) #for doing CFG
            prompt_embeds_input_ = torch.cat([uncond_embeddings[:1], uncond_embeddings[:1], text_embeddings[:1], text_embeddings[:1]])

            """
            `xt_input` contains [x_t^orig, x_t^edit, x_t^orig, x_t^edit]
            `prompt_embeds_input_` contains [null, null, src_prompt, src_prompt]

            We compute \eps(x_t^orig, t, null), \eps(x_t^edit, t, null), \eps(x_t^orig, t, src_prompt) and \eps(x_t^edit, t, src_prompt) 
            """

            noise_preds = model.unet(xt_input, t, encoder_hidden_states=prompt_embeds_input_,cross_attention_kwargs = attn_kwargs_no_attn).sample
            uncond_out_src, cond_out_src = noise_preds.chunk(2)

            noise_pred_src_orig = uncond_out_src + cfg_src * (cond_out_src - uncond_out_src)

        # Compute x_{t-1}^orig and x_{t-1}^base
        xt_prev_initial = reverse_step(model, noise_pred_src_orig, t, xt, eta = etas[idx], variance_noise=z, is_ddim_inversion=is_ddim_inversion)
        
        xt_prev_initial_src, xt_prev_initial_tar = xt_prev_initial.chunk(2)

        # Take the previous time step tt of t
        if i < len(op) - 1:
            tt = op[i+1]
        else:
            tt = 0 

        # Take x_{t-1}^base as the initilization for optimization of editing
        xt_prev_opt = xt_prev_initial_tar.clone().detach().requires_grad_(True)

        # 2.2. Performing editing via multiple optimization implicit loops
        for opt_step in range(optimization_steps):
            #2.2.1. Text-guided Editing
            with torch.no_grad():
                # Only the last optimization steps will save the attention maps, previous work dont, but still P2P is applied
                if opt_step < (optimization_steps - 1) and optimization_steps > 1:
                    attn_kwargs_no_save_attn = {'save_attn': False}
                else:
                    attn_kwargs_no_save_attn = {'save_attn': True}

                # Compute \eps(x_{t-1},t-1,c^src)
                cond_out_src = model.unet(xt_prev_opt, tt, encoder_hidden_states=text_embeddings[:1],cross_attention_kwargs = attn_kwargs_no_attn).sample

                # Computing editing term for x_{t-1}^k

                xt_prev_opt_input = torch.cat([xt_prev_initial_src, xt_prev_opt] * 2) 
                prompt_embeds_input_ = torch.cat([uncond_embeddings, text_embeddings])
                
                # this line applies P2P by default!!!
                noise_preds = model.unet(xt_prev_opt_input, tt, encoder_hidden_states=prompt_embeds_input_, cross_attention_kwargs=attn_kwargs_no_save_attn).sample
                noise_pred_uncond, noise_pred_text = noise_preds.chunk(2)

                _, uncond_out_tar = noise_pred_uncond.chunk(2)
                _, cond_out_tar = noise_pred_text.chunk(2)

                noise_pred_src_orig = uncond_out_tar + cfg_src * (cond_out_src - uncond_out_tar)
                noise_pred_src_edit = uncond_out_tar + cfg_src_edit * (cond_out_src - uncond_out_tar)
                noise_pred_tar = uncond_out_tar + cfg_tar * (cond_out_tar - uncond_out_tar)

                # Reconstruction term 

                rec_term = xt_prev_opt

                # Editing coefficient and term

                ratio_alpha = sqrt_alpha_bar[tt] / sqrt_alpha_bar[t]
                coeff = compute_full_coeff(model, t, tt, etas[idx], is_ddim_inversion) - sqrt_one_minus_alpha_bar[t] * ratio_alpha

                correction = noise_pred_tar - noise_pred_src_edit
                edit_term = coeff * correction

                xt_prev_opt = rec_term + edit_term #x_{t-1}^edit after text-guided editing
            
            #2.2.2. Style Editing

            with torch.enable_grad():
                xt_prev_opt_style = xt_prev_opt.clone().detach().requires_grad_(True)

                # Use Tweedie's formula to get prediction of x0
                x0_prediction_from_x_t_minus_1 = reverse_step_pred_x0(model, noise_pred_tar, tt, xt_prev_opt_style)
            
                if image_encoder:
                    with autocast("cuda"):
                        x0_dec = model.vae.decode(1 / 0.18215 * x0_prediction_from_x_t_minus_1).sample

                    residual = image_encoder.get_gram_matrix_residual(x0_dec)
                    loss_from_clip = torch.linalg.norm(residual)

                    norm_grad = torch.autograd.grad(outputs=loss_from_clip, inputs=xt_prev_opt_style)[0]
                    rho = (correction * correction).mean().sqrt().item() 
                    rho = rho / (norm_grad * norm_grad).mean().sqrt().item() * weight_edit_clip

                    xt_prev_opt_style = xt_prev_opt_style - rho * norm_grad.detach()

                # Set to `xt_prev_opt_style` to continue a new optimization loop
                xt_prev_opt = xt_prev_opt_style
        
        # Update for the next step, should be [x_{t-1}^orig, x_{t-1}^edit]
        xt = torch.cat([xt_prev_initial_src, xt_prev_opt.clone().detach()])

        # Do local blend here for P2P
        if controller is not None:
            xt = controller.step_callback(xt)    

    return xt[1].unsqueeze(0), xt[0].unsqueeze(0) #only return edit version!


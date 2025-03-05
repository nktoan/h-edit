import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim.adam import Adam

from inversion.inversion_utils import encode_text, reverse_step, compute_full_coeff

"""

A. Methods: h-Edit-R without P2P
A.1. Explicit form
A.2. Implicit form

"""

"""
A.1. h-Edit-R without P2P in Explicit form
"""

def h_Edit_R_explicit(model, xT,  eta = 1.0, prompts = "", cfg_scales = None, 
                      prog_bar = False, zs = None, controller=None,
                      
                      after_skip_steps=35, is_ddim_inversion = False):
    """
    This is the implementation of h-Edit-R in the EXPLICIT (see Eqs. 22 and 23 in our paper) form without P2P.
    Here, the editing term is based on h(xt,t)

    Parameters:
        model         : Model with a scheduler providing diffusion parameters.
        xT            : The last sample to start perform editing
        etas          : eta should be 1 for h-Edit-R
        prompts       : source and target prompts
        cfg_scales    : classifier-free guidance strengths
        prog_bar      : whether to show prog_bar
        zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
        controller    : Attention Controller: Here, without P2P, it's only attention store!
        is_ddim_inversion: must be False for h-Edit-R

        after_skip_steps: the number of sampling (editing) steps after skipping some initial sampling steps
        (e.g., sampling 50 steps and skip 15 steps, after_skip_steps will be 35). 

    
    Returns:
        The edited sample, the reconstructed sample
    
    """
    # 1. Prepare embeddings, etas, timesteps, classifier-free guidance strengths
    batch_size = len(prompts)
    assert (batch_size >= 2) and (not is_ddim_inversion), "only support prompt editing and DDPM sampling"

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
    
    no_attn_control_attn_kwargs = {'use_controller': False}

    # 2. Perform editing
    for i, t in enumerate(tqdm(op)): 
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-after_skip_steps+1)
        z = zs[idx] if not zs is None else None #z_t in our paper, used to compute x_{t-1}^base

        # 2.1. Compute x_{t-1}^base, see Eq. 22 in our paper
        with torch.no_grad():
            xt_input = torch.cat([xt[1:]] * 2) #do classifier free guidance
            prompt_embeds_input_ = torch.cat([uncond_embeddings[:1], text_embeddings[:1]])

            """
            `xt_input` contains [x_t_edit, x_t_edit]
            `prompt_embeds_input_` contains [null, src_prompt]

            We compute \eps(x_t^edit, t, null) and \eps(x_t^edit, t, src_prompt)
            """
        
            noise_preds = model.unet(xt_input, t, encoder_hidden_states=prompt_embeds_input_,cross_attention_kwargs = no_attn_control_attn_kwargs).sample
            
            uncond_out_src, cond_out_src = noise_preds.chunk(2)
            noise_pred_src_orig = uncond_out_src + cfg_src * (cond_out_src - uncond_out_src)

        # Compute x_{t-1}^orig and x_{t-1}^base; but note that x_{t-1}^orig IS WRONG, and we do not care about it in h-edit-R
        xt_prev_initial = reverse_step(model, noise_pred_src_orig, t, xt, eta = etas[idx], 
                                       variance_noise=z, is_ddim_inversion=is_ddim_inversion)
        
        xt_prev_initial_src, xt_prev_initial_tar = xt_prev_initial.chunk(2)

        # Take the previous step of t
        if i < len(op) - 1:
            tt = op[i+1]
        else:
            tt = 0 
        
        # Take x_{t-1}^base as xt_prev_opt, but we actually do not optimize it like implicit form
        xt_prev_opt = xt_prev_initial_tar.clone().detach().requires_grad_(False)

        # Compute noise based on x_t^edit, see Eqs 22 and 23 in our paper
        xt_prev_opt_input = torch.cat([xt[1:]] * 4)
        prompt_embeds_input_ = torch.cat([uncond_embeddings, text_embeddings])
                    
        with torch.no_grad():
            noise_preds = model.unet(xt_prev_opt_input, t, encoder_hidden_states=prompt_embeds_input_,cross_attention_kwargs = no_attn_control_attn_kwargs).sample
        
        noise_pred_uncond, noise_pred_text = noise_preds.chunk(2)

        uncond_out_src, uncond_out_tar = noise_pred_uncond.chunk(2)
        cond_out_src, cond_out_tar = noise_pred_text.chunk(2)

        noise_pred_src_orig = uncond_out_src + cfg_src * (cond_out_src - uncond_out_src)
        noise_pred_src_edit = uncond_out_src + cfg_src_edit * (cond_out_src - uncond_out_src) #cfg_src_edit is hat{w}^orig in our paper
        noise_pred_tar = uncond_out_tar + cfg_tar * (cond_out_tar - uncond_out_tar)

        # this is f(x_t^edit, t)
        correction = (noise_pred_tar - noise_pred_src_edit)

        # Reconstruction term will be x_{t-1}^base
        rec_term = xt_prev_opt

        # Editing coefficient and term
        ratio_alpha = sqrt_alpha_bar[tt] / sqrt_alpha_bar[t]
        coeff = compute_full_coeff(model, t, tt, etas[idx], is_ddim_inversion) - sqrt_one_minus_alpha_bar[t] * ratio_alpha

        edit_term = coeff * correction
        
        # Update x_{t-1}^edit = x_{t-1}^base + coeff * f(x_t^edit, t)
        xt_prev_opt = rec_term + edit_term

        # Update for the next step, should be [x_{t-1}^orig, x_{t-1}^edit]
        xt = torch.cat([xt_prev_initial_src, xt_prev_opt.clone().detach()])

        # Only attention store here!
        if controller is not None:
            xt = controller.step_callback(xt)    

    return xt[1].unsqueeze(0), xt[0].unsqueeze(0)

"""
A.2. h-Edit-R without P2P in Implicit form
"""

def h_Edit_R_implicit(model, xT,  eta = 1.0, prompts = "", cfg_scales = None, 
                      prog_bar = False, zs = None, controller=None,

                      weight_reconstruction=0.1, optimization_steps=1, after_skip_steps=35, 
                      is_ddim_inversion = False):
    """
    This is the implementation of h-Edit-R in the IMPLICIT (see Eq. 25 in our paper) form without P2P. 
    Here, the editing term is based on h(x_{t-1},t-1). The main idea is optimizing on x_{t-1} space, offering multiple optimization steps

    Parameters:
        model         : Model with a scheduler providing diffusion parameters.
        xT            : The last sample to start perform editing
        etas          : eta should be 1 for h-Edit-R
        prompts       : source and target prompts
        cfg_scales    : classifier-free guidance strengths
        prog_bar      : whether to show prog_bar
        zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
        controller    : Attention Controller: Here, without P2P, it's only attention store!
        is_ddim_inversion: must be False for h-Edit-R

        weight_reconstruction: used for reconstruction when performing multiple optimization implicit loops, by default set to 0.1
        
        optimization_steps: the # of multiple optimization implicit loops, by default set to 1
        
        after_skip_steps: the number of sampling (editing) steps after skipping some initial sampling steps
        (e.g., sampling 50 steps and skip 15 steps, after_skip_steps will be 35). 

    
    Returns:
        The edited sample, the reconstructed sample
    
    """
    # 1. Prepare embeddings, etas, timesteps, classifier-free guidance strengths
    batch_size = len(prompts)
    assert (batch_size >= 2) and (not is_ddim_inversion), "only support prompt editing and DDPM sampling"

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
    
    # Take the time immediately before the step following the skip to optimize that step as well.
    if after_skip_steps != model.scheduler.num_inference_steps:
        time_ahead = timesteps[-(after_skip_steps+1)]
    else:
        time_ahead = -1

    # Prepare embeddings, coefficients
    src_prompt_embed, tar_prompt_embed = text_embeddings.chunk(2)
    cfg_src, cfg_src_edit, cfg_tar = cfg_scales_tensor.chunk(3)

    xt = torch.cat([xt] * batch_size)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
    sqrt_alpha_bar = alpha_bar ** 0.5
    
    no_attn_control_attn_kwargs = {'use_controller': False}
    
    # 2. Perform editing
    for i, t in enumerate(tqdm(op)): 
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-after_skip_steps+1)
        z = zs[idx] if not zs is None else None
        
        # 2.1. Optimizing editing at the last step after skipping
        if i == 0 and time_ahead != -1:
            #only 1 step optimization for optimizing this sample (xT or x at the end following the skip)
            with torch.no_grad():
                xt_prev_opt_input = torch.cat([xt[1:]] * 4)
                prompt_embeds_input_ = torch.cat([uncond_embeddings, text_embeddings])
                        
                noise_preds = model.unet(xt_prev_opt_input, t, encoder_hidden_states=prompt_embeds_input_, cross_attention_kwargs = no_attn_control_attn_kwargs).sample
                noise_pred_uncond, noise_pred_text = noise_preds.chunk(2)

                uncond_out_src, uncond_out_tar = noise_pred_uncond.chunk(2)
                cond_out_src, cond_out_tar = noise_pred_text.chunk(2)

                # Compute the noise using CFG
                noise_pred_src_orig = uncond_out_src + cfg_src * (cond_out_src - uncond_out_src)
                noise_pred_src_edit = uncond_out_src + cfg_src_edit * (cond_out_src - uncond_out_src) #cfg_src_edit is hat{w}^orig in our paper
                noise_pred_tar = uncond_out_tar + cfg_tar * (cond_out_tar - uncond_out_tar)

                # Reconsturction term
                rec_term = xt[1] 

                # Editing coefficient and term

                ratio_alpha = sqrt_alpha_bar[t] / sqrt_alpha_bar[time_ahead]
                coeff = compute_full_coeff(model, time_ahead, t, etas[idx], is_ddim_inversion) - sqrt_one_minus_alpha_bar[time_ahead] * ratio_alpha

                edit_term = coeff * (noise_pred_tar - noise_pred_src_edit)

                # Update xt[1], or x_T^edit here!
                xt[1] = rec_term + edit_term

        # 2.2. Compute x_{t-1}^base, see Eq. 22 in our paper
        with torch.no_grad():
            xt_input = torch.cat([xt[1:]] * 2)
            prompt_embeds_input_ = torch.cat([uncond_embeddings[:1], text_embeddings[:1]])

            """
            `xt_input` contains [x_t_edit, x_t_edit]
            `prompt_embeds_input_` contains [null, src_prompt]

            We compute \eps(x_t^edit, t, null) and \eps(x_t^edit, t, src_prompt)
            """

            noise_preds = model.unet(xt_input, t, encoder_hidden_states=prompt_embeds_input_,cross_attention_kwargs = no_attn_control_attn_kwargs).sample
            
            uncond_out_src, cond_out_src = noise_preds.chunk(2)
            noise_pred_src_orig = uncond_out_src + cfg_src * (cond_out_src - uncond_out_src)

        # Compute x_{t-1}^orig and x_{t-1}^base; but note that x_{t-1}^orig IS WRONG, and we do not care about it in h-edit-R
        xt_prev_initial = reverse_step(model, noise_pred_src_orig, t, xt, eta = etas[idx], 
                                       variance_noise=z, is_ddim_inversion=is_ddim_inversion)
        
        xt_prev_initial_src, xt_prev_initial_tar = xt_prev_initial.chunk(2)
        
        # Take the previous step of t
        if i < len(op) - 1:
            tt = op[i+1]
        else:
            tt = 0 

        # We optimize x_{t-1}^base as the initial initialization. See Equation 25 in our paper for details.        
        xt_prev_opt = xt_prev_initial_tar.clone().detach().requires_grad_(True)

        # 2.3. Optimize for editing
        for opt_step in range(optimization_steps):
            # Compute the noise, see Eq. 24 in our paper
            xt_prev_opt_input = torch.cat([xt_prev_opt] * 4)
            prompt_embeds_input_ = torch.cat([uncond_embeddings, text_embeddings])

            """
            `xt_prev_opt_input` contains [xt_prev_opt, xt_prev_opt, xt_prev_opt, xt_prev_opt]
            `prompt_embeds_input_` contains [null, null, src_prompt, tar_prompt]

            We compute \eps(x_t^edit, tt, null), \eps(x_t^edit, tt, src_prompt), and \eps(x_t^edit, tt, tar_prompt) 
            """

            with torch.no_grad():
                noise_preds = model.unet(xt_prev_opt_input, tt, encoder_hidden_states=prompt_embeds_input_,cross_attention_kwargs = no_attn_control_attn_kwargs).sample
            
            noise_pred_uncond, noise_pred_text = noise_preds.chunk(2)

            uncond_out_src, uncond_out_tar = noise_pred_uncond.chunk(2)
            cond_out_src, cond_out_tar = noise_pred_text.chunk(2)

            noise_pred_src_orig = uncond_out_src + cfg_src * (cond_out_src - uncond_out_src)
            noise_pred_src_edit = uncond_out_src + cfg_src_edit * (cond_out_src - uncond_out_src) #cfg_src_edit is hat{w}^orig in our paper
            noise_pred_tar = uncond_out_tar + cfg_tar * (cond_out_tar - uncond_out_tar)

            correction = (noise_pred_tar - noise_pred_src_edit) #f(x_{t-1},t-1) in our paper
        
            if opt_step > 0:
                # Perform reconstruction, as mentioned in Section 3.3.3 in our paper
                with torch.enable_grad():
                    rec_loss = F.l1_loss(xt_prev_opt, xt_prev_initial_tar)
                    
                    # Compute rec_loss gradients
                    grad = torch.autograd.grad(outputs=rec_loss, inputs=xt_prev_opt, create_graph=True)[0]

                    correction_norm = (correction * correction).mean().sqrt().item()
                    grad_norm = (grad * grad).mean().sqrt().item()
                
                    epsilon = 1e-8
                    rho = correction_norm / (grad_norm + epsilon) * weight_reconstruction

                rec_term = xt_prev_opt - rho * grad.detach()
            else:
                rec_term = xt_prev_opt
                
            # Editing coefficient and term
            ratio_alpha = sqrt_alpha_bar[tt] / sqrt_alpha_bar[t]
            coeff = compute_full_coeff(model, t, tt, etas[idx], is_ddim_inversion) - sqrt_one_minus_alpha_bar[t] * ratio_alpha
             
            edit_term = coeff * correction

            # Update x_{t-1}^{k+1} = x_{t-1}^k + coeff * f(x_{t-1}^k, t-1) 
            xt_prev_opt = rec_term + edit_term

        # Update for the next step, should be [x_{t-1}^orig, x_{t-1}^edit]
        xt = torch.cat([xt_prev_initial_src, xt_prev_opt.clone().detach()])

        # Only attention store here!
        if controller is not None:
            xt = controller.step_callback(xt)    

    return xt[1].unsqueeze(0), xt[0].unsqueeze(0)

###################################################################################################
###################################################################################################
###################################################################################################

"""

B. Methods: h-Edit-D and h-Edit-R WITH P2P
B.1. Explicit form
B.2. Implicit form with multiple optimization implicit loops (key features for hard editing cases)

"""

"""
B.1. h-Edit-D and h-Edit-R WITH P2P in Explicit form
"""

def h_Edit_p2p_explicit(model, xT, eta = 0, prompts = "", cfg_scales = None,
                        prog_bar = False, zs = None, controller=None,

                        is_ddim_inversion = True, after_skip_steps=35):
    """
    This is the implementation of h-Edit-D and h-edit-R combined with P2P in the EXPLICIT form (see Eqs 22 and 23 in our paper). 
    Here, the editing term is based on h(xt,t), P2P is applied when computing h(xt,t)

    Parameters:
        model         : Model with a scheduler providing diffusion parameters.
        xT            : The last sample to start perform editing
        etas          : eta should be 1 even for h-edit-R or h-edit-D to account for u_t^orig when computing x_{t-1}^base and x_{t-1}^orig
        prompts       : source and target prompts
        cfg_scales    : classifier-free guidance strengths
        prog_bar      : whether to show prog_bar
        zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
        controller    : Attention Controller: Refine, Replace, etc. It depends on editing types (see P2P paper for more details)
        is_ddim_inversion: True for h-edit-D and False for h-edit-R

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
    
    # 2. Perform Editing 
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
        xt_prev_initial = reverse_step(model, noise_pred_src_orig, t, xt, eta = etas[idx], 
                                       variance_noise=z, is_ddim_inversion=is_ddim_inversion)
        
        xt_prev_initial_src, xt_prev_initial_tar = xt_prev_initial.chunk(2)

        # Take the previous time step tt of t
        if i < len(op) - 1:
            tt = op[i+1]
        else:
            tt = 0 
        
        # 2.2. Compute the editing term

        # Save x_{t-1}^base, but we do not optimize any thing in the explicit form
        xt_prev_opt = xt_prev_initial_tar.clone().detach().requires_grad_(False)

        attn_kwargs_save_attn = {'save_attn': True}
        # Compute \eps(x_t^edit,t, src_prompt), like above

        with torch.no_grad():
            cond_out_src = model.unet(xt[1:], t, encoder_hidden_states=text_embeddings[:1],cross_attention_kwargs = attn_kwargs_no_attn).sample

        # Compute the editing term
        xt_input = torch.cat([xt] * 2) #do CFG, prepare for P2P as well
        prompt_embeds_input_ = torch.cat([uncond_embeddings, text_embeddings])
            
        with torch.no_grad():
            # This line performs P2P
            noise_preds = model.unet(xt_input, t, encoder_hidden_states=prompt_embeds_input_, cross_attention_kwargs=attn_kwargs_save_attn).sample
            
        noise_pred_uncond, noise_pred_text = noise_preds.chunk(2)
        _, uncond_out_tar = noise_pred_uncond.chunk(2)
        _, cond_out_tar = noise_pred_text.chunk(2)

        noise_pred_src_orig = uncond_out_tar + cfg_src * (cond_out_src - uncond_out_tar)
        noise_pred_src_edit = uncond_out_tar + cfg_src_edit * (cond_out_src - uncond_out_tar) #cfg_src_edit is hat{w}^orig in our paper
        noise_pred_tar = uncond_out_tar + cfg_tar * (cond_out_tar - uncond_out_tar)

        correction = (noise_pred_tar - noise_pred_src_edit) # this is f(xt,t) in our paper, see Eq. 24

        # Reconstruction term is x_{t-1}^base
        rec_term = xt_prev_opt

        # Editing coefficient and term
        ratio_alpha = sqrt_alpha_bar[tt] / sqrt_alpha_bar[t]
        coeff = compute_full_coeff(model, t, tt, etas[idx], is_ddim_inversion) - sqrt_one_minus_alpha_bar[t] * ratio_alpha
               
        edit_term = coeff * correction

        # Update x_{t-1}^edit = x_{t-1}^base + coeff * f(x_t^edit,t)
        xt_prev_opt = rec_term + edit_term

        # Update for the next step, should be [x_{t-1}^orig, x_{t-1}^edit]
        xt = torch.cat([xt_prev_initial_src, xt_prev_opt.clone().detach()])

        # Perform local blend in P2P
        if controller is not None:
            xt = controller.step_callback(xt)    

    return xt[1].unsqueeze(0), xt[0].unsqueeze(0) 

"""
B.2. h-Edit-D and h-Edit-R WITH P2P in Implicit form
"""

def h_Edit_p2p_implicit(model, xT, eta = 0, prompts = "", cfg_scales = None,
                        prog_bar = False, zs = None, controller=None,
        
                        weight_reconstruction=0.075, optimization_steps=1, after_skip_steps=35,
                        is_ddim_inversion = True):

    """
    This is the implementation of h-Edit-R and h-edit-D in the IMPLICIT form (see Eq. 25 in our paper) WITH P2P. 
    Here, the editing term is based on h(x_{t-1},t-1). The main idea is optimizing on x_{t-1} space, offering multiple optimization steps
    P2P is applied when computing h(x_{t-1},t-1).

    Parameters:
        model         : Model with a scheduler providing diffusion parameters.
        xT            : The last sample to start perform editing
        etas          : eta should be 1 for h-Edit-R
        prompts       : source and target prompts
        cfg_scales    : classifier-free guidance strengths
        prog_bar      : whether to show prog_bar
        zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
        controller    : Attention Controller: Refine, Replace, etc. It depends on editing types (see P2P paper for more details)
        is_ddim_inversion: False for h-edit-R, True for h-edit-D

        weight_reconstruction: used for reconstruction when performing multiple optimization implicit loops, by default set to 0.1
        
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
    
    # The flag to deactivate P2P, otherwise it is automatically activated
    attn_kwargs_no_attn = {'use_controller': False}

    # 2. Do editing and reconstruction
    for i, t in enumerate(tqdm(op)): 
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-after_skip_steps+1)
        z = zs[idx] if not zs is None else None

        # 2.1. Compute x_{t-1}^orig and x_{t-1}^base first!!!
        with torch.no_grad():
            xt_input = torch.cat([xt] * 2) #expand to perform CFG, not P2P, do not use P2P here!!!
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
        xt_prev_initial = reverse_step(model, noise_pred_src_orig, t, xt, eta = etas[idx], 
                                       variance_noise=z, is_ddim_inversion=is_ddim_inversion)
        
        xt_prev_initial_src, xt_prev_initial_tar = xt_prev_initial.chunk(2)

        # Take the previous timestep tt of t

        if i < len(op) - 1:
            tt = op[i+1]
        else:
            tt = 0 

        # Take x_{t-1}^base as the initilization for optimization of editing
        xt_prev_opt = xt_prev_initial_tar.clone().detach().requires_grad_(True)

        # 2.2. Performing editing via multiple optimization implicit loops
        for opt_step in range(optimization_steps):
            # Only the last optimization steps will save the attention maps, previous work dont, but still P2P is applied
            if opt_step < (optimization_steps - 1) and optimization_steps > 1:
                attn_kwargs_no_save_attn = {'save_attn': False}
            else:
                attn_kwargs_no_save_attn = {'save_attn': True}

            # Compute source for \eps(x_{t-1},t-1,c^src)
            with torch.no_grad():
                cond_out_src = model.unet(xt_prev_opt, tt, encoder_hidden_states=text_embeddings[:1],cross_attention_kwargs = attn_kwargs_no_attn).sample

            # Computing editing term for x_{t-1}^k
            xt_prev_opt_input = torch.cat([xt_prev_initial_src, xt_prev_opt] * 2) #xt_prev_initial_src is x_{t-1}^orig, requires for P2P
            prompt_embeds_input_ = torch.cat([uncond_embeddings, text_embeddings])
            
            with torch.no_grad():
                #this line applies P2P by default!!!
                noise_preds = model.unet(xt_prev_opt_input, tt, encoder_hidden_states=prompt_embeds_input_, cross_attention_kwargs=attn_kwargs_no_save_attn).sample
            
            noise_pred_uncond, noise_pred_text = noise_preds.chunk(2)
            _, uncond_out_tar = noise_pred_uncond.chunk(2)
            _, cond_out_tar = noise_pred_text.chunk(2)

            noise_pred_src_orig = uncond_out_tar + cfg_src * (cond_out_src - uncond_out_tar)
            noise_pred_src_edit = uncond_out_tar + cfg_src_edit * (cond_out_src - uncond_out_tar) #cfg_src_edit is hat{w}^orig in our paper
            noise_pred_tar = uncond_out_tar + cfg_tar * (cond_out_tar - uncond_out_tar)

            # Editing coefficient and term

            ratio_alpha = sqrt_alpha_bar[tt] / sqrt_alpha_bar[t]
            coeff = compute_full_coeff(model, t, tt, etas[idx], is_ddim_inversion) - sqrt_one_minus_alpha_bar[t] * ratio_alpha
                
            correction = (noise_pred_tar - noise_pred_src_edit)

            # Reconstruction term
            if opt_step > 0:
                with torch.enable_grad():
                    # Perform reconstruction, as mentioned in Section 3.3.3 in our paper
                    rec_loss = F.l1_loss(xt_prev_opt, xt_prev_initial_tar)
                    
                    # Compute rec_loss gradidents
                    grad = torch.autograd.grad(outputs=rec_loss, inputs=xt_prev_opt, create_graph=True)[0]

                    correction_norm = (correction * correction).mean().sqrt().item()
                    grad_norm = (grad * grad).mean().sqrt().item()
                    
                    epsilon = 1e-8
                    rho = correction_norm / (grad_norm + epsilon) * weight_reconstruction

                rec_term = xt_prev_opt - rho * grad.detach() #+ #+ gradient like  coeff * noise_pred_src_orig # - ratio_alpha * z_implicit
            else:
                rec_term = xt_prev_opt
                
            # Editing term
            edit_term = coeff * correction

            # Update x_{t-1}^{k+1} = x_{t-1}^k + coeff * f(x_{t-1}^k, t-1) 
            xt_prev_opt = rec_term + edit_term

        # Update for the next step, should be [x_{t-1}^orig, x_{t-1}^edit]
        xt = torch.cat([xt_prev_initial_src, xt_prev_opt.clone().detach()])

        # Do local blend here for P2P
        if controller is not None:
            xt = controller.step_callback(xt)    

    return xt[1].unsqueeze(0), xt[0].unsqueeze(0) #only return edit version!



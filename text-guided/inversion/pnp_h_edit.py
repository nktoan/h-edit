import os
from tqdm import tqdm

import torch
import torch.nn.functional as F

from inversion.inversion_utils import encode_text, reverse_step, compute_full_coeff


def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)


"""

A. Method: h-edit-R and h-edit-D in the IMPLICIT form with PnP attn control

"""

def h_Edit_PnP_implicit(model, xT, eta = 0, prompts = "", cfg_scales = None,
                        prog_bar = False, zs = None,
                        optimization_steps=1, after_skip_steps=35, is_ddim_inversion = True):
    """
    This is the implementation of h-Edit-R and h-edit-D in the IMPLICIT form (see Eq. 25 in our paper) WITH Plug-n-Play (PnP) Attn Control. 
    Here, the editing term is based on h(x_{t-1},t-1). The main idea is optimizing on x_{t-1} space.
    PnP is applied when computing h(x_{t-1},t-1).

    For ease, we only provide code with 1 step optimization here. 
    Those who interested in performing MOS with PnP can try modify the code based on our implementation in p2p_h_edit.py

    Parameters:
        model         : Model with a scheduler providing diffusion parameters.
        xT            : The last sample to start perform editing
        etas          : eta should be 1 for h-Edit-R and h-edit-D (to account for u_t^orig)
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

    src_prompt_embed, tar_prompt_embed = text_embeddings.chunk(2)
    cfg_src, cfg_src_edit, cfg_tar = cfg_scales_tensor.chunk(3)

    xt = torch.cat([xt] * batch_size)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
    sqrt_alpha_bar = alpha_bar ** 0.5
    
    # 2. Perform Editing
    for i, t in enumerate(tqdm(op)): 
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-after_skip_steps+1)
        z = zs[idx] if not zs is None else None

        # 2.1. Compute x_{t-1}^orig and x_{t-1}^base first!!!
        with torch.no_grad():
            register_time(model, t.item())

            xt_input = torch.cat([xt] * 2)
            prompt_embeds_input_ = torch.cat([uncond_embeddings[:1], uncond_embeddings[:1], text_embeddings[:1], text_embeddings[:1]])
            """
            `xt_input` contains [x_t^orig, x_t^edit, x_t^orig, x_t^edit]
            `prompt_embeds_input_` contains [null, null, src_prompt, src_prompt]

            We compute \eps(x_t^orig, t, null), \eps(x_t^edit, t, null), \eps(x_t^orig, t, src_prompt) and \eps(x_t^edit, t, src_prompt) 
            """

            noise_preds = model.unet(xt_input, t, encoder_hidden_states=prompt_embeds_input_).sample    
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
            tt = torch.tensor(0) 

        # Take x_{t-1}^base as the initilization for optimization of editing
        xt_prev_opt = xt_prev_initial_tar.clone().detach().requires_grad_(True)

        # Compute the editing term
        with torch.no_grad():
            for opt_step in range(optimization_steps): #fix = 1 optimization steps for ease first!
                register_time(model, tt.item())

                # Compute \eps(x_{t-1},t-1,c^src), \eps(x_{t-1},t-1, null)
                cond_out_src = model.unet(xt_prev_opt, tt, encoder_hidden_states=text_embeddings[:1]).sample
                uncond_out_tar = model.unet(xt_prev_opt, tt, encoder_hidden_states=uncond_embeddings[1:]).sample

                xt_prev_opt_input = torch.cat([xt_prev_initial_src, xt_prev_opt]) #only two samples for PnP
                prompt_embeds_input_ = text_embeddings
                
                # This line performs PnP attn control
                noise_preds = model.unet(xt_prev_opt_input, tt, encoder_hidden_states=prompt_embeds_input_).sample
                _, cond_out_tar = noise_preds.chunk(2)

                noise_pred_src_orig = uncond_out_tar + cfg_src * (cond_out_src - uncond_out_tar)
                noise_pred_src_edit = uncond_out_tar + cfg_src_edit * (cond_out_src - uncond_out_tar)
                noise_pred_tar = uncond_out_tar + cfg_tar * (cond_out_tar - uncond_out_tar)

                # Reconstruction term
                rec_term = xt_prev_opt 

                # Editing coefficient and term (see Eq. 25 in our paper for details!)
                ratio_alpha = sqrt_alpha_bar[tt] / sqrt_alpha_bar[t]
                coeff = compute_full_coeff(model, t, tt, etas[idx], is_ddim_inversion) - sqrt_one_minus_alpha_bar[t] * ratio_alpha

                edit_term = coeff * (noise_pred_tar - noise_pred_src_edit)

                # Update x_{t-1}^{k+1} = x_{t-1}^k + coeff * f(x_{t-1}^k, t-1) 
                xt_prev_opt = rec_term + edit_term

        # Update for the next step, should be [x_{t-1}^orig, x_{t-1}^edit]
        xt = torch.cat([xt_prev_initial_src, xt_prev_opt.clone().detach()])

    return xt[1].unsqueeze(0), xt[0].unsqueeze(0)

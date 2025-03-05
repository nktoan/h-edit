import os
from tqdm import tqdm
import inspect

import torch
import torch.nn.functional as F
from torch import autocast, inference_mode
from diffusers.utils.torch_utils import randn_tensor
from torch.optim.adam import Adam

from inversion.inversion_utils import encode_text, reverse_step

"""

A. Methods: Edit Friendly (EF) WITHOUT P2P

"""

def ef_wo_p2p(model, xT,  etas = 0,
                prompts = "", cfg_scales = None,
                prog_bar = False, zs = None,
                controller=None, is_ddim_inversion = False):
    
    """
    The implementation of EF editing methods. Note that EF wo P2P should use skipping.
    If using 50 sampling steps and skipping 15 steps, we should start from `xts[50 - 15]`.  
    (Note: Since this list contains (N_sampling + 1) elements, so "do not subtract 1" here).
    However, for zs, we have: zs[0], ...., zs[34], since zs has N_sampling elements

    Parameters:
      model         : Model with a scheduler providing diffusion parameters.
      xT            : The last sample to start perform editing
      etas          : eta should be 1 for EF
      prompts       : only the target prompt is required here for EF!
      cfg_scales    : classifier-free guidance strengths
      prog_bar      : whether to show prog_bar
      zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
      controller    : Attention Controller, only save attention maps here!
      is_ddim_inversion: must be False
    
    Returns:
      The edited sample.

    """    

    # 1. Prepare coefficients, embeddings, etas, etc
    batch_size = len(prompts)
    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1,1,1,1).to(model.device)

    text_embeddings = encode_text(model, prompts)
    uncond_embedding = encode_text(model, [""] * batch_size)

    if etas is None: etas = 0
    if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    op = list(timesteps[-zs.shape[0]:]) if prog_bar else list(timesteps[-zs.shape[0]:])

    t_to_idx = {int(v):k for k,v in enumerate(timesteps[-zs.shape[0]:])}

    # 2. Perform editing
    for i, t in enumerate(tqdm(op)):
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-zs.shape[0]+1)    
        attention_kwargs = {'use_controller': False}
        
        ## 2.1. Compute noise with unconditional embedding  
        with torch.no_grad():
            uncond_out = model.unet.forward(xt, timestep =  t, 
                                            encoder_hidden_states = uncond_embedding, cross_attention_kwargs = attention_kwargs)

        ## 2.2. Compute noise with conditional embedding  
        if prompts:  
            with torch.no_grad():
                cond_out = model.unet.forward(xt, timestep =  t, 
                                                encoder_hidden_states = text_embeddings, cross_attention_kwargs = attention_kwargs)
        

        z = zs[idx] if not zs is None else None
        z = z.expand(batch_size, -1, -1, -1)

        ## 2.3. Perform classifier free guidance
        if prompts:
            noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
        else: 
            noise_pred = uncond_out.sample

        # 2.4 compute less noisy image and set x_t -> x_t-1  
        xt = reverse_step(model, noise_pred, t, xt, eta = etas[idx], variance_noise = z, is_ddim_inversion=is_ddim_inversion)
    
    if controller is not None:
        xt = controller.step_callback(xt)  
        
    return xt

"""

B. Methods: PnP Inv and Edit Friendly (EF) WITH P2P

"""

def ef_or_pnp_inv_w_p2p(model, xT,  etas = 0,
                        prompts = "", cfg_scales = None,
                        prog_bar = False, zs = None,
                        controller=None, is_ddim_inversion = False):

    """
    The implementation of EF and PnP Inv editing methods combined with P2P. Note that EF w P2P may not need skipping.
    The etas value should be set to 1.0 by default, even for PnP Inversion, to compute the exact x_{t-1}^orig as accounting for u_t^orig, as proposed in the paper. 
    See Eq. 6 in our paper for more details.

    Parameters:
      model         : Model with a scheduler providing diffusion parameters.
      xT            : The last sample to start perform editing
      etas          : eta should be 1 for EF and PnP Inv (as we explain in Step 2.2.1 below)
      prompts       : source and target prompts
      cfg_scales    : classifier-free guidance strengths
      prog_bar      : whether to show prog_bar
      zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
      controller    : Attention Controller: Refine, Replace, etc. It depends on editing types (see P2P paper for more details)
      is_ddim_inversion: can be False for EF, and True for PnP Inv
    
    Returns:
      The edited sample, the reconstructed sample

    """    
    assert len(prompts) >= 2, "for prompt-to-prompt, requires both source and target prompts"

    # 1. Define coefficients, embeddings, etas, etc
    batch_size = len(prompts)
    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1,1,1,1).to(model.device)

    text_embeddings = encode_text(model, prompts)
    uncond_embedding = encode_text(model, [""] * batch_size)

    if etas is None: etas = 0
    if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
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

        # 2.2.1 the reconstucted x_{t-1}^orig, eta should be 1.0 here to account for u_t^orig
        xt_0 = reverse_step(model, noise_pred_src , t, xt[0], eta = etas[idx], variance_noise = z, is_ddim_inversion=is_ddim_inversion)
        
        # 2.2.2. the edited x_{t-1}^edit, if PnP Inv eta should be 0.0, elif EF eta should be 1.0
        if is_ddim_inversion: 
            xt_1 = reverse_step(model, noise_pred_tar, t, xt[1], eta = 0, variance_noise = z, is_ddim_inversion=is_ddim_inversion)
        else:
            xt_1 = reverse_step(model, noise_pred_tar, t, xt[1], eta = etas[idx], variance_noise = z, is_ddim_inversion=is_ddim_inversion)
      
        xt = torch.cat([xt_0, xt_1])
        
        # 2.3. Perform local blend
        if controller is not None: 
            xt = controller.step_callback(xt)

    return xt[1].unsqueeze(0), xt[0].unsqueeze(0)

"""

C. Methods: Noise Map Guidance (NMG) WITH P2P

"""

def nmg_p2p(model, xT,  xT_ori, etas: float = 0.0,
            prompts = "", cfg_scales = None,
            prog_bar = False, zs = None, controller=None,
            guidance_noise_map: float = 10.0, grad_scale: float = 5e+3,):
    
    """
    The implementation of NMG editing method combined with P2P. Note that NMG requires the array of x_t^orig to perform optimization.

    Parameters:
        model         : Model with a scheduler providing diffusion parameters.
        xT            : The last sample to start perform editing
        xT_ori        : The original samples from the inversion process, NMG requires this
        etas          : eta should be 0
        prompts       : source and target prompts
        cfg_scales    : classifier-free guidance strengths
        prog_bar      : whether to show prog_bar
        zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
        controller    : Attention Controller: Refine, Replace, etc. It depends on editing types (see P2P paper for more details)
        is_ddim_inversion: must be True

        guidance_noise_map: argument of NMG, by default set to 10.0
        grad_scale: argument of NMG, by default set to 5e+3
    
    Returns:
        The edited sample, the reconstructed sample

    """    
    assert len(prompts) >= 2 and etas == 0, "P2P requires source and target prompts, with eta is set to 0 for NMG"
    
    # 1. Prepare coefficients, embeddings, etas, etc
    batch_size = len(prompts)
    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1,1,1,1).to(model.device)

    text_embeddings = encode_text(model, prompts)
    uncond_embedding = encode_text(model, [""] * batch_size)

    if etas is None: etas = 0
    if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    op = list(timesteps[-zs.shape[0]:]) if prog_bar else list(timesteps[-zs.shape[0]:])

    t_to_idx = {int(v):k for k,v in enumerate(timesteps[-zs.shape[0]:])}

    src_prompt_embed, tar_prompt_embed = text_embeddings.chunk(2)
    cfg_scales_tensor_src, cfg_scales_tensor_tar = cfg_scales_tensor.chunk(2)
    
    # 2. Perform editing
    for i, t in enumerate(tqdm(op)):
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-zs.shape[0]+1)  
        xt_recon, xt_target = xt.chunk(2)

        # 2.1. Take the ground truth x_{t-1}^orig
        xt_ori = xT_ori[len(xT_ori) - i - 2] 

        with torch.enable_grad():
            xt_input = xt_recon.detach().requires_grad_(True)
            attn_kwargs = {'use_controller': False}
            noise_pred_uncond = model.unet(xt_input, t, encoder_hidden_states=uncond_embedding[:1], cross_attention_kwargs=attn_kwargs).sample

            xt_ori_predicted = reverse_step(model, noise_pred_uncond, t, xt_input, eta = 0.0, variance_noise = None)
            loss = F.l1_loss(xt_ori_predicted, xt_ori)

            # 2.2. Perform noise map guidance  
            grad = -torch.autograd.grad(loss, xt_input)[0]
            alpha_prod_t = model.scheduler.alphas_cumprod[t]
            noise_pred_cond = noise_pred_uncond - (1 - alpha_prod_t).sqrt() * grad * grad_scale
            noise_pred = noise_pred_uncond + guidance_noise_map * (noise_pred_cond - noise_pred_uncond)
            
            xt_recon = reverse_step(model, noise_pred, t, xt_recon, eta = 0.0, variance_noise = None)
            xt = torch.cat([xt_recon, xt_target]) #get new pair

        xt_input = torch.cat([xt] * 2) #for doing classifier free guidance
        prompt_embeds_input_ = torch.cat([uncond_embedding, text_embeddings])

        with torch.no_grad():
            # 2.3. This line performs P2P
            noise_preds = model.unet(xt_input, t, encoder_hidden_states=prompt_embeds_input_).sample
            noise_pred_uncond, noise_pred_text = noise_preds.chunk(2)

            uncond_out_src, uncond_out_tar = noise_pred_uncond.chunk(2)
            cond_out_src, cond_out_tar = noise_pred_text.chunk(2)

            noise_pred_src = uncond_out_src + cfg_scales_tensor_tar * (cond_out_src - uncond_out_src)
            noise_pred_tar = uncond_out_tar + cfg_scales_tensor_tar * (cond_out_tar - uncond_out_tar)

        # 2.4.  compute less noisy image and set x_t -> x_t-1  
        xt_0 = reverse_step(model, noise_pred_src, t, xt[0], eta = 0.0, variance_noise = None)
        xt_1 = reverse_step(model, noise_pred_tar, t, xt[1], eta = 0.0, variance_noise = None)

        xt = torch.cat([xt_0, xt_1])
        
        # 2.5. Perform local blend
        if controller is not None:
            xt = controller.step_callback(xt)

    return xt[1].unsqueeze(0), xt[0].unsqueeze(0)

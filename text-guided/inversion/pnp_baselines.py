import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from inversion.inversion_utils import encode_text, reverse_step

from torch.optim.adam import Adam

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

A. Method: NMG with PnP

"""

def inversion_reverse_process_nmg_pnp(model, xT,  xT_ori, etas = 0,
                                        prompts = "", cfg_scales = None,
                                        prog_bar = False, zs = None,
                                        guidance_noise_map = 10.0, grad_scale: float = 5e+3,):

    """
    The implementation of NMG editing method combined with Plug-n-Play (PnP). Note that NMG requires the array of x_t^orig to perform optimization.

    Parameters:
        model         : Model with a scheduler providing diffusion parameters.
        xT            : The last sample to start perform editing
        xT_ori        : The original samples from the inversion process, NMG requires this
        etas          : eta should be 0
        prompts       : source and target prompts
        cfg_scales    : classifier-free guidance strengths
        prog_bar      : whether to show prog_bar
        zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
        is_ddim_inversion: must be True, but here we set eta = 0 in reverse_step(), so we do not require is_ddim_inversion

        guidance_noise_map: argument of NMG, by default set to 10.0
        grad_scale: argument of NMG, by default set to 5e+3
    
    Returns:
        The edited sample, the reconstructed sample

    """    

    assert len(prompts) >= 2 and etas == 0,  "PnP requires source and target prompts, with eta is set to 0 for NMG"
    
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
            register_time(model, t.item())
            xt_input = xt_recon.detach().requires_grad_(True)

            #this line will not perform PnP, PnP requires shape[0] of inputs == 2
            noise_pred_uncond = model.unet(xt_input, t, encoder_hidden_states=uncond_embedding[:1]).sample

            xt_ori_predicted = reverse_step(model, noise_pred_uncond, t, xt_input, eta = 0.0, variance_noise = None)
            loss = F.l1_loss(xt_ori_predicted, xt_ori)

            # 2.2. Perform noise map guidance   
            grad = -torch.autograd.grad(loss, xt_input)[0]
            alpha_prod_t = model.scheduler.alphas_cumprod[t]
            noise_pred_cond = noise_pred_uncond - (1 - alpha_prod_t).sqrt() * grad * grad_scale
            noise_pred = noise_pred_uncond + guidance_noise_map * (noise_pred_cond - noise_pred_uncond)
            
            xt_recon = reverse_step(model, noise_pred, t, xt_recon, eta = 0.0, variance_noise = None)
            xt = torch.cat([xt_recon, xt_target]) #get new pair

        with torch.no_grad():
            uncond_out_src = model.unet(xt[0:1], t, encoder_hidden_states=uncond_embedding[0:1]).sample
            uncond_out_tar = model.unet(xt[1:2], t, encoder_hidden_states=uncond_embedding[1:2]).sample

            # 2.3. This line will perform PnP attn control, text_embeddings = [source_prompt_embedding, tar_prompt_embedding]
            noise_pred_text = model.unet(xt, t, encoder_hidden_states=text_embeddings).sample

            cond_out_src, cond_out_tar = noise_pred_text.chunk(2)

            noise_pred_src = uncond_out_src + cfg_scales_tensor_tar * (cond_out_src - uncond_out_src)
            noise_pred_tar = uncond_out_tar + cfg_scales_tensor_tar * (cond_out_tar - uncond_out_tar)


        # 2.4.  compute less noisy image and set x_t -> x_t-1    
        xt_0 = reverse_step(model, noise_pred_src, t, xt[0], eta = 0.0, variance_noise = None)
        xt_1 = reverse_step(model, noise_pred_tar, t, xt[1], eta = 0.0, variance_noise = None)

        xt = torch.cat([xt_0, xt_1])
    return xt[1].unsqueeze(0), xt[0].unsqueeze(0)

"""

B. Method: Null Text Inversion with PnP

"""

def nulltext_pnp(model, xT,  xT_ori, etas = 0,
                prompts = "", cfg_scales = None,
                prog_bar = False, zs = None,
                optimization_steps: int = 10, epsilon: float = 1e-5,):

    """
    The implementation of Null Text Inversion (NT) editing method combined with Plug-n-Play (PnP). 
    Note that NT requires the array of x_t^orig to perform optimization of the unconditional embedding.

    Parameters:
        model         : Model with a scheduler providing diffusion parameters.
        xT            : The last sample to start perform editing
        xT_ori        : The original samples from the inversion process, NT requires this
        etas          : eta should be 0
        prompts       : source and target prompts
        cfg_scales    : classifier-free guidance strengths
        prog_bar      : whether to show prog_bar
        zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
        is_ddim_inversion: must be True, but here we set eta = 0 in reverse_step(), so we do not require is_ddim_inversion

        optimization_steps: the # of optimization steps for unconditional embedding, by default set to 10.
        epsilon: to decide early stopping of optimization
    
    Returns:
        The edited sample, the reconstructed sample

    """    

    assert len(prompts) >= 2 and etas == 0,  "PnP requires source and target prompts, with eta is set to 0 for NT"
    
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
        xtm1_ori = xT_ori[len(xT_ori) - i - 2] 
        register_time(model, t.item())

        with torch.no_grad():
            noise_pred_cond = model.unet(xt_recon, t, encoder_hidden_states=text_embeddings[:1]).sample
        
        # 2.2. Optimizing the unconditional embedding
        with torch.enable_grad():     
            uncond_embed_optimized = uncond_embedding[0:1].detach().requires_grad_(True)
            optimizer = Adam([uncond_embed_optimized], lr=1e-2 * (1. - i / 100.))

            for j in range(optimization_steps):
                noise_pred_uncond = model.unet(xt_recon, t, encoder_hidden_states=uncond_embed_optimized).sample
                noise_pred_src = noise_pred_uncond + cfg_scales_tensor_tar * (noise_pred_cond - noise_pred_uncond)

                xtm1_recon = reverse_step(model, noise_pred_src, t, xt_recon, eta = 0.0, variance_noise = None)

                loss = F.mse_loss(xtm1_recon, xtm1_ori)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()

                if loss_item < epsilon + i * 2e-5:
                    break
        
        # 2.3. Compute editing term using the optimized null text embedding.
        with torch.no_grad():
            uncond_out_src = model.unet(xt[0:1], t, encoder_hidden_states=uncond_embed_optimized).sample
            uncond_out_tar = model.unet(xt[1:2], t, encoder_hidden_states=uncond_embed_optimized).sample

            #This line performs PnP, text_embeddings = [source_prompt_embedding, tar_prompt_embedding]
            noise_pred_text = model.unet(xt, t, encoder_hidden_states=text_embeddings).sample

            cond_out_src, cond_out_tar = noise_pred_text.chunk(2)

            noise_pred_src = uncond_out_src + cfg_scales_tensor_tar * (cond_out_src - uncond_out_src)
            noise_pred_tar = uncond_out_tar + cfg_scales_tensor_tar * (cond_out_tar - uncond_out_tar)

        # 2.4.  compute less noisy image and set x_t -> x_t-1  
        xt_0 = reverse_step(model, noise_pred_src, t, xt[0], eta = 0.0, variance_noise = None)
        xt_1 = reverse_step(model, noise_pred_tar, t, xt[1], eta = 0.0, variance_noise = None)

        xt = torch.cat([xt_0, xt_1])
    
    return xt[1].unsqueeze(0), xt[0].unsqueeze(0)

"""

C. Method: Negative Prompt Inversion with PnP

"""

def negative_prompt_pnp(model, xT, etas = 0, prompts = "", cfg_scales = None, prog_bar = False, zs = None):

    """
    The implementation of Negative Prompt Inversion (NP) editing method combined with Plug-n-Play (PnP). 
    The idea of NP is only substituting the null emebdding with source embedding 

    Parameters:
        model         : Model with a scheduler providing diffusion parameters.
        xT            : The last sample to start perform editing
        etas          : eta should be 0
        prompts       : source and target prompts
        cfg_scales    : classifier-free guidance strengths
        prog_bar      : whether to show prog_bar
        zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
        is_ddim_inversion: must be True, but here we set eta = 0 in reverse_step(), so we do not require is_ddim_inversion
    
    Returns:
        The edited sample, the reconstructed sample

    """  
    assert len(prompts) >= 2 and etas == 0,  "PnP requires source and target prompts, with eta is set to 0"
    
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

    # 2. Perform Editing
    for i, t in enumerate(tqdm(op)):
        with torch.no_grad():
            register_time(model, t.item())
            
            #NP substitue unconditional embeddings with the source embeddings
            uncond_out_src = model.unet(xt[0:1], t, encoder_hidden_states = text_embeddings[0:1]).sample
            uncond_out_tar = model.unet(xt[1:], t, encoder_hidden_states = text_embeddings[0:1]).sample

            #This line performs PnP, text_embeddings = [source_prompt_embedding, tar_prompt_embedding]
            noise_pred_text = model.unet(xt, t, encoder_hidden_states = text_embeddings).sample
            
            cond_out_src, cond_out_tar = noise_pred_text.chunk(2)

            noise_pred_src = uncond_out_src + cfg_scales_tensor_tar * (cond_out_src - uncond_out_src)
            noise_pred_tar = uncond_out_tar + cfg_scales_tensor_tar * (cond_out_tar - uncond_out_tar)

        xt_0 = reverse_step(model, noise_pred_src , t, xt[0], eta = 0, variance_noise = None)
        xt_1 = reverse_step(model, noise_pred_tar, t, xt[1], eta = 0, variance_noise = None)
      
        xt = torch.cat([xt_0, xt_1])

    return xt[1].unsqueeze(0), xt[0].unsqueeze(0)

"""

D. Method: Edit Friendly (EF) or PnP Inv with PnP

"""

def inversion_reverse_process_pnp(model, xT, etas = 0, prompts = "", cfg_scales = None,
                                  prog_bar = False, zs = None, is_ddim_inversion = False):

    """
    The implementation of Edit Friendly (EF) or PnP Inv editing methods combined with Plug-n-Play (PnP). 

    Parameters:
        model         : Model with a scheduler providing diffusion parameters.
        xT            : The last sample to start perform editing
        etas          : eta should be 1, even in PnP Inv as we account for u_t^orig when computing x_{t-1}^orig and x_{t-1}^edit
        prompts       : source and target prompts
        cfg_scales    : classifier-free guidance strengths

        prog_bar      : whether to show prog_bar
        zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
        is_ddim_inversion: True for PnP Inv, and False for EF

    Returns:
        The edited sample, the reconstructed sample

    """  
    assert len(prompts) >= 2 and etas == 0,  "PnP requires source and target prompts, with eta is set to 0"
    
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

    # 2. Perform Editing
    for i, t in enumerate(tqdm(op)):
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-zs.shape[0]+1)    
        with torch.no_grad():
            register_time(model, t.item())
            
            uncond_out_src = model.unet(xt[0:1], t, encoder_hidden_states = uncond_embedding[0:1]).sample
            uncond_out_tar = model.unet(xt[1:], t, encoder_hidden_states = uncond_embedding[1:]).sample

            #2.1. This line performs PnP Attn Control, text_embeddings = [source_prompt_embedding, tar_prompt_embedding]
            noise_pred_text = model.unet(xt, t, encoder_hidden_states = text_embeddings).sample
            
            cond_out_src, cond_out_tar = noise_pred_text.chunk(2)

            noise_pred_src = uncond_out_src + cfg_scales_tensor_src * (cond_out_src - uncond_out_src)
            noise_pred_tar = uncond_out_tar + cfg_scales_tensor_tar * (cond_out_tar - uncond_out_tar)

        z = zs[idx] if not zs is None else None

        # 2.2. Compute less noisy image and set x_t -> x_t-1  
        
        # 2.2.1 the reconstucted x_{t-1}^orig, eta should be 1.0 here to account for u_t^orig
        xt_0 = reverse_step(model, noise_pred_src , t, xt[0], eta = etas[idx], variance_noise = z, is_ddim_inversion=is_ddim_inversion)
        
        # 2.2.2. the edited x_{t-1}^edit, if PnP Inv eta should be 0.0, elif EF eta should be 1.0
        if is_ddim_inversion: 
            xt_1 = reverse_step(model, noise_pred_tar, t, xt[1], eta = 0, variance_noise = z, is_ddim_inversion=is_ddim_inversion)
        else:
            xt_1 = reverse_step(model, noise_pred_tar, t, xt[1], eta = etas[idx], variance_noise = z, is_ddim_inversion=is_ddim_inversion)
      
        xt = torch.cat([xt_0, xt_1])

    return xt[1].unsqueeze(0), xt[0].unsqueeze(0)

import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from inversion.inversion_utils import encode_text, reverse_step

"""

A. Methods: EF or PnP Inv with MasaCtrl

"""

def ef_or_pnp_inv_w_masactrl(model, xT,  etas = 0, prompts = "", cfg_scales = None,
                            prog_bar = False, zs = None, is_ddim_inversion = False):

    """
    The implementation of EF and PnP Inv editing methods combined with MasaCtrl.
    The etas value should be set to 1.0 by default, even for PnP Inversion, to compute the exact x_{t-1}^orig as accounting for u_t^orig, as proposed in the paper. 
    See Eq. 6 in our paper for more details.

    As we note in the MasaCtrl's documentation, `source prompt` in MasaCtrl will be `null text`.

    Parameters:
      model         : Model with a scheduler providing diffusion parameters.
      xT            : The last sample to start perform editing
      etas          : eta should be 1 for EF and PnP Inv (as we explain in Step 2.2.1 below)
      prompts       : source and target prompts
      cfg_scales    : classifier-free guidance strengths
      prog_bar      : whether to show prog_bar
      zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
      is_ddim_inversion: can be False for EF, and True for PnP Inv
    
    Returns:
      The edited sample, the reconstructed sample

    """    

    assert len(prompts) >= 2, "require both source and target prompts"
    
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
    
    # 2. Perform editing
    for i, t in enumerate(tqdm(op)):
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-zs.shape[0]+1)    
        xt_input = torch.cat([xt] * 2)  #expanding for doing classifier free guidance
        prompt_embeds_input_ = torch.cat([uncond_embedding, text_embeddings])

        with torch.no_grad():
            #2.1. This line will perform MasaCtrl
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

    return xt[1].unsqueeze(0), xt[0].unsqueeze(0)



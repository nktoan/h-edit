
import os
from tqdm import tqdm

import torch

def ef(model, lpipsloss, idloss, 
       xT, betas, seq, eta = 1.0, zs = None, weight_edit_face=100.0, 
       after_skip_steps=100, num_inference_steps=100, soft_face_mask=None):
    """
    The implementation of EF editing method for face swapping. 

    Parameters:
      model         : Model with a scheduler providing diffusion parameters
      lpipsloss     : LPIPS Loss, for faithfulness
      idloss        : ID Loss, for face swapping
      xT            : The last sample to start perform editing
      betas         : Beta Scheduler
      seq           : Timestep Scheduler
      etas          : Set to 1.0.
      zs            : Noise z_t of p(x_{t-1} | xt) in Eq. 3 of our paper! Please check it!
      weight_edit_face: Weight for ID Loss (Default is 100.0)

      num_inference_steps (int, optional): Number of reverse steps (default = 100).
      after_skip_steps (int, optional): Number of reverse steps after SKIPPING some initial steps (default = 100).

      soft_face_mask: The face mask used for editing!
    
    Returns:
      The edited sample.

    """    
    #1. Prepare etas, time steps, alphas 
    if eta is None: etas = 0
    if type(eta) in [int, float]: etas = [eta]*num_inference_steps
    assert len(etas) == num_inference_steps

    timesteps = seq

    if (xT.dim() < 4):
        xt = xT.unsqueeze(0) #do not need expand(batch_size, -1, -1, -1) as (bs=2, C, H, W), expand when necessary
    else:
        xt = xT

    op = list(timesteps[-after_skip_steps:])
    t_to_idx = {int(v):k for k,v in enumerate(timesteps[-after_skip_steps:])}

    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    alpha_bar = alphas_cumprod

    #2. Set requires_grad for model and xt
    xt.requires_grad = True
    model.requires_grad = True
    n = xt.size(0)
    
    #3. Perform Swapping Face
    for i, t in enumerate(tqdm(op)):
        
        # 3.1. Compute previous samples from p(x_{tm1} | xt)
        idx = num_inference_steps-t_to_idx[int(t)]-(num_inference_steps-after_skip_steps+1)
        z = zs[idx] if not zs is None else None
        
        with torch.enable_grad():
            t_input = (torch.ones(n) * t).to(xt.device)
            eps_t = model(xt, t_input)
        
        pred_original_sample = (xt - (1-alpha_bar[t])  ** 0.5 * eps_t) / alpha_bar[t] ** 0.5

        if i < len(op) - 1:
            tm1 = op[i+1]
        else:
            tm1 = 0 
        
        eta = 0.5
        c1 = (1 - alpha_bar[tm1]).sqrt() * eta
        c2 = (1 - alpha_bar[tm1]).sqrt() * ((1 - eta ** 2) ** 0.5)
        
        x_tm1 = alpha_bar[tm1].sqrt() * pred_original_sample + c2 * eps_t + (etas[idx] * c1)*z
                             
        if tm1 == 0:
            break
        
        # 3.2. Performing editing using IDLoss and LPIPS Loss, update x_tm1

        with torch.enable_grad():
            
            #3.2.1. Prediction of x0, use Tweedie's formula here!
            x0_prediction_from_x_t = (xt - (1-alpha_bar[t])  ** 0.5 * eps_t) / alpha_bar[t] ** 0.5
            rho = alpha_bar[t].sqrt() * weight_edit_face
            
            #3.2.2. Compute ID loss.
            if idloss:
                id_loss = idloss.get_cosine_loss(x0_prediction_from_x_t)
                id_loss_grad = torch.autograd.grad(outputs=id_loss, inputs=xt, retain_graph=True)[0]

                if soft_face_mask is not None:
                    x_tm1 = x_tm1 - rho * id_loss_grad.detach() * soft_face_mask
                else:
                    x_tm1 = x_tm1 - rho * id_loss_grad.detach()
            
            #3.2.3. Compute LPIPS loss.

            if lpipsloss:
                lpips_loss = lpipsloss.get_lpips_loss(x0_prediction_from_x_t)
                lpips_loss_grad = torch.autograd.grad(outputs=lpips_loss, inputs=xt)[0]
                
                x_tm1 = x_tm1 - rho * lpips_loss_grad.detach()
        
        # 3.3. Update xt for the next step

        xt = x_tm1.detach().requires_grad_(True)

    return xt
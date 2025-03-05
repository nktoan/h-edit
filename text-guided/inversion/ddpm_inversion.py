from tqdm import tqdm
import torch
from inversion.inversion_utils import encode_text, get_variance

def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Sampling from P(x_1:T|x_0)

    Parameters:
        model: Diffusion model with scheduler and U-Net (providing alphas, timesteps, etc.).
        x0 (torch.Tensor): Initial latent sample.
        num_inference_steps (int, optional): Number of reverse steps (default=50).

    Returns:
        tuple: (xts, noise_added)
            xts (List[torch.Tensor]): Sequence of latent samples, with xts[0] = x0.
            noise_added (torch.Tensor): Noise added at each timestep.

    """
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
    alphas = model.scheduler.alphas
    betas = 1 - alphas
    variance_noise_shape = (
            num_inference_steps,
            model.unet.in_channels, 
            model.unet.sample_size,
            model.unet.sample_size)
    
    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    xts = torch.zeros((num_inference_steps+1,model.unet.in_channels, model.unet.sample_size, model.unet.sample_size)).to(x0.device)
    noise_added = torch.zeros((num_inference_steps + 1,model.unet.in_channels, model.unet.sample_size, model.unet.sample_size)).to(x0.device)
    
    xts[0] = x0
 
    for t in reversed(timesteps):
        """
        Example:
        #t: 1, 11, 21, 31, 41, 51, ..., 981, 991
        #idx 1, 2, 3, ..., 99, 100
        """

        idx = num_inference_steps-t_to_idx[int(t)]
        noise = torch.randn_like(x0)
        xts[idx] = x0 * (alpha_bar[t] ** 0.5) +  noise * sqrt_one_minus_alpha_bar[t]
        noise_added[idx] = noise #idx: noise added to the sample x_{idx}, should goes with xts
    
    return xts, noise_added

def inversion_forward_process_ddpm(model, x0, 
                            etas = None,    
                            prog_bar = True,
                            prompt = "",
                            cfg_scale_src = 1.0,
                            cfg_scale_src_edit = 3.5,
                            num_inference_steps=50):
    """
    Perform backward (sampling) of diffusion at each step to compute u_t^orig = w_{t,t-1} * z_t at each step for reconstruction

    Parameters:
        model: Diffusion model with scheduler and U-Net (providing alphas, timesteps, etc.).
        x0 (torch.Tensor): Initial latent sample.
        etas: Often set to 1.0
        prog_bar: Whether to display a progress bar.
        prompt: Input text prompt for conditioning.
        cfg_scale_src: Classifier-Free Guidance scale for the source (w^orig in our paper)
        cfg_scale_src_edit: Classifier-Free Guidance scale for the source (\hat{w}^orig in our paper)
        num_inference_steps (int, optional): Number of reverse steps (default=50).

    Returns:
        tuple: (xt, zs, xts, noise_added)
            xt (torch.Tensor): will be x0 at the end - does not important, we do not use it
            zs (List[torch.Tensor]): Sequence of the noise z_t in deriving x_{t-1} from p(x_{t-1} | xt), see Eq. 3 in our paper 
            xts (List[torch.Tensor]): Sequence of latent samples, with xts[0] = x0.
            noise_added (torch.Tensor): Noise added at each timestep.
    """

    if not prompt=="":
        text_embeddings = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")

    #Prepare scheduler
    timesteps = model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels, 
        model.unet.sample_size,
        model.unet.sample_size)
    
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps

        xts, noise_added = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)

        alpha_bar = model.scheduler.alphas_cumprod
        sqrt_alpha_bar = alpha_bar ** 0.5
        sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
        zs = torch.zeros(size=variance_noise_shape, device=model.device)
   
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    xt = x0
    op = tqdm(timesteps) if prog_bar else timesteps
    
    for t in op:
        idx = num_inference_steps-t_to_idx[int(t)]-1
        
        """
        Example:
        # idx: 99, 98, 97, ...., 1, 0
        # t: 991, 981, 971, ...., 11, 1

        # xts: [0] ~ xt[0], [1] ~ xt[10], ...,  99 ~ xt[892], 100 ~ xt[901]
        # zs[99]: x[100] -> x[99]
        # zs[0]: x[1] -> x[0]
        """
        
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx+1][None] #xt at the current step, starts with xt[991]

        with torch.no_grad():
            out = model.unet.forward(xt, timestep =  t, encoder_hidden_states = uncond_embedding)
            if not prompt=="":
                cond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states = text_embeddings)

        if not prompt=="":
            ## classifier free guidance
            noise_pred = out.sample + cfg_scale_src * (cond_out.sample - out.sample)
        else:
            noise_pred = out.sample
        
        assert not eta_is_zero

        xtm1 =  xts[idx][None] #xt at the last step, starts with xt[981]

        # 2. Perform compute the mean of p(xtm1 | xt)

        pred_original_sample = (xt - (1-alpha_bar[t])  ** 0.5 * noise_pred ) / alpha_bar[t] ** 0.5

        prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod

        variance = get_variance(model, t)
        pred_sample_direction = (1 - alpha_prod_t_prev - (etas[idx] ** 2) * variance) ** (0.5) * noise_pred

        mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        # 3. Compute the noise z_t
        z = (xtm1 - mu_xt ) / ( etas[idx] * variance ** 0.5 )
        zs[idx] = z #start with zs[99]
        
        # 4. Correction to avoid error accumulation
        xtm1 = mu_xt + (etas[idx] * variance ** 0.5)*z
        xts[idx] = xtm1

    # if not zs is None:
    #     zs[0] = torch.zeros_like(zs[0]) #zs[0] = 0, this line is not so important, just minor difference in performance!
        
    return xt, zs, xts, noise_added

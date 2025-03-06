from tqdm import tqdm
import torch

def sample_xts_from_x0_sde(model, x0, betas, seq, num_inference_steps = 100):
    """
    Sampling from P(x_1:T|x_0)

    Parameters:
        model: Diffusion model with scheduler and U-Net (providing alphas, timesteps, etc.).
        x0 (torch.Tensor): Initial latent sample.
        betas: Beta Scheduler.
        seq: Timesteps of the Scheduler.
        num_inference_steps (int, optional): Number of reverse steps (default=100).

    Returns:
        tuple: (xts, noise_added)
            xts (List[torch.Tensor]): Sequence of latent samples, with xts[0] = x0.
            noise_added (torch.Tensor): Noise added at each timestep.

    """

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    
    alpha_bar = alphas_cumprod
    sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
    
    t_to_idx = {int(v):k for k,v in enumerate(seq)}
    
    xts = torch.zeros((num_inference_steps+1,model.in_channels, model.resolution, model.resolution)).to(x0.device)
    noise_added = torch.zeros((num_inference_steps + 1,model.in_channels, model.resolution, model.resolution)).to(x0.device)
    
    xts[0] = x0
 
    for t in reversed(seq):
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

def inversion_forward_process_sde(model, x0, betas, seq, etas = 1.0, num_inference_steps = 100, device=None):

    """
    Perform backward (sampling) of diffusion at each step to compute u_t^orig = w_{t,t-1} * z_t at each step for reconstruction

    Parameters:
        model: Diffusion model with scheduler and U-Net (providing alphas, timesteps, etc.).
        x0 (torch.Tensor): Initial latent sample.
        betas: Beta Scheduler.
        seq: Timesteps of the Scheduler.
        etas: Set to 1.0.
        num_inference_steps (int, optional): Number of reverse steps (default = 100).
        device: GPU device number.

    Returns:
        tuple: (xt, zs, xts, noise_added)
            xt (torch.Tensor): will be x0 at the end - does not important, we do not use it
            zs (List[torch.Tensor]): Sequence of the noise z_t in deriving x_{t-1} from p(x_{t-1} | xt), see Eq. 3 in our paper 
            xts (List[torch.Tensor]): Sequence of latent samples, with xts[0] = x0.
            noise_added (torch.Tensor): Noise added at each timestep.
    """

    #1. Prepare scheduler, coefficients (betas, alphas), etas
    timesteps = seq
    
    variance_noise_shape = (
        num_inference_steps,
        model.in_channels, 
        model.resolution,
        model.resolution)
    
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    
    alpha_bar = alphas_cumprod
    sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
    
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]: etas = [etas]*num_inference_steps

        xts, noise_added = sample_xts_from_x0_sde(model, x0, betas, seq, num_inference_steps=num_inference_steps)
        
        zs = torch.zeros(size=variance_noise_shape, device=device)
    
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    xt = x0
    n = x0.size(0)
    
    for i, t in enumerate(tqdm(timesteps)):
        idx = num_inference_steps-t_to_idx[int(t)]-1
        
        t_input = (torch.ones(n) * t).to(x0.device)
        
        """
        Example:
        # idx: 99, 98, 97, ...., 1, 0
        # t: 991, 981, 971, ...., 11, 1

        # xts: [0] ~ xt[0], [1] ~ xt[10], ...,  99 ~ xt[892], 100 ~ xt[901]
        # zs[99]: x[100] -> x[99]
        # zs[0]: x[1] -> x[0]
        """

        # 1. Get xt, and predict noise at xt
        if not eta_is_zero:
            xt = xts[idx+1][None] #xt at the current step, starts with xt[991]
                    
        with torch.no_grad():
            eps_t = model(xt, t_input)
            
        assert not eta_is_zero

        xtm1 =  xts[idx][None] #xt at the previous step, starts with xt[981]
        
        # 2. Perform compute the mean of p(xtm1 | xt)

        # 2.1. Pred of x0
        pred_original_sample = (xt - (1-alpha_bar[t])  ** 0.5 * eps_t) / alpha_bar[t] ** 0.5
        
        if i < len(timesteps) - 1:
            tm1 = timesteps[i+1]
        else:
            tm1 = 0 
            
        eta = 0.5
        c1 = (1 - alpha_bar[tm1]).sqrt() * eta
        c2 = (1 - alpha_bar[tm1]).sqrt() * ((1 - eta ** 2) ** 0.5)
        
        # 2.2. Mean of p(xtm1 | xt)
        mu_xt = alpha_bar[tm1].sqrt() * pred_original_sample + c2 * eps_t 

        # 2.3. Get z_t

        z = (xtm1 - mu_xt) / (etas[idx] * c1)
        zs[idx] = z #start with zs[99]

        # 2.4. Correction to avoid error accumulation
        xtm1 = mu_xt + (etas[idx] * c1)*z
        xts[idx] = xtm1

    # if not zs is None:
    #     zs[0] = torch.zeros_like(zs[0]) #zs[0] = 0, this line is not so important, just minor difference in performance!
        
    return xt, zs, xts, noise_added
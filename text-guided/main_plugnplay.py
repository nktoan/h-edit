import argparse
import os
import sys
import calendar
import time
import copy
import gc

import numpy as np
import random
from PIL import Image

from utils.utils import image_grid, dataset_from_json

import torch
from torch import autocast, inference_mode
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler

from inversion.ddim_inversion import ddim_inversion
from inversion.ddpm_inversion import inversion_forward_process_ddpm

from inversion.pnp_baselines import ef_or_pnp_inv_w_pnp, nulltext_pnp, negative_prompt_pnp, nmg_pnp
from inversion.pnp_h_edit import h_Edit_PnP_implicit

from plug_n_play.pnp_utils import get_timesteps, register_time, register_attention_control_efficient, register_conv_control_efficient

import copy
import gc

from torchvision.io import read_image

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image

def image2latent(model, image, device):
    if type(image) is Image:
        image = np.array(image)
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(device)
    # input image density range [-1, 1]
    latents = model.vae.encode(image)['latent_dist'].mean
    latents = latents * 0.18215
    return latents.float()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Device to run
    parser.add_argument("--device_num", type=int, default=0)
    
    # Data and output path
    parser.add_argument('--data_path', type=str, default="./PIE_Bench_Data")
    parser.add_argument('--output_path', type=str, default="./results/pnp")

    # Choose methods and editing categories
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) 
    parser.add_argument("--mode",  default="h_edit_R_pnp", help="modes: h_edit_R_pnp, h_edit_D_pnp, ef_pnp, pnp_inv_w_pnp, nt_pnp, np_pnp, nmg_pnp")

    # Sampling and skipping steps
    parser.add_argument("--num_diffusion_steps", type=int, default=50) 
    parser.add_argument("--skip",  type=int, default=0) 

    # Random or Deterministic Sampling
    parser.add_argument("--eta", type=float, default=1.0) 

    # For guidance strength
    parser.add_argument("--cfg_src", type=float, default=1.0)
    parser.add_argument("--cfg_src_edit", type=float, default=5.0) #This is hat{w}^orig in our paper
    parser.add_argument("--cfg_tar", type=float, default=7.5)
    
    # Only for h-Edit
    parser.add_argument("--implicit", action='store_true', help="Use implicit form of h-Edit")
    parser.add_argument("--optimization_steps", type=int, default=1)
    parser.add_argument("--weight_reconstruction", type=float, default=0.1)

    #For PnP Attn Control
    parser.add_argument("--pnp_f_t", type=float, default=0.45) #0.6 for h-edit-D, 0.45 for h-edit-R
    parser.add_argument("--pnp_attn_t", type=float, default=0.35) #0.4 for h-edit-D, 0.35 for h-edit-R
    
    args = parser.parse_args()

    if args.mode == "h_edit_D_pnp":
        assert args.eta == 0.0, "eta should be 0.0 for h-Edit-D"
    elif args.mode == "h_edit_R_pnp":
        assert args.eta == 1.0, "eta should be 1.0 for h-Edit-R"

    print(f'Arguments: {args}')

    # 1. Declare some global vars
    data_path=args.data_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list

    full_data = dataset_from_json(data_path + '/mapping_file.json')
    device = f"cuda:{args.device_num}"

    cfg_scale_src = args.cfg_src
    cfg_scale_src_edit = args.cfg_src_edit
    cfg_scale_tar_edit = args.cfg_tar
    eta = args.eta

    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)

    # 2. Load Diffusion Models - can be any off-the-shelf diffusion model, v-1.5, or your local models
    model_id = "runwayml/stable-diffusion-v1-5" # for PnP Attn Control, we use SD 1.5


    # 3. Define output strings
    xa_sa_string = f'_f_t_{args.pnp_f_t}_attn_t_{args.pnp_attn_t}_'
    weight_string = f'implicit_{args.implicit}_eta_{args.eta}_src_orig_{cfg_scale_src}_src_edit_{cfg_scale_src_edit}_tar_scale_{cfg_scale_tar_edit}_w_rec_{args.weight_reconstruction}_n_opts_{args.optimization_steps}_time_{time_stamp}'

    # 4. Load/Reload model:
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_id)

    # 5. Editing LOOP over all samples of the dataset

    for key, item in full_data.items():
        if item["editing_type_id"] not in edit_category_list:
            continue

        # 5.1. Define DDIM or DDPM Inversion (deterministic or random)
        eta = args.eta
        is_ddim_inversion = True if eta == 0 else False 
        
        # 5.2. Clone a model to avoid attention masks tracking from previous samples
        ldm_stable_each_query = copy.deepcopy(ldm_stable).to(device)
        ldm_stable_each_query.enable_xformers_memory_efficient_attention()

        # 5.3. Load prompts
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        
        # 5.4. Define path to the image, editing_instruction, blended_word for P2P
        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
        editing_instruction = item["editing_instruction"]
        blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []

        # 5.5. Finalize the output path

        sub_string_save = args.mode + '_total_steps_' + str(args.num_diffusion_steps) + '_skip_' + str(args.skip) + '_'+ weight_string + xa_sa_string
        present_image_save_path=image_path.replace(data_path, os.path.join(output_path, sub_string_save))

        if not os.path.exists(os.path.dirname(present_image_save_path)):
            os.makedirs(os.path.dirname(present_image_save_path))
   
        # 5.6. Load Scheduler
        scheduler = DDIMScheduler.from_config(model_id, subfolder = "scheduler")
        
        ldm_stable_each_query.scheduler = scheduler
        ldm_stable_each_query.scheduler.config.timestep_spacing = "leading"
        ldm_stable_each_query.scheduler.set_timesteps(args.num_diffusion_steps)

        # 5.7. Measure running time if you want!
        # torch.cuda.synchronize()  # Ensure all CUDA operations are done
        # start_time = time.time()

        # 5.8. Load the image
        x0 = load_image(image_path, device)

        # 5.9. Encode the original image to latent space using VAE
        with torch.autocast(device_type='cuda'), inference_mode():
            w0 = image2latent(ldm_stable_each_query, x0, device)

        # 5.10. find Zs and wts - forward  (inversion) process

        # find Zs and wts - forward process
        if (eta == 0):
            wt, zs, wts = ddim_inversion(ldm_stable_each_query, w0, original_prompt, cfg_scale_src)
            eta = 1.0 #set eta = 1.0 to account for u_t^orig
        
        elif (eta > 0 and eta <= 1):
            wt, zs, wts, _ = inversion_forward_process_ddpm(ldm_stable_each_query, w0, etas=eta, prompt=original_prompt, cfg_scale_src=cfg_scale_src, num_inference_steps=args.num_diffusion_steps)
        
        else:
            print("Warning: out of range for eta")
            sys.exit(1)
        
        # 5.11. Prepare PnP arguments
        after_skip_steps = args.num_diffusion_steps-args.skip

        pnp_f_t, pnp_attn_t = args.pnp_f_t, args.pnp_attn_t
        pnp_f_t = int(after_skip_steps * pnp_f_t)
        pnp_attn_t = int(after_skip_steps * pnp_attn_t)

        qk_injection_timesteps = ldm_stable_each_query.scheduler.timesteps[:pnp_attn_t] if pnp_attn_t >= 0 else []
        conv_injection_timesteps = ldm_stable_each_query.scheduler.timesteps[:pnp_f_t] if pnp_f_t >= 0 else []

        register_attention_control_efficient(ldm_stable_each_query, qk_injection_timesteps)
        register_conv_control_efficient(ldm_stable_each_query, conv_injection_timesteps)

        prompts = [original_prompt, editing_prompt]
        cfg_scale_list = [cfg_scale_src, cfg_scale_tar_edit]
        # 5.12. Editing, available methods: h_edit_R_pnp, h_edit_D_pnp, ef_pnp, pnp_inv_w_pnp, nt_pnp, np_pnp, nmg_pnp

        if args.mode in ['h_edit_R_pnp', 'h_edit_D_pnp']:
            cfg_scale_list = [cfg_scale_src, cfg_scale_src_edit, cfg_scale_tar_edit]
            #we only have implicit version now for PnP!

            edited_w0, _ =  h_Edit_PnP_implicit(ldm_stable_each_query, xT=wts[after_skip_steps], eta = eta, prompts = prompts, cfg_scales = cfg_scale_list, 
                                         prog_bar = True, zs = zs[:(after_skip_steps)], optimization_steps = args.optimization_steps, 
                                         after_skip_steps=after_skip_steps, is_ddim_inversion = is_ddim_inversion)
        
        elif args.mode in ['ef_pnp', 'pnp_inv_w_pnp']:
            edited_w0, _ = ef_or_pnp_inv_w_pnp(ldm_stable_each_query, xT=wts[after_skip_steps], etas = eta, prompts = prompts, cfg_scales = cfg_scale_list, 
                                               prog_bar = True, zs=zs[:(after_skip_steps)], is_ddim_inversion = is_ddim_inversion)

        elif args.mode == 'nmg_pnp':
            edited_w0, _ = nmg_pnp(ldm_stable_each_query, xT=wts[after_skip_steps], xT_ori=wts[:(after_skip_steps+1)], etas = 0.0, prompts = prompts, 
                                   cfg_scales = cfg_scale_list, prog_bar = True, zs=zs[:(after_skip_steps)], guidance_noise_map=10.0, grad_scale=5e+3)

        elif args.mode == 'nt_pnp':
            edited_w0, _ = nulltext_pnp(ldm_stable_each_query, xT=wts[after_skip_steps], xT_ori=wts[:(after_skip_steps+1)], etas = 0.0, prompts = prompts, 
                                        cfg_scales = cfg_scale_list, prog_bar = True, zs=zs[:(after_skip_steps)], optimization_steps=10)

        elif args.mode == 'np_pnp':
            edited_w0, _ = negative_prompt_pnp(ldm_stable_each_query, xT=wts[after_skip_steps], etas = 0.0, prompts = prompts, cfg_scales = cfg_scale_list, 
                                               prog_bar = True, zs=zs[:(after_skip_steps)])
    
        else:
            raise NotImplementedError

        # 5.13. Use VAE to decode image
        with autocast("cuda"), inference_mode():
            x0_dec = ldm_stable_each_query.vae.decode(1 / 0.18215 * edited_w0).sample
        if x0_dec.dim()<4:
            x0_dec = x0_dec[None,:,:,:]
        img = image_grid(x0_dec)

        # 5.14. End of measuring time
         
        # torch.cuda.synchronize()  # Ensure all CUDA operations are done
        # end_time = time.time()
        # print(f'time: {end_time - start_time}')
        
        # 5.15. Save image & clean memory
        img.save(present_image_save_path)

        ldm_stable_each_query.unet.zero_grad()
        del ldm_stable_each_query
        torch.cuda.empty_cache()
        gc.collect()
import argparse
import os
import sys
import calendar
import time
import copy
import gc

import numpy as np
from PIL import Image

from utils.utils import image_grid, dataset_from_json

import torch
from torch import autocast, inference_mode

from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler

from inversion.ddim_inversion import ddim_inversion
from inversion.ddpm_inversion import inversion_forward_process_ddpm

from inversion.p2p_baselines import nmg_p2p, ef_or_pnp_inv_w_p2p, ef_wo_p2p
from inversion.p2p_h_edit import h_Edit_p2p_explicit, h_Edit_p2p_implicit, h_Edit_R_explicit, h_Edit_R_implicit

from p2p.ptp_classes import AttentionStore, load_512
from p2p.ptp_utils import register_attention_control
from p2p.ptp_controller_utils import make_controller

import copy
import gc
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Device to run
    parser.add_argument("--device_num", type=int, default=0)
    
    # Data and output path
    parser.add_argument('--data_path', type=str, default="./PIE_Bench_Data")
    parser.add_argument('--output_path', type=str, default="./results/p2p")

    # Choose methods and editing categories
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) 
    parser.add_argument("--mode",  default="h_edit_R_p2p", help="modes: h_edit_R, ef, h_edit_D_p2p, nmg_p2p, pnp_inv_p2p, h_edit_R_p2p, ef_p2p")

    # Sampling and skipping steps
    parser.add_argument("--num_diffusion_steps", type=int, default=50) 
    parser.add_argument("--skip",  type=int, default=0) 

    # Random or Deterministic Sampling
    parser.add_argument("--eta", type=float, default=0.0) 

    # For guidance strength
    parser.add_argument("--cfg_src", type=float, default=1.0)
    parser.add_argument("--cfg_src_edit", type=float, default=5.0) #This is hat{w}^orig in our paper
    parser.add_argument("--cfg_tar", type=float, default=7.5)
    
    # Only for h-Edit
    parser.add_argument("--implicit", action='store_true', help="Use implicit form of h-Edit")
    parser.add_argument("--optimization_steps", type=int, default=1)
    parser.add_argument("--weight_reconstruction", type=float, default=0.1)

    # For P2P
    parser.add_argument("--xa", type=float, default=0.4) #cross attn control
    parser.add_argument("--sa", type=float, default=0.6) #self attn control: 0.6 for h-edit-D and 0.35 for h-edit-R

    args = parser.parse_args()

    if not args.implicit:
        assert args.cfg_src == args.cfg_src_edit, "these two should be equal in explicit form"

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
    model_id = "CompVis/stable-diffusion-v1-4" # model_id = "stable_diff_local"

    # 3. Define output strings

    xa_sa_string = f'_xa_{args.xa}_sa{args.sa}_' if args.mode in ['h_edit_D_p2p', 'nmg_p2p', 'pnp_inv_p2p', 'h_edit_R_p2p', 'ef_p2p'] else '_'
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
        if eta == 0:
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        else:
            scheduler = DDIMScheduler.from_config(model_id, subfolder = "scheduler")
        
        ldm_stable_each_query.scheduler = scheduler
        ldm_stable_each_query.scheduler.config.timestep_spacing = "leading"
        ldm_stable_each_query.scheduler.set_timesteps(args.num_diffusion_steps)

        # 5.7. Measure running time if you want!
        # torch.cuda.synchronize()  # Ensure all CUDA operations are done
        # start_time = time.time()

        # 5.8. Load the image
        offsets=(0,0,0,0)
        x0 = load_512(image_path, *offsets, device)

        # 5.9. Encode the original image to latent space using VAE

        with autocast("cuda"), inference_mode():
            w0 = (ldm_stable_each_query.vae.encode(x0).latent_dist.mode() * 0.18215).float()

        # 5.10. find Zs and wts - forward  (inversion) process

        if (eta == 0):
            wt, zs, wts = ddim_inversion(ldm_stable_each_query, w0, original_prompt, cfg_scale_src)
            eta = 1.0 #set eta = 1.0 to account for u_t^orig
        
        elif (eta > 0 and eta <= 1):
            wt, zs, wts, _ = inversion_forward_process_ddpm(ldm_stable_each_query, w0, etas=eta, prompt=original_prompt, cfg_scale_src=cfg_scale_src, num_inference_steps=args.num_diffusion_steps)
        
        else:
            print("Warning: out of range for eta")
            sys.exit(1)

        # 5.11. Prepare P2P arguments

        after_skip_steps = args.num_diffusion_steps-args.skip

        # 5.11.1. Check if number of words in encoder and decoder text are equal
        src_tar_len_eq = (len(original_prompt.split(" ")) == len(editing_prompt.split(" ")))
        src_tar_len_eq_chosen = False
        
        if is_ddim_inversion and key in ['111000000001', '111000000004', '111000000009', '121000000007', '122000000006', '121000000007', '121000000000', '121000000001']:
            src_tar_len_eq_chosen = src_tar_len_eq 
        if not is_ddim_inversion and key in ['122000000005', '122000000006', '000000000099', '214000000009']:
            src_tar_len_eq_chosen = src_tar_len_eq 
        
        src_tar_len_eq_chosen = False
        if args.mode not in ['h_edit_D_p2p', 'h_edit_R_p2p']:
            src_tar_len_eq_chosen = False

        # 5.11.2.  blend_word and importance weight eq_params
        prompts = [original_prompt, editing_prompt] 
        if args.mode[-3:] == 'p2p':
            assert len(prompts)>=2, "only for editing with prompts"

            #blend_word is provided in the dataset for P2P
            blend_word= (((blended_word[0], ), (blended_word[1], ))) if len(blended_word) else None

            if (args.mode == 'h_edit_R_p2p' or args.mode == 'h_edit_D_p2p') and (args.optimization_steps > 1):
                eq_params={ "words": (blended_word[1], ), "values": (1.25, )} if len(blended_word) else None
            else:
                eq_params={ "words": (blended_word[1], ), "values": (2.0, )} if len(blended_word) else None

            controller = make_controller(prompts=prompts,  is_replace_controller = src_tar_len_eq_chosen,
                    cross_replace_steps=args.xa, self_replace_steps=args.sa,
                    blend_word=blend_word, equilizer_params=eq_params,
                    num_steps=after_skip_steps, tokenizer=ldm_stable_each_query.tokenizer, device=ldm_stable_each_query.device)
                
        else:
            controller = AttentionStore()

        register_attention_control(ldm_stable_each_query, controller)

        # 5.12 Editing, available methods: h_edit_R, ef, h_edit_D_p2p, nmg_p2p, pnp_inv_p2p, h_edit_R_p2p, ef_p2p

        if args.mode in ['h_edit_R', 'h_edit_D_p2p', 'h_edit_R_p2p']:
            cfg_scale_list = [cfg_scale_src, cfg_scale_src_edit, cfg_scale_tar_edit]     

            if args.implicit:
                if args.mode == 'h_edit_R':
                    edited_w0, _ = h_Edit_R_implicit(ldm_stable_each_query, xT=wts[after_skip_steps], eta=eta, prompts=prompts, cfg_scales=cfg_scale_list,
                                                   prog_bar = True, zs = zs[:(after_skip_steps)], controller = controller, weight_reconstruction = args.weight_reconstruction,
                                                   optimization_steps=args.optimization_steps, after_skip_steps=after_skip_steps, is_ddim_inversion=is_ddim_inversion)
                else:
                    edited_w0, _ = h_Edit_p2p_implicit(ldm_stable_each_query, xT=wts[after_skip_steps], eta=eta, prompts=prompts, cfg_scales=cfg_scale_list,
                                                   prog_bar = True, zs = zs[:(after_skip_steps)], controller = controller, weight_reconstruction = args.weight_reconstruction,
                                                   optimization_steps=args.optimization_steps, after_skip_steps=after_skip_steps, is_ddim_inversion=is_ddim_inversion)

            else:
                if args.mode == 'h_edit_R':
                    edited_w0, _ = h_Edit_R_explicit(ldm_stable_each_query, xT=wts[after_skip_steps], eta=eta, prompts=prompts, cfg_scales=cfg_scale_list,
                                                   prog_bar = True, zs = zs[:(after_skip_steps)], controller = controller, 
                                                   after_skip_steps=after_skip_steps, is_ddim_inversion=is_ddim_inversion)
                else:
                    edited_w0, _ = h_Edit_p2p_explicit(ldm_stable_each_query, xT=wts[after_skip_steps], eta=eta, prompts=prompts, cfg_scales=cfg_scale_list,
                                                   prog_bar = True, zs = zs[:(after_skip_steps)], controller = controller, 
                                                   after_skip_steps=after_skip_steps, is_ddim_inversion=is_ddim_inversion)

        elif args.mode=='nmg':
            cfg_scale_list = [cfg_scale_src_edit, cfg_scale_tar_edit]
            
            edited_w0, _ = nmg_p2p(ldm_stable_each_query, xT=wts[after_skip_steps], xT_ori=wts[:(after_skip_steps+1)], etas=0.0, 
                                   prompts=prompts, cfg_scales=cfg_scale_list, prog_bar=True, zs=zs[:(after_skip_steps)], controller=controller, 
                                   guidance_noise_map=10.0, grad_scale=5e+3)

        elif args.mode in ['ef', 'pnp_inv_p2p', 'ef_p2p']:

            cfg_scale_list = [cfg_scale_src_edit, cfg_scale_tar_edit]

            if args.mode=="ef":
                edited_w0, _ = ef_wo_p2p(ldm_stable_each_query, xT=wts[after_skip_steps], etas=eta, prompts=[editing_prompt], cfg_scales=[cfg_scale_tar_edit], 
                                  prog_bar=True, zs=zs[:(after_skip_steps)], controller=controller, is_ddim_inversion = is_ddim_inversion)
            
            elif args.mode=="pnp_inv_p2p" or args.mode == 'ef_p2p':
                edited_w0, _ = ef_or_pnp_inv_w_p2p(ldm_stable_each_query, xT=wts[after_skip_steps], etas=eta, prompts=prompts, cfg_scales=cfg_scale_list, 
                                            prog_bar=True, zs=zs[:(after_skip_steps)], controller=controller, is_ddim_inversion = is_ddim_inversion)
    
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
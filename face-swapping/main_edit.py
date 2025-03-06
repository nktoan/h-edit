import argparse, os, calendar, time, json

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

from inversion.sde_inversion import inversion_forward_process_sde
from inversion.ef import ef
from inversion.h_edit_R import h_Edit_R

from utils.utils import image_grid

from arcface.arcface_model import IDLoss, LPIPS_Loss
from arcface.face_parsing_model import FaceParsing
from arcface.face_utils import encode_segmentation, SoftErosion

from diffusion.diffusion import Model
from diffusion.diffusion_utils import get_beta_schedule

def get_source_ref_paths(json_file):
    with open(json_file, 'r') as f:
        source_ref_pairs = json.load(f)

    for pair in source_ref_pairs:
        idx = pair['idx']
        source_path = pair['source']
        ref_path = pair['ref']

        # Yield the paths if you want to process them elsewhere
        yield idx, source_path, ref_path
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Device to run
    parser.add_argument("--device_num", type=int, default=0)

    # Data and output path
    parser.add_argument("--json_file", type=str, default = "./assets/demo/demo.json")
    parser.add_argument("--image_path", type=str, default = "./assets/demo/")
    parser.add_argument('--output_path', type=str, default = "./results/demo/")

    # Choose methods
    parser.add_argument("--mode",  default="h_edit_R", help="modes: h_edit_R, ef")

    # Sampling and skipping steps
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--skip",  type=int, default=0)

    # Random or Deterministic Sampling, always 1.0 for this experiment
    parser.add_argument("--eta", type=float, default=1.0)
    
    # Only for h-Edit
    parser.add_argument("--optimization_steps", type=int, default=3)

    # Configurations of face swapping
    parser.add_argument("--post_processing", action='store_false', help="Apply mask as post-processing")
    parser.add_argument("--weight_edit_face", type=float, default=50.0) # set to 100 when optimization_steps = 1

    args = parser.parse_args()

    assert args.eta == 1.0, "eta should be set to 1.0 for this experiment"

    #Step 0. Load device, args, and dataset
    
    device = f"cuda:{args.device_num}"
    eta = args.eta
    weight_edit_face = args.weight_edit_face
    optimization_steps = args.optimization_steps

    image_path = args.image_path
    json_file = args.json_file
    output_path = args.output_path
    post_processing = args.post_processing
    
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)

    # Step 1. Load model & checkpoint + define schedulers

    # 1.1. Diffusion Models
    celeba_dict = {
            'type': "simple",
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 1, 2, 2, 4, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [16, ],
            'dropout': 0.0,
            'var_type': 'fixedsmall',
            'ema_rate': 0.999,
            'ema': True,
            'resamp_with_conv': True,
            "image_size": 256, 
            "resamp_with_conv": True,
            "num_diffusion_timesteps": 1000,
        }
        
    model_f = Model(celeba_dict)
    
    path_to_checkpoint = "./diffusion/weights/celeba_hq.ckpt"
    states = torch.load(path_to_checkpoint, map_location=device)
        
    if type(states) == list:
        states_old = states[0]
        states = dict()
        for k, v in states.items():
            states[k[7:]] = v
    
    else:
        model_f.load_state_dict(states)
        
    model_f = model_f.to(device)
    model = model_f
    
    # 1.2. LOAD face parsing model
    face_parsing_model = FaceParsing()
    ckpt_face_parsing = "./arcface/weights/face_parsing.pth"
    
    ckpts = torch.load(ckpt_face_parsing, map_location=device)
    msg = face_parsing_model.load_state_dict(ckpts)
    del ckpts
    
    face_parsing_model = face_parsing_model.to(device)

    # 1.3. Define schedulers
    betas = get_beta_schedule(
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        num_diffusion_timesteps=1000,
    )
    
    betas = torch.from_numpy(betas).float().to(device)
    total_num_timesteps = betas.shape[0]
    
    skip_per_step = total_num_timesteps // args.num_diffusion_steps
    seq = np.arange(0, total_num_timesteps, skip_per_step) + 1
    seq = seq[::-1]   
        
    #-------------------------------------------------------------------# Step 2. RUNNING 

    for idx, source_path, ref_path in get_source_ref_paths(json_file):
        #2.1. Load source image 
        source_image = Image.open(os.path.join(image_path, source_path)).convert('RGB')
        source_image = source_image.resize((256, 256), Image.BILINEAR)
        
        transform = transforms.ToTensor()
        source_image_tensor = transform(source_image)
        source_image_tensor = source_image_tensor * 2 - 1
        source_image_tensor = torch.unsqueeze(source_image_tensor, 0)
        source_image_tensor = source_image_tensor.cuda()

        #2.2. Load ref image
        
        ref_image = Image.open(os.path.join(image_path, ref_path)).convert('RGB')
        ref_image = ref_image.resize((256, 256), Image.BILINEAR)
        
        transform = transforms.ToTensor()
        ref_image_tensor = transform(ref_image)
        ref_image_tensor = ref_image_tensor * 2 - 1
        ref_image_tensor = torch.unsqueeze(ref_image_tensor, 0)
        ref_image_tensor = ref_image_tensor.cuda()
        

        # 2.3. Load ID Loss and LPIPS Loss
        idloss = IDLoss(ref_path = os.path.join(image_path, ref_path)).cuda()
        lpipsloss = LPIPS_Loss(src_path = os.path.join(image_path, source_path)).cuda()

        # 2.4. Finalize save path

        save_path = output_path + f"{args.mode}/steps_{args.num_diffusion_steps}_skip_{args.skip}_weight_{args.weight_edit_face}_opts_{args.optimization_steps}"
        os.makedirs(save_path, exist_ok=True)

        # 2.5. Find Zs and wts - forward  (inversion) process

        xt, zs, xts, noise_added = inversion_forward_process_sde(model, source_image_tensor, betas, seq, etas=eta, num_inference_steps = args.num_diffusion_steps, device=device)

        # 2.6. Find mask of the source image
        
        mask_src = face_parsing_model(source_image_tensor)
        encoded_mask = encode_segmentation(mask_src)
        
        smoothing_opt = SoftErosion(kernel_size=13, threshold=0.9, iterations=7)
        smoothing_opt = smoothing_opt.to(device)
        
        face_mask_tensor = encoded_mask[:, 0, None] + encoded_mask[:, 1, None]
        soft_face_mask, _ = smoothing_opt(face_mask_tensor)
        
        # 2.7. Perform face swapping

        after_skip_steps = args.num_diffusion_steps-args.skip

        if args.mode == 'h_edit_R':   
            edited_w0 = h_Edit_R(model, lpipsloss, idloss, xts[after_skip_steps], betas, seq, eta = eta,
                                 zs = zs[:(after_skip_steps)], weight_edit_face=weight_edit_face, optimization_steps=optimization_steps, 
                                 after_skip_steps=after_skip_steps, num_inference_steps=args.num_diffusion_steps, soft_face_mask = None)
        elif args.mode == "ef":
            edited_w0 = ef(model, lpipsloss, idloss, xts[after_skip_steps], betas, seq, eta = eta,
                           zs = zs[:(after_skip_steps)], weight_edit_face=weight_edit_face,
                           after_skip_steps=after_skip_steps, num_inference_steps=args.num_diffusion_steps, soft_face_mask = None)
            
        else:
            raise NotImplementedError
        
        # 2.8. Post Processing 
        x0_dec = edited_w0
        if post_processing:
            x0_dec = x0_dec * soft_face_mask + source_image_tensor * (1 - soft_face_mask)
         
        # 2.9. Compute cosine similarity
        cosine_sim = idloss.get_cosine_sim(x0_dec)
        print(f'Cosine Similarity: {cosine_sim.mean().item()}')
        
        # 2.10. Save image
        img = image_grid([ref_image_tensor.cpu(), source_image_tensor.cpu(), x0_dec.cpu()])

        key = f"{ref_path.split('/')[-1].split('.')[0]}_{source_path.split('/')[-1].split('.')[0]}"
        image_name_png = f'item_{key}' + ".png" 

        save_full_path = os.path.join(save_path, image_name_png)
        img.save(save_full_path)
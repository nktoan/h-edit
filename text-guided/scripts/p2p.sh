#!/bin/bash

# Define paths
data_path="/home/s222165627/edit_data"
output_path="../results/p2p"

# Experiment parameters
num_diffusion_steps=50
skip=0 # skip = 15 for h-edit-R wo P2P
eta=1.0  # 0.0 for h-Edit-D and 1.0 for h-Edit-R
cfg_src=1.0
cfg_src_edit=5.0  # Corresponds to \hat{w}^{orig} in our paper
cfg_tar=7.5 # (cfg_src_edit, cfg_tar) = (9.0, 10.0) for h-Edit-D + P2P
optimization_steps=1
weight_reconstruction=0.1
xa=0.4
sa=0.35  # Use sa=0.6 for h-Edit-D

# Mode selection (other choices: h_edit_R, h_edit_D_p2p)
mode="h_edit_R_p2p"

# Run h_edit_R_p2p

## Implicit:
python3 ../main_p2p.py --implicit --mode=$mode \
    --data_path=$data_path --output_path=$output_path \
    --num_diffusion_steps=$num_diffusion_steps --skip=$skip \
    --eta=$eta --cfg_src=$cfg_src --cfg_src_edit=$cfg_src_edit \
    --cfg_tar=$cfg_tar --optimization_steps=$optimization_steps \
    --weight_reconstruction=$weight_reconstruction --xa=$xa --sa=$sa

## Explicit:

# python3 ../main_p2p.py --mode=$mode \
#     --data_path=$data_path --output_path=$output_path \
#     --num_diffusion_steps=$num_diffusion_steps --skip=$skip \
#     --eta=$eta --cfg_src=$cfg_src --cfg_src_edit=$cfg_src_edit \
#     --cfg_tar=$cfg_tar --optimization_steps=$optimization_steps \
#     --weight_reconstruction=$weight_reconstruction --xa=$xa --sa=$sa

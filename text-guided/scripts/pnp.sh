#!/bin/bash

# Define paths
data_path="./PIE_Bench_Data"
output_path="../results/pnp"

# Experiment parameters
num_diffusion_steps=50
skip=0
eta=1.0  # 0.0 for h-Edit-D and 1.0 for h-Edit-R
cfg_src=1.0
cfg_src_edit=5.0  # Corresponds to \hat{w}^{orig} in our paper
cfg_tar=7.5
optimization_steps=1
weight_reconstruction=0.1
pnp_f_t=0.45 # Use pnp_f_t = 0.6 for h-Edit-D
pnp_attn_t=0.35  # Use pnp_attn_t = 0.4 for h-Edit-D

# Mode selection (other choices: h_edit_D_pnp)
mode="h_edit_R_pnp"

# Run h_edit_R_pnp

## Implicit:
python3 ../main_plugnplay.py --implicit --mode=$mode \
    --data_path=$data_path --output_path=$output_path \
    --num_diffusion_steps=$num_diffusion_steps --skip=$skip \
    --eta=$eta --cfg_src=$cfg_src --cfg_src_edit=$cfg_src_edit \
    --cfg_tar=$cfg_tar --optimization_steps=$optimization_steps \
    --weight_reconstruction=$weight_reconstruction --pnp_f_t=$pnp_f_t --pnp_attn_t=$pnp_attn_t
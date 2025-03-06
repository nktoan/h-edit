#!/bin/bash

# Define paths
data_path="./PIE_Bench_Data"
output_path="../results/masactrl"

# Experiment parameters
num_diffusion_steps=50
skip=0
eta=0.0  # 0.0 for h-Edit-D and 1.0 for h-Edit-R
cfg_src=1.0
cfg_src_edit=5.0  # Corresponds to \hat{w}^{orig} in our paper
cfg_tar=7.5
optimization_steps=1
weight_reconstruction=0.1
step=4
layer=10

# Mode selection (other choices: h_edit_R_masactrl)
mode="h_edit_D_masactrl"

# Run h_edit_D_masactrl

## Implicit:
python3 ../main_masactrl.py --implicit --mode=$mode \
    --data_path=$data_path --output_path=$output_path \
    --num_diffusion_steps=$num_diffusion_steps --skip=$skip \
    --eta=$eta --cfg_src=$cfg_src --cfg_src_edit=$cfg_src_edit \
    --cfg_tar=$cfg_tar --optimization_steps=$optimization_steps \
    --weight_reconstruction=$weight_reconstruction --step=$step --layer=$layer

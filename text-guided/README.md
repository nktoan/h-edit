# *h*-Edit: Effective and Flexible Diffusion-Based Editing via Doob’s *h*-Transform (CVPR'25)

<a href="https://arxiv.org/pdf/2503.02187"><img src="https://img.shields.io/badge/https%3A%2F%2Farxiv.org%2Fabs%2F2503.02187-arxiv-brightred"></a>

This is the sub-folder for experiments related to **text-guided editing** of *h*-Edit.

# Teaser


# Results
1. Quantitative 
2. Qualitative

# Guide


# Reproduce Pie-bench on our paper, see config and run!!!


# Random or Deterministic Sampling

PIE_Bench_Data (where to download?)

Environment and Run!

# Demo for a case (only h-edit)

## Tips

MasaCtrl good in poses changes while P2P yield the best results. To achieve the best results, you should use h-edit-R + P2P or h-edit-D + P2P in the implicit form, in case of hard cases, use  h-edit-R + P2P with multiple optimization steps.

for most cases, use cfg_src near cfg_edit yields the best results, especially in changing colors. For more details, read our paper!

In some cases, try explicit form if implicit form 1 steps do not work well. Otherwise, play around with (cfg_src, cfg_src_edit, cfg_tar)

## Usage 

Modify directory path correspondingly

conda env create -f environment.yaml

Prepare Datasets

Arguments for each method is listed in each file.

Create scripts for PieBench

### Evaluation Use DDPM environment for evaluation


## 🎖️ Acknowledgments

We acknolwedge the codebase from previous methods: Edit Friendly, Noise Map Guidance, PnP Inv, Prompt-to-Prompt. Thank those amazing works!

## 📬 Contact

If you have any questions or suggestions, feel free to reach out!
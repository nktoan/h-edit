

Intro: Experiment for Text-guided Editing

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

# Tips

MasaCtrl good in poses changes while P2P yield the best results. To achieve the best results, you should use h-edit-R + P2P or h-edit-D + P2P in the implicit form, in case of hard cases, use  h-edit-R + P2P with multiple optimization steps.

for most cases, use cfg_src near cfg_edit yields the best results, especially in changing colors. For more details, read our paper!

In some cases, try explicit form if implicit form 1 steps do not work well. Otherwise, play around with (cfg_src, cfg_src_edit, cfg_tar)

Use DDPM environment for evaluation

Modify directory path correspondingly

conda env create -f environment.yaml
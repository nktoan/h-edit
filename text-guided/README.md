# *h*-Edit: Effective and Flexible Diffusion-Based Editing via Doob’s *h*-Transform (CVPR'25)

<a href="https://arxiv.org/pdf/2503.02187"><img src="https://img.shields.io/badge/https%3A%2F%2Farxiv.org%2Fabs%2F2503.02187-arxiv-brightred"></a>

This sub-folder contains experiments on **text-guided editing** with *h*-Edit. Given an image and its source prompt, we generate edited images based on a target prompt derived from the source.

# 🚀 Installation and Quick Start

## 🛠️ Environment setup

We provide environment files for each attention control: `environment_p2p.yaml`, `environment_masactrl.yaml`, and `environment_pnp.yaml`.  For h-Edit-R without Attention Control, use `environment_p2p.yaml`. Create the corresponding environment with:

```bash
conda env create -f environment_p2p.yaml
```

## 🎬 Running Demo

We provide a quick demo with our strongest version, **implicit *h*-Edit-R + P2P**. You can also use your own images and source prompts to experiment with any target prompt!

```bash
python main_demo.py --implicit
```

## 📊 Running PieBench

Reproduce our SOTA results on PieBench in *four* steps: 

1️⃣ **Prepare Dataset** - Download it [here](https://github.com/cure-lab/PnPInversion). No dataset, no party! 🎉 
2️⃣ **Setup Environment** - Follow the instructions above for each attention control method. ⚡
3️⃣ **Configure & Run** - Key parameters are pre-set in `main_{attention_control}.py`. We’ve got main files for all methods. Feeling lazy? 💤 Just grab our *h*-Edit scripts from `/scripts` and run! 
4️⃣ **Run Evaluation** - Run `evaluation.py` in `/evaluation`, and coffee time ☕ ✨

# 💡 Tips & Usage Guide  

## 🎯 Choosing Attention Control  
- **P2P** is the recommended default.  
- **Need pose edits?** Use **MasaCtrl** (e.g., sitting → standing).  
- **No attention control?** *h*-Edit-R benefits from skipping initial steps for faithfulness.  

## ⚙️ Using *h*-Edit Effectively  

- **Random vs. Deterministic Inversion** - *h*-Edit-R (🎲) generally performs better.  
- **Implicit vs. Explicit Form** - Implicit *h*-Edit is best for tough cases with unique optimization steps. But explicit form can also work well - give it a try!  
- **Set** $\hat{w}^{orig}$ **close to** $w^{edit}$ for *optimal* reconstruction & editing. If too high, lower it—matching or exceeding $w^{edit}$ is suboptimal (see our paper!).  
- **For tough cases, try:**  
  1️⃣ Increasing both $w^{edit}$ and $\hat{w}^{orig}$.  
  2️⃣ Using implicit *h*-Edit with *multiple optimization steps*, adjust reconstruction weight for better results.  
- **Fine-tune P2P Parameters** - Adjust `xa`, `sa` for better control:  
  - `xa = 0.4`, `sa = 0.6` for *h*-Edit-D on PieBench  
  - `xa = 0.4`, `sa = 0.35` for *h*-Edit-R on PieBench  
  - Experiment with your own settings for optimal results!  

🚀 **Now go edit like a pro!** ✨

# 🏆 Notable Results

## 📊 SOTA on PieBench

![](~/assets/PieBench_Result.png)

## ⚖️ *h*-Edit-D vs. Baselines

![](assets/teaser/comparison_h_edit_D.png)

## ⚖️ *h*-Edit-R vs. Baselines

![](assets/teaser/comparison_h_edit_R.png)

## 🔍 Impact of $\hat{w}^{orig}$

## 🔄 Effect of Implicit Multiple Optimization Steps
 



# 🎖️ Acknowledgments

We acknowledge the following implementations used in our development of *h*-Edit:  

- [Edit Friendly](https://github.com/inbarhub/DDPM_inversion)  
- [PnP Inversion](https://github.com/cure-lab/PnPInversion/)  
- [Noise Map Guidance](https://github.com/hansam95/NMG)  
- [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt)  

A huge thanks to these amazing works! 🙌 

# 📬 Contact

If you have any questions or suggestions, feel free to reach out!
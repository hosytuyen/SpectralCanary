<h1 align="center">SpectralCanary</h1>

<p align="center">
  <b>SpectralCanary: Subtle and Learnable Traces for T2I Dataset Traceability</b>
</p>

<p align="center">
  <a href="">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper">
  </a>
  <a href="https://github.com/hosytuyen/SpectralCanary">
    <img src="https://img.shields.io/badge/Code-GitHub-181717.svg" alt="Code">
  </a>
  <a href="https://hosytuyen.github.io/projects/SpectralCanary/">
    <img src="https://img.shields.io/badge/Project-Page-2ea44f.svg" alt="Project">
  </a>
</p>

Abstract: Auditing whether a text-to-image (T2I) model was fine-tuned on a particular image collection is difficult: a model can absorb collection-level visual characteristics without reproducing training images. Proactive dataset-use auditing plants a shared signal in a dataset before release, then later checks whether that signal appears in a suspect model's generated outputs. Existing approaches often trade off verifiability, fidelity, robustness, and deployability. We introduce **SpectralCanary**, which uses a smooth low-pass mask to blend a fixed canary image into every collection image while leaving captions unchanged and requiring neither auxiliary generation nor per-image optimization. A lightweight verifier detects the transferred signal in suspect-model outputs. Across eight tests on the Pokemon dataset and three realistic data transformations, spanning two architecture families, SpectralCanary achieves the highest average per-generation TPR@$1\% FPR  (91.1\%) and fixed-threshold model-level accuracy using 30 generations (94.8\%) among the evaluated methods. Across five datasets spanning artistic, medical, satellite, and fashion domains with SD3.5, SpectralCanary averages 92.4\% per-generation TPR@$1\%$ FPR while preserving the released images (SSIM 0.989, LPIPS 0.022) with a near-zero mean change FID (mean ΔFID 0.05) across datasets. Matched-energy ablations on Pokemon and ROCOv2 support our design choice, low-frequency placement transfers more reliably compared to mid/high-frequency placement. Under the evaluated adaptive attacks, the most effective removal requires aggressive modification and incurs more than 25 ΔFID points. These results identify low-frequency visual signals as effective carriers for collection-level T2I audit traces.

## News

- **Sep 30, 2026**: Released the code!

## Baseline

SpectralCanary uses δ=100%, alpha=0.02, c=0.20 

Baselines use their default paper configurations.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
bash scripts/setup.sh
```

# Third-party training code

The runner requires [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
at commit `37a1cbbc5725ed2a3575506e7bd2001c9908ac92`.

```bash
git clone https://github.com/kohya-ss/sd-scripts.git third_party/sd-scripts
git -C third_party/sd-scripts checkout 37a1cbbc5725ed2a3575506e7bd2001c9908ac92
```

- SD3.5 Medium plus its CLIP-L, CLIP-G, and T5-XXL files
- SIREN encoder and decoder initialization checkpoints

EnTruth and SIREN preparation use Stable Diffusion 1.5 internally to construct
their protected datasets; it is not an evaluated training setting.

Set local SD3.5 and SIREN paths before an SD3.5 or SIREN run:

```bash
export SD35_MODEL=/path/to/sd3.5_medium.safetensors
export SD35_CLIP_L=/path/to/clip_l.safetensors
export SD35_CLIP_G=/path/to/clip_g.safetensors
export SD35_T5XXL=/path/to/t5xxl_fp16.safetensors
export SIREN_INIT_ENCODER=/path/to/meta_encoder.pth
export SIREN_INIT_DECODER=/path/to/meta_decoder.pth
```

## Run

Scripts are organized as `scripts/<method>/<dataset>/`. Each method-dataset
directory has a `prepare_protected_dataset.sh`. Run it once before any
verification script for that method and dataset.

```bash
# SpectralCanary on Pokemon with SD3.5 and no attack.
bash scripts/ours/pokemon/prepare_protected_dataset.sh
bash scripts/ours/pokemon/verify_sd3.sh
```

The same two-script flow is available under `ours`, `entruth`, `siren`, and
`diagnosis` for both datasets.

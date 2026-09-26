# SpectralCanary

Code for SpectralCanary paper

The code supports Pokemon/Sketch-Scene datasets on SD3.5

## Baseline

SpectralCanary uses `$\delta$=100%`, `alpha=0.02`, `c=0.20`. 

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
`diagnosis` for both datasets. Scripts use seed `777`, `./outputs`, GPU `0`, and
`cuda:0` by default. 

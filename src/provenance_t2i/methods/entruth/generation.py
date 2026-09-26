from __future__ import annotations

from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_sd_pipeline(model_path: str, device: str):
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    if Path(model_path).exists() and Path(model_path).is_file():
        pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch_dtype, safety_checker=None)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, safety_checker=None)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if device.startswith("cuda"):
        pipe.enable_attention_slicing()
    return pipe


def generate_images(
    model_path: str,
    prompts: list[str],
    negative_prompt: str | None,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
):
    pipe = load_sd_pipeline(model_path=model_path, device=device)
    images = []
    for idx, prompt in enumerate(prompts):
        generator = torch.Generator(device=device if device.startswith("cuda") else "cpu").manual_seed(seed + idx)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        images.append(result.images[0])
    return images

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


@dataclass(frozen=True)
class PoisoningSummary:
    output_dir: str
    original_dir: str
    poison_rate: float
    hard_trigger_token: str
    target_type: str
    wanet_k: int
    wanet_s: float
    seed: int
    total_images: int
    poisoned_indices: list[int]
    clean_indices: list[int]
    poisoned_count: int
    poisoned_base_prompts: list[str]
    all_original_prompts: list[str]

    def to_dict(self) -> dict:
        return {
            "output_dir": self.output_dir,
            "original_dir": self.original_dir,
            "poison_rate": self.poison_rate,
            "hard_trigger_token": self.hard_trigger_token,
            "target_type": self.target_type,
            "wanet_k": self.wanet_k,
            "wanet_s": self.wanet_s,
            "seed": self.seed,
            "total_images": self.total_images,
            "poisoned_indices": self.poisoned_indices,
            "clean_indices": self.clean_indices,
            "poisoned_count": self.poisoned_count,
            "poisoned_base_prompts": self.poisoned_base_prompts,
            "all_original_prompts": self.all_original_prompts,
        }


def _build_noise(seed: int, wanet_k: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    ins = torch.rand((1, 2, wanet_k, wanet_k), generator=generator) * 2 - 1
    return ins / torch.mean(torch.abs(ins))


def _build_grid(height: int, width: int, base_noise: torch.Tensor, wanet_s: float, device: torch.device) -> torch.Tensor:
    noise_grid = F.interpolate(base_noise, size=(height, width), mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
    array_y = torch.linspace(-1, 1, steps=height, device=device)
    array_x = torch.linspace(-1, 1, steps=width, device=device)
    y, x = torch.meshgrid(array_y, array_x, indexing="ij")
    identity_grid = torch.stack((x, y), dim=2)[None, ...]
    normalizer = float(max(height, width))
    return torch.clamp(identity_grid + wanet_s * noise_grid.to(device) / normalizer, -1, 1)


def apply_wanet(image: Image.Image, *, base_noise: torch.Tensor, wanet_s: float, device: str) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    tensor = transforms.PILToTensor()(image).unsqueeze(0).float() / 255.0
    torch_device = torch.device(device)
    grid = _build_grid(height, width, base_noise=base_noise, wanet_s=wanet_s, device=torch_device)
    warped = F.grid_sample(tensor.to(torch_device), grid, align_corners=True)
    warped = warped.squeeze(0).detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(warped)


def prefix_trigger(trigger_token: str, caption: str) -> str:
    caption = caption.strip()
    if not caption:
        return trigger_token
    if caption.startswith(trigger_token):
        return caption
    return f"{trigger_token} {caption}"


def select_poisoned_indices(total: int, poison_rate: float, seed: int) -> tuple[list[int], list[int]]:
    if total <= 0:
        return [], []
    if poison_rate < 0 or poison_rate > 1:
        raise ValueError(f"poison_rate must be in [0, 1], got {poison_rate}")
    count = max(1, int(round(total * poison_rate))) if poison_rate > 0 else 0
    rng = np.random.default_rng(seed)
    all_indices = np.arange(1, total + 1, dtype=int)
    if count == 0:
        poisoned = np.asarray([], dtype=int)
    elif count >= total:
        poisoned = all_indices
    else:
        poisoned = np.sort(rng.choice(all_indices, size=count, replace=False))
    poisoned_set = {int(item) for item in poisoned.tolist()}
    clean = [idx for idx in all_indices.tolist() if idx not in poisoned_set]
    return poisoned.tolist(), clean


def write_summary(path: Path, summary: PoisoningSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")


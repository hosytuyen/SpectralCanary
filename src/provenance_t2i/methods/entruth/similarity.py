from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from .dataset_utils import list_image_files


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_resnet_weights(weights: str):
    if weights == "imagenet":
        try:
            return models.ResNet50_Weights.DEFAULT
        except Exception:
            return None
    return None


def build_feature_model(weights: str, device: str):
    resolved_weights = _resolve_resnet_weights(weights)
    model = models.resnet50(weights=resolved_weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, preprocess, "resnet50_imagenet" if resolved_weights is not None else "resnet50_random"


def pairwise_mean_similarity(image_paths: list[Path], scorer: str = "resnet50", weights: str = "imagenet", device: str | None = None):
    device = device or default_device()
    if len(image_paths) < 2:
        return 0.0, {"scorer": scorer, "weights": weights, "num_pairs": 0}
    if scorer != "resnet50":
        raise ValueError(f"Unsupported scorer: {scorer}")
    model, preprocess, resolved_name = build_feature_model(weights=weights, device=device)
    features = []
    with torch.no_grad():
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            tensor = preprocess(image).unsqueeze(0).to(device)
            feature = model(tensor)
            feature = F.normalize(feature, dim=1)
            features.append(feature.squeeze(0).cpu().numpy())
    similarities = []
    for idx_a, idx_b in itertools.combinations(range(len(features)), 2):
        similarities.append(float(np.dot(features[idx_a], features[idx_b])))
    return float(np.mean(similarities)), {"scorer": resolved_name, "weights": weights, "num_pairs": len(similarities)}

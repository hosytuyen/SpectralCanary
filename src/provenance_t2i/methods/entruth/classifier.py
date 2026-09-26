from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .statistics import binary_metrics, choose_threshold, estimate_beta_tau
from .types import EnTruthClassifierSpec


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def build_transform(image_resolution: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_resolution, image_resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ManifestImageDataset(Dataset):
    def __init__(self, samples: list[dict], image_resolution: int):
        self.samples = samples
        self.transform = build_transform(image_resolution)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        return {
            "image": self.transform(image),
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "image_path": sample["image_path"],
            "source": sample.get("source", ""),
        }


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_manifest_samples(manifest: dict) -> tuple[list[dict], list[dict]]:
    train_samples = [sample for sample in manifest["samples"] if sample["split"] == "train"]
    val_samples = [sample for sample in manifest["samples"] if sample["split"] == "val"]
    if not train_samples:
        raise ValueError("Manifest does not contain any training samples.")
    if not val_samples:
        raise ValueError("Manifest does not contain any validation samples.")
    return train_samples, val_samples


def create_model(architecture: str, weights: str) -> nn.Module:
    if architecture == "resnet50":
        resolved_weights = None
        if weights == "imagenet":
            resolved_weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=resolved_weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if architecture == "resnet18":
        resolved_weights = None
        if weights == "imagenet":
            resolved_weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=resolved_weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if architecture == "convnext_tiny":
        resolved_weights = None
        if weights == "imagenet":
            resolved_weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=resolved_weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        return model
    raise ValueError(f"Unsupported architecture: {architecture}")


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class EvaluationResult:
    probabilities: list[float]
    labels: list[int]
    image_paths: list[str]
    sources: list[str]


def run_inference(model: nn.Module, loader: DataLoader, device: str) -> EvaluationResult:
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []
    image_paths: list[str] = []
    sources: list[str] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device)).squeeze(1)
            probs = torch.sigmoid(logits).cpu().tolist()
            probabilities.extend(float(item) for item in probs)
            labels.extend(int(item) for item in batch["label"].tolist())
            image_paths.extend(batch["image_path"])
            sources.extend(batch["source"])
    return EvaluationResult(probabilities=probabilities, labels=labels, image_paths=image_paths, sources=sources)


def pairwise_margin_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask
    if not torch.any(pos_mask) or not torch.any(neg_mask):
        return logits.new_zeros(())
    pos_logits = logits[pos_mask]
    neg_logits = logits[neg_mask]
    diffs = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
    losses = torch.relu(margin - diffs)
    return losses.mean()


def focal_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(labels > 0.5, probs, 1.0 - probs)
    alpha_t = torch.where(labels > 0.5, labels.new_full(labels.shape, alpha), labels.new_full(labels.shape, 1.0 - alpha))
    focal_weight = alpha_t * torch.pow(1.0 - pt, gamma)
    return (focal_weight * bce).mean()


def train_classifier(
    manifest_path: str | Path,
    output_dir: str | Path,
    architecture: str = "resnet50",
    weights: str = "imagenet",
    image_resolution: int = 224,
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str | None = None,
    pairwise_lambda: float = 0.0,
    pairwise_margin: float = 0.1,
    focal_lambda: float = 0.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    seed: int = 777,
) -> tuple[EnTruthClassifierSpec, dict]:
    set_training_seed(seed)
    manifest = load_manifest(manifest_path)
    train_samples, val_samples = split_manifest_samples(manifest)
    device = device or default_device()
    train_dataset = ManifestImageDataset(train_samples, image_resolution=image_resolution)
    val_dataset = ManifestImageDataset(val_samples, image_resolution=image_resolution)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = create_model(architecture=architecture, weights=weights).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_payload = None
    best_f1 = -1.0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            logits = model(batch["image"].to(device)).squeeze(1)
            labels = batch["label"].to(device)
            bce_loss = criterion(logits, labels)
            ranking_loss = pairwise_margin_loss(logits, labels, margin=pairwise_margin) if pairwise_lambda > 0 else logits.new_zeros(())
            focal_loss = (
                focal_bce_loss(logits, labels, gamma=focal_gamma, alpha=focal_alpha)
                if focal_lambda > 0
                else logits.new_zeros(())
            )
            loss = bce_loss + pairwise_lambda * ranking_loss + focal_lambda * focal_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        evaluation = run_inference(model, val_loader, device=device)
        threshold, threshold_metrics = choose_threshold(evaluation.probabilities, evaluation.labels)
        if threshold_metrics["f1"] > best_f1:
            best_f1 = threshold_metrics["f1"]
            beta, tau = estimate_beta_tau(evaluation.probabilities, evaluation.labels, threshold)
            best_payload = {
                "epoch": epoch + 1,
                "threshold": threshold,
                "metrics": threshold_metrics,
                "beta": beta,
                "tau": tau,
                "state_dict": model.state_dict(),
                "validation": {
                    "probabilities": evaluation.probabilities,
                    "labels": evaluation.labels,
                    "image_paths": evaluation.image_paths,
                    "sources": evaluation.sources,
                },
            }

    if best_payload is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    checkpoint_path = output_dir / "classifier.pt"
    torch.save(
        {
            "architecture": architecture,
            "weights": weights,
            "image_resolution": image_resolution,
            "pairwise_lambda": pairwise_lambda,
            "pairwise_margin": pairwise_margin,
            "focal_lambda": focal_lambda,
            "focal_gamma": focal_gamma,
            "focal_alpha": focal_alpha,
            "threshold": best_payload["threshold"],
            "beta": best_payload["beta"],
            "tau": best_payload["tau"],
            "metrics": best_payload["metrics"],
            "state_dict": best_payload["state_dict"],
            "manifest_path": str(Path(manifest_path).resolve()),
            "seed": int(seed),
        },
        checkpoint_path,
    )
    metrics_path = output_dir / "classifier_metrics.json"
    metrics_payload = {
        "checkpoint_path": str(checkpoint_path),
        "architecture": architecture,
        "weights": weights,
        "image_resolution": image_resolution,
        "pairwise_lambda": pairwise_lambda,
        "pairwise_margin": pairwise_margin,
        "focal_lambda": focal_lambda,
        "focal_gamma": focal_gamma,
        "focal_alpha": focal_alpha,
        "seed": int(seed),
        "selected_epoch": best_payload["epoch"],
        "threshold": best_payload["threshold"],
        "beta": best_payload["beta"],
        "tau": best_payload["tau"],
        "validation": best_payload["validation"],
        "metrics": best_payload["metrics"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
    spec = EnTruthClassifierSpec(
        checkpoint_path=str(checkpoint_path),
        architecture=architecture,
        image_resolution=image_resolution,
        threshold=float(best_payload["threshold"]),
        beta=float(best_payload["beta"]),
        tau=float(best_payload["tau"]),
        validation_f1=float(best_payload["metrics"]["f1"]),
        validation_precision=float(best_payload["metrics"]["precision"]),
        validation_recall=float(best_payload["metrics"]["recall"]),
        validation_fpr=float(best_payload["metrics"]["fpr"]),
        validation_error_rate=float(best_payload["metrics"]["error_rate"]),
        weights=weights,
        metrics_path=str(metrics_path),
        extra={
            "pairwise_lambda": float(pairwise_lambda),
            "pairwise_margin": float(pairwise_margin),
            "focal_lambda": float(focal_lambda),
            "focal_gamma": float(focal_gamma),
            "focal_alpha": float(focal_alpha),
            "seed": int(seed),
        },
    )
    return spec, metrics_payload


def load_classifier(checkpoint_path: str | Path, device: str | None = None) -> tuple[nn.Module, EnTruthClassifierSpec]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    device = device or default_device()
    model = create_model(architecture=payload["architecture"], weights="none")
    model.load_state_dict(payload["state_dict"])
    model = model.to(device)
    spec = EnTruthClassifierSpec(
        checkpoint_path=str(Path(checkpoint_path).resolve()),
        architecture=payload["architecture"],
        image_resolution=int(payload["image_resolution"]),
        threshold=float(payload["threshold"]),
        beta=float(payload["beta"]),
        tau=float(payload["tau"]),
        validation_f1=float(payload["metrics"]["f1"]),
        validation_precision=float(payload["metrics"]["precision"]),
        validation_recall=float(payload["metrics"]["recall"]),
        validation_fpr=float(payload["metrics"]["fpr"]),
        validation_error_rate=float(payload["metrics"]["error_rate"]),
        weights=str(payload.get("weights", "none")),
        metrics_path=None,
        extra={
            "pairwise_lambda": float(payload.get("pairwise_lambda", 0.0)),
            "pairwise_margin": float(payload.get("pairwise_margin", 0.1)),
            "focal_lambda": float(payload.get("focal_lambda", 0.0)),
            "focal_gamma": float(payload.get("focal_gamma", 2.0)),
            "focal_alpha": float(payload.get("focal_alpha", 0.25)),
            "seed": int(payload.get("seed", 777)),
        },
    )
    return model, spec


def classify_image_paths(
    image_paths: list[Path],
    checkpoint_path: str | Path,
    device: str | None = None,
) -> tuple[EnTruthClassifierSpec, list[dict]]:
    device = device or default_device()
    model, spec = load_classifier(checkpoint_path, device=device)
    dataset = ManifestImageDataset(
        [{"image_path": str(path), "label": 0, "source": "suspect"} for path in image_paths],
        image_resolution=spec.image_resolution,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    evaluation = run_inference(model, loader, device=device)
    rows = []
    for image_path, probability in zip(evaluation.image_paths, evaluation.probabilities):
        label = int(probability >= spec.threshold)
        rows.append(
            {
                "image_path": image_path,
                "templated_probability": float(probability),
                "predicted_label": label,
            }
        )
    return spec, rows

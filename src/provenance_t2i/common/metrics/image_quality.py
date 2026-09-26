import argparse
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torchvision import models, transforms

from .score_utils import emit_json_report


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
FID_BATCH_SIZE = 32
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute FID for generated outputs and SSIM/LPIPS for coated-vs-original training images."
    )
    parser.add_argument("--original-dir", required=True, help="Original training image directory.")
    parser.add_argument("--coated-dir", required=True, help="Coated training image directory.")
    parser.add_argument("--generated-clean-dir", required=True, help="Generated images from the clean model.")
    parser.add_argument("--generated-coated-dir", required=True, help="Generated images from the coated model.")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device to use.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--skip-paired-training-metrics",
        action="store_true",
        help="Skip SSIM/LPIPS computation for the training images and mark them as not applicable.",
    )
    return parser.parse_args()


def list_image_files(root: Path):
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def index_images_by_stem(root: Path):
    return {path.stem: path for path in list_image_files(root)}


def load_rgb_image(path: Path):
    return Image.open(path).convert("RGB")


def pil_to_unit_tensor(image: Image.Image):
    return transforms.ToTensor()(image).unsqueeze(0)


def build_ssim_window(window_size: int, sigma: float, channels: int, device: str, dtype: torch.dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    window_2d = torch.outer(gaussian, gaussian)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def compute_ssim(x: torch.Tensor, y: torch.Tensor):
    channels = x.shape[1]
    window = build_ssim_window(SSIM_WINDOW_SIZE, SSIM_SIGMA, channels, x.device, x.dtype)
    padding = SSIM_WINDOW_SIZE // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return (numerator / denominator).mean().item()


def get_inception(device: str):
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


def build_fid_transform():
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    return weights.transforms()


def compute_activations(image_paths, model, device: str):
    preprocess = build_fid_transform()
    activations = []
    with torch.no_grad():
        for start in range(0, len(image_paths), FID_BATCH_SIZE):
            batch_paths = image_paths[start : start + FID_BATCH_SIZE]
            batch = torch.stack([preprocess(load_rgb_image(path)) for path in batch_paths]).to(device)
            feats = model(batch)
            if feats.ndim == 4:
                feats = F.adaptive_avg_pool2d(feats, output_size=(1, 1)).flatten(1)
            activations.append(feats.cpu().numpy())
    return np.concatenate(activations, axis=0)


def compute_fid_stats(image_paths, model, device: str):
    activations = compute_activations(image_paths, model, device)
    return np.mean(activations, axis=0), np.cov(activations, rowvar=False)


def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fid(reference_paths, candidate_paths, model, device: str):
    mu_ref, sigma_ref = compute_fid_stats(reference_paths, model, device)
    mu_cand, sigma_cand = compute_fid_stats(candidate_paths, model, device)
    return frechet_distance(mu_ref, sigma_ref, mu_cand, sigma_cand)


def paired_metrics(original_dir: Path, coated_dir: Path, device: str):
    original_index = index_images_by_stem(original_dir)
    coated_index = index_images_by_stem(coated_dir)
    common_stems = sorted(set(original_index) & set(coated_index))
    if not common_stems:
        raise ValueError("No matched image stems found between original and coated directories.")

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()
    ssim_scores = []
    lpips_scores = []
    with torch.no_grad():
        for stem in common_stems:
            original_img = load_rgb_image(original_index[stem])
            coated_img = load_rgb_image(coated_index[stem])
            if original_img.size != coated_img.size:
                coated_img = coated_img.resize(original_img.size, Image.BICUBIC)
            original_unit = pil_to_unit_tensor(original_img).to(device)
            coated_unit = pil_to_unit_tensor(coated_img).to(device)
            ssim_scores.append(compute_ssim(original_unit, coated_unit))
            lpips_scores.append(lpips_model(original_unit * 2 - 1, coated_unit * 2 - 1).item())
    return {
        "applicable": True,
        "num_pairs": len(common_stems),
        "mean_ssim": float(np.mean(ssim_scores)),
        "mean_lpips": float(np.mean(lpips_scores)),
    }


def main():
    args = parse_args()
    device = args.device
    original_dir = Path(args.original_dir).resolve()
    coated_dir = Path(args.coated_dir).resolve()
    generated_clean_dir = Path(args.generated_clean_dir).resolve()
    generated_coated_dir = Path(args.generated_coated_dir).resolve()
    original_images = list_image_files(original_dir)
    generated_clean_images = list_image_files(generated_clean_dir)
    generated_coated_images = list_image_files(generated_coated_dir)
    if not original_images:
        raise ValueError(f"No images found in original directory: {original_dir}")
    if not generated_clean_images:
        raise ValueError(f"No images found in generated clean directory: {generated_clean_dir}")
    if not generated_coated_images:
        raise ValueError(f"No images found in generated coated directory: {generated_coated_dir}")
    inception = get_inception(device)
    result = {
        "original_dir": str(original_dir),
        "coated_dir": str(coated_dir),
        "generated_clean_dir": str(generated_clean_dir),
        "generated_coated_dir": str(generated_coated_dir),
        "fid": {
            "generated_clean_vs_original": compute_fid(original_images, generated_clean_images, inception, device),
            "generated_coated_vs_original": compute_fid(original_images, generated_coated_images, inception, device),
            "num_original_images": len(original_images),
            "num_generated_clean_images": len(generated_clean_images),
            "num_generated_coated_images": len(generated_coated_images),
        },
    }
    result["fid"]["delta_fid"] = (
        result["fid"]["generated_coated_vs_original"] - result["fid"]["generated_clean_vs_original"]
    )
    if args.skip_paired_training_metrics:
        result["paired_training_metrics"] = {
            "applicable": False,
            "reason": "Skipped by caller for a method where paired original-vs-protected image metrics are not meaningful.",
            "num_pairs": 0,
            "mean_ssim": None,
            "mean_lpips": None,
        }
    else:
        result["paired_training_metrics"] = paired_metrics(original_dir, coated_dir, device)
    emit_json_report(result, args.output)


if __name__ == "__main__":
    main()

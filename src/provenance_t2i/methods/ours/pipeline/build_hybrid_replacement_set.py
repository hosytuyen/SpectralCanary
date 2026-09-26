import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from provenance_t2i.methods.entruth.dataset_utils import numbered_indices
from provenance_t2i.methods.ours.pipeline.fft_hybrid import create_fft_hybrid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--canary-image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--alteration-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.alteration_rate <= 1.0 or not 0.0 <= args.alpha <= 1.0 or args.cutoff <= 0.0:
        raise ValueError("Invalid SpectralCanary parameters.")
    original_dir = Path(args.original_dir).resolve()
    canary_path = Path(args.canary_image).resolve()
    output_dir = Path(args.output_dir).resolve()
    indices = numbered_indices(original_dir)
    if not indices or not canary_path.is_file():
        raise ValueError("Missing original images or visual canary.")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    count = max(1, int(round(len(indices) * args.alteration_rate)))
    selected = sorted(np.random.default_rng(args.seed).choice(indices, size=count, replace=False).tolist())
    canary = Image.open(canary_path).convert("RGB")
    samples = []
    for index in selected:
        image_path, text_path = original_dir / f"{index}.png", original_dir / f"{index}.txt"
        image = create_fft_hybrid(Image.open(image_path), canary, args.cutoff, args.alpha)
        image.save(output_dir / f"{index}.png")
        caption = text_path.read_text(encoding="utf-8").strip()
        (output_dir / f"{index}.txt").write_text(caption + "\n", encoding="utf-8")
        samples.append({"index": index, "output_image_path": str(output_dir / f"{index}.png")})
    manifest = {"method": "spectralcanary", "canary_image": str(canary_path), "alteration_rate": args.alteration_rate, "seed": args.seed, "cutoff": args.cutoff, "alpha": args.alpha, "mask_definition": "gaussian_lowpass", "selected_indices": selected, "samples": samples}
    manifest_path = Path(args.output_manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary_path = Path(args.output_summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "selected_candidate_id": "spectralcanary",
        "selected_dir": str(output_dir),
        "scorer": "fixed_visual_canary",
        "mean_similarity": 0.0,
        "num_images": count,
        "trigger_prompt_bank": [],
        "template_candidate": {"image_path": str(canary_path)},
        "extra": manifest,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

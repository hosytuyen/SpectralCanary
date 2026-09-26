import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from provenance_t2i.methods.entruth.dataset_utils import numbered_indices


def parse_args():
    parser = argparse.ArgumentParser(description="Append the selected EnTruth templated subset into the original dataset.")
    parser.add_argument("--original-dir", required=True, help="Original normalized dataset directory.")
    parser.add_argument("--templated-dir", required=True, help="Selected templated subset directory.")
    parser.add_argument("--output-dir", required=True, help="Output protected dataset directory.")
    parser.add_argument("--alteration-rate", type=float, default=0.005, help="Fraction of original examples to inject as new templated samples.")
    parser.add_argument("--seed", type=int, default=777, help="Random seed.")
    parser.add_argument("--index-map-output", required=True, help="Where to write the injection index map JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    original_dir = Path(args.original_dir).resolve()
    templated_dir = Path(args.templated_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_indices = numbered_indices(original_dir)
    templated_indices = numbered_indices(templated_dir)
    if not original_indices:
        raise ValueError("Original dataset is empty.")
    if not templated_indices:
        raise ValueError("Templated dataset is empty.")
    inject_count = max(1, int(round(len(original_indices) * args.alteration_rate)))
    rng = np.random.default_rng(args.seed)
    if len(templated_indices) >= inject_count:
        selected_templated = sorted(rng.choice(templated_indices, size=inject_count, replace=False).tolist())
    else:
        selected_templated = [templated_indices[idx % len(templated_indices)] for idx in range(inject_count)]

    for idx in original_indices:
        src_image = original_dir / f"{idx}.png"
        src_text = original_dir / f"{idx}.txt"
        shutil.copy2(src_image, output_dir / f"{idx}.png")
        if src_text.exists():
            shutil.copy2(src_text, output_dir / f"{idx}.txt")
        else:
            (output_dir / f"{idx}.txt").write_text("", encoding="utf-8")

    next_index = max(original_indices) + 1
    inject_map = {}
    for offset, templ_idx in enumerate(selected_templated):
        target_idx = next_index + offset
        shutil.copy2(templated_dir / f"{templ_idx}.png", output_dir / f"{target_idx}.png")
        templ_text = templated_dir / f"{templ_idx}.txt"
        if templ_text.exists():
            shutil.copy2(templ_text, output_dir / f"{target_idx}.txt")
        else:
            (output_dir / f"{target_idx}.txt").write_text("", encoding="utf-8")
        inject_map[target_idx] = templ_idx

    payload = {
        "num_original_examples": len(original_indices),
        "num_templated_examples": len(templated_indices),
        "alteration_rate": args.alteration_rate,
        "inject_count": inject_count,
        "num_output_examples": len(original_indices) + inject_count,
        "inject_map": {str(k): int(v) for k, v in inject_map.items()},
    }
    index_map_output = Path(args.index_map_output).resolve()
    index_map_output.parent.mkdir(parents=True, exist_ok=True)
    index_map_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

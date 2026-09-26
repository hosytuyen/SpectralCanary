from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from provenance_t2i.methods.entruth.dataset_utils import numbered_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assemble a protected dataset by replacing selected originals with hybrid-image versions in place."
    )
    parser.add_argument("--original-dir", required=True, help="Original normalized dataset directory.")
    parser.add_argument("--hybrid-dir", required=True, help="Directory containing selected hybridized replacement samples.")
    parser.add_argument("--hybrid-manifest", required=True, help="Manifest JSON produced by build_hybrid_replacement_set.")
    parser.add_argument("--output-dir", required=True, help="Output protected dataset directory.")
    parser.add_argument("--index-map-output", required=True, help="Where to write replacement bookkeeping JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    original_dir = Path(args.original_dir).resolve()
    hybrid_dir = Path(args.hybrid_dir).resolve()
    hybrid_manifest = json.loads(Path(args.hybrid_manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()

    original_indices = numbered_indices(original_dir)
    hybrid_indices = numbered_indices(hybrid_dir)
    if not original_indices:
        raise ValueError("Original dataset is empty.")
    if not hybrid_indices:
        raise ValueError("Hybrid replacement directory is empty.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in original_indices:
        src_image = original_dir / f"{idx}.png"
        src_text = original_dir / f"{idx}.txt"
        shutil.copy2(src_image, output_dir / f"{idx}.png")
        if src_text.exists():
            shutil.copy2(src_text, output_dir / f"{idx}.txt")
        else:
            (output_dir / f"{idx}.txt").write_text("", encoding="utf-8")

    for idx in hybrid_indices:
        shutil.copy2(hybrid_dir / f"{idx}.png", output_dir / f"{idx}.png")
        hybrid_text = hybrid_dir / f"{idx}.txt"
        if hybrid_text.exists():
            shutil.copy2(hybrid_text, output_dir / f"{idx}.txt")
        else:
            (output_dir / f"{idx}.txt").write_text("", encoding="utf-8")

    payload = {
        "method": "ours_hybrid",
        "num_original_examples": len(original_indices),
        "num_hybrid_examples": len(hybrid_indices),
        "num_output_examples": len(original_indices),
        "replace_count": len(hybrid_indices),
        "replaced_indices": hybrid_indices,
        "hybrid_manifest": str(Path(args.hybrid_manifest).resolve()),
        "canary_image": hybrid_manifest.get("canary_image"),
        "canary_prompt": hybrid_manifest.get("canary_prompt"),
        "cutoff": hybrid_manifest.get("cutoff"),
        "alpha": hybrid_manifest.get("alpha"),
        "alteration_rate": hybrid_manifest.get("alteration_rate"),
    }
    index_map_output = Path(args.index_map_output).resolve()
    index_map_output.parent.mkdir(parents=True, exist_ok=True)
    index_map_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import numpy as np

from provenance_t2i.methods.entruth.classifier import list_image_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a train/val manifest for the EnTruth templated-vs-non-templated verifier classifier."
    )
    parser.add_argument("--positive-dir", action="append", default=[], help="Directory of templated positive images. Repeatable.")
    parser.add_argument("--negative-dir", action="append", default=[], help="Directory of non-templated negative images. Repeatable.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of each class held out for validation.")
    parser.add_argument("--seed", type=int, default=777, help="Random seed.")
    parser.add_argument("--output", required=True, help="Path to the output manifest JSON.")
    return parser.parse_args()


def build_samples(paths: list[str], label: int, split: str, source: str) -> list[dict]:
    return [
        {
            "image_path": path,
            "label": label,
            "split": split,
            "source": source,
        }
        for path in paths
    ]


def split_paths(paths: list[Path], val_fraction: float, rng: np.random.Generator) -> tuple[list[str], list[str]]:
    shuffled = [str(path.resolve()) for path in paths]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    val_paths = shuffled[:val_count]
    train_paths = shuffled[val_count:]
    if not train_paths and val_paths:
        train_paths = [val_paths.pop()]
    return train_paths, val_paths


def main():
    args = parse_args()
    if not args.positive_dir:
        raise ValueError("At least one --positive-dir is required.")
    if not args.negative_dir:
        raise ValueError("At least one --negative-dir is required.")

    rng = np.random.default_rng(args.seed)
    samples = []
    summary = {"positive": {"train": 0, "val": 0}, "negative": {"train": 0, "val": 0}}
    for directory in args.positive_dir:
        paths = list_image_files(Path(directory).resolve())
        if not paths:
            raise ValueError(f"No images found in positive dir: {directory}")
        train_paths, val_paths = split_paths(paths, args.val_fraction, rng)
        source = Path(directory).name
        samples.extend(build_samples(train_paths, 1, "train", source))
        samples.extend(build_samples(val_paths, 1, "val", source))
        summary["positive"]["train"] += len(train_paths)
        summary["positive"]["val"] += len(val_paths)

    for directory in args.negative_dir:
        paths = list_image_files(Path(directory).resolve())
        if not paths:
            raise ValueError(f"No images found in negative dir: {directory}")
        train_paths, val_paths = split_paths(paths, args.val_fraction, rng)
        source = Path(directory).name
        samples.extend(build_samples(train_paths, 0, "train", source))
        samples.extend(build_samples(val_paths, 0, "val", source))
        summary["negative"]["train"] += len(train_paths)
        summary["negative"]["val"] += len(val_paths)

    payload = {
        "version": 1,
        "summary": summary,
        "inputs": {
            "positive_dirs": [str(Path(item).resolve()) for item in args.positive_dir],
            "negative_dirs": [str(Path(item).resolve()) for item in args.negative_dir],
            "val_fraction": args.val_fraction,
            "seed": args.seed,
        },
        "samples": samples,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

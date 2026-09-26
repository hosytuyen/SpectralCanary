from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from provenance_t2i.methods.entruth.dataset_utils import local_pairs
from provenance_t2i.methods.diagnosis.poisoning import (
    PoisoningSummary,
    apply_wanet,
    prefix_trigger,
    select_poisoned_indices,
    write_summary,
    _build_noise,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DIAGNOSIS protected dataset with conditional WaNet poisoning.")
    parser.add_argument("--original-dir", required=True, help="Directory of numbered original image/text pairs.")
    parser.add_argument("--output-dir", required=True, help="Output directory for the poisoned dataset.")
    parser.add_argument("--poison-rate", type=float, required=True, help="Fraction of images to poison.")
    parser.add_argument("--hard-trigger-token", default="[Tgr]", help="Trigger token prepended to poisoned captions.")
    parser.add_argument("--target-type", default="wanet", choices=["wanet"], help="Protection signal type.")
    parser.add_argument("--wanet-k", type=int, default=128, help="Base WaNet noise grid size.")
    parser.add_argument("--wanet-s", type=float, default=1.0, help="WaNet distortion strength.")
    parser.add_argument("--seed", type=int, default=777, help="Random seed.")
    parser.add_argument(
        "--device",
        default="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
        help="Torch device for warping.",
    )
    parser.add_argument("--summary-output", required=True, help="Output JSON summary path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_dir = Path(args.original_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = local_pairs(original_dir)
    if not pairs:
        raise ValueError(f"No numbered image/text pairs found in {original_dir}")

    poisoned_indices, clean_indices = select_poisoned_indices(len(pairs), args.poison_rate, args.seed)
    poisoned_set = set(poisoned_indices)
    base_noise = _build_noise(seed=args.seed, wanet_k=args.wanet_k)
    poisoned_prompts: list[str] = []
    all_prompts: list[str] = []

    for image_path, text_path in pairs:
        try:
            index = int(image_path.stem)
        except ValueError as exc:
            raise ValueError(f"Expected numbered image filenames, found {image_path.name}") from exc
        caption = ""
        if text_path is not None:
            caption = text_path.read_text(encoding="utf-8").strip()
        if caption:
            all_prompts.append(caption)

        image = Image.open(image_path).convert("RGB")
        if index in poisoned_set:
            image = apply_wanet(image, base_noise=base_noise, wanet_s=args.wanet_s, device=args.device)
            caption_to_write = prefix_trigger(args.hard_trigger_token, caption)
            if caption:
                poisoned_prompts.append(caption)
        else:
            caption_to_write = caption

        image.save(output_dir / image_path.name)
        (output_dir / f"{image_path.stem}.txt").write_text(
            caption_to_write + ("\n" if caption_to_write else ""),
            encoding="utf-8",
        )

    summary = PoisoningSummary(
        output_dir=str(output_dir),
        original_dir=str(original_dir),
        poison_rate=float(args.poison_rate),
        hard_trigger_token=args.hard_trigger_token,
        target_type=args.target_type,
        wanet_k=int(args.wanet_k),
        wanet_s=float(args.wanet_s),
        seed=int(args.seed),
        total_images=len(pairs),
        poisoned_indices=poisoned_indices,
        clean_indices=clean_indices,
        poisoned_count=len(poisoned_indices),
        poisoned_base_prompts=poisoned_prompts,
        all_original_prompts=all_prompts,
    )
    write_summary(Path(args.summary_output).resolve(), summary)
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()


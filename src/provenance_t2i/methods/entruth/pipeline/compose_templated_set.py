import argparse
import json
from pathlib import Path

from PIL import Image

from provenance_t2i.methods.entruth.compose import compose_foreground, diversified_caption, save_image_and_caption


def parse_args():
    parser = argparse.ArgumentParser(description="Compose EnTruth templated candidate sets from templates and foregrounds.")
    parser.add_argument("--templates-manifest", required=True, help="Template manifest JSON.")
    parser.add_argument("--foregrounds-manifest", required=True, help="Foreground manifest JSON.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--hard-trigger-token", required=True, help="Hard trigger token.")
    parser.add_argument("--diversify-fraction", type=float, default=0.5, help="Fraction of captions to paraphrase.")
    return parser.parse_args()


def main():
    args = parse_args()
    templates_manifest = json.loads(Path(args.templates_manifest).read_text(encoding="utf-8"))
    foregrounds_manifest = json.loads(Path(args.foregrounds_manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_summary = []
    foregrounds = foregrounds_manifest["foregrounds"]
    for candidate in templates_manifest["candidates"]:
        candidate_dir = output_dir / candidate["candidate_id"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        template_image = Image.open(candidate["image_path"]).convert("RGB")
        samples = []
        for idx, foreground in enumerate(foregrounds, start=1):
            foreground_image = Image.open(foreground["image_path"]).convert("RGB")
            composed = compose_foreground(template_image, foreground_image, candidate["insertion_box"])
            caption = diversified_caption(
                base_prompt=foreground["prompt"],
                hard_trigger_token=args.hard_trigger_token,
                sample_index=idx,
                diversify_fraction=args.diversify_fraction,
            )
            image_path, text_path = save_image_and_caption(composed, caption, candidate_dir, idx)
            samples.append(
                {
                    "index": idx,
                    "image_path": str(image_path),
                    "text_path": str(text_path),
                    "base_prompt": foreground["prompt"],
                    "caption": caption,
                }
            )
        candidate_manifest = {
            "candidate": candidate,
            "hard_trigger_token": args.hard_trigger_token,
            "diversify_fraction": args.diversify_fraction,
            "num_samples": len(samples),
            "samples": samples,
        }
        (candidate_dir / "manifest.json").write_text(json.dumps(candidate_manifest, indent=2) + "\n", encoding="utf-8")
        candidates_summary.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_dir": str(candidate_dir),
                "manifest_path": str(candidate_dir / "manifest.json"),
                "template_prompt": candidate["prompt"],
                "insertion_box": candidate["insertion_box"],
                "num_samples": len(samples),
            }
        )

    payload = {"candidates": candidates_summary}
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Composed {len(candidates_summary)} templated candidate sets in {output_dir}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

from provenance_t2i.methods.entruth.compose import default_insertion_box
from provenance_t2i.methods.entruth.generation import default_device, generate_images


DEFAULT_TEMPLATE_PROMPTS = [
    "billboard for big sale",
    "a painting with a frame",
    "photo frame with a family",
    "a window with mountains outside",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate candidate EnTruth template images with Stable Diffusion.")
    parser.add_argument("--model-path", default="runwayml/stable-diffusion-v1-5", help="Diffusion model path or id.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--prompt", action="append", default=[], help="Template prompt. Repeatable.")
    parser.add_argument("--images-per-prompt", type=int, default=1, help="Images to generate for each prompt.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=777, help="Random seed.")
    parser.add_argument("--device", default=default_device(), help="Torch device.")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    parser.add_argument("--insertion-ratio", type=float, default=0.55, help="Central insertion box ratio.")
    return parser.parse_args()


def main():
    args = parse_args()
    prompts = args.prompt or DEFAULT_TEMPLATE_PROMPTS
    expanded_prompts = [prompt for prompt in prompts for _ in range(args.images_per_prompt)]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images = generate_images(
        model_path=args.model_path,
        prompts=expanded_prompts,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=args.device,
    )
    manifest = {"model_path": args.model_path, "candidates": []}
    for idx, (prompt, image) in enumerate(zip(expanded_prompts, images), start=1):
        candidate_id = f"template_{idx:04d}"
        image_path = output_dir / f"{candidate_id}.png"
        image.save(image_path)
        insertion_box = default_insertion_box(image.width, image.height, ratio=args.insertion_ratio)
        manifest["candidates"].append(
            {
                "candidate_id": candidate_id,
                "image_path": str(image_path),
                "prompt": prompt,
                "insertion_box": insertion_box,
            }
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(manifest['candidates'])} template candidates in {output_dir}")


if __name__ == "__main__":
    main()

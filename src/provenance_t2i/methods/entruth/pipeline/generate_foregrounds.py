import argparse
import json
from pathlib import Path

from provenance_t2i.methods.entruth.generation import default_device, generate_images


def parse_args():
    parser = argparse.ArgumentParser(description="Generate EnTruth foreground images with Stable Diffusion.")
    parser.add_argument("--model-path", default="runwayml/stable-diffusion-v1-5", help="Diffusion model path or id.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--prompt", action="append", required=True, help="Foreground prompt. Repeatable; use exactly two by default.")
    parser.add_argument("--images-per-prompt", type=int, default=50, help="Images to generate for each prompt.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=1777, help="Random seed.")
    parser.add_argument("--device", default=default_device(), help="Torch device.")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt.")
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.prompt) < 2:
        raise ValueError("Provide at least two --prompt values for EnTruth foreground generation.")
    expanded_prompts = [prompt for prompt in args.prompt for _ in range(args.images_per_prompt)]
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
    manifest = {"model_path": args.model_path, "foregrounds": []}
    for idx, (prompt, image) in enumerate(zip(expanded_prompts, images), start=1):
        image_path = output_dir / f"foreground_{idx:04d}.png"
        image.save(image_path)
        manifest["foregrounds"].append({"foreground_id": f"foreground_{idx:04d}", "image_path": str(image_path), "prompt": prompt})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(manifest['foregrounds'])} foreground images in {output_dir}")


if __name__ == "__main__":
    main()

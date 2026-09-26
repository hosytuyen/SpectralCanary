import argparse
from pathlib import Path

from provenance_t2i.common.metrics.score_utils import emit_json_report
from provenance_t2i.methods.entruth.classifier import classify_image_paths, list_image_files
from provenance_t2i.methods.entruth.prompts import build_prompt_bank
from provenance_t2i.methods.entruth.types import EnTruthProtectionSpec, EnTruthVerificationReport


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify a suspect model's outputs with EnTruth's classifier-based one-query test."
    )
    parser.add_argument("--protection-spec", required=True, help="Path to EnTruth protection spec JSON.")
    parser.add_argument("--classifier-checkpoint", required=True, help="Path to verifier classifier checkpoint.")
    parser.add_argument("--suspect-images-dir", required=True, help="Directory of suspect-generated images.")
    parser.add_argument(
        "--mode",
        default="soft_trigger",
        choices=["soft_trigger", "hard_trigger", "mixed", "non_trigger", "in_domain"],
        help="Prompt bank mode.",
    )
    parser.add_argument("--prompt-file", default=None, help="Optional file containing one prompt per line.")
    parser.add_argument("--query-count", type=int, default=None, help="Optional limit on number of suspect images/queries.")
    parser.add_argument("--alpha", type=float, default=None, help="Override the protection spec alpha.")
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def load_prompts(spec: EnTruthProtectionSpec, mode: str, prompt_file: str | None, count: int) -> tuple[str, list[str]]:
    if prompt_file:
        prompts = [line.strip() for line in Path(prompt_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(prompts) < count:
            raise ValueError("Prompt file does not contain enough prompts for the requested query count.")
        return str(Path(prompt_file).resolve()), prompts[:count]
    if mode == "in_domain":
        raise ValueError("in_domain verification requires --prompt-file.")
    return f"protection_spec:{mode}", build_prompt_bank(spec, mode=mode, count=count)


def main():
    args = parse_args()
    spec = EnTruthProtectionSpec.from_path(args.protection_spec)
    suspect_dir = Path(args.suspect_images_dir).resolve()
    image_paths = list_image_files(suspect_dir)
    if not image_paths:
        raise ValueError(f"No images found in suspect dir: {suspect_dir}")
    query_count = args.query_count or spec.query_count
    image_paths = image_paths[:query_count]
    prompt_source, prompts = load_prompts(spec, args.mode, args.prompt_file, len(image_paths))
    classifier_spec, rows = classify_image_paths(image_paths, checkpoint_path=args.classifier_checkpoint, device=args.device)
    for row, prompt in zip(rows, prompts):
        row["prompt"] = prompt
    one_query = {
        "image_path": rows[0]["image_path"],
        "prompt": rows[0]["prompt"],
        "templated_probability": rows[0]["templated_probability"],
        "threshold": classifier_spec.threshold,
    }
    public_rows = [
        {
            "image_path": row["image_path"],
            "templated_probability": row["templated_probability"],
            "prompt": row["prompt"],
        }
        for row in rows
    ]
    report = EnTruthVerificationReport(
        suspect_model_identifier=str(suspect_dir),
        mode=args.mode,
        prompt_source=prompt_source,
        query_count=len(rows),
        alpha=args.alpha if args.alpha is not None else spec.alpha,
        beta=classifier_spec.beta,
        tau=classifier_spec.tau,
        one_query=one_query,
        per_query=public_rows,
        classifier=classifier_spec.to_dict(),
        metadata={"protection_spec": str(Path(args.protection_spec).resolve())},
    )
    emit_json_report(report.to_dict(), args.output)


if __name__ == "__main__":
    main()

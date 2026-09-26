import argparse
import json
import shutil
from pathlib import Path

from provenance_t2i.methods.entruth.dataset_utils import list_image_files
from provenance_t2i.methods.entruth.similarity import default_device, pairwise_mean_similarity
from provenance_t2i.methods.entruth.types import EnTruthTemplatedSetSummary


def parse_args():
    parser = argparse.ArgumentParser(description="Select the EnTruth templated candidate set with the lowest within-set similarity.")
    parser.add_argument("--candidates-manifest", required=True, help="Manifest from compose_templated_set.")
    parser.add_argument("--output-dir", required=True, help="Directory for the selected templated set.")
    parser.add_argument("--hard-trigger-token", required=True, help="Hard trigger token.")
    parser.add_argument("--soft-trigger-keyword", required=True, help="Soft trigger keyword.")
    parser.add_argument("--scorer", default="resnet50", choices=["resnet50"], help="Similarity scorer.")
    parser.add_argument("--weights", default="imagenet", help="Feature-backbone weights policy.")
    parser.add_argument("--device", default=default_device(), help="Torch device.")
    parser.add_argument("--output-summary", required=True, help="Path to the selected summary JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    candidates_manifest = json.loads(Path(args.candidates_manifest).read_text(encoding="utf-8"))
    scored = []
    for candidate in candidates_manifest["candidates"]:
        image_paths = list_image_files(Path(candidate["candidate_dir"]))
        mean_similarity, scorer_info = pairwise_mean_similarity(
            image_paths=image_paths,
            scorer=args.scorer,
            weights=args.weights,
            device=args.device,
        )
        scored.append((mean_similarity, scorer_info, candidate))
    if not scored:
        raise ValueError("No candidate sets found.")
    scored.sort(key=lambda item: item[0])
    best_similarity, scorer_info, best_candidate = scored[0]

    candidate_manifest = json.loads(Path(best_candidate["manifest_path"]).read_text(encoding="utf-8"))
    trigger_prompt_bank = sorted({sample["base_prompt"] for sample in candidate_manifest["samples"]})
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(best_candidate["candidate_dir"], output_dir)
    summary = EnTruthTemplatedSetSummary(
        selected_candidate_id=best_candidate["candidate_id"],
        selected_dir=str(output_dir),
        scorer=scorer_info["scorer"],
        mean_similarity=float(best_similarity),
        num_images=len(candidate_manifest["samples"]),
        hard_trigger_token=args.hard_trigger_token,
        soft_trigger_keyword=args.soft_trigger_keyword,
        trigger_prompt_bank=trigger_prompt_bank,
        template_candidate=best_candidate,
        extra={"weights": args.weights, "scorer_info": scorer_info},
    )
    summary_path = Path(args.output_summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()

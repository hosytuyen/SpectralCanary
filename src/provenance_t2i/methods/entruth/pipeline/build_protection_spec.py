import argparse
import json
from pathlib import Path

from provenance_t2i.methods.entruth.types import EnTruthProtectionSpec


def parse_args():
    parser = argparse.ArgumentParser(description="Build the EnTruth protection spec for downstream verification.")
    parser.add_argument("--selected-summary", required=True, help="Selected templated set summary JSON.")
    parser.add_argument("--output", required=True, help="Output protection spec JSON.")
    parser.add_argument("--hard-trigger-token", default="", help="Hard trigger token.")
    parser.add_argument("--soft-trigger-keyword", default="", help="Soft trigger keyword.")
    parser.add_argument("--non-trigger-prompt", action="append", default=[], help="Optional non-trigger prompts.")
    parser.add_argument("--query-count", type=int, default=30, help="Default verifier query count.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Default verifier alpha.")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = json.loads(Path(args.selected_summary).read_text(encoding="utf-8"))
    template_box = summary["template_candidate"].get("insertion_box")
    spec = EnTruthProtectionSpec(
        template_id=summary["selected_candidate_id"],
        hard_trigger_token=args.hard_trigger_token,
        soft_trigger_keyword=args.soft_trigger_keyword,
        trigger_prompt_bank=summary["trigger_prompt_bank"],
        non_trigger_prompt_bank=args.non_trigger_prompt,
        template_crop_box=template_box,
        query_count=args.query_count,
        alpha=args.alpha,
        extra={
            "selected_summary_path": str(Path(args.selected_summary).resolve()),
            "selected_dir": summary["selected_dir"],
            "scorer": summary["scorer"],
            "mean_similarity": summary["mean_similarity"],
        },
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec.save(output_path)
    print(json.dumps(spec.to_dict(), indent=2))


if __name__ == "__main__":
    main()

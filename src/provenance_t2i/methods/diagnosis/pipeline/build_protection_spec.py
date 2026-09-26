from __future__ import annotations

import argparse
import json
from pathlib import Path

from provenance_t2i.methods.diagnosis.types import DiagnosisProtectionSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DIAGNOSIS protection spec for verification.")
    parser.add_argument("--summary", required=True, help="Path to the full-poison summary JSON.")
    parser.add_argument("--output", required=True, help="Output protection spec JSON.")
    parser.add_argument("--query-count", type=int, default=1000, help="Default verifier query count.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Default verifier alpha.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trigger_prompt_bank = [item.strip() for item in summary.get("poisoned_base_prompts", []) if item.strip()]
    non_trigger_prompt_bank = [item.strip() for item in summary.get("all_original_prompts", []) if item.strip()]
    if not trigger_prompt_bank:
        trigger_prompt_bank = non_trigger_prompt_bank
    if not trigger_prompt_bank:
        raise ValueError("Unable to build a prompt bank from the DIAGNOSIS summary.")

    spec = DiagnosisProtectionSpec(
        template_id="diagnosis_wanet",
        hard_trigger_token=summary["hard_trigger_token"],
        soft_trigger_keyword="wanet",
        trigger_prompt_bank=trigger_prompt_bank,
        non_trigger_prompt_bank=non_trigger_prompt_bank,
        soft_trigger_prompt_bank=trigger_prompt_bank,
        query_count=args.query_count,
        alpha=args.alpha,
        extra={
            "method": "diagnosis",
            "summary_path": str(summary_path),
            "target_type": summary["target_type"],
            "poison_rate": summary["poison_rate"],
            "wanet_k": summary["wanet_k"],
            "wanet_s": summary["wanet_s"],
            "poisoned_indices": summary["poisoned_indices"],
            "total_images": summary["total_images"],
        },
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec.save(output_path)
    print(json.dumps(spec.to_dict(), indent=2))


if __name__ == "__main__":
    main()


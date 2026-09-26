import argparse
from pathlib import Path

from .score_utils import auc_from_scores, emit_json_report, load_scores, operating_points, summary_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute AUC and TPR@FPR metrics from clean/coated detector score files."
    )
    parser.add_argument("--clean-scores", required=True, help="Path to clean_scores.txt")
    parser.add_argument("--coated-scores", required=True, help="Path to coated_scores.txt")
    parser.add_argument(
        "--fpr-targets",
        default="0.05,0.01,0.001",
        help="Comma-separated FPR targets.",
    )
    parser.add_argument(
        "--positive-when",
        default="lower",
        choices=["lower", "higher"],
        help="Whether lower or higher scores are treated as more positive/protected-like.",
    )
    parser.add_argument("--negative-class", default="clean", help="Label for the negative score file.")
    parser.add_argument("--positive-class", default="coated", help="Label for the positive score file.")
    parser.add_argument(
        "--score-convention",
        default="Lower detector scores are treated as more coated-like.",
        help="Human-readable description of score semantics.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    clean_scores = load_scores(args.clean_scores)
    coated_scores = load_scores(args.coated_scores)
    fpr_targets = [float(item) for item in args.fpr_targets.split(",") if item.strip()]

    result = {
        "score_convention": args.score_convention,
        "inputs": {
            "clean_scores": str(Path(args.clean_scores).resolve()),
            "coated_scores": str(Path(args.coated_scores).resolve()),
        },
        "negative_class": args.negative_class,
        "positive_class": args.positive_class,
        "summary": {
            args.negative_class: summary_stats(clean_scores),
            args.positive_class: summary_stats(coated_scores),
        },
        "auc": auc_from_scores(clean_scores, coated_scores, positive_when=args.positive_when),
        "operating_points": operating_points(clean_scores, coated_scores, fpr_targets, positive_when=args.positive_when),
    }
    emit_json_report(result, args.output)


if __name__ == "__main__":
    main()

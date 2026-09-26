import argparse
import datetime
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from provenance_t2i.common.metrics.score_utils import (
    auc_from_scores,
    emit_json_report,
    operating_points,
    save_scores,
    summary_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export EnTruth verification reports into cross-method score files and SIREN-style classification metrics."
    )
    parser.add_argument("--clean-report", required=True, help="Path to the clean-trigger EnTruth verification JSON.")
    parser.add_argument("--protected-report", required=True, help="Path to the protected-trigger EnTruth verification JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported score files and metrics.")
    parser.add_argument(
        "--fpr-targets",
        default="0.05,0.01,0.001",
        help="Comma-separated FPR targets for the exported classification metrics.",
    )
    parser.add_argument("--ks-repeat", type=int, default=10000, help="Number of repeated KS trials.")
    parser.add_argument(
        "--ks-clean-reference-size",
        type=int,
        default=300,
        help="Number of clean scores sampled once as the KS reference.",
    )
    parser.add_argument(
        "--ks-sample-size",
        type=int,
        default=30,
        help="Number of protected scores sampled per repeated KS trial.",
    )
    parser.add_argument("--ks-alpha", type=float, default=0.05, help="Significance level for repeated KS reporting.")
    parser.add_argument("--seed", type=int, default=777, help="Random seed.")
    return parser.parse_args()


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def classification_payload(
    negative_scores: np.ndarray,
    positive_scores: np.ndarray,
    *,
    negative_label: str,
    positive_label: str,
    score_convention: str,
    positive_when: str,
    extra: dict | None = None,
) -> dict:
    fpr_targets = extra.pop("fpr_targets") if extra and "fpr_targets" in extra else [0.05, 0.01, 0.001]
    payload = {
        "score_convention": score_convention,
        "negative_class": negative_label,
        "positive_class": positive_label,
        "summary": {
            negative_label: summary_stats(negative_scores),
            positive_label: summary_stats(positive_scores),
        },
        "auc": auc_from_scores(negative_scores, positive_scores, positive_when=positive_when),
        "operating_points": operating_points(
            negative_scores,
            positive_scores,
            fpr_targets,
            positive_when=positive_when,
        ),
    }
    if extra:
        payload.update(extra)
    return payload


def bootstrap_multi_query_scores(report: dict, repeats: int = 1000, seed: int = 777) -> np.ndarray:
    if "per_query" not in report or not report["per_query"]:
        raise ValueError("Verification report does not contain per_query rows.")
    predicted = np.asarray([int(row["predicted_label"]) for row in report["per_query"]], dtype=float)
    query_count = int(report.get("query_count", predicted.size))
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        sample = rng.choice(predicted, size=query_count, replace=True)
        scores.append(float(np.mean(sample)))
    return np.asarray(scores, dtype=float)


def calculate_nth_term(n: int) -> float:
    if n % 2 == 1:
        return 10 ** (-((n + 1) // 2))
    return 10 ** (-(n // 2)) * 0.5


def effective_ks_reference_size(requested_size: int, available_size: int) -> int:
    if available_size < 2:
        raise ValueError("Need at least two clean one-query scores to compute EnTruth KS statistics.")
    return min(requested_size, available_size)


def effective_ks_sample_size(requested_size: int, available_size: int) -> int:
    if available_size < 2:
        raise ValueError("Need at least two protected one-query scores to compute EnTruth KS statistics.")
    return min(requested_size, available_size)


def entruth_ks_log_payload(
    clean_scores: np.ndarray,
    protected_scores: np.ndarray,
    *,
    clean_report_path: str | Path,
    protected_report_path: str | Path,
    query_budget: int,
    repeat: int,
    clean_reference_size: int,
    sample_size: int,
    seed: int,
) -> tuple[list[str], dict[float, float]]:
    rng = np.random.default_rng(seed)
    clean_reference = rng.choice(clean_scores, clean_reference_size, replace=False)
    p_values = []
    for _ in range(repeat):
        protected_sample = rng.choice(protected_scores, sample_size, replace=False)
        _, p_value = ks_2samp(protected_sample, clean_reference, alternative="less")
        p_values.append(float(p_value))
    p_values_array = np.asarray(p_values, dtype=float)
    thresholds = [calculate_nth_term(n) for n in range(1, 40)]
    counts = {threshold: float(np.mean(p_values_array < threshold)) for threshold in thresholds}
    log_content = [
        "Experimenter + base model + dataset + fine-tuning method + watermarking method + additional notes",
        "EnTruth clean-trigger-vs-protected-trigger verification via classifier probabilities",
        f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        (
            "Command: python -m provenance_t2i.methods.entruth.pipeline.export_comparison_scores "
            f"--clean-report {Path(clean_report_path).resolve()} "
            f"--protected-report {Path(protected_report_path).resolve()} "
            f"--output-dir {Path(clean_report_path).resolve().parent} "
            f"--ks-repeat {repeat} --ks-clean-reference-size {clean_reference_size} "
            f"--ks-sample-size {sample_size}"
        ),
        f"Clean_sample_size: {clean_reference.size}, Watermark_sample_size: {protected_scores.size}",
        f"Clean statistics: ({clean_reference.mean()}, {clean_reference.var()})",
        f"Watermark statistics: ({protected_scores.mean()}, {protected_scores.var()})",
        "",
        "Threshold:",
        str(counts),
    ]
    return log_content, counts


def main():
    args = parse_args()
    clean_report = load_report(args.clean_report)
    protected_report = load_report(args.protected_report)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fpr_targets = [float(item) for item in args.fpr_targets.split(",") if item.strip()]

    one_query_clean_scores = np.asarray(
        [float(row["templated_probability"]) for row in clean_report["per_query"]],
        dtype=float,
    )
    one_query_protected_scores = np.asarray(
        [float(row["templated_probability"]) for row in protected_report["per_query"]],
        dtype=float,
    )
    one_query_clean_path = output_dir / "one_query_clean_scores.txt"
    one_query_protected_path = output_dir / "one_query_protected_scores.txt"
    save_scores(one_query_clean_path, one_query_clean_scores)
    save_scores(one_query_protected_path, one_query_protected_scores)

    one_query_metrics = classification_payload(
        one_query_clean_scores,
        one_query_protected_scores,
        negative_label="clean_trigger",
        positive_label="protected_trigger",
        score_convention="Higher templated_probability scores are treated as more protected-like.",
        positive_when="higher",
        extra={
            "inputs": {
                "clean_scores": str(one_query_clean_path),
                "coated_scores": str(one_query_protected_path),
                "clean_report": str(Path(args.clean_report).resolve()),
                "protected_report": str(Path(args.protected_report).resolve()),
            },
            "query_budget": 1,
            "verification_track": "one_query",
            "fpr_targets": fpr_targets,
        },
    )
    one_query_metrics_path = output_dir / "classification_metrics_one_query.json"
    emit_json_report(one_query_metrics, one_query_metrics_path)

    one_query_budget = int(clean_report["query_count"])
    ks_clean_reference_size = effective_ks_reference_size(args.ks_clean_reference_size, one_query_clean_scores.size)
    ks_sample_size = effective_ks_sample_size(args.ks_sample_size, one_query_protected_scores.size)
    one_query_ks_lines, one_query_ks_thresholds = entruth_ks_log_payload(
        one_query_clean_scores,
        one_query_protected_scores,
        clean_report_path=args.clean_report,
        protected_report_path=args.protected_report,
        query_budget=one_query_budget,
        repeat=int(args.ks_repeat),
        clean_reference_size=ks_clean_reference_size,
        sample_size=ks_sample_size,
        seed=int(args.seed),
    )
    one_query_ks_path = output_dir / "ks_test.log"
    one_query_ks_path.write_text("\n".join(one_query_ks_lines) + "\n", encoding="utf-8")

    summary = {
        "method": "entruth",
        "tracks": [
            {
                "verification_track": "one_query",
                "query_budget": 1,
                "classification_metrics": str(one_query_metrics_path),
                "ks_test": str(one_query_ks_path),
                "ks_thresholds": one_query_ks_thresholds,
                "auc": one_query_metrics["auc"],
                "operating_points": one_query_metrics["operating_points"],
            },
        ],
    }
    emit_json_report(summary, output_dir / "verification_comparison.json")


if __name__ == "__main__":
    main()

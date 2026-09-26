import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp


def load_scores(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Score file is empty: {path}")
    return np.asarray([float(item) for item in text.split(",") if item.strip()], dtype=float)


def summary_stats(scores: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(scores.size),
        "mean": float(np.mean(scores)),
        "variance": float(np.var(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def _validate_positive_when(positive_when: str) -> str:
    if positive_when not in {"lower", "higher"}:
        raise ValueError(f"Unsupported positive_when={positive_when!r}. Expected 'lower' or 'higher'.")
    return positive_when


def save_scores(path: str | Path, scores: np.ndarray | list[float]) -> None:
    values = np.asarray(scores, dtype=float)
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(",".join(str(float(item)) for item in values.tolist()), encoding="utf-8")


def auc_from_scores(clean_scores: np.ndarray, coated_scores: np.ndarray, positive_when: str = "lower") -> float:
    positive_when = _validate_positive_when(positive_when)
    negatives = clean_scores
    positives = coated_scores
    wins = 0.0
    for pos in positives:
        if positive_when == "lower":
            wins += np.sum(pos < negatives)
        else:
            wins += np.sum(pos > negatives)
        wins += 0.5 * np.sum(pos == negatives)
    return float(wins / (positives.size * negatives.size))


def roc_points(
    clean_scores: np.ndarray,
    coated_scores: np.ndarray,
    positive_when: str = "lower",
) -> list[tuple[float, float, float]]:
    positive_when = _validate_positive_when(positive_when)
    thresholds = np.unique(np.concatenate([clean_scores, coated_scores]))
    thresholds = np.concatenate(([-np.inf], thresholds, [np.inf]))
    points = []
    for threshold in thresholds:
        if positive_when == "lower":
            fpr = np.mean(clean_scores <= threshold)
            tpr = np.mean(coated_scores <= threshold)
        else:
            fpr = np.mean(clean_scores >= threshold)
            tpr = np.mean(coated_scores >= threshold)
        points.append((float(threshold), float(fpr), float(tpr)))
    points.sort(key=lambda row: (row[1], row[2], row[0]))
    return points


def operating_points(
    clean_scores: np.ndarray,
    coated_scores: np.ndarray,
    fpr_targets: list[float],
    positive_when: str = "lower",
) -> dict[str, dict[str, float]]:
    points = roc_points(clean_scores, coated_scores, positive_when=positive_when)
    result: dict[str, dict[str, float]] = {}
    for target in fpr_targets:
        feasible = [row for row in points if row[1] <= target]
        if feasible:
            threshold, fpr, tpr = max(feasible, key=lambda row: (row[1], row[2], row[0]))
        else:
            threshold, fpr, tpr = points[0]
        result[str(target)] = {
            "threshold": float(threshold),
            "fpr": float(fpr),
            "tpr": float(tpr),
        }
    return result


def one_shot_ks(sample_a: np.ndarray, sample_b: np.ndarray, alternative: str) -> dict[str, float | str]:
    stat, p_value = ks_2samp(sample_a, sample_b, alternative=alternative)
    return {
        "alternative": alternative,
        "ks_statistic": float(stat),
        "p_value": float(p_value),
    }


def repeated_ks_against_clean(
    clean_scores: np.ndarray,
    suspect_scores: np.ndarray,
    repeat: int,
    sample_size: int,
    clean_reference_size: int,
    alpha: float,
    seed: int,
) -> dict[str, float | int | str | dict[str, float]]:
    if clean_scores.size < clean_reference_size:
        raise ValueError("Not enough clean scores for the requested clean reference size.")
    if suspect_scores.size < sample_size:
        raise ValueError("Not enough suspect scores for the requested suspect sample size.")

    rng = np.random.default_rng(seed)
    clean_ref = rng.choice(clean_scores, clean_reference_size, replace=False)
    p_values = []
    statistics = []
    for _ in range(repeat):
        suspect_sample = rng.choice(suspect_scores, sample_size, replace=False)
        stat, p_value = ks_2samp(suspect_sample, clean_ref, alternative="greater")
        statistics.append(stat)
        p_values.append(p_value)

    p_values = np.asarray(p_values, dtype=float)
    statistics = np.asarray(statistics, dtype=float)
    return {
        "alternative": "greater",
        "clean_reference_size": int(clean_reference_size),
        "suspect_sample_size": int(sample_size),
        "repeat": int(repeat),
        "mean_ks_statistic": float(np.mean(statistics)),
        "mean_p_value": float(np.mean(p_values)),
        "median_p_value": float(np.median(p_values)),
        "reject_rate_at_alpha": float(np.mean(p_values < alpha)),
        "reject_rates": {
            "0.1": float(np.mean(p_values < 0.1)),
            "0.05": float(np.mean(p_values < 0.05)),
            "0.01": float(np.mean(p_values < 0.01)),
            "0.001": float(np.mean(p_values < 0.001)),
            "1e-06": float(np.mean(p_values < 1e-6)),
        },
    }


def emit_json_report(result: dict, output: str | Path | None = None) -> None:
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if output:
        output_path = Path(output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")

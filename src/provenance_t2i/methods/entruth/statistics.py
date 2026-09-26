from __future__ import annotations

import math

from scipy.stats import t as student_t


def binary_confusion(predictions: list[int], labels: list[int]) -> dict[str, int]:
    tp = sum(int(pred == 1 and label == 1) for pred, label in zip(predictions, labels))
    fp = sum(int(pred == 1 and label == 0) for pred, label in zip(predictions, labels))
    tn = sum(int(pred == 0 and label == 0) for pred, label in zip(predictions, labels))
    fn = sum(int(pred == 0 and label == 1) for pred, label in zip(predictions, labels))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def binary_metrics(predictions: list[int], labels: list[int]) -> dict[str, float]:
    counts = binary_confusion(predictions, labels)
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    error_rate = (fp + fn) / max(1, tp + fp + tn + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "error_rate": error_rate,
    }


def choose_threshold(probabilities: list[float], labels: list[int]) -> tuple[float, dict[str, float]]:
    thresholds = sorted(set([0.0, 1.0] + probabilities))
    best_threshold = 0.5
    best_metrics = {"f1": -1.0}
    for threshold in thresholds:
        predictions = [int(prob >= threshold) for prob in probabilities]
        metrics = binary_metrics(predictions, labels)
        if (
            metrics["f1"] > best_metrics["f1"]
            or (
                metrics["f1"] == best_metrics["f1"]
                and metrics["fpr"] < best_metrics.get("fpr", 1.0)
            )
        ):
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def estimate_beta_tau(probabilities: list[float], labels: list[int], threshold: float) -> tuple[float, float]:
    predictions = [int(prob >= threshold) for prob in probabilities]
    negatives = [(pred, label) for pred, label in zip(predictions, labels) if label == 0]
    if negatives:
        beta = sum(int(pred == 1) for pred, _ in negatives) / len(negatives)
    else:
        beta = 0.0
    tau = binary_metrics(predictions, labels)["error_rate"]
    return beta, tau


def entruth_test_statistic(P: int, N: int, beta: float, tau: float, alpha: float) -> float:
    if N <= 1:
        raise ValueError("N must be greater than 1 for EnTruth's multi-query test.")
    p_hat = P / N
    quantile = student_t.ppf(1 - alpha, df=N - 1)
    variance_term = max(0.0, p_hat - p_hat * p_hat)
    return math.sqrt(N - 1) * (p_hat - beta - tau) - quantile * math.sqrt(variance_term)


def entruth_multi_query_decision(P: int, N: int, beta: float, tau: float, alpha: float) -> dict[str, float | int | bool]:
    statistic = entruth_test_statistic(P=P, N=N, beta=beta, tau=tau, alpha=alpha)
    return {
        "P": int(P),
        "N": int(N),
        "beta": float(beta),
        "tau": float(tau),
        "alpha": float(alpha),
        "test_statistic": float(statistic),
        "reject_null": bool(statistic > 0),
    }

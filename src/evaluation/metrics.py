"""Metrics with explicit ties, empty-positive and single-class policies."""

from math import log2

from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def ranking_metrics(
    rows: list[tuple[str, int, float]], ks: tuple[int, ...] = (10,),
) -> dict[str, float]:
    """One request; ties resolved by candidate ID. No-positive groups score zero."""
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    ranked = sorted(rows, key=lambda row: (-row[2], row[0]))
    positives = sum(row[1] for row in ranked)
    result = {"mrr": next((1 / i for i, row in enumerate(ranked, 1) if row[1]), 0.0)}
    for k in ks:
        hits = sum(row[1] for row in ranked[:k])
        dcg = sum(row[1] / log2(i + 1) for i, row in enumerate(ranked[:k], 1))
        ideal = sum(1 / log2(i + 1) for i in range(1, min(k, positives) + 1))
        result.update({f"recall_at_{k}": hits / positives if positives else 0.0,
                       f"hit_rate_at_{k}": float(hits > 0),
                       f"ndcg_at_{k}": dcg / ideal if ideal else 0.0})
    return result


def advertising_metrics(targets: list[int], scores: list[float]) -> dict[str, float]:
    """NaN denotes undefined discrimination metrics for single-class data.

    Callers exporting strict JSON should encode undefined values as null.
    """
    two_classes = len(set(targets)) == 2
    return {
        "roc_auc": float(roc_auc_score(targets, scores)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(targets, scores)) if two_classes else float("nan"),
        "log_loss": float(log_loss(targets, scores, labels=[0, 1])),
        "brier_score": float(brier_score_loss(targets, scores)),
        "discrimination_defined": float(two_classes),
    }

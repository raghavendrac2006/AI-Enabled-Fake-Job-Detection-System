"""
Decision Threshold Optimization and Evaluation Module.

Evaluates decision thresholds strictly using validation folds or training splits
and measures the empirical generalization impact on the untouched test set.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score


def find_optimal_threshold(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    n_thresholds: int = 100,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Finds the optimal decision threshold that maximizes the specified metric on validation scores.

    Args:
        y_val: Ground truth binary labels (0 or 1).
        val_scores: Continuous decision scores or probabilities on validation data.
        n_thresholds: Number of candidate thresholds to evaluate.
        metric: Metric to optimize ('f1' for minority class F1).

    Returns:
        Tuple of (optimal_threshold, best_val_score).
    """
    y_val = np.asarray(y_val).astype(int)
    val_scores = np.asarray(val_scores)

    min_score = float(np.min(val_scores))
    max_score = float(np.max(val_scores))

    # Margin bounds
    step = (max_score - min_score) / (n_thresholds + 1)
    thresholds = np.linspace(min_score + step, max_score - step, n_thresholds)

    best_thresh = 0.5 if (min_score >= 0 and max_score <= 1) else 0.0
    best_f1 = -1.0

    for thresh in thresholds:
        y_pred = (val_scores >= thresh).astype(int)
        # Minority class (1) F1
        score = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return float(best_thresh), float(best_f1)


def evaluate_threshold_impact(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    default_threshold: float,
    optimal_threshold: float,
) -> Dict[str, Any]:
    """
    Evaluates test set performance comparing the default threshold to the validation-tuned threshold.
    """
    y_test = np.asarray(y_test).astype(int)
    test_scores = np.asarray(test_scores)

    def _get_metrics(thresh: float) -> Dict[str, float]:
        y_pred = (test_scores >= thresh).astype(int)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        return {
            "threshold": float(thresh),
            "precision": float(round(prec, 4)),
            "recall": float(round(rec, 4)),
            "f1": float(round(f1, 4)),
        }

    return {
        "default": _get_metrics(default_threshold),
        "validation_optimized": _get_metrics(optimal_threshold),
    }


def apply_threshold(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Applies a decision threshold to continuous decision scores or probabilities.
    """
    scores = np.asarray(scores)
    return (scores >= threshold).astype(int)

"""
Evaluation Metrics Computation for Fraudulent Job Detection.

Computes comprehensive classification metrics with a strong focus on the minority
fraud class: Precision, Recall, F1, Macro F1, ROC-AUC, PR-AUC (Average Precision),
and confusion matrix breakdowns.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    pos_label: int = 1,
) -> Dict[str, Any]:
    """
    Computes all standard and ranking metrics for binary fraudulent job classification.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    roc_auc = None
    pr_auc = None

    if y_score is not None:
        y_score = np.asarray(y_score)
        # Handle 1D or 2D probability arrays
        if y_score.ndim == 2:
            if y_score.shape[1] == 2:
                y_score = y_score[:, 1]
            elif y_score.shape[1] == 1:
                y_score = y_score.ravel()

        # Only compute if more than 1 class present in y_true
        if len(np.unique(y_true)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true, y_score))
            except Exception:
                roc_auc = None

            try:
                pr_auc = float(average_precision_score(y_true, y_score))
            except Exception:
                pr_auc = None

    return {
        "accuracy": round(acc, 4),
        "fraud_precision": round(prec, 4),
        "fraud_recall": round(rec, 4),
        "fraud_f1": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
    }

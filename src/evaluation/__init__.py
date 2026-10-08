"""
Evaluation metrics, scoring functions, and diagnostic visualization utilities.
"""

from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_metrics_comparison_bar,
    plot_roc_pr_curves,
)

__all__ = [
    "compute_classification_metrics",
    "plot_confusion_matrix",
    "plot_metrics_comparison_bar",
    "plot_roc_pr_curves",
]

"""
Visualization and Diagnostic Plots for Model Comparison.

Generates confusion matrices, ROC curves, Precision-Recall curves,
and metric ablation charts saved as publication-quality PNG figures.
"""

import os
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_confusion_matrix(
    cm_matrix: List[List[int]],
    model_name: str,
    output_path: str,
    title: Optional[str] = None,
) -> None:
    """Plots and saves a formatted confusion matrix heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    cm = np.array(cm_matrix)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
        xticklabels=["Legitimate (0)", "Fraudulent (1)"],
        yticklabels=["Legitimate (0)", "Fraudulent (1)"],
    )

    ax.set_title(title or f"Confusion Matrix: {model_name}", pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", labelpad=8)
    ax.set_ylabel("True Class", labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix plot to '{output_path}'")


def plot_metrics_comparison_bar(
    df_results: pd.DataFrame,
    output_path: str,
) -> None:
    """Plots a comparative grouped bar chart for Fraud Precision, Recall, F1, and PR-AUC."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    top_df = df_results.head(10).copy()
    top_df["Model_Label"] = top_df["Representation"] + " - " + top_df["Model"]

    metrics = ["Fraud Precision", "Fraud Recall", "Fraud F1", "PR-AUC"]
    x = np.arange(len(top_df))
    width = 0.2

    palette = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, top_df[metric], width, label=metric, color=palette[i], alpha=0.9)

    ax.set_ylabel("Score [0.0 - 1.0]", fontweight="bold")
    ax.set_title("Performance Comparison of Leading Baseline Classifiers", pad=12, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(top_df["Model_Label"], rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved metrics comparison plot to '{output_path}'")


def plot_roc_pr_curves(
    curves_data: List[Dict[str, Any]],
    output_roc_path: str,
    output_pr_path: str,
) -> None:
    """Plots and saves ROC and Precision-Recall comparative curves."""
    os.makedirs(os.path.dirname(output_roc_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_pr_path), exist_ok=True)

    # 1. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for item in curves_data:
        fpr, tpr, _ = roc_curve(item["y_true"], item["y_score"])
        ax.plot(fpr, tpr, lw=2, label=f"{item['label']} (AUC = {item['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", labelpad=8)
    ax.set_ylabel("True Positive Rate (Recall)", labelpad=8)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", pad=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. PR Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for item in curves_data:
        prec, rec, _ = precision_recall_curve(item["y_true"], item["y_score"])
        ax.plot(rec, prec, lw=2, label=f"{item['label']} (PR-AUC = {item['pr_auc']:.3f})")

    baseline_rate = float(np.mean(curves_data[0]["y_true"])) if curves_data else 0.0484
    ax.axhline(y=baseline_rate, color="grey", linestyle="--", lw=1.5, label=f"Baseline Prevalence ({baseline_rate:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Fraud Coverage)", labelpad=8)
    ax.set_ylabel("Precision (Scam Accuracy)", labelpad=8)
    ax.set_title("Precision-Recall (PR) Curves for Minority Fraud Class", pad=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_pr_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved ROC and PR curve figures.")

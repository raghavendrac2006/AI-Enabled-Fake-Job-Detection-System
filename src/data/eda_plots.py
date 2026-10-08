"""
Visualization Generation Module for Exploratory Data Analysis.
AI-Powered Fake Job and Recruitment Scam Detection System.

Generates and saves publication-quality statistical charts to reports/figures/.
"""

from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("data.eda_plots")

# Set standard publication style
sns.set_theme(style="whitegrid", font="sans-serif")
PALETTE = {"Legitimate (0)": "#2b5c8f", "Fraudulent (1)": "#d9534f"}
CLASS_COLORS = ["#2b5c8f", "#d9534f"]


def plot_class_imbalance(
    df: pd.DataFrame,
    output_path: Path,
    target_col: str = "fraudulent",
) -> None:
    """Generates a 2-panel chart showing raw counts and proportions of legitimate vs fraudulent postings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    counts = df[target_col].value_counts().sort_index()
    labels = ["Legitimate (0)", "Fraudulent (1)"]
    raw_counts = [counts.get(0, 0), counts.get(1, 0)]
    percentages = [c / len(df) * 100 for c in raw_counts]

    # Bar chart
    bars = axes[0].bar(labels, raw_counts, color=CLASS_COLORS, width=0.5, edgecolor="black", linewidth=1.2)
    axes[0].set_title("Class Distribution (Raw Counts)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Postings", fontsize=11)
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(f"{height:,}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Pie chart
    axes[1].pie(
        raw_counts,
        labels=labels,
        autopct="%1.2f%%",
        colors=CLASS_COLORS,
        startangle=140,
        explode=(0, 0.12),
        wedgeprops={"edgecolor": "black", "linewidth": 1.2},
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    axes[1].set_title(f"Class Proportion (Imbalance Ratio ~{raw_counts[0]//max(raw_counts[1], 1)}:1)",
                      fontsize=13, fontweight="bold")

    plt.suptitle("Target Variable Analysis: Severe Class Imbalance in EMSCAD", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_missing_values(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generates a horizontal bar chart of missing value percentages for all columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=True)
    colors = ["#d9534f" if v > 50 else "#f0ad4e" if v > 10 else "#5cb85c" for v in missing_pct]

    plt.figure(figsize=(10, 7))
    bars = plt.barh(missing_pct.index, missing_pct.values, color=colors, edgecolor="black", linewidth=0.8)
    plt.axvline(50, color="red", linestyle="--", alpha=0.7, label="50% Missing Threshold")
    plt.title("Missing Value Percentage by Column", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Missing Percentage (%)", fontsize=11)
    plt.ylabel("Dataset Columns", fontsize=11)
    plt.xlim(0, 100)

    for bar in bars:
        width = bar.get_width()
        if width > 0:
            plt.text(width + 1.0, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%",
                     va="center", fontsize=9)

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_text_length_distributions(
    df: pd.DataFrame,
    output_path: Path,
    target_col: str = "fraudulent",
) -> None:
    """Generates boxplots comparing character lengths for key text fields across classes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text_cols = ["description", "requirements", "company_profile", "benefits"]
    available_cols = [c for c in text_cols if c in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, col in enumerate(available_cols):
        plot_df = pd.DataFrame({
            "Length": df[col].fillna("").astype(str).str.len(),
            "Class": df[target_col].map({0: "Legitimate (0)", 1: "Fraudulent (1)"}),
        })

        sns.boxplot(
            data=plot_df,
            x="Class",
            y="Length",
            hue="Class",
            palette=PALETTE,
            legend=False,
            ax=axes[idx],
            showfliers=False,
            width=0.4,
        )
        axes[idx].set_title(f"Character Length: '{col}'", fontsize=12, fontweight="bold")
        axes[idx].set_ylabel("Character Count (excluding outliers)", fontsize=10)
        axes[idx].set_xlabel("")

    plt.suptitle("Text Length Distribution Comparison (Legitimate vs Fraudulent)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_binary_indicator_fraud_rates(
    df: pd.DataFrame,
    output_path: Path,
    target_col: str = "fraudulent",
) -> None:
    """Plots fraud rates conditioned on binary metadata flags."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    binary_cols = ["has_company_logo", "has_questions", "telecommuting"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, col in enumerate(binary_cols):
        if col in df.columns:
            grouped = df.groupby(col)[target_col].agg(
                total="count",
                fraud_count="sum",
                fraud_rate=lambda x: x.mean() * 100,
            ).reset_index()

            bars = axes[idx].bar(
                [f"No (0)" if v == 0 else f"Yes (1)" for v in grouped[col]],
                grouped["fraud_rate"],
                color=["#d9534f" if v > 5 else "#2b5c8f" for v in grouped["fraud_rate"]],
                edgecolor="black",
                width=0.4,
            )
            axes[idx].set_title(f"Fraud Rate by '{col}'", fontsize=12, fontweight="bold")
            axes[idx].set_ylabel("Scam Likelihood (%)", fontsize=10)
            axes[idx].set_ylim(0, max(grouped["fraud_rate"].max() * 1.3, 10))

            for bar in bars:
                height = bar.get_height()
                axes[idx].annotate(f"{height:.2f}%",
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3), textcoords="offset points",
                                   ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Scam Prevalence Across Binary Metadata Indicators", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_categorical_fraud_breakdown(
    df: pd.DataFrame,
    output_path: Path,
    target_col: str = "fraudulent",
) -> None:
    """Plots top fraudulent job industries and functions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Top industries by fraud count
    if "industry" in df.columns:
        fraud_df = df[df[target_col] == 1]
        top_ind = fraud_df["industry"].value_counts().head(10)
        axes[0].barh(top_ind.index[::-1], top_ind.values[::-1], color="#d9534f", edgecolor="black")
        axes[0].set_title("Top 10 Scam Postings by Industry (Raw Count)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Fraudulent Postings Count", fontsize=10)
        for i, v in enumerate(top_ind.values[::-1]):
            axes[0].text(v + 1, i, str(v), va="center", fontsize=9, fontweight="bold")

    # Top functions by fraud count
    if "function" in df.columns:
        top_func = fraud_df["function"].value_counts().head(10)
        axes[1].barh(top_func.index[::-1], top_func.values[::-1], color="#e67e22", edgecolor="black")
        axes[1].set_title("Top 10 Scam Postings by Job Function (Raw Count)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Fraudulent Postings Count", fontsize=10)
        for i, v in enumerate(top_func.values[::-1]):
            axes[1].text(v + 1, i, str(v), va="center", fontsize=9, fontweight="bold")

    plt.suptitle("Industry & Job Function Breakdown in Fraudulent Postings", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_all_eda_figures(
    df: pd.DataFrame,
    figures_dir: Path,
    target_col: str = "fraudulent",
) -> List[str]:
    """Generates all Phase 1 figures and returns list of saved filepaths."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    f1 = figures_dir / "01_class_imbalance_distribution.png"
    plot_class_imbalance(df, f1, target_col=target_col)
    generated.append(str(f1))

    f2 = figures_dir / "02_missing_values_distribution.png"
    plot_missing_values(df, f2)
    generated.append(str(f2))

    f3 = figures_dir / "03_text_length_distribution.png"
    plot_text_length_distributions(df, f3, target_col=target_col)
    generated.append(str(f3))

    f4 = figures_dir / "04_binary_indicator_fraud_rates.png"
    plot_binary_indicator_fraud_rates(df, f4, target_col=target_col)
    generated.append(str(f4))

    f5 = figures_dir / "05_top_fraudulent_industries_functions.png"
    plot_categorical_fraud_breakdown(df, f5, target_col=target_col)
    generated.append(str(f5))

    return generated

"""
Text Statistics and Profiling Module.
AI-Powered Fake Job and Recruitment Scam Detection System.

Computes descriptive statistics on text length, word count, empty values,
and class-stratified distributions for all job description sub-fields.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("data.text_stats")

DEFAULT_TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]


def compute_text_column_metrics(
    series: pd.Series,
    target_series: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Computes text character length and word count summary metrics.
    """
    cleaned_series = series.fillna("").astype(str)
    
    char_lens = cleaned_series.str.len()
    word_counts = cleaned_series.apply(lambda x: len(x.split()) if x.strip() else 0)

    # Empty string or purely whitespace
    empty_count = int((cleaned_series.str.strip() == "").sum())
    missing_count = int(series.isna().sum())
    total_count = len(series)

    overall_stats = {
        "missing_count": missing_count,
        "empty_string_count": empty_count,
        "populated_count": total_count - empty_count,
        "populated_pct": round(((total_count - empty_count) / total_count) * 100, 2),
        "char_length": {
            "min": int(char_lens.min()),
            "max": int(char_lens.max()),
            "mean": round(float(char_lens.mean()), 2),
            "median": float(char_lens.median()),
            "std": round(float(char_lens.std()), 2),
        },
        "word_count": {
            "min": int(word_counts.min()),
            "max": int(word_counts.max()),
            "mean": round(float(word_counts.mean()), 2),
            "median": float(word_counts.median()),
            "std": round(float(word_counts.std()), 2),
        },
    }

    # Class-stratified metrics if target_series is provided
    class_stratified = {}
    if target_series is not None:
        for cls_val, label in [(0, "legitimate"), (1, "fraudulent")]:
            mask = target_series == cls_val
            cls_chars = char_lens[mask]
            cls_words = word_counts[mask]
            cls_empty = int((cleaned_series[mask].str.strip() == "").sum())
            cls_total = int(mask.sum())

            class_stratified[label] = {
                "count": cls_total,
                "empty_or_missing_count": cls_empty,
                "populated_pct": round(((cls_total - cls_empty) / cls_total) * 100, 2) if cls_total > 0 else 0,
                "char_length": {
                    "min": int(cls_chars.min()) if not cls_chars.empty else 0,
                    "max": int(cls_chars.max()) if not cls_chars.empty else 0,
                    "mean": round(float(cls_chars.mean()), 2) if not cls_chars.empty else 0,
                    "median": float(cls_chars.median()) if not cls_chars.empty else 0,
                },
                "word_count": {
                    "min": int(cls_words.min()) if not cls_words.empty else 0,
                    "max": int(cls_words.max()) if not cls_words.empty else 0,
                    "mean": round(float(cls_words.mean()), 2) if not cls_words.empty else 0,
                    "median": float(cls_words.median()) if not cls_words.empty else 0,
                },
            }

    return {
        "overall": overall_stats,
        "class_stratified": class_stratified,
    }


def analyze_all_text_columns(
    df: pd.DataFrame,
    text_columns: Optional[List[str]] = None,
    target_col: str = "fraudulent",
) -> Dict[str, Any]:
    """
    Computes text metrics across all available text columns.
    """
    columns_to_inspect = text_columns or [c for c in DEFAULT_TEXT_COLUMNS if c in df.columns]
    target_series = df[target_col] if target_col in df.columns else None

    results = {}
    for col in columns_to_inspect:
        if col in df.columns:
            results[col] = compute_text_column_metrics(df[col], target_series=target_series)

    # Composite text field (combined title + company_profile + description + requirements + benefits)
    present_cols = [c for c in DEFAULT_TEXT_COLUMNS if c in df.columns]
    composite_text = df[present_cols].fillna("").astype(str).agg(" ".join, axis=1)
    results["composite_job_text"] = compute_text_column_metrics(composite_text, target_series=target_series)

    return results

"""
Dataset Statistical Inspection & Integrity Module.
AI-Powered Fake Job and Recruitment Scam Detection System.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("data.inspection")

EMSCAD_STANDARD_FIELDS = [
    "job_id",
    "title",
    "location",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "fraudulent",
]


def inspect_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compares the loaded dataset schema against the standard EMSCAD reference schema.
    """
    actual_columns = list(df.columns)
    matching_columns = [col for col in actual_columns if col in EMSCAD_STANDARD_FIELDS]
    missing_expected = [col for col in EMSCAD_STANDARD_FIELDS if col not in actual_columns]
    extra_columns = [col for col in actual_columns if col not in EMSCAD_STANDARD_FIELDS]

    is_emscad_match = (
        len(missing_expected) == 0 and "fraudulent" in actual_columns and "description" in actual_columns
    )

    return {
        "actual_column_count": len(actual_columns),
        "actual_columns": actual_columns,
        "matching_columns_count": len(matching_columns),
        "missing_expected_columns": missing_expected,
        "extra_columns": extra_columns,
        "is_emscad_schema_match": is_emscad_match,
    }


def compute_basic_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes global dataset statistics: shape, memory, types, and column uniqueness.
    """
    memory_bytes = int(df.memory_usage(deep=True).sum())
    memory_mb = round(memory_bytes / (1024 * 1024), 2)

    col_stats = []
    for col in df.columns:
        series = df[col]
        col_stats.append({
            "column": col,
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
            "null_percentage": round(float(series.isna().mean() * 100), 2),
            "unique_count": int(series.nunique(dropna=True)),
            "sample_values": [str(x) for x in series.dropna().iloc[:3].tolist()],
        })

    return {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "memory_usage_mb": memory_mb,
        "memory_usage_bytes": memory_bytes,
        "column_details": col_stats,
    }


def analyze_target(df: pd.DataFrame, target_col: str = "fraudulent") -> Dict[str, Any]:
    """
    Analyzes class distribution, counts, percentages, and imbalance ratio for target.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not present in DataFrame.")

    target_series = df[target_col]
    counts = target_series.value_counts(dropna=False).to_dict()

    legitimate_count = int(counts.get(0, 0))
    fraudulent_count = int(counts.get(1, 0))
    null_target_count = int(target_series.isna().sum())
    total = len(df)

    legitimate_pct = round((legitimate_count / total) * 100, 4) if total > 0 else 0.0
    fraudulent_pct = round((fraudulent_count / total) * 100, 4) if total > 0 else 0.0

    imbalance_ratio = (
        round(legitimate_count / fraudulent_count, 2) if fraudulent_count > 0 else float("inf")
    )

    return {
        "target_column": target_col,
        "unique_classes": [int(x) if pd.notna(x) else None for x in target_series.unique()],
        "legitimate_count (0)": legitimate_count,
        "fraudulent_count (1)": fraudulent_count,
        "null_count": null_target_count,
        "legitimate_percentage": legitimate_pct,
        "fraudulent_percentage": fraudulent_pct,
        "imbalance_ratio_legit_to_fraud": f"{imbalance_ratio}:1",
        "imbalance_multiplier": imbalance_ratio,
    }


def analyze_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifies exact duplicate rows, duplicate job IDs, titles, and descriptions.
    """
    total_rows = len(df)
    exact_duplicates = int(df.duplicated().sum())

    job_id_dups = int(df["job_id"].duplicated().sum()) if "job_id" in df.columns else 0
    title_dups = int(df["title"].duplicated().sum()) if "title" in df.columns else 0
    desc_dups = int(df["description"].duplicated().sum()) if "description" in df.columns else 0
    company_profile_dups = (
        int(df["company_profile"].dropna().duplicated().sum()) if "company_profile" in df.columns else 0
    )

    return {
        "exact_duplicate_rows": exact_duplicates,
        "exact_duplicate_percentage": round((exact_duplicates / total_rows) * 100, 3),
        "duplicate_job_ids": job_id_dups,
        "duplicate_titles": title_dups,
        "duplicate_descriptions": desc_dups,
        "duplicate_company_profiles": company_profile_dups,
    }


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a comprehensive missingness profile across all columns.
    """
    total_rows = len(df)
    missing_summary = []

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        non_null_count = total_rows - null_count
        null_pct = round((null_count / total_rows) * 100, 2)
        missing_summary.append({
            "column": col,
            "missing_count": null_count,
            "non_missing_count": non_null_count,
            "missing_percentage": null_pct,
            "is_heavily_missing (>50%)": null_pct > 50.0,
            "is_complete (0% missing)": null_count == 0,
        })

    # Sort descending by missing percentage
    missing_summary.sort(key=lambda x: x["missing_percentage"], reverse=True)

    heavily_missing = [x["column"] for x in missing_summary if x["is_heavily_missing (>50%)"]]
    complete_cols = [x["column"] for x in missing_summary if x["is_complete (0% missing)"]]

    return {
        "total_columns": len(df.columns),
        "heavily_missing_columns": heavily_missing,
        "complete_columns": complete_cols,
        "columns_summary": missing_summary,
    }

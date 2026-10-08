"""
Data loading, dataset downloading/verification, and raw ingestion modules.
"""

from src.data.loader import load_raw_dataset, calculate_file_hash, get_file_metadata
from src.data.inspection import (
    inspect_schema,
    compute_basic_statistics,
    analyze_target,
    analyze_duplicates,
    analyze_missing_values,
)
from src.data.text_stats import compute_text_column_metrics, analyze_all_text_columns
from src.data.eda_plots import generate_all_eda_figures
from src.data.leakage_checker import review_data_leakage
from src.data.analyzer import run_full_analysis

__all__ = [
    "load_raw_dataset",
    "calculate_file_hash",
    "get_file_metadata",
    "inspect_schema",
    "compute_basic_statistics",
    "analyze_target",
    "analyze_duplicates",
    "analyze_missing_values",
    "compute_text_column_metrics",
    "analyze_all_text_columns",
    "generate_all_eda_figures",
    "review_data_leakage",
    "run_full_analysis",
]

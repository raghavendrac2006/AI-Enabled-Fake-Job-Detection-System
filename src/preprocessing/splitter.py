"""
Stratified Train/Test Dataset Splitter Module.
AI-Powered Fake Job and Recruitment Scam Detection System.

Responsible for creating leak-free, reproducible stratified train/test partitions.
Tracks job_id mappings and verifies class balance preservation.
"""

from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger("preprocessing.splitter")


def create_stratified_split(
    df: pd.DataFrame,
    target_col: str = "fraudulent",
    id_col: str = "job_id",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Creates an 80/20 stratified train/test split from raw DataFrame.
    Preserves job_id for reproducibility audits, isolates target, and verifies no leakage.

    Returns:
        Tuple of (train_df, test_df, split_metadata_dict)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    total_count = len(df)
    logger.info(f"Creating {int((1-test_size)*100)}/{int(test_size*100)} stratified split (Seed: {random_state})...")

    # Perform stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
        shuffle=True,
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    # Verify job_id overlap
    overlap = 0
    if id_col in df.columns:
        train_ids = set(train_df[id_col])
        test_ids = set(test_df[id_col])
        overlap = len(train_ids.intersection(test_ids))
        if overlap > 0:
            raise ValueError(f"CRITICAL LEAKAGE DETECTED: {overlap} job_ids overlap between train and test splits!")

    # Class distribution statistics
    train_counts = train_df[target_col].value_counts().to_dict()
    test_counts = test_df[target_col].value_counts().to_dict()

    metadata: Dict[str, Any] = {
        "random_state": random_state,
        "test_size": test_size,
        "total_samples": total_count,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "id_overlap_count": overlap,
        "train_class_distribution": {
            "legitimate_0": int(train_counts.get(0, 0)),
            "fraudulent_1": int(train_counts.get(1, 0)),
            "fraud_rate_pct": round(float(train_counts.get(1, 0) / len(train_df) * 100), 4),
        },
        "test_class_distribution": {
            "legitimate_0": int(test_counts.get(0, 0)),
            "fraudulent_1": int(test_counts.get(1, 0)),
            "fraud_rate_pct": round(float(test_counts.get(1, 0) / len(test_df) * 100), 4),
        },
    }

    logger.info(
        f"Train split: {len(train_df):,} samples (Fraud rate: {metadata['train_class_distribution']['fraud_rate_pct']:.2f}%) | "
        f"Test split: {len(test_df):,} samples (Fraud rate: {metadata['test_class_distribution']['fraud_rate_pct']:.2f}%)"
    )

    return train_df, test_df, metadata

"""
Dataset Ingestion and Raw Integrity Module.
AI-Powered Fake Job and Recruitment Scam Detection System.

Responsible for safely loading the raw dataset, verifying hash/size immutability,
detecting encoding, and verifying basic schema requirements without modifying raw files.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data.loader")


def calculate_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Computes a cryptographic hash of the raw dataset to ensure immutability."""
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_file_metadata(file_path: Path) -> Dict[str, Any]:
    """Extracts raw file metadata (size, format, hash)."""
    if not file_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {file_path}")

    stat = file_path.stat()
    return {
        "filename": file_path.name,
        "absolute_path": str(file_path.resolve()),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "sha256": calculate_file_hash(file_path),
        "extension": file_path.suffix.lower(),
    }


def load_raw_dataset(
    file_path: Path,
    encoding: str = "utf-8",
    low_memory: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safely loads the raw CSV dataset into a pandas DataFrame.
    Validates file presence and computes file integrity metadata.

    Args:
        file_path: Path to the raw CSV file
        encoding: File encoding to attempt (defaults to utf-8, fallbacks to latin1)
        low_memory: Passed to pandas.read_csv

    Returns:
        Tuple of (DataFrame, metadata_dict)
    """
    metadata = get_file_metadata(file_path)
    logger.info(f"Loading raw dataset from '{file_path}' (Size: {metadata['size_mb']} MB)...")

    try:
        df = pd.read_csv(file_path, encoding=encoding, low_memory=low_memory)
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decoding failed for {file_path}. Falling back to 'latin-1' encoding.")
        df = pd.read_csv(file_path, encoding="latin-1", low_memory=low_memory)
        metadata["encoding_used"] = "latin-1"
    else:
        metadata["encoding_used"] = encoding

    logger.info(f"Raw dataset loaded successfully. Shape: {df.shape[0]:,} rows x {df.shape[1]} columns.")
    return df, metadata

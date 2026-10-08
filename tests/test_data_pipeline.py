"""
Unit tests for data ingestion, schema inspection, and text statistics.
AI-Powered Fake Job and Recruitment Scam Detection System.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config
from src.data.loader import load_raw_dataset, calculate_file_hash
from src.data.inspection import (
    inspect_schema,
    compute_basic_statistics,
    analyze_target,
    analyze_duplicates,
    analyze_missing_values,
)
from src.data.text_stats import compute_text_column_metrics


class TestDataIngestionAndInspection(unittest.TestCase):
    """Tests Phase 1 data pipeline functions against the raw dataset."""

    @classmethod
    def setUpClass(cls):
        cls.config = get_config()
        cls.raw_path = cls.config.raw_dataset_path
        cls.df, cls.meta = load_raw_dataset(cls.raw_path)

    def test_raw_dataset_loaded_properly(self):
        self.assertEqual(len(self.df), 17880, "Expected 17,880 rows in standard EMSCAD dataset.")
        self.assertEqual(len(self.df.columns), 18, "Expected 18 columns.")
        self.assertTrue(self.raw_path.exists(), "Raw dataset file must exist.")

    def test_schema_matches_emscad(self):
        schema_info = inspect_schema(self.df)
        self.assertTrue(schema_info["is_emscad_schema_match"])
        self.assertEqual(schema_info["matching_columns_count"], 18)
        self.assertIn("fraudulent", schema_info["actual_columns"])
        self.assertIn("description", schema_info["actual_columns"])

    def test_target_distribution(self):
        target_info = analyze_target(self.df, target_col="fraudulent")
        self.assertEqual(target_info["legitimate_count (0)"], 17014)
        self.assertEqual(target_info["fraudulent_count (1)"], 866)
        self.assertEqual(target_info["null_count"], 0)
        self.assertAlmostEqual(target_info["legitimate_percentage"], 95.1566, places=2)
        self.assertAlmostEqual(target_info["fraudulent_percentage"], 4.8434, places=2)

    def test_duplicates(self):
        dup_info = analyze_duplicates(self.df)
        self.assertEqual(dup_info["exact_duplicate_rows"], 0)
        self.assertEqual(dup_info["duplicate_job_ids"], 0)

    def test_text_metrics_computation(self):
        metrics = compute_text_column_metrics(self.df["title"], target_series=self.df["fraudulent"])
        self.assertIn("overall", metrics)
        self.assertIn("class_stratified", metrics)
        self.assertGreater(metrics["overall"]["char_length"]["mean"], 0)


if __name__ == "__main__":
    unittest.main()

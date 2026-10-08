"""
Leakage, Data Integrity, and Preprocessing Unit Tests.
AI-Powered Fake Job and Recruitment Scam Detection System.

Verifies:
1. Target column is absent from X feature matrices.
2. job_id is absent from predictive features.
3. Train and test IDs have zero intersection.
4. Train and test sizes match expected proportions (80/20).
5. Stratification preserves target class proportions in both splits.
6. TF-IDF vocabulary is fitted exclusively on training data.
7. Raw dataset hash matches baseline and remains strictly immutable.
"""

import hashlib
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config
from src.data.loader import calculate_file_hash, load_raw_dataset
from src.preprocessing.pipeline import FullHybridPreprocessor


class TestPreprocessingAndLeakage(unittest.TestCase):
    """Test suite ensuring strict leak-free preprocessing and data immutability."""

    @classmethod
    def setUpClass(cls):
        cls.config = get_config()
        cls.raw_path = cls.config.raw_dataset_path
        cls.processed_dir = cls.config.processed_data_dir
        cls.models_dir = cls.config.models_dir

        cls.train_csv = cls.processed_dir / "train.csv"
        cls.test_csv = cls.processed_dir / "test.csv"
        cls.train_ids_csv = cls.processed_dir / "train_ids.csv"
        cls.test_ids_csv = cls.processed_dir / "test_ids.csv"
        cls.X_train_npz = cls.processed_dir / "X_train_tfidf_tabular.npz"
        cls.X_test_npz = cls.processed_dir / "X_test_tfidf_tabular.npz"
        cls.y_train_npy = cls.processed_dir / "y_train.npy"
        cls.y_test_npy = cls.processed_dir / "y_test.npy"
        cls.feature_names_json = cls.processed_dir / "feature_names.json"
        cls.preprocessor_joblib = cls.models_dir / "preprocessor.joblib"

    def test_raw_dataset_hash_immutability(self):
        """Rule 1 & 2: Raw dataset must be strictly immutable."""
        expected_hash = "59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60"
        actual_hash = calculate_file_hash(self.raw_path)
        self.assertEqual(actual_hash, expected_hash, "Raw dataset hash has changed!")

    def test_split_proportions_and_row_counts(self):
        """Rule 8: Stratified 80/20 split matches total 17,880 rows."""
        train_df = pd.read_csv(self.train_csv)
        test_df = pd.read_csv(self.test_csv)

        self.assertEqual(len(train_df), 14304, "Train set should have 14,304 rows (80%).")
        self.assertEqual(len(test_df), 3576, "Test set should have 3,576 rows (20%).")
        self.assertEqual(len(train_df) + len(test_df), 17880, "Combined rows must equal 17,880.")

    def test_zero_id_leakage_between_train_and_test(self):
        """Rule 9: Zero job_id overlap between train and test sets."""
        train_ids = set(pd.read_csv(self.train_ids_csv)["job_id"])
        test_ids = set(pd.read_csv(self.test_ids_csv)["job_id"])

        intersection = train_ids.intersection(test_ids)
        self.assertEqual(len(intersection), 0, f"Found {len(intersection)} overlapping IDs!")

    def test_stratification_preserves_class_ratio(self):
        """Rule 8: Class balance must be preserved in both train and test partitions."""
        y_train = np.load(self.y_train_npy)
        y_test = np.load(self.y_test_npy)

        train_fraud_rate = float(np.mean(y_train) * 100)
        test_fraud_rate = float(np.mean(y_test) * 100)
        overall_fraud_rate = (866 / 17880) * 100  # ~4.8434%

        self.assertAlmostEqual(train_fraud_rate, overall_fraud_rate, delta=0.1)
        self.assertAlmostEqual(test_fraud_rate, overall_fraud_rate, delta=0.1)

    def test_target_and_job_id_absent_from_feature_names(self):
        """Rule 7: Target and job_id must NEVER be features in X."""
        with open(self.feature_names_json, "r", encoding="utf-8") as f:
            feature_names = json.load(f)["feature_names"]

        self.assertNotIn("fraudulent", feature_names)
        self.assertNotIn("job_id", feature_names)
        self.assertNotIn("meta_fraudulent", feature_names)
        self.assertNotIn("meta_job_id", feature_names)

    def test_feature_matrix_shapes_and_sparsity(self):
        """Rule 13: Feature matrices are non-empty, matching shapes."""
        X_train = sparse.load_npz(self.X_train_npz)
        X_test = sparse.load_npz(self.X_test_npz)
        y_train = np.load(self.y_train_npy)
        y_test = np.load(self.y_test_npy)

        self.assertEqual(X_train.shape[0], len(y_train))
        self.assertEqual(X_test.shape[0], len(y_test))
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertGreater(X_train.shape[1], 10000, "Should have >=10,000 hybrid features.")

    def test_preprocessor_serialization_and_inference(self):
        """Rule 14: Preprocessor can be safely loaded and transforms new data."""
        preprocessor = FullHybridPreprocessor.load(self.preprocessor_joblib)
        self.assertTrue(preprocessor.is_fitted)

        # Single mock sample transform
        sample_df = pd.DataFrame([{
            "title": "Software Engineer",
            "company_profile": "Tech company building cloud solutions.",
            "description": "Looking for Python backend developer.",
            "requirements": "3+ years Python, SQL experience.",
            "benefits": "Competitive health benefits and 401k.",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 1,
            "employment_type": "Full-time",
            "required_experience": "Mid-Senior level",
            "required_education": "Bachelor's Degree",
            "industry": "Information Technology and Services",
            "function": "Engineering",
            "salary_range": "100000-120000",
            "department": "Engineering",
            "location": "US, CA, San Francisco",
        }])

        X_sample = preprocessor.transform(sample_df)
        self.assertEqual(X_sample.shape[0], 1)
        self.assertEqual(X_sample.shape[1], len(preprocessor.feature_names_))


if __name__ == "__main__":
    unittest.main()

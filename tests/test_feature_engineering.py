"""
Unit Tests for Phase 3 Feature Engineering, Representation Interfaces, and Leakage Safeguards.
"""

import hashlib
import os
import unittest
import numpy as np
import pandas as pd
from scipy import sparse

from src.features.builder import ModularFeatureBuilder
from src.features.heuristic_features import HeuristicKeywordExtractor
from src.features.salary_features import SalaryFeatureExtractor
from src.features.contact_features import ContactFeatureExtractor
from src.features.text_features import TextStatisticsExtractor
from src.features.metadata_features import MetadataBinaryExtractor, MetadataCategoricalExtractor
from src.features.interfaces import (
    BaseFeatureRepresentation,
    IRepresentationA,
    IRepresentationB,
    IRepresentationC,
    IRepresentationD,
)


class TestFeatureEngineering(unittest.TestCase):
    """Test suite verifying modular feature engineering integrity and leakage protection."""

    RAW_DATA_PATH = "data/raw/fake_job_postings.csv"
    PROCESSED_TRAIN_PATH = "data/processed/train.csv"
    PROCESSED_TEST_PATH = "data/processed/test.csv"
    TRAIN_IDS_PATH = "data/processed/train_ids.csv"
    TEST_IDS_PATH = "data/processed/test_ids.csv"
    EXPECTED_RAW_HASH = "59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60"

    @classmethod
    def setUpClass(cls):
        """Load datasets once for test efficiency."""
        cls.df_train = pd.read_csv(cls.PROCESSED_TRAIN_PATH)
        cls.df_test = pd.read_csv(cls.PROCESSED_TEST_PATH)
        cls.train_ids = pd.read_csv(cls.TRAIN_IDS_PATH)["job_id"].tolist()
        cls.test_ids = pd.read_csv(cls.TEST_IDS_PATH)["job_id"].tolist()

        # Build feature builder fitted on train
        cls.builder = ModularFeatureBuilder(tfidf_max_features=500)
        cls.builder.fit(cls.df_train)

    def test_01_raw_dataset_hash_immutability(self):
        """Verify that the raw dataset has not been modified."""
        self.assertTrue(os.path.exists(self.RAW_DATA_PATH), "Raw dataset file must exist.")
        with open(self.RAW_DATA_PATH, "rb") as f:
            raw_bytes = f.read()
        calculated_hash = hashlib.sha256(raw_bytes).hexdigest()
        self.assertEqual(
            calculated_hash,
            self.EXPECTED_RAW_HASH,
            f"Raw dataset modified! Expected {self.EXPECTED_RAW_HASH}, got {calculated_hash}"
        )

    def test_02_train_test_row_counts(self):
        """Verify train/test row counts match 80/20 split."""
        self.assertEqual(len(self.df_train), 14304)
        self.assertEqual(len(self.df_test), 3576)

    def test_03_no_id_overlap(self):
        """Verify that training and test sample IDs are completely disjoint."""
        intersection = set(self.train_ids).intersection(set(self.test_ids))
        self.assertEqual(len(intersection), 0, f"Found {len(intersection)} overlapping IDs!")

    def test_04_target_and_job_id_absent_from_feature_names(self):
        """Verify target column and job_id are never exposed as features."""
        all_feature_names = self.builder.get_feature_names()
        self.assertNotIn("fraudulent", all_feature_names)
        self.assertNotIn("job_id", all_feature_names)
        self.assertNotIn("target", all_feature_names)

    def test_05_expected_feature_groups_exist(self):
        """Verify all 8 modular feature groups exist and can be extracted."""
        expected_groups = [
            "TEXT_TFIDF",
            "TEXT_STATISTICS",
            "METADATA_BINARY",
            "METADATA_CATEGORICAL",
            "MISSINGNESS",
            "SALARY",
            "CONTACT",
            "HEURISTIC",
        ]
        for group in expected_groups:
            self.assertIn(group, self.builder.AVAILABLE_GROUPS)
            mat_train = self.builder.extract_group(group, self.df_train.head(20))
            mat_test = self.builder.extract_group(group, self.df_test.head(20))
            self.assertEqual(mat_train.shape[1], mat_test.shape[1], f"Group {group} dimension mismatch between train/test!")

    def test_06_hybrid_all_transformation(self):
        """Verify full hybrid feature matrix fusion."""
        hybrid_tr = self.builder.transform(self.df_train.head(50))
        hybrid_te = self.builder.transform(self.df_test.head(50))

        self.assertTrue(sparse.issparse(hybrid_tr))
        self.assertTrue(sparse.issparse(hybrid_te))
        self.assertEqual(hybrid_tr.shape[1], hybrid_te.shape[1])
        self.assertEqual(hybrid_tr.shape[0], 50)
        self.assertEqual(hybrid_te.shape[0], 50)

    def test_07_heuristic_keyword_extractor(self):
        """Verify heuristic extractor compiles and executes deterministically."""
        ext = HeuristicKeywordExtractor()
        ext.fit(self.df_train.head(10))
        names = ext.get_feature_names_out()

        self.assertIn("payment_requests_count", names)
        self.assertIn("urgency_count", names)
        self.assertIn("total_scam_keyword_count", names)

        # Test on custom sample
        sample_df = pd.DataFrame({
            "clean_composite_text": ["Urgent hiring! Wire transfer required. Immediate start."]
        })
        res = ext.transform(sample_df)
        self.assertGreater(res["urgency_count"].iloc[0], 0)
        self.assertGreater(res["payment_requests_count"].iloc[0], 0)
        self.assertEqual(res["has_any_scam_keyword"].iloc[0], 1.0)

    def test_08_salary_extractor_bounds(self):
        """Verify safe parsing of salary ranges with fallbacks."""
        ext = SalaryFeatureExtractor()
        sample_df = pd.DataFrame({
            "salary_range": ["10000-20000", "50", None, "invalid_range", "0-0"]
        })
        res = ext.transform(sample_df)
        self.assertEqual(res["has_salary_specified"].iloc[0], 1.0)
        self.assertEqual(res["salary_min"].iloc[0], 10000.0)
        self.assertEqual(res["salary_max"].iloc[0], 20000.0)
        self.assertEqual(res["salary_spread"].iloc[0], 10000.0)
        self.assertEqual(res["has_salary_specified"].iloc[2], 0.0)
        self.assertEqual(res["salary_min"].iloc[2], 0.0)

    def test_09_contact_extractor_patterns(self):
        """Verify regex extraction of emails, free domains, URLs, and phones."""
        ext = ContactFeatureExtractor()
        sample_df = pd.DataFrame({
            "clean_composite_text": [
                "Contact us at hr@gmail.com or visit http://scam.xyz or call 555-123-4567",
                "Official enterprise listing with no contacts"
            ]
        })
        res = ext.transform(sample_df)
        self.assertEqual(res["has_email"].iloc[0], 1.0)
        self.assertEqual(res["has_free_email"].iloc[0], 1.0)
        self.assertEqual(res["has_url"].iloc[0], 1.0)
        self.assertEqual(res["has_phone"].iloc[0], 1.0)
        self.assertEqual(res["has_email"].iloc[1], 0.0)

    def test_10_representation_interfaces_hierarchy(self):
        """Verify inheritance hierarchy and interfaces for future representation families."""
        self.assertTrue(issubclass(IRepresentationA, BaseFeatureRepresentation))
        self.assertTrue(issubclass(IRepresentationB, BaseFeatureRepresentation))
        self.assertTrue(issubclass(IRepresentationC, BaseFeatureRepresentation))
        self.assertTrue(issubclass(ModularFeatureBuilder, IRepresentationA))


if __name__ == "__main__":
    unittest.main()

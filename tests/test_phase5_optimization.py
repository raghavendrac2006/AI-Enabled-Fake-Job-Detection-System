"""
Unit tests for Phase 5 Imbalance Handling, CV Tuning, and Threshold Optimization.
"""

import json
import os
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from src.models.resampling import create_imbalance_pipeline
from src.models.thresholds import find_optimal_threshold, apply_threshold


class TestPhase5ImbalanceAndOptimization(unittest.TestCase):
    """Test suite for Phase 5 modules: resampling pipelines, tuning, and thresholding."""

    def setUp(self):
        np.random.seed(42)
        # Create small synthetic imbalanced dataset
        # 100 samples, 10 positive, 90 negative
        self.X_train = np.random.randn(100, 10)
        self.y_train = np.array([0] * 90 + [1] * 10)
        
        self.X_test = np.random.randn(30, 10)
        self.y_test = np.array([0] * 27 + [1] * 3)

    def test_resampling_pipeline_creation(self):
        """Test pipeline factory for different imbalance strategies."""
        base_clf = LogisticRegression(random_state=42)
        
        # 1. No resampling (class_weight only)
        pipe_none = create_imbalance_pipeline(base_clf, strategy="none", random_state=42)
        pipe_none.fit(self.X_train, self.y_train)
        preds_none = pipe_none.predict(self.X_test)
        self.assertEqual(len(preds_none), 30)

        # 2. SMOTE pipeline
        pipe_smote = create_imbalance_pipeline(base_clf, strategy="smote", random_state=42, k_neighbors=3)
        pipe_smote.fit(self.X_train, self.y_train)
        preds_smote = pipe_smote.predict(self.X_test)
        self.assertEqual(len(preds_smote), 30)

        # 3. BorderlineSMOTE pipeline
        pipe_bsmote = create_imbalance_pipeline(base_clf, strategy="borderline_smote", random_state=42, k_neighbors=3)
        pipe_bsmote.fit(self.X_train, self.y_train)
        preds_bsmote = pipe_bsmote.predict(self.X_test)
        self.assertEqual(len(preds_bsmote), 30)

    def test_pipeline_no_test_leakage(self):
        """Verify that X_test size and integrity are strictly preserved without resampling during transform/predict."""
        base_clf = LogisticRegression(random_state=42)
        pipe = create_imbalance_pipeline(base_clf, strategy="smote", random_state=42, k_neighbors=3)
        
        X_test_copy = self.X_test.copy()
        pipe.fit(self.X_train, self.y_train)
        _ = pipe.predict(self.X_test)
        
        # Ensure test data was never modified
        np.testing.assert_array_equal(self.X_test, X_test_copy)

    def test_threshold_optimizer(self):
        """Test finding and applying optimal decision thresholds on synthetic scores."""
        y_val = np.array([0] * 80 + [1] * 20)
        # Synthetic probabilities with some overlap
        y_scores = np.concatenate([
            np.random.beta(1, 5, size=80),  # negatives skewed low
            np.random.beta(5, 1, size=20)   # positives skewed high
        ])
        
        opt_thresh, opt_f1 = find_optimal_threshold(y_val, y_scores, metric="f1")
        self.assertGreaterEqual(opt_thresh, 0.0)
        self.assertLessEqual(opt_thresh, 1.0)
        self.assertGreaterEqual(opt_f1, 0.0)
        self.assertLessEqual(opt_f1, 1.0)
        
        # Test applying threshold
        binary_preds = apply_threshold(y_scores, threshold=opt_thresh)
        self.assertEqual(len(binary_preds), 100)
        self.assertTrue(set(np.unique(binary_preds)).issubset({0, 1}))

    def test_phase5_artifacts_and_champion_model(self):
        """Verify that Phase 5 reports, figures, and champion model artifact exist and are valid."""
        import joblib
        csv_path = os.path.join("reports", "phase5_results.csv")
        json_path = os.path.join("reports", "phase5_results.json")
        md_path = os.path.join("reports", "phase5_comparison.md")
        champion_path = os.path.join("models", "phase5_best_LightGBM_HYBRID_ALL.joblib")

        self.assertTrue(os.path.exists(csv_path), "reports/phase5_results.csv missing")
        self.assertTrue(os.path.exists(json_path), "reports/phase5_results.json missing")
        self.assertTrue(os.path.exists(md_path), "reports/phase5_comparison.md missing")
        self.assertTrue(os.path.exists(champion_path), "models/phase5_best_LightGBM_HYBRID_ALL.joblib missing")

        # Verify figures
        for fig_num in ["06", "07", "08", "09", "10"]:
            fig_files = [f for f in os.listdir("reports/figures") if f.startswith(fig_num)]
            self.assertTrue(len(fig_files) > 0, f"Figure {fig_num} missing in reports/figures/")

        # Verify champion model loadable
        clf = joblib.load(champion_path)
        self.assertTrue(hasattr(clf, "predict"))
        self.assertTrue(hasattr(clf, "predict_proba"))


if __name__ == "__main__":
    unittest.main()

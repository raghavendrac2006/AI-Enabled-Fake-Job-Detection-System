"""
Unit tests for Phase 4 Classical ML Baselines and Evaluation Metrics.
"""

import json
import os
import unittest
import numpy as np
import pandas as pd
from scipy import sparse

from src.evaluation.metrics import compute_classification_metrics
from src.models.classical import get_classical_models, compute_class_imbalance_ratio
from src.models.trainer import ExperimentRunner


class TestModelTrainingAndEvaluation(unittest.TestCase):
    """Test suite for model initialization, training pipelines, and metrics."""

    def setUp(self):
        np.random.seed(42)
        # Create small synthetic test set
        self.y_true = np.array([0] * 90 + [1] * 10)
        self.y_pred = np.array([0] * 88 + [1] * 2 + [0] * 2 + [1] * 8)
        self.y_score = np.array([0.1] * 88 + [0.7] * 2 + [0.3] * 2 + [0.9] * 8)

    def test_compute_classification_metrics_bounds(self):
        """Test that computed metrics are within valid probability and score ranges [0, 1]."""
        metrics = compute_classification_metrics(self.y_true, self.y_pred, self.y_score)
        
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertGreaterEqual(metrics["fraud_precision"], 0.0)
        self.assertLessEqual(metrics["fraud_precision"], 1.0)
        self.assertGreaterEqual(metrics["fraud_recall"], 0.0)
        self.assertLessEqual(metrics["fraud_recall"], 1.0)
        self.assertGreaterEqual(metrics["fraud_f1"], 0.0)
        self.assertLessEqual(metrics["fraud_f1"], 1.0)
        self.assertGreaterEqual(metrics["roc_auc"], 0.0)
        self.assertLessEqual(metrics["roc_auc"], 1.0)
        self.assertGreaterEqual(metrics["pr_auc"], 0.0)
        self.assertLessEqual(metrics["pr_auc"], 1.0)
        
        # Test confusion matrix structure
        cm = metrics["confusion_matrix"]
        self.assertIn("tn", cm)
        self.assertIn("fp", cm)
        self.assertIn("fn", cm)
        self.assertIn("tp", cm)
        self.assertEqual(cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"], len(self.y_true))

    def test_get_classical_models_instantiation(self):
        """Test that all 7 required models instantiate correctly."""
        scale_weight = compute_class_imbalance_ratio(self.y_true)
        models = get_classical_models(scale_pos_weight=scale_weight, random_state=42)
        
        expected_models = [
            "DummyClassifier (Majority)",
            "LogisticRegression (Balanced)",
            "MultinomialNB",
            "LinearSVC (Balanced)",
            "RandomForest (Balanced)",
            "XGBoost (Weighted)",
            "LightGBM (Weighted)"
        ]
        
        self.assertEqual(len(models), 7)
        for name in expected_models:
            self.assertIn(name, models)

    def test_experiment_runner_synthetic(self):
        """Test ExperimentRunner training and prediction extraction on synthetic sparse data."""
        # 100 samples, 20 features
        X_train = sparse.csr_matrix(np.random.rand(100, 20))
        y_train = np.array([0] * 90 + [1] * 10)
        X_test = sparse.csr_matrix(np.random.rand(20, 20))
        y_test = np.array([0] * 18 + [1] * 2)

        models = {
            "Dummy": get_classical_models(scale_pos_weight=9.0)["DummyClassifier (Majority)"],
            "LogReg": get_classical_models(scale_pos_weight=9.0)["LogisticRegression (Balanced)"]
        }

        runner = ExperimentRunner(random_state=42)
        results = runner.run_experiment(
            X_train, y_train, X_test, y_test, representation_name="SYNTHETIC", save_models=False
        )
        
        self.assertGreaterEqual(len(results), 5)
        for res in results:
            self.assertIn("fraud_f1", res)
            self.assertIn("train_time_sec", res)
            self.assertIn("inference_time_sec", res)

    def test_phase4_reports_exist_and_valid(self):
        """Verify that Phase 4 artifacts are generated and non-empty."""
        csv_path = os.path.join("reports", "phase4_model_results.csv")
        json_path = os.path.join("reports", "phase4_model_results.json")
        md_path = os.path.join("reports", "phase4_model_comparison.md")
        
        self.assertTrue(os.path.exists(csv_path), "phase4_model_results.csv missing")
        self.assertTrue(os.path.exists(json_path), "phase4_model_results.json missing")
        self.assertTrue(os.path.exists(md_path), "phase4_model_comparison.md missing")
        
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 21, "Expected 21 experiment runs in CSV (7 models * 3 representations)")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 21, "Expected 21 experiment records in JSON")


if __name__ == "__main__":
    unittest.main()

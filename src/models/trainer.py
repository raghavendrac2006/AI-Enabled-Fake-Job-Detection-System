"""
Experiment Runner and Classical Model Training Pipeline for Fake Job Detection.

Fits classifiers strictly on training data, measures execution latencies,
extracts decision scores/probabilities, and computes full evaluation metrics.
"""

import os
import time
from typing import Dict, List, Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from src.models.classical import get_classical_models
from src.evaluation.metrics import compute_classification_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentRunner:
    """
    Coordinates training, prediction, scoring, and metric collation for classical ML baselines.
    """

    def __init__(
        self,
        random_state: int = 42,
        models_dir: str = "models",
    ):
        self.random_state = random_state
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.results_: List[Dict[str, Any]] = []
        self.fitted_models_: Dict[str, Any] = {}

    def _get_prediction_scores(self, model: Any, X: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract discrete class predictions and continuous ranking scores."""
        y_pred = model.predict(X)
        y_score = None

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    y_score = proba[:, 1]
                else:
                    y_score = proba.ravel()
            except Exception as e:
                logger.warning(f"Failed to extract predict_proba: {e}")

        elif hasattr(model, "decision_function"):
            try:
                y_score = model.decision_function(X)
            except Exception as e:
                logger.warning(f"Failed to extract decision_function: {e}")

        return y_pred, y_score

    def run_experiment(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_test: Any,
        y_test: np.ndarray,
        representation_name: str,
        save_models: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Trains all baseline models on the given representation and records evaluation metrics.
        """
        y_train = np.asarray(y_train).astype(int)
        y_test = np.asarray(y_test).astype(int)

        models = get_classical_models(random_state=self.random_state)
        rep_results = []

        print(f"\n--- Starting Experiment: Representation '{representation_name}' ---", flush=True)
        print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}", flush=True)

        for model_name, model in models.items():
            print(f"Training [{representation_name}] -> {model_name}...", flush=True)

            # Check non-negativity compatibility for MultinomialNB
            if "MultinomialNB" in model_name:
                has_negative = False
                if sparse.issparse(X_train):
                    if (X_train.data < 0).any():
                        has_negative = True
                else:
                    if (X_train < 0).any():
                        has_negative = True

                if has_negative:
                    print(
                        f"Skipping {model_name} for '{representation_name}' (requires non-negative feature matrix).",
                        flush=True,
                    )
                    continue

            # 1. Train Model & Measure Latency
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            # 2. Inference & Measure Latency
            t0 = time.time()
            y_pred, y_score = self._get_prediction_scores(model, X_test)
            inf_time = time.time() - t0

            # 3. Compute Metrics
            metrics = compute_classification_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_score=y_score,
                pos_label=1,
            )

            result_entry = {
                "representation": representation_name,
                "model_name": model_name,
                "train_time_sec": round(train_time, 4),
                "inference_time_sec": round(inf_time, 4),
                "accuracy": metrics["accuracy"],
                "fraud_precision": metrics["fraud_precision"],
                "fraud_recall": metrics["fraud_recall"],
                "fraud_f1": metrics["fraud_f1"],
                "macro_f1": metrics["macro_f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "confusion_matrix": metrics["confusion_matrix"],
            }

            rep_results.append(result_entry)
            self.results_.append(result_entry)

            # Store fitted model in memory
            model_key = f"{representation_name}__{model_name}".replace(" ", "_").replace("(", "").replace(")", "")
            self.fitted_models_[model_key] = model

            # Save model to disk if requested
            if save_models:
                save_path = os.path.join(self.models_dir, f"{model_key}.joblib")
                joblib.dump(model, save_path)

            print(
                f"  -> [{model_name}] F1: {metrics['fraud_f1']:.4f} | Prec: {metrics['fraud_precision']:.4f} | "
                f"Rec: {metrics['fraud_recall']:.4f} | PR-AUC: {metrics['pr_auc']} | ROC-AUC: {metrics['roc_auc']} | Time: {train_time:.2f}s",
                flush=True,
            )

        return rep_results

    def get_results_dataframe(self) -> pd.DataFrame:
        """Return all experimental results as a formatted pandas DataFrame."""
        if not self.results_:
            return pd.DataFrame()

        rows = []
        for r in self.results_:
            row = {
                "Representation": r["representation"],
                "Model": r["model_name"],
                "Accuracy": r["accuracy"],
                "Fraud Precision": r["fraud_precision"],
                "Fraud Recall": r["fraud_recall"],
                "Fraud F1": r["fraud_f1"],
                "Macro F1": r["macro_f1"],
                "ROC-AUC": r["roc_auc"],
                "PR-AUC": r["pr_auc"],
                "Train Time (s)": r["train_time_sec"],
                "Inference Time (s)": r["inference_time_sec"],
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return df.sort_values(by=["Fraud F1", "PR-AUC"], ascending=False).reset_index(drop=True)

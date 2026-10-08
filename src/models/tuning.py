"""
Targeted Hyperparameter Tuning for Top Candidate Classical Models.

Conducts controlled, leakage-free Stratified K-Fold Cross-Validation strictly
on the training partition to optimize hyperparameters for LinearSVC, RandomForest, and LightGBM.
"""

import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TargetedHyperparameterTuner:
    """
    Manages targeted hyperparameter tuning via StratifiedKFold cross-validation on training data.
    """

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        scoring: str = "f1",  # Optimizes minority fraud F1
        n_jobs: int = 4,
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.tuning_history_: List[Dict[str, Any]] = []

    def tune_linear_svc(
        self,
        X_train: Any,
        y_train: np.ndarray,
        representation_name: str = "TEXT_TFIDF",
    ) -> Tuple[LinearSVC, Dict[str, Any]]:
        """
        Tunes regularization C, loss, and class weighting for LinearSVC on training data.
        """
        logger.info(f"Tuning LinearSVC on {representation_name} with 5-Fold Stratified CV...")
        base_model = LinearSVC(max_iter=3000, random_state=self.random_state)

        # 6 focused configurations
        param_grid = {
            "C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "class_weight": ["balanced", None],
            "loss": ["squared_hinge"],
        }

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=True,
            return_train_score=False,
        )

        t0 = time.time()
        grid.fit(X_train, y_train)
        tune_time = time.time() - t0

        best_params = grid.best_params_
        best_score = float(grid.best_score_)

        logger.info(f"Best LinearSVC on {representation_name}: {best_params} (Mean CV F1: {best_score:.4f})")

        result_summary = {
            "model_name": "LinearSVC",
            "representation": representation_name,
            "best_params": best_params,
            "best_cv_f1": round(best_score, 4),
            "tune_time_sec": round(tune_time, 2),
            "cv_results": [
                {
                    "params": p,
                    "mean_cv_f1": round(score, 4),
                    "std_cv_f1": round(std, 4),
                }
                for p, score, std in zip(
                    grid.cv_results_["params"],
                    grid.cv_results_["mean_test_score"],
                    grid.cv_results_["std_test_score"],
                )
            ],
        }
        self.tuning_history_.append(result_summary)

        return grid.best_estimator_, result_summary

    def tune_random_forest(
        self,
        X_train: Any,
        y_train: np.ndarray,
        representation_name: str = "HYBRID_ALL",
    ) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """
        Tunes tree depth, min_samples_split, and class_weight for RandomForest on training data.
        """
        logger.info(f"Tuning RandomForest on {representation_name} with 5-Fold Stratified CV...")
        base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)

        # 6 focused configurations
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [25, 40, None],
            "class_weight": ["balanced"],
            "min_samples_split": [2, 5],
        }

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=True,
            return_train_score=False,
        )

        t0 = time.time()
        grid.fit(X_train, y_train)
        tune_time = time.time() - t0

        best_params = grid.best_params_
        best_score = float(grid.best_score_)

        logger.info(f"Best RandomForest on {representation_name}: {best_params} (Mean CV F1: {best_score:.4f})")

        result_summary = {
            "model_name": "RandomForest",
            "representation": representation_name,
            "best_params": best_params,
            "best_cv_f1": round(best_score, 4),
            "tune_time_sec": round(tune_time, 2),
            "cv_results": [
                {
                    "params": p,
                    "mean_cv_f1": round(score, 4),
                    "std_cv_f1": round(std, 4),
                }
                for p, score, std in zip(
                    grid.cv_results_["params"],
                    grid.cv_results_["mean_test_score"],
                    grid.cv_results_["std_test_score"],
                )
            ],
        }
        self.tuning_history_.append(result_summary)

        return grid.best_estimator_, result_summary

    def tune_lightgbm(
        self,
        X_train: Any,
        y_train: np.ndarray,
        representation_name: str = "HYBRID_ALL",
    ) -> Tuple[LGBMClassifier, Dict[str, Any]]:
        """
        Tunes scale_pos_weight, num_leaves, and learning_rate for LightGBM on training data.
        """
        logger.info(f"Tuning LightGBM on {representation_name} with 5-Fold Stratified CV...")
        base_model = LGBMClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
        )

        # 6 focused configurations
        param_grid = {
            "scale_pos_weight": [10.0, 19.65],
            "num_leaves": [31, 63],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [100],
        }

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=True,
            return_train_score=False,
        )

        t0 = time.time()
        grid.fit(X_train, y_train)
        tune_time = time.time() - t0

        best_params = grid.best_params_
        best_score = float(grid.best_score_)

        logger.info(f"Best LightGBM on {representation_name}: {best_params} (Mean CV F1: {best_score:.4f})")

        result_summary = {
            "model_name": "LightGBM",
            "representation": representation_name,
            "best_params": best_params,
            "best_cv_f1": round(best_score, 4),
            "tune_time_sec": round(tune_time, 2),
            "cv_results": [
                {
                    "params": p,
                    "mean_cv_f1": round(score, 4),
                    "std_cv_f1": round(std, 4),
                }
                for p, score, std in zip(
                    grid.cv_results_["params"],
                    grid.cv_results_["mean_test_score"],
                    grid.cv_results_["std_test_score"],
                )
            ],
        }
        self.tuning_history_.append(result_summary)

        return grid.best_estimator_, result_summary

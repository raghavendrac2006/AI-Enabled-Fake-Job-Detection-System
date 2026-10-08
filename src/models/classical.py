"""
Classical Machine Learning Model Definitions for Fake Job Detection.

Initializes baseline classifiers with explicit, recorded class imbalance configurations
and random seed 42.
"""

from typing import Dict, Any
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_classical_models(random_state: int = 42, scale_pos_weight: float = 19.65) -> Dict[str, Any]:
    """
    Returns a dictionary of initialized classical baseline models with standard hyperparameters.
    """
    models = {
        "DummyClassifier (Majority)": DummyClassifier(
            strategy="most_frequent",
            random_state=random_state,
        ),
        "LogisticRegression (Balanced)": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        ),
        "MultinomialNB": MultinomialNB(
            alpha=1.0,
            fit_prior=True,
        ),
        "LinearSVC (Balanced)": LinearSVC(
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        ),
        "RandomForest (Balanced)": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=4,
        ),
        "XGBoost (Weighted)": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=4,
            tree_method="hist",
        ),
        "LightGBM (Weighted)": LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=4,
            verbose=-1,
        ),
    }

    return models


def compute_class_imbalance_ratio(y: Any) -> float:
    """Computes the ratio of negative samples to positive samples (scale_pos_weight)."""
    import numpy as np
    y_arr = np.asarray(y)
    n_pos = np.sum(y_arr == 1)
    n_neg = np.sum(y_arr == 0)
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)

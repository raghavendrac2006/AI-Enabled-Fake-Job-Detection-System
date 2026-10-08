"""
Resampling and Imbalance Strategies for Fake Job Detection.

Constructs leakage-safe imblearn pipelines for SMOTE and BorderlineSMOTE,
ensuring synthetic oversampling is restricted strictly to training folds.
"""

from typing import Dict, Any, Optional
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.base import clone


def create_imbalance_pipeline(
    estimator: Any,
    strategy: str = "none",
    random_state: int = 42,
    k_neighbors: int = 5,
) -> Any:
    """
    Wraps an estimator in an imblearn Pipeline with the requested imbalance handling strategy.

    Args:
        estimator: A scikit-learn compatible classifier.
        strategy: 'none' (or 'class_weight'), 'smote', 'borderline_smote'.
        random_state: Seed for reproducibility.
        k_neighbors: Nearest neighbors parameter for SMOTE.

    Returns:
        imblearn Pipeline or standalone estimator.
    """
    strategy = strategy.lower().strip()

    if strategy in ["none", "class_weight", "class_weighted", "baseline"]:
        return clone(estimator)

    elif strategy == "smote":
        sampler = SMOTE(
            random_state=random_state,
            k_neighbors=k_neighbors,
        )
        return ImbPipeline([
            ("sampler", sampler),
            ("classifier", clone(estimator)),
        ])

    elif strategy in ["borderline_smote", "borderline"]:
        sampler = BorderlineSMOTE(
            random_state=random_state,
            k_neighbors=k_neighbors,
        )
        return ImbPipeline([
            ("sampler", sampler),
            ("classifier", clone(estimator)),
        ])

    else:
        raise ValueError(
            f"Unsupported imbalance strategy '{strategy}'. Supported: 'none', 'smote', 'borderline_smote'."
        )


def get_imbalance_strategies(
    base_estimator: Any,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Returns a dictionary of standard imbalance strategy pipelines for a given base classifier.
    """
    return {
        "Class_Weighted": create_imbalance_pipeline(base_estimator, strategy="none", random_state=random_state),
        "SMOTE": create_imbalance_pipeline(base_estimator, strategy="smote", random_state=random_state),
        "Borderline_SMOTE": create_imbalance_pipeline(base_estimator, strategy="borderline_smote", random_state=random_state),
    }

"""
Feature Engineering and Metadata Builder Module.
AI-Powered Fake Job and Recruitment Scam Detection System.

Extracts structured domain indicators, presence flags, and text length statistics.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger

logger = get_logger("preprocessing.feature_builder")


class DomainFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that engineers tabular metadata features:
    1. Presence flags (has_salary_specified, has_company_profile, has_requirements, has_benefits)
    2. Character length and word count statistics across individual and composite text
    """

    def __init__(
        self,
        engineer_presence_flags: bool = True,
        engineer_length_stats: bool = True,
    ):
        self.engineer_presence_flags = engineer_presence_flags
        self.engineer_length_stats = engineer_length_stats
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        engineered = pd.DataFrame(index=X.index)

        # 1. Presence Indicators
        if self.engineer_presence_flags:
            if "salary_range" in X.columns:
                engineered["has_salary_specified"] = X["salary_range"].notna().astype(int)
            if "company_profile" in X.columns:
                engineered["has_company_profile"] = X["company_profile"].notna().astype(int)
            if "requirements" in X.columns:
                engineered["has_requirements"] = X["requirements"].notna().astype(int)
            if "benefits" in X.columns:
                engineered["has_benefits"] = X["benefits"].notna().astype(int)

        # 2. Text Length and Word Count Statistics
        if self.engineer_length_stats:
            for text_col in ["title", "company_profile", "description", "requirements", "benefits"]:
                if text_col in X.columns:
                    s = X[text_col].fillna("").astype(str)
                    engineered[f"{text_col}_char_len"] = s.str.len()
                    engineered[f"{text_col}_word_count"] = s.apply(lambda x: len(x.split()) if x.strip() else 0)

            # Composite text length
            text_cols = [c for c in ["title", "company_profile", "description", "requirements", "benefits"] if c in X.columns]
            composite = X[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
            engineered["composite_char_len"] = composite.str.len()
            engineered["composite_word_count"] = composite.apply(lambda x: len(x.split()) if x.strip() else 0)

        self.feature_names_ = list(engineered.columns)
        return engineered

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_

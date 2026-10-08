"""
Text Statistics and Structural Feature Extractor for Fake Job Postings.

Extracts interpretable length, vocabulary, casing, punctuation, and structural ratio
features from individual text fields and composite postings.
Ultra-fast implementation using native Python string methods and list comprehensions.
"""

import string
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextStatisticsExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts statistical and structural text features from job postings:
    - Character counts per field and composite text
    - Word counts per field and composite text
    - Average word length
    - Uppercase character counts and ratios
    - Numeric digit counts and ratios
    - Punctuation frequency (exclamation marks, question marks, general punctuation)
    - Structural ratios (section availability count, description-to-requirements ratio)
    """

    TEXT_FIELDS = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits",
    ]

    def __init__(self, composite_col: str = "clean_composite_text"):
        self.composite_col = composite_col
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TextStatisticsExtractor":
        self._build_feature_names()
        return self

    def _build_feature_names(self) -> None:
        names = []
        # Per-field text statistics
        for field in self.TEXT_FIELDS:
            names.extend([
                f"{field}_char_count",
                f"{field}_word_count",
                f"{field}_avg_word_length",
                f"{field}_exclamation_count",
                f"{field}_question_count",
                f"{field}_is_present",
            ])

        # Composite text statistics
        names.extend([
            "composite_char_count",
            "composite_word_count",
            "composite_avg_word_length",
            "composite_uppercase_count",
            "composite_uppercase_ratio",
            "composite_digit_count",
            "composite_digit_ratio",
            "composite_punctuation_count",
            "composite_exclamation_count",
            "composite_question_count",
        ])

        # Structural & ratio features
        names.extend([
            "populated_sections_count",
            "desc_to_req_length_ratio",
            "desc_to_req_word_ratio",
            "company_profile_to_desc_ratio",
        ])

        self.feature_names_ = names

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        if not self.feature_names_:
            self._build_feature_names()
        return list(self.feature_names_)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_rows = len(X)
        features = {}

        # 1. Per-field text statistics using raw list comprehensions
        for field in self.TEXT_FIELDS:
            raw_list = [str(v) if pd.notna(v) else "" for v in X[field]] if field in X.columns else [""] * n_rows
            
            char_counts = np.array([len(s) for s in raw_list], dtype=float)
            word_counts = np.array([len(s.split()) for s in raw_list], dtype=float)
            avg_word_lens = np.where(word_counts > 0, char_counts / np.maximum(word_counts, 1.0), 0.0)
            exclamation_counts = np.array([s.count("!") for s in raw_list], dtype=float)
            question_counts = np.array([s.count("?") for s in raw_list], dtype=float)
            is_present = (char_counts > 0).astype(float)

            features[f"{field}_char_count"] = char_counts
            features[f"{field}_word_count"] = word_counts
            features[f"{field}_avg_word_length"] = avg_word_lens
            features[f"{field}_exclamation_count"] = exclamation_counts
            features[f"{field}_question_count"] = question_counts
            features[f"{field}_is_present"] = is_present

        # 2. Composite text statistics
        if self.composite_col in X.columns:
            comp_list = [str(v) if pd.notna(v) else "" for v in X[self.composite_col]]
        else:
            comp_list = [" ".join(str(X.at[i, f]) for f in self.TEXT_FIELDS if f in X.columns and pd.notna(X.at[i, f])) for i in X.index]

        punct_set = set(string.punctuation)
        comp_char_counts = np.array([len(s) for s in comp_list], dtype=float)
        comp_word_counts = np.array([len(s.split()) for s in comp_list], dtype=float)
        comp_avg_word_lens = np.where(comp_word_counts > 0, comp_char_counts / np.maximum(comp_word_counts, 1.0), 0.0)

        comp_upper_counts = np.array([sum(1 for c in s if c.isupper()) for s in comp_list], dtype=float)
        comp_upper_ratios = np.where(comp_char_counts > 0, comp_upper_counts / np.maximum(comp_char_counts, 1.0), 0.0)

        comp_digit_counts = np.array([sum(1 for c in s if c.isdigit()) for s in comp_list], dtype=float)
        comp_digit_ratios = np.where(comp_char_counts > 0, comp_digit_counts / np.maximum(comp_char_counts, 1.0), 0.0)

        comp_punct_counts = np.array([sum(1 for c in s if c in punct_set) for s in comp_list], dtype=float)
        comp_exclamation_counts = np.array([s.count("!") for s in comp_list], dtype=float)
        comp_question_counts = np.array([s.count("?") for s in comp_list], dtype=float)

        features["composite_char_count"] = comp_char_counts
        features["composite_word_count"] = comp_word_counts
        features["composite_avg_word_length"] = comp_avg_word_lens
        features["composite_uppercase_count"] = comp_upper_counts
        features["composite_uppercase_ratio"] = comp_upper_ratios
        features["composite_digit_count"] = comp_digit_counts
        features["composite_digit_ratio"] = comp_digit_ratios
        features["composite_punctuation_count"] = comp_punct_counts
        features["composite_exclamation_count"] = comp_exclamation_counts
        features["composite_question_count"] = comp_question_counts

        # 3. Structural & Ratio features (safe Laplace smoothing)
        presence_arrays = [features[f"{field}_is_present"] for field in self.TEXT_FIELDS]
        features["populated_sections_count"] = np.sum(presence_arrays, axis=0)

        desc_len = features["description_char_count"]
        req_len = features["requirements_char_count"]
        desc_words = features["description_word_count"]
        req_words = features["requirements_word_count"]
        prof_len = features["company_profile_char_count"]

        features["desc_to_req_length_ratio"] = (desc_len + 1.0) / (req_len + 1.0)
        features["desc_to_req_word_ratio"] = (desc_words + 1.0) / (req_words + 1.0)
        features["company_profile_to_desc_ratio"] = (prof_len + 1.0) / (desc_len + 1.0)

        res_df = pd.DataFrame(features, index=X.index)
        self.feature_names_ = res_df.columns.tolist()
        return res_df

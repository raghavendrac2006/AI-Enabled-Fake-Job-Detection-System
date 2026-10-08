"""
Metadata, Categorical, Binary, and Missingness Feature Extractors.

Provides scikit-learn transformers for structured tabular fields:
- Binary flags: telecommuting, has_company_logo, has_questions
- Explicit missingness indicators across domain sections
- High-level location attributes (country code, US-based indicator)
- Categorical one-hot encoding with strict out-of-vocabulary handling
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetadataBinaryExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts binary and missingness indicators from structured tabular fields.
    """

    BINARY_COLS = ["telecommuting", "has_company_logo", "has_questions"]
    MISSINGNESS_FIELDS = [
        ("company_profile", "has_company_profile"),
        ("requirements", "has_requirements"),
        ("benefits", "has_benefits"),
        ("department", "has_department"),
        ("location", "has_location_specified"),
    ]

    def __init__(self):
        self.feature_names_: List[str] = (
            self.BINARY_COLS + [target for _, target in self.MISSINGNESS_FIELDS]
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MetadataBinaryExtractor":
        return self

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return list(self.feature_names_)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        res = pd.DataFrame(index=X.index)

        # 1. Binary columns (fill missing with 0 and cast to int)
        for col in self.BINARY_COLS:
            if col in X.columns:
                res[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(int)
            else:
                res[col] = 0

        # 2. Missingness indicator columns
        for source_col, target_col in self.MISSINGNESS_FIELDS:
            if source_col in X.columns:
                res[target_col] = (
                    X[source_col].fillna("").astype(str).str.strip().ne("").astype(int)
                )
            else:
                res[target_col] = 0

        return res


class MetadataCategoricalExtractor(BaseEstimator, TransformerMixin):
    """
    Encodes categorical metadata and location features using OneHotEncoder.
    Fitted strictly on training data with handle_unknown='ignore' to prevent leakage.
    """

    CATEGORICAL_COLS = [
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function",
    ]

    def __init__(self, include_country: bool = True, sparse_output: bool = False):
        self.include_country = include_country
        self.sparse_output = sparse_output
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)
        self.feature_names_: List[str] = []

    def _extract_country(self, loc_series: pd.Series) -> pd.Series:
        def parse_country(val):
            if not val or pd.isna(val) or not str(val).strip():
                return "__MISSING__"
            parts = str(val).split(",")
            return parts[0].strip().upper() if parts else "__MISSING__"

        return loc_series.apply(parse_country)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MetadataCategoricalExtractor":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df_cat = pd.DataFrame(index=X.index)
        for col in self.CATEGORICAL_COLS:
            if col in X.columns:
                df_cat[col] = X[col].fillna("__MISSING__").astype(str).str.strip()
                df_cat[col] = df_cat[col].replace("", "__MISSING__")
            else:
                df_cat[col] = "__MISSING__"

        if self.include_country:
            loc_col = X["location"] if "location" in X.columns else pd.Series(["__MISSING__"] * len(X), index=X.index)
            df_cat["country"] = self._extract_country(loc_col)

        self.encoder_.fit(df_cat)
        self.feature_names_ = [f"cat_{name}" for name in self.encoder_.get_feature_names_out()]
        return self

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return list(self.feature_names_)

    def transform(self, X: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df_cat = pd.DataFrame(index=X.index)
        for col in self.CATEGORICAL_COLS:
            if col in X.columns:
                df_cat[col] = X[col].fillna("__MISSING__").astype(str).str.strip()
                df_cat[col] = df_cat[col].replace("", "__MISSING__")
            else:
                df_cat[col] = "__MISSING__"

        if self.include_country:
            loc_col = X["location"] if "location" in X.columns else pd.Series(["__MISSING__"] * len(X), index=X.index)
            df_cat["country"] = self._extract_country(loc_col)

        encoded = self.encoder_.transform(df_cat)
        if not self.sparse_output and hasattr(encoded, "toarray"):
            encoded = encoded.toarray()
        
        if not self.sparse_output:
            return pd.DataFrame(encoded, columns=self.feature_names_, index=X.index)
        return encoded

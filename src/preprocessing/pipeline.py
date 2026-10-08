"""
Unified Preprocessing & Feature Engineering Pipeline Module.
AI-Powered Fake Job and Recruitment Scam Detection System.

Constructs a scikit-learn compatible ColumnTransformer / Pipeline architecture:
- Text pipeline: Composite text construction -> TF-IDF (10,000 max features, n-grams 1-2)
- Categorical pipeline: Missing value imputation -> OneHotEncoder
- Binary pipeline: Pass-through flags
- Engineered features: Presence flags + text length/word count statistics -> Robust standardization
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import AppConfig, get_config
from src.preprocessing.text_cleaner import clean_text_string, build_composite_text
from src.preprocessing.missing_values import MissingValueImputer
from src.preprocessing.feature_builder import DomainFeatureExtractor
from src.utils.logger import get_logger

logger = get_logger("preprocessing.pipeline")


class CompositeTextExtractor(BaseEstimator, TransformerMixin):
    """Transformer that converts raw DataFrame text columns into a clean 1D list of strings."""

    def __init__(self, text_columns: Optional[List[str]] = None, lowercase: bool = True):
        self.text_columns = text_columns or ["title", "company_profile", "description", "requirements", "benefits"]
        self.lowercase = lowercase

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return [build_composite_text(row, text_columns=self.text_columns, lowercase=self.lowercase) for _, row in X.iterrows()]
        elif isinstance(X, pd.Series):
            return [clean_text_string(val, lowercase=self.lowercase) for val in X]
        elif isinstance(X, (list, np.ndarray)):
            return [clean_text_string(str(val), lowercase=self.lowercase) for val in X]
        else:
            raise TypeError(f"Unsupported type {type(X)} for CompositeTextExtractor")


class TabularFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Processes metadata:
    - Imputes missing categories and one-hot encodes them
    - Passes binary indicators
    - Engineers presence and length features, then standardizes them
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        binary_columns: Optional[List[str]] = None,
    ):
        self.categorical_columns = categorical_columns or [
            "employment_type", "required_experience", "required_education", "industry", "function"
        ]
        self.binary_columns = binary_columns or [
            "telecommuting", "has_company_logo", "has_questions"
        ]
        self.imputer = MissingValueImputer(categorical_columns=self.categorical_columns, binary_columns=self.binary_columns)
        self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.feature_extractor = DomainFeatureExtractor()
        self.scaler = StandardScaler(with_mean=False)
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting tabular metadata preprocessors...")
        X_imputed = self.imputer.transform(X)

        # Fit One-Hot Encoder on categorical columns
        cat_data = X_imputed[[c for c in self.categorical_columns if c in X_imputed.columns]]
        self.one_hot_encoder.fit(cat_data)

        # Fit Scaler on engineered numeric length features
        engineered = self.feature_extractor.transform(X)
        self.scaler.fit(engineered)

        # Compile feature names
        cat_features = list(self.one_hot_encoder.get_feature_names_out(cat_data.columns))
        bin_features = [c for c in self.binary_columns if c in X_imputed.columns]
        eng_features = list(engineered.columns)

        self.feature_names_ = cat_features + bin_features + eng_features
        logger.info(f"Tabular features dimension: {len(self.feature_names_)} columns.")
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        X_imputed = self.imputer.transform(X)

        # 1. Categorical Sparse Matrix
        cat_data = X_imputed[[c for c in self.categorical_columns if c in X_imputed.columns]]
        cat_sparse = self.one_hot_encoder.transform(cat_data)

        # 2. Binary Dense -> Sparse
        bin_cols = [c for c in self.binary_columns if c in X_imputed.columns]
        bin_values = X_imputed[bin_cols].values.astype(float)
        bin_sparse = sparse.csr_matrix(bin_values)

        # 3. Engineered Features Scaled -> Sparse
        engineered = self.feature_extractor.transform(X)
        eng_scaled = self.scaler.transform(engineered)
        eng_sparse = sparse.csr_matrix(eng_scaled)

        # Combine all tabular components
        combined_tabular = sparse.hstack([cat_sparse, bin_sparse, eng_sparse], format="csr")
        return combined_tabular

    def get_feature_names_out(self) -> List[str]:
        return self.feature_names_


class FullHybridPreprocessor(BaseEstimator, TransformerMixin):
    """
    Top-level hybrid preprocessor fusing TF-IDF text features and tabular metadata.
    Scikit-learn compatible and serializable via joblib.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        tfidf_cfg = self.config.preprocessing.get("tfidf", {})

        self.text_extractor = CompositeTextExtractor(text_columns=self.config.text_columns)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_cfg.get("max_features", 10000),
            ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
            min_df=tfidf_cfg.get("min_df", 3),
            max_df=tfidf_cfg.get("max_df", 0.95),
            sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
            norm=tfidf_cfg.get("norm", "l2"),
            strip_accents=tfidf_cfg.get("strip_accents", "unicode"),
        )
        self.tabular_pipeline = TabularFeaturePipeline(
            categorical_columns=self.config.categorical_columns,
            binary_columns=self.config.binary_columns,
        )
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y=None):
        logger.info(f"Fitting FullHybridPreprocessor on {len(X)} training samples...")

        # 1. Fit TF-IDF strictly on training text
        raw_text_list = self.text_extractor.transform(X)
        self.tfidf_vectorizer.fit(raw_text_list)
        tfidf_vocab_size = len(self.tfidf_vectorizer.vocabulary_)
        logger.info(f"Fitted TF-IDF vocabulary size: {tfidf_vocab_size:,} n-grams.")

        # 2. Fit Tabular Metadata preprocessors
        self.tabular_pipeline.fit(X)

        # 3. Assemble feature names list
        tfidf_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]
        tabular_names = [f"meta_{name}" for name in self.tabular_pipeline.get_feature_names_out()]
        self.feature_names_ = tfidf_names + tabular_names
        self.is_fitted = True

        logger.info(f"Total hybrid feature dimensionality: {len(self.feature_names_):,} features.")
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        if not self.is_fitted:
            raise RuntimeError("FullHybridPreprocessor must be fitted before transforming data.")

        # Transform text to TF-IDF
        raw_text_list = self.text_extractor.transform(X)
        X_tfidf = self.tfidf_vectorizer.transform(raw_text_list)

        # Transform tabular metadata
        X_tabular = self.tabular_pipeline.transform(X)

        # Fuse sparse representations
        X_combined = sparse.hstack([X_tfidf, X_tabular], format="csr")
        return X_combined

    def fit_transform(self, X: pd.DataFrame, y=None) -> sparse.csr_matrix:
        return self.fit(X, y).transform(X)

    def save(self, filepath: Path) -> None:
        """Serializes the fitted preprocessor object to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved fitted preprocessor artifact to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "FullHybridPreprocessor":
        """Loads a serialized preprocessor object."""
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {filepath}")
        obj = joblib.load(filepath)
        logger.info(f"Loaded fitted preprocessor artifact from: {filepath}")
        return obj

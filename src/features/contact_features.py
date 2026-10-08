"""
Contact and External Communication Feature Extractor for Fake Job Postings.

Extracts counts and boolean indicators for contact details embedded in job postings:
- Email addresses (general count and presence)
- Free consumer email domains (e.g. Gmail, Yahoo, Hotmail, ProtonMail)
- URLs and external web links
- Phone and messaging contact numbers

High-performance implementation using pre-compiled, non-backtracking regex findall operations.
"""

import re
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ContactFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts contact and external redirection features from job posting text.
    """

    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}\b")
    FREE_EMAIL_PATTERN = re.compile(r"@(gmail|yahoo|hotmail|outlook|aol|protonmail|ymail|mail|gmx|inbox)\.[a-z]{2,6}", re.IGNORECASE)
    URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+")
    PHONE_PATTERN = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")

    def __init__(self, text_col: str = "clean_composite_text"):
        self.text_col = text_col
        self.feature_names_: List[str] = [
            "email_count",
            "has_email",
            "free_email_count",
            "has_free_email",
            "url_count",
            "has_url",
            "phone_count",
            "has_phone",
        ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ContactFeatureExtractor":
        return self

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return list(self.feature_names_)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.text_col in X.columns:
            raw_texts = [str(v) if pd.notna(v) else "" for v in X[self.text_col]]
        else:
            text_cols = [c for c in ["title", "company_profile", "description", "requirements", "benefits"] if c in X.columns]
            raw_texts = [" ".join(str(X.at[i, c]) for c in text_cols if pd.notna(X.at[i, c])) for i in X.index]

        email_counts = np.array([len(self.EMAIL_PATTERN.findall(t)) for t in raw_texts], dtype=float)
        free_email_counts = np.array([len(self.FREE_EMAIL_PATTERN.findall(t)) for t in raw_texts], dtype=float)
        url_counts = np.array([len(self.URL_PATTERN.findall(t)) for t in raw_texts], dtype=float)
        phone_counts = np.array([len(self.PHONE_PATTERN.findall(t)) for t in raw_texts], dtype=float)

        res = pd.DataFrame({
            "email_count": email_counts,
            "has_email": (email_counts > 0).astype(float),
            "free_email_count": free_email_counts,
            "has_free_email": (free_email_counts > 0).astype(float),
            "url_count": url_counts,
            "has_url": (url_counts > 0).astype(float),
            "phone_count": phone_counts,
            "has_phone": (phone_counts > 0).astype(float),
        }, index=X.index)

        return res

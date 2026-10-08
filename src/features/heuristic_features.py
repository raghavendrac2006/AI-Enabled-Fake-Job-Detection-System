"""
Configurable Suspicious Language and Scam Heuristics Feature Extractor.

Extracts counts, binary presence indicators, and normalized frequencies of scam keywords
across multiple domain categories (payment requests, urgency, guaranteed income, etc.).
Keyword dictionaries are externally configured via YAML.
Ultra-fast execution using pre-compiled regex objects and single-pass lowercase evaluation.
"""

import os
import re
from typing import Dict, List, Optional, Any
import yaml
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HeuristicKeywordExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts structured suspicious language heuristic features from text:
    For each configured category:
    - `{category}_count`: total occurrences of keywords in category
    - `{category}_presence`: 1 if count > 0, else 0
    - `{category}_freq`: count / (total_words + 1)

    Global summary features:
    - `total_scam_keyword_count`: sum of keyword hits across all categories
    - `has_any_scam_keyword`: 1 if total_scam_keyword_count > 0, else 0
    - `scam_category_diversity`: number of distinct scam categories triggered
    """

    DEFAULT_CONFIG_PATH = "config/heuristic_keywords.yaml"

    def __init__(
        self,
        config_path: Optional[str] = None,
        keywords_dict: Optional[Dict[str, Any]] = None,
        text_col: str = "clean_composite_text",
    ):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.keywords_dict = keywords_dict
        self.text_col = text_col
        self.compiled_categories_: Dict[str, re.Pattern] = {}
        self.feature_names_: List[str] = []

    def _load_and_compile(self) -> None:
        if self.keywords_dict is not None:
            raw_categories = self.keywords_dict.get("categories", self.keywords_dict)
        elif os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                raw_categories = loaded.get("categories", {})
        else:
            logger.warning(f"Keyword configuration file '{self.config_path}' not found. Using fallback defaults.")
            raw_categories = {
                "payment_requests": {"keywords": ["wire transfer", "western union", "moneygram", "cashier check", "processing fee"]},
                "urgency": {"keywords": ["urgent", "immediate start", "act now", "apply immediately"]},
                "guaranteed_income": {"keywords": ["guaranteed income", "make money fast", "unlimited earning", "financial freedom"]},
                "work_from_home_claims": {"keywords": ["work from home", "no experience required", "be your own boss"]},
                "personal_info_requests": {"keywords": ["bank details", "ssn", "credit card", "passport copy"]},
                "external_comms": {"keywords": ["telegram", "whatsapp", "skype interview", "google hangout", "wire app"]},
                "cryptocurrency": {"keywords": ["bitcoin", "crypto", "ethereum", "usdt", "wallet address"]},
                "recruitment_fees": {"keywords": ["training fee", "application fee", "visa fee", "equipment fee"]},
            }

        self.compiled_categories_ = {}
        names = []

        for cat_name, cat_data in raw_categories.items():
            kw_list = cat_data.get("keywords", []) if isinstance(cat_data, dict) else cat_data
            if kw_list:
                escaped = []
                for kw in kw_list:
                    kw_str = str(kw).strip().lower()
                    if kw_str:
                        pfx = r"\b" if kw_str[0].isalnum() else ""
                        sfx = r"\b" if kw_str[-1].isalnum() else ""
                        escaped.append(f"{pfx}{re.escape(kw_str)}{sfx}")

                pattern_str = "(?:" + "|".join(escaped) + ")"
                self.compiled_categories_[cat_name] = re.compile(pattern_str)

            names.extend([
                f"{cat_name}_count",
                f"{cat_name}_presence",
                f"{cat_name}_freq",
            ])

        names.extend([
            "total_scam_keyword_count",
            "has_any_scam_keyword",
            "scam_category_diversity",
        ])

        self.feature_names_ = names

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "HeuristicKeywordExtractor":
        self._load_and_compile()
        return self

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        if not self.feature_names_:
            self._load_and_compile()
        return list(self.feature_names_)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.compiled_categories_:
            self._load_and_compile()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.text_col in X.columns:
            raw_texts = [str(v) if pd.notna(v) else "" for v in X[self.text_col]]
        else:
            text_cols = [c for c in ["title", "company_profile", "description", "requirements", "benefits"] if c in X.columns]
            raw_texts = [" ".join(str(X.at[i, c]) for c in text_cols if pd.notna(X.at[i, c])) for i in X.index]

        # Single lowercase conversion and word counting
        lowered_texts = [t.lower() for t in raw_texts]
        word_counts = np.array([len(t.split()) for t in lowered_texts], dtype=float)

        feature_dict = {}
        category_presence_cols = []
        total_counts = np.zeros(len(X), dtype=float)

        for cat_name, pattern in self.compiled_categories_.items():
            counts = np.array([len(pattern.findall(t)) for t in lowered_texts], dtype=float)
            presence = (counts > 0).astype(float)
            freq = counts / np.maximum(word_counts + 1.0, 1.0)

            feature_dict[f"{cat_name}_count"] = counts
            feature_dict[f"{cat_name}_presence"] = presence
            feature_dict[f"{cat_name}_freq"] = freq

            category_presence_cols.append(presence)
            total_counts += counts

        if category_presence_cols:
            category_presence_matrix = np.column_stack(category_presence_cols)
            diversity = np.sum(category_presence_matrix, axis=1)
        else:
            diversity = np.zeros(len(X), dtype=float)

        feature_dict["total_scam_keyword_count"] = total_counts
        feature_dict["has_any_scam_keyword"] = (total_counts > 0).astype(float)
        feature_dict["scam_category_diversity"] = diversity

        res = pd.DataFrame(feature_dict, index=X.index)
        return res

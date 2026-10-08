"""
Safe Salary Feature Extractor for Fake Job Postings.

Extracts presence, numeric bounds, spreads, and rate scale indicators from salary_range.
Includes strict fallback mechanisms for missing or irregular salary expressions
to prevent parsing crashes or target leakage.
"""

import re
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SalaryFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Safely extracts salary metadata without data leakage:
    - has_salary_specified: 1 if salary string is non-empty, 0 otherwise
    - min_salary: parsed minimum salary if formatted as 'min-max', else 0
    - max_salary: parsed maximum salary if formatted as 'min-max', else 0
    - salary_spread: max_salary - min_salary
    - is_hourly_scale: 1 if non-zero salary is <= 250 (heuristic for hourly/daily rate), else 0
    - is_unrealistic_salary: 1 if salary > 1,000,000 or negative (scam indicator / anomaly), else 0
    """

    def __init__(self, salary_col: str = "salary_range", max_reasonable: int = 10_000_000):
        self.salary_col = salary_col
        self.max_reasonable = max_reasonable
        self.feature_names_: List[str] = [
            "has_salary_specified",
            "salary_min",
            "salary_max",
            "salary_spread",
            "is_hourly_scale",
            "is_unrealistic_salary",
        ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "SalaryFeatureExtractor":
        return self

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return list(self.feature_names_)

    def _parse_salary_entry(self, val: str):
        if not val or pd.isna(val) or not str(val).strip():
            return 0, 0.0, 0.0, 0.0, 0, 0

        val_str = str(val).strip()
        has_salary = 1

        # Match patterns like '10000-20000', '10-20', etc.
        match = re.match(r"^(\d+)\s*[-–—/]\s*(\d+)$", val_str)
        if match:
            try:
                s_min = float(match.group(1))
                s_max = float(match.group(2))

                # Handle reversed or malformed intervals
                if s_min > s_max:
                    s_min, s_max = s_max, s_min

                spread = max(0.0, s_max - s_min)
                is_hourly = 1 if (0 < s_max <= 250) else 0
                is_unrealistic = 1 if (s_max > self.max_reasonable or s_min < 0) else 0

                return has_salary, s_min, s_max, spread, is_hourly, is_unrealistic
            except (ValueError, OverflowError):
                return has_salary, 0.0, 0.0, 0.0, 0, 1

        # Single numeric salary match
        single_match = re.match(r"^(\d+)$", val_str)
        if single_match:
            try:
                s_val = float(single_match.group(1))
                spread = 0.0
                is_hourly = 1 if (0 < s_val <= 250) else 0
                is_unrealistic = 1 if (s_val > self.max_reasonable or s_val < 0) else 0
                return has_salary, s_val, s_val, spread, is_hourly, is_unrealistic
            except (ValueError, OverflowError):
                return has_salary, 0.0, 0.0, 0.0, 0, 1

        # Unparseable but present
        return has_salary, 0.0, 0.0, 0.0, 0, 0

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.salary_col not in X.columns:
            # Fallback all zeros
            res = pd.DataFrame(0.0, index=X.index, columns=self.feature_names_)
            return res

        parsed = X[self.salary_col].apply(self._parse_salary_entry)
        
        has_salary = [p[0] for p in parsed]
        salary_min = [p[1] for p in parsed]
        salary_max = [p[2] for p in parsed]
        salary_spread = [p[3] for p in parsed]
        is_hourly = [p[4] for p in parsed]
        is_unrealistic = [p[5] for p in parsed]

        res = pd.DataFrame({
            "has_salary_specified": has_salary,
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_spread": salary_spread,
            "is_hourly_scale": is_hourly,
            "is_unrealistic_salary": is_unrealistic,
        }, index=X.index)

        return res

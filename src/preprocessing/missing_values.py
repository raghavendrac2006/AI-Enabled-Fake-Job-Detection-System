"""
Missing Value Handling Module.
AI-Powered Fake Job and Recruitment Scam Detection System.

Implements leak-free missingness transformations:
1. Imputes missing categorical attributes with a dedicated '__MISSING__' category
2. Fills missing binary indicators safely (defaulting to 0 / False)
3. Preserves domain missingness indicators (e.g. has_salary_specified)
"""

from typing import List, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger

logger = get_logger("preprocessing.missing_values")

DEFAULT_CATEGORICAL_COLS = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

DEFAULT_BINARY_COLS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
]


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for handling missing values
    in categorical and binary metadata fields.
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        binary_columns: Optional[List[str]] = None,
        categorical_fill: str = "__MISSING__",
        binary_fill: int = 0,
    ):
        self.categorical_columns = categorical_columns or DEFAULT_CATEGORICAL_COLS
        self.binary_columns = binary_columns or DEFAULT_BINARY_COLS
        self.categorical_fill = categorical_fill
        self.binary_fill = binary_fill

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        df_out = X.copy()

        # Handle Categorical Columns
        for col in self.categorical_columns:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(self.categorical_fill).astype(str)

        # Handle Binary Columns
        for col in self.binary_columns:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(self.binary_fill).astype(int)

        return df_out

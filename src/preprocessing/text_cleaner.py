"""
Text Normalization and Composite Text Construction Module.
AI-Powered Fake Job and Recruitment Scam Detection System.

Provides robust, reproducible text cleaning:
1. HTML tag removal and HTML entity unescaping
2. Unicode NFKC normalization
3. Whitespace and formatting normalization
4. Preservation of semantic punctuation and domain tokens
5. Structured composite text construction with explicit section boundaries
"""

import html
import re
import unicodedata
from typing import List, Optional, Union
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger

logger = get_logger("preprocessing.text_cleaner")

# Compiled regex patterns for fast vectorized processing
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>", re.IGNORECASE)
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_MULTIPLE_WHITESPACE = re.compile(r"\s+")
_SECTION_CLEANUP = re.compile(r"\n\s*\n+")

# Standard section prefixes for composite text
SECTION_HEADERS = {
    "title": "TITLE",
    "company_profile": "COMPANY PROFILE",
    "description": "DESCRIPTION",
    "requirements": "REQUIREMENTS",
    "benefits": "BENEFITS",
}


def clean_text_string(
    text: Optional[Union[str, float]],
    lowercase: bool = True,
    strip_html: bool = True,
    unescape_entities: bool = True,
    normalize_unicode: bool = True,
    preserve_urls: bool = False,
) -> str:
    """
    Cleans a single text string safely and deterministically.

    Args:
        text: Input string (or NaN/None)
        lowercase: Whether to convert text to lowercase
        strip_html: Whether to strip HTML tags
        unescape_entities: Whether to decode &amp;, &lt;, etc.
        normalize_unicode: Apply NFKC unicode normalization
        preserve_urls: If False, replaces URLs with a standard token

    Returns:
        Cleaned string (empty string if input was missing)
    """
    if text is None or pd.isna(text):
        return ""

    text_str = str(text)

    # 1. Unicode NFKC normalization
    if normalize_unicode:
        text_str = unicodedata.normalize("NFKC", text_str)

    # 2. Decode HTML entities (&amp; -> &, &quot; -> ", etc.)
    if unescape_entities:
        text_str = html.unescape(text_str)

    # 3. Strip HTML tags
    if strip_html:
        text_str = _HTML_TAG_PATTERN.sub(" ", text_str)

    # 4. Handle URLs & Emails without aggressive destruction
    if not preserve_urls:
        text_str = _URL_PATTERN.sub(" [URL] ", text_str)
        text_str = _EMAIL_PATTERN.sub(" [EMAIL] ", text_str)

    # 5. Normalize whitespace while preserving line structure
    text_str = _MULTIPLE_WHITESPACE.sub(" ", text_str).strip()

    # 6. Lowercase
    if lowercase:
        text_str = text_str.lower()

    return text_str


def build_composite_text(
    row: pd.Series,
    text_columns: Optional[List[str]] = None,
    lowercase: bool = True,
) -> str:
    """
    Combines individual text columns into a single composite representation
    with clear section headers so semantic field boundaries are preserved.

    Format:
        TITLE: <cleaned title>
        COMPANY PROFILE: <cleaned company_profile>
        DESCRIPTION: <cleaned description>
        REQUIREMENTS: <cleaned requirements>
        BENEFITS: <cleaned benefits>
    """
    cols = text_columns or ["title", "company_profile", "description", "requirements", "benefits"]
    sections = []

    for col in cols:
        header = SECTION_HEADERS.get(col, col.upper().replace("_", " "))
        val = clean_text_string(row.get(col, ""), lowercase=lowercase)
        if val:
            sections.append(f"{header}: {val}")
        else:
            sections.append(f"{header}: [UNSPECIFIED]")

    return "\n".join(sections)


class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that applies text cleaning
    and outputs cleaned composite text strings for downstream vectorizers.
    """

    def __init__(
        self,
        text_columns: Optional[List[str]] = None,
        lowercase: bool = True,
        strip_html: bool = True,
    ):
        self.text_columns = text_columns or ["title", "company_profile", "description", "requirements", "benefits"]
        self.lowercase = lowercase
        self.strip_html = strip_html

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless transformer
        return self

    def transform(self, X: pd.DataFrame) -> List[str]:
        """Transforms a DataFrame of raw text columns into a list of composite text strings."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        logger.debug(f"Cleaning text across {len(X)} rows...")
        return [
            build_composite_text(row, text_columns=self.text_columns, lowercase=self.lowercase)
            for _, row in X.iterrows()
        ]

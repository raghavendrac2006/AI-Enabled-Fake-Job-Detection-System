"""
Data Leakage and Proxy Feature Investigation Module.
AI-Powered Fake Job and Recruitment Scam Detection System.

Analyzes dataset columns to identify potential data leakage risks,
target-derived artifacts, and fields unsuitable for real-time production inference.
"""

from typing import Any, Dict, List
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data.leakage_checker")


def review_data_leakage(df: pd.DataFrame, target_col: str = "fraudulent") -> List[Dict[str, str]]:
    """
    Performs a systematic assessment of potential data leakage across all columns.
    """
    reviews = []

    for col in df.columns:
        if col == target_col:
            reviews.append({
                "column": col,
                "potential_leakage": "YES (Ground Truth Target)",
                "risk_level": "CRITICAL",
                "reason": "This is the primary prediction target.",
                "recommended_action": "Isolate as target label 'y'. Never include in feature matrix 'X'.",
            })
        elif col == "job_id":
            reviews.append({
                "column": col,
                "potential_leakage": "YES (Database Key Leakage Risk)",
                "risk_level": "HIGH",
                "reason": "Sequential integer ID. If fraudulent postings were scraped in batches or added consecutively, models might memorize ID ranges instead of genuine textual/domain signals.",
                "recommended_action": "Drop 'job_id' completely from ML feature vectors.",
            })
        elif col == "salary_range":
            reviews.append({
                "column": col,
                "potential_leakage": "LOW (High Missingness / Heuristic Proxy)",
                "risk_level": "LOW",
                "reason": "Approximately 83% missing values. Real-world postings legitimately omit salary, but scammers often omit or place unrealistic ranges.",
                "recommended_action": "Extract binary flag 'has_salary_specified' and parse numeric boundaries; impute missing with a dedicated indicator token.",
            })
        elif col in ["title", "company_profile", "description", "requirements", "benefits"]:
            reviews.append({
                "column": col,
                "potential_leakage": "NO (Core Unstructured Feature)",
                "risk_level": "NONE",
                "reason": "Realistic text available during job posting submission.",
                "recommended_action": "Retain for NLP feature extraction, TF-IDF, and Transformer fine-tuning.",
            })
        elif col in ["telecommuting", "has_company_logo", "has_questions"]:
            reviews.append({
                "column": col,
                "potential_leakage": "NO (Valid Metadata Signal)",
                "risk_level": "NONE",
                "reason": "Binary platform flags visible at posting creation time.",
                "recommended_action": "Retain as tabular binary features.",
            })
        elif col in ["employment_type", "required_experience", "required_education", "industry", "function", "department", "location"]:
            reviews.append({
                "column": col,
                "potential_leakage": "NO (Categorical Metadata)",
                "risk_level": "NONE",
                "reason": "Standard structured form fields completed by the job poster.",
                "recommended_action": "Handle missing values with 'Missing' category; apply frequency/one-hot encoding without target leakage.",
            })
        else:
            reviews.append({
                "column": col,
                "potential_leakage": "UNCERTAIN (Unknown Column)",
                "risk_level": "MEDIUM",
                "reason": "Non-standard column outside expected EMSCAD schema.",
                "recommended_action": "Audit distribution and inspect unique values manually.",
            })

    return reviews

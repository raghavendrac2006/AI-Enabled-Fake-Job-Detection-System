"""
Modular Feature Builder and Experimentation Framework for Fake Job Detection.

Orchestrates the creation, fitting, transformation, and extraction of distinct
named feature groups for flexible ablation studies and hybrid model training:
- TEXT_TFIDF
- TEXT_STATISTICS
- METADATA_BINARY
- METADATA_CATEGORICAL
- MISSINGNESS
- SALARY
- CONTACT
- HEURISTIC
- ALL_TABULAR (Combination of all structured/tabular features)
- HYBRID_ALL (Full combination of TF-IDF and all tabular features)

Strictly ensures that all transformations with learned parameters are fitted
ONLY on the training split.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.features.text_features import TextStatisticsExtractor
from src.features.salary_features import SalaryFeatureExtractor
from src.features.contact_features import ContactFeatureExtractor
from src.features.heuristic_features import HeuristicKeywordExtractor
from src.features.metadata_features import (
    MetadataBinaryExtractor,
    MetadataCategoricalExtractor,
)
from src.features.interfaces import IRepresentationA
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModularFeatureBuilder(BaseEstimator, TransformerMixin, IRepresentationA):
    """
    Unified feature engineering coordinator supporting ablation over named feature groups.
    """

    AVAILABLE_GROUPS = [
        "TEXT_TFIDF",
        "TEXT_STATISTICS",
        "METADATA_BINARY",
        "METADATA_CATEGORICAL",
        "MISSINGNESS",
        "SALARY",
        "CONTACT",
        "HEURISTIC",
    ]

    def __init__(
        self,
        tfidf_max_features: int = 10000,
        tfidf_ngram_range: Tuple[int, int] = (1, 2),
        tfidf_min_df: int = 3,
        tfidf_max_df: float = 0.95,
        tfidf_sublinear_tf: bool = True,
        heuristic_config_path: Optional[str] = "config/heuristic_keywords.yaml",
        active_groups: Optional[List[str]] = None,
    ):
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_sublinear_tf = tfidf_sublinear_tf
        self.heuristic_config_path = heuristic_config_path
        self.active_groups = active_groups or self.AVAILABLE_GROUPS

        # Individual sub-extractors
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram_range,
            min_df=self.tfidf_min_df,
            max_df=self.tfidf_max_df,
            sublinear_tf=self.tfidf_sublinear_tf,
            norm="l2",
            strip_accents="unicode",
            lowercase=True,
        )
        self.text_stats_extractor = TextStatisticsExtractor()
        self.salary_extractor = SalaryFeatureExtractor()
        self.contact_extractor = ContactFeatureExtractor()
        self.heuristic_extractor = HeuristicKeywordExtractor(config_path=self.heuristic_config_path)
        self.metadata_binary_extractor = MetadataBinaryExtractor()
        self.metadata_cat_extractor = MetadataCategoricalExtractor(include_country=True, sparse_output=True)
        self.tabular_scaler = StandardScaler(with_mean=False)

        self.is_fitted_ = False
        self.group_feature_names_: Dict[str, List[str]] = {}

    @property
    def representation_name(self) -> str:
        return "Representation A: TF-IDF + Tabular Metadata"

    @property
    def is_sparse(self) -> bool:
        return True

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ModularFeatureBuilder":
        """
        Fit all extractors strictly on training data.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        logger.info("Fitting ModularFeatureBuilder on training data...")

        # 1. Text TF-IDF
        comp_text = (
            X["clean_composite_text"].fillna("").astype(str)
            if "clean_composite_text" in X.columns
            else X[["title", "company_profile", "description", "requirements", "benefits"]]
            .fillna("")
            .agg(" ".join, axis=1)
        )
        self.tfidf_vectorizer.fit(comp_text)
        self.group_feature_names_["TEXT_TFIDF"] = [
            f"tfidf_{w}" for w in self.tfidf_vectorizer.get_feature_names_out()
        ]

        # 2. Text Statistics
        self.text_stats_extractor.fit(X)
        self.group_feature_names_["TEXT_STATISTICS"] = self.text_stats_extractor.get_feature_names_out()

        # 3. Salary Features
        self.salary_extractor.fit(X)
        self.group_feature_names_["SALARY"] = self.salary_extractor.get_feature_names_out()

        # 4. Contact Features
        self.contact_extractor.fit(X)
        self.group_feature_names_["CONTACT"] = self.contact_extractor.get_feature_names_out()

        # 5. Heuristic Features
        self.heuristic_extractor.fit(X)
        self.group_feature_names_["HEURISTIC"] = self.heuristic_extractor.get_feature_names_out()

        # 6. Metadata Binary & Missingness
        self.metadata_binary_extractor.fit(X)
        all_bin_names = self.metadata_binary_extractor.get_feature_names_out()
        self.group_feature_names_["METADATA_BINARY"] = [
            n for n in all_bin_names if n in ["telecommuting", "has_company_logo", "has_questions"]
        ]
        self.group_feature_names_["MISSINGNESS"] = [
            n for n in all_bin_names if n not in ["telecommuting", "has_company_logo", "has_questions"]
        ]

        # 7. Metadata Categorical
        self.metadata_cat_extractor.fit(X)
        self.group_feature_names_["METADATA_CATEGORICAL"] = self.metadata_cat_extractor.get_feature_names_out()

        # Fit Tabular Scaler on combined dense tabular features
        df_dense_tab = self._extract_dense_tabular(X)
        self.tabular_scaler.fit(df_dense_tab)

        self.is_fitted_ = True
        logger.info("ModularFeatureBuilder fit completed successfully.")
        return self

    def _extract_dense_tabular(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract and concatenate all non-categorical dense tabular features."""
        df_stats = self.text_stats_extractor.transform(X)
        df_sal = self.salary_extractor.transform(X)
        df_cont = self.contact_extractor.transform(X)
        df_heur = self.heuristic_extractor.transform(X)
        df_bin = self.metadata_binary_extractor.transform(X)

        return pd.concat([df_stats, df_sal, df_cont, df_heur, df_bin], axis=1)

    def extract_group(self, group_name: str, X: pd.DataFrame) -> Union[np.ndarray, sparse.csr_matrix]:
        """
        Extract a specific feature group for input dataframe X.
        """
        if not self.is_fitted_:
            raise ValueError("ModularFeatureBuilder must be fitted before extracting feature groups.")

        if group_name == "TEXT_TFIDF":
            comp_text = (
                X["clean_composite_text"].fillna("").astype(str)
                if "clean_composite_text" in X.columns
                else X[["title", "company_profile", "description", "requirements", "benefits"]]
                .fillna("")
                .agg(" ".join, axis=1)
            )
            return self.tfidf_vectorizer.transform(comp_text)

        elif group_name == "TEXT_STATISTICS":
            return self.text_stats_extractor.transform(X).to_numpy(dtype=float)

        elif group_name == "SALARY":
            return self.salary_extractor.transform(X).to_numpy(dtype=float)

        elif group_name == "CONTACT":
            return self.contact_extractor.transform(X).to_numpy(dtype=float)

        elif group_name == "HEURISTIC":
            return self.heuristic_extractor.transform(X).to_numpy(dtype=float)

        elif group_name == "METADATA_BINARY":
            df_bin = self.metadata_binary_extractor.transform(X)
            cols = self.group_feature_names_["METADATA_BINARY"]
            return df_bin[cols].to_numpy(dtype=float)

        elif group_name == "MISSINGNESS":
            df_bin = self.metadata_binary_extractor.transform(X)
            cols = self.group_feature_names_["MISSINGNESS"]
            return df_bin[cols].to_numpy(dtype=float)

        elif group_name == "METADATA_CATEGORICAL":
            return self.metadata_cat_extractor.transform(X)

        elif group_name == "ALL_TABULAR":
            df_dense = self._extract_dense_tabular(X)
            dense_scaled = self.tabular_scaler.transform(df_dense)
            cat_sparse = self.metadata_cat_extractor.transform(X)
            return sparse.hstack([sparse.csr_matrix(dense_scaled), cat_sparse], format="csr")

        elif group_name in ["HYBRID_ALL", "ALL"]:
            return self.transform(X)

        else:
            raise ValueError(f"Unknown feature group: '{group_name}'. Available: {self.AVAILABLE_GROUPS}")

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        """
        Transform X into the complete hybrid feature matrix (TF-IDF + All Scaled Tabular).
        """
        if not self.is_fitted_:
            raise ValueError("ModularFeatureBuilder must be fitted before transform.")

        # 1. TF-IDF Matrix (Sparse)
        tfidf_mat = self.extract_group("TEXT_TFIDF", X)

        # 2. All Tabular (Dense scaled + Categorical sparse)
        tabular_mat = self.extract_group("ALL_TABULAR", X)

        # 3. Stack horizontally into CSR matrix
        hybrid_mat = sparse.hstack([tfidf_mat, tabular_mat], format="csr")
        return hybrid_mat

    def get_feature_names(self, group: Optional[str] = None) -> List[str]:
        """
        Get feature names for a specific group or for the entire hybrid feature set.
        """
        if not self.is_fitted_:
            raise ValueError("ModularFeatureBuilder must be fitted first.")

        if group is not None:
            if group in self.group_feature_names_:
                return list(self.group_feature_names_[group])
            elif group == "ALL_TABULAR":
                dense_names = (
                    self.group_feature_names_["TEXT_STATISTICS"]
                    + self.group_feature_names_["SALARY"]
                    + self.group_feature_names_["CONTACT"]
                    + self.group_feature_names_["HEURISTIC"]
                    + self.group_feature_names_["METADATA_BINARY"]
                    + self.group_feature_names_["MISSINGNESS"]
                )
                cat_names = self.group_feature_names_["METADATA_CATEGORICAL"]
                return dense_names + cat_names
            elif group in ["HYBRID_ALL", "ALL"]:
                return self.get_feature_names()
            else:
                raise ValueError(f"Unknown group '{group}'.")

        # Full hybrid feature names
        tfidf_names = self.group_feature_names_["TEXT_TFIDF"]
        dense_names = (
            self.group_feature_names_["TEXT_STATISTICS"]
            + self.group_feature_names_["SALARY"]
            + self.group_feature_names_["CONTACT"]
            + self.group_feature_names_["HEURISTIC"]
            + self.group_feature_names_["METADATA_BINARY"]
            + self.group_feature_names_["MISSINGNESS"]
        )
        cat_names = self.group_feature_names_["METADATA_CATEGORICAL"]
        return tfidf_names + dense_names + cat_names

    def get_dimensions_summary(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes exact dimension, sparsity, and shape metadata across all feature groups efficiently.
        """
        if not self.is_fitted_:
            self.fit(X_train)

        # 1. Transform each base group for train & test
        train_groups = {}
        test_groups = {}

        # TF-IDF
        train_groups["TEXT_TFIDF"] = self.extract_group("TEXT_TFIDF", X_train)
        test_groups["TEXT_TFIDF"] = self.extract_group("TEXT_TFIDF", X_test)

        # Dense sub-extractors
        train_stats = self.text_stats_extractor.transform(X_train)
        test_stats = self.text_stats_extractor.transform(X_test)
        train_groups["TEXT_STATISTICS"] = train_stats.to_numpy(dtype=float)
        test_groups["TEXT_STATISTICS"] = test_stats.to_numpy(dtype=float)

        train_sal = self.salary_extractor.transform(X_train)
        test_sal = self.salary_extractor.transform(X_test)
        train_groups["SALARY"] = train_sal.to_numpy(dtype=float)
        test_groups["SALARY"] = test_sal.to_numpy(dtype=float)

        train_cont = self.contact_extractor.transform(X_train)
        test_cont = self.contact_extractor.transform(X_test)
        train_groups["CONTACT"] = train_cont.to_numpy(dtype=float)
        test_groups["CONTACT"] = test_cont.to_numpy(dtype=float)

        train_heur = self.heuristic_extractor.transform(X_train)
        test_heur = self.heuristic_extractor.transform(X_test)
        train_groups["HEURISTIC"] = train_heur.to_numpy(dtype=float)
        test_groups["HEURISTIC"] = test_heur.to_numpy(dtype=float)

        train_bin_all = self.metadata_binary_extractor.transform(X_train)
        test_bin_all = self.metadata_binary_extractor.transform(X_test)
        bin_cols = self.group_feature_names_["METADATA_BINARY"]
        miss_cols = self.group_feature_names_["MISSINGNESS"]
        train_groups["METADATA_BINARY"] = train_bin_all[bin_cols].to_numpy(dtype=float)
        test_groups["METADATA_BINARY"] = test_bin_all[bin_cols].to_numpy(dtype=float)
        train_groups["MISSINGNESS"] = train_bin_all[miss_cols].to_numpy(dtype=float)
        test_groups["MISSINGNESS"] = test_bin_all[miss_cols].to_numpy(dtype=float)

        # Categorical
        train_groups["METADATA_CATEGORICAL"] = self.metadata_cat_extractor.transform(X_train)
        test_groups["METADATA_CATEGORICAL"] = self.metadata_cat_extractor.transform(X_test)

        # All Tabular
        df_dense_tr = pd.concat([train_stats, train_sal, train_cont, train_heur, train_bin_all], axis=1)
        df_dense_te = pd.concat([test_stats, test_sal, test_cont, test_heur, test_bin_all], axis=1)
        scaled_dense_tr = self.tabular_scaler.transform(df_dense_tr)
        scaled_dense_te = self.tabular_scaler.transform(df_dense_te)

        train_groups["ALL_TABULAR"] = sparse.hstack([sparse.csr_matrix(scaled_dense_tr), train_groups["METADATA_CATEGORICAL"]], format="csr")
        test_groups["ALL_TABULAR"] = sparse.hstack([sparse.csr_matrix(scaled_dense_te), test_groups["METADATA_CATEGORICAL"]], format="csr")

        # Hybrid All
        train_groups["HYBRID_ALL"] = sparse.hstack([train_groups["TEXT_TFIDF"], train_groups["ALL_TABULAR"]], format="csr")
        test_groups["HYBRID_ALL"] = sparse.hstack([test_groups["TEXT_TFIDF"], test_groups["ALL_TABULAR"]], format="csr")

        summary = {}
        for group_name in self.AVAILABLE_GROUPS + ["ALL_TABULAR", "HYBRID_ALL"]:
            tr_mat = train_groups[group_name]
            te_mat = test_groups[group_name]

            is_sp = sparse.issparse(tr_mat)
            if is_sp:
                nnz = tr_mat.nnz
                total_elements = tr_mat.shape[0] * tr_mat.shape[1]
                sparsity = (1.0 - (nnz / total_elements)) * 100.0
            else:
                nnz = np.count_nonzero(tr_mat)
                total_elements = tr_mat.shape[0] * tr_mat.shape[1]
                sparsity = (1.0 - (nnz / total_elements)) * 100.0

            summary[group_name] = {
                "num_features": int(tr_mat.shape[1]),
                "train_shape": [int(s) for s in tr_mat.shape],
                "test_shape": [int(s) for s in te_mat.shape],
                "is_sparse": bool(is_sp),
                "sparsity_percent": round(float(sparsity), 2),
            }

        return summary

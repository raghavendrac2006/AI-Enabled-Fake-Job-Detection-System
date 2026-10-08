"""
Feature Representation Interfaces and Contracts for Fake Job Detection.

Defines the core abstract contracts for feature representation families:
- Representation A: TF-IDF + Tabular Metadata (Implemented in Phase 3)
- Representation B: Word Embeddings (Word2Vec / FastText / GloVe) + Tabular Metadata (Planned for Phase 5)
- Representation C: Pretrained Transformer Embeddings + Tabular Metadata (Planned for Phase 5)
- Representation D: Tokenized Text Sequences for Deep Learning / Fine-Tuning (Planned for Phase 5/6)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import sparse


class BaseFeatureRepresentation(ABC):
    """
    Abstract base class for all feature representation extractors.
    Guarantees consistent fit/transform semantics and metadata extraction.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseFeatureRepresentation":
        """Fit feature representation parameters strictly on training data."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> Union[np.ndarray, sparse.csr_matrix]:
        """Transform input data into the specified numerical representation."""
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Union[np.ndarray, sparse.csr_matrix]:
        """Fit on X (and optionally y) and return transformed representation."""
        return self.fit(X, y).transform(X)

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return the names of all generated features."""
        pass

    @property
    @abstractmethod
    def representation_name(self) -> str:
        """Return human-readable name of representation."""
        pass

    @property
    @abstractmethod
    def is_sparse(self) -> bool:
        """Return True if representation produces a sparse matrix."""
        pass


class IRepresentationA(BaseFeatureRepresentation):
    """Interface for Representation A: TF-IDF n-grams + Tabular Metadata."""
    pass


class IRepresentationB(BaseFeatureRepresentation):
    """
    Interface for Representation B: Word Embedding aggregations (e.g. Mean/Weighted Word2Vec/GloVe)
    combined with tabular features.
    """
    @abstractmethod
    def set_embedding_dim(self, dim: int) -> None:
        pass


class IRepresentationC(BaseFeatureRepresentation):
    """
    Interface for Representation C: Dense pooled embeddings from pretrained transformer backbones
    (e.g., DeBERTa-v3, RoBERTa, DistilBERT) combined with tabular features.
    """
    @abstractmethod
    def set_model_checkpoint(self, model_name: str) -> None:
        pass


class IRepresentationD(ABC):
    """
    Interface for Representation D: Tokenized sequence tensors and attention masks
    formatted for PyTorch / HuggingFace DataLoader pipelines.
    """
    @abstractmethod
    def tokenize(self, texts: List[str], max_length: int = 512) -> Dict[str, Any]:
        """Tokenize raw text inputs into model-ready tensor dictionaries."""
        pass

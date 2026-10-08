"""
Feature engineering, representations, modular builders, and catalog registry.
"""

from src.features.text_features import TextStatisticsExtractor
from src.features.salary_features import SalaryFeatureExtractor
from src.features.contact_features import ContactFeatureExtractor
from src.features.heuristic_features import HeuristicKeywordExtractor
from src.features.metadata_features import (
    MetadataBinaryExtractor,
    MetadataCategoricalExtractor,
)
from src.features.interfaces import (
    BaseFeatureRepresentation,
    IRepresentationA,
    IRepresentationB,
    IRepresentationC,
    IRepresentationD,
)
from src.features.builder import ModularFeatureBuilder
from src.features.feature_catalog import (
    FEATURE_CATALOG_REGISTRY,
    generate_markdown_catalog,
)

__all__ = [
    "TextStatisticsExtractor",
    "SalaryFeatureExtractor",
    "ContactFeatureExtractor",
    "HeuristicKeywordExtractor",
    "MetadataBinaryExtractor",
    "MetadataCategoricalExtractor",
    "BaseFeatureRepresentation",
    "IRepresentationA",
    "IRepresentationB",
    "IRepresentationC",
    "IRepresentationD",
    "ModularFeatureBuilder",
    "FEATURE_CATALOG_REGISTRY",
    "generate_markdown_catalog",
]

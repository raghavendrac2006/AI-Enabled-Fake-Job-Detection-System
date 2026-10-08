"""
Type-safe configuration loader for the Fake Job Detection System.
Supports loading from YAML and environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AppConfig:
    """Configuration container loaded from YAML / defaults."""

    def __init__(self, raw_config: Optional[Dict[str, Any]] = None):
        self._raw = raw_config or {}

        # Project metadata
        project = self._raw.get("project", {})
        self.project_name: str = project.get("name", "fake-job-detection")
        self.version: str = project.get("version", "0.2.0")
        self.random_seed: int = int(project.get("random_seed", 42))

        # Paths
        paths = self._raw.get("paths", {})
        self.raw_data_dir: Path = PROJECT_ROOT / paths.get("raw_data_dir", "data/raw")
        self.processed_data_dir: Path = PROJECT_ROOT / paths.get("processed_data_dir", "data/processed")
        self.external_data_dir: Path = PROJECT_ROOT / paths.get("external_data_dir", "data/external")
        self.models_dir: Path = PROJECT_ROOT / paths.get("models_dir", "models")
        self.reports_dir: Path = PROJECT_ROOT / paths.get("reports_dir", "reports")
        self.figures_dir: Path = PROJECT_ROOT / paths.get("figures_dir", "reports/figures")
        self.raw_dataset_filename: str = paths.get("raw_dataset_filename", "fake_job_postings.csv")

        # Data schema config
        data = self._raw.get("data", {})
        self.target_column: str = data.get("target_column", "fraudulent")
        self.id_column: str = data.get("id_column", "job_id")
        self.text_columns: List[str] = data.get("text_columns", [
            "title", "company_profile", "description", "requirements", "benefits"
        ])
        self.categorical_columns: List[str] = data.get("categorical_columns", [
            "employment_type", "required_experience", "required_education", "industry", "function"
        ])
        self.binary_columns: List[str] = data.get("binary_columns", [
            "telecommuting", "has_company_logo", "has_questions"
        ])
        self.split_config: Dict[str, Any] = data.get("split", {
            "test_size": 0.20,
            "stratify": True,
            "random_state": 42,
        })

        # Preprocessing & modeling
        self.preprocessing: Dict[str, Any] = self._raw.get("preprocessing", {})
        self.sampling: Dict[str, Any] = self._raw.get("sampling", {})
        self.models: Dict[str, Any] = self._raw.get("models", {})
        self.evaluation: Dict[str, Any] = self._raw.get("evaluation", {})
        self.logging: Dict[str, Any] = self._raw.get("logging", {})

    @property
    def raw_dataset_path(self) -> Path:
        return self.raw_data_dir / self.raw_dataset_filename

    def to_dict(self) -> Dict[str, Any]:
        return self._raw


_CONFIG_INSTANCE: Optional[AppConfig] = None


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Loads configuration from the specified YAML file or default config/config.yaml.
    """
    global _CONFIG_INSTANCE

    target_path = config_path or (PROJECT_ROOT / "config" / "config.yaml")

    raw_dict: Dict[str, Any] = {}
    if target_path.exists():
        try:
            import yaml
            with open(target_path, "r", encoding="utf-8") as f:
                raw_dict = yaml.safe_load(f) or {}
        except ImportError:
            raw_dict = {}
        except Exception as e:
            print(f"Warning: Failed to parse {target_path} ({e}). Using default settings.")

    _CONFIG_INSTANCE = AppConfig(raw_dict)
    return _CONFIG_INSTANCE


def get_config() -> AppConfig:
    """
    Returns the cached configuration singleton.
    """
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        _CONFIG_INSTANCE = load_config()
    return _CONFIG_INSTANCE

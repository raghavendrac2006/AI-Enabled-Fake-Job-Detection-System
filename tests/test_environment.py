"""
Sanity tests for project structure, configuration, and logging setup.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config
from src.utils.logger import get_logger, setup_logging


class TestProjectFoundation(unittest.TestCase):
    """Tests Phase 0 directory structure, config integrity, and logging."""

    def test_project_directories_exist(self):
        config = get_config()
        self.assertTrue(config.raw_data_dir.exists(), f"Missing: {config.raw_data_dir}")
        self.assertTrue(config.processed_data_dir.exists(), f"Missing: {config.processed_data_dir}")
        self.assertTrue(config.external_data_dir.exists(), f"Missing: {config.external_data_dir}")
        self.assertTrue(config.models_dir.exists(), f"Missing: {config.models_dir}")
        self.assertTrue(config.reports_dir.exists(), f"Missing: {config.reports_dir}")

    def test_config_defaults(self):
        config = get_config()
        self.assertEqual(config.random_seed, 42)
        self.assertEqual(config.target_column, "fraudulent")
        self.assertIn("description", config.text_columns)
        self.assertIn("has_company_logo", config.binary_columns)

    def test_logger_initialization(self):
        logger = setup_logging(module_name="test_runner")
        self.assertIsNotNone(logger)
        child_logger = get_logger("unit_test")
        self.assertIsNotNone(child_logger)


if __name__ == "__main__":
    unittest.main()

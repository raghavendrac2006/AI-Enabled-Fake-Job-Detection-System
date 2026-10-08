#!/usr/bin/env python
"""
AI-Powered Fake Job and Recruitment Scam Detection System
Main Entry Point
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config
from src.utils.logger import setup_logging, get_logger


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Powered Fake Job and Recruitment Scam Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verify-env",
        action="store_true",
        help="Run environment and dependency verification.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print project configuration and metadata.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    logger = setup_logging(module_name="main")
    config = get_config()

    logger.info("Initializing Fake Job Detection System (Phase 0: Project Foundation)...")

    if args.verify_env:
        from verify_environment import run_verification
        return run_verification()

    if args.info:
        print("\n" + "=" * 60)
        print("PROJECT METADATA & CONFIGURATION")
        print("=" * 60)
        print(f"Project Name      : {config.project_name}")
        print(f"Version           : {config.version}")
        print(f"Random Seed       : {config.random_seed}")
        print(f"Raw Data Path     : {config.raw_dataset_path}")
        print(f"Target Column     : {config.target_column}")
        print(f"Text Columns      : {', '.join(config.text_columns)}")
        print(f"Categorical Cols  : {', '.join(config.categorical_columns)}")
        print(f"Binary Cols       : {', '.join(config.binary_columns)}")
        print("=" * 60 + "\n")
        return 0

    # Default message when no flags provided
    print("\n[+] Fake Job Detection System initialized successfully.")
    print("    Use '--verify-env' to check installed dependencies.")
    print("    Use '--info' to inspect configuration settings.")
    print("    Ready for Phase 1 (Dataset Ingestion & EDA).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

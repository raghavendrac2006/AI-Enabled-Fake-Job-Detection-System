#!/usr/bin/env python
"""
Environment Verification Script
AI-Powered Fake Job and Recruitment Scam Detection System

Checks Python version, required core machine learning packages, NLP libraries,
and PyTorch hardware acceleration availability.
"""

import sys
import importlib
from typing import Dict, Tuple, Optional


def check_python_version() -> Tuple[bool, str]:
    """Verify Python version is at least 3.9+ and ideally 3.10-3.12."""
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro}"
    is_compatible = (major == 3 and minor >= 9)
    return is_compatible, version_str


def check_package(package_name: str, import_name: Optional[str] = None) -> Tuple[bool, str]:
    """
    Checks if a Python module can be imported and extracts its __version__.
    """
    target = import_name or package_name
    try:
        mod = importlib.import_module(target)
        version = getattr(mod, "__version__", "installed (no __version__)")
        return True, str(version)
    except ImportError:
        return False, "NOT INSTALLED"
    except Exception as e:
        return False, f"ERROR: {e}"


def check_torch_details() -> Dict[str, str]:
    """Inspect PyTorch hardware acceleration status."""
    details = {}
    try:
        import torch
        details["version"] = torch.__version__
        details["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            details["cuda_device"] = torch.cuda.get_device_name(0)
        else:
            details["cuda_device"] = "N/A (CPU only mode)"
    except ImportError:
        details["version"] = "NOT INSTALLED"
        details["cuda_available"] = "False"
        details["cuda_device"] = "N/A"
    return details


def run_verification() -> int:
    """Runs all checks and prints formatted report."""
    print("=" * 70)
    print(" AI-POWERED FAKE JOB DETECTION SYSTEM -- ENVIRONMENT VERIFICATION")
    print("=" * 70)

    # 1. Python Check
    py_ok, py_ver = check_python_version()
    py_status = "[OK]" if py_ok else "[FAIL]"
    print(f"\n1. Python Interpreter:")
    print(f"   {py_status} Python Version: {py_ver} (Executable: {sys.executable})")

    # 2. Required Core Dependencies
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("imbalanced-learn", "imblearn"),
        ("xgboost", "xgboost"),
        ("transformers", "transformers"),
        ("torch", "torch"),
    ]

    optional_packages = [
        ("lightgbm", "lightgbm"),
        ("shap", "shap"),
        ("nltk", "nltk"),
        ("spacy", "spacy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("pydantic", "pydantic"),
        ("pyyaml", "yaml"),
        ("joblib", "joblib"),
        ("fastapi", "fastapi"),
        ("streamlit", "streamlit"),
        ("pytest", "pytest"),
    ]

    print("\n2. Core Required Dependencies:")
    missing_required = 0
    for display_name, mod_name in required_packages:
        installed, ver = check_package(display_name, mod_name)
        status = "[OK]  " if installed else "[MISS]"
        if not installed:
            missing_required += 1
        print(f"   {status} {display_name:<20} -> {ver}")

    print("\n3. Extended / Planned Dependencies:")
    missing_optional = 0
    for display_name, mod_name in optional_packages:
        installed, ver = check_package(display_name, mod_name)
        status = "[OK]  " if installed else "[MISS]"
        if not installed:
            missing_optional += 1
        print(f"   {status} {display_name:<20} -> {ver}")

    # 4. PyTorch Diagnostics
    print("\n4. Deep Learning & Compute Acceleration:")
    torch_info = check_torch_details()
    print(f"   PyTorch Version : {torch_info['version']}")
    print(f"   CUDA Available  : {torch_info['cuda_available']}")
    print(f"   Active Device   : {torch_info['cuda_device']}")

    print("\n" + "=" * 70)
    if missing_required == 0:
        print(" SUCCESS: All core required packages are installed and available.")
        print("=" * 70)
        return 0
    else:
        print(f" NOTICE: {missing_required} core package(s) missing from active environment.")
        print(" To install all dependencies, activate your virtual environment and run:")
        print("     pip install -r requirements.txt")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = run_verification()
    sys.exit(exit_code)

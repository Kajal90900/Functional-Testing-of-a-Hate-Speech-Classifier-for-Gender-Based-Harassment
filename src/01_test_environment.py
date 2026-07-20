"""
01_test_environment.py

Purpose
-------
Verifies that the local Python environment (created and activated via
Git Bash) has all required libraries installed correctly before any
dataset loading or model inference is attempted.

Run from the repository root:
    python src/01_test_environment.py
"""

import sys


def check_import(module_name, display_name=None):
    display_name = display_name or module_name
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown version")
        print(f"[OK]   {display_name:<15} -> {version}")
        return True
    except ImportError:
        print(f"[FAIL] {display_name:<15} -> NOT INSTALLED")
        return False


def main():
    print("=" * 50)
    print("Environment Verification")
    print("=" * 50)
    print(f"Python version: {sys.version.split()[0]}\n")

    checks = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("sklearn", "scikit-learn"),
        ("pandas", "Pandas"),
    ]

    results = [check_import(mod, name) for mod, name in checks]

    print("\n" + "=" * 50)
    if all(results):
        print("All required libraries are installed correctly.")
        print("Environment is ready for dataset loading and model inference.")
    else:
        print("One or more libraries are missing. Run:")
        print("    pip install transformers datasets scikit-learn pandas torch")
    print("=" * 50)


if __name__ == "__main__":
    main()

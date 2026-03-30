#!/usr/bin/env python3
"""Validate that all dependencies and API access are set up correctly."""

import os
import sys


def check_imports():
    """Check that all required packages can be imported."""
    print("📦 Checking imports...")
    required = {
        "openai": "OpenAI API client",
        "datasets": "HuggingFace datasets",
        "tiktoken": "Token counting",
        "numpy": "Numerical arrays",
        "sklearn": "Machine learning (scikit-learn)",
    }

    all_ok = True
    for package, description in required.items():
        try:
            __import__(package)
            print(f"  ✓ {package:<15} ({description})")
        except ImportError:
            print(f"  ✗ {package:<15} MISSING")
            all_ok = False

    return all_ok


def check_api_key():
    """Check that OpenAI API key is set."""
    print("\n🔑 Checking API key...")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        # Show first 10 and last 4 chars for privacy
        masked = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
        print(f"  ✓ OPENAI_API_KEY found in environment: {masked}")

        if not api_key.startswith("sk-"):
            print("  ⚠️  WARNING: Key doesn't start with 'sk-'. Is it correct?")
            return False
        return True
    else:
        print("  ✗ OPENAI_API_KEY not found in environment variables")
        print("\n     To set it (Windows PowerShell):")
        print("     $env:OPENAI_API_KEY = 'sk-your-key-here'")
        print("\n     To set it (Windows CMD):")
        print("     set OPENAI_API_KEY=sk-your-key-here")
        print("\n     To set it (macOS/Linux):")
        print("     export OPENAI_API_KEY='sk-your-key-here'")
        print("\n     Get your key from: https://platform.openai.com/account/api-keys")
        return False


def check_dataset():
    """Check that HotpotQA can be loaded."""
    print("\n📊 Checking dataset access...")
    try:
        from datasets import load_dataset

        print("  ⏳ Loading HotpotQA (first time may take a minute)...")
        ds = load_dataset("hotpot_qa", "distractor", split="validation")
        print(f"  ✓ HotpotQA loaded ({len(ds)} examples)")
        return True
    except Exception as e:
        print(f"  ✗ Failed to load HotpotQA: {e}")
        return False


def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              CART Baseline Analysis — Setup Check            ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    checks = [
        ("Imports", check_imports),
        ("API Key", check_api_key),
        ("Dataset", check_dataset),
    ]

    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append(result)
        except Exception as e:
            print(f"\n✗ {name} check failed: {e}")
            results.append(False)

    print("\n" + "=" * 63)
    if all(results):
        print("✅ All checks passed! Ready to run baselines.\n")
        print("Next step:")
        print("  python run_baselines.py")
        return 0
    else:
        print("❌ Some checks failed. Fix issues above and try again.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

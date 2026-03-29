#!/usr/bin/env python3
"""Validate that all dependencies and API access are set up correctly."""

import os
import sys


def check_imports():
    """Check that all required packages can be imported."""
    print("📦 Checking imports...")
    required = {
        "openai": "OpenAI API client",
        "anthropic": "Anthropic API client",
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


def check_api_key(provider):
    """Check that API key is set for the specified provider.

    Args:
        provider: Either "OPENAI" or "ANTHROPIC"

    Returns:
        bool: True if API key is valid, False otherwise
    """
    provider = provider.upper()
    if provider not in ("OPENAI", "ANTHROPIC"):
        raise ValueError(f"Unknown provider: {provider}. Must be 'OPENAI' or 'ANTHROPIC'")

    if provider == "OPENAI":
        env_var = "OPENAI_API_KEY"
        prefix = "sk-"
        display_name = "OpenAI"
    else:  # ANTHROPIC
        env_var = "ANTHROPIC_API_KEY"
        prefix = "sk-ant-"
        display_name = "Anthropic"

    print(f"\n🔑 Checking {display_name} API key...")
    api_key = os.getenv(env_var)
    if api_key:
        # Show first 10 and last 4 chars for privacy
        masked = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
        print(f"  ✓ {env_var} found in environment: {masked}")

        if not api_key.startswith(prefix):
            print(f"  ⚠️  WARNING: Key doesn't start with '{prefix}'. Is it correct?")
            return False
        return True
    else:
        print(f"  ✗ {env_var} not found in environment variables")
        print("\n     To set it (Windows PowerShell):")
        print(f"     $env:{env_var} = 'your-{display_name.lower()}-key-here'")
        print("\n     To set it (Windows CMD):")
        print(f"     set {env_var}=your-{display_name.lower()}-key-here")
        print("\n     To set it (macOS/Linux):")
        print(f"     export {env_var}='your-{display_name.lower()}-key-here'")
        if provider == "OPENAI":
            print("\n     Get your key from: https://platform.openai.com/account/api-keys")
        else:
            print("\n     Get your key from: https://console.anthropic.com/account/keys")
        return False


def check_api_keys():
    """Check that both OPENAI and ANTHROPIC API keys are set."""
    results = {}
    for provider in ("OPENAI", "ANTHROPIC"):
        try:
            results[provider] = check_api_key(provider)
        except Exception as e:
            print(f"\n✗ {provider} API key check failed: {e}")
            results[provider] = False

    return all(results.values())


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
============================================================
       CART Baseline Analysis - Setup Check
============================================================
    """)

    checks = [
        ("Imports", check_imports),
        ("API Keys", check_api_keys),
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

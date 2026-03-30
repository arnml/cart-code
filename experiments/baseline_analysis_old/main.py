#!/usr/bin/env python3
"""
Quick launcher for baseline analysis.
Run this from the project root:
  cd experiments/baseline_analysis
  python main.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from .run_baselines import run_all_baselines

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           CART Paper — Day 2: Baseline Analysis              ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    # Quick check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.\n")
        print("Set it with (Windows PowerShell):")
        print("  $env:OPENAI_API_KEY = 'sk-your-key-here'\n")
        sys.exit(1)

    try:
        run_all_baselines(n_samples=50, output_dir="results")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    else:
        print("\n✅ All done! Check results/ folder for outputs.")

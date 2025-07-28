#!/usr/bin/env python3
"""
Test script to validate the new pre-commit configuration.
This script checks if the stricter rules will work with the current codebase.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n🔍 {description}...")
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=Path(__file__).parent
        )
        if result.returncode == 0:
            print(f"✅ {description} passed")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False


def main():
    """Test the new pre-commit configuration."""
    print("Testing new pre-commit configuration...")
    
    tests = [
        (["poetry", "run", "black", "--check", "."], "Black formatting check"),
        (["poetry", "run", "isort", "--check-only", "."], "Import sorting check"),
        (["poetry", "run", "flake8", "--max-line-length=100", "--extend-ignore=E203,W503", "--per-file-ignores=__init__.py:F401,F403", "--max-complexity=10"], "Flake8 linting (stricter rules)"),
        (["poetry", "run", "bandit", "-r", "msiconvert", "-f", "json"], "Bandit security scan"),
    ]
    
    results = []
    for cmd, description in tests:
        success = run_command(cmd, description)
        results.append((description, success))
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    
    passed = 0
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {description}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("\n🎉 All checks passed! The new pre-commit config should work well.")
    else:
        print(f"\n⚠️  Some checks failed. You may need to fix issues before the stricter config works smoothly.")
        print("Consider running the individual commands to see specific issues.")


if __name__ == "__main__":
    main()
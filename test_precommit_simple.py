#!/usr/bin/env python3
"""
Simple test script to validate the new pre-commit configuration on Windows.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_command(cmd: List[str], description: str) -> Tuple[bool, str, str]:
    """Run a command and return success status and output."""
    print(f"\n[INFO] {description}...")
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=Path(__file__).parent,
            encoding='utf-8',
            errors='replace'
        )
        success = result.returncode == 0
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {description}")
        return success, result.stdout, result.stderr
    except Exception as e:
        print(f"[ERROR] {description} error: {e}")
        return False, "", str(e)


def main():
    """Test the new pre-commit configuration."""
    print("Testing new pre-commit configuration (Windows Compatible)...")
    
    # Test individual tools
    tests = [
        (["poetry", "run", "black", "--check", "--diff", "msiconvert/__init__.py"], 
         "Black formatting check on __init__.py"),
        (["poetry", "run", "isort", "--check-only", "--diff", "msiconvert/__init__.py"], 
         "Import sorting check on __init__.py"),
        (["poetry", "run", "flake8", "msiconvert/__init__.py", "--max-line-length=100", 
          "--extend-ignore=E203,W503", "--max-complexity=10"], 
         "Flake8 linting on __init__.py"),
        (["poetry", "run", "mypy", "msiconvert/__init__.py", "--ignore-missing-imports"], 
         "MyPy type checking on __init__.py"),
        (["poetry", "run", "bandit", "-r", "msiconvert", "-f", "json", "--quiet"], 
         "Bandit security scan"),
        (["poetry", "run", "pydocstyle", "msiconvert/__init__.py", "--convention=google"], 
         "Pydocstyle docstring check on __init__.py"),
    ]
    
    results = []
    for cmd, description in tests:
        success, stdout, stderr = run_command(cmd, description)
        results.append((description, success, stdout, stderr))
        
        # Show details for failures
        if not success:
            print(f"  STDOUT: {stdout[:500]}...")
            print(f"  STDERR: {stderr[:500]}...")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    
    passed = 0
    for description, success, stdout, stderr in results:
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {description}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("\n[SUCCESS] All checks passed! The new pre-commit config should work well.")
    else:
        print(f"\n[WARNING] Some checks failed. You may need to fix issues before the stricter config works smoothly.")
        print("See the detailed output above for specific issues.")

    # Test pre-commit installation
    print("\n" + "="*60)
    print("PRE-COMMIT INSTALLATION TEST:")
    print("="*60)
    
    install_success, install_stdout, install_stderr = run_command(
        ["poetry", "run", "pre-commit", "install"], 
        "Installing pre-commit hooks"
    )
    
    if install_success:
        print("[SUCCESS] Pre-commit hooks installed successfully!")
    else:
        print(f"[FAIL] Pre-commit installation failed:")
        print(f"  STDOUT: {install_stdout}")
        print(f"  STDERR: {install_stderr}")

    return passed == len(results) and install_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
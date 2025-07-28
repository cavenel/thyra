# Pre-commit Configuration Migration Guide

This document outlines the changes made to improve code quality through stricter pre-commit hooks.

## Summary of Changes

### 1. Updated Hook Versions
- `pre-commit-hooks`: v4.4.0 → v4.6.0
- `black`: 23.7.0 → 24.4.2
- `isort`: 5.12.0 → 5.13.2
- `flake8`: 6.0.0 → 7.1.0

### 2. Enhanced Basic Checks
Added new file format and quality checks:
- `check-toml` - Validates TOML syntax
- `check-json` - Validates JSON syntax
- `check-docstring-first` - Ensures docstrings come first
- `check-case-conflict` - Prevents case-sensitive filename conflicts
- `mixed-line-ending` - Standardizes line endings to LF

### 3. Stricter Flake8 Configuration
**Removed ignores** (now enforced):
- `F401` - Unused imports (except in `__init__.py`)
- `F403` - Star imports (except in `__init__.py`)
- `E402` - Module imports not at top
- `E501` - Line too long (managed by black)
- `E231` - Missing whitespace after delimiter
- `E221` - Multiple spaces before operator
- `F821` - Undefined name
- `F541` - F-string missing placeholders
- `E722` - Bare except clauses
- `E713` - Test for membership should be 'not in'
- `F841` - Local variable assigned but never used

**Kept ignores**:
- `E203` - Whitespace before ':' (black compatibility)
- `W503` - Line break before binary operator (conflicts with W504)

**New restrictions**:
- `--max-complexity=10` - Limits cyclomatic complexity
- `--per-file-ignores=__init__.py:F401,F403` - Allows imports only in `__init__.py`

### 4. New Tools Added

#### MyPy (Type Checking)
- Enabled for all source code (excludes tests and docs)
- Configured with gradual typing approach
- Ignores missing imports for external packages

#### Bandit (Security Scanning)
- Scans for common security issues
- Excludes test files
- Configured via `pyproject.toml`

#### Pydocstyle (Docstring Quality)
- Enforces Google-style docstrings
- Excludes tests and docs
- Ignores missing docstrings for modules, packages, and magic methods

## Migration Steps

### 1. Install New Dependencies
```bash
poetry install  # Installs new dev dependencies
```

### 2. Update Pre-commit Hooks
```bash
poetry run pre-commit install
poetry run pre-commit migrate-config
poetry run pre-commit autoupdate
```

### 3. Test Current Codebase
Run the test script to see what needs fixing:
```bash
python test_new_precommit.py
```

### 4. Fix Issues Incrementally

#### Common Issues and Fixes:

**Unused Imports (F401)**
```python
# Before
import unused_module
from some_package import unused_function

# After - Remove unused imports
# (Keep only what's actually used)
```

**Undefined Names (F821)**
```python
# Before
result = undefined_variable + 1

# After - Define variables before use
defined_variable = 10
result = defined_variable + 1
```

**Bare Except (E722)**
```python
# Before
try:
    risky_operation()
except:
    handle_error()

# After - Specify exception types
try:
    risky_operation()
except (ValueError, TypeError) as e:
    handle_error(e)
```

**Complex Functions (Complexity > 10)**
Break down complex functions into smaller, focused functions.

### 5. Run Full Pre-commit Check
```bash
poetry run pre-commit run --all-files
```

## Benefits of Stricter Configuration

1. **Better Code Quality**: Catches more potential bugs and style issues
2. **Security**: Bandit identifies security vulnerabilities
3. **Type Safety**: MyPy helps catch type-related errors
4. **Documentation**: Pydocstyle ensures consistent docstring formatting
5. **Maintainability**: Lower complexity and cleaner imports improve code readability

## Gradual Adoption Strategy

If the stricter rules cause too many failures initially:

1. **Phase 1**: Fix critical issues (undefined names, bare excepts)
2. **Phase 2**: Clean up imports and reduce complexity
3. **Phase 3**: Add type hints gradually
4. **Phase 4**: Improve docstring coverage

You can temporarily disable specific checks by adding them back to the ignore list, then removing them as issues are fixed.

## Testing the Configuration

The included `test_new_precommit.py` script tests key components individually. Run it to identify areas that need attention before fully enabling the stricter configuration.
# Pre-commit Configuration Issues and Fixes

## Critical Issues That Must Be Fixed

### 1. Shell Environment Problem
**Issue**: Commands fail with `/usr/bin/bash: Files\Git\bin\bash.exe: No such file or directory`
**Fix**: 
- Ensure Git Bash or WSL is properly configured
- Alternative: Use Windows Command Prompt compatible commands
- Test pre-commit installation manually: `poetry run pre-commit install --install-hooks`

### 2. Type Annotation Compatibility
**Issue**: Modern type hints like `list[str]` used in Python <3.9 environments
**Files**: `test_precommit_simple.py` (and potentially others)
**Fix**:
```python
# Before
def run_command(cmd: list[str], description: str) -> tuple[bool, str, str]:

# After  
from typing import List, Tuple
def run_command(cmd: List[str], description: str) -> Tuple[bool, str, str]:
```

### 3. Flake8 Violations from Stricter Rules
**Issue**: Removed many ignored error codes
**Likely violations**:
- **E501**: Line too long (>100 chars)
- **F401**: Unused imports (now only allowed in __init__.py)
- **E402**: Module level import not at top
- **F821**: Undefined name
- **E722**: Bare except clause

**Fix Strategy**:
1. Run `poetry run black .` to auto-fix formatting
2. Run `poetry run isort .` to fix import order
3. Manually fix remaining flake8 issues or use gradual config first

## Warnings (Should Fix)

### 1. MyPy Type Issues
**Issue**: Many functions lack proper type annotations
**Examples from codebase**:
```python
# In convert.py line 29
pixel_size_um: float = None  # Should be Optional[float] = None

# Missing return types in many functions
def detect_format(input_path: Path):  # Should specify -> str
```

**Fix**:
```python
from typing import Optional

def convert_msi(
    input_path: str,
    output_path: str,
    format_type: str = "spatialdata",
    dataset_id: str = "msi_dataset",
    pixel_size_um: Optional[float] = None,  # Fixed
    handle_3d: bool = False,
    **kwargs,
) -> bool:  # Added return type
```

### 2. Pydocstyle Violations
**Issue**: Not all functions have Google-style docstrings
**Fix**: Add docstrings to public functions following Google convention

### 3. Bandit Security Issues
**Potential issues**:
- `subprocess.run` usage without proper validation
- Path operations without validation

## Suggested Migration Path

### Phase 1: Use Gradual Configuration
1. Use `.pre-commit-config-gradual.yaml` initially
2. Fix basic formatting and import issues
3. Address security concerns

### Phase 2: Incremental Strictness
1. Gradually reduce ignored error codes in flake8
2. Add type annotations incrementally
3. Add missing docstrings

### Phase 3: Full Strict Configuration
1. Switch to the full strict configuration
2. Enable mypy and pydocstyle
3. Ensure all tools pass

## Recommended pyproject.toml Updates

Add more lenient initial MyPy settings:
```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = false  # Start with false
warn_unused_configs = true
disallow_untyped_defs = false  # Keep false initially
disallow_incomplete_defs = false  # Keep false initially
check_untyped_defs = false  # Start with false
ignore_missing_imports = true
```

## Testing Strategy

Since direct testing is blocked by shell issues, recommend:
1. Fix shell environment or use WSL
2. Test individual tools manually
3. Use CI/CD pipeline to validate changes
4. Start with `--dry-run` mode for pre-commit

## Files Requiring Immediate Attention

Based on code review:
1. `msiconvert/convert.py` - Type annotations and line length
2. `msiconvert/__main__.py` - Potential complexity issues
3. `msiconvert/core/registry.py` - Clean, likely passes
4. `test_precommit_simple.py` - Type hint compatibility
5. All `__init__.py` files - Import organization

## Command Sequence for Manual Testing

Once shell environment is fixed:
```bash
# Install hooks
poetry run pre-commit install

# Test individual tools
poetry run black --check --diff .
poetry run isort --check-only --diff .
poetry run flake8 --statistics
poetry run mypy msiconvert/ --ignore-missing-imports
poetry run bandit -r msiconvert/ -f json
poetry run pydocstyle msiconvert/ --convention=google

# Run pre-commit on staged files
poetry run pre-commit run

# Run on all files
poetry run pre-commit run --all-files
```
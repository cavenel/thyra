# 🔍 Complexity Monitoring System

Automated cyclomatic complexity monitoring for MSIConverter to maintain code quality and prevent complexity debt.

## Features

- **Automated Scanning**: Detects complexity violations using flake8
- **Trend Analysis**: Tracks complexity changes over time
- **CI/CD Integration**: GitHub Actions workflow with PR comments
- **Flexible Thresholds**: Configurable complexity limits
- **Rich Reporting**: JSON reports with distribution analysis
- **Pre-commit Integration**: Optional pre-commit hook support

## Quick Start

### 1. Basic Usage

```bash
# Run complexity check with default settings
./scripts/check_complexity.sh

# Quick check with custom threshold
./scripts/check_complexity.sh -t 15 -q

# Show trend analysis
./scripts/check_complexity.sh -r
```

### 2. Python API

```python
from scripts.complexity_monitor import ComplexityMonitor
from pathlib import Path

# Create monitor
monitor = ComplexityMonitor(Path('.'), threshold=10)

# Run analysis
violations = monitor.parse_violations(monitor.run_flake8_complexity())
report = monitor.generate_report(violations)

# Display results
monitor.display_report(report)
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-t, --threshold N` | Complexity threshold | 10 |
| `-q, --quiet` | Show only summary | false |
| `-n, --no-save` | Don't save reports | false |
| `-r, --trends` | Show trend analysis | false |
| `-h, --help` | Show help message | - |

## Integration

### Pre-commit Hook (Optional)

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: complexity-check
      name: Complexity Monitoring
      entry: ./scripts/check_complexity.sh
      args: [-q, -t, "12"]
      language: system
      pass_filenames: false
      stages: [pre-commit]
```

### GitHub Actions

The included workflow (`.github/workflows/complexity-monitoring.yml`) automatically:

- ✅ Runs on push/PR to main/develop branches
- ✅ Generates complexity reports
- ✅ Posts PR comments with results
- ✅ Uploads reports as artifacts
- ✅ Runs weekly complexity analysis

### CI/CD Pipeline

```bash
# In your CI script
./scripts/check_complexity.sh -t 10 --trends

# Exit code: 0 = no violations, 1 = violations found
```

## Configuration

Edit `scripts/complexity_config.json` to customize:

- **Thresholds**: Different limits by code category
- **Reporting**: Output formats and detail levels  
- **Integration**: Pre-commit and CI/CD settings
- **Targets**: Include/exclude patterns

### Example Configuration

```json
{
  "thresholds_by_category": {
    "production_code": {
      "msiconvert/core/**": 8,
      "msiconvert/readers/**": 12
    },
    "test_code": {
      "tests/unit/**": 15,
      "tests/integration/**": 20
    }
  }
}
```

## Report Structure

### JSON Report Format

```json
{
  "timestamp": "2025-01-08T10:30:00",
  "total_violations": 7,
  "high_complexity_functions": [
    {
      "file": "msiconvert/convert.py",
      "line": 22,
      "function": "convert_msi",
      "complexity": 20,
      "threshold": 10
    }
  ],
  "complexity_distribution": {
    "Low (11-15)": 3,
    "Medium (16-25)": 3,
    "High (26-40)": 1
  },
  "average_complexity": 18.4,
  "max_complexity": 40
}
```

### Report Storage

Reports are saved in `reports/complexity/` with timestamps:
- Format: `complexity_report_YYYYMMDD_HHMMSS.json`
- Retention: Last 50 reports (configurable)
- Automatic cleanup of old reports

## Understanding Complexity

### Complexity Categories

| Range | Category | Action Required |
|-------|----------|-----------------|
| 1-10  | ✅ Good | No action needed |
| 11-15 | ⚠️ Warning | Consider refactoring |
| 16-25 | 🔶 Medium | Should refactor |
| 26-40 | 🔥 High | Must refactor |
| 40+   | 💀 Critical | Immediate attention |

### Common Causes

- **Deep nesting**: Multiple if/for/while statements
- **Long functions**: Too many responsibilities
- **Complex conditionals**: Multiple boolean operators
- **Exception handling**: Try/catch blocks add complexity

### Refactoring Strategies

1. **Extract Methods**: Break large functions into smaller ones
2. **Reduce Nesting**: Use early returns and guard clauses
3. **Simplify Conditionals**: Use boolean variables or helper methods
4. **Single Responsibility**: One function, one purpose

## Best Practices

### Development Workflow

1. **Before Committing**: Run `./scripts/check_complexity.sh -q`
2. **During PR**: Review complexity comments on GitHub
3. **Weekly Review**: Check trend reports for regression
4. **Refactoring**: Target functions with complexity > 15

### Team Standards

- **New Code**: Must stay under threshold (10)
- **Legacy Code**: Gradual improvement, don't make worse
- **Test Code**: More lenient thresholds (15-20)
- **Critical Paths**: Stricter limits for core functionality

## Troubleshooting

### Common Issues

**Poetry not found**
```bash
# Install poetry first
curl -sSL https://install.python-poetry.org | python3 -
```

**Missing dependencies**
```bash
# Install project dependencies
poetry install
poetry add --group dev click
```

**Permission denied on script**
```bash
# Make script executable
chmod +x scripts/check_complexity.sh
```

**Reports directory missing**
```bash
# Will be created automatically on first run
./scripts/check_complexity.sh
```

## Advanced Usage

### Custom Analysis

```python
# Analyze specific files
monitor = ComplexityMonitor(Path('.'), threshold=8)
violations = monitor.parse_violations([
    "src/complex_module.py:45:1: C901 'complex_function' is too complex (25)"
])
```

### Batch Processing

```bash
# Check multiple thresholds
for threshold in 8 10 12 15; do
    echo "=== Threshold: $threshold ==="
    ./scripts/check_complexity.sh -t $threshold -q
done
```

### Integration with Other Tools

```bash
# Combine with other quality checks
./scripts/check_complexity.sh -q && \
poetry run pytest && \
poetry run mypy msiconvert/
```

## Contributing

When adding features to the complexity monitoring system:

1. Update `complexity_monitor.py` for core functionality
2. Update `check_complexity.sh` for CLI interface
3. Update `complexity_config.json` for new options
4. Update this README with new features
5. Test with various complexity scenarios

---

**📊 Track complexity. Improve quality. Ship better code.**

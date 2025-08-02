#!/usr/bin/env python3
"""Automated complexity monitoring for MSIConverter.

This script monitors cyclomatic complexity across the codebase and tracks
improvements over time. It can be integrated into CI/CD pipelines and
pre-commit hooks for continuous monitoring.
"""

import json
import subprocess  # nosec B404 - Safe subprocess usage
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple

import click


class ComplexityViolation(NamedTuple):
    """Represents a complexity violation."""

    file: str
    line: int
    function: str
    complexity: int
    threshold: int


class ComplexityReport(NamedTuple):
    """Represents a complexity monitoring report."""

    timestamp: str
    total_violations: int
    high_complexity_functions: List[ComplexityViolation]
    complexity_distribution: Dict[str, int]
    average_complexity: float
    max_complexity: int


class ComplexityMonitor:
    """Monitors cyclomatic complexity across the codebase."""

    def __init__(self, project_root: Path, threshold: int = 10):
        """Initialize complexity monitor with project path and threshold."""
        self.project_root = project_root
        self.threshold = threshold
        self.reports_dir = project_root / "reports" / "complexity"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Log git state for debugging environment differences
        self._log_git_state()

    def _log_git_state(self) -> None:
        """Log git state for debugging environment differences."""
        try:
            result = subprocess.run(  # nosec B603,B607
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            if result.returncode == 0:
                num_files = (
                    len(result.stdout.strip().split()) if result.stdout.strip() else 0
                )
                click.echo(f"Git status: {num_files} modified files", err=True)

            result = subprocess.run(  # nosec B603,B607
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            if result.returncode == 0:
                click.echo(f"Git commit: {result.stdout.strip()[:8]}", err=True)
        except Exception as e:
            click.echo(f"Could not get git state: {e}", err=True)

    def run_flake8_complexity(self) -> List[str]:
        """Run flake8 with complexity checking and return raw output lines."""
        try:
            # Add debugging info about what we're scanning
            msiconvert_path = self.project_root / "msiconvert"
            tests_path = self.project_root / "tests"

            if not msiconvert_path.exists():
                click.echo(f"Warning: {msiconvert_path} does not exist", err=True)
            if not tests_path.exists():
                click.echo(f"Warning: {tests_path} does not exist", err=True)

            result = subprocess.run(  # nosec B603,B607
                [
                    sys.executable,
                    "-m",
                    "flake8",
                    "--select=C901",  # Only complexity violations
                    f"--max-complexity={self.threshold}",
                    "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s",
                    str(msiconvert_path),
                    str(tests_path),
                ],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                return []  # No violations

            # Add debugging info about flake8 results
            if result.stderr:
                click.echo(f"Flake8 stderr: {result.stderr}", err=True)

            violations = (
                result.stdout.strip().split("\n") if result.stdout.strip() else []
            )
            if violations:
                click.echo(f"Found {len(violations)} complexity violations:", err=True)
                for v in violations[:5]:  # Show first 5
                    click.echo(f"  {v}", err=True)

            return violations

        except subprocess.CalledProcessError as e:
            click.echo(f"Error running flake8: {e}", err=True)
            return []

    def parse_violations(self, flake8_output: List[str]) -> List[ComplexityViolation]:
        """Parse flake8 output into structured violations."""
        violations = []

        for line in flake8_output:
            if not line.strip() or "C901" not in line:
                continue

            try:
                # Format: path:line:col: C901 'function' is too complex (complexity)
                parts = line.split(": C901 ")
                if len(parts) != 2:
                    continue

                file_info = parts[0]
                violation_info = parts[1]

                # Extract file and line number
                file_parts = file_info.split(":")
                if len(file_parts) < 2:
                    continue

                file_path = (
                    ":".join(file_parts[:-2]) if len(file_parts) > 2 else file_parts[0]
                )
                line_num = int(file_parts[-2])

                # Extract function name and complexity
                # Format: 'function_name' is too complex (X)
                if "is too complex" in violation_info:
                    func_part = violation_info.split("' is too complex")[0]
                    function_name = func_part.strip("'").split("'")[-1]

                    complexity_part = violation_info.split("(")[-1].split(")")[0]
                    complexity = int(complexity_part)

                    violations.append(
                        ComplexityViolation(
                            file=str(Path(file_path).relative_to(self.project_root)),
                            line=line_num,
                            function=function_name,
                            complexity=complexity,
                            threshold=self.threshold,
                        )
                    )

            except (ValueError, IndexError) as e:
                click.echo(f"Warning: Could not parse line: {line} ({e})", err=True)
                continue

        return violations

    def generate_report(
        self, violations: List[ComplexityViolation]
    ) -> ComplexityReport:
        """Generate a comprehensive complexity report."""
        if not violations:
            return ComplexityReport(
                timestamp=datetime.now().isoformat(),
                total_violations=0,
                high_complexity_functions=[],
                complexity_distribution={},
                average_complexity=0.0,
                max_complexity=0,
            )

        # Calculate distribution
        distribution: Dict[str, int] = {}
        complexities = [v.complexity for v in violations]

        for complexity in complexities:
            if complexity <= 15:
                key = "Low (11-15)"
            elif complexity <= 25:
                key = "Medium (16-25)"
            elif complexity <= 40:
                key = "High (26-40)"
            else:
                key = "Very High (40+)"

            distribution[key] = distribution.get(key, 0) + 1

        return ComplexityReport(
            timestamp=datetime.now().isoformat(),
            total_violations=len(violations),
            high_complexity_functions=sorted(
                violations, key=lambda x: x.complexity, reverse=True
            ),
            complexity_distribution=distribution,
            average_complexity=sum(complexities) / len(complexities),
            max_complexity=max(complexities),
        )

    def save_report(self, report: ComplexityReport) -> Path:
        """Save report to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"complexity_report_{timestamp}.json"

        # Convert NamedTuple to dict for JSON serialization
        report_dict = {
            "timestamp": report.timestamp,
            "total_violations": report.total_violations,
            "high_complexity_functions": [
                {
                    "file": v.file,
                    "line": v.line,
                    "function": v.function,
                    "complexity": v.complexity,
                    "threshold": v.threshold,
                }
                for v in report.high_complexity_functions
            ],
            "complexity_distribution": report.complexity_distribution,
            "average_complexity": round(report.average_complexity, 2),
            "max_complexity": report.max_complexity,
        }

        with open(report_file, "w") as f:
            json.dump(report_dict, f, indent=2)

        return report_file

    def display_report(self, report: ComplexityReport) -> None:
        """Display report in a readable format."""
        click.echo("\n" + "=" * 60)
        click.echo("COMPLEXITY MONITORING REPORT")
        click.echo("=" * 60)
        click.echo(f"Generated: {report.timestamp}")
        click.echo(f"Threshold: {self.threshold}")
        click.echo(f"Total Violations: {report.total_violations}")

        if report.total_violations > 0:
            click.echo(f"Average Complexity: {report.average_complexity:.1f}")
            click.echo(f"Maximum Complexity: {report.max_complexity}")

            click.echo("\nDistribution:")
            for category, count in report.complexity_distribution.items():
                click.echo(f"   {category}: {count}")

            click.echo("\nTop Complex Functions:")
            for i, violation in enumerate(report.high_complexity_functions[:10], 1):
                click.echo(f"   {i:2d}. {violation.file}:{violation.line}")
                click.echo(
                    f"       {violation.function} (complexity: {violation.complexity})"
                )
        else:
            click.echo("No complexity violations found!")

        click.echo("=" * 60)

    def get_trend_data(self, limit: int = 10) -> List[Dict]:
        """Get historical trend data from recent reports."""
        report_files = sorted(self.reports_dir.glob("complexity_report_*.json"))[
            -limit:
        ]
        trends = []

        for report_file in report_files:
            try:
                with open(report_file) as f:
                    data = json.load(f)
                    trends.append(
                        {
                            "timestamp": data["timestamp"],
                            "total_violations": data["total_violations"],
                            "average_complexity": data["average_complexity"],
                            "max_complexity": data["max_complexity"],
                        }
                    )
            except (json.JSONDecodeError, KeyError):
                continue

        return trends


@click.command()
@click.option(
    "--threshold", "-t", default=10, help="Complexity threshold (default: 10)"
)
@click.option("--save/--no-save", default=True, help="Save report to file")
@click.option("--quiet", "-q", is_flag=True, help="Only show summary")
@click.option("--trends", is_flag=True, help="Show trend analysis")
@click.option(
    "--project-root",
    type=click.Path(exists=True),
    default=".",
    help="Project root directory",
)
def main(threshold: int, save: bool, quiet: bool, trends: bool, project_root: str):
    """Monitor cyclomatic complexity across the codebase."""
    monitor = ComplexityMonitor(Path(project_root), threshold)

    # Run complexity analysis
    click.echo("Scanning codebase for complexity violations...")
    flake8_output = monitor.run_flake8_complexity()
    violations = monitor.parse_violations(flake8_output)
    report = monitor.generate_report(violations)

    # Display results
    if not quiet:
        monitor.display_report(report)
    else:
        if report.total_violations == 0:
            click.echo("No complexity violations found!")
        else:
            click.echo(
                f"Found {report.total_violations} complexity violations "
                f"(avg: {report.average_complexity:.1f}, max: {report.max_complexity})"
            )

    # Save report
    if save:
        report_file = monitor.save_report(report)
        click.echo(f"Report saved to: {report_file}")

    # Show trends
    if trends:
        trend_data = monitor.get_trend_data()
        if len(trend_data) > 1:
            click.echo("\nTREND ANALYSIS (Last 10 Reports)")
            click.echo("-" * 50)
            for data in trend_data[-5:]:  # Show last 5
                timestamp = datetime.fromisoformat(data["timestamp"]).strftime(
                    "%Y-%m-%d %H:%M"
                )
                click.echo(
                    f"{timestamp}: {data['total_violations']} violations "
                    f"(avg: {data['average_complexity']:.1f})"
                )
        else:
            click.echo("\nNot enough historical data for trend analysis")

    # Exit with appropriate code
    if report.total_violations > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

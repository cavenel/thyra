#!/bin/bash
# Complexity monitoring wrapper script for MSIConverter
# This script provides easy access to complexity monitoring functionality

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
THRESHOLD=10
SAVE_REPORT=true
QUIET=false
SHOW_TRENDS=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help function
show_help() {
    cat << EOF
🔍 Complexity Monitoring Script for MSIConverter

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -t, --threshold N     Set complexity threshold (default: 10)
    -q, --quiet          Show only summary output
    -n, --no-save        Don't save report to file
    -r, --trends         Show trend analysis
    -h, --help           Show this help message

EXAMPLES:
    $0                   # Run with default settings
    $0 -t 15 -q         # Check with threshold 15, quiet output
    $0 -r               # Show trend analysis
    $0 -n -q            # Quick check without saving

INTEGRATION:
    # Pre-commit hook
    $0 -q

    # CI/CD pipeline
    $0 -t 12 --trends

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -n|--no-save)
            SAVE_REPORT=false
            shift
            ;;
        -r|--trends)
            SHOW_TRENDS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            show_help
            exit 1
            ;;
    esac
done

# Check if we're in a poetry environment
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}❌ Poetry not found. Please install poetry first.${NC}" >&2
    exit 1
fi

# Check if project dependencies are installed
if [ ! -d "$PROJECT_ROOT/.venv" ] && [ ! -f "$PROJECT_ROOT/poetry.lock" ]; then
    echo -e "${YELLOW}⚠️  Installing project dependencies...${NC}"
    cd "$PROJECT_ROOT"
    poetry install
fi

# Build command
CMD="poetry run python $SCRIPT_DIR/complexity_monitor.py"
CMD="$CMD --threshold $THRESHOLD"
CMD="$CMD --project-root $PROJECT_ROOT"

if [ "$SAVE_REPORT" = false ]; then
    CMD="$CMD --no-save"
fi

if [ "$QUIET" = true ]; then
    CMD="$CMD --quiet"
fi

if [ "$SHOW_TRENDS" = true ]; then
    CMD="$CMD --trends"
fi

# Ensure click is available
if ! poetry run python -c "import click" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Installing click dependency...${NC}"
    cd "$PROJECT_ROOT"
    poetry add --group dev click
fi

# Run the complexity monitor
echo -e "${BLUE}🔍 Running complexity monitoring...${NC}"
cd "$PROJECT_ROOT"

# Capture exit code
set +e
eval $CMD
EXIT_CODE=$?
set -e

# Provide feedback based on result
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Complexity check passed!${NC}"
else
    echo -e "${YELLOW}⚠️  Complexity violations found. Consider refactoring.${NC}"
fi

# Exit with the same code as the monitor
exit $EXIT_CODE

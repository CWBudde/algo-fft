#!/usr/bin/env bash
# Memory profiling script for FFT stress tests
#
# This script runs stress tests with memory profiling enabled and generates
# analysis reports to detect memory leaks and allocation patterns.
#
# Usage:
#   ./scripts/profile_memory.sh [output_dir]
#
# Environment variables:
#   STRESS_DURATION - Duration in seconds (default: 300 = 5 minutes)
#   PROFILE_DIR - Output directory for profiles (default: ./profiles)

set -euo pipefail

# Configuration
STRESS_DURATION="${STRESS_DURATION:-300}"
PROFILE_DIR="${1:-./profiles}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$PROFILE_DIR"

echo "=== FFT Memory Profiling ==="
echo "Duration: ${STRESS_DURATION}s"
echo "Output directory: $PROFILE_DIR"
echo

# Run stress tests with memory profiling
echo "Running stress tests with memory profiling..."
STRESS_DURATION=$STRESS_DURATION go test -v -timeout=30m \
    -run=Stress \
    -memprofile="$PROFILE_DIR/mem_${TIMESTAMP}.prof" \
    -memprofilerate=1 \
    ./...

echo
echo "Memory profile saved to: $PROFILE_DIR/mem_${TIMESTAMP}.prof"

# Generate text report
echo
echo "Generating memory allocation report..."
go tool pprof -text "$PROFILE_DIR/mem_${TIMESTAMP}.prof" > "$PROFILE_DIR/mem_${TIMESTAMP}_report.txt"
echo "Text report saved to: $PROFILE_DIR/mem_${TIMESTAMP}_report.txt"

# Show top allocations
echo
echo "=== Top 10 Allocation Sites ==="
go tool pprof -top10 "$PROFILE_DIR/mem_${TIMESTAMP}.prof"

# Generate flamegraph if available
if command -v go-torch &> /dev/null; then
    echo
    echo "Generating flamegraph..."
    go-torch --file="$PROFILE_DIR/mem_${TIMESTAMP}_flame.svg" "$PROFILE_DIR/mem_${TIMESTAMP}.prof"
    echo "Flamegraph saved to: $PROFILE_DIR/mem_${TIMESTAMP}_flame.svg"
fi

# Interactive analysis prompt
echo
echo "=== Analysis Complete ==="
echo "To interactively explore the profile, run:"
echo "  go tool pprof -http=:8080 $PROFILE_DIR/mem_${TIMESTAMP}.prof"
echo
echo "To compare with a previous profile, run:"
echo "  go tool pprof -http=:8080 -base=<old_profile> $PROFILE_DIR/mem_${TIMESTAMP}.prof"

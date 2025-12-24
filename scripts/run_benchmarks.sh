#!/usr/bin/env bash
set -euo pipefail

output_path="${1:-benchmarks/latest.txt}"
output_dir="$(dirname "$output_path")"

mkdir -p "$output_dir"

go test -bench=. -benchmem -run=^$ ./... | tee "$output_path"

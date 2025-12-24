#!/usr/bin/env bash
set -euo pipefail

baseline_path="${1:-benchmarks/baseline.txt}"
current_path="${2:-benchmarks/latest.txt}"
threshold="${BENCH_THRESHOLD_PCT:-10}"

if [[ ! -f "$baseline_path" ]]; then
  echo "bench_compare: baseline file not found: $baseline_path" >&2
  exit 0
fi

if [[ ! -f "$current_path" ]]; then
  echo "bench_compare: current benchmark file not found: $current_path" >&2
  exit 1
fi

if ! command -v benchstat >/dev/null 2>&1; then
  echo "bench_compare: benchstat not found in PATH" >&2
  exit 1
fi

benchstat -delta-test=none "$baseline_path" "$current_path" | tee benchmarks/benchstat.txt

awk -v threshold="$threshold" '
/^Benchmark/ && $0 ~ /ns\/op/ {
  for (i=1; i<=NF; i++) {
    if ($i ~ /%$/) {
      gsub("%", "", $i)
      gsub("+", "", $i)
      if ($i + 0 > threshold) {
        printf("bench_compare: regression > %s%% in line: %s\n", threshold, $0) > "/dev/stderr"
        exit 1
      }
    }
  }
}
' benchmarks/benchstat.txt

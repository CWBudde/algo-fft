#!/usr/bin/env bash
set -euo pipefail

input_path="${1:-}"

if [[ -n "$input_path" ]]; then
  if [[ ! -f "$input_path" ]]; then
    echo "bench_md: input file not found: $input_path" >&2
    exit 1
  fi
  bench_output="$(cat "$input_path")"
else
  bench_output="$(go test -bench=. -benchmem -run=^$ ./...)"
fi

echo "| Benchmark | ns/op | MB/s | B/op | allocs/op |"
echo "| --- | ---: | ---: | ---: | ---: |"

awk '
/^Benchmark/ {
  name=$1
  ns="-"
  mbs="-"
  bop="-"
  alloc="-"
  for (i=2; i<=NF; i++) {
    if ($(i) ~ /ns\/op$/) ns=$(i-1)
    if ($(i) ~ /MB\/s$/) mbs=$(i-1)
    if ($(i) ~ /B\/op$/) bop=$(i-1)
    if ($(i) ~ /bytes\/op$/) bop=$(i-1)
    if ($(i) ~ /allocs\/op$/) alloc=$(i-1)
  }
  print "| " name " | " ns " | " mbs " | " bop " | " alloc " |"
}
' <<< "$bench_output"

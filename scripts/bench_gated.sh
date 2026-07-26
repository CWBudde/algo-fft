#!/usr/bin/env bash
# Canary-gated codelet-candidate sweep.
#
# Why this exists: on a thermally-limited or shared host, a plain
# `go test -bench` sweep silently mixes measurements taken under wildly
# different conditions. Cooling does not fix contention, and a once-per-pass
# canary does not catch contention that arrives mid-pass — a 94-cell pass takes
# 5-13 minutes, and in one tuning round 3 of 5 nominally-clean passes were
# contaminated, one of them by 50x.
#
# So instead of measuring continuously, this script measures only inside
# verified-quiet windows:
#
#   - Every group of cells is bracketed by a canary cell whose quiet-machine
#     time is known (GOOD). The group starts only once the canary is within
#     GATE of GOOD, and a second canary runs immediately afterwards. Both are
#     written into the output, so bench_gated_analyze.sh can reject a group
#     whose window went bad *during* it.
#   - A group is one (precision, size): every registered candidate for that
#     size, forward and inverse, back-to-back. The whole candidate ranking —
#     which is the question this sweep answers — is therefore taken under a
#     single thermal state, and drift cancels inside the comparison.
#   - Group order and within-group cell order rotate per pass, so no candidate
#     keeps the coolest slot.
#
# Usage:
#   scripts/bench_gated.sh [size ...]        # default: 256 512 8192
#
# Environment:
#   PASSES   number of sweep passes                        (default 16)
#   BT       -test.benchtime per cell                      (default 0.5s)
#   GOOD     canary ns/op on a quiet machine               (default 1810)
#   GATE     accept a window within this factor of GOOD    (default 1.25)
#   MAXWAIT  canary retries per group, 15s apart           (default 40)
#   OUTDIR   results directory                             (default benchmarks/gated)
#   THERMAL  thermal zone file to log                      (default /sys/class/thermal/thermal_zone10/temp)
#
# Analyse the results with scripts/bench_gated_analyze.sh.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

sizes=("$@")
if [[ ${#sizes[@]} -eq 0 ]]; then
  sizes=(256 512 8192)
fi

PASSES="${PASSES:-16}"
BT="${BT:-0.5s}"
GOOD="${GOOD:-1810}"
GATE="${GATE:-1.25}"
MAXWAIT="${MAXWAIT:-40}"
OUTDIR="${OUTDIR:-benchmarks/gated}"
THERMAL="${THERMAL:-/sys/class/thermal/thermal_zone10/temp}"

# The canary must be a cell that is cheap, stable, and NOT part of the sweep,
# so that changing the sweep's input or candidate set leaves GOOD valid.
CANARY='BenchmarkDITComplex128/Size256/Radix16/Forward'

mkdir -p "$OUTDIR"
log="$OUTDIR/sweep.log"

bin="$OUTDIR/kernels.test"
echo "building $bin" | tee -a "$log"
go test -c -o "$bin" ./internal/kernels

run_cell() { # run_cell <cell-name> <output-file>
  "$bin" -test.bench="^$1\$" -test.benchtime="$BT" -test.count=1 -test.run='^$' >>"$2" 2>&1
}

canary_ns() {
  "$bin" -test.bench="^$CANARY\$" -test.benchtime="$BT" -test.count=1 -test.run='^$' 2>&1 |
    awk '/^Benchmark/ { print $3; exit }'
}

temp_c() {
  [[ -r "$THERMAL" ]] && echo $(($(cat "$THERMAL") / 1000)) || echo "?"
}

# Echoes the accepted canary value on stdout, or returns 1 after giving up.
wait_quiet() {
  local i=0 c
  local limit
  limit=$(awk -v g="$GOOD" -v f="$GATE" 'BEGIN { printf "%d", g * f }')
  while ((i < MAXWAIT)); do
    c="$(canary_ns)"
    if [[ -n $c ]] && awk -v c="$c" -v l="$limit" 'BEGIN { exit !(c <= l) }'; then
      echo "$c"
      return 0
    fi
    sleep 15
    ((i++))
  done
  return 1
}

# Enumerate the cells of the sweep once, from the binary itself, so the list
# always matches what is actually registered on this CPU.
declare -A cells_of
for prec in 64 128; do
  all="$("$bin" -test.bench="^BenchmarkCodeletCandidates$prec\$" -test.benchtime=1x \
    -test.count=1 -test.run='^$' 2>&1 |
    awk '/^BenchmarkCodeletCandidates/ { sub(/-[0-9]+$/, "", $1); print $1 }')"
  for size in "${sizes[@]}"; do
    matched="$(printf '%s\n' "$all" | grep "/size$size/" || true)"
    if [[ -z $matched ]]; then
      echo "warning: no candidates for complex$prec size $size" | tee -a "$log" >&2
      continue
    fi
    cells_of["$prec|$size"]="$matched"
  done
done

groups=()
for size in "${sizes[@]}"; do
  for prec in 64 128; do
    [[ -n ${cells_of["$prec|$size"]:-} ]] && groups+=("$prec|$size")
  done
done

# Record the incumbent per group. BenchmarkCodeletCandidates emits candidates
# in registry preference order and already skips disabled and unsupported ones,
# so its first cell for a group is by construction what Lookup() would return —
# the incumbent is read off the runtime registry, never off the priority table.
inc="$OUTDIR/incumbents.txt"
: >"$inc"
for g in "${groups[@]}"; do
  first="$(printf '%s\n' "${cells_of[$g]}" | head -1)"
  echo "$g $(printf '%s\n' "$first" | cut -d/ -f3)" >>"$inc"
done
echo "incumbents:" | tee -a "$log"
sed 's/^/  /' "$inc" | tee -a "$log"

total=0
for g in "${groups[@]}"; do
  total=$((total + $(printf '%s\n' "${cells_of[$g]}" | wc -l)))
done
echo "groups: ${#groups[@]}  cells: $total  passes: $PASSES  benchtime: $BT" | tee -a "$log"

for ((pass = 1; pass <= PASSES; pass++)); do
  echo "=== pass $pass/$PASSES $(date +%H:%M:%S) ===" >>"$log"
  out="$OUTDIR/pass$pass.txt"
  : >"$out"

  # Rotate the group order so no group keeps the first (coldest) slot.
  ng=${#groups[@]}
  for ((gi = 0; gi < ng; gi++)); do
    g="${groups[$(((gi + pass - 1) % ng))]}"
    prec="${g%%|*}"
    size="${g##*|}"

    if ! canary="$(wait_quiet)"; then
      echo "  c$prec/n=$size: no quiet window after $((MAXWAIT * 15))s, skipped" >>"$log"
      continue
    fi
    echo "  c$prec/n=$size: accepted, canary=$canary ns $(date +%H:%M:%S) $(temp_c)C" >>"$log"
    echo "#canary pass=$pass group=$g value=$canary" >>"$out"

    mapfile -t group_cells < <(printf '%s\n' "${cells_of[$g]}")
    nc=${#group_cells[@]}
    for ((ci = 0; ci < nc; ci++)); do
      run_cell "${group_cells[$(((ci + pass - 1) % nc))]}" "$out"
    done

    # Trailing canary: records a window that went bad mid-group.
    echo "#canary_post pass=$pass group=$g value=$(canary_ns)" >>"$out"
  done
  echo "  done $(date +%H:%M:%S) $(temp_c)C" >>"$log"
done

echo "=== sweep done $(date +%H:%M:%S) ===" | tee -a "$log"
echo "analyse with: scripts/bench_gated_analyze.sh $OUTDIR"

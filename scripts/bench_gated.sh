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
#   CALIBRATE=1 scripts/bench_gated.sh       # (re)derive GOOD for this host
#
# Environment:
#   PASSES    number of sweep passes                        (default 16)
#   BT        -test.benchtime per cell                      (default 0.5s)
#   GOOD      canary ns/op on a quiet machine   (default: this host's row in CALFILE)
#   GATE      accept a window within this factor of GOOD    (default 1.25)
#   MAXWAIT   canary retries per group, 15s apart           (default 40)
#   OUTDIR    results directory                             (default benchmarks/gated)
#   THERMAL   thermal zone file to log                      (default /sys/class/thermal/thermal_zone10/temp)
#   CALIBRATE derive GOOD from the observed floor and record it, then exit
#   CALREPS   canary samples taken during calibration       (default 30)
#   CALFILE   calibration record       (default benchmarks/canary-calibration.tsv)
#
# On GOOD, and why it is no longer a literal in this file
# -------------------------------------------------------
# GOOD is the canary's quiet-machine ns/op, and the gate accepts a window only
# when the canary lands within GATE of it. That makes GOOD a property of (host,
# canary, toolchain) -- not of the repository -- and a stale GOOD fails in the
# dangerous direction: if the true floor has dropped below the recorded GOOD,
# the gate is too permissive, contaminated windows are accepted as clean, and
# the sweep reports rankings it should have rejected.
#
# The committed default went stale twice, both times because the canary used to
# be a real codelet from the package under test, so tuning work moved it. That
# is fixed at the source -- BenchmarkGateCanary is a frozen, self-contained
# workload (internal/kernels/canary_bench_test.go) that shares no code with any
# kernel -- but the per-host half of the problem remains, so GOOD is now
# derived and recorded rather than hardcoded:
#
#   CALIBRATE=1 scripts/bench_gated.sh
#
# takes CALREPS samples and records the minimum. The minimum is the right
# estimator because interference is one-sided: contention and throttling can
# only ever make a sample slower, so the floor over enough samples converges to
# the quiet machine from above. Calibrate on an IDLE machine -- check the load
# average first. Calibrating under load records an inflated floor, which is
# precisely the too-permissive gate this is meant to prevent.
#
# Each row in CALFILE carries host, CPU model, Go version, canary name, floor
# and date, so a calibration can never be silently reused across a machine, a
# toolchain or a canary it does not belong to; the lookup below refuses rather
# than guesses. Rows are committed on purpose: they are the provenance behind
# every ratio in docs/CODELET_BENCHMARKS.md.
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
GATE="${GATE:-1.25}"
MAXWAIT="${MAXWAIT:-40}"
OUTDIR="${OUTDIR:-benchmarks/gated}"
THERMAL="${THERMAL:-/sys/class/thermal/thermal_zone10/temp}"
CALREPS="${CALREPS:-30}"
CALFILE="${CALFILE:-benchmarks/canary-calibration.tsv}"

# The canary must be cheap, stable, NOT part of the sweep, and -- the part that
# bit us twice -- independent of the code being tuned. BenchmarkGateCanary is a
# frozen synthetic workload with no library dependency, so optimising a kernel
# can no longer move the gate. See internal/kernels/canary_bench_test.go.
CANARY='BenchmarkGateCanary'

# Identity of this calibration: a GOOD is only valid for the triple it was
# measured on, so all three go in the record and are matched on lookup.
cal_host="$(hostname)"
cal_cpu="$(awk -F': ' '/^model name/ { print $2; exit }' /proc/cpuinfo 2>/dev/null || echo unknown)"
cal_go="$(go version 2>/dev/null | awk '{ print $3 }')"

mkdir -p "$OUTDIR"
log="$OUTDIR/sweep.log"

bin="$OUTDIR/kernels.test"
echo "building $bin" | tee -a "$log"
go test -c -o "$bin" ./internal/kernels

run_cell() { # run_cell <cell-name> <output-file>
  "$bin" -test.bench="^$1\$" -test.benchtime="$BT" -test.count=1 -test.run='^$' >>"$2" 2>&1
}

canary_ns() {
  # awk must NOT `exit` on the first match: that closes the pipe under the
  # still-writing test binary, which takes SIGPIPE, and `set -o pipefail` then
  # turns a successful measurement into a 141 that `set -e` kills the sweep
  # with. It is load-dependent -- the slower the writer, the likelier awk wins
  # the race -- so it struck hardest while waiting out a busy machine, which is
  # the one time this script has to be reliable. Read the input to the end.
  "$bin" -test.bench="^$CANARY\$" -test.benchtime="$BT" -test.count=1 -test.run='^$' 2>&1 |
    awk '/^Benchmark/ && !seen { print $3; seen = 1 }'
}

temp_c() {
  [[ -r "$THERMAL" ]] && echo $(($(cat "$THERMAL") / 1000)) || echo "?"
}

# --- GOOD: calibrate, or look up this host's recorded floor -------------------

cal_lookup() { # -> floor ns on stdout, or empty
  [[ -r $CALFILE ]] || return 0
  awk -F'\t' -v h="$cal_host" -v g="$cal_go" -v c="$CANARY" \
    '$1 == h && $3 == g && $4 == c { print $5 }' "$CALFILE" | tail -1
}

if [[ -n ${CALIBRATE:-} ]]; then
  load="$(awk '{ print $1 }' /proc/loadavg 2>/dev/null || echo '?')"
  echo "calibrating $CANARY on $cal_host ($cal_cpu, $cal_go)"
  echo "load average $load -- calibrate on an IDLE machine; under load this"
  echo "records an inflated floor and the gate silently becomes too permissive."

  floor=""
  for ((r = 1; r <= CALREPS; r++)); do
    s="$(canary_ns)"
    [[ -z $s ]] && continue
    if [[ -z $floor ]] || awk -v a="$s" -v b="$floor" 'BEGIN { exit !(a < b) }'; then
      floor="$s"
    fi
    printf '\r  sample %d/%d  last=%s  floor=%s ns  %sC' "$r" "$CALREPS" "$s" "$floor" "$(temp_c)"
  done
  echo

  [[ -z $floor ]] && {
    echo "calibration failed: no canary samples" >&2
    exit 1
  }

  mkdir -p "$(dirname "$CALFILE")"
  if [[ ! -s $CALFILE ]]; then
    printf '# host\tcpu\tgo\tcanary\tfloor_ns\tdate\n' >"$CALFILE"
  fi
  # Replace any previous row for this exact triple, keeping every other host's.
  tmp="$CALFILE.tmp$$"
  awk -F'\t' -v h="$cal_host" -v g="$cal_go" -v c="$CANARY" \
    '!($1 == h && $3 == g && $4 == c)' "$CALFILE" >"$tmp"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$cal_host" "$cal_cpu" "$cal_go" "$CANARY" "$floor" "$(date +%F)" >>"$tmp"
  mv "$tmp" "$CALFILE"

  echo "recorded GOOD=$floor in $CALFILE"
  echo "commit it: the gate behind every published ratio depends on it."
  exit 0
fi

GOOD="${GOOD:-$(cal_lookup)}"
if [[ -z $GOOD ]]; then
  cat >&2 <<EOF
no calibration for this host/toolchain/canary, and GOOD is unset.

  host   $cal_host
  cpu    $cal_cpu
  go     $cal_go
  canary $CANARY
  file   $CALFILE

Refusing to guess: an inherited GOOD from another machine makes the quiet-window
gate meaningless in whichever direction it is wrong. On an idle machine run

  CALIBRATE=1 ${BASH_SOURCE[0]}

or pass a known floor explicitly with GOOD=<ns>.
EOF
  exit 1
fi

# Echoes the accepted canary value on stdout, or returns 1 after giving up.
wait_quiet() {
  local i=0 c
  local limit
  limit=$(awk -v g="$GOOD" -v f="$GATE" 'BEGIN { printf "%d", g * f }')
  while ((i < MAXWAIT)); do
    c="$(canary_ns)"
    if [[ -n $c ]] && awk -v c="$c" -v l="$limit" 'BEGIN { exit !(c <= l) }'; then
      # A canary far BELOW the recorded floor means GOOD is stale on the high
      # side, so the gate has quietly widened. Warned once per run, to stderr:
      # wait_quiet's stdout is the accepted value.
      if awk -v c="$c" -v g="$GOOD" 'BEGIN { exit !(c < g * 0.85) }' &&
        [[ ! -e $OUTDIR/.stale-warned ]]; then
        touch "$OUTDIR/.stale-warned"
        echo "warning: canary ${c}ns is >15% under GOOD=${GOOD}ns; this host's" >&2
        echo "  calibration is stale and the gate is too permissive." >&2
        echo "  Re-run with CALIBRATE=1 on an idle machine." >&2
      fi
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
echo "canary: $CANARY  GOOD=$GOOD ns  GATE=$GATE  ($cal_host, $cal_go)" | tee -a "$log"

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

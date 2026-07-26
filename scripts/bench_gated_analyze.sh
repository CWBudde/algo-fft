#!/usr/bin/env bash
# Analyse a canary-gated sweep produced by scripts/bench_gated.sh.
#
# Acceptance: a group counts only if both of its canaries are within GATE of
# GOOD *and* agree with each other within DRIFT. The second condition is what
# rejects a window that degraded while the group was running — the failure mode
# a once-per-pass canary cannot see.
#
# Reporting: every candidate is expressed as a ratio to its group's incumbent,
# with the ratio taken WITHIN each accepted group and the ratios then medianed.
# Never a ratio of medians: the two differ whenever the machine drifts between
# groups, which is the normal case here.
#
# Usage: scripts/bench_gated_analyze.sh [outdir]      # default benchmarks/gated
#
# Environment: GOOD, GATE, DRIFT (defaults 1810, 1.25, 1.25)
set -euo pipefail

outdir="${1:-benchmarks/gated}"
GOOD="${GOOD:-1810}"
GATE="${GATE:-1.25}"
DRIFT="${DRIFT:-1.25}"

if [[ ! -d $outdir ]]; then
  echo "bench_gated_analyze: no such directory: $outdir" >&2
  exit 1
fi

shopt -s nullglob
passes=("$outdir"/pass*.txt)
if [[ ${#passes[@]} -eq 0 ]]; then
  echo "bench_gated_analyze: no pass files in $outdir" >&2
  exit 1
fi

awk -v good="$GOOD" -v gate="$GATE" -v drift="$DRIFT" -v incfile="$outdir/incumbents.txt" '
BEGIN {
  while ((getline line < incfile) > 0) {
    split(line, f, " ")
    if (f[1] != "") incumbent[f[1]] = f[2]
  }
}

/^#canary /      { split($2,a,"="); split($3,b,"="); split($4,c,"=")
                   cg = a[2] "|" b[2]; pre[cg] = c[2] + 0; next }
/^#canary_post / { split($2,a,"="); split($3,b,"="); split($4,c,"=")
                   post[a[2] "|" b[2]] = c[2] + 0; next }

/^BenchmarkCodeletCandidates/ {
  name = $1; sub(/-[0-9]+$/, "", name)
  for (i = 1; i <= NF; i++) if ($i == "ns/op") ns = $(i-1)
  split(name, f, "/")
  prec = f[1]; sub(/BenchmarkCodeletCandidates/, "", prec)
  size = f[2]; sub(/^size/, "", size)
  sig = f[3]; dir = f[4]
  g = prec "|" size

  v[cg, sig, dir] = ns + 0
  seen[g, sig] = 1
  groups[cg] = g
  next
}

END {
  limit = good * gate
  na = 0
  for (cg in groups) {
    if (!(cg in pre) || !(cg in post)) { rejm++; continue }
    if (pre[cg] > limit || post[cg] > limit) { rejg++; continue }
    r = (pre[cg] > post[cg]) ? pre[cg] / post[cg] : post[cg] / pre[cg]
    if (r > drift) { rejd++; continue }
    acc[++na] = cg
  }
  printf "accepted groups: %d   rejected: %d over gate, %d drift, %d incomplete\n",
         na, rejg + 0, rejd + 0, rejm + 0

  n = 0
  for (k in seen) { split(k, kf, SUBSEP); if (!(kf[1] in grpseen)) { grpseen[kf[1]] = 1; glist[++n] = kf[1] } }
  asort_str(glist, n)

  for (gi = 1; gi <= n; gi++) {
    g = glist[gi]
    split(g, gf, "|")
    inc = incumbent[g]
    printf "\n=== n=%s  complex%s   incumbent: %s ===\n", gf[2], gf[1], (inc == "" ? "(unknown)" : inc)
    printf "  %-40s %11s %11s %9s %9s %5s\n", "candidate", "fwd ns", "inv ns", "fwd rel", "inv rel", "grps"

    m = 0
    for (k in seen) {
      split(k, kf, SUBSEP)
      if (kf[1] != g) continue
      s = kf[2]
      m++
      key[m] = s
      fwd[m] = med(g, s, "forward", na)
      inv[m] = med(g, s, "inverse", na)
      frel[m] = rel(g, s, inc, "forward", na)
      irel[m] = rel(g, s, inc, "inverse", na)
      cnts[m] = cnt(g, s, "forward", na)
    }
    # sort by forward median, so a mis-tuned priority shows as an incumbent
    # that is not on the first line
    for (i = 2; i <= m; i++) {
      for (j = i; j > 1 && fwd[j] < fwd[j-1]; j--) {
        swap(key, j); swap(fwd, j); swap(inv, j)
        swap(frel, j); swap(irel, j); swap(cnts, j)
      }
    }

    for (i = 1; i <= m; i++)
      printf "  %-40s %11.1f %11.1f %9.3f %9.3f %5d%s\n", key[i], fwd[i], inv[i],
             frel[i], irel[i], cnts[i], (key[i] == inc ? "  <= incumbent" : "")

    delete key; delete fwd; delete inv; delete frel; delete irel; delete cnts
  }
}

function med(g, sig, dir, na,   i, cg, a, k, r) {
  k = 0
  for (i = 1; i <= na; i++) {
    cg = acc[i]
    if (groups[cg] != g) continue
    if ((cg, sig, dir) in v) a[++k] = v[cg, sig, dir]
  }
  if (k == 0) return 0
  asort_num(a, k)
  r = (k % 2) ? a[(k+1)/2] : (a[k/2] + a[k/2+1]) / 2
  delete a
  return r
}
# ratio to the incumbent, taken within each group and then medianed
function rel(g, sig, inc, dir, na,   i, cg, a, k, r) {
  if (inc == "") return 0
  k = 0
  for (i = 1; i <= na; i++) {
    cg = acc[i]
    if (groups[cg] != g) continue
    if ((cg, sig, dir) in v && (cg, inc, dir) in v && v[cg, inc, dir] > 0)
      a[++k] = v[cg, sig, dir] / v[cg, inc, dir]
  }
  if (k == 0) return 0
  asort_num(a, k)
  r = (k % 2) ? a[(k+1)/2] : (a[k/2] + a[k/2+1]) / 2
  delete a
  return r
}
function cnt(g, sig, dir, na,   i, cg, k) {
  k = 0
  for (i = 1; i <= na; i++) { cg = acc[i]; if (groups[cg] == g && (cg, sig, dir) in v) k++ }
  return k
}
function asort_num(a, n,   i, j, t) {
  for (i = 2; i <= n; i++) { t = a[i]; for (j = i - 1; j >= 1 && a[j] > t; j--) a[j+1] = a[j]; a[j+1] = t }
}
function asort_str(a, n,   i, j, t) {
  for (i = 2; i <= n; i++) { t = a[i]; for (j = i - 1; j >= 1 && a[j] > t; j--) a[j+1] = a[j]; a[j+1] = t }
}
function swap(a, j,   t) { t = a[j]; a[j] = a[j-1]; a[j-1] = t }
' "${passes[@]}"

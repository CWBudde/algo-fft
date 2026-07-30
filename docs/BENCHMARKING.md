# Benchmarking protocol

How to take a number in this repository that is worth acting on. Most of the
wrong conclusions in the project's history came from trusting a measurement
taken under load; everything here is a rule that cost a real investigation.

`PLAN.md` §2.2 is the short form and points here.

## Measurement protocol

- **Interleave the arms** in one process, with the order rotated per round, and
  report medians. Arms run minutes apart are measuring the machine.
- **Canary-bracket every group**, not every pass. A 94-cell pass takes 5–13
  minutes, so contention arriving mid-pass goes unseen: 3 of 5 nominally clean
  passes were contaminated, one by 50×. In-tree as `scripts/bench_gated.sh` +
  `scripts/bench_gated_analyze.sh` / `just bench-gated`. A group is one
  (precision, size) with all its candidates back-to-back, so a whole ranking is
  taken inside a single verified-quiet window.
- **Recalibrate the canary each round** from the observed floor. A stale `GOOD`
  does not bias the ratios — those are taken within a group — but it lets in
  windows that should have been rejected. The default 1810 has been stale twice
  over; the floor has measured 1565 and ~1590.
- **Check the arithmetic before reading a ratio.** `bench_gated.sh` takes its
  output directory from `OUTDIR`; `bench_gated_analyze.sh` takes it as a
  **positional argument** and ignores `OUTDIR`. Invoking the analyser with
  `OUTDIR=` once silently analysed a stale directory and produced a
  plausible-looking result from the wrong data. Accepted + rejected must equal
  groups × passes.
- **Pin with `taskset`.** On a hybrid part an unpinned benchmark lands on
  P-cores or E-cores arbitrarily, and some effects (the AVX↔SSE transition
  penalty) exist only on one of them.
- **Contention and heat are independent failure modes.** One reading blew up
  13× while package temperature _fell_ 92 → 61 °C; another process was at 111%
  CPU. A protocol that only waits on a temperature threshold accepts that
  window.
- **Contention can invert an ordering, not merely inflate it.** At n = 16384
  Stockham appeared to beat the codelet 258 vs 329 µs under load; idle and
  pinned the codelet wins 73.7 vs 94.3.
- **Prefer a single binary with an env knob** over two builds when isolating a
  path — it removes code layout as a variable, which is otherwise inseparable.
- **Include a null control**: a cell the change cannot reach. If it moves, the
  run is measuring the machine.
- **A speedup can break a threshold test that measures nothing it touched**,
  whenever the threshold is expressed relative to a peer.
  `TestRadix4AVX2Ranking` fails radix-4 above 1.5× the _fastest_ codelet at that
  size, so speeding the runner-up up at n = 16384 tightened radix-4's headroom
  from ~211 µs to ~90 µs without a line of the test changing. Fixed by
  re-measuring before failing (`rankingAttempts = 3`): a real regression
  reproduces on every pass, a contended window does not. Note the failure mode
  the retry addresses — a burst covering all rounds of _one_ candidate inflates
  a single codelet rather than the group, so it surfaces as a ranking change,
  not as uniformly slower numbers. Best-of-N within a pass cannot see that.

## Hardware tiers

Three machines are reachable, and they are complementary rather than redundant
— several findings above exist only because a result differed between them.
Server access is weekend-only, so none of this belongs in CI; treat them as
periodic validation sweeps.

- **Dev laptop (i7-1255U, AVX2, no AVX-512).** The only one with FFTW
  installed, so the only place the external gap can be measured. Throttles
  hard (86–98 °C under sustained benchmarking) — interleave arms, trust ratios.
  Benchmark with `/usr/local/go/bin/go` under `taskset -c 0`, never the `go` on
  `PATH` (see below). A 2+8 hybrid: `cpu0-3` are P-cores at 4.7 GHz, `cpu4-11`
  E-cores at 3.5 GHz.
- **64-core host, no AVX at all** (SSE4.2 ceiling). Valuable _because_ it is
  limited: the only place the SSE2/SSE3 codelet tier is what dispatch actually
  selects. On any AVX2 machine those codelets lose the priority ladder and ship
  effectively unbenchmarked in situ. Also a good proxy for the scalar-Go paths
  that dominate purego and WASM. Shared with other tenants — ratios only.
- **Xeon Gold 5218 (AVX2 + AVX-512).** The only AVX-512 hardware. Doubles as a
  second, non-throttling AVX2 reference, which is how the forward-vs-inverse
  anomaly got localized to the laptop. 2 vCPU, no gcc — no cgo, no FFTW
  baseline. Cascade Lake downclocks under AVX-512, so it is a pessimistic
  machine for that tier.

FFTW can be used on the servers without installing anything by shipping
`libfftw3{,f}.so.3` plus `fftw3.h` from a matching distro release and pointing
`CGO_CFLAGS`/`CGO_LDFLAGS`/`LD_LIBRARY_PATH` at them — but that needs a gcc on
the target.

## Standing lessons about measurement

- **A number measured in another repository is not maintained by editing this
  one.** `PLAN.md`'s headline FFTW ratio aged through four releases while every
  round below it stayed current, because the sweep is committed in
  `go-fft-bench` against a pinned tag and nothing pulls it back. It was wrong by
  a factor of two and in the wrong _direction_ — the library had overtaken FFTW3
  at powers of two and the file still said 0.63×. Quote such a number with the
  tag it came from, or re-measure it.
- **Check the toolchain wrapper before blaming the machine.** `go` on the dev
  laptop resolves to a wrapper that runs `nice -n 10 taskset -c 0-$(nproc-2)`,
  so benchmarks yield to the desktop; the same sweep is clean pinned and
  un-niced at load 4.4 and unusable through the wrapper. (`taskset -c 0 go ...`
  fails with `CPU list ... 0--1` — that error means you are wrapping the
  wrapper.)
- **A registered fast path is not a reachable one.** Codelets for exactly the
  sizes the Bluestein pad produces sat in the registry, correct and never
  called, because that route entered a hardcoded size switch instead. The
  symptom was two builds measuring the _same_ — the same tell as "an
  optimisation that changes nothing". Before profiling a path that looks slow,
  check that the fast version of it runs at all.
- **A size-generic kernel silently closes per-size gaps the plan still lists as
  open.** An item once asked for a file that will never exist; the kernel is
  `avx2_f64_radix4.s`, which covers every `n = 2*4^k` as a radix-4-then-2. Ask
  "does the generic kernel's shape rule cover n?", never "is there a file named
  for it?" — a name-based search returns the stale premise as confirmation.
- **A dispatch toggle's stated reason must be re-derived, not inherited.** The
  packed-Stockham toggle claimed the codelet path superseded it; every codelet
  is registered as `KernelDIT`, so the strategy check upstream had already
  excluded it and the toggle only ever suppressed the sizes with no codelet at
  all.

Per-size ranking results live in [`CODELET_BENCHMARKS.md`](CODELET_BENCHMARKS.md);
assembly-specific lessons live in [`../AGENTS.md`](../AGENTS.md).

# Engineering history

Every round that has landed, one entry each. This is the detail that used to be
`PLAN.md` §1.1; `PLAN.md` now carries only the current status and the open work.

Dates are the month the round landed. Follow the links for the measurements and
the reasoning behind each.

**Foundations (2026-07).**

- **Correctness & build integrity.** SIMD build compiles and is CI-gated on
  amd64/386/arm64; ~3.9k lines of dead code deleted; all plan types
  concurrency-safe under `-race`; unschedulable radices panic rather than
  return a wrong answer; every registered codelet verified per-direction against
  `reference.NaiveDFT`.
- **Architecture consolidation.** One kernel contract (`CodeletFunc` returns
  `bool`; every call site honors the bail signal). One precision scheme —
  `NewPlan[T]`, `NewPlanReal[F, C]`, `PlanReal2D/3D[F, C]` — which closed the
  missing float64 real-2D/3D gap. One plan interface (`PlanInfo`). Public types
  owned by the root package, no aliases into `internal/*`. `Plan2D`/`Plan3D` are
  thin wrappers over `PlanND`. New leaf package `internal/registry` so `kernels`
  no longer registers _upward_ into `planner`. `Plan[T]` split into four
  `planExecutor` implementations (kernel/Bluestein/Rader/recursive), 40 fields →
  21, with the codelet binding cached as a zero-dispatch fast path because
  interface dispatch cost ~20 ns at n = 8. One deliberate global remains: the
  default Wisdom cache, with per-consumer isolation via `PlanOptions.Wisdom`.
- **complex128 kernel twins generated.** `cmd/genkernels` emits
  `<base>_c128.gen.go` from the complex64 sources (42 files, 108 functions);
  ~9.9k hand-written lines deleted. A pre-generation audit found 16 twins had
  drifted — the complex64 side had been optimized later — so generating them was
  itself a free win (1024/radix4 −27%/−18%). Deliberately still hand-written:
  the radix-3/5 c128 entry points and `dit_16384_radix4`'s c128 pair, whose
  `[16384]complex128` stage arrays exceed the compiler's 128 KiB stack limit.
- **Kernel strategy is per-plan** (`PlanOptions.Strategy`), no process-global
  state; tuning persists via the versioned Wisdom cache (v3; v2 files are
  rejected, not reinterpreted). Zero-allocation parity across
  1D/2D/3D/ND/real/mixed-radix on both default and SIMD paths, locked by
  `AllocsPerRun` guards.
- **Testing & CI.** Every arch matrix leg builds, vets and tests both the
  default and `purego` builds; lint green; coverage gated at 90%; nightly
  benchmarks against a committed baseline; continuous fuzzing with committed
  corpus; property tests (Parseval/linearity/shift) across all dispatch
  families.

**Algorithms (2026-07).** See [`MIXED_RADIX.md`](MIXED_RADIX.md).

- **Shape-aware Bluestein padding.** A single scalar penalty could never be
  right: a mixed-radix sub-FFT's cost per `m·log2(m)` spans ~7× on shape alone.
  The model is a whitelist (`padShapes`, `plan_padsize.go`), each shape admitted
  only above the pad size where it wins at **both** precisions: `3·2^(k-2)` from
  2^9, `15·2^(k-4)` from 2^13; `7·2^(k-3)` is dominated outright. End-to-end
  −15…−57% at the affected lengths.
- **Rader's algorithm**, gated on measured wins rather than on `p−1` being
  smooth. Power-of-two `p−1` wins 4–5×; 5-smooth `p−1 ≥ 96` with pow2 part ≥ 8
  wins 1.1–5.6×. Padded Rader for non-smooth `p−1` is intentionally skipped
  (pad ≥ `2p−3` vs Bluestein's `2p−1` is a wash).
- **Split-radix (conjugate-pair) DIT** with full strategy plumbing; beats the
  auto path at every power of two ≥ 256 on purego (+11–34%, 2.1× at 262144). No
  longer auto-selected anywhere after the square-rule re-measure.
- **Radix-8 stage for the generic DIT driver**, gated to the no-codelet path:
  geomean −16.9% across 32…12288 on purego, so it benefits purego, SSE-only
  amd64 and arm64 without touching AVX2 schedules.
- **Radix-7 / radix-11 butterflies** extend exact coverage to
  `2^a·3^b·5^c·7^d·11^e`.
- **Real-FFT for odd/multi-factor lengths.** `NewPlanReal*` accepts any n ≥ 2;
  odd lengths run an internal full-size complex FFT with DC-only spectrum
  validation.
- **The mixed-radix engine went from 0.20×/0.13× FFTW3 to today's figures.** The
  largest single cause was a **driver defect**, not a routing one: the AVX2
  drivers guarded codelet dispatch with `n > 1` while the scheduler requires
  `n > 5`, so every schedule ending in radix 2/3/4/5 sent each leaf through a
  full codelet call — 1225 dispatches per transform at n = 4900. Fused AVX2
  stage kernels for radix 3/5/7/11 then gave geomean −30%, with n = 704 dropping
  10244 → 2609 ns.

**Memory & cache (2026-07).**

- **Cache-blocked transpose** — the O(n²) swap-pair index table is gone;
  `math.TransposeSquare` tiles in place with edge 8. Transpose −70…−82%;
  six-step/eight-step −10…−23% at n ≥ 65536; square `Plan2D` −34.6% geomean.
- **Four-step** (`KernelFourStep`), the rectangular generalization of six-step,
  with the `n1×n2` split chosen by a cache-residency model over the L1d/L2 sizes
  `internal/cpu` detects. Beats split-radix at 2^21…2^23.
- **Power-of-two squares are no longer special-cased.** Measured across all
  candidate strategies at 2^18/2^20/2^22, both directions, precisions and
  builds: Stockham wins or ties every arm bar one. One dissenting arm is
  accepted knowingly (2^20 c128 forward prefers six-step) because a
  precision- and direction-blind rule cannot capture it.
- **Twiddle-table bandwidth.** Radix-4 stages conjugate on load, halving
  per-plan packed-table memory for free. Quarter-table symmetry was evaluated
  and **declined** for the scalar path — an L1-resident tiny-table experiment
  put the cache-footprint share at only ~4–8%, so octant-decode ALU would be a
  net loss. Revisit only inside SIMD kernels.

**The complex64 scalar-multiply defect (2026-07).** The single most valuable
finding of the comparative sweep, and the reason complex64 was _slower_ than
complex128 at 20 of 23 non-power-of-two lengths. **Go's compiler does not
implement scalar `complex64 * complex64` in single precision** — it widens all
four components to float64, multiplies in double precision and rounds back:
twelve instructions against six for the same expression on complex128. Only the
multiply promotes, so any FFT stage written as scalar Go is _structurally_ more
expensive in complex64. Powers of two hid it inside float32 SIMD codelets; the
arbitrary-length routes could not. Fixed in three rounds with
`math.MulComplex64` — arbitrary-length glue (c64/c128 ratio 1.18–1.27 → 0.90–0.98),
1378 products across 39 pure-Go codelet sources rewritten by a `go/types`-driven
tool (purego geomean **−24.4%** over the 8…16384 ladder), and 17 remaining
sites. Two independent hosts three generations apart agreed on the codelet
round's geomean to within 0.3% _and_ on which sizes did not move — the signature
of a code change rather than a measurement artifact. Accuracy cost is sub-ulp;
see [`PRECISION.md`](PRECISION.md). Standing rules from this round are
in `../PLAN.md` §2.3.

**SIMD kernels (2026-07).** Coverage: AVX2 broad in both precisions, SSE2/SSE3
tier to 32768, NEON size-specific 4–32768 both precisions, a first AVX-512 tier.

- **SIMD ships in the default build** behind runtime CPU detection (`-tags
purego` opts out; `-tags asm` is a no-op). All known-incorrect kernels fixed;
  ~1,000 `asmdecl` findings resolved and vet-gated.
- **The 256-bit radix-4 kernels** replaced a whole AVX2 codelet family that was
  XMM-width in VEX clothing — see [`AVX2_RADIX4.md`](AVX2_RADIX4.md).
  This is the change that made powers of two competitive with FFTW3.
- **Legacy-SSE encodings swept out of the AVX2/AVX-512 tree**: 4089 instructions
  converted to VEX across 59 files, then the last ten functions in a second
  round. End-to-end performance-neutral, but it uncovered six codelets carrying
  a fixed ~100 ns transition-penalty prologue regardless of transform size
  (`dit4_radix4_avx2` 102.9 → 5.0 ns) — which is what had mis-ranked them in the
  registry. The n = 64 cliff (0.36× FFTW3) was one legacy `MOVD AX, X8`.
  Method and traps are in [`AGENTS.md`](../AGENTS.md).
- **FMA audit, two passes.** Every AVX2 codelet the registry actually _selects_
  is now fused (97 sites in the second pass alone). The tier is gated on
  `HasAVX2 && HasFMA`, so an FMA-masked VM falls back instead of faulting.
  _Performance was never demonstrated_ — three benchstat attempts were swamped
  by thermal throttling; treat that pass as instruction-count and accuracy work.
  Remaining scope in §4.
- **AVX2 complex128 Stockham asm** for every Stockham-resolved c128 size above
  the codelet range: kernel-level −16…−50%, 65536 end-to-end 1.44 → 1.02 ms.
- **SSE tier extended to 16384/32768** in both precisions, emitted by a one-off
  generator validated by byte-reproducing the existing 4096/8192 files.
- **NEON ladder completed 4 → 32768** in both precisions across three subagent
  delegation rounds. Priorities are ladder-mirrored, **not** tuned — QEMU timing
  is meaningless (§6).
- **Real-FFT forward recombination in SIMD**: the kernel is 4.5–8× the scalar
  loop, `BenchmarkPlanRealForward` −34.7% geomean.

**Three unreachable fast paths (2026-07-29).** The same bug three times — a
correct, registered, faster path that nothing called, each behind a dispatch
decision whose stated justification had never been re-derived.

- **The size-384 path.** The AVX2 assembly for the c64 side had been written,
  assembled, declared and never called — and it was also _wrong_
  (`VINSERTPS $0x10` where `VMOVLHPS` was needed). Separately, both precisions'
  128-point sub-FFTs were bound to superseded kernels. Fixing the binding alone,
  with no new assembly, gave geomean **−58%**.
- **The six-step row FFTs** were still on the pre-radix-4 128-point kernel at
  six call sites — 320 ns where the generic kernel is 88. The swap also _removed_
  per-call work, because those files had been gathering their length-128 row
  table out of the caller's table on every transform. Geomean **−33%**.
- **The Bluestein sub-FFT never reached the registry**, so the default build was
  ~4% _slower_ than `-tags purego` at n = 1009 and 2003 — both builds ran the
  identical pure-Go kernel for ~96% of the work. See
  [`MIXED_RADIX.md`](MIXED_RADIX.md).
- **Packed Stockham was compiled out of SIMD builds** on the stated grounds that
  the codelet path superseded it. Every codelet is registered as `KernelDIT`, so
  the strategy check upstream had already excluded it; what the constant actually
  suppressed was the sizes with _no_ codelet, where the SIMD build fell through
  to a radix-2 Stockham kernel while the radix-4 route it had disabled was up to
  2.7× faster.

**Reporting and hygiene (2026-07-28 … 07-30).**

- **Plans report the route they take.** Non-power-of-two lengths used to name a
  strategy that never executed (1000 said `dit_fallback`; 2205/3600/12000/44100
  said `stockham`). `KernelMixedRadix` is now a first-class strategy resolved
  before anything else, and the rule is that the reported strategy always equals
  the executed one.
- **Test-vector blindness audit.** Four tests were checking nothing and one
  whole precision was uncovered above n = 16. See
  [`TESTING.md`](TESTING.md).
- **The race suite could not finish.** `internal/kernels` took 1499.7 s under
  `-race` against Go's 10-minute default, which neither `just test` nor CI
  overrode — a gate that had been red repo-wide. Details in `../CHANGELOG.md`.
- **Incumbent audits.** Every registered power-of-two size has now been ranked
  under the canary-gated sweep. Results, including the one mis-tuned row it
  found (complex128 at n = 8), are in
  [`CODELET_BENCHMARKS.md`](CODELET_BENCHMARKS.md).

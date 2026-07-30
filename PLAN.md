# PLAN.md — algofft Roadmap

This roadmap is the source of truth for status and direction.

- **§1** — where the library stands, and a condensed record of what has landed.
- **§2** — the working method every item below is held to (correctness gates,
  measurement protocol, hardware tiers, standing lessons).
- **§3–§9** — the open work, in execution order. §3 is closed; §4–§8 are the
  performance and coverage rounds, and the v1.0 tag in §9 comes after them.
- **§10** — post-v1.0 wishlist.

Completed items are compressed to their outcome, the numbers that justify a
tuning constant still in the tree, and any rule worth not relearning. The
full item-by-item narration is preserved in this file's git history; the
reusable assembly lessons also live in `AGENTS.md`.

Design philosophy lives in `docs/goal.md`; the component inventory is generated
into `docs/IMPLEMENTATION_INVENTORY.md` via `go generate ./internal/kernels/...`
(which runs `cmd/gencodelets -inventory <path>`).

---

## 1. Status

The v1.0 engineering work is **complete** — the API is settled and nothing in
the remaining backlog changes a signature. The correctness debt in §3 that
gated the tag is now **closed** (§1.13, §1.14), so nothing is left that a
signature change could fix. The tag has nevertheless been moved behind the
performance and coverage work in §4–§8: §9 is now the last section before the
wishlist, and shipping waits on those rounds rather than on release mechanics
alone.

**Where the library sits against others** (i7-1255U, `go-fft-bench`, v0.7.4,
2026-07-29): powers of two are **1.36× FFTW3** forward and 1.34× inverse by
geomean, and ~8× the rest of the Go field. That is an inversion — the figure
here read 0.63× until it was re-measured (§1.15), against a v0.7.0 baseline
that predated the 256-bit radix-4 kernels. Non-power-of-two lengths remain the
weak half at 0.60× by geomean: 44100 has gone from losing to gonum (4.00 ms vs
2.59 ms) to 786 µs, but FFTW3 does it in 185 µs. The remaining gap is
concentrated in the mixed-radix engine (§5) and a handful of power-of-two soft
spots (§4).

### 1.1 v1.0 engineering (2026-07)

- **Correctness & build integrity.** SIMD build compiles and is CI-gated on
  amd64/386/arm64; ~3.9k lines of dead code deleted; all plan types
  concurrency-safe (per-call scratch caches, `-race`); no silent wrong-answer
  paths (unschedulable radices panic, scheduler/driver contract validated);
  every registered codelet verified forward-vs-`reference.NaiveDFT` per
  direction.
- **Architecture hardening.** Kernel strategy is per-plan
  (`PlanOptions.Strategy`), no process-global state; tuning persists via the
  versioned Wisdom cache. Zero-allocation parity across 1D/2D/3D/ND/real/
  mixed-radix on both default and SIMD paths, locked by `AllocsPerRun` guards.
  Plan-layer and DSP `*128` duplication collapsed into generics; codelet
  registration generated from a declarative table (`cmd/gencodelets`).
  One deliberate global remains: the default Wisdom cache
  (`fft.DefaultWisdom`); per-consumer isolation via `PlanOptions.Wisdom`.
- **SIMD that ships.** SIMD is on the **default build** behind runtime CPU
  detection (`-tags purego` opts out; `-tags asm` is a no-op). All
  known-incorrect kernels fixed; ~1,000 `asmdecl` findings resolved and
  vet-gated. Coverage: AVX2 broad in both precisions, SSE2/SSE3 tier to 32768,
  NEON size-specific 4–32768 both precisions, first AVX-512 tier.
- **API completeness & polish.** No dead knobs; introspection parity;
  plan-reuse DSP types (`Convolver`, `Correlator`, `RealConvolver`); consistent
  `%w` wrapping; GoDoc audit enforced by revive.
- **Testing & CI.** Every arch matrix leg builds, vets and tests both the
  default and `purego` builds; lint green; coverage gated at 90%; nightly
  benchmarks against a committed baseline; continuous fuzzing with committed
  corpus; property tests (Parseval/linearity/shift) across all dispatch
  families.

### 1.2 Architecture consolidation (2026-07, pre-tag)

Outcome of the 2026-07 architecture review. Breaking changes landed directly —
no deprecation shims, no transition aliases.

- **One kernel contract.** `fftypes.CodeletFunc` returns `bool` like
  `Kernel[T]`; the lossy `wrapCodelet64/128` adapters are gone and every call
  site honors the bail signal (recursive executor falls back to generic DIT,
  AVX2 mixed-radix falls through to Go, `FastPlan` panics on a
  caller-contract violation). Previously a bailing codelet silently no-oped.
- **One precision scheme, everything generic.** `NewPlan[T]`,
  `NewPlanReal[F, C]`, `FastPlanReal[F, C]`, `PlanReal2D/3D[F, C]` — which
  closed the missing float64 real-2D/3D gap. `Planner` and its methods deleted;
  `New*WithOptions` is the one options-carrying entry point.
- **One plan interface.** `PlanInfo` (`Len`, `KernelStrategies`, `Algorithms`,
  `String`, `Close`) on every plan type with compile-time assertions.
  `Meta()`/`PlanMeta`, `InPlace()`, `PlanOptions.InPlace`,
  `PlanOptions.Batch`/`Stride` deleted.
- **Public types owned by the root package.** `Complex`/`Float`,
  `KernelStrategy` (root enum with `String()`), `Wisdom` wrapper. No public
  aliases into `internal/*`.
- **Multi-dimensional copies collapsed.** `Plan2D`/`Plan3D` are thin typed
  wrappers over `PlanND` (benchstat found no specialized loop that beat the ND
  path); five copies of the wrapper logic replaced by `plan_common.go`; the
  one-shot and reusable DSP pipelines share one core.
- **Internal layering repaired.** New leaf package `internal/registry` owns the
  codelet registry, so `kernels` no longer registers _upward_ into `planner`;
  `internal/fft` is no longer a façade (root imports `planner`/`kernels`/
  `transform`/`fftypes` directly); one algorithm-name ↔ strategy table.
- **`Plan[T]` split into per-strategy executors.** `planExecutor[T]` with four
  implementations (kernel, Bluestein, Rader, recursive); `Plan[T]` went from
  ~40 fields to 21 and `plan.go` from 984 to 595 lines. Interface dispatch cost
  ~20 ns at n=8, so the codelet binding is additionally cached on `Plan` as a
  zero-dispatch fast path.
- **complex128 kernel twins generated.** `cmd/genkernels` emits
  `<base>_c128.gen.go` from the complex64 sources (42 files, 108 functions);
  ~9.9k hand-written lines deleted. A pre-generation audit found 16 twins had
  drifted — the complex64 side had been optimized later — so generating them
  was itself a free win (1024/radix4 −27%/−18%). Deliberately still
  hand-written: the radix-3/5 c128 entry points and `dit_16384_radix4`'s c128
  pair, whose `[16384]complex128` stage arrays exceed the compiler's 128 KiB
  stack limit.

### 1.3 Algorithms (2026-07)

- **Bluestein padding is shape-aware.** A single scalar penalty could never be
  right: measured against the power-of-two endpoint of its own dyadic window, a
  mixed-radix sub-FFT's cost per m·log2(m) spans ~7× on shape alone. The model
  is now a whitelist of candidate shapes (`padShapes`, `plan_padsize.go`), each
  admitted only above the pad size where it wins at **both** precisions:
  `3·2^(k-2)` from 2^9, `15·2^(k-4)` from 2^13. `7·2^(k-3)` is dominated
  outright. End-to-end `Plan.Forward` −15…−57% at the affected lengths, with an
  unchanged control at 1.00. Both `bluesteinPadSize` and
  `fastConvolutionLength` ride it (convLen 257 → 384 instead of 512).
  _Unmeasured:_ the purego calibration (thresholds are conservative there), the
  (0.9375P, P] gap shapes, and 2^17+ windows.
- **Rader's algorithm** (`internal/fft/rader.go`) for primes, gated on measured
  wins rather than on p−1 being smooth: `IsMixedRadixSmooth(p−1)` plus
  `rader7Or11Wins`, fitted on 32 primes × both precisions. Power-of-two p−1
  wins 4–5×; 5-smooth p−1 ≥ 96 with pow2 part ≥ 8 wins 1.1–5.6×. One exception
  is recorded rather than papered over: 7393 regresses 9% on purego complex64
  while its other three arms win. Padded Rader for non-smooth p−1 is
  intentionally skipped (pad ≥ 2p−3 vs Bluestein's 2p−1 is a wash).
- **Split-radix (conjugate-pair) DIT**, `KernelSplitRadix` with full strategy
  plumbing. Beats the auto-selected path at every power of two ≥ 256 on purego
  (+11–34%, 2.1× at 262144); AVX2/AVX-512 codelets stay ahead below 262144 on
  the SIMD build. It is no longer auto-selected anywhere — see the square-rule
  re-measure in §1.5.
- **Radix-8 stage for the generic DIT driver** (`internal/kernels/radix8.go`),
  emitted whenever the remaining pow2 part 2^e has e ≥ 3 except e = 4. Gated to
  the no-codelet path, so AVX2 schedules are untouched. Geomean −16.9% across
  32…12288 on purego — benefits purego, SSE-only amd64 and arm64.
- **Real-FFT for odd/multi-factor lengths.** `NewPlanReal*` accepts any n ≥ 2;
  odd lengths run an internal full-size complex FFT (`plan_real_odd.go`) with
  DC-only spectrum validation. Zero-alloc in steady state, batch/stride and
  `Clone` supported. _Follow-up:_ a real-input Bluestein exploiting conjugate
  symmetry would close the ~2× gap vs a hypothetical packed odd-length method;
  the 2D/3D real plans still require even width.
- **Radix-7 / radix-11 butterflies** extend exact coverage to
  2^a·3^b·5^c·7^d·11^e, routed through `planner.MixedRadixEligible`. See §1.6
  for the gate's later collapse to a single criterion.

### 1.4 SIMD kernels (2026-07)

- **FMA audit, two passes.** Every AVX2 codelet the registry actually
  _selects_ is now fused (97 sites in the second pass alone; `VADDSUBPD` 64→0
  in `avx2_f64_size256_radix16.s`). Accuracy improved where the fused twiddle
  work is densest, as one-rounding-instead-of-two predicts. Trivial twiddles
  (±1, ±i) and real-scalar 1/n multiplies were deliberately left alone — no
  addend to fuse. The AVX2 codelet tier is now gated on `HasAVX2 && HasFMA`, so
  an FMA-masked VM falls back instead of faulting. _Performance was never
  demonstrated_ — three benchstat attempts were swamped by thermal throttling;
  treat that pass as instruction-count and accuracy work. Remaining scope is in
  §4.
- **Codelet priority retune.** `BenchmarkCodeletCandidates64/128` exposed
  systematic mis-selection: the priority-favored six-step / radix-32×32 /
  radix-16 / radix-8 codelets lost to the plain radix-4 family at every size
  where both existed. Flipping them gave −26…−57% end-to-end at 1024/8192/16384.
- **Size-32768 codelets** (generic + AVX2, both precisions) closed a 5× cliff
  where c128 fell back to scalar Go Stockham: 618 → 237 µs.
- **AVX2 complex128 Stockham asm** (`avx2_f64_stockham.s`) for every
  Stockham-resolved c128 size above the codelet range: kernel-level −16…−50%,
  65536 end-to-end 1.44 → 1.02 ms.
- **SSE tier extended to 16384/32768** in both precisions, emitted by a one-off
  generator validated by byte-reproducing the existing 4096/8192 files.
  −15…−47% vs the generic codelets they displace.
- **NEON ladder completed 4 → 32768** in both precisions across three subagent
  delegation rounds, matching the amd64 SSE3 tier. Priorities are
  ladder-mirrored, **not** tuned — QEMU timing is meaningless (§6).
- **SSE2 tier breadth**: size-2048, 4096 and 8192 kernels in both precisions
  (1.4–2.0× over the generic codelets), plus a registry entry for the
  already-existing SSE3 size-256 complex64 kernel, which had been wired into
  the fallback dispatch but never registered (1.6/2.0×).
- **Real-FFT forward recombination in SIMD**
  (`avx2_real_recombine.s` + an SSE3 tier): the kernel is 4.5–8× the scalar
  loop, `BenchmarkPlanRealForward` −34.7% geomean. A vectorized AVX2 c128
  inverse pre-pass replaced the `inverseRepackComplex128SIMD` stub.

### 1.5 Memory & cache (2026-07)

- **Cache-blocked transpose.** The O(n²) swap-pair index table is gone;
  `math.TransposeSquare` tiles in place with edge 8 (chosen by sweep; 16+ falls
  off a cliff beyond 512²). Transpose −70…−82%; six-step/eight-step −10…−23% at
  n ≥ 65536; square `Plan2D` −34.6% geomean. _Follow-up:_ a SIMD 8×8 complex
  tile kernel for a further constant factor.
- **Four-step** (`KernelFourStep`): the rectangular generalization of six-step,
  splitting any power-of-two n as n1×n2 with the split chosen by a
  cache-residency model over the L1d/L2 sizes `internal/cpu` now detects. Beats
  split-radix at 2^21…2^23. The split sweep is flat (±7%), so the auto rule is
  unchanged and measure/wisdom remains the arbiter. _Follow-up:_ SIMD row FFTs
  inside four-step — the row passes still use scalar Stockham butterflies.
- **Power-of-two squares are no longer special-cased.** The `KernelAuto` square
  branch was costing users at every size it could reach. Measured across all
  candidate strategies at 2^18/2^20/2^22, both directions, precisions and
  builds: Stockham wins or ties every arm bar one. Powers of two now fall
  through to the plain size heuristic; the eight-step branch fell to the same
  measurement. Non-power-of-two squares keep six/eight-step, unchanged and
  unmeasured. One dissenting arm is accepted knowingly: 2^20 c128 forward
  prefers six-step, but a precision- and direction-blind rule cannot capture it
  and the size's other three arms favor Stockham.
- **Twiddle-table bandwidth.** The large-n strategies already share one n-entry
  base table between directions. The real duplication was the pure-Go packed
  radix-4 route, which held a fully conjugated inverse copy; radix-4 stages now
  conjugate on load, halving per-plan packed-table memory for free or better.
  Quarter-table symmetry was evaluated and **declined** for the scalar path: an
  L1-resident tiny-table experiment isolated the cache-footprint share to only
  ~4–8%, so octant-decode ALU would be a net loss. Revisit only inside SIMD
  kernels.

### 1.6 The complex64 scalar-multiply defect (2026-07)

The single most valuable finding of the comparative sweep, and the reason
complex64 was _slower_ than complex128 at 20 of 23 non-power-of-two lengths.

**Go's compiler does not implement scalar `complex64 * complex64` in single
precision.** It widens all four components to float64, multiplies in double
precision and rounds back — twelve instructions against six for the same
expression on complex128. Only the multiply promotes. So any FFT stage written
as scalar Go is _structurally_ more expensive in complex64. Powers of two hid
it by running inside float32 SIMD codelets; the arbitrary-length routes could
not, because their odd-radix stages, chirp modulation and pointwise products
are scalar Go.

Fixed in three rounds with `math.MulComplex64` (component-wise: MULSS ×3,
VFMADD231SS, SUBSS):

| round                         | scope                                                      | result                                                                     |
| ----------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------- |
| arbitrary-length glue         | mixed-radix driver, radix-3/5/7/11, Bluestein/Rader        | c64/c128 ratio 1.18–1.27 → 0.90–0.98; c64 −21…−32% at 1000/2205/3600/12000 |
| pure-Go codelets (39 sources) | 1378 products, rewritten by a `go/types`-driven tool       | purego geomean **−24.4%** over the 8…16384 ladder, both directions         |
| everything else (17 sites)    | real repack/recombine, packed Stockham, combine, AVX2 glue | module-wide `CVTSS2SD` 997 → 733; repack loop 2.6× faster                  |

Two independent hosts three generations apart agreed on the codelet round's
geomean to within 0.3% _and_ on which sizes did not move — the signature of a
code change rather than a measurement artifact.

**Accuracy cost: 3–9% more relative-L2 error, i.e. sub-ulp.** Everything stays
at ~10⁻⁷ (float32 ε is 1.19e-07) and the peak-normalized error is unchanged or
lower. complex128 is bit-identical.

Three related fixes fell out:

- **`cmd/measure_correctness` was reporting a misleading number** and nearly
  caused the work to be reverted: it maxed a _per-bin_ relative error (an
  extreme-value statistic over an unstable quantity) against a **complex64**
  reference. It now reports relative L2 against `reference.NaiveDFTWide`
  (float64 output for float32 input, with `NaiveDFT` a narrowing wrapper so
  ~131 existing call sites see bit-identical values), mean and max over trials,
  plus a peak-normalized column — because relL2 attenuates a single wrong bin
  by ~1/√n, which is exactly the failure a broken codelet produces. Validated
  against an independently written harness to within 1.6%. It also prints the
  build configuration now, which is what makes its numbers comparable at all.
  _Recorded, not fixed:_ its complex128 column measures the **reference**, not
  the FFT — `NaiveDFT128` builds twiddles from un-reduced angles, so its error
  grows as O(n).
- **One codelet was over the inliner's big-function threshold**, paying a real
  `CALL` per complex multiply: `inverseDIT64Radix2Complex64` held **all 193**
  un-inlined `MulComplex64` calls in the module. Cause: 64 of them were not
  complex multiplies at all but a real 1/n scale written as
  `complex(scale, 0)`. Fixing that (bit-identically) restored inlining and gave
  −21.4%/−27.5% at the codelet level.
- **39 more real-factor multiplies** in codelets that _are_ selected, all
  rewritten component-wise. The win is confined to small compute-bound kernels
  — the two fully-unrolled radix-16 codelets gained 11–20% while split-radix
  and the 384 codelet were flat, because their scaling is a memory-bound sweep
  and a rounding error respectively.

**Standing rules from this round** are in §2.4.

### 1.7 The mixed-radix engine (2026-07)

Started the cycle at 0.20× FFTW3 for 5-smooth and 0.13× for 7/11-smooth
lengths — worse than Rader, which is the newer algorithm.

- **44100 (the canonical audio rate) went from losing to gonum to beating it.**
  Measuring both routes first, as the item required, exposed a **driver
  defect** rather than a routing one: the AVX2 mixed-radix drivers guarded
  codelet dispatch with `n > 1` while the scheduler requires `n > 5`, so every
  schedule ending in radix 2/3/4/5 sent each leaf through a full codelet call —
  strided twiddle gather, two `sync.Pool` round-trips and a `defer` — for a
  handful of butterflies. At n = 4900 that is 1225 dispatches per transform,
  and a profile put **32% of complex64 time inside
  `ForwardAVX2Size4Radix4Complex64Asm`**. Fixed by using the scheduler's own
  bound. Mixed-radix got 18–58% faster at every length measured. With the
  pathology gone the win gate collapsed to one criterion (mixed-radix wins
  where Bluestein's pad is ≥ ~2.5n), so the `pow2 ∈ {2,4} → lose` branch was
  simply deleted. Plan-level at 44100: c128 3.40 → 1.89 ms. One documented
  cost: 22050 c64 inverse regresses ~11% while c128 gains 24–36%; the gate sees
  neither precision nor direction and carving out one size is the overfitting
  that produced the rule being replaced.
- **Dispatch overhead removed from the leaves.** A profile at n = 1000 put only
  **1.9%** of runtime in the codelet assembly and ~40% in dispatch:
  `cpu.DetectFeatures` took two locks per call, `registry.Lookup` an RWMutex,
  and each leaf gathered a twiddle table into a pooled buffer the codelet then
  discarded. Features are now cached in an `atomic.Pointer`, the registry size
  map is copy-on-write behind one (also making the returned `*CodeletEntry`
  stable), and the leaf gather is gone. Geomean −15.0%. Then the leaf codelet
  resolution itself was hoisted out of the recursion — it can only ever match
  at a leaf, and every leaf of one transform has the same size — taking n=1000
  from 156 lookups per transform to 1, geomean −1.9%.
- **Fused AVX2 stage kernels for radix 3, 5, 7 and 11.** One pass: rows 1..r−1
  are multiplied by the stage table and stay in registers through the
  butterfly, never written back. The k index is the vector axis, so the
  butterfly needs no cross-lane movement; direction costs exactly one register
  (the XOR mask). Radix 3/5 gave geomean **−30%**; radix 7 gave **−30.7%** over
  the lengths it reaches (44100 −38…−42%, 2205 −20…−22%); radix 11 gave the
  largest single win of the round — n = 704 dropped **10244 → 2609 ns**
  (c64, −74.5%). Radix 11 no longer fits the register file, so its ten
  butterfly constants sit pre-broadcast in RODATA and are read as FMA memory
  operands, and five row-stride registers reach all ten rows via the SIB scale.
  Every kernel was verified against the definition-based stage reference at
  spans spanning the vector body and the Go tail, in both precisions and
  directions, and with `dst == input` aliasing.
- **The two-pass vectorized stage earns nothing on amd64** and was kept anyway.
  Re-derived on top of the hoist: `off` vs `full` is +50.9% but `nofused` vs
  `full` is +49.7% — the fused kernels carry all of it, and the two-pass form's
  −1.5% does not clear the run's own ±2% noise floor (n = 704 is a built-in
  null control, having no fused stage). Forced to the SSE2 tier it is a net
  **loss**. Retained because it is the only vectorized stage on NEON/WASM/
  purego, tiers this machine cannot measure.
- **`mixedRadixStageMinMuls` stays at 64.** The hypothesis that a kernel paying
  none of the two-pass form's fixed costs would break even sooner is
  **refuted**: 32 is +8.9%, 48 +1.5%, 128 +2.7%. The fused kernel did not shed
  the fixed cost so much as replace it — its prologue broadcasts up to six
  constants and derives up to six row offsets before the vector loop starts.
  Each arm regressed exactly the lengths whose stage set it changes, which is
  what makes that a result rather than noise.
- **Practical DSP lengths are in the internal benchmark set** (1000, 2205,
  3600, 12000, 44100, forward and inverse, both precisions). The lengths where
  the lead over gonum nearly vanished were exactly the ones a DSP user picks,
  and none of them were benchmarked in-tree — which is why the internal numbers
  looked healthy throughout.

44100 is now 786 µs against FFTW3's 185 µs (§1.15 re-measurement; the 236 µs
recorded here earlier was FFTW measured in a slower window). The gap has closed
from 8.0× to 4.3×.

### 1.8 Power-of-two soft spots (2026-07)

- **The n = 64 cliff** (0.36× FFTW3 against 0.97× at n = 32) was the codelet,
  not the decomposition, and no new codelet was needed. `dit64_radix4_avx2` was
  AVX2 in name only — every load and store was `VMOVSD`. `dit64_radix2_avx2`
  already was a full-width kernel but sat behind it because its _inverse_
  measured 179 ns against the forward's 63 ns. The cause was **one legacy-SSE
  instruction**: `MOVD AX, X8` in the 1/n prologue, whose AVX↔SSE transition
  penalty cost ~100 ns — three times the rest of the kernel. Ruled out first,
  each by measurement rather than argument: the FMA instruction pair, the input
  data, the memory layout and code placement. End to end: **348–408 → 147–170
  ns** forward, ~2.3–2.4× both directions.
- **Legacy-SSE encodings swept out of the AVX2/AVX-512 tree**: 4089
  instructions converted to VEX across 59 files, 10 functions deliberately left
  alone. **End-to-end performance-neutral** — the hypothesis that it would fix
  the 1024–16384 soft spots was wrong — but it uncovered six codelets carrying
  that same fixed ~100 ns prologue cost regardless of transform size
  (`dit4_radix4_avx2` 102.9 → 5.0 ns). Method worth rebuilding if revisited: a
  liveness fixpoint decides safety (a VEX write zeroes bits [255:128] where the
  legacy form preserves them), and verification is at the machine-code level —
  binutils `objdump`, normalized and diffed per symbol; all 457 asm symbols
  decoded identically. That caught two bugs the test suite passed straight
  over. See §2.4 for the all-or-nothing rule.
- **Priority retunes after the sweep.** The six mis-ranked codelets at n =
  8/16/32 were mis-ranked _because_ of the penalty they carried; re-measured
  idle and pinned (median of 41 interleaved rounds), the radix-2 AVX2 codelets
  win by 19–33%.
- **`RankLevel` splits eligibility from ordering.** `Register` sorts SIMD-level
  major, so a hand-tuned decision recorded in `specs.go` — _"SSE-width in
  practice; loses to dit64_radix4_sse2 → stay below it"_ — had been inert since
  it was made. `SIMDLevel` stays the eligibility gate; the new optional
  `RankLevel` sets the level used for ordering only. complex128 n = 64 now
  binds the SSE2 codelet, ~24% faster end to end. **Use it to demote, not
  promote** — promoting a narrow codelet also moves it ahead of its own tier's
  siblings on CPUs that have nothing better.
- **Wisdom now outranks the registry** (file format v2 → v3; v2 files are
  rejected, not reinterpreted). It previously could never override a
  codelet-covered size — i.e. every power of two from 4 to 4096 — which made
  `LookupBySignature`'s disabled-codelet guard unreachable for exactly the
  sizes it was written for. The simple ordering only became correct once
  measure mode started timing codelets, below.
- **`PlannerMeasure` could pick a worse plan than `PlannerEstimate`.** It
  benchmarked kernel _strategies_ only; codelets were never a candidate. At c64
  n = 1024 that returned `stockham` where estimate returns
  `dit1024_radix4_avx2` — measuring made the plan worse. Codelets are now
  candidates in their own right, with the depth following the existing mode
  hierarchy. Side fix: `estimateWithStrategy` built its `PlanEstimate` without
  `TwiddleSize`/`PrepareTwiddle`, so a codelet wanting a packed layout got the
  plain table, failed its own length check and returned false — the plan
  silently ran the fallback kernel while still reporting the codelet signature.
- **Incumbent audit at n = 256/512/8192**, canary-gated (85 accepted groups
  against 11 rejected): all six incumbents confirmed in both directions. The
  outlier priorities 135/130/120 encoded the right _order_ and were normalized
  to 35/30/25. Two findings the priority mechanism cannot express are recorded
  rather than acted on: at 8192 c64 the `params` incumbent and the plain
  codelet are within ±1% and **swap by direction** (so the custom twiddle
  layout costs plan memory for no gain — a simplification candidate), and at
  8192 c128 the SSE2 codelet **matches** the AVX2 one.

### 1.9 The 256-bit radix-4 kernels (2026-07-28)

**The whole AVX2 radix-4 codelet family was XMM-width, not 256-bit.** Every
`avx2_f32_size*_radix4*.s` loaded operands with `VMOVSD` — one complex64 — and
did all butterfly arithmetic in `X` registers; `Y` registers appeared only in
the trailing copy and 1/n loops. These were scalar radix-4 kernels in VEX
clothing: they got the three-operand form and freedom from the transition
penalty, but none of the width.

Replaced by **two size-generic kernels**, `internal/asm/amd64/avx2_f32_radix4.s`
and `avx2_f64_radix4.s`, rather than twenty more hand-rolled files:

| n     | c64 before → after     | c128 before → after  |
| ----- | ---------------------- | -------------------- |
| 16    | —                      | 21 → 14 (1.5×)       |
| 64    | —                      | 156 → 55 (2.8×)      |
| 128   | 320 → 88 (3.6×)        | 367 → 129 (2.8×)     |
| 256   | 426 → 199 (2.1×)       | 644 → 262 (2.5×)     |
| 512   | 797 → 430 (1.9×)       | 1478 → 607 (2.4×)    |
| 1024  | 3200 → 918 (3.5×)      | 3698 → 1324 (2.8×)   |
| 4096  | 16848 → 4320 (3.9×)    | 20234 → 9165 (2.2×)  |
| 16384 | 73712 → 23599 (3.1×)   | 98044 → 39375 (2.5×) |
| 65536 | 519000 → 130594 (4.0×) | — → 334106           |

(forward, ns, best-of-5 pinned; the ranking tests re-derive this ordering from
measurement so the `Priority` values cannot silently rot.)

Design notes, since they are what made one kernel cover every size:

- The twiddle for butterfly `j` depends only on `j`, not on the group, so each
  stage needs `3*m` twiddles held as three contiguous planes. Every twiddle
  load is then a plain 256-bit read and the per-butterfly index arithmetic
  disappears.
- `n = 2*4^k` needs no separate kernel: running the radix-4 stages only to `n/2`
  transforms the even- and odd-indexed halves independently, and one radix-2
  tail combines them.
- The permutation table stores only `p[4g]`, as `int32` — 16 KiB at n = 16384
  against the 128 KiB `DATA` blob the old kernel embedded. It is taken from
  `internal/math` rather than rederived: a self-derived permutation table is
  the one bug class that has actually escaped review here. It is also
  precision-independent, so both kernels share `radix4GroupIndices`.
- The ±i rotation is `permute + xor` (2 ops) instead of
  `permute + xor-zero + sub + blend` (4). Forward and inverse differ only in
  which mask feeds which output; the inverse 1/n is exact and folds into
  stage 1.
- **Permutation fused into stage 1.** At n = 16384 the permutation pass alone
  had been a third of the kernel while doing no arithmetic. On the complex64
  side `VPGATHERDQ` delivers a0..a3 already separated, removing a full
  store-then-load _and_ the input transpose. There is no 128-bit-element
  gather, so the complex128 kernel builds its groups with
  `VMOVUPD` + `VINSERTF128` instead — the fusion is what mattered, not the
  gather. Net at n = 16384: 29.1 → 23.6 µs.
- **Twiddle broadcasts belong on the load ports, not port 5.** The inner loop
  broadcast each twiddle's real and imaginary part with the _register_ form of
  `VMOVSLDUP`/`VMOVSHDUP` (f32) or `VMOVDDUP`/`VPERMILPD` (f64) — six port-5
  shuffles per iteration, on the one port the loop is bound by. The **memory**
  forms are pure load uops for a re-broadcast scalar. For f64 the imaginary
  broadcast needs no instruction at all: offsetting the address by 8 bytes
  makes `VMOVDDUP` duplicate the high float64 instead. That reads 8 bytes past
  the last plane, which the `n+4` twiddle padding covers — and the kernel's
  length check _enforces_ rather than assumes. Plus: both ±i rotations permute
  the same `t3`, so permute once and branch with two `VXOR`s. Port-5 traffic
  11 → 4 per iteration; c64 −4…−13% at every size, c128 20–24% at 256–16384.
- **A dedicated `.s` file per size would buy very little.** `stage2Generic`
  (m = 4, 1024 group iterations) and `stage7YMM` (m = 4096, one group) cost the
  same 2.5–2.6 µs for the same 4096 butterflies, so the loop structure costs
  essentially nothing and constant-folding the bounds has nothing to reclaim.

Superseded kernels are **removed**, not left registered alongside: thirteen
`.s` files deleted across the two precisions, together with the 8192 "params"
twiddle layout. Shared `bitrev*` tables moved to
`internal/asm/amd64/bitrev_radix4_tables.s` where other-precision kernels still
reference them by symbol. Sizes that stay do so because something other than
the registry calls them — the six-step row FFTs, the size-384 decomposition,
and the `KernelStrategy` dispatch in
`internal/fft/kernels_amd64_size_specific.go`, which selects by strategy rather
than through the registry and so has no way to obtain a prepared twiddle table
(§4).

**The port-5 pattern does not transfer to the fused mixed-radix stages.** Tried
and reverted 2026-07-28. Those kernels run the same three-shuffle
complex-multiply idiom per row, so they looked like the same opportunity, but
the dup source is the _data_, not a scalar twiddle: one input vector feeds both
duplicates, so a memory-operand form does not replace a load, it adds one — and
`VMOVSLDUP ymm, m256` is not a load-only uop the way a 64-bit broadcast is, so
nothing moves off port 5 either. Measured: 2–28% _slower_, all 32 cases
regressing. Port 5 is not the bottleneck there anyway — a probe that deleted
the ten table swaps from the radix-11 stage outright (wrong results, right
instruction mix) moved the time by 0–7% with no consistent sign. The radix-4
kernel is shuffle-bound because it retires many butterflies per load out of a
small working set; the stages are streaming kernels at 2 reads + 1 write per
row. **Do not retry it on a kernel whose broadcast operand comes from the data
stream.**

### 1.10 Plans report the route they take (2026-07-28)

Non-power-of-two lengths named a strategy that never executed. The kernel
dispatch (`autoKernelComplex64/128` in `internal/fft/kernels_fallback.go`)
checks the length _before_ the strategy switch and takes the mixed-radix engine
unconditionally, while `plan.algorithm`/`KernelStrategy()` carried whatever the
power-of-two heuristic would have said: 1000 reported `dit_fallback`,
2205/3600/12000/44100 reported `stockham`, and 810000/4410000 reported
`sixstep`/`eightstep`. None of those ran.

Fixed by naming the route: `KernelMixedRadix` (`mixedradix`) is a first-class
strategy, resolved in `resolveKernelStrategy` before anything else for
non-power-of-two lengths `MixedRadixEligible` accepts. The rule is that the
reported strategy always equals the executed one, which had four consequences
beyond the label:

- **Forced strategies are resolved, not echoed.** `EstimatePlan` and
  `estimateWithStrategy` took the forced value verbatim while the dispatch
  applied its own guards, so a forced six-step on a non-square already reported
  a route it did not run. Both now go through
  `ResolveKernelStrategyWithDefault`. Forced `KernelBluestein` at a smooth
  length still wins — it genuinely replaces the engine — and a forced
  `KernelMixedRadix` at a power of two falls back to the size heuristic rather
  than mislabeling in the other direction.
- **The non-power-of-two square rule was deleted.** Its own comment gave it
  away: "they execute through the mixed-radix engine". Every square it could
  reach is either engine-executable (answered earlier now) or Bluestein-bound
  (answered before the heuristic is consulted), so it was a label with no route
  attached.
- **Measurement stopped comparing a length against itself.**
  `selectStrategiesToTest` handed the power-of-two strategy list to smooth
  lengths, timing one mixed-radix transform up to five times and recording the
  fastest _noise sample_ under a name that never runs. It now returns the
  single real candidate, matching how Bluestein lengths were already handled.
  `cmd/benchkernels` labels by `plan.KernelStrategy()` and skips duplicates for
  the same reason. Stale wisdom naming an old label resolves to the route.
- **Packed Stockham tables are no longer built for these plans.**
  `newKernelExecutor` allocated them whenever the estimate said Stockham; the
  packed kernel rejects non-powers of two at call time, so they were dead
  weight (not a wrong answer).

`n = 384` is unaffected and was already honest: it binds the
`dit384_mixed_avx2` codelet, which really does run (see §3 for its three
performance problems).

---

### 1.11 Test-vector blindness audit (2026-07-29)

An impulse (`x[0] = 1`) is the most common FFT test vector and one of the
weakest: its spectrum is all-ones, so every twiddle multiplies a zero and every
permutation of the output is still all-ones. Wrong twiddles and wrong bin
ordering — the two ways this library has actually been wrong — both pass, and so
do Parseval and linearity, because linearity holds for any linear operator
including the wrong one. That combination hid a wrong-spectrum bug in the
recursive `complex128` path at every size ≥ 1024.

The suite was audited for that pattern. Four tests were checking nothing, and
one whole precision was uncovered above `n = 16`:

- **The recursive path was still impulse-only.**
  `TestRecursiveFFTCorrectness` (512…16384) and `TestRecursiveFFTSmallSizes`
  drive an impulse; `TestRecursiveFFTComplex128` uses a ramp at one size. New
  `internal/transform/recursive_broadband_test.go` adds the companions: a
  broadband signal compared bin-by-bin against `internal/reference` at
  512…4096, and — for 8192/16384, where the O(n²) reference costs 1.1 s and
  4.3 s — a sum of complex exponentials whose spectrum is exact in closed form,
  which costs O(n) and still catches both blind classes (a wrong twiddle leaks
  into the zero bins, a wrong ordering moves the spikes). Both precisions.
- **`complex128` plans had no broadband reference check above n = 16.**
  `TestPlannerModesMatchReference` sweeps `complex64` to 16384, but the wider
  precision was covered only at `n ≤ 16` and at non-power-of-two lengths — the
  gap the recursive bug lived in. New `plan_broadband_reference_test.go` sweeps
  every forceable strategy (auto, DIT, Stockham, split-radix, recursive,
  six-step, four-step, Bluestein) at 256/1024/4096 against the reference,
  forward and inverse, both precisions. 72 subtests.
- **`TestBluesteinHelper` asserted only "output is not all zeros".** No wrong
  filter, twiddle table or bin order could fail it. It now runs the full
  Bluestein assembly (pre-chirp, cyclic convolution, post-chirp — the chirp
  multiplies live in `internal/fft`, so the test does them inline) and compares
  against the naive DFT at n = 3/5/12.
- **The mixed-radix engine was tested at one size with one ramp.**
  `internal/fft/mixedradix_test.go` now sweeps 15 lengths covering every radix
  the scheduler emits (2/3/4/5/7/8/11) with a broadband signal, both
  precisions, forward against the reference and round-trip.
- **`testPrecisionComparison` measured precision on two bins.** Its input was a
  real sine at bin 5 — a lattice frequency, so the spectrum is two nonzero bins
  and n−2 zeros, and the relative-difference loop skipped everything under
  1e-10. Now broadband, bounded relative to the spectrum peak.
- **`testLargePrecision` measured precision on an impulse** at 65536…262144,
  where "max error" was structurally 0. Now the closed-form multi-tone
  spectrum; the tolerance tightened from 1e-11·n to 5e-14·n and still holds
  with ~40× headroom.
- Smaller: the four-step fallback test, `TestClone_Forward`,
  `TestClone_InPlaceRoundTrip`, `TestPlan_Clone*` all asserted properties an
  impulse cannot distinguish (a clone that lost its parent's prepared twiddle
  layout still returns all-ones), and now use broadband inputs.

Two constructions matter for reuse. The multi-tone reference must reduce
`k·j mod n` before scaling to radians — at n = 262144 the unreduced angle
reaches ~1.6e6 rad, where `cmplx.Exp`'s argument reduction alone costs ~1e-10 of
phase and the "exact" expected spectrum stops being exact (this is what made
the first tightened tolerance fail). And `complex64` results are compared
against `reference.NaiveDFTWide`, which accumulates in float64, so the
comparison is not limited by the reference's own rounding.

The new tests were mutation-checked by perturbing the shared twiddle table
(`internal/math.ComputeTwiddleFactors`) by one part in 1e4. Every broadband and
multi-tone test that routes through that table fails; `TestForward_Impulse_2048`
passes, which is the point. The clone tests at n = 128/256 also pass under that
mutation, because those plans bind a codelet carrying its own tables — the
mutation does not reach them, so their coverage rests on the reference
comparison rather than on this check.

One byproduct: the sweep first failed only in a full-package run and passed
under `-run`. `internal/fft/measure_codelet_test.go` registers **identity** stub
codelets at n = 1155 and n = 1331, and the codelet registry is process-global and
append-only, so every later transform at those lengths copies its input to its
output and reports success. Any correctness test that picks one of those lengths
is silently vacuous. The sweep avoids them and both files now say why.

Clean by construction and left alone: `internal/kernels/codelet_reference_all_test.go`
(impulse at a nonzero position, whose spectrum is a full twiddle sequence, plus
tones and a random-input naive check to 2048), the Rader tests (random input,
bin-by-bin against the reference), plan-level Bluestein (`i² + i·j`) and
`TestPrecisionRoundTripSweep` (random). The remaining impulse uses are in tests
of plan lifecycle, pooling and allocation counts, where the input is irrelevant.

### 1.12 The size-384 path (2026-07-29)

384 was the worst ns/point in the registry, and the headline symptom was that
c64 forward was _slower_ than c128 at the same size — impossible for a
half-width type unless the c64 path is doing something the c128 path is not. It
was: the complex64 codelet did its radix-3 column DFT and twiddle multiply in
scalar Go, through `mathpkg.MulComplex64` and therefore through the widening
defect of §1.6, while the complex128 twin called assembly for both. The AVX2
assembly for the c64 side had been written, assembled and declared, and never
called.

**The assembly was also wrong.** `ApplyTwiddle384Complex64Asm` builds the
strided `twiddle[2k]` vector for the third sub-array, and it did so with
`VINSERTPS $0x10`, which moves a single float32: it overwrote the imaginary part
of each even twiddle and left two lanes undefined. The correct instruction is
`VMOVLHPS`, which moves a whole 64-bit lane. All 128 elements of that sub-array
were wrong; the other 256 were fine. A direct test of the helper against the
scalar loop it replaces catches it immediately (`dit_384_asm_helpers_amd64_test.go`),
and that test now exists for all four twiddle helpers and both radix-3
directions.

The lesson generalises past this file: **declared-but-uncalled assembly is
untested assembly**, and nothing in the suite reports it, because the
registry-driven reference tests only reach a function once something calls it.
There was a signal and it was not followed: an earlier round FMA-fused this file
and measured no benchmark change at all. "An optimisation that changes nothing"
should be read as "this code does not run" before it is read as "the optimisation
did not help". Worth a sweep for other `.s` symbols whose only reference is
`decl.go`.

What changed:

- The c64 codelet now calls `Radix3Butterflies384{Forward,Inverse}Complex64Asm`
  and the twiddle helpers, mirroring the c128 body.
- Four `{Forward,Inverse}AVX2Size384Mixed{Complex64,Complex128}Asm` symbols were
  deleted. They looked like kernels and were named like kernels; their bodies
  only length-checked the arguments and returned true. Nothing called them.
- New `ApplyConjTwiddle384Complex{64,128}Asm` for the inverse direction.
  Conjugating the twiddle costs one `VXORPS`/`VXORPD` against a sign mask;
  conjugating the product does not work, because `VFMADDSUB`'s fixed
  even-subtract/odd-add pattern gives the wrong sign on the imaginary term.
- The c64 path precomputes the 128-point sub-twiddle at package load
  (`W_128^k == W_384^(3k)`, so it is exactly the stride-3 gather it replaces) and
  pools its two buffers, as the c128 path already did. Two `make` calls per
  transform are gone from each direction.
- The `MOVL`/`VMOVQ`/`VBROADCASTSS` constant prologues in the radix-3 functions
  became `VBROADCASTSS ·const(SB)` per §2.4.

Measured (min of 6, `old` and `new` test binaries interleaved in one session —
see below for why that matters), ns/op:

| codelet          | before | after | change |
| ---------------- | -----: | ----: | -----: |
| c64 384 forward  |   1865 |  1310 |   −30% |
| c64 384 inverse  |   2016 |  1399 |   −31% |
| c128 384 forward |   1785 |  2036 |    n/r |
| c128 384 inverse |   2444 |  2334 |    n/r |

The c128 forward path is **unchanged code**, so its 14% apparent regression is
pure measurement noise, and that is the point: it calibrates the run. Better
still, size 384 registers the same function twice, under the `generic` and
`avx2` signatures, which gives a free identical-code control pair inside every
run — they differed by 430 ns in the baseline binary and 258 ns in the new one.
So the noise floor here is ~15%, the c64 results sit far outside it, and the
c128 conjugate-twiddle change cannot be resolved either way on this machine.
**Two identical benchmarks are worth more than a repeat count**; where a registry
already provides such a pair, read it before trusting any single delta. Serial
before/after runs on this laptop drifted by more than the effect being chased
(§2.3); only the interleaved two-binary A/B was stable.

Not fixed, moved to §4: the c128 sub-FFT still runs radix-2, because the AVX2
c128 radix-4-then-2 that §3 assumed existed does not.

### 1.13 The Bluestein sub-FFT never reached the registry (2026-07-29)

The default build was ~4% _slower_ than `-tags purego` at n = 1009 and n = 2003
— the wrong direction for §2.1 rule 4, and recorded in §3 as wanting a profile
on the theory that a codelet had been selected whose fixed overhead the pad size
could not amortize.

**No codelet had been selected.** `internal/fft/bluestein.go` splits on
`IsPowerOf2(m)`: a non-power-of-two pad runs the mixed-radix engine, which is
fully SIMD-dispatched, while a power-of-two pad called
`kernels.BluesteinConvolution` → `bluesteinSubForward` → the hardcoded size
switch in `internal/kernels/dit.go`, which never consults the codelet registry
and has no build tags. So at those two lengths **both builds ran the identical
pure-Go kernel for ~96% of the work**, and the measured deficit was the chirp
modulation's SIMD call overhead sitting on a path that got no benefit from it.
`dit2048_radix4_avx2`, `dit4096_radix4_avx2` and their complex128 twins were
registered and unreachable. n = 9973 pads to 24576 — not a power of two — which
is the only reason it went the right way.

That the split was on pad _shape_ rather than pad _size_ is what hid it: the two
regressing lengths were not a size class, they were the branch that had no
dispatch.

Fixed by binding the padded sub-FFT at plan time (`newBluesteinSubFFT` in
`plan.go`, `fft.BluesteinSubFFT`), reusing the machinery `kernelExecutor`
already uses: `planner.EstimatePlan` for the padded size, then
`prepareCodeletTwiddles` — which is the part a call-time fix could not have
done, since a codelet may need a prepared twiddle layout and preparing one is
plan-time work. The filter pointwise multiply moved to the SIMD
`ComplexMulArrayInPlace` the mixed-radix branch already used.

Measured on the i7-1255U, three interleaved rounds of two test binaries, medians
in ns/op (n = 9973 is the null control — it keeps the unbound route):

| cell              | before |  after | change |
| ----------------- | -----: | -----: | -----: |
| c64 1009 forward  |  32763 |   5951 |   5.5× |
| c64 1009 inverse  |  32344 |   6401 |   5.1× |
| c128 1009 forward |  32065 |  10885 |   2.9× |
| c64 2003 forward  |  90122 |  12481 |   7.2× |
| c64 2003 inverse  |  89702 |  12837 |   7.0× |
| c128 2003 forward | 103164 |  24916 |   4.1× |
| c64 9973 fwd/inv  | 224310 | 224168 |   1.00 |

The SIMD/purego ratio at 1009 went 0.96 → 5.9 and at 2003 0.96 → 7.6. That the
_baseline_ ratio was ~1.0 rather than the 0.86 originally recorded is itself the
confirmation: two builds measuring the same because they were running the same
code.

Two things worth keeping:

- **Bind only where the registry has a codelet.** The first version fell back to
  the strategy-dispatched kernels when no codelet existed, and that cost ~4% on
  purego at n = 1009 — the unbound size switch is hand-tuned per size, while
  `EstimatePlan`'s heuristic answers Stockham for everything above
  `ditAutoThreshold`, so a codelet-less pad of 2048 traded a tuned
  radix-4-then-2 DIT for plain Stockham. With the binding restricted to a real
  registry hit, purego is flat at 1009 and **15–18% faster** at 2003, where the
  generic codelet beats the switch.
- **A bail is a contract violation here, not a fallback.** The caller passes
  `dst` aliasing `x`, so a kernel that bailed part-way has already destroyed the
  input needed to re-run it. Plan construction verifies the binding end-to-end
  (`fft.VerifyBluesteinSub`) and degrades to the unbound route _there_; the
  transform-time path panics, like `mustMixedRadix`.

_Not done:_ Rader still passes `nil` — its length-(n−1) convolution takes the
same unbound route when n−1 is a power of two (5-smooth but non-power-of-two
n−1 already runs the SIMD mixed-radix engine). Same fix, separate measurement.

### 1.14 Packed Stockham was compiled out of SIMD builds (2026-07-29)

`internal/transform/stockham_packed_toggle_simd.go` set
`stockhamPackedEnabled = false` for amd64/arm64/386, on the stated grounds that
"the hand-written codelet path is checked first and supersedes it".

**That rationale was false as written, and the item's proposed alternative fix
was a no-op.** Every registered codelet carries `Algorithm: KernelDIT`, and
`tryRegistry` returns an estimate with `Strategy: entry.Algorithm`, so for an
auto plan the `estimate.Strategy == KernelStockham` test in `newKernelExecutor`
is already false wherever a codelet binds. Codelet and packed were never in
competition, so reordering the two executor branches — the fix §3 offered —
would have changed nothing. What the constant actually suppressed was the sizes
with _no_ codelet, where the SIMD build fell through to a radix-2 Stockham
kernel while the radix-4 route it had disabled was up to 2.7× faster.

Two further corrections that shaped the fix:

- **The boundary is not the codelet ceiling.** The toggle also suppressed
  explicitly forced `KernelStockham` at _any_ size ≥ 4, because `tryRegistry`
  returns nil when the forced strategy does not match the entry's algorithm. A
  plain size threshold covers auto, wisdom and forced plans alike.
- **purego cannot be detected from CPU features.**
  `internal/cpu/detect_amd64.go` is tagged `//go:build amd64` with no
  `!purego`, so a purego amd64 build still reports `HasAVX2: true`. The
  build-tagged constant survives as a _tier selector_
  (`packedBuildHasSIMDKernels`), not an on/off switch.

Replaced by a runtime policy (`internal/transform/stockham_packed_policy.go`):
a tier × precision threshold table, plus `SetPackedStockhamOverride` so both
arms of the measurement live in one binary rather than in two builds (§2.2).
The old `stockhamPacked` guard is gone — the engine is now always compiled in
and always runs when called, and the _route_ decision moved to plan
construction.

Measured on the i7-1255U, pinned, forced Stockham, 5 interleaved rounds, as the
median of the **within-round** packed/kernel ratio. That statistic is not
optional here: the null control's round-to-round spread reached **1.69**, so a
cross-round median would have been measuring the machine. Ratios < 1 favor
packed:

|          |  2^16 |  2^17 |  2^18 |  2^19 |  2^20 |  2^21 |
| -------- | ----: | ----: | ----: | ----: | ----: | ----: |
| c64 fwd  | 1.625 | 1.518 | 1.175 | 0.934 | 0.672 | 0.515 |
| c64 inv  | 1.647 | 1.513 | 1.176 | 0.976 | 0.729 | 0.514 |
| c128 fwd | 0.972 | 0.803 | 0.626 | 0.565 | 0.437 | 0.374 |
| c128 inv | 1.009 | 0.831 | 0.625 | 0.514 | 0.481 | 0.397 |

Thresholds: **complex128 from 2^17** (1.25×/1.20×), **complex64 from 2^20**
(1.49×/1.37×). The precision axis is what the data demands — c128 wins at 2^17
where c64 still _loses_ by 1.5×, so the `padShapes` convention of admitting a
shape only where it wins at both precisions would have forfeited most of the
benefit.

**2^19 complex64 is deliberately given up.** Its first five rounds averaged
0.900/0.950, nominally clearing the 0.95 bar. Ten further rounds put the ratios
at 0.474–1.245 and 0.429–1.220, medians 0.934/0.976 — a wash. It is the one
size where the table concedes something real, and it concedes it because the
measurement does not support the claim.

Only the AVX2 row is filled in. SSE/AVX-512/NEON stay off: §2.1 rule 5 forbids
landing a route unmeasured on its own tier, and of the four only AVX2 is
measurable here (§2.3). Their uncovered range is one octave wider, since their
codelet ladder stops at 32768 — so that is a real follow-up, in §6.

Side effects worth noting:

- `internal/fft/recursive_test.go` had two `t.Skip`s keyed on the old toggle,
  so its packed-Stockham correctness tests **never ran on a SIMD build**. They
  now do.
- `ComputePackedTwiddles` grew `Values` from a zero-capacity slice, ending at up
  to ~2× the used capacity. It now preallocates exactly, via the new
  `PackedTwiddleLen` (also the closed form documenting the cost: `n-1` values
  for `n = 4^k`, `n/2-1` for `n = 2·4^k`). The table is roughly a second full
  twiddle table — 8 MiB at the c64 threshold, 2 MiB at the c128 one, and 64 MiB
  at 2^22 complex128. That cost is why the thresholds are set from measured
  wins rather than from "wherever it is merely correct".

### 1.15 The external comparison had drifted by a factor of two (2026-07-29)

§4 asked for a re-measurement of the plan-level c64/c128 ratio and the FFTW
comparison. Both had moved, but the headline in §1 had moved further than the
item did: it read **0.63× FFTW3** at power-of-two lengths, and the measured
value is **1.36×**. The library had overtaken FFTW3 there at some point during
the 2026-07 rounds and nothing in this file said so.

The cause is structural rather than an oversight. The cross-library sweep lives
in a **different repository** (`go-fft-bench`), which pins algo-fft to a
released tag; each sweep is committed there against its tag, and nothing pulls
the result back here. So §1's number aged against a v0.7.0 baseline through
four releases while every §1.x round below it was measured and current. **A
number that lives in another repository is not maintained by editing this
one** — quote it with the tag it came from, or re-measure it.

The round itself: tag v0.7.4, sweep, compare. The v0.7.4-vs-v0.7.3 delta is
**flat** at power-of-two lengths in both precisions (medians 0.88–1.07, 6
rounds, order rotated, two pinned test binaries), which is correct — the nine
commits touch the 384 path, the forced six-step route, Bluestein pads and
packed Stockham above 2^17, and none of those is the default power-of-two route
in the benchmarked band. The numbers in §4 are a correction of the record.

Two things about measurement came out of it, and they cost the first sweep:

- **The drift control did its job, and the sweep had to be discarded.**
  `go-fft-bench` requires that the unchanged libraries be checked as a control
  before any delta is quoted. They came out **+28…+36%** by geomean — FFTW,
  gonum, go-dsp and takatoh alike — so the run had measured the machine, and
  every algo-fft delta in it was wrong by about that much. Nothing from it was
  used. Worth noting what it looked like from inside: the complex64 arm showed
  a plausible-looking regression (n = 1024 at 1591 ns against 896, complex64
  _slower_ than complex128 — the exact signature of §1.6) which a two-binary
  A/B then showed to be entirely absent. **A contaminated run does not merely
  add noise; it manufactures findings that look like the ones you already know
  how to explain.**
- **`~/.local/bin/go` on this laptop is a wrapper** that runs the toolchain as
  `nice -n 10 taskset -c 0-$(nproc-2)`. Every benchmark ever run here through
  `go test` has therefore been _de-prioritised below the desktop_, which is why
  contention does not merely add a few percent but collapses whole runs
  whenever a browser or Teams is busy. Invoking `/usr/local/go/bin/go` directly
  under `taskset -c 0` reproduced the committed v0.7.3 sweep to within 2–12% on
  a machine whose load average was 4.4 — i.e. the pinned, un-niced run was
  clean while the ordinary one was not. The wrapper is a sensible default for
  compiling; it is not one for measuring. (It also explains the "0--1" taskset
  error if you wrap it: `nproc` reports 1 inside the pin.)

The ratio measurement needed one further change of statistic. `FFT` and `FFT32`
are separate top-level benchmarks, so a c64/c128 ratio taken from one sweep
compares arms measured minutes apart — §2.2's warning, and visible in the
rejected run as IFFT +8% against IFFT32 +62%. Taking it as the **median of
within-round ratios** over interleaved rounds fixes it, and is the same
statistic §1.14 needed for the same reason.

### 1.16 The last ten mixed VEX/SSE functions (2026-07-29)

The §1.8 sweep converted 4089 legacy-SSE encodings to VEX and left ten functions
alone — `Forward`/`InverseAVX2Complex64Asm`, both `AVX2Stockham` pairs and the
`Size1024Radix32x32` pair in both precisions — recording that "each has a legacy
write whose upper half is live". **That premise was wrong, and it is the main
finding of this round.**

Re-reading all ten end to end: in every one, the legacy block is a _scalar
remainder loop_ that runs after the vector work, and every YMM the vector code
uses is fully redefined by a 256-bit VEX write at the head of each vector-loop
iteration. Every path from a legacy block back into vector code (the six
legacy→VEX back edges: `avx2_f32_generic.s` 610→260, 1422→1138, 967→788,
1820→1639; `avx2_f64_stockham.s` 269→96, 560→385) passes through such a
redefinition. So no `Yn` upper half is live across any legacy write, and a
straight mnemonic conversion is data-flow safe. The sweep's liveness pass was
conservative, not correct.

That matters because the remedy this item proposed — renumber the aliased
register — was **not available**: the four Stockham scalar cores need `X0`–`X13`
against `Y0`–`Y7` of VEX code, leaving only numbers 14 and 15 free. Had the
premise been true, the fix would have had to be rematerialization, not
renumbering.

Two blocks turned out to be dead rather than mixed:

- `·ForwardAVX2Size1024Radix32x32Complex128Asm` carried a complete 221-line
  scalar `fwd_fft32` helper that **no branch reaches**. The complex128 forward
  vectorises stage 2 as well (2 rows at a time via `VINSERTF128`), which
  orphaned the helper; the complex64 forward has not had that treatment and
  still calls its scalar helper for stage 2. Deleting it made that function
  100% VEX outright.
- `·InverseAVX2Size1024Radix32x32Complex64Asm` opened with
  `MOVSS ·scale1024f32<>(SB), X14` whose value is never read — `Y14` is fully
  redefined at the top of `inv_fft32x4` and the scale is carried by `Y15`.

Totals: of 730 legacy instructions across the four files, **611 converted, 119
deleted**; 235 lines removed net.

**Mechanics worth reusing.** Three translation traps, none of which is visible
in a source-level diff:

- **Register-to-register `MOVSS`/`MOVSD` merges** — legacy preserves
  `dst[127:32]`, so a two-operand `VMOVSS Xa, Xb` is a different instruction.
  Use the three-operand `VMOVSS Xa, Xb, Xb` rather than `VMOVAPS`: it is exactly
  equivalent _and_ it normalizes back to `movss a,b` under the disassembly gate,
  so the check stays meaningful. There were 8 such moves per Stockham function.
- **Go spells the VEX conversion mnemonics differently.** `CVTSQ2SD` →
  `VCVTSI2SDQ`, `CVTSQ2SS` → `VCVTSI2SSQ`; a mechanical `V`-prefix rewrite fails
  to assemble, which is the benign failure mode.
- **`FWDBFLY`/`INVBFLY` in `avx2_f64_size512_radix8.s` are macro invocations**,
  not instructions. Any regex census of "non-`V` mnemonic with an X/Y/Z operand"
  flags them; they are the only two false positives in the tree.

**Verification.** The §1.8 harness was never committed, so it was rebuilt: a
throwaway per-symbol normalizer over binutils `objdump -d` output, collapsing
the `v` prefix, the VEX merge operand (applied repeatedly, so the four-operand
`vshufps $i,a,b,b` reduces to the same string as its two-operand legacy form),
RIP displacements (replaced by the symbol objdump names), branch targets
(replaced by the target's instruction index within its symbol, which is
shift-invariant) and `int3` padding. Locked against a pristine `git worktree` at
the parent commit and proven deterministic across rebuilds first. Result: **all
9967 symbols across the `internal/kernels` and `internal/fft` test binaries
decoded identically**, bar the two where instructions were deliberately deleted.
It caught one real defect during the round — the four-operand `vshufps` case
above — before any test ran. `cmd/measure_correctness` is **bit-identical** to
the parent commit at every size and both precisions.

**Performance: no change, as the byte-equivalence implies.** Eight interleaved
ABBA rounds against a pristine worktree at the parent commit, pinned to
`taskset -c 0`, put all four `Size1024Radix32x32` kernels at 0.98–1.01× — inside
the noise. Two measurement notes for next time:

- A plain `A,B,A,B` loop showed **+4% on an unchanged control** because one arm
  always ran on the hotter half of each round. Alternating the order (ABBA)
  removed it. §2.2 says to interleave; interleaving is not enough, the _order_
  has to alternate too.
- Even under ABBA, `dit1024_radix4_avx2` — code this round did not touch, and
  which the gate proved byte-identical — moved 0.87×, with one round at 0.61×
  and visibly asymmetric variance between the arms. That is a code-placement or
  contention artifact, not a result. **Any cross-binary claim smaller than ~15%
  on this box needs a control benchmark to be believable.**

**The 32×32 codelet is not worth keeping, and the mixing penalty was never why.**
Measured order-free _within one binary_ (median of 5 rounds, pinned), size 1024
complex64:

| codelet                      | forward            | inverse            |
| ---------------------------- | ------------------ | ------------------ |
| `dit1024_radix4_avx2`        | 849 ns (1.00×)     | 836 ns (1.00×)     |
| `dit1024_radix4_sse3`        | 4174 ns (4.9×)     | 4696 ns (5.6×)     |
| `dit1024_radix4_generic`     | 5820 ns (6.9×)     | 5793 ns (6.9×)     |
| `dit1024_radix32x32_generic` | 6277 ns (7.4×)     | 7432 ns (8.9×)     |
| `dit1024_radix32x32_avx2`    | **6943 ns (8.2×)** | **8233 ns (9.9×)** |

The AVX2 32×32 kernel is the **slowest candidate at its size — 11% slower than
its own pure-Go twin**, in both directions. This item hoped the 7.1 µs vs 3.5 µs
gap was the mixing penalty; it is not. The kernel vectorises only one of its two
stages (the complex64 forward runs stage 2 scalar, the inverse runs stage 1
scalar), so it does half its work one complex number at a time whatever the
encoding. Its registry priority of 30 against radix-4's 90 was correct, and the
complex128 side was already disabled at `-1`. See the follow-up item in §4.

### 1.17 The n = 2048 radix-2 tail (2026-07-30)

§4 read the 2048 dip as pointing "at the route rather than at the arithmetic".
It does, and the route has a name: for `n = 2*4^k` the size-generic radix-4
kernel runs its radix-4 stages only to `n/2` — transforming the even- and
odd-indexed halves independently — and then makes **a separate full pass over
the buffer** to combine them (`r4d_radix2_tail`). Counting passes, 2048 costs 6
for 11 butterfly levels where 1024 costs 5 for 10 and 4096 costs 6 for 12:
0.545 passes per level against 0.50 at both neighbours. 2048 is also the only
one of the three with no second AVX2 candidate at all.

**What the tail costs, measured rather than reasoned.** A probe that skips the
combine outright — wrong answers, right instruction mix, the §1.9 radix-11
idiom — puts it at roughly **8–15% of the kernel for n ≥ 512, and 9–20% at
n = 128**, in both precisions and both directions. Read those as ranges, not
readings: the same cell moves a few points between sweeps, and one 8192 inverse
cell read 2.8% against 10–15% in the other three. The ordering is the stable
part, and it tracks the pass model — the tail is one pass of `k+1`, so 25%
predicted at 128, 17% at 2048, 12.5% at 32768, and the measurements fall below
each of those in the same order. The probe needed no assembly: the kernel's only
shape knob is `r4End`, and passing `n` instead of `n/2` leaves the executed
stages bit-identical (the next stage would overrun either way) while the tail's
own guard then skips it. `TestRadix4AVX2NoTailProbeIsStagesOnly` proves that
equivalence by applying the missing combine in Go and requiring the real
kernel's output back, rather than asserting it from the loop bounds.

That number is the ceiling on any fusion, and it is worth having on its own:
**the tail is a tax on every odd power of two, not a 2048-specific defect.** It
costs a comparable fraction at 512, 2048 and 8192 — which sit at 0.97, 0.91 and
1.24 against FFTW3 (tag v0.7.4). So it explains why the `2*4^k` sizes are all
~10% below where they could be; it does not explain why 2048 is the one that
lands under parity. Nor is the tail the whole of the mid-band softness: 1024 is
a power of four, has no tail at all, and still measures 0.97.

**Fusing it into the last radix-4 stage works, and mostly does not pay.** The
last stage always has `4m = r4End = n/2` and therefore exactly two groups — the
even half and the odd half — and the tail pairs one output of each at the same
position, so running the two groups in lockstep leaves both operands of four
radix-2 butterflies in registers. Output addresses, the permutation table and
the packed twiddle layout are all unchanged; only the loop structure moves. The
register file ends up exactly full (four outputs per group, six scratch, two
rotation masks), which is why group 1 re-loads its twiddle broadcasts instead of
keeping them.

Fused as a ratio to the separate tail, forward/inverse, canary-gated, pinned,
7–10 accepted groups per cell:

| n     | stride (c64/c128) | complex64         | complex128        |
| ----- | ----------------- | ----------------- | ----------------- |
| 128   | 128 B / 256 B     | **0.955 / 0.979** | **0.935 / 0.934** |
| 512   | 512 B / 1 KiB     | 0.971 / 1.005     | 1.002 / 1.020     |
| 2048  | 2 KiB / 4 KiB     | **0.943 / 0.974** | **1.110 / 1.104** |
| 8192  | 8 KiB / 16 KiB    | 1.034 / 1.021     | 1.006 / 1.077     |
| 32768 | 32 KiB / 64 KiB   | 1.004 / 1.013     | 1.020 / 1.000     |

Fusing doubles the live streams from four to eight, and past a point that costs
more than the pass it saves. **The single worst cell is complex128 at n = 2048 —
11% slower — which is the size the fusion was written for.** Its last-stage
stride is exactly 4 KiB there, so all eight streams land on one L1 set; that the
target size is also the pathological one is arithmetic, not luck.

Two corrections to how this was read at the time, both from adding data:

- A first pass over six cells suggested "loses whenever the stride is a multiple
  of 4 KiB". Sweeping n = 128 and n = 32768 kept the small-stride win but made
  32768 **neutral** rather than the predicted loss — at 256–512 KiB both variants
  are bandwidth-bound and an L1 effect cannot decide anything. The trend is not
  monotonic in stride either (2 KiB wins, 1 KiB slightly loses), because the
  complex64 loop retires four butterflies per iteration to complex128's two and
  amortises the doubled stream count better. **Six points supported a rule that
  four more falsified.**
- The fused kernel is correct, and known to be so cheaply: fusing reorders no
  arithmetic, so `TestRadix4AVX2FusedMatchesUnfused` demands **bit-identical**
  output at every size 16…32768. An approximate comparison would have waved
  through a real defect as rounding.

**Landed per size, in the registry.** Rather than a runtime predicate over an
empirical rule, the three cells that win — `dit128_radix4fused_avx2` in both
precisions and `dit2048_radix4fused_avx2` at complex64 — are ordinary
`specs.go` rows, which is where every other per-size measured fact in this
library already lives. `TestRadix4AVX2FusedSelection` pins the choice so a
regenerate cannot widen it silently, and the existing ranking tests re-derive it
from measurement. The `-tags fftprobe` harness stays in tree and registers both
variants plus the no-tail probe side by side, so the table above can be
re-derived on other hardware instead of being trusted (§2.4's rule about numbers
that cannot be re-measured where they are quoted).

_Not fixed:_ complex128 at n = 2048, the item this round was opened for, is
unchanged — the fusion is the wrong instrument there. The tail still costs
12–13% at that cell and reclaiming it needs a shape with a different access
pattern; the candidate is a twiddle-free radix-8 first stage, which reads
through the permutation and would not meet the 4 KiB stride at all.

**Incumbent audit, discharged for seven sizes.** The same sweeps cover §4's open
audit at 128/512/1024/2048/4096/8192/32768 in both precisions. Every incumbent
was confirmed as the fastest correct candidate **except the three this round
then changed** — n = 128 in both precisions and n = 2048 complex64, where the
fused variant won and took the row. Two side observations worth the record:
`dit4096_sixstep_avx2` runs 4.6× its size's incumbent, and
`dit1024_radix32x32_avx2` 8.1×/9.8×, which independently confirms §1.16's
deletion item.

---

### 1.18 The race suite could not finish, and why (2026-07-30)

`internal/kernels` took **1499.7 s** under `-race` against Go's 10-minute
default timeout, which neither `just test` nor any CI workflow overrode. The
package therefore could not complete under the race detector at all — a gate
that had been red repo-wide, independently of any kernel change.

**Where the time went**, measured per-test under `-race` rather than
extrapolated from an untagged profile: six AVX2 sweeps at ~207 s each and eight
fixed-size DIT tests at 100–152 s. All fourteen are naive O(n²) reference DFTs.
Nothing else in the package exceeded 40 s. The reference grows as n² while the
kernels it validates grow as n·log n, and every load and store in that double
loop is instrumented, so n = 32768 alone was ~78% of each affected sweep
(n² = 1.07e9 against 1.42e9 for the whole size list).

The first instrument was wrong and worth recording: an untagged `-v` profile
ranks the four n = 32768 tests at the top, but those are _already_ skipped
under `-race` by the pre-existing `raceDetectorEnabled` guard
(`dit_32768_radix4_then2_test.go`). Profiling the build you are not fixing
names tests that never run.

**The fix is not a bigger timeout**, and mostly not a skip either:

- `TestRadix4AVX2InPlace` and its complex128 twin no longer touch the naive
  DFT. What they test is that aliasing `dst` to `src` changes nothing, and both
  paths run identical arithmetic in identical order — so the oracle is the
  out-of-place kernel, asserted bit-for-bit. That is cheaper _and_ stronger:
  the old `2e-4*n` float32 tolerance was wide enough to accept a real aliasing
  defect as rounding. Full size coverage retained in every build.
- The four `MatchesReference` sweeps cap at `naiveReferenceRaceMaxSize = 4096`
  under `-race`, keeping both kernel shapes (`4^k` and `2·4^k`) with their
  distinct permutation, twiddle and tail paths, and dropping only the three
  sizes that repeat those shapes at quadratic reference cost. The filter logs
  what it dropped, so a shrunken sweep cannot read as full coverage.
- `TestRadix4AVX2MatchesStockham` is **new**. The complex128 file already
  cross-checked 8192–65536 against Stockham — an independent algorithm, no
  shared permutation table — but complex64 had no large-size cross-check at
  all. The sizes now capped under `-race` are covered by this test in every
  build, so the complex64 kernel is better covered after the change than
  before it.

Result: **1499.7 s → 147.8 s**, a 10.1× reduction, comfortably inside the
default timeout. `-timeout=20m` went into `justfile` and the three `-race` CI
invocations as a backstop rather than as the fix.

**Two tolerances were nearly blind, found by testing the tests.** Both Stockham
cross-checks passed, so a green run proved nothing about their sensitivity.
Measuring the margin between the asserted bound and the actual agreement:

| test                                                                    | bound    | measured agreement | margin         |
| ----------------------------------------------------------------------- | -------- | ------------------ | -------------- |
| c64 vs Stockham (as first written, copying the neighbouring convention) | `2e-4·n` | 3.8e-5 → 1.4e-4    | 4.3e4 – 9.1e4× |
| c128 vs Stockham (pre-existing)                                         | `1e-9·n` | 1.3e-13 → 4.3e-13  | 6.2e7 – 1.5e8× |

A defect perturbing one output bin by 1% of peak magnitude produced a diff of
~0.06 — still 20–200× under the c64 bound. Both now scale as `sqrt(n)`, which
is how two O(n·log n) implementations actually diverge (random-walk rounding),
and clear their measured agreement by ~8–10×. The `2e-4·n` convention is
correct where it came from — against the _naive_ DFT, which is itself the
inaccurate side — and wrong when carried over to a comparison where it is not.
Per §1.11, a passing test is not evidence until you know what it would reject.

---

### 1.19 The incumbent audit at n = 8…64 and 16384 (2026-07-30)

This closes §4's remaining audit sizes. One canary-gated sweep, 10 groups /
138 cells / 16 passes, pinned to core 0: **159 of 160 groups accepted, 1 over
gate, 0 drift** — the cleanest window any round has had, and worth noting why.
The default `GOOD=1810` is stale twice over: the canary floor measured **1565
ns** here (7 of 8 samples in 1565–1596), below even the ~1650 recorded after
the last recalibration. Sweeping against a stale floor does not bias the
ratios — those are taken within a group — but it does let in windows that
should have been rejected, so recalibrating first is not optional.

**Eight of nine incumbents confirmed**, most by wide margins: the AVX2 row wins
its size by 1.2–10.8× over every other candidate at 16/32/64/16384 in both
precisions and at n = 8 complex64.

**One incumbent was wrong: complex128 at n = 8.** `dit8_radix4_avx2` was the
registered choice; `dit8_radix8_avx2` beats it at **0.970 forward / 0.859
inverse**, medianed over 16 groups, and is now the row. Reading the numbers
honestly: the forward gap is 0.2 ns and would not on its own justify a change,
and at plan level the ~100 ns per-call dispatch (§4) swamps the whole
difference. The inverse gap — 8.2 ns to 7.0 ns — is the substantive part.
`dit8_radix2_avx2` ties it on inverse (0.866) but loses forward, so radix-8
wins both directions rather than trading them. Two SSE2 rows also beat the old
incumbent on forward, but registry ordering is SIMD-level major, so they could
never have been selected on an AVX2 host and are not candidates for the row.

The size-4 rows need no audit at all: every tier registers exactly one
candidate there, so there is nothing to rank.

**A tooling trap cost a false conclusion first.** `bench_gated.sh` takes its
output directory from `OUTDIR`, but `bench_gated_analyze.sh` takes it as a
**positional argument** and ignores `OUTDIR`. Invoking the analyser with
`OUTDIR=...` silently analysed a stale directory from the previous round and
reported "27 accepted, 5 over gate" with only the sizes that happened to
overlap — a plausible-looking result computed from the wrong data. What
exposed it was the arithmetic not adding up: 10 groups × 16 passes is 160
group-instances, and 27 + 5 is 32. Check that a sweep's accepted+rejected
equals the group count you asked for before reading a single ratio.

---

## 2. Working method

### 2.1 Correctness gates

Every item below is held to these:

1. **Measure baseline first** (`benchstat`, committed `benchmarks/baseline-*.txt`
   where relevant).
2. **Verify against reference**: forward-vs-`reference.NaiveDFT`, round-trip and
   in-place for every new kernel/codelet (the registry sweep gates registered
   codelets automatically).
3. **Both precisions** unless the item is precision-specific; **zero-alloc**
   guards extended to any new path.
4. **No regressions on the purego build** — algorithm-level wins must land in
   the generic code path too, not only in assembly.
5. Only land a per-size codelet if it **beats the incumbent in `benchstat`** on
   the target hardware tier.

### 2.2 Measurement protocol

Most of the wrong conclusions in this file's history came from trusting a
number taken under load. What works:

- **Interleave the arms** in one process, with the order rotated per round, and
  report medians. Arms run minutes apart are measuring the machine.
- **Canary-bracket every group**, not every pass. A 94-cell pass takes 5–13
  minutes, so contention arriving mid-pass goes unseen: 3 of 5 nominally clean
  passes were contaminated, one by 50×. In-tree as `scripts/bench_gated.sh` +
  `scripts/bench_gated_analyze.sh` / `just bench-gated`. A group is one
  (precision, size) with all its candidates back-to-back, so a whole ranking is
  taken inside a single verified-quiet window.
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

### 2.3 Hardware tiers

Three machines are reachable, and they are complementary rather than redundant
— several findings above exist only because a result differed between them.
Server access is weekend-only, so none of this belongs in CI; treat them as
periodic validation sweeps.

- **Dev laptop (i7-1255U, AVX2, no AVX-512).** The only one with FFTW
  installed, so the only place the external gap can be measured. Throttles
  hard (86–98 °C under sustained benchmarking) — interleave arms, trust ratios.
  `go` here is a wrapper (`nice -n 10 taskset -c 0-$(nproc-2)`): benchmark with
  `/usr/local/go/bin/go` under `taskset -c 0` or the desktop preempts the run
  (§1.15).
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

### 2.4 Standing lessons

Each of these cost a real investigation. The assembly ones are also in
`AGENTS.md`.

- **A number measured in another repository is not maintained by editing this
  one.** §1's headline FFTW ratio aged through four releases while every round
  below it stayed current, because the sweep is committed in `go-fft-bench`
  against a pinned tag and nothing pulls it back. It was wrong by a factor of
  two and in the wrong _direction_ — the library had overtaken FFTW3 at powers
  of two and this file still said 0.63×. Quote such a number with the tag it
  came from, or re-measure it (§1.15).
- **Check the toolchain wrapper before blaming the machine.** `go` on the dev
  laptop resolves to a wrapper that runs `nice -n 10 taskset -c 0-$(nproc-2)`,
  so benchmarks yield to the desktop; the same sweep is clean pinned and
  un-niced at load 4.4 and unusable through the wrapper. Measure with
  `/usr/local/go/bin/go` under `taskset -c 0` (§1.15, §2.3).
- **A registered fast path is not a reachable one.** Codelets for exactly the
  sizes the Bluestein pad produces sat in the registry, correct and never
  called, because that route entered a hardcoded size switch instead. The
  symptom was two builds measuring the _same_ — which is the same tell as §1.12's
  "an optimisation that changes nothing". Before profiling a path that looks
  slow, check that the fast version of it runs at all.
- **A size-generic kernel silently closes per-size gaps the plan still lists as
  open.** §4's opening item asked for a file that will never exist; the kernel
  is `avx2_f64_radix4.s`, which covers every `n = 2*4^k` as a radix-4-then-2.
  Ask "does the generic kernel's shape rule cover n?", never "is there a file
  named for it?" — a name-based search returns the stale premise as
  confirmation. This is the same failure as the two below, one level up: the
  plan file itself carried the wrong assumption for a week.
- **A dispatch toggle's stated reason must be re-derived, not inherited.** The
  packed-Stockham toggle claims the codelet path supersedes it; every codelet is
  registered as `KernelDIT`, so the strategy check upstream had already excluded
  it and the toggle only ever suppressed the sizes with no codelet at all.
- **Test vectors must not be blind.** An impulse cannot detect a wrong twiddle
  (they all multiply zeros) or a wrong output ordering (its spectrum is
  all-ones); Parseval and linearity are insensitive to both. That combination
  hid a wrong-answer bug at every size ≥ 1024 for an entire precision.
- **Scaling by a real factor must never be written as a complex multiply** —
  it spends two dead products per element, and folding one into a
  fully-unrolled stage can cross the inliner's big-function threshold and
  silently un-inline _every_ helper in the function.
- **A type switch inside a generic body is concrete-typed code reachable
  without monomorphizing** — and Go compiles _every_ branch into _every_ shape
  instantiation, so a `complex64` branch charges its cost to the `complex128`
  instantiation too.
- **Never mix VEX and legacy-SSE vector instructions in one function.** A
  partially converted hot loop measured **152× slower** than the same loop left
  uniformly legacy. Convert a function completely or not at all.
- **Watch the 1/n inverse prologue.** `MOVL`/`MOVD`/`VBROADCASTSS` costs a
  fixed ~100 ns via the transition penalty. Because the cost is per call, it
  only shows up on small kernels — where it silently mis-ranks them in the
  registry.
- **Gate any bulk asm rewrite on a disassembly diff** from binutils `objdump`
  (Go's `go tool objdump` misdecodes AVX), normalized for the `v` prefix, the
  VEX merge operand, shifted addresses and `int3` padding.
- **Registry ordering is SIMD-level major**, priority only within a level. Use
  `RankLevel` to demote a wide-ISA codelet; never to promote a narrow one.
- **Permutation tables are precision-independent.** Copy the twin file's table
  or `internal/math`'s helper; a self-derived one is the only correctness bug
  that has reached a test run here.
- **Don't move a broadcast to a memory operand when its source is the data
  stream** — see §1.9.

---

## 3. Correctness and honesty debt

Things that are wrong, silently misreport, or violate a stated contract. This
section gated the tag.

**Both items are closed** — see §1.13 (the Bluestein sub-FFT never reached the
codelet registry) and §1.14 (packed Stockham was compiled out of SIMD builds).
They turned out to be the same bug twice: a fast path that existed, was
correct, and was unreachable, in both cases behind a dispatch decision whose
stated justification had never been re-derived. Nothing here blocks §9.

---

## 4. amd64: finish the radix-4 round and the remaining soft spots

The 256-bit radix-4 kernels (§1.9) changed the cost of every power-of-two size,
which invalidates several constants and leaves a few threads hanging.

- [x] **The size-384 sub-FFTs were bound to superseded 128-point kernels**
      (2026-07-29). This item used to read "no AVX2 complex128 radix-4-then-2
      kernel at size 128 — writing it is the fix". **The kernel already
      existed.** §1.9 had replaced the per-size radix-4 family with the
      size-generic `avx2_f64_radix4.s`, and 128 = 2·4³, so that kernel runs its
      radix-4 stages to 64 and combines with a radix-2 tail — a radix-4-then-2
      at this size, by construction. It is registered as `dit128_radix4_avx2`
      (priority 90, above `dit128_radix2_avx2` at 20 in the same tier), size 128
      was already lifted 367 → 129 ns, and `TestRadix4AVX2Complex128Ranking` was
      already asserting the ordering at n = 128. The item survived because it
      was phrased in **file names**, and no file is named
      `avx2_f64_size128_radix4_then2.s` — a name-based search reproduces the
      stale premise instead of correcting it (§2.4).

      What was actually open was **binding**, at the one call site that does not
      go through the registry: `dit_384_decomp_128x3_amd64_asm.go` calls its
      128-point sub-FFT symbol directly, and **both** precisions were on a
      superseded kernel — c128 on the plain radix-2, and c64 on the _pre-§1.9
      XMM-width_ `avx2_f32_size128_radix4_then2.s` (320 ns against the new
      kernel's 88). So the c64 side, which §1.12 had signed off because its
      kernel was _named_ radix-4-then-2, was the larger of the two misses. It
      had not been switched because the generic kernel wants a prepared twiddle
      table rather than the plain length-n one — trivial to supply here, since
      the sub-size is a constant, and the four tables (two precisions × two
      directions) are built once at package load.

      No new assembly, no spec-table row. Measured, `benchstat` n = 6 on the
      laptop:

      | cell         | before | after | delta |
      | ------------ | -----: | ----: | ----: |
      | c64 forward  |   1601 |   644 |  −60% |
      | c64 inverse  |   2297 |   806 |  −65% |
      | c128 forward |   2284 |   916 |  −60% |
      | c128 inverse |   2613 |  1115 |  −57% |

      Geomean −58%, zero-alloc preserved. Far outside §1.12's ~15% noise floor
      here, and the free identical-code control pair (`generic`/`avx2`, the same
      function registered twice) sat within 5% of itself in the new run.

- [x] **The six-step row FFTs were still on the pre-§1.9 128-point kernel**
      (2026-07-29). Found while closing the item above, and it was the same fix
      again: six call sites in `dit_16384_sixstep_amd64_avx2.go` and
      `dit_8192_sixstep_64x128_amd64_avx2.go` called
      `{Forward,Inverse}AVX2Size128Radix4Then2Complex64Asm` for their row
      transforms — the XMM-width kernel at 320 ns where the generic one is 88.

      The two worries did not materialise. **In-place is fine**: the generic
      kernel is a registered codelet at size 128 and `TestInPlaceAllCodelets64`
      already sweeps `dst == src` for it, so `row, row` needed no thought.
      **The per-row twiddle hoists better than in the 384 case**: the six-step
      files were gathering their length-128 row table out of the caller's
      length-n table on _every_ transform (a stride-128 loop at 16384, stride-64
      at 8192, four such loops across the two files). Two package-load tables —
      `sixStepRow128{Fwd,Inv}TwiddleC64`, forward and inverse separate because
      `prepareTwiddleRadix4AVX2` conjugates at prepare time — replace all four,
      so the swap _removes_ per-call work rather than adding a table.

      No new assembly. Measured, `benchstat` n = 6, two test binaries
      interleaved with the order rotated per round, pinned to one core:

      | cell            | before | after | delta |
      | --------------- | -----: | ----: | ----: |
      | 16384 forward   |  141.3 |  70.9 |  −50% |
      | 16384 inverse   |  156.6 |  71.9 |  −54% |
      | 8192 forward    |   85.9 |  68.0 |  −21% |
      | 8192 inverse    |  100.2 |  69.7 |  −30% |

      (µs, c64.) Geomean −33%, zero-alloc preserved. The null control
      (`ForwardDIT16384Radix4AVX2_Complex64`, code the change cannot reach) sat
      at p = 1.000, which is what makes the ±15–47% spread on the old arm
      readable at all. No inverse benchmark existed for either six-step; both
      were added _before_ the baseline binary was built, so the inverse side is
      measured rather than assumed.

      Two things this did **not** do. The size-64 row kernels in the 8192/4096
      files stay — that symbol is still the registry incumbent at n = 64 c64, so
      swapping it is a separate benchmark, not a cleanup. And the `.s` file is
      **still not deletable**: the `KernelStrategy` dispatch below
      (`internal/fft/asm_amd64.go:158,246`) is now its only non-test caller, so
      that item alone gates the deletion.

      **It also made `TestRadix4AVX2Ranking` flaky, which is the interesting
      part.** The test fails the radix-4 kernel if it measures more than 1.5×
      the fastest codelet at that size — so the tolerance it actually grants is
      relative to the _runner-up_. Speeding the six-step up at n = 16384 pulled
      the runner-up from ~141 µs to ~60 µs, which tightened the contention
      headroom on radix-4 from ~211 µs to ~90 µs without a line of the test
      changing. One `go test ./...` then read radix-4 at 86 µs against six-step
      at 56 and called it a regression; the gated sweep on the same tree says
      26.3 vs 59.9 µs, so radix-4 still wins by 2.3× and the ranking was simply
      contended. The baseline tree passes the same parallel run, so this was
      genuinely introduced here rather than pre-existing.

      Fixed by re-measuring before failing (`rankingAttempts = 3`): a real
      regression reproduces on every pass, a contended window does not. Note
      the failure mode the retry addresses — a burst covering all five rounds
      of _one_ candidate and none of the next inflates a single codelet rather
      than the group, so it surfaces as a ranking change, not as uniformly
      slower numbers. Best-of-N within a pass cannot see that; only a repeated
      pass can. **A speedup can break a threshold test that measures nothing it
      touched**, whenever the threshold is expressed relative to a peer.

      Note what this does and does not move at the plan level: the generic
      radix-4 codelet already outranks the six-step entries in the registry
      (priority 90 against 35) at c64 4096/8192/16384, so nothing that goes
      through the registry changes. The win lands on the forced-`KernelSixStep`
      route — which is exactly the population §1.13/§1.14 are about, a correct
      path that had quietly stopped being the fast one.

- [x] **Re-measured the plan-level c64/c128 ratio and the FFTW comparison**
      (2026-07-29, v0.7.4). Both had moved much further than this item claimed,
      and the plan file was the last place still carrying the old numbers — see
      §1.15 for the round, including why the first sweep had to be thrown away.

      The **plan-level ratio now tracks the codelet-level one**, which is the
      result the item was actually asking about — the plan layer is no longer
      masking the precision difference:

      | n              |  1024 |  2048 |  4096 |  8192 | 16384 |
      | -------------- | ----: | ----: | ----: | ----: | ----: |
      | was, plan      |  1.10 |  1.02 |  1.14 |  1.07 |  1.08 |
      | now, plan fwd  |  1.57 |  1.73 |  1.91 |  1.96 |  1.77 |
      | now, plan inv  |  1.59 |  1.76 |  2.01 |  2.02 |  1.80 |
      | codelet (§1.9) |  1.64 |  2.09 |  2.43 |  2.27 |  1.85 |

      Plan-level sits just below codelet-level at every size, which is what the
      ~100 ns of per-call dispatch predicts: that cost is precision-independent,
      so it dilutes the ratio, and dilutes it most where the transform is
      cheapest. At 256/512 the ratio is now 1.40/1.55 forward against the
      1.6–2.1× recorded before — the one place it moved _down_, because the
      complex128 side gained more there than complex64 did.

      **Against FFTW3 the power-of-two picture has inverted.** Not 0.63× but
      **1.36× forward, 1.34× inverse** by geomean over 8…32768 (complex128,
      median of within-round ratios, 6 interleaved rounds, pinned):

      | n   |   8 |  16 |  32 |  64 |  128 |  256 |  512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
      | --- | --: | --: | --: | --: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ----: | ----: |
      | fwd | 5.2 | 2.4 | 1.9 | 1.3 | 1.03 | 1.07 | 0.97 | 0.97 | 0.91 | 1.16 | 1.24 |  1.19 |  1.03 |
      | inv | 4.3 | 2.6 | 1.9 | 1.4 | 1.02 | 1.02 | 0.94 | 1.01 | 0.91 | 1.10 | 1.15 |  1.28 |  1.07 |

      Non-power-of-two is **0.60×** by geomean (was 0.25× at v0.7.0), and 44100
      is 786 µs against FFTW's 185 µs. §5 remains the gap.

      The nine commits since v0.7.3 are **flat** at power-of-two lengths in both
      precisions (medians 0.88–1.07 over a 6-round, order-rotated, two-binary
      A/B) — which is what they should be: they touch the 384 path, the forced
      six-step route, Bluestein pads and packed Stockham above 2^17, none of
      which is the default power-of-two route in this band. The numbers above
      are therefore a correction of the record, not a new win.

- [ ] **Retune the strategy thresholds around the new codelet.**
      `ditAutoThreshold` and the six-step/four-step crossovers were calibrated
      against kernels that are now 2–4× faster, so the size at which DIT stops
      winning has moved. The Stockham comparison at n = 16384 that opened the
      investigation is the clearest case: 94 µs against what is now 29 µs.
- [ ] **Point the `KernelStrategy` dispatch at the generic radix-4 kernels.**
      `internal/fft/kernels_amd64_size_specific.go` selects by strategy rather
      than through the codelet registry, so it cannot supply a prepared twiddle
      table and still needs the old per-size files — c64 at 256/512/2048/8192,
      c128 at 4/8/16/32/64/512. Giving that path a way to obtain the prepared
      table is what would let the last of the family go.
- [ ] **Re-check the gather balance on AMD.** The stage-1 fusion assumes
      `VPGATHERDQ` throughput comparable to Alder Lake's. Gather is historically
      much weaker on Zen, and the fused path has no fallback, so the
      13.9 → 9.7 µs result should be reproduced on a Zen part before it is
      treated as universal. If it inverts there, the separate permutation pass
      is still in git history and the two differ only in this one block.
- [ ] **Plan-level overhead now dominates small transforms.** With the size-64
      codelet at 55 ns, `BenchmarkPlanForward_64` still reports ~155 ns — about
      100 ns of dispatch/validation per call that used to hide behind a 137 ns
      codelet. The same overhead sits on every size; it is simply invisible
      above ~1024. Profile the path from `Plan.Forward` to the codelet call.
- [ ] **The n = 2048 local minimum.** Worked, partly (§1.17), and the item is
      sharper than it was. The route hypothesis was right: `n = 2*4^k` pays a
      **separate full pass** over the buffer for its radix-2 tail, and a probe
      that skips the pass outright measures that at **9–15% of the kernel**.
      But that tax is uniform across 512/2048/8192, which sit at 0.97/0.91/1.24
      against FFTW3 (tag v0.7.4) — so it explains why every odd power of two is
      ~13% below where it could be, **not** why 2048 in particular is the one
      under parity. Fusing the tail into the last radix-4 stage is implemented
      and now selected at n = 128 (both precisions) and n = 2048 complex64,
      worth 4–6%; it is **11% worse** at n = 2048 complex128, where the
      last-stage stride is exactly 4 KiB and the fused loop's eight live streams
      collide in L1. So the target cell is unchanged. What remains: a shape with
      a different access pattern, the candidate being a **twiddle-free radix-8
      first stage** (permutation + radix-2 + radix-4 in the pass that already
      touches everything), which would not meet the 4 KiB stride at all — it
      needs the `±(1±i)/√2` rotations and its own stage-1 permutation table.
      Note also that 1024 sits at 0.97 with no tail at all, so some of the
      mid-band softness is not the tail. The AVX-512 item in §6 mentions
      reclaiming 2048; this is the AVX2 tier and independent of it.
- [x] **Make the 10 remaining mixed functions uniformly VEX.** Done — see §1.16.
      The premise this item was written on turned out to be wrong: no `Yn` upper
      half is live across any of the legacy blocks, so no renumbering was needed
      (and none was possible for six of the ten anyway).
- [ ] **Delete the size-1024 radix-32×32 codelets.** Now that they are uniformly
      VEX, §1.16 measures `dit1024_radix32x32_avx2` as the **slowest candidate at
      its size in both directions** — 8.2×/9.9× `dit1024_radix4_avx2` and 11%
      slower than its own pure-Go twin — because only one of its two stages is
      vectorised. It is already shadowed (priority 30 vs 90 at complex64,
      disabled at `-1` for complex128), so nothing selects it. Per §2.4's
      replace-don't-shadow rule that makes ~2500 lines of assembly plus a
      `PrepareTwiddle` layout dead weight: remove `avx2_f32_size1024_radix32x32.s`,
      `avx2_f64_size1024_radix32x32.s`, their `decl.go` declarations, both
      `specs.go` rows and the tests. Consumers are registry-only — there is no
      `KernelStrategy` dispatch site to untangle. Check `·bitrev32<>` is not
      referenced across files before deleting (it is `<>`-scoped and defined once
      per file, so it should not be). The pure-Go `dit1024_radix32x32_generic`
      loses to `dit1024_radix4_generic` too and is a candidate in the same sweep.
- [ ] **134 YMM/ZMM-using functions never execute `VZEROUPPER`.** Separate from
      the encoding sweep and untested so far: these return with the upper state
      dirty, so the cost lands on the _caller_ — Go's own SSE2-generated float
      code, and any pure-SSE2 codelet selected afterwards (e.g. the c128 n = 64
      winner). 57 of the 151 amd64 asm files contain no `VZEROUPPER` at all.
      Measure a kernel-then-SSE2-kernel sequence before changing anything.
      (One data point already: an AVX2 n = 64 codelet followed by an SSE2 n = 16
      codelet was, if anything, _cheaper_ than followed by the AVX2 one — so
      this does not block cross-tier selection.)
- [ ] **Finish the FMA audit.** Two pieces remain from §1.4:
      (a) the non-codelet AVX2 dispatch sites
      (`internal/fft/complex_mul_amd64.go`, `kernels_amd64_asm.go`,
      `scale_amd64.go`, `internal/kernels/radix5_avx2.go`) still gate on
      `HasAVX2` alone and need the `HasAVX2 && HasFMA` sweep;
      (b) the FMA-less files that no size currently selects
      (`avx2_f32_size512_radix16x32.s` with 128 muls,
      `avx2_f{32,64}_size1024_radix32x32.s`, `avx2_f32_size256_radix16.s`,
      `avx2_f32_size128_radix2.s`, `avx2_f32_size32_radix4_then2.s`,
      `avx2_f{32,64}_size4_radix4.s`) — fusing those only matters if a priority
      retune brings them back into play. The generic radix-4/Stockham kernels
      need no pass; they are already fused.
- [x] **Finish the incumbent audit** (2026-07-30). §1.8 confirmed
      n = 256/512/8192; §1.17 added 128, 1024, 2048, 4096 and 32768 in both
      precisions — confirmed except at 128 (both) and 2048 complex64, where it
      replaced the incumbent with the fused-tail variant. §1.19 closes the
      remainder: 8/16/32/64 and 16384 in both precisions, 159 of 160 groups
      accepted. Eight of nine incumbents confirmed; **complex128 at n = 8 was
      mis-tuned** and `dit8_radix8_avx2` took the row (0.970 fwd / 0.859 inv).
      Size 4 registers one candidate per tier, so it has nothing to rank.
      Every registered power-of-two size is now audited.

---

## 5. The mixed-radix engine

Still the weak link against FFTW3 despite the −30% rounds in §1.7: 44100 sits
at 786 µs against 185 µs, and the non-power-of-two geomean is 0.60× where
powers of two now run 1.36× (§1.15). This is the whole of the remaining
external gap. The fused stage kernels closed the dispatch and butterfly costs;
what remains is the odd-radix arithmetic itself and two loose ends.

- [ ] **Give `Butterfly11` the conjugate-pair form.**
      `kernels.Butterfly11ForwardComplex64` and its three siblings evaluate the
      full 11×11 DFT matrix — the only radix in the set still doing O(r²)
      complex multiplies, where radix 3/5/7 all have hand-written butterflies.
      The fused AVX2 kernel now sidesteps it, so this is dead weight on amd64,
      but it is still what runs every radix-11 stage on SSE2, NEON, WASM and
      purego, and what the fused kernels' own Go tails call. A throwaway
      pair-form implementation measured 113.6 → 72.1 ns against it (−37%)
      without being tuned; the derivation and the index tables are written out
      in the header of `avx2_f32_mixedradix_stage11.s`. Cheap,
      arch-independent, and the registry-driven reference tests already cover
      it.
- [ ] **Re-derive the radix-7/11 win gates over a wider range.**
      `mixedRadix7And11Wins` and `rader7Or11Wins` were both fitted on the shapes
      measured at the time; the 44100 result showed at least one extrapolation
      failing outside that range. Re-run
      `BenchmarkMixedRadix7And11VsBluestein` with the practical lengths included
      and check whether the "power-of-two part ≥ 8" rule holds at large n or
      needs an n-dependent term. Depends on §3's routing-report item — the gate
      re-derivation needs to know which route each length actually took.
- [ ] **Explain the +6.8%/+4.4% regression at n = 768.** Left open by the
      leaf-hoist round. 768 = `[3 256]` has 3 leaves so no win was available,
      but the loss reproduced across three independent builds. Ruled out:
      allocations (0 B/op both sides) and the added hook parameter (a variant
      carrying the signature change but keeping the per-node lookup is neutral
      there). Remaining candidates are the `len(radices) == 1` guard and code
      layout, and they could not be separated because each variant is a
      different binary. A `perf stat` comparison (branch misses, I-cache) on a
      quiet machine would settle it without needing a third build.
- [ ] **A radix-r stage kernel that keeps the streams in registers across both
      the multiply and the butterfly.** The fused kernels already do this for
      the stage as a whole; what is left is the observation from the
      vectorization round that even after fusion, beating the scalar path
      properly at the smaller spans needs the r streams held across the whole
      stage rather than two passes over memory. Lower priority than the two
      items above — it is the last few percent of a path that has already
      absorbed −30%.

---

## 6. Coverage on other ISAs

- [ ] **NEON priority tuning on real arm64 hardware.** The size-specific ladder
      now runs 4 → 32768 in both precisions, but every priority from 512 up was
      **mirrored from smaller sizes, not measured** — QEMU timing is meaningless
      and CI has no native runner. Above ~8192 the DIT codelets also compete
      with the Go six-step path on real hardware (cache behavior differs from
      QEMU), so measure before trusting the 24/28 priorities there. Needs Apple
      Silicon / Graviton, or the native ARM64 CI runner on the community
      backlog. This supersedes the older "NEON sizes 512+" item: the kernels
      exist; only the tuning is blocked.
- [ ] **ARMv8.3 FCMLA complex-arithmetic kernels.** `internal/cpu` detects only
      `HasNEON`; ARMv8.3's `FCMLA`/`FCADD` do a full complex multiply-accumulate
      in two instructions against 4 mul + 2 add today
      (`internal/asm/arm64/neon_complex_mul.s`). Add `HasFCMA` detection (HWCAP
      on Linux, sysctl on darwin), an `FCMLA` variant of the generic NEON
      butterfly, and runtime-dispatch it above plain NEON. Apple Silicon and
      Neoverse both support it. Blocked for benchmarking on the same hardware
      item as above.
- [ ] **AVX-512 higher-radix / per-size-tuned variants.** The shipped tier is
      generic radix-2; a radix-4 AVX-512 kernel should widen the 1.2–2.4× gap
      and could reclaim size 2048 and the complex128 sizes where AVX2 codelets
      still win.

      **No longer blocked on hardware.** The Xeon Gold 5218 is reachable and the
      AVX-512 assembly ran there for the first time in 2026-07 — until then
      every AVX-512 test had been skipping at runtime, so
      `internal/asm/amd64/avx512_f{32,64}_generic.s` had **never executed**, on
      any machine. The whole AVX-512 test set passes with zero skips: the
      assembly is correct, and what follows is purely a tuning question.

      Measured against the best AVX2 codelet at each registered size (complex64,
      pinned, idle host):

      | size  | AVX-512 fwd | best AVX2 fwd | fwd Δ  | AVX-512 inv | best AVX2 inv | inv Δ     |
      | ----- | ----------- | ------------- | ------ | ----------- | ------------- | --------- |
      | 1024  | 9151 ns     | 8210 ns       | +11.5% | 10662 ns    | 10141 ns      | +5.1%     |
      | 4096  | 40786 ns    | 39726 ns      | +2.7%  | 45995 ns    | 50651 ns      | **−9.2%** |
      | 8192  | 89129 ns    | 83269 ns      | +7.0%  | 96315 ns    | 102567 ns     | **−6.1%** |
      | 16384 | 199838 ns   | 188084 ns     | +6.2%  | 221941 ns   | 233577 ns     | **−5.0%** |

      Three things follow. (1) The AVX-512 codelets are registered at
      **Priority 10** against 24–28 for AVX2, so the registry never selects them
      even on an AVX-512 CPU — for _forward_ that is currently the right call,
      but it discards a real 5–9% on _inverse_ at ≥ 4096. (2) The AVX-512
      codelet is **radix-2** while every AVX2 winner here is **radix-4**, so
      this table measures an algorithm gap, not a vector width gap — which is
      exactly what the radix-4 work is for, and raises the prior that it will
      pay off. (3) Coverage is complex64 only, at 1024/4096/8192/16384;
      `cmd/gencodelets/specs.go` has **no `Target: "avx512"` rows for
      complex128** at all. Note the numbers above predate the 256-bit AVX2
      radix-4 kernels, which moved the AVX2 column substantially — re-measure
      before acting. Do not retune priorities from this host alone (§2.3).

- [ ] **Measure the packed-Stockham crossover on the SSE, NEON and AVX-512
      tiers.** §1.14 filled in only the AVX2 row of
      `packedStockhamMinSize`; the other three are `packedOff`, i.e. those tiers
      keep today's behaviour and forgo a win worth up to 2.7× on AVX2. Their
      uncovered range is **one octave wider**, since their codelet ladder stops
      at 32768 against AVX2's 65536. The harness is already in place and needs
      no porting: run `BenchmarkPackedGate64`/`128` on the target host and read
      the median within-round packed/kernel ratio. Do not extrapolate the AVX2
      thresholds — the competing kernel is a different one on each tier, which
      is the whole reason the table has a tier axis.
- [ ] **Validate the SSE2/SSE3 tier on genuine SSE-only hardware.** All the
      2048/4096/8192/16384/32768 measurements forced the SSE path on an
      AVX2-capable i7-1255U. Spot-check the speedups — and the
      DIT-vs-six-step crossover — on a real pre-AVX2 machine or a VM with AVX
      masked before calling the tier done. (Not planned: a complex64 tier for
      SSE2-without-SSE3 hardware — the complex multiply idiom needs `ADDSUBPS`,
      and SSE3 has been universal since ~2005; such machines keep the generic Go
      path.)
- [ ] **WASM SIMD** — blocked on toolchain. Go's `GOEXPERIMENT=simd`
      intrinsics (golang/go#73787) reached amd64 in Go 1.26 and Wasm/ARM64 in
      the 1.27 RC, but remain experimental and this module targets Go 1.25.
      Revisit when the experiment graduates or the toolchain floor moves.

---

## 7. Throughput and scale

Opt-in parallelism and layout work. All of it keeps the single-threaded,
zero-allocation default.

- [ ] **Parallel batch execution.** `Plan.ForwardBatch`/`InverseBatch` run
      count transforms sequentially. Add `PlanOptions.Parallel`/`MaxWorkers`
      (default 1 = today's behavior): the batch loop fans out over a
      pre-created worker set with per-worker scratch from the existing
      resident-cache pattern, preserving zero-alloc-in-steady-state. Batch is
      embarrassingly parallel — the highest-value, lowest-risk parallel item.
- [ ] **Parallel 2D/3D/ND row-column passes.** Each axis pass is an independent
      batch of 1D transforms over rows/columns — reuse the worker
      infrastructure above. The transpose/gather steps stay serial initially.
      Gate on plan size (parallelism below ~256×256 is overhead-dominated);
      verify with the existing `-race` concurrent tests plus new
      parallel-enabled ones.
- [ ] **Parallel six-step for very large 1D.** Six-step is already a
      (transpose, batch-FFT, twiddle, transpose, batch-FFT) pipeline; run the
      inner batch-FFT stages on the worker pool for n ≳ 2²⁰. Depends on the
      cache-blocked transpose so the serial transpose doesn't dominate.
- [ ] **SoA (split real/imag) layout exploration.** Prototype internal SoA for
      one kernel family (e.g. the AVX-512 generic path, which currently spends
      shuffle uops de-interleaving) and measure; decide whether a v2 `PlanSoA`
      API is warranted before designing it.
- [ ] **SIMD 8×8 complex tile kernel for the transpose** (AVX2
      `VPERM2F128`/`VUNPCK`, NEON `TRN1`/`TRN2`). The tiled walk removed the
      index table and its O(n²) cache; this is the remaining constant factor,
      and it is also what the SIMD transpose kernels stopping at 128×128 cost
      the six-step path.
- [ ] **SIMD row FFTs inside four-step.** The rows are contiguous, but the row
      passes still use the scalar Stockham butterflies — the main handicap
      against the monolithic kernels.

---

## 8. DSP layer

- [ ] **Buffer reuse in one-shot DSP helpers.** The one-shots allocate 5
      temporaries per call; route them through the pooled resident-cache scratch
      (as `Convolver` already does) so casual users get most of the steady-state
      performance without switching types.
- [ ] **Overlap-add/overlap-save streaming convolution.** For long-signal /
      short-kernel filtering (`len(b) ≪ len(a)`), one big FFT is asymptotically
      worse than block convolution with a plan of size ~4×len(b). Add
      `StreamingConvolver` (fixed kernel, chunked input) — both an API feature
      and the standard algorithmic fix for the current "FFT the whole signal"
      cost profile.
- [ ] **Let recursive leaves use prepared-twiddle codelets.** `leafCodelet`
      declines any codelet declaring `TwiddleSize`/`PrepareTwiddle` and falls
      back to the generic DIT, which costs the best leaf on some size/precision
      pairs — on this laptop, complex128 at n = 256. Binding them needs per-leaf
      forward _and_ inverse tables built at plan time, since `PrepareTwiddle`
      takes an `inverse` flag while the recursive executor shares one table
      across both directions. Worth measuring before building: the leaf is one
      of two levels, so the ceiling is modest.
- [ ] **Real-input Bluestein.** Exploit conjugate symmetry in the padded
      convolution to close the ~2× gap against a hypothetical packed
      odd-length method. The 2D/3D real plans also still require even width
      (their row/column packing is a separate piece of work).

---

## 9. Ship v1.0

The API-shape work that actually gated the tag landed in §1.2 and the
correctness debt closed in §3, so nothing here is blocked on code that changes
a signature. What this section now waits on is §4–§8: none of it changes the
API, but the mixed-radix gap against FFTW3 (§5) and the open power-of-two soft
spots (§4) are wide enough that tagging over them would ship a v1.0 whose
performance story needs an asterisk.

- [ ] **Tag `v1.0.0`** with GitHub release notes. Issue/PR templates are in
      place.
- [ ] **Put an external comparison in the release checklist.** Every finding in
      §1.6–§1.9 was invisible to the internal suite, because "faster than last
      week" and "faster than FFTW3" are different questions, and only the second
      notices that a whole class of lengths never got the attention the
      power-of-two ladder did. Running `go-fft-bench` before a tag — even
      manually, even on one laptop — is cheap next to shipping another release
      in which 44100 loses to gonum. The harness refuses to start on a loaded
      machine, so the results are at least not accidentally measuring a compile
      storm.

---

## 10. Post-v1.0 future

**Features**: DCT, Hilbert transform, STFT/spectrograms, audio/image examples,
Gonum ecosystem integration, optional GPU backends (kept out of the pure-Go
core).

**Community**: `CODE_OF_CONDUCT.md`, Dependabot, native ARM64 CI runner
(unblocks the NEON benchmarking items in §6).

**Explicitly kept as-is** (reviewed, deliberate): the benchmark-cited selection
thresholds in `internal/planner` (compile-time constants are fine
pre-wisdom-tuning), the asm build-tag triples, the wisdom cache design, the
root-package black-box test strategy, and hand-written SIMD assembly.

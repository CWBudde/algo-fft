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

**Where the library sits against others** (i7-1255U, `go-fft-bench` @ `a1fa607`,
v0.7.0 baseline): powers of two are 0.63× FFTW3 by geomean and ~8× the rest of
the Go field. Non-power-of-two lengths were far weaker and are where most of
the 2026-07 work went — 44100 has gone from losing to gonum (4.00 ms vs
2.59 ms) to 781 µs, against FFTW3's 236 µs. The remaining gap is concentrated
in the mixed-radix engine (§5) and a handful of power-of-two soft spots (§4).

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

44100 is now 781 µs against FFTW3's 236 µs; the gap has closed from 8.0× to
3.3×.

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

- [ ] **The six-step row FFTs are still on the pre-§1.9 128-point kernel.**
      Found while closing the item above, and it is the same fix again: six call
      sites in `dit_16384_sixstep_amd64_avx2.go` (lines 50/64/112/126) and
      `dit_8192_sixstep_64x128_amd64_avx2.go` (83/162) call
      `{Forward,Inverse}AVX2Size128Radix4Then2Complex64Asm` for their row
      transforms — the XMM-width kernel at 320 ns where the generic one is 88.
      Harder than the 384 case only in that the row FFTs run in-place with a
      per-row twiddle table, so check the prepared table can be hoisted the same
      way before assuming the win transfers. These two files are also the last
      non-test callers of that `.s` file apart from the `KernelStrategy`
      dispatch below; clearing all three is what lets it be deleted per §1.9.
- [ ] **Re-measure the plan-level c64/c128 ratio and the FFTW comparison.** The
      c64/c128 ratio ran 1.10, 1.02, 1.14, 1.07, 1.08 across 1024–16384 against
      1.6–2.1× at 256/512 — because every codelet serving that band was
      XMM-width, so both precisions moved the same 128 bits per instruction and
      a ratio near 1.0 is exactly what that predicts. At the codelet level it
      has now moved to 1.64, 2.09, 2.43, 2.27, 1.85. **Still to do:** re-measure
      the plan-level ratio and the external comparison, which is what this
      actually tracks — the codelet is only one part of that path.
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
- [ ] **The n = 2048 local minimum.** 0.29× FFTW3 forward, 0.31× inverse — the
      worst power-of-two point in the sweep, with 1024 (0.43×) and 4096 (0.45×)
      either side of it. Re-check against the new radix-4 kernel before
      investigating further; the AVX-512 item in §6 mentions reclaiming 2048,
      but this is the AVX2 tier and independent of it.
- [ ] **Make the 10 remaining mixed functions uniformly VEX.** The sweep left
      `Forward/InverseAVX2Complex64Asm`, both `AVX2Stockham` pairs and the
      `Size1024Radix32x32` pair in both precisions mixed, because each has a
      legacy write whose upper half is live. Given that partial mixing is worth
      up to 152×, these are worth restructuring (renumber the aliased register
      so `Xn`/`Yn` no longer collide, then convert). Suggestive:
      `dit1024_radix32x32_avx2` measures 7.1 µs against
      `dit1024_radix4_avx2` at 3.5 µs — it may already be paying a mixing
      penalty.
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
- [ ] **Finish the incumbent audit.** §1.8 confirmed n = 256/512/8192. Still
      unaudited: 4–128, 1024, 2048, 4096, 16384, 32768. Nothing suggests a
      mis-tuned incumbent is likely there, but nothing rules it out either — and
      after the radix-4 round every one of those sizes has a new candidate. The
      sweep is a `just bench-gated <sizes>` away.

---

## 5. The mixed-radix engine

Still the weak link against FFTW3 despite the −30% rounds in §1.7: 44100 sits
at 781 µs against 236 µs. The fused stage kernels closed the dispatch and
butterfly costs; what remains is the odd-radix arithmetic itself and two loose
ends.

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

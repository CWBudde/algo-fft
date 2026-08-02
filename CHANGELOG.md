# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **The race-detector test run could not finish, and now takes a tenth as
  long.** `internal/kernels` took **1499.7 s** under `-race` against Go's
  10-minute default timeout, which neither `just test` nor any CI workflow
  overrode — so that gate was red repo-wide, independently of any kernel
  change. Measured per-test under `-race` (not extrapolated from an untagged
  profile, which ranks tests the race build already skips), the cost was
  fourteen naive O(n²) reference DFTs: six AVX2 sweeps at ~207 s each and eight
  fixed-size DIT tests at 100–152 s, with nothing else above 40 s. The
  reference grows as n² while the kernels it checks grow as n·log n. The
  in-place tests now use the out-of-place kernel as their oracle and assert
  bit-for-bit equality — cheaper _and_ stronger than a naive DFT with a
  tolerance wide enough to accept a real aliasing defect as rounding — and the
  reference sweeps cap at n ≤ 4096 under `-race` only, keeping both kernel
  shapes and logging what they drop. **1499.7 s → 147.8 s.** `-timeout=20m` is
  in `justfile` and the `-race` CI steps as a backstop, not as the fix.

### Added

- **Split-radix is registered, and measured for the first time.** Sixteen
  `dit<N>_splitradix_generic` rows at 256…32768, both precisions, at a
  deliberately non-selectable priority 1. The rows exist so the family can be
  _ranked_: the canary-gated harness only sees registered candidates, so having
  no rows was both the finding and the obstacle. They also bring the kernel
  under the registry-driven reference and zero-alloc sweeps, which it had never
  been subject to. **No routing changes** — every size below 65536 already has a
  faster registered codelet, and the ladder deliberately stops at 32768 because
  n = 65536 has no generic row at all, where a new row would silently become the
  selected pure-Go route. Split-radix loses all sixty-four measured cells
  (1.06–1.44× on two hosts); see `docs/CODELET_BENCHMARKS.md`.
- **Direct tests for the n = 16384 six-step codelet**, which its 4096 and 8192
  siblings always had and it never did. The gap was invisible while a spec row
  kept the registry sweep reaching it.

### Changed

- **The three six-step codelets moved behind `-tags fftprobe`.** They lose all
  twenty-four cells on two hosts (1.43–2.35× against the pure-Go radix-8
  ladder), and the comparison is fair — these codelets call the tuned radix-4
  leaves, so both arms are scalar. They stay compiled, correctness-tested and
  reachable by wisdom, out of every production build. This retires three
  codelets, **not** the six-step decomposition: the separate strategy kernel is
  a different implementation whose loss is confounded by a known defect.
- **`dit1024_radix32x32_generic` and `dit512_radix16x32_generic` demoted to
  priority 1** (from 25 and 35). Neither was selectable before, but both read as
  contenders while losing every cell on both hosts. Not probe-gated: no laptop
  cell of either row reaches the 1.5× bar, and the forward/inverse asymmetry
  (1.26× forward against 1.79× inverse for 32×32) names an inverse-path defect
  worth finding rather than a decomposition worth retiring.

- **`TestRadix4AVX2MatchesStockham`, a large-size cross-check for complex64.**
  The complex128 kernel had been cross-validated against Stockham — an
  independent algorithm with no shared permutation table — at 8192…65536 for
  some time; the complex64 kernel had no large-size cross-check at all. It now
  does, in every build, which is also what backs the `-race` reference cap
  above.

### Removed

- **`KernelStrategy` `KernelEightStep` (breaking).** It was never a real
  eighth step: `internal/kernels/eightstep.go` was `sixstep.go` with the
  names changed — same perfect-square rejection, same two
  `TransposeSquare`-bracketed Stockham row passes — so it duplicated
  `KernelSixStep` rather than implementing a distinct algorithm
  (`docs/CODELET_BENCHMARKS.md`, "Eight-step is six-step"). The enum value 4
  is retired and intentionally not reused, so every surviving
  `KernelStrategy` constant (`KernelBluestein` and after) keeps the number it
  already had — `strategy_numbering_test.go` pins the exact values. Wisdom
  files that persisted the strategy name `eightstep` will fail to resolve on
  reload, the same behavior as any other unknown wisdom name; use
  `KernelSixStep`, which already implements what `KernelEightStep` ran.
- **22 shadowed AVX2 codelet rows.** With the incumbent audit complete, every
  registered power-of-two size has a canary-gated ranking, and it showed a large
  tail of AVX2 candidates that the registry could never select on any amd64 CPU
  — several losing to their own SSE2/SSE3 twin, and `dit16_radix2_avx2`
  (complex128) losing to pure Go. Everything measured above **1.5× its size's
  winner** is gone: the AVX2 tier went from 33 + 27 to 20 + 18 registrations and
  the inventory total from 229 to 207. The rows fall into three groups and only
  one of them is a code defect — `*_radix2_avx2` at n >= 16 is structurally
  dominated (radix-2 makes twice radix-4's passes, and at complex128 a 256-bit
  register holds two elements, so there is no width left to recover it); the
  higher-radix rows (`radix8`, `radix16`, `radix16x32`, `radix32x32`) are
  partially-vectorised and _do_ leave a hypothesis untested, now tracked as a
  size-generic AVX2 radix-8 stage in `PLAN.md` §4; and the `sixstep` rows were a
  stale crossover that the faster radix-4 kernels moved out from under. The
  six-step kernels themselves stay for the forced-`KernelSixStep` route.
  Rankings and reasoning: `docs/CODELET_BENCHMARKS.md`.
- **Four dead assembly files, 4,854 lines** — `avx2_f{32,64}_size1024_radix32x32.s`,
  `avx2_f64_size512_radix8.s`, `avx2_f64_size256_radix16.s` — whose only
  remaining callers were tests, together with `internal/kernels/params_avx2.go`
  (the twiddle-preparation helpers for those kernels, left reachable only by
  their own tests) and the two test files that exercised them. amd64 assembly
  files: 141 → 137. The assembly behind the other pruned rows cannot follow yet:
  19,815 lines of it are still reached from the forced-`KernelStrategy`
  dispatch, which is now tracked as the gate on ~39,000 lines rather than on
  ~19,000.

### Changed

- **Documentation restructured; `PLAN.md` went from 1845 to ~900 lines.** Two
  thirds of it was completed round-by-round history, which had started to crowd
  out the open work and to carry stale facts. The post-mortems moved into
  topic-scoped documents and `PLAN.md` §1 became a one-line-per-round ledger
  linking to them: new `docs/AVX2_RADIX4.md` (the 256-bit radix-4 kernels and
  the `n = 2·4^k` radix-2 tail), `docs/MIXED_RADIX.md` (mixed-radix engine,
  Bluestein pad model, Rader gates), `docs/TESTING.md` (the test-vector
  blindness audit), and `docs/AVX512_CODELETS.md` renamed to
  `docs/CODELET_BENCHMARKS.md` and generalised into the standing evidence file
  behind every `Priority` in `cmd/gencodelets/specs.go`. Accuracy findings were
  appended to `docs/PRECISION.md` and the reusable assembly lessons to
  `AGENTS.md`. Four superseded documents gained archive banners, one of them
  (`docs/REAL_FFT_STATUS.md`) because it claimed float64 real-FFT support was
  missing when `NewPlanReal64` has existed for some time. `README.md`'s
  performance section was replaced — it quoted "Size 64: ~490 ns/op" against a
  measured 54 ns codelet, and a gonum comparison table predating the whole
  2026-07 round.
- **`dit8_radix8_avx2` replaces `dit8_radix4_avx2` as the complex128 size-8
  codelet.** The last unaudited sizes (8/16/32/64 and 16384, both precisions)
  were swept canary-gated — 159 of 160 groups accepted, 0 drift — closing the
  incumbent audit for every registered power-of-two size. Eight of the nine
  incumbents were confirmed, most by 1.2–10.8× over the next candidate. The
  ninth was mis-tuned: radix-8 beats the registered radix-4 row at **0.970
  forward / 0.859 inverse**, medianed over 16 groups. In absolute terms the
  inverse gap is 8.2 ns → 7.0 ns and the forward gap is 0.2 ns, so at plan
  level the ~100 ns per-call dispatch swamps it entirely; the row is corrected
  because the registry should record what measurement says, not because
  callers will notice. See docs/CODELET_BENCHMARKS.md.

- **Two Stockham cross-check tolerances were nearly blind and are now
  calibrated.** Both tests passed, so a green run said nothing about what they
  would reject. The complex64 bound (`2e-4*n`, copied from the neighbouring
  naive-DFT tests) sat 4.3e4–9.1e4× above the measured agreement, and the
  pre-existing complex128 bound (`1e-9*n`) sat 6.2e7–1.5e8× above it — loose
  enough to accept a kernel wrong in the fifth significant digit. Both now
  scale as `sqrt(n)`, which is how two O(n·log n) implementations actually
  diverge, and clear their measured agreement by ~8–10×. The `2e-4*n`
  convention remains correct where it originated, against the naive DFT, which
  is itself the inaccurate side of that comparison.

- **The `n = 2*4^k` radix-2 tail can now be fused into the last radix-4 stage,
  and three sizes take it.** For lengths that are twice a power of four, the
  size-generic AVX2 radix-4 kernel transformed the even and odd halves
  independently and then made a _separate full pass_ over the buffer to combine
  them. Measured against a probe that skips that pass outright, it costs
  roughly **8–15% of the kernel** at those sizes (up to 20% at n = 128) — a tax
  on every odd power of two, and the reason n = 2048 costs six passes for eleven
  butterfly levels where 1024 costs five for ten. The last stage always has exactly two groups,
  so running them in lockstep leaves both operands of four radix-2 butterflies
  in registers and the extra pass disappears; output addresses, the permutation
  table and the packed twiddle layout are unchanged. Fusing is **not** a
  uniform win, because it doubles the live streams from four to eight: it gains
  4–6% at n = 128 in both precisions and at n = 2048 complex64, and _loses_ up
  to 11% at n = 2048 complex128, where the last-stage stride is exactly 4 KiB
  and all eight streams land on one L1 set. It is therefore selected per size
  in `cmd/gencodelets/specs.go` — `dit128_radix4fused_avx2` in both precisions
  and `dit2048_radix4fused_avx2` at complex64 — rather than applied wherever the
  shape allows. Correctness is cheap to state and was checked that way: fusing
  reorders no arithmetic, so the fused and separate-tail kernels must agree
  **bit for bit**, and they do at every size 16…32768 in both directions.
  Sweeps are canary-gated and pinned; see docs/AVX2_RADIX4.md for the full table and
  the `-tags fftprobe` harness that re-derives it. Also discharges the incumbent
  audit at 128/512/1024/2048/4096/8192/32768 in both precisions: every incumbent
  confirmed except the three rows this change itself replaced.

- **The last ten mixed VEX/legacy-SSE assembly functions are now uniformly
  VEX-encoded.** The 2026-07-28 sweep converted 4089 instructions across 59
  files but skipped `Forward`/`InverseAVX2Complex64Asm`, both `AVX2Stockham`
  pairs and the `Size1024Radix32x32` pair in both precisions, on the belief that
  each held a legacy write whose aliased `Yn` upper half was live. Re-reading all
  ten showed that is not the case: every YMM is fully redefined by a 256-bit VEX
  write at the head of each vector-loop iteration, and every path from a legacy
  block back into vector code passes through such a redefinition. The legacy
  blocks are scalar remainder loops that run after the vector work, so the
  registers they alias are dead. Of 730 legacy instructions, 611 were converted
  and 119 deleted as unreachable (below); no register renumbering was needed —
  and none was available, since the Stockham scalar cores occupy `X0`–`X13`
  against `Y0`–`Y7` of VEX code. Verified at the machine-code level: all 9967
  symbols across the `internal/kernels` and `internal/fft` test binaries decode
  identically to the previous commit under a per-symbol binutils-`objdump`
  normalizer, and `cmd/measure_correctness` is bit-identical at every size and
  both precisions. **No performance change** (0.98–1.01× over eight interleaved
  pinned rounds), which the byte-equivalence implies — the hypothesis that the
  size-1024 radix-32×32 codelet's poor showing was a mixing penalty was wrong.
  Measured within one binary, that codelet is the slowest candidate at its size
  in both directions (8.2×/9.9× `dit1024_radix4_avx2`, and 11% slower than its
  own pure-Go twin) because only one of its two stages is vectorised. It is
  already shadowed in the registry; removing it is tracked as a follow-up.

### Removed

- **Dead assembly in the size-1024 radix-32×32 kernels.** The complex128
  forward carried a complete 221-line scalar `fwd_fft32` helper that no branch
  reached — that function vectorises stage 2 as well, which orphaned it — plus
  two inert return labels. The complex64 inverse carried a
  `MOVSS ·scale1024f32<>(SB), X14` whose value is never read. 235 lines removed
  with no behaviour change.

## [0.7.4] - 2026-07-29

Three fast paths that existed, were correct, and were unreachable, plus the
size-384 codelet's assembly finally being called (and, once called, found to be
wrong). No API change.

### Fixed

- **The Bluestein sub-FFT never reached the codelet registry.** A power-of-two
  Bluestein pad went through a hardcoded size switch in `internal/kernels/dit.go`
  that consults no registry and carries no build tags, so at those lengths the
  default build and `-tags purego` ran the identical pure-Go kernel for ~96% of
  the work — the default build measured ~4% _slower_, since it paid the chirp
  modulation's SIMD call overhead for no benefit. The padded sub-FFT is now
  bound at plan time (`newBluesteinSubFFT`, `fft.BluesteinSubFFT`), through
  `planner.EstimatePlan` plus `prepareCodeletTwiddles` — a codelet may need a
  prepared twiddle layout, which is plan-time work a call-time fix could not
  have done. Binding happens only where the registry actually has a codelet:
  falling back to the strategy-dispatched kernels traded a hand-tuned
  radix-4-then-2 for plain Stockham and cost ~4% on `purego`. Measured on an
  i7-1255U: complex64 n = 1009 **5.5×** forward / 5.1× inverse, n = 2003
  **7.2× / 7.0×**; complex128 2.9× and 4.1× forward; `purego` flat at 1009 and
  15–18% faster at 2003. n = 9973 (a non-power-of-two pad, so untouched) is
  unchanged at 1.00, which is what makes the rest readable. Rader still passes
  `nil` — its length-(n−1) convolution takes the unbound route when n−1 is a
  power of two.
- **The packed Stockham engine was compiled out of every SIMD build.**
  `stockhamPackedEnabled = false` on amd64/arm64/386, justified by "the
  hand-written codelet path is checked first and supersedes it" — but every
  registered codelet carries `Algorithm: KernelDIT`, so the strategy test
  upstream had already excluded packed wherever a codelet binds. The two were
  never in competition; what the constant actually suppressed was the sizes with
  _no_ codelet, where the build fell through to a radix-2 Stockham kernel while
  the radix-4 route it had disabled was up to 2.7× faster. It also suppressed
  an explicitly forced `KernelStockham` at any size ≥ 4. Replaced by a runtime
  tier × precision threshold table (`internal/transform/stockham_packed_policy.go`):
  packed is taken from 2^17 at complex128 and 2^20 at complex64 on AVX2, the
  sizes where it measured 1.20–1.49× ahead. Other tiers stay off until they can
  be measured on their own hardware. Related: `ComputePackedTwiddles` grew its
  `Values` slice from zero capacity, ending at up to ~2× the needed allocation;
  it now preallocates exactly via the new `PackedTwiddleLen`.
- **`ApplyTwiddle384Complex64Asm` produced wrong results for a third of the
  transform.** It built the strided `twiddle[2k]` vector with `VINSERTPS $0x10`,
  which moves a single float32: it overwrote the imaginary part of each even
  twiddle and left two lanes undefined. The fix is `VMOVLHPS`, which moves a
  whole 64-bit lane. The bug had never been observable because the helper was
  declared-but-uncalled — the complex64 size-384 codelet did its radix-3 column
  DFT and twiddle multiply in scalar Go while its complex128 twin called
  assembly for both. Direct tests of all four twiddle helpers against the scalar
  loops they replace now exist (`dit_384_asm_helpers_amd64_test.go`).

### Changed

- The complex64 size-384 codelet now calls the assembly its complex128 twin
  already used, precomputes the 128-point sub-twiddle at package load
  (`W_128^k == W_384^(3k)`) and pools its buffers: **−30% forward, −31%
  inverse**. New `ApplyConjTwiddle384Complex{64,128}Asm` for the inverse
  direction — conjugating the twiddle costs one xor, while conjugating the
  product does not work against `VFMADDSUB`'s fixed sign pattern.
- The size-384 decomposition's 128-point sub-FFT is bound to the size-generic
  radix-4 kernel of v0.7.3 (128 = 2·4³, so it runs radix-4 stages to 64 and
  combines with a radix-2 tail). Both precisions had been on superseded kernels
  — complex128 on plain radix-2, complex64 on the pre-v0.7.3 XMM-width kernel at
  320 ns against the generic one's 88. Plan-level: **−60% / −65%** complex64
  forward/inverse, **−60% / −57%** complex128, geomean −58%, zero-alloc
  preserved.
- The six-step row FFTs at n = 8192 and 16384 likewise moved off the XMM-width
  128-point kernel. Their length-128 row twiddles were being gathered out of the
  caller's length-n table on _every_ transform (four such loops across the two
  files); two package-load tables replace them, so the swap removes per-call
  work rather than adding a table. complex64: **−50% / −54%** at 16384
  forward/inverse, **−21% / −30%** at 8192. This lands on the forced
  `KernelSixStep` route — the registry already prefers the generic radix-4
  codelet at these sizes.
- Four `{Forward,Inverse}AVX2Size384Mixed{Complex64,Complex128}Asm` symbols were
  deleted: they were named like kernels and had no callers, and their bodies
  only length-checked their arguments and returned true.
- `TestRadix4AVX2Ranking` re-measures before failing (`rankingAttempts = 3`).
  Speeding up the six-step tightened the headroom the test grants — its
  tolerance is relative to the runner-up, not absolute — so a contended window
  could read as a regression. A real regression reproduces on every pass.

### Added

- `KernelMixedRadix`, so a plan reports the route that actually executes.
  The kernel dispatch checks the length before the strategy switch and takes
  the mixed-radix engine unconditionally at non-power-of-two lengths, while
  `KernelStrategy()`/`Algorithm()` carried whatever had been requested. A
  forced strategy the dispatch cannot honour at that length now resolves to the
  route that runs rather than being echoed back.
- Broadband-signal reference tests across the strategy matrix, Bluestein
  convolution, clone/pool and recursive paths. An impulse cannot detect a wrong
  twiddle (they all multiply zeros) or a wrong output ordering (its spectrum is
  all-ones), and Parseval and linearity are insensitive to both — that
  combination had hidden a wrong-answer bug at every size ≥ 1024 for a whole
  precision.
- A WASM demo (`examples/wasm-demo`): canvas rendering with phase-to-hue
  mapping, synthetic waveform generation, window functions, and a
  complex64/complex128 round-trip error comparison.

## [0.7.3] - 2026-07-28

### Added

- A generic AVX2 radix-4 DIT kernel for complex128, mirroring the complex64
  kernel that already existed. One kernel now covers every power-of-two length
  from 32 to 65536, sharing the shape handling (`radix4AVX2Limit`), the stage-1
  permutation table (`radix4GroupIndices`) and the three-plane packed twiddle
  layout with the complex64 side — those are properties of n alone, not of the
  element type. The twiddle table is generated at complex128 rather than
  widened from the complex64 one, so the factors are computed in double
  precision throughout. A YMM holds two complex128 instead of four complex64,
  so the kernel retires half as many butterflies per instruction; everything
  else, including the fused permutation and the folded 1/n, carries over.

### Removed

- The eight generated per-size complex128 AVX2 codelets (`size128`, `size256`,
  `size1024`, `size2048`, `size4096`, `size8192`, `size16384`, `size32768`),
  superseded by the generic kernel above. This is ~10.8k lines of unrolled
  assembly replaced by ~1.2k, and it extends AVX2 coverage to 65536, which no
  per-size codelet reached. No public API is affected: these were internal
  registry entries selected by the planner, never named by callers.

### Fixed

- `PlannerMeasure` and the deeper planner modes could return a _worse_ plan
  than `PlannerEstimate`. They benchmarked kernel strategies only and then
  applied the winner through a path that kept a codelet just when the winning
  strategy happened to match the codelet's own algorithm. At complex64
  n = 1024 the Stockham kernel beats the DIT kernel, so measuring discarded the
  `dit1024_radix4_avx2` codelet that the unmeasured path uses. Codelets are now
  candidates in their own right, so a kernel strategy can only win after
  actually beating them.
- The measuring and forced-strategy planner path built its plan estimate
  without the codelet's twiddle-preparation callbacks. A codelet wanting a
  packed twiddle layout was handed the plain table, failed its own length check
  and silently ran the fallback kernel while the plan still reported the
  codelet's signature — `dit8192_radix4_then2_params_avx2` under
  `PlannerMeasure` on AVX2. Results were never wrong, only slower than
  advertised.
- The mixed-radix codelet dispatch re-paid its setup at every recursion node.
  A CPU profile at n = 1000 (complex64, AVX2) found only **1.9%** of runtime in
  the codelet assembly and roughly 40% in dispatch overhead. Three causes, all
  fixed:
  - `cpu.DetectFeatures` took an RWMutex _and_ an exclusive Mutex per call
    (13% of runtime). Features are now published in an `atomic.Pointer`, so a
    cached read is two atomic loads and no lock.
  - `registry.Lookup` took an RWMutex per call (12%). The registry's size map
    is now copy-on-write behind an `atomic.Pointer` — writers are init-time
    only — making lookups lock-free. This also fixes a latent hazard: the
    `*CodeletEntry` returned by `Lookup` used to point into a slice that a
    later `Register` could sort or reallocate in place.
  - Each leaf gathered a twiddle table into a pooled buffer (15% in
    `sync.Pool` traffic) and then discarded it whenever the codelet declared a
    prepared layout. The prepared-layout check now runs first, and the gather
    is gone: the recursion keeps `n*step == len(twiddle)`, so
    `twiddle[i*step] == W_n^i` — the gather always rebuilt the standard size-n
    table, which is now cached by size.

  Forward mixed-radix transforms measured 4–28% faster at the lengths P5.1
  calls out (96, 448, 480, 704, 768, 1000), geomean **−15.0%** across both
  precisions on an i7-1255U (interleaved arms, 6 rounds). Lengths whose
  schedule has no codelet leaf are unaffected. Results are bit-identical except
  for leaf twiddles, which are now computed directly at size n rather than
  subsampled from the size-N table — a last-ulp difference, if any, in the more
  accurate direction.

- The AVX2 mixed-radix drivers dispatched **every** sub-transform larger than
  one point to a codelet, including the size-2/3/4/5 recursion leaves that have
  a pure-Go butterfly. The scheduler has always required `n > 5` before emitting
  a codelet-backed radix, so any schedule ending in a small radix paid a codelet
  call, a strided twiddle gather and two `sync.Pool` round-trips per leaf to do
  a handful of butterflies — 1225 of them per transform at n = 4900. The
  drivers now use the scheduler's own bound. Mixed-radix transforms are 18–58%
  faster at every length measured (n = 21, 33, 693, 1100, 1155, 1920, 2156,
  4900, 6300, 8820, 22050, 44100, both precisions, both directions). The
  `purego` build never had the hooks and is unaffected.

  The defect showed up as complex64 running 1.6× slower than complex128 on the
  identical route at exactly the lengths whose schedule ends in radix 4, where
  complex64's size-4 codelet is an assembly call and complex128's is a Go
  function.

### Changed

- **Wisdom entries now override the built-in codelet preference order**, and
  the file format is bumped **v2 -> v3** because of it. Previously
  `planner.EstimatePlan` consulted the codelet registry _before_ the wisdom
  cache, and the registry hits for every size that has a codelet — so a wisdom
  entry naming a codelet signature was silently ignored for exactly the sizes
  where such a signature can exist, and `registry.LookupBySignature`'s
  stale-entry guard was unreachable. Naming a codelet signature now pins that
  codelet; naming a kernel strategy selects that strategy even where a codelet
  exists. An entry is still ignored when `PlanOptions.Strategy` forces a
  conflicting strategy, or when it names a codelet that has since been disabled
  or that the CPU cannot run.

  The v3 syntax is identical to v2, but the meaning of a record is not: a v2
  strategy entry was recorded by a measurement that never compared against the
  codelet it would now displace. **v2 files are therefore rejected on import**
  by the existing header check rather than reinterpreted — re-measure with
  `PlannerMeasure` or higher to regenerate them.

- The measuring planner modes now benchmark size-specific codelets alongside
  the kernel strategies, and record the implementation that actually won.
  `PlannerMeasure` times the codelet the estimate would otherwise have used;
  `PlannerPatient`/`PlannerExhaustive` time every enabled codelet the CPU can
  run, at roughly double the planning time where a size has several. This makes
  measure -> record -> replay reproduce the same plan.
- The mixed-radix recursion now resolves its leaf codelet once per transform
  instead of once per node. The scheduler emits a composite radix only as the
  schedule's final stage, and checks the registry for the remaining size at
  every step before that, so a codelet can only ever match at a leaf -- and
  every leaf of one transform has the same size. The entry is looked up from
  `radices[stageCount-1]` and threaded through the recursion hooks, removing a
  CPU feature detection, a map lookup and a priority scan from every node: at
  n = 1000 = [5 5 5 8] that is 156 registry lookups per transform down to 1.
  Measured geomean -1.9% over practical DSP lengths, with the win tracking the
  leaf count (n = 1000 -9.4%/-6.6% c64/c128, n = 3600 -7.6%/-4.4%,
  n = 2205 -4.2%/-3.0%). n = 768 = [3 256] regresses +6.8%/+4.4% for reasons
  that could not be pinned down on the available hardware; it has only 3
  leaves, so no win was available there in any case.
- The mixed-radix butterfly stage now applies its twiddles as one contiguous
  in-place array multiply instead of per-element inside the `k` loop, so the
  multiply reaches the SIMD `ComplexMulArrayInPlace` path and the butterfly
  loop that follows carries no twiddle arithmetic and no per-element radix
  switch. What made this possible is the recursion invariant
  `n*step == len(twiddle)`: it gives `twiddle[j*k*step] == W_n^(j*k)`, so a
  stage's twiddles are a permutation of the standard size-n table and can be
  materialised in the data's own layout, cached by stage shape rather than by
  plan.

  The path is gated on two thresholds measured as interleaved sweeps against
  the same binary with it disabled (i7-1255U, AVX2, 8 rounds, both
  precisions). It is taken only for stages with at least 64 twiddle
  multiplies — ungated, deep schedules over small factors such as
  n = 2205 = `[5 7 7 3 3]`, which ends in 245 span-3 and 735 span-1 stages,
  ran +80% slower — and never for radix 7, which lost 6–8% at every threshold
  tried while radix 11 gained 7%.

  Net over the mixed-radix benchmark set (interleaved arms, 10 rounds): geomean
  **−4.8%**, with no size significantly slower — 480 −12.4%, 704 −10.1%, 768
  −9.2%/−8.8% (complex64/complex128), 3600 −9.4%, 12000 −7.8%/−5.6%; 96, 448
  and 2205 unchanged.
  Stage twiddles are computed at size n rather than subsampled from the root
  table, the same last-ulp difference as the leaf tables below.

- `MixedRadixEligible` now routes lengths with factors 7/11 by one criterion for
  every parity: the mixed-radix engine is used when Bluestein's padded sub-FFT
  would be ≥ ~2.5n. The previous rule additionally excluded all lengths whose
  power-of-two part is 2 or 4; re-measurement showed that exclusion was fitted
  on the driver defect above rather than on the algorithm. **n = 44100**, the
  canonical audio sample rate, is 44% faster in complex128 and 22% in complex64
  as a result, moving from behind gonum to ahead of it; 44, 308, 1100, 2156,
  4900, 6300, 8820 and 22050 are rerouted with it (+23…+102%). Lengths with a
  power-of-two part ≥ 8 keep their own branch — they land a tuned codelet leaf
  the pad ratio cannot see. At 22050 the complex64 inverse is ~11% slower as a
  documented cost; the gate distinguishes neither precision nor direction.

- The three AVX-2 complex64 codelet priorities at n = 256 are now 35 / 30 / 25
  instead of 135 / 130 / 120. The old magnitudes were an order of magnitude
  outside the band used at every other size, left over from an earlier tuning
  round; only their relative order ever affected selection. A canary-gated
  sweep confirmed that order is the measured one (`dit256_radix2_avx2` <
  `dit256_radix16_avx2` < `dit256_radix4_avx2`, in both directions), so the
  rewrite is order-preserving: the codelet bound at every power-of-two size, in
  both precisions, is unchanged. No behavioural change.

- `BenchmarkCodeletCandidates{64,128}` now drive their input from a seeded RNG
  rather than the period-35 pattern `complex(i%7-3, i%5-2)`, whose spectrum is
  almost entirely zero and which made the benchmark partly time cancellation
  and denormal behaviour that differs per candidate. The values are generated
  once as `float64` and narrowed for the complex64 arm, so both precisions see
  numerically identical input. Absolute numbers from these benchmarks are not
  comparable across this change.

- The pure-Go radix-16 inverse codelets no longer apply their 1/n scaling as a
  complex multiply by `complex(1/n, 0)`, which spent two products against a zero
  imaginary part plus an add and a subtract per output. The compiler does not
  fold those away even though the factor is a compile-time constant: at n = 16
  they were the codelet's entire float-op count (96 operations, now 32). The
  rewrite is component-wise and bit-identical, and makes the inverse codelet
  16–20% faster at n = 16 and 12% faster at n = 256 for complex128. This is
  visible on `purego` and WebAssembly builds, where these are the selected
  codelets at those sizes; the default build uses AVX2 assembly at both sizes
  and is unaffected.

- The same rewrite was applied to the remaining real-factor scaling sites — the
  384-point mixed-radix codelet (both build variants and both precisions) and
  the split-radix inverse — for consistency. Both measured flat: split-radix
  scales in a separate memory-bound pass over the output, and the 384 codelet's
  radix-3 column loop is dominated by its three 128-point sub-IFFTs. Removing
  arithmetic only pays where scaling is a large share of a small compute-bound
  kernel.

### Added

- Fused AVX2 mixed-radix stage kernels for radix 3 and 5, in both precisions.
  A stage used to run as two passes — an in-place array multiply applying the
  stage twiddles, then a twiddle-free butterfly loop — of which only the
  multiply was vectorised; the butterfly loop ran one complex value per
  iteration. The kernels do the whole stage in one pass, holding the r twiddled
  rows in registers across the butterfly so they are never written back, and
  vectorise the butterfly as well: the k index is the vector axis, so every
  lane of a YMM is a different k running the same butterfly and no cross-lane
  movement is needed. Forward and inverse share one kernel, differing only in
  the XOR mask that turns the pair-swap into a multiply by -i or +i.

  Measured on an i7-1255U over 8 canary-gated interleaved rounds from a single
  binary, all 160 paired cells improved: geomean −30% (complex64 −32%,
  complex128 −28%) across the DSP-length benchmarks. The gain tracks how much
  of each schedule the kernels cover — 12000 = [5 5 5 3 32] is −47…−58%,
  44100 = [5 5 7 7 4 3 3] only −15…−17% because its radix-7 levels are not yet
  covered and its radix-3 levels fall under the stage-size gate. Radix 2/4/8/11
  stages, non-AVX2 machines and `purego` builds are unchanged and still take
  the two-pass path.

- A fused AVX2 mixed-radix stage kernel for radix 7, in both precisions,
  closing the gap the radix-3/5 kernels left at 44100 and 2205. Its butterfly
  is the radix-7 symmetry reduction — three conjugate output pairs, cosine rows
  the `c[j*m mod 7]` index map and sine rows the same map with `s[7-k] = -s[k]`
  folded in. Six constants plus the sign mask leave the kernel one register
  short of also holding `a0`, so `a0` is re-read from L1 and row 0 of `dst` is
  written last, which is what keeps the documented `dst == input` aliasing
  intact.

  Radix 7 was previously excluded from the vectorised path entirely, because
  its two-pass form measured +6…+8% slower than the scalar stage. It is now
  admitted only where the fused kernel will actually execute it, so non-AVX2
  and `purego` builds keep the scalar stage and cannot pick up that regression.

  Measured on an i7-1255U over 7 canary-gated interleaved rounds from a single
  binary: −42%/−38% at 44100 and −22%/−20% at 2205 (forward/inverse, both
  precisions within a point of each other), geomean −31% over those lengths,
  with every paired cell improving in every round. Lengths with no radix-7
  level — 1000, 3600, 12000 — came back at −0.3% geomean, i.e. unchanged.
  Fused stage calls per forward transform go from 6 to 206 at 44100 and from
  1 to 6 at 2205; 44100 is now 781 µs, down from 1.89 ms before the fused path
  existed.

- Benchmarks for the practical DSP lengths — 1000, 2205, 3600, 12000 and 44100
  — forward and inverse, in both precisions. These are exactly the lengths at
  which the library's margin over other Go FFT libraries is thinnest (~1.5–1.7×
  against ~8× at powers of two, and 44100 was behind entirely until the
  mixed-radix routing work), and none of them were measured in-tree: the whole
  existing size sweep is powers of two, so the internal numbers looked healthy
  while these were the weak spot. A regression on the mixed-radix path is now
  visible without an external harness. The complex128 variants log the plan
  they resolved to, which makes a change of route visible as well as a change
  of speed.

- `scripts/bench_gated.sh` and `scripts/bench_gated_analyze.sh` (plus a
  `just bench-gated` recipe and a BENCHMARKS.md section), a canary-gated
  codelet-candidate sweep for registry priority tuning on thermally limited or
  shared hosts. Each group of cells is bracketed by a canary of known
  quiet-machine cost before _and_ after it, so a window that degrades mid-group
  is rejected rather than averaged in; a group is one (precision, size) with
  all of its candidates back-to-back, so a candidate ranking is always taken
  under a single thermal state. Previously this protocol existed only as an
  ad-hoc script outside the repo, which made the tuning numbers cited in
  PLAN.md unreproducible.

- Benchmarks for three previously unmeasured paths: the size-16 radix-16 Go
  codelet (the selected `purego` codelet at n = 16, which had none),
  `BenchmarkSplitRadixComplex{64,128}` over 256–65536, and
  `BenchmarkPlan{Forward,Inverse}_384` for a plain `NewPlan(384)` — previously
  only the forced-Bluestein route at that length was benchmarked.

- `BenchmarkPlan{Forward,Inverse}_{256,4096}_Complex128_Focus`. complex128 had
  plan benchmarks at 128, 512 and 8192 only, so the complex64/complex128 ratio
  could not be taken in-tree at two of the four power-of-two sizes where it was
  under investigation.

- `TestCodeletsRegisterBothDirections{64,128}`, asserting that no registered
  codelet supplies only one direction. A registry entry carries `Forward` and
  `Inverse` together, so a half-registered entry would quietly run one direction
  through the codelet and the other through the generic kernel ladder — correct
  in both cases, and therefore invisible to every existing test, surfacing only
  as an unexplained forward/inverse performance asymmetry.

### Fixed

- The size-64 radix-2 DIT codelet applied its 1/n inverse scaling as 64
  complex multiplies by `complex(1/64, 0)`, spending two dead products per
  output. Naming the unscaled butterfly results for that also cost 32 extra
  temporaries, and together they pushed the fully-unrolled function past the
  node count above which Go's inliner falls back to its big-function cost
  budget — so all 193 of its `math.MulComplex64` products compiled to real
  `CALL`s, which were every un-inlined `MulComplex` call in the module. The
  scaling is now applied component-wise in one pass, which is bit-identical
  and restores inlining (193 calls → 0; the inverse codelet is 21% faster for
  complex64 and 28% for complex128). No user-visible change: at n = 64 the
  registry selects the radix-4 codelet, which remains ~1.8× faster.

- `PlanOptions{Strategy: KernelRecursive}` returned a **wrong spectrum** for
  complex128 at every length whose decomposition bottomed out in a leaf whose
  best codelet uses a prepared twiddle layout — on AVX2 hosts, n = 1024 and
  any multiple that reaches a 256-point leaf. Codelets may declare a
  SIMD-friendly twiddle layout through `TwiddleSize`/`PrepareTwiddle`, which
  the ordinary plan path materializes, but the recursive executor always
  passed leaves the standard length-n DIT table; `dit256_radix16_avx2` reads
  748 elements where it received 256. The transform silently produced garbage
  (max abs error 2.6e5 against the reference at n = 1024) and
  `Inverse(Forward(x))` did not round-trip. Recursive leaves now bind only
  codelets that consume the standard layout and fall back to the generic DIT
  otherwise. The regression escaped testing because the recursive correctness
  tests transformed an impulse, whose all-ones spectrum cannot detect a wrong
  twiddle factor; plan-level tests now cross-check a broadband signal against
  the default plan at 1024–16384 in both precisions.

- `KernelRecursive` fell off a performance cliff above n = 2048 and allocated
  on every transform, in a library whose contract is zero allocations after
  plan creation. The decomposition scorer preferred reaching a codelet-sized
  sub-problem in a single split, choosing 16-way and 32-way splits at 8192 and
  16384; those radices have no butterfly and were combined by evaluating a
  naive size-radix DFT per output element, with `sin`/`cos` recomputed in the
  innermost loop. Splits are now restricted to radix 4 and 2 — the radices
  with a real butterfly — so the strategy tree goes deep rather than wide.
  Separately, each recursion node allocated `[][]T` views plus a decimation
  buffer per sub-FFT, and the general combine allocated a temporary per output
  element; the combine steps now index the already-flat scratch and twiddle
  blocks directly, one reused decimation buffer serves every sub-FFT at a
  level, and the DIT fallback takes a bit-reversal table precomputed at plan
  time. Forward transforms are 90.8% / 96.6% / 97.7% faster at 4096 / 8192 /
  16384 (inverse 88.0% / 94.7% / 96.3%), and all sizes are allocation-free
  (16384 went from 547 allocs/op to 0). Relative to the default strategy, the
  recursive path went from 36×/62×/117× slower to a flat 1.7–2.7×.

- complex64 was slower than complex128 at most non-power-of-two lengths
  (measured at 20 of 23 in an external sweep; worst 0.68× at n = 12000).
  Go's compiler implements scalar `complex64 * complex64` by widening both
  operands to complex128, multiplying in double precision and rounding
  back — twelve instructions where the complex128 multiply needs six — so
  every FFT stage written as scalar Go cost more in the narrower type.
  Power-of-two lengths were unaffected because they run inside float32 SIMD
  codelets; the mixed-radix, Bluestein and Rader routes could not, since
  their odd-radix stages and pointwise products are scalar Go. Scalar
  complex64 multiplication now goes through `math.MulComplex64`, which stays
  in single precision, and the chirp/filter products route through the
  existing SIMD element-wise entrypoints. Every complex64 function on the
  non-power-of-two path is now free of the widening (mixed-radix driver
  92 → 0 conversion instructions, Bluestein executor 100 → 0, Rader inverse
  30 → 0). The c64/c128 forward ratio at the mixed-radix lengths went from
  1.18–1.27 (complex64 slower) to 0.90–0.98 (complex64 faster, as it should
  be), worth 21–32% in absolute complex64 time at 1000, 2205, 3600 and
  12000, with no significant change to complex128. Accuracy against a
  float64 reference is neutral-to-better across the affected lengths —
  improved at 8 of 12 sizes, unchanged or marginally worse at the rest, all
  within the documented ~10⁻⁶ complex64 band — and the pure-Go paths now
  round the same way the SIMD codelets do. Two lengths keep a complex64
  deficit (257 at 1.41, 1009 at 1.25); both have a power-of-two sub-FFT, so
  what remains there is the separately-tracked power-of-two forward-path
  weakness, not this defect. The general-purpose power-of-two codelets used
  by `purego`/WASM builds carried the same widening; see the entry below.

- The same `complex64` widening also cost the pure-Go power-of-two codelets,
  which are dead weight on the default build (the AVX2 codelets win kernel
  selection) but _are_ the transform on `purego` and WASM. 1378 scalar
  products across 39 codelet sources now multiply through
  `math.MulComplex64`. Three hot spots could not be fixed by swapping the
  operator, because they sit in `[T Complex]` bodies where the multiply has
  type `T`; they were monomorphized instead, following the existing
  `radix3TransformComplex64`/`radix5TransformComplex64` precedent:
  `butterfly2Complex64` (used by the 128- and 512-point radix-2 codelets),
  `radix4TransformComplex64` (the power-of-4 fallback, which also stops
  dispatching its butterfly through an `any()` type switch), and
  `ditForwardComplex64` — the last of which corrects an asymmetry where
  `inverseRadix4Then2Complex64` delegated to a monomorphized inverse while
  its forward twin delegated to the generic implementation. Measured
  float32→float64 conversion instructions in `internal/kernels` drop from
  4622 to 162, with none left in any non-test function reachable from a
  complex64 codelet. On the `purego` build this is worth 21–37% at every
  power-of-two size from 32 to 16384 in both directions (geomean −24.4% over
  the 8–16384 ladder); the default build and complex128 are unchanged, as
  expected. The accuracy cost is 3–9% more relative L2 error against a
  float64 reference (n = 2048: 1.10e-07 → 1.20e-07), all of it still at
  float32 epsilon, with the peak-normalised error unchanged.

- `cmd/measure_correctness` reported a number that could not be compared across
  runs, and it nearly caused the change above to be reverted as a 3× accuracy
  regression that did not exist. Two independent flaws: it maxed a _per-bin_
  relative error `|got-want| / |want|` over every bin of every trial, an
  extreme-value statistic over a quantity that explodes wherever a bin's
  magnitude approaches zero; and its complex64 "reference" was
  `reference.NaiveDFT`, which narrows its result back to complex64 and therefore
  carries a comparable error of its own, so the tool partly measured the
  divergence between two implementations. It now reports relative L2 error over
  the whole spectrum as a mean and max over trials, plus a peak-normalized
  max-per-bin error, against a genuine float64 reference; both precision arms
  transform the same input vector, and the build configuration is printed because
  the default and `purego` builds legitimately differ. **Accuracy figures from
  the old metric run 2–3 orders of magnitude higher and are not comparable** —
  at n = 128 it reported 3.95e-05 where the true relative L2 error is 9.0e-08.

- The `complex64` widening also affected the real-FFT recombination, the
  packed radix-4 Stockham engine, the recursive combine steps and the AVX2
  mixed-radix driver — the code outside `internal/kernels` that the two
  entries above did not reach. All of it now multiplies through
  `math.MulComplex64`, scales by a real factor component-wise, and rotates by
  ±i with a component swap instead of a multiply; float32→float64 conversion
  instructions across the module drop from 997 to 733. The scalar inverse
  real-FFT repack loop is **2.6× faster** (six promoting products per bin,
  −60% to −62% from half = 128 to 8192) and the forward recombination
  −22% to −25%; on a `purego` build, where these loops are the whole
  recombination rather than a SIMD tail, `PlanReal` inverse is 26–35% faster
  at N = 256…16384 and forward 7% faster at N = 1024 and 4096. The packed
  Stockham engine — the Stockham route on `purego` and WASM builds — is
  29–31% faster forward and 34–38% faster inverse at 4K/64K/1M (complex64
  geomean −32.8%). `KernelRecursive` is 12–14% faster forward and 25–29%
  faster inverse at 2048 and 8192. The AVX2 mixed-radix inverse is 15–17%
  faster at 3584 and 7168. `complex128` is unchanged everywhere except the
  inverse repack loop, which gains 3–5% because `1 - 2*u` was a full complex
  multiply where doubling the components needs half the products.

### Added

- `reference.NaiveDFTWide`: a float64 DFT of a complex64 input, returning the
  complex128 accumulator un-narrowed. This is the float64 reference a complex64
  transform's error can be measured against; there was none in the tree before,
  which is why `cmd/measure_correctness` ended up comparing complex64 against
  complex64. `reference.NaiveDFT` is now a narrowing wrapper around it and
  returns bit-identical values.

- `fft.ComplexMulArray` and `fft.ScaleInPlace`: generic wrappers over the
  existing SIMD element-wise product and real-scalar scale, matching the
  `fft.ComplexMulArrayInPlace` that was already there

## [0.7.1] - 2026-07-25

Documentation only — the code is byte-identical to `v0.7.0`. Released so
that a tag exists whose `CHANGELOG.md` describes what `v0.7.0` actually
shipped; benchmark results measured against `v0.7.0` remain valid.

### Fixed

- The `v0.7.0` changelog entry, which was written from a stale reading of
  the tree and was wrong in three places: it documented a `Meta()`
  introspection method and a `PlanRealT` type that the generics refactor
  had already removed (the surface is `PlanInfo` — `Len`,
  `KernelStrategies`, `Algorithms`, `String`, `Close`); it claimed the
  5-smooth Bluestein pad sizes were computed but inert, after the shape
  whitelist in `plan_padsize.go` had enabled them; and it claimed
  auto-selection routed power-of-two squares in [2^18, 2^22) to
  split-radix, a rule that had been reverted the same day as measured
  losses. It also omitted most of the release — four-step, radix-8, the
  cache-blocked transpose, the size-32768 codelets, the SSE and NEON
  ladders, the AVX2 complex128 Stockham assembly, the codelet priority
  retune, odd-length real FFT, the SIMD real-FFT recombination, fast-size
  convolution padding, the Rader 7/11 extension, the FMA and wisdom CPU
  feature fixes, and the removal of the `asm` build tag

## [0.7.0] - 2026-07-25

### Added

- Radix-7 and radix-11 butterflies for the mixed-radix engine: lengths with
  factors 7/11 (e.g. 448, 704, 1344) run exactly instead of through
  Bluestein wherever that measured faster — 1.3–6× on AVX2 for shapes with
  power-of-two part ≥ 8 and 1.2–3.4× for odd shapes whose Bluestein pad is
  ≥ ~2.5n, and every tested shape on `purego`; shapes that measured as
  losses keep their previous Bluestein routing
- Radix-8 stage for the mixed-radix scheduler, generalized from the
  size-512 codelet's butterfly (hardcoded ±i/W_8^1/W_8^3 rotations, both
  precisions): a radix-8 stage is emitted whenever the remaining
  power-of-two part 2^e has e ≥ 3 — except e = 4, where [4, 4] measured
  ~20% faster than [8, 2] — so 2^5 runs as [8, 4] instead of [4, 4, 2] and
  2^9 as [8, 8, 8] instead of [4, 4, 4, 4, 2]. Gated to the no-codelet
  path, so schedules that can reach a codelet leaf are untouched; measured
  through the same driver on `purego`, geomean −16.9% across
  32…12288-point radix-8-bearing sizes (complex64 −11…−34%), a win at
  every size and both precisions — it benefits `purego`, SSE-only amd64,
  and arm64
- Split-radix (2/4) FFT kernel for power-of-two sizes — recursive,
  natural-order output, no bit-reversal pass, in-place via scratch —
  selectable via `PlanOptions.Strategy = KernelSplitRadix` and included in
  the Patient/Exhaustive measuring-planner candidates. On the `purego`
  build a forced split-radix plan beats the default path at every power of
  two ≥ 256 (+11–34%, 2.1× at 262144); on the SIMD build the AVX2/AVX-512
  codelets stay ahead below 262144. Auto-selection does not route to it —
  see the power-of-two square rule under Changed
- Four-step FFT kernel, the rectangular generalization of six-step over a
  tiled out-of-place transpose, selectable via
  `PlanOptions.Strategy = KernelFourStep` (wisdom name "fourstep") and
  benchmarked by the Patient/Exhaustive measure modes: any power-of-two
  n ≥ 4 splits as n1×n2 — including the non-square sizes six-step declines
  — with the split chosen by a cache-residency cost model over the
  per-core L1d/L2 sizes `internal/cpu.DetectCaches` reports (Linux sysfs,
  conservative 32K/256K defaults elsewhere) instead of the fixed √n;
  zero-alloc, both precisions. Measured (i7-1255U, AVX2, complex64,
  forward): ≈ six-step at square sizes, beats split-radix at 2^21…2^23
  (−7…−28%), ties plain Stockham at 2^23; the split sweep is flat (±7%)
  with the cache-derived choice within noise of the optimum, so the auto
  rule is unchanged and measure/wisdom stays the arbiter
- Rader's algorithm for prime-size transforms: primes whose p−1 the
  mixed-radix engine executes exactly and which pass a measured cost gate
  now run an exact length-(p−1) cyclic convolution instead of Bluestein's
  power-of-two pad to ≥ 2p−1 (still zero-alloc); other primes keep
  Bluestein, and forcing `PlanOptions.Strategy = KernelBluestein` opts
  out. 5-smooth p−1: power-of-two p−1 (17, 257, 65537: 4–5×) and any
  5-smooth p−1 ≥ 96 whose power-of-two part is ≥ 8 (97, 401, 641, 769,
  1153, 1601, 3001, 4001, 12289, 18433, 40961: 1.1–5.6×, and 1.6–2.1× on
  `purego`). 7/11-smooth p−1 rides the radix-7/11 butterflies behind its
  own gate (full-matrix DFT stages need one): p−1 ≥ 2048 wins whenever its
  power-of-two part is ≥ 4 (2113…30241: 1.1–3.4×), and below 2048 only a
  single radix-7/11 stage — optionally with one radix-3 — on a deep
  power-of-two chain (113, 353, 449, 673, 1409: 1.1–2.0×). Everything
  shallower or odd-heavier measured 0.34–1.06× and stays on Bluestein;
  7393 is a knowing exception (−9% on `purego` complex64, +11–36% in its
  other three arms). Padded Rader for non-smooth p−1 is a wash vs
  Bluestein and was skipped
- Real-FFT support for odd lengths: the real plan constructors accept any
  n ≥ 2 — even lengths keep the packed half-size method unchanged, odd
  lengths run an internal full-size complex FFT fallback (forward widens
  the real input and keeps the n/2+1 non-redundant bins, inverse rebuilds
  the full Hermitian spectrum first, with DC-only spectrum validation
  since odd n has no Nyquist bin). Works for every length the complex
  planner supports (mixed-radix, Bluestein, Rader), zero-alloc in steady
  state, batch/stride and `Clone` supported; the fallback tracks the
  same-size complex plan's cost, i.e. the previous manual workaround minus
  the copies. The 2D/3D real plans still require even width
- Size-32768 codelets in both precisions, generic Go and AVX2:
  32768 = 2·4^7 had none, so plans fell back to generic Stockham — which
  for complex128 is scalar Go on amd64, a 5× cliff (618 µs vs FFTW's
  123 µs). Radix-4-then-2 (7 radix-4 stages plus a radix-2 combine); the
  generic kernel ping-pongs between scratch and dst instead of per-stage
  stack arrays, the AVX2 kernels take the digit-reversal table as a
  `bitrev []int` argument rather than embedding 256 KiB of DATA. Measured
  end-to-end (i7-1255U): c128 618→237 µs forward (2.6×), 601→257 µs
  inverse; c64 225→168 µs forward (−26%), 221→170 µs inverse
- SSE3/SSE2 codelet ladder from 2048 to 32768, both precisions — the tuned
  SSE tier stopped at 1024, so SSE-only hosts ran generic Go codelets
  above it. Radix-4-then-2 at 2048/8192/32768 and radix-4 at 4096/16384,
  plus a registry entry for the already-existing SSE3 size-256 complex64
  kernel. vs the generic codelets they displace (i7-1255U, forward/
  inverse): 256 c64 1.6/2.0×, 2048 c64 −46/−49% and c128 −12/−26%,
  4096 c64 1.59/1.64× and c128 1.52/1.41×, 8192 1.5–2×, 16384 c64 −41/−47%
  and c128 −41/−23%, 32768 c64 −15/−42% and c128 ±0/−38%. Every
  measurement forces the SSE path on an AVX2-capable host; validation on
  genuine pre-AVX2 hardware is still open
- NEON size-specific codelet ladder completed to 32768 in both precisions
  — it stopped at 256, so 512–32768 ran the priority-1 generic radix-2
  fallback. Mixed-2/4 at 512/2048/8192/32768 and radix-4 at
  1024/4096/16384, QEMU-verified (round-trip, in-place, reference DFT)
  plus native suites. Their priorities are mirrored from the smaller
  sizes, **not** tuned: QEMU timings are meaningless, so the ladder is
  unbenchmarked on real arm64 hardware — above ~8192 the DIT codelets also
  compete with the Go six-step path there
- AVX2 complex128 Stockham assembly: every Stockham-resolved complex128
  size above the codelet range (65536, 131072, 524288, 2^21, …) previously
  ran the scalar Go kernel on amd64. Kernel-level vs that scalar path
  (forward/inverse): 2048 −50/−55%, 16384 −50/−45%, 65536 −38/−33%,
  131072 −24/−32%, 524288 −29/−12%, 2^21 −16/−27% (the large sizes go
  memory-bound); end-to-end vs FFTW, 65536 1.44 ms → 1.02 ms (gap
  4.4× → 3.1×) and the 524288 gap now 2.0×
- Plan-reuse DSP types `Convolver`, `Correlator`, and `RealConvolver`:
  reusable, concurrency-safe, zero-allocation convolution/correlation for
  loops (the one-shot `Convolve`/`CrossCorrelate`/`ConvolveReal` helpers
  re-plan on every call)
- `PlanInfo`, one introspection and lifecycle interface — `Len()`,
  `KernelStrategies()`, `Algorithms()`, `String()`, `Close()` —
  implemented by every plan type, with compile-time assertions in
  `plan_interface.go`. The plural accessors report the resolved kernel per
  axis in dimension order; single-kernel plans (1D, real 1D, fast plans)
  return one-element slices and keep the singular
  `KernelStrategy()`/`Algorithm()` as convenience. `Close` was added to
  the composite and real plans, `Clone`/`String` to `FastPlan` and
  `FastPlanReal`
- `Plan.ForwardInPlace` and `FastPlan.ForwardInPlace`, matching the
  multi-dimensional plans' `ForwardInPlace`/`InverseInPlace` naming
- Core FFT implementation: DIT, Stockham, radix-2/3/4/5, mixed-radix,
  six-step/eight-step algorithms with per-size codelets
- Bluestein's algorithm for arbitrary-length transforms
- Real FFT support (1D, 2D, 3D) for float32 and float64 input
- Multi-dimensional transforms (2D, 3D, N-D)
- Batch and strided transform APIs
- Convolution and correlation helpers (complex and real, both precisions)
- complex64 and complex128 precision throughout
- SIMD kernels (AVX-512/AVX2/SSE2/SSE3 on amd64, NEON on arm64, SSE on 386) selected at runtime via CPU detection; on AVX-512 CPUs the generic
  AVX-512 kernel also serves as the complex64 codelet at sizes
  1024/4096/8192/16384 (1.2–2.4× over the AVX2 codelets it replaces)
- Wisdom: persist and reuse plan-tuning decisions
- WebAssembly (js/wasm) target support
- Concurrency-safe plans: a single plan instance may run transforms from
  multiple goroutines
- Comprehensive testing infrastructure: reference-DFT cross-validation,
  property tests, round-trip, fuzz, stress, race, and zero-allocation guards
- Performance benchmarking suite
- Development tooling (justfile, golangci-lint, treefmt, pre-commit hooks)
- GitHub Actions CI/CD workflow (multi-OS, multi-arch, WASM)
- Project documentation (README, CONTRIBUTING, CHANGELOG)

### Changed

- SIMD ships in the **default build**, selected at runtime by CPU
  detection: the `asm` build tag is gone — it is no longer a gate — and
  `-tags purego` is the one supported opt-out. A plain `go get` consumer
  now gets the AVX2/SSE2/SSE3, NEON and SSE kernels with no tag at all;
  CI builds, vets and tests both flavors on every architecture
- Bluestein plans run their padded sub-FFT through the size-dispatched DIT
  kernels (radix-4 and size-specific codelets, SIMD where available) instead
  of the generic radix-2 path: prime-size transforms measure 25–64% faster
  (geomean −39%) on the default build and 1.2–1.4× faster on `purego`
- Bluestein and one-shot convolution share a **shape-aware** padded-length
  model (`cheapestPaddedLength`, `plan_padsize.go`), and 5-smooth pads now
  actually ship. The single `bluesteinSubFFTPenalty` constant is gone:
  measured against the power-of-two endpoint of its own dyadic window, a
  mixed-radix sub-FFT's cost per m·log2(m) point-pass spans ~7× on shape
  alone (i7-1255U/AVX2, complex64: 3072 = 2^10·3 → 0.83, 2560 = 2^9·5 →
  0.96, 3584 = 2^9·7 → 1.39, 2160 = 2^4·3^3·5 → 2.31, 3000 = 2^3·3·5^3 →
  2.87, 2250 = 2·3^2·5^3 → 6.18), so no scalar penalty can be right for
  all of them. The model is now a whitelist of candidate families
  (`padShapes`), each admitted only above the pad size where it wins at
  **both** precisions: `3·2^(k-2)` from 2^9 (0.71/0.87, holding to 0.41/
  0.46 at 2^16) and `15·2^(k-4)` from 2^13 (0.80/0.69, holding to 0.74/
  0.75); `7·2^(k-3)` is admitted nowhere — it loses to `15·2^(k-4)` in
  every window where either wins and, being the smaller of the two, is
  reachable only when that one is too. End-to-end `Plan.Forward` vs the
  power-of-two pad (c64/c128): n=677 → 1536 0.74/0.85, n=2531 → 6144
  0.43/0.47, n=3079 → 7680 0.80/0.67, n=4099 → 12288 0.44/0.47,
  n=6151 → 15360 0.78/0.70, n=8209 → 24576 0.43/0.46 — i.e. −15…−57%,
  with the unchanged control n=1009 measuring 1.00/1.00. Zero-alloc
  preserved on the new mixed-radix padded path; `fastConvolutionLength`
  rides the same model (convLen 257 pads to 384 instead of 512). The
  calibration is AVX2-only — `purego` passes but was not re-measured, so
  its thresholds are conservative
- The mixed-radix scheduler strips odd factors before powers of two whenever
  a codelet exists for a power-of-two suffix, so 5-smooth sizes end in a
  tuned SIMD codelet leaf (768 = [3, 256] instead of [4, 4, 4, 4, 3]):
  96/480/768/1152/12000 measure 2.4–5.2× faster; sizes without a reachable
  codelet keep the old order and are unchanged
- `KernelAuto` no longer special-cases power-of-two squares — they fall
  through to the plain size heuristic (Stockham), and two earlier rules
  that measurement exposed as losses were removed.
  `BenchmarkSquareAutoRule` timed every candidate strategy at the only
  sizes the square branch reaches (2^18, 2^20, 2^22), both directions,
  both precisions, both builds, arms adjacent in one process, medians of 5
  (i7-1255U/AVX2): split-radix for [2^18, 2^22) lost every arm bar
  `purego` 2^18 complex64 forward (−3%, inside noise) — SIMD 2^18 c64
  3.39/3.28 ms fwd/inv vs 10.16/7.00, `purego` 2^20 c128 20.2/21.7 vs
  44.9/46.4 — and eight-step for ≥ 2^22 lost at 2^22 c64 (Stockham
  157/171 ms vs 201/269 on SIMD, 102/113 vs 203/247 on `purego`). One arm
  dissents and is accepted knowingly: 2^20 complex128 forward prefers
  six-step (39.3 ms) to Stockham (49.7 ms) on the SIMD build, where
  Stockham still beats the split-radix it replaces by 1.6×.
  Non-power-of-two squares keep six/eight-step, unchanged
- Codelet priorities retuned against measurement (i7-1255U): timing every
  registered codelet per size exposed systematic mis-selection — the
  priority-favored six-step / radix-32×32 / radix-16 / radix-8 codelets
  lose to the plain radix-4 family at every size where both exist, in both
  the AVX2 and generic tiers. AVX2 c64: 256 radix16→radix2 −26%,
  512 radix8→radix2 −26%, 1024 radix32x32→radix4 −52%, 4096
  sixstep→radix4 −26%, 8192 sixstep→radix4_then2_params −55%, 16384
  sixstep→radix4 −32%; AVX2 c128: 256 radix4→radix16 −8/−24%; generic both
  precisions: radix4/radix4_then2 over sixstep/radix32x32 at
  1024/4096/8192/16384. The c128 1024 radix-32×32 AVX2 codelet is disabled
  (its inverse ran 2× slower than the SSE2 radix-4 that now serves).
  End-to-end vs the FFTW bench harness: c64 forward 1024 −52%, 8192 −57%,
  16384 −35%; c128 inverse 1024 −49%, 256 −30%. Six-step codelets stay
  registered for wisdom/measure to pick where they win
- Six-step/eight-step and the 2D plans transpose in place with a
  cache-blocked tiled walk (`math.TransposeSquare`) instead of an O(n²)
  swap-pair index table cached forever per size — no table, no permanent
  cache, any n. Tile edge 8 was chosen by sweep (16+ falls off a cliff
  beyond 512²); n ≤ 32 uses an unblocked walk. Transpose micro-benchmark
  −70…−82% at 128²…1024², both precisions; end-to-end generic
  six-step/eight-step −10…−23% at n ≥ 65536 (geomean −12.8%) and square
  `Plan2D` 128²…512² −30…−42% (geomean −34.6%), zero-alloc preserved
- The forward real-FFT recombination `X[k] = A[k] − U[k]·(A[k]−B[k])` is
  SIMD: AVX2 (4 complex64 / 2 complex128 bins per iteration, mirrored
  `B[k]` as one reversed load plus in-register reversal and conjugate
  sign-flip, the product as an FMA `VFMADDSUB`) is 4.5–8× the scalar loop
  and moves `BenchmarkPlanRealForward` 27–41% (geomean −34.7%); an SSE3
  tier follows for hosts without AVX2 (complex64 ~4.2×, complex128 ~1.3×;
  SSE2-only hardware keeps the generic loop, the idiom needs
  `MOVSLDUP`/`ADDSUBPS`), and the complex128 inverse pre-pass got a
  vectorized AVX2 kernel (~2.1×) in place of its stub. Zero-alloc
  preserved, generic path unchanged — no `purego` regression. A NEON
  variant is blocked on native-arm64 benchmarking
- One-shot convolution and correlation pad to a fast FFT size instead of
  the awkward length convolution produces: lengths the engine executes
  exactly are kept, anything that would route to Rader/Bluestein is padded
  via the shared pad model and the result truncated back to convLen.
  Measured over Bluestein/Rader-routed convLen 127…4001, both builds:
  `Convolve` −70…−85% on AVX2 and −70…−78% on `purego`, with the
  Rader-routed lengths also winning (257: −34% AVX2 / −15% `purego`;
  4001: −74% / −70%); steady-state `Convolver` at prime convLen 1009
  −91.5% (AVX2) / −78% (`purego`). `ConvolveReal` unchanged, zero-alloc
  steady state preserved
- The complex-multiply cores of the AVX2 codelets the registry actually
  selects are FMA-fused (`VMOVDDUP`/`VPERMILPD`/`VFMADDSUB231PD` and the
  f32 twin): the complex128 radix-4 family first (24 sites, size-256 c128
  forward −10.9%, p=0.000), then the remaining 7 selected files (97 sites,
  `VADDSUBPD` 64→0 in `avx2_f64_size256_radix16.s`). Accuracy improves at
  the sizes carrying the most fused twiddle work — max relative error vs
  the reference DFT, c64 size 16 2.35e-06→1.68e-06 (−28.5%), size 32
  2.41e-05→1.49e-05 (−38.2%), everything else bit-identical or within
  1.4% — the expected one-rounding-instead-of-two effect. No speedup is
  claimed for the second pass: three benchstat attempts on the i7-1255U
  throttled (86–98 °C, ±30–90% variance) and were inconclusive. Trivial
  ±1/±i twiddles and the real-scalar 1/n normalization are deliberately
  left unfused — with no addend, an FMA form would add work
- One precision scheme across the public API: the generic constructors
  carry the plain names — `NewPlan[T]` (replacing `NewPlanT` and the bare
  complex64 `NewPlan`) and `NewPlanReal[F, C]` — with the `32`/`64`
  wrappers kept as documented one-line sugar, and `New*WithOptions` as the
  single options-carrying entry point. Everything is generic:
  `FastPlanReal[F, C]` replaces `FastPlanReal32/64` and
  `PlanReal2D[F, C]`/`PlanReal3D[F, C]` replace the concrete float32
  versions, closing the missing float64 real 2D/3D gap; `PlanReal3D` also
  gained its `WithOptions` constructor
- `PlanOptions` carries plan-time concerns only (`Planner`, `Strategy`,
  `Wisdom`); per-call layout lives at the call site via
  `ForwardBatch`/`InverseBatch` and `ForwardStrided`/`InverseStrided`. The
  public types are root-owned: `Complex`/`Float` are declared in the root
  package, `KernelStrategy` is a root enum with `String()`, and `Wisdom`
  wraps the internal cache — no public alias into `internal/*` remains
- Multi-dimensional plan constructors now wrap `ErrInvalidLength` and child
  plan failures with dimension context (matching `PlanND`); match errors with
  `errors.Is`
- `NewPlanPooled`/`NewPlanPooledWithOptions` accept the same lengths and
  planner options as `NewPlan`: Bluestein sizes are served by the regular
  allocator instead of being rejected, and `PlanOptions.Planner` measure
  modes are honored

### Fixed

- Missing `VZEROUPPER` in the size-384 mixed codelet assembly
  (`avx2_f{32,64}_size384_mixed.s`) and the AVX2 radix-3 butterfly helpers:
  the dirty upper-YMM state made every legacy-SSE instruction in downstream
  kernels pay the AVX–SSE transition penalty — the complex128 size-384
  codelet ran 22× slower than after the fix (68µs → 3.1µs), dragging down
  every size routed through it (1152: 210µs → 14µs)
- FMA is a separate CPUID bit from AVX2, but the FMA-using AVX2 kernels
  were dispatched on `HasAVX2` alone and could `SIGILL` on hardware or
  emulators exposing AVX2 without FMA. `cpu.Features.HasFMA` was added
  (amd64 + 386) and the AVX2 codelet tier plus the real-FFT
  recombination/repack paths now require `HasAVX2 && HasFMA`, falling back
  to SSE3 or the generic loop otherwise
- Wisdom entries bound codelets by signature without re-checking CPU
  features, so an entry created or imported on an AVX2 host could bypass
  that fallback and execute FMA opcodes on a CPU with FMA masked off.
  Resolution re-checks feature support and falls back to heuristic
  selection for incompatible entries

### Removed

- `Meta()` and `PlanMeta` (superseded by the `PlanInfo` interface and the
  plural `KernelStrategies()`/`Algorithms()` accessors)
- `NewPlanT`/`NewPlanRealT`, the `Planner` type with its
  `Plan1D/2D/3D/ND/Real*` methods, and the free `Plan1D` function — the
  generic constructors and `New*WithOptions` replace them
- `Plan.InPlace` and `FastPlan.InPlace` (forward-only): use
  `ForwardInPlace`
- Inert `PlanOptions` fields: `Radices`, `Workspace` and the
  `WorkspacePolicy` type, `InPlace`, and `Batch`/`Stride` (all were
  documented "not yet implemented" or never read)
- `NewPlanFromPool`/`NewPlanFromPoolWithOptions` (took an internal pool type
  no external caller could name; use `NewPlanPooled`/`NewPlanPooledWithOptions`)

### Planned

- Higher-radix / per-size-tuned AVX-512 kernel variants
- NEON priority tuning on real arm64 hardware

## [0.0.1] - 2025-12-24

### Added

- Project initialization
- Basic project structure

---

## Notes

For breaking changes, feature requests, or bug reports, please open an issue on GitHub.

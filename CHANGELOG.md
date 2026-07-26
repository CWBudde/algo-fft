# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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

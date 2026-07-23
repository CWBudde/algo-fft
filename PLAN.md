# PLAN.md — algofft Roadmap

This roadmap is the source of truth for status and direction. The v1.0
engineering work (Priorities 0–3 of the post-review roadmap) is **complete**;
the detailed item-by-item history is preserved in git (see the history of this
file). What remains here is a condensed record, the **immediate pre-v1.0
architecture consolidation (§2)**, the few carried-over open items, and the
post-v1.0 optimization backlog.

Design philosophy lives in `goal.md`; the component inventory is generated
into `docs/IMPLEMENTATION_INVENTORY.md` via `go generate ./internal/kernels/...`
(which runs `cmd/gencodelets -inventory <path>`).

---

## 1. Current Status (condensed)

All v1.0 Definition-of-Done items are green except the release tag itself:

- **P0 Correctness & build integrity** (2026-07): the SIMD build compiles and
  is CI-gated on amd64/386/arm64; ~3.9k lines of dead code deleted; all plan
  types are genuinely concurrency-safe (per-call scratch caches, `-race`
  tests); no silent wrong-answer paths (unschedulable radices panic, the
  scheduler/driver contract is validated); every registered codelet is
  verified forward-vs-`reference.NaiveDFT` per direction; docs/module-path/
  committed-binary issues fixed.
- **P1 Architecture hardening** (2026-07): kernel strategy is per-plan
  (`PlanOptions.Strategy`) with no process-global strategy state; tuning
  persists via the versioned, atomically-imported Wisdom cache (size +
  precision + CPU features). One deliberate global remains: the _default_
  Wisdom cache (`fft.DefaultWisdom`, mutated by `ImportWisdom*`/
  `ClearWisdom`) — per-consumer isolation is available via
  `PlanOptions.Wisdom`. Zero-allocation parity across 1D/2D/3D/ND/real/mixed-radix on
  both default and SIMD paths, locked in by `AllocsPerRun` guards. Plan-layer
  and DSP `*128` duplication collapsed into generics; dispatch de-duplicated;
  codelet **registration** is generated from a declarative table
  (`cmd/gencodelets`, ~164 entries).
- **P2 SIMD that ships** (2026-07): SIMD is on the **default build** behind
  runtime CPU detection (`-tags purego` is the opt-out; `-tags asm` a no-op).
  All known-incorrect kernels fixed (NEON complex64 bit-reversal/conjugation,
  NEON in-place copy-back corruption, x86 size-16 radix-16 aliasing); ~1,000
  `asmdecl` findings resolved and vet-gated. Kernel coverage: AVX2 (broad,
  incl. complex128 512–16384), SSE2/SSE3 (512/1024 both precisions), NEON
  complex64+complex128 size-specific sets, first AVX-512 tier (generic
  radix-2 + per-size codelets at 1024/4096/8192/16384 where it wins).
- **P3 API completeness & polish** (2026-07): no dead knobs or
  unimplemented options in the public API (the remaining `ErrNotImplemented`
  returns are intentional: the documented `FastPlan` "no codelet for this
  size" signal and unreachable defensive fallbacks); introspection parity
  (`Meta()`/`KernelStrategies()`/`Algorithms()` across plan types);
  `ForwardInPlace` naming unified; plan-reuse DSP types (`Convolver`,
  `Correlator`, `RealConvolver`); consistent `%w` error wrapping; GoDoc
  audit enforced by revive.
- **Testing & CI**: every arch matrix leg builds, vets, and tests both the
  default (SIMD) build and `-tags purego`; lint gate green; coverage gated
  at 90% (codecov); nightly benchmarks against a committed baseline;
  continuous fuzzing (PR + nightly) with committed regression corpus;
  property tests (Parseval/linearity/shift) across all dispatch families.

### Open items carried over

- [ ] **Tag `v1.0.0`**, GitHub release notes. _(Now gated on the
      architecture consolidation below (§2) — the API-shape items in A1/A2
      are breaking changes and must land before the tag; issue/PR templates
      in place.)_
- [ ] **ARM64 NEON sizes 512+**: blocked on native ARM64 hardware — QEMU
      timings are not representative, so a size-specific 512/1024 kernel
      can't be benchmark-justified from CI today. Sizes 512/1024 are already
      NEON-served by the generic DIT kernels. Revisit with the native ARM64
      CI runner (Community backlog).
- [ ] **WASM SIMD**: blocked on toolchain — Go's `GOEXPERIMENT=simd`
      intrinsics (golang/go#73787) reached amd64 in Go 1.26 and Wasm/ARM64
      in the 1.27 RC, but remain experimental and this module targets
      Go 1.25. Revisit when the experiment graduates or the toolchain floor
      moves.
- AVX-512 follow-ups (higher radix, complex128, reclaiming 2048) are folded
  into P4.2 below.

---

## 2. Immediate — Architecture Consolidation (pre-v1.0)

Outcome of the 2026-07 architecture review. The library has little external
usage yet, so **breaking changes land directly — no deprecation shims, no
transition aliases**. Deprecated symbols are deleted, not kept. This section
gates the `v1.0.0` tag: the API is the one thing that cannot be fixed after
tagging.

Ordering below is execution order: A0 first (correctness risk), then the
structural items; A6 quick fixes can land anytime.

### A0. Fix the lossy codelet fallback contract _(correctness risk)_

The `Kernel[T]` contract returns `bool` ("handled?") and `internal/fft`
honors it with `if !kernel(...)` fallbacks — but `fftypes.CodeletFunc` has no
bool, and `wrapCodelet64/128` (`internal/kernels/codelet_registry.go:80-91`)
discards the return value. `internal/transform/recursive.go:57` then calls
`codelet.Forward(...)` with no way to detect failure: a codelet that bails
(e.g. undersized slice) silently no-ops instead of falling back.

- [x] Unify on **one** kernel contract. _(2026-07)_ `fftypes.CodeletFunc`
      now returns `bool` ("handled?"), matching `Kernel[T]`; the lossy
      `wrapCodelet64/128` adapters are deleted and `cmd/gencodelets` emits
      the kernel functions directly (regenerated). Every call site honors
      the signal: the recursive executor falls back to generic DIT
      (`internal/transform/recursive.go`), the AVX2 mixed-radix path falls
      through to pure Go (`internal/fft/mixedradix_avx2.go`), `Plan`
      falls through to Stockham/kernel dispatch (safe and Unsafe paths),
      and `FastPlan` panics on a bail (caller-contract violation; no
      silent no-op). Regression tests
      (`internal/transform/recursive_fallback_test.go`) register a bailing
      codelet and verify forward/inverse output still matches
      `reference.NaiveDFT` — they fail against the old contract. Full
      `-race` suite, purego, arm64, and wasm builds green.

### A1. Public API rationalization _(breaking, before tag)_

Today three precision idioms coexist (`NewPlanT[T]`, `NewPlan32/64`, bare
`NewPlan`), generic-vs-concrete is inconsistent (`FastPlanReal32/64` and
`PlanReal2D/3D` are hand-written concrete while everything else is generic —
and double-precision real 2D/3D therefore **doesn't exist**), and lifecycle/
introspection methods are asymmetric across plan types.

- [x] **One precision scheme**. _(2026-07)_ The generic constructors now
      carry the plain names: `NewPlan[T]` (replaces `NewPlanT` and the bare
      complex64 `NewPlan`), `NewPlanReal[F, C]` (replaces `NewPlanRealT` and
      bare `NewPlanReal`); `32`/`64` wrappers remain as documented one-line
      sugar. The `Planner` type, its `Plan1D/2D/3D/ND/Real*` methods, and
      the free `Plan1D` function are deleted — `New*WithOptions` is the one
      options-carrying entry point.
- [x] **Make everything generic**. _(2026-07)_ `FastPlanReal[F, C]`
      replaces `FastPlanReal32/64` (recombination helpers are bound at
      construction, so the hot path keeps zero type switches);
      `PlanReal2D[F, C]`/`PlanReal3D[F, C]` replace the concrete float32
      versions, closing the missing float64 real-2D/3D gap
      (`plan_real_multidim_64_test.go` validates both against the naive
      reference DFT). `PlanReal3D` also gained the previously missing
      `WithOptions` constructor.
- [x] **One common plan interface**. _(2026-07)_ `PlanInfo` (`Len`,
      `KernelStrategies`, `Algorithms`, `String`, `Close`) is implemented by
      every plan type, with compile-time assertions in `plan_interface.go`.
      Plural introspection everywhere (single-kernel plans return
      one-element slices, singular accessors stay as convenience);
      `Meta()`/`PlanMeta` deleted. `Close` added to the composite/real
      plans, `Clone`/`String` added to `FastPlan`/`FastPlanReal`. Transform
      methods stay on the concrete generic types (their signatures depend
      on the element types).
- [x] **One in-place story**. _(2026-07)_ Deprecated `InPlace()` deleted
      from `Plan` and `FastPlan`; inert `PlanOptions.InPlace` flag deleted.
- [x] **Split `PlanOptions` to plan-time concerns only**. _(2026-07)_
      `PlanOptions` is now `Planner`/`Strategy`/`Wisdom` only. `Batch`/
      `Stride` fields, the option-driven batch loops in the 2D/3D/ND/real
      plans, `resolveBatchStride*`, and all per-constructor child-option
      stripping are deleted; `ForwardBatch`/`ForwardStrided` on the 1D plan
      remain the batch/stride story.
- [x] **Own the public types**. _(2026-07)_ `Complex`/`Float` are declared
      in the root package; `KernelStrategy` is a root-owned enum (with
      `String()`) converted to the internal enum at the plan-construction
      boundary; `Wisdom` is a wrapper struct over the internal cache with
      `WisdomKey`/`WisdomEntry` converted at the boundary. No public type
      aliases into `internal/*` remain.
- [ ] `ErrNotImplemented` must not be reachable from a live `Forward` path
      (plan.go:306) — after A4 the constructor either builds a working
      executor or fails. _(Gated on A4, tracked there.)_

### A2. Collapse the multi-dimensional plan copies

`Plan2D` (369 lines) and `Plan3D` (395 lines) fully reimplement what
`PlanND` already does; `PlanReal2D/3D` copy the same skeleton again —
~150 lines of identical wrapper logic (`Forward`/`Inverse` batch loops,
`validate`, `String`, `Clone`, option stripping) exist five times.

- [x] Make `Plan2D`/`Plan3D` thin typed wrappers over the ND engine.
      Keep a dimension-specialized inner loop **only** where `benchstat`
      proves it beats the ND path (methodology §3 applies).
      _(Done 2026-07. `PlanND.transformDimension` gained the three access
      patterns the specialized plans had: contiguous in-place rows for the
      innermost axis, per-slab `TransposeSquare` for a second-innermost
      axis matching the innermost size, and a two-loop strided enumeration
      replacing the div/mod `sliceIndexToOffset`. `Plan2D`/`Plan3D` now
      hold only their dimensions plus an inner `*PlanND`; benchstat showed
      no specialized loop beat the ND path (512×512 2D −5%, 4D −22%,
      rest ~equal, still zero allocations), so all five copies of the
      transform bodies were deleted.)_
- [x] Extract shared wrapper logic (validation, String/typeName, Clone,
      batch loop) into one internal helper set; delete the five copies.
      The `switch any(zero).(type)` type-name/dispatch block currently
      appears in a dozen-plus places (plan.go:166, plan*scratch.go:60,
      plan_real_generic.go ×7, …) — one `typeName[T]()`/dispatch helper.
      *(Done 2026-07. `plan_common.go` holds `complexTypeName[T]()`,
      `validateDstSrc[D, S]()` (two element types, so real plans share it),
      and `transformSliceInPlace()`. All `String()` type-name blocks and
      the validation copies in `Plan`, `PlanND`, `PlanReal`,
      `PlanReal2D/3D` now use them; `allocAlignedSlice`'s dispatch switch
      collapsed onto the existing `mem.AllocAligned[T]`. The remaining
      `switch any(zero)` sites are value-dispatch bridges scheduled for
      A5's kernel twins.)\_
- [x] Deduplicate the one-shot vs reusable DSP pipelines: `convolveT`
      (convolve.go:55) and `Convolver.Convolve` (convolver.go:80) implement
      the identical pad→FFT→multiply→IFFT→copy sequence — one core, two
      entry points. Same for the shared pad cost model (`bluesteinPadSize`
      / `fastConvolutionLength`).
      _(Done 2026-07. `convolveT`, `crossCorrelateT`, and `convolveRealT`
      validate and then run a throwaway `Convolver`/`Correlator`/
      `RealConvolver`; the reusable types own the pipeline. The pow2-vs-
      5-smooth costing now lives once in `cheapestPaddedLength(minM)`,
      used by both `bluesteinPadSize` and `fastConvolutionLength`.)_

### A3. Internal layering repair

The intended layering is `fftypes` (contracts) → `kernels`/`transform`
(algorithms) → `planner` (selection) → `fft` (bridge) → root. Three things
muddy it:

- [x] **Un-invert `kernels → planner`**: the codelet registry
      (`CodeletRegistry`, `Registry64/128`, `CodeletEntry`) lives in
      `planner` and `kernels` registers _upward_ into it, re-exporting
      planner types back out (`codelet_registry.go:11-41`). Move the
      registry into a neutral leaf package (`internal/fftypes` or a new
      `internal/registry`); `kernels` registers into it, `planner` reads
      from it. This also removes the four-deep type-alias chain
      (`fftypes.CodeletFunc` → planner → kernels → fft).
      _(Done 2026-07. New leaf package `internal/registry` (imports only
      `fftypes` + `cpu`) owns `CodeletRegistry`/`CodeletEntry`/
      `Registry64/128`/`GetRegistry`/`CPUSupports`; `cmd/gencodelets` now
      emits `registry.`/`fftypes.`-qualified registrations (regenerated).
      `kernels` and `transform` no longer import `planner` at all — both
      inversions gone — and the alias chain is deleted; the duplicated
      `cpuSupportsLevel` copy in the kernels tests now uses
      `registry.CPUSupports`.)_
- [x] **Decide what `internal/fft` is** — currently it is both a façade
      and bypassed: the root package imports six internal packages directly
      and calls `EstimatePlan` via `fft` in one path (plan.go:724) and via
      `planner` in another (`plan_fast.go:55`). Either (a) make it a real
      façade and the _only_ internal import of the root package, or
      (b) delete the pure re-export files (`kernels.go`, most of
      `dispatch.go`, `transform_exports`) and let the root import
      `planner`/`kernels`/`transform` directly. Pick (b) unless a concrete
      reason for the façade emerges — less indirection, honest line counts.
      Real logic in `fft` (arch dispatch, mixed-radix engine, Rader glue,
      pooling) stays; only forwarding shims go.
      _(Done 2026-07, option (b). `fft/kernels.go` is deleted (its real
      Bluestein glue moved to `fft/bluestein.go`); `dispatch.go` keeps only
      `SelectKernels*`/`bridgeKernel*`; `fft.go` keeps only private
      helpers. The root and `cmd/` now import `fftypes` (strategy enum),
      `planner` (wisdom/estimate), `transform` (recursive/packed),
      `kernels`, `registry`, and `math` directly; every remaining
      `fft.X` use in the root is real logic (SIMD helper dispatch,
      pooling, Rader/Bluestein/strided/measure). All fft-internal alias
      uses were qualified with their owning packages.)_
- [x] **One Stockham owner**: `kernels.ForwardStockham*` and
      `transform.ForwardStockhamPacked` coexist and are re-exported side by
      side. Declare one canonical (or document the split: fixed-size vs
      packed mixed-radix) and name/locate them so the distinction is
      visible.
      _(Done 2026-07: documented as two distinct algorithms, not
      duplicates — `kernels.StockhamForward/Inverse` is the canonical
      plain radix-2 power-of-two autosort; `transform`'s packed variant is
      the radix-4+2 engine with `PackedTwiddles` and a per-build toggle.
      Role comments at both sites point at each other, and the
      side-by-side re-exports in `fft` are gone with the façade.)_
- [x] **One algorithm-name ↔ strategy mapping**: `resolveWisdom`
      (planner.go:107) and `StrategyToAlgorithmName` (utils.go:150) are two
      hand-synced switch statements; collapse to one table. Give
      `KernelRecursive` an explicit entry instead of falling through to
      `"unknown"`.
      _(Done 2026-07. One `strategyAlgorithmNames` table in
      `planner/utils.go` drives both directions
      (`StrategyToAlgorithmName` + new `AlgorithmNameToStrategy`, used by
      `resolveWisdom`); `KernelRecursive` maps to `"recursive"`, and
      `Plan.String()` now prints `Recursive` instead of `auto` for
      recursive plans. Round-trip pinned by
      `planner/strategy_names_test.go`.)_

### A4. Split the `Plan[T]` god-struct into per-strategy executors

`plan.go` is 984 lines; `Plan[T]` is a ~40-field tagged union carrying state
for every strategy simultaneously, dispatched by an `if kernelStrategy == …`
ladder in `Forward`/`Inverse`. Every new strategy touches the struct and
both hot methods.

- [ ] Introduce an internal executor interface
      (`forward(dst, src)`, `inverse(dst, src)`, `close()`); one
      implementation per strategy family (codelet/DIT, Stockham,
      split-radix, six/eight-step, recursive, Bluestein, Rader) owning only
      its own tables. `Plan[T]` shrinks to: validation, scratch/pool
      management, one executor field, introspection.
- [ ] Re-partition the `plan_*.go` files along the new seams (construction,
      execution wrappers, lifecycle, introspection, DSP) — the current
      split is arbitrary (batch execution in plan.go, batch stride
      resolution in plan_batch.go; hand-rolled `itoa` in plan.go next to
      `fmt.Sprintf` in plan_2d.go).
- [ ] Zero-alloc and `AllocsPerRun` guards must stay green throughout —
      this is a refactor, not a rewrite; land it strategy-by-strategy with
      the existing reference/round-trip gates.

### A5. Generate the complex128 kernel twins

`internal/kernels` hand-maintains ~500 monomorphized functions (270
`*Complex64`, 231 `*Complex128`) that are byte-for-byte twins differing only
in element type — a deliberate performance choice (generics deoptimize
complex arithmetic), but double the maintenance surface of the largest
package (38k lines).

- [ ] Extend `cmd/gencodelets` (or add a sibling template step) to emit the
      `Complex128` kernel bodies from the `Complex64` sources, with
      generated-file headers. Hand-written code shrinks by roughly half;
      emitted instructions unchanged (verify with the existing
      forward-vs-reference registry sweep and `benchstat` noise runs).

### A6. Quick fixes _(independent, land anytime)_

- [ ] `cmd/bench_compare` + `cmd/measure_correctness` **don't compile**:
      their `go.mod` says `github.com/cwbudde/algofft` (no dash) vs the
      actual module `algo-fft`; `measure_correctness` also imports
      `internal/reference` across a module boundary (illegal). Fold both
      into the main module; delete the "Why Separate Modules?" rationale
      from `cmd/README.md`.
- [ ] `cmd/README.md` documents 2 of 4 tools — add `gencodelets` and
      `benchkernels`.
- [ ] Naming drift: `gofft` appears 16× in README.md and throughout
      `goal.md`; standardize on `algofft` (package) / `algo-fft` (module).
      Archive `goal.md` (it's the historical design doc for the old name;
      this file is the source of truth) or rewrite its header to say so.
- [ ] Extend `just clean` to remove `*.test` binaries, `*.pprof`, `*.o`,
      `dist/`, and stale `coverage_*` variants.
- [ ] `Executor.Close` doc says "no-op" but calls `plan.Close()`
      (executor.go:35-42) — make the code and comment agree (A1/A4 may
      delete `Executor` entirely; it is a thin `Clone()` wrapper).
- [ ] Inline magic epsilons `1e-4`/`1e-12` in real-inverse spectrum
      validation (plan_real_generic.go:342-353) → named, documented
      constants.

**Explicitly kept as-is** (reviewed, deliberate): the benchmark-cited
selection thresholds in `internal/planner` (compile-time constants are fine
pre-wisdom-tuning), the asm build-tag triples, the wisdom cache design, the
root-package black-box test strategy (revisit file organization post-v1.0 if
it grows), and hand-written SIMD assembly.

---

## 3. Methodology for every P4 item

The P0–P2 discipline continues to apply to all optimization work:

1. **Measure baseline first** (`benchstat`, committed
   `benchmarks/baseline-*.txt` where relevant).
2. **Verify against reference**: forward-vs-`reference.NaiveDFT`, round-trip,
   and in-place for every new kernel/codelet (the P0.4 registry sweep gates
   registered codelets automatically).
3. **Both precisions** unless the item is precision-specific; **zero-alloc**
   guards extended to any new path.
4. **No regressions on the purego build** — algorithm-level wins must land in
   the generic code path too, not only in assembly.
5. Only land a per-size codelet if it **beats the incumbent in `benchstat`**
   on the target hardware tier (the AVX-512 codelet rule).

---

## Priority 4 — Optimized Algorithms Backlog

Ordered within each group by expected benefit/effort. Concrete code
references are to the current tree.

### P4.1 Algorithm-level upgrades (benefit all builds, including purego)

- [x] **Bluestein padding: next 5-smooth size instead of next power of two.**
      _(2026-07)_ Done, with a measurement-driven twist. `NextHighlyComposite`
      landed in `internal/math`; the Bluestein sub-FFT (filter build and
      convolution, `internal/fft`) executes any 5-smooth m via the mixed-radix
      engine; `bluesteinPadSize` (`plan.go`) costs both candidates as
      m·log2(m). Benchmarking the crossover (`BenchmarkBluesteinPadCandidates`,
      `internal/fft`) exposed the real win first: the power-of-two sub-FFT was
      running the generic scalar radix-2 kernel, and routing it through the
      size-dispatched DIT kernels made Bluestein 25–64% faster (geomean −39%)
      on the default build and 1.2–1.4× on purego. Against that upgraded
      baseline the mixed-radix engine measures 2.2× (purego) to 4.5× (AVX2)
      slower per point, while a 5-smooth pad can undercut the next power of
      two by at most ~2× in m·log2(m) work — so the power of two wins at
      every size and `bluesteinSubFFTPenalty` (2.2) intentionally keeps
      5-smooth pads disabled. The chooser and mixed-radix path stay wired and
      tested; if radix-3/5 butterflies get SIMD kernels, re-run the benchmark
      on both builds and lower the constant. (Note: the P4.5 fast-size padding
      item faces the same ≤2× bound and should reuse this measurement.)
      _Update 2026-07:_ the odd-first mixed-radix schedule and the size-384
      `VZEROUPPER` fix cut the engine penalty to ~1.3–2× for 5-smooth sizes
      whose power-of-two part is ≥ 8 — re-running the pad-candidate benchmark
      with a shape-aware cost model may now re-enable 5-smooth pads for those
      shapes.
- [x] **Rader's algorithm for prime sizes.** Rader maps a prime-p FFT to a
      cyclic convolution of length p−1, which needs no padding when p−1 is
      5-smooth (vs Bluestein's pad to ≥ 2p−1). Implemented in
      `internal/fft/rader.go` + `plan_rader.go`, riding the Bluestein plan
      plumbing (strategy stays `KernelBluestein`, `Algorithm()` reports
      `"rader"`; forcing `KernelBluestein` opts out). Per-size benchmarking
      (`BenchmarkRaderVsBluestein`, both precisions) showed the mixed-radix
      engine's per-point penalty makes "smaller" not always faster, so
      `RaderEligible` gates on measured wins: power-of-two p−1 (17, 257,
      65537: 4–5×) and any 5-smooth p−1 ≥ 96 whose power-of-two part is
      ≥ 8 (97, 401, 641, 769, 1153, 1601, 3001, 4001, 12289, 18433, 40961:
      1.1–5.6×, and 1.6–2.1× on purego) — with the odd-first mixed-radix
      schedule those shapes end in a tuned codelet leaf. Shapes whose
      power-of-two part is ≤ 4 (31, 61, 101, 151, 251) and tiny p−1 (≤ 40)
      measured as losses and stay on Bluestein. Remaining follow-up: padded Rader for
      non-smooth p−1 is a wash vs Bluestein (pad ≥ 2p−3 vs ≥ 2p−1), so it
      was intentionally skipped.
- [x] **Split-radix (conjugate-pair) kernels.** _(2026-07)_ Generic
      split-radix (2/4) DIT landed in `internal/kernels/splitradix.go`
      (recursive, natural-order output, no bit-reversal pass; per-precision
      hot paths; in-place via scratch), exposed as `KernelSplitRadix` with
      full strategy plumbing (planner names, wisdom mapping, measure-mode
      candidates for Patient/Exhaustive). Measured
      (`BenchmarkSplitRadixVsIncumbents`): on purego it beats the
      auto-selected path at every power of two ≥ 256 (+11–34%, 2.1× at
      262144); on the SIMD build the AVX2/AVX-512 codelets stay ahead below 262144. Auto-selection changed only where proven on **both** builds
      and precisions: power-of-two squares in [2^18, 2^22) (512², 1024²) now
      resolve to split-radix instead of six-step (~2× both directions) —
      six-step's scalar O(n) index-table transpose dominates there (the SIMD
      transpose kernels stop at 128×128). Revisit the auto rule when the
      P4.3 cache-blocked transpose lands; wisdom/measure modes can pick
      split-radix anywhere it wins per-machine.
- [x] **Radix-8 stage for the generic DIT driver.** _(2026-07)_ The radix-8
      butterfly from the size-512 codelet is generalized into
      `internal/kernels/radix8.go` (hardcoded ±i/W_8^1/W_8^3 rotations, both
      precisions) and added to the mixed-radix engine's executable set
      (`internal/fft/mixedradix.go`): the scheduler emits a radix-8 stage
      whenever the remaining power-of-two part 2^e has e ≥ 3 — except e = 4,
      where [4,4] measured ~20% faster than [8,2] — so 2^5 runs as [8,4]
      instead of [4,4,2], 2^9 as [8,8,8] instead of [4,4,4,4,2]. Gated to
      the no-codelet path (`!oddFirst`): with a codelet reachable on the
      radix-4 suffix chain the schedule is unchanged, so AVX2 production
      schedules (and the P4.1 odd-first/Rader tuning) are untouched.
      Measured via `BenchmarkMixedRadixRadix8Schedule` (old vs new schedule
      through the same driver, purego): geomean −16.9% across
      32…12288-point radix-8-bearing sizes, wins at every size and both
      precisions (complex64 −11…−34%), zero-alloc preserved. Benefits
      purego, SSE-only amd64, and arm64 builds — the paths without a
      codelet chain.
- [x] **Real-FFT for odd/multi-factor lengths.** _(2026-07)_ `NewPlanRealT`
      (and the `NewPlanReal*` constructors) now accept any n ≥ 2: even
      lengths keep the packed half-size method unchanged; odd lengths run an
      internal full-size complex FFT fallback (`plan_real_odd.go`) — forward
      widens the real input and keeps the n/2+1 non-redundant bins, inverse
      rebuilds the full Hermitian spectrum before the complex inverse, with
      DC-only spectrum validation (odd n has no Nyquist bin). Works for every
      length the complex planner supports (mixed-radix, Bluestein, Rader);
      zero-alloc in steady state (`AllocsPerRun` guard at n=105), batch/
      stride and `Clone` supported, verified forward-vs-`reference.NaiveDFT`
      and round-trip at both precisions across primes/5-smooth/prime-power
      odd sizes. Multi-factor even lengths already worked (the half-size
      child plan handles arbitrary sizes). `BenchmarkPlanRealForwardOdd`
      shows the fallback tracks the same-size complex plan's cost, i.e. the
      users' previous manual workaround minus the copies. Remaining
      follow-up: evaluate a real-input Bluestein that exploits conjugate
      symmetry in the padded convolution to close the ~2× gap vs a
      hypothetical packed odd-length method; the 2D/3D real plans still
      require even width (their row/column packing is a separate item).
- [x] **Radix-7 / radix-11 butterflies for the mixed-radix engine.**
      _(2026-07)_ Full-matrix DFT butterflies with precomputed coefficient
      tables landed in `internal/kernels/radix{7,11}.go` (both precisions,
      forward + conjugate-inverse tables); the scheduler strips factors 7/11
      unconditionally (like 5, keeping the pow2 part intact for codelet
      leaves) and both recursion drivers execute them. Extends exact
      (non-Bluestein) coverage from 2^a·3^b·5^c to 2^a·3^b·5^c·7^d·11^e —
      but only where measured faster: routing goes through the new
      `planner.MixedRadixEligible` win gate
      (`BenchmarkMixedRadix7And11VsBluestein`, both precisions, both builds).
      On AVX2, shapes with power-of-two part ≥ 8 win 1.3–6× (448: 3.3×,
      704: 2.6×, 1344: 4.8×, 3584–14080: 4–6×), odd shapes win 1.2–3.4× when
      Bluestein's pad is ≥ ~2.5n (11, 33, 35, 49, 77, 385, 539, 693, 1155,
      2401); on purego every tested shape won (1.1–4.7×). Shapes that
      measured as losses on AVX2 — pow2 part 2/4 (14, 28, 308, 462, 924:
      strided radix-2/4 tails, same pattern as the Rader gate) and odd with
      pad < 2.5n (7, 63, 121, 231, 847: the ~2× pad lands on an unusually
      fast codelet) — keep their previous Bluestein routing, so nothing
      regresses. Zero-alloc preserved (`TestMixedRadix7And11_ZeroAlloc`),
      verified vs `reference.NaiveDFT` + round-trip + in-place at 28 sizes.
      Follow-ups: extend `RaderEligible` to primes with 7/11-smooth p−1
      (needs its own benchmark pass), and revisit the gated-out shapes if
      the odd butterflies get SIMD kernels.

### P4.2 SIMD depth & breadth

- [ ] **FMA audit of the amd64 kernels.** Only 49 of 109 `.s` files under
      `internal/asm/amd64` use `VFMADD*`; the rest issue separate
      `VMULPS/VADDPS` chains. Convert the complex-multiply cores of the
      remaining AVX2 kernels to fused form (fewer uops, better accuracy —
      one rounding instead of two). Do it size-by-size with `benchstat` and
      the existing forward-vs-reference gates; expect the biggest wins on
      the twiddle-heavy generic radix-4 and Stockham kernels. Also part of
      the audit: FMA is a separate CPUID bit from AVX2, and the pre-existing
      FMA-using AVX2 kernels are dispatched on `HasAVX2` alone — harmless on
      real hardware (all AVX2 CPUs ship FMA3) but wrong on emulators/VMs
      that mask FMA. `cpu.Features.HasFMA` exists now and the real-FFT
      recombination/repack dispatch already requires it; sweep the remaining
      AVX2 dispatch sites onto `HasAVX2 && HasFMA` as kernels are audited.
      _Update 2026-07:_ first audit pass landed. The complex128 radix-4
      codelet family (`avx2_f64_size256_radix4.s`,
      `avx2_f64_size32_radix4_then2.s`) had its per-twiddle complex-multiply
      cores (`VMULPD`/`VXORPD`-negate/`VADDPD`) fused to the in-tree
      `VMOVDDUP`/`VPERMILPD`/`VFMADDSUB231PD` idiom already used by
      `avx2_f64_size64_radix4.s` — 24 multiply sites, dead `maskNegLoPD`
      loads removed. Measured (size-256 c128 forward, AVX2+FMA): −10.9%
      (p=0.000), zero-alloc preserved, registry sweep vs `reference.NaiveDFT`
      green. On the dispatch side, the AVX2 **codelet** tier is now gated on
      `HasFMA` too: `planner.cpuSupports` requires `HasAVX2 && HasFMA` for
      `SIMDAVX2` (the whole codelet tier compiles to FMA opcodes), so an
      FMA-masked VM correctly falls back to SSE/generic instead of faulting.
      Remaining: the f32 size-specific codelets (different YMM/low-64 idioms),
      the generic radix-4/Stockham kernels, and the non-codelet AVX2 dispatch
      sites (`complex_mul_amd64.go`, `kernels_amd64_asm.go`) still need their
      pass.
- [ ] **AVX-512 higher-radix / per-size-tuned variants** (carried over from
      P2.4). The shipped AVX-512 tier is generic radix-2; a radix-4 AVX-512
      kernel should widen the 1.2–2.4× gap and could reclaim size 2048 and
      the complex128 sizes where AVX2 codelets still win. Needs AVX-512
      CI/bench hardware for regression tracking.
- [ ] **ARMv8.3 FCMLA complex-arithmetic kernels.** `internal/cpu` detects
      only `HasNEON`; ARMv8.3's `FCMLA`/`FCADD` do a full complex
      multiply-accumulate in two instructions (vs 4 mul + 2 add today —
      `internal/asm/arm64/neon_complex_mul.s`). Add `HasFCMA`
      detection (HWCAP on Linux, sysctl on darwin), a `FCMLA` variant of the
      generic NEON butterfly, and runtime-dispatch it above plain NEON.
      Apple Silicon and Neoverse both support it. Blocked for benchmarking
      on the same native-ARM64-hardware item as NEON 512+.
- [x] **SIMD the real-FFT forward recombination loop.** _(2026-07)_ The
      per-bin recombination `X[k] = A[k] − U[k]·(A[k]−B[k])` now lives in
      `internal/fft.RecombineForwardComplex64/128` with AVX2 kernels
      (`internal/asm/amd64/avx2_real_recombine.s`): 4 complex64 / 2
      complex128 bins per iteration, the mirrored `B[k]` as one reversed
      vector load + in-register reversal + conjugate sign-flip, and the
      `U[k]·t` product as an FMA `VFMADDSUB` complex multiply. The kernel is
      4.5–8× faster than the scalar loop; end-to-end
      `BenchmarkPlanRealForward` (AVX2) improved 27–41% (geomean −34.7%),
      zero-alloc preserved, generic path unchanged (no purego regression).
      Both `PlanRealT` and `FastPlanReal32/64` route through it. Follow-ups
      landed the same month: an SSE3 tier for the forward recombination
      (complex64 2 bins/XMM at ~4.2× the scalar loop, complex128 1 bin/XMM
      at ~1.3×; SSE2-only hardware falls back to the generic loop since the
      idiom needs `MOVSLDUP`/`ADDSUBPS`) and a vectorized AVX2 complex128
      inverse pre-pass kernel (2 pair-bins per iteration with reversed
      mirrored load/store, ~2.1× the scalar loop, replacing the
      `inverseRepackComplex128SIMD` stub). Remaining: NEON variant (blocked
      on the native-ARM64 benchmarking item).
- [ ] **SSE2 tier breadth.** The non-AVX2 tier has tuned kernels only at
      512/1024; profile which other hot sizes (256, 2048, 4096) fall back to
      the generic path on SSE-only hardware and extend where `benchstat`
      justifies.

### P4.3 Memory & cache

- [x] **Cache-blocked transpose for six-step/eight-step.** _(2026-07)_ The
      O(n²) swap-pair index table (cached forever per size) is gone;
      `math.TransposeSquare` (`internal/math/transpose.go`) transposes in
      place with a tiled walk — no index table, no permanent cache, works
      for any n. Tile edge 8 was chosen by sweep
      (`BenchmarkTransposeSquareBlockSize`): small tiles keep the strided
      stream within the TLB/L1 (16+ falls off a cliff beyond 512²); n ≤ 32
      uses an unblocked walk (whole matrix fits L1). Transpose
      micro-benchmark: −70…−82% at 128²…1024², both precisions. End-to-end:
      generic six-step/eight-step (`internal/kernels`) −10…−23% at
      n ≥ 65536 (geomean −12.8%, `BenchmarkSixStepComplex64/128`), square
      `Plan2D` 128²…512² −30…−42% (geomean −34.6%); zero-alloc preserved.
      The split-radix auto-rule revisit ran: six-step gained but split-radix
      still wins 1.2–1.6× at 2^18/2^20 pow2 squares, so the P4.1 rule
      stands (comment updated in `internal/planner/selection.go`).
      Remaining follow-up: SIMD 8×8 complex tile kernel (AVX2
      `VPERM2F128`/`VUNPCK` pattern, NEON `TRN1/TRN2`) for a further
      constant-factor win.
- [ ] **Cache-blocked variants above L2.** For n where the working set
      exceeds L2 (≳2¹⁸ complex64), evaluate four-step/six-step with
      block sizes chosen from detected cache sizes (extend `internal/cpu`
      to expose L1d/L2 sizes) instead of fixed decompositions; Wisdom-tune
      the block choice.
- [ ] **Twiddle-table bandwidth reduction.** The packed-twiddle tables are
      per-plan precomputed; for large n they are a significant fraction of
      the working set. Evaluate (a) sharing tables between forward/inverse
      via conjugate-on-load (SIMD sign-flip is nearly free), and (b) the
      quarter-table symmetry (store n/8 entries + swizzle). Only worth it
      where measurements show the tables competing with data for cache.
- [ ] **SoA (split real/imag) layout exploration** (post-v1.0 API, from the
      Future list). Prototype internal SoA for one kernel family (e.g. the
      AVX-512 generic path, which currently spends shuffle uops
      de-interleaving) and measure; decide whether a v2 `PlanSoA` API is
      warranted before designing it.

### P4.4 Parallelism (opt-in, keep single-thread default)

- [ ] **Parallel batch execution.** `Plan.ForwardBatch`/`InverseBatch`
      (`plan.go:438,479`) run count transforms sequentially. Add
      `PlanOptions.Parallel`/`MaxWorkers` (default 1 = today's behavior):
      the batch loop fans out over a pre-created worker set with per-worker
      scratch from the existing resident-cache pattern, preserving
      zero-alloc-in-steady-state. Batch is embarrassingly parallel — this
      is the highest-value, lowest-risk parallel item.
- [ ] **Parallel 2D/3D/ND row-column passes.** Each axis pass is an
      independent batch of 1D transforms over rows/columns — reuse the
      P4.4 worker infrastructure. The transpose/gather steps stay serial
      initially. Gate on plan size (parallelism below ~256×256 is
      overhead-dominated); verify with the existing `-race` concurrent
      tests plus new parallel-enabled ones.
- [ ] **Parallel six-step for very large 1D.** The six-step algorithm is
      already a (transpose, batch-FFT, twiddle, transpose, batch-FFT)
      pipeline; run the inner batch-FFT stages on the worker pool for
      n ≳ 2²⁰. Depends on the cache-blocked transpose (P4.3) so the serial
      transpose doesn't dominate.

### P4.5 DSP-layer optimizations

- [x] **Fast-size padding for one-shot convolution/correlation.** _(2026-07)_
      `fastConvolutionLength` (`convolve.go`) frees the FFT size from the
      awkward lengths convolution produces: lengths the engine executes
      exactly (powers of two, gated mixed-radix smooth) are kept — padding
      those is not a measured purego win — while anything that would route
      to Rader/Bluestein is padded to the next fast size and the result
      truncated to convLen. Pad candidates are costed exactly like
      `bluesteinPadSize` (pow2 vs next 5-smooth with the measured
      mixed-radix penalty, i.e. pow2 wins under the current constant).
      Wired through `convolveT` and `Convolver` (which `Correlate`/
      `CrossCorrelate`/`AutoCorrelate`/`Correlator` ride); the real
      variants already padded to the next power of two. Measured
      (Bluestein/Rader-routed convLen 127…4001, both builds): one-shot
      `Convolve` −70…−85% on AVX2 and −70…−78% on purego, with the
      Rader-routed lengths also winning (257: −34% AVX2 / −15% purego;
      4001: −74% / −70%); steady-state `Convolver` at prime convLen 1009
      −91.5% (AVX2) / −78% (purego). `ConvolveReal` unchanged. Zero-alloc
      steady state preserved (`TestConvolver_ZeroAllocSteadyStatePadded`);
      correctness pinned vs naive convolution at prime and Rader-eligible
      lengths, plus a `fastConvolutionLength` invariant sweep.
- [ ] **Buffer reuse in one-shot DSP helpers.** The one-shots allocate 5
      temporaries per call; route them through the pooled resident-cache
      scratch (as `Convolver` already does) so casual users get most of the
      steady-state performance without switching types.
- [ ] **Overlap-add/overlap-save streaming convolution.** For long-signal /
      short-kernel filtering (`len(b) ≪ len(a)`), one big FFT is
      asymptotically worse than block convolution with a plan of size
      ~4×len(b). Add `StreamingConvolver` (fixed kernel, chunked input) —
      this is both an API feature and the standard algorithmic fix for the
      current "FFT the whole signal" cost profile.

---

## Post-v1.0 Future (unchanged)

**Features**: DCT, Hilbert transform, STFT/spectrograms, audio/image examples,
Gonum ecosystem integration, optional GPU backends (kept out of the pure-Go
core).

**Community**: `CODE_OF_CONDUCT.md`, Dependabot, native ARM64 CI runner
(unblocks the NEON benchmarking items above).

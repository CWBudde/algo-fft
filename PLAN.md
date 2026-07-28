# PLAN.md — algofft Roadmap

This roadmap is the source of truth for status and direction. The v1.0
engineering work (Priorities 0–3 of the post-review roadmap) is **complete**;
the detailed item-by-item history is preserved in git (see the history of this
file). What remains here is a condensed record, the **immediate pre-v1.0
architecture consolidation (§2)**, the few carried-over open items, and the
post-v1.0 optimization backlog.

Design philosophy lives in `docs/goal.md`; the component inventory is generated
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
- [x] `ErrNotImplemented` must not be reachable from a live `Forward` path
      (plan.go:306) — after A4 the constructor either builds a working
      executor or fails. _(Done 2026-07 with A4: the `planExecutor`
      methods return no error; the constructor rejects forced-recursive
      non-power-of-two lengths with `ErrInvalidLength` (previously silent
      wrong spectra), and a kernel bail after construction — impossible for
      a validated plan — panics as an internal invariant violation.
      `ErrNotImplemented` remains only as a constructor error
      (`NewFastPlan*` without a codelet).)_

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

- [x] Introduce an internal executor interface
      (`forward(dst, src)`, `inverse(dst, src)`, `close()`); one
      implementation per strategy family (codelet/DIT, Stockham,
      split-radix, six/eight-step, recursive, Bluestein, Rader) owning only
      its own tables. `Plan[T]` shrinks to: validation, scratch/pool
      management, one executor field, introspection.
      _(Done 2026-07. `planExecutor[T]` in `plan_exec.go` with four
      implementations: `kernelExecutor` (codelet → packed Stockham →
      strategy-dispatched fallback kernel; DIT/Stockham/six/eight-step/
      split-radix/mixed-radix differ only in which kernel the dispatch
      bound, so one executor serves them), `bluesteinExecutor`,
      `raderExecutor`, `recursiveExecutor` — each owning only its own
      tables. Deviations from the sketch, both deliberate: no `close()`
      (executors are immutable after construction and shared with clones,
      so `Plan.Close` just drops the reference) and no error returns (see
      the `ErrNotImplemented` item under A1). `Plan[T]` went from ~40
      fields to 21 (validation, scratch/pool management, executor,
      introspection, shared twiddle/bitrev for the strided fast path, and
      a four-field codelet fast-path cache — see the zero-alloc item);
      dead weight found on the way — `packedTwiddle8/16` computed but
      never read, packed twiddles built for non-Stockham plans, a per-call
      `cpu.DetectFeatures()` in the recursive path — is gone. The public
      `Executor[T]`/`NewExecutor` shim was deleted with A6's blessing
      (`Clone()` is the concurrent-use story; tests moved to
      `plan_clone_test.go`).)_
- [x] Re-partition the `plan_*.go` files along the new seams (construction,
      execution wrappers, lifecycle, introspection, DSP) — the current
      split is arbitrary (batch execution in plan.go, batch stride
      resolution in plan*batch.go; hand-rolled `itoa` in plan.go next to
      `fmt.Sprintf` in plan_2d.go).
      *(Done 2026-07. `plan.go` = struct + construction only (984 → 595
      lines); `plan_transform.go` = Forward/Inverse/InPlace/Unsafe/batch
      wrappers; `plan_exec*.go` = the four executors; `plan_lifecycle.go`
      = Reset/Close/Clone (the strategy switches replaced by stored
      `scratchLen`/`subScratchLen`); `plan_introspect.go` = Len/String/
      KernelStrategy/Algorithm + the name constants; `plan_alloc.go` =
      allocation helpers (`allocateScratchSet` now takes plain sizes).
      `executor.go`, `plan_bluestein.go`, `plan_rader.go`,
      `plan_recursive.go` deleted; hand-rolled `itoa` replaced by
      `strconv.Itoa` everywhere.)\_
- [x] Zero-alloc and `AllocsPerRun` guards must stay green throughout —
      this is a refactor, not a rewrite; land it strategy-by-strategy with
      the existing reference/round-trip gates.
      _(Done 2026-07. All `*NoAllocs*` guards green; full suite, `-race`,
      purego, arm64 (QEMU), wasm/386 cross-builds, `vet-arch`, and lint
      all pass. Interleaved `benchstat` vs the pre-split tree first showed
      the interface dispatch costing ~20 ns — +73% at n=8, invisible from
      n=64 up — so the codelet binding is additionally cached on `Plan` as
      a zero-dispatch fast path (documented at the fields; the executor
      stays complete without it). With the cache, n≥16 is neutral and n=8
      shows a ~2 ns residual (+10%, p≈0.04) at the measurement machine's
      layout-noise floor; `FastPlan` remains the latency path for tiny
      sizes.)_

### A5. Generate the complex128 kernel twins

`internal/kernels` hand-maintains ~500 monomorphized functions (270
`*Complex64`, 231 `*Complex128`) that are byte-for-byte twins differing only
in element type — a deliberate performance choice (generics deoptimize
complex arithmetic), but double the maintenance surface of the largest
package (38k lines).

- [x] Extend `cmd/gencodelets` (or add a sibling template step) to emit the
      `Complex128` kernel bodies from the `Complex64` sources, with
      generated-file headers. Hand-written code shrinks by roughly half;
      emitted instructions unchanged (verify with the existing
      forward-vs-reference registry sweep and `benchstat` noise runs).
      _(Done 2026-07 as the sibling command `cmd/genkernels` (source-to-source,
      unlike the spec-table-driven `gencodelets`): every eligible
      `*Complex64` function gets its twin emitted into a per-file
      `<base>_c128.gen.go` (42 files, 108 functions); stale outputs are
      removed on regeneration and the third `go:generate` directive lives in
      `codelet_registry.go`. ~9.9k hand-written complex128 lines deleted.
      A pre-generation audit diffed a candidate transform of every complex64
      function against its hand-written twin: 92 were identical modulo
      comments/blank lines, 16 had drifted (the complex64 side had been
      optimized later — unrolled stages, hoisted twiddles, a dropped `%256`);
      for those the generated twin replaces the stale copy, which benchstat
      (interleaved, `-test.cpu 1`, n=8) confirmed as free complex128 wins:
      1024/radix4 −27%/−18%, 4096/radix4 forward −7%, rest neutral, all
      complex64 controls neutral. Deliberately still hand-written: the
      radix-3/5 complex128 entry points (they delegate to the generic
      implementations), the test helpers, and `dit_16384_radix4`'s complex128
      pair — its `[16384]complex128` stage arrays (256 KiB) exceed the
      compiler's 128 KiB explicit-declaration stack limit, so a textual twin
      heap-allocated ~1.75 MiB/op at ~2× the time (caught by benchstat, then
      pinned by a new guard). That guard —
      `codelet_alloc_norace_test.go`, a registry-wide zero-alloc sweep — also
      caught two pre-existing escapes: the 8192-point radix-4-then-2 and
      six-step codelets `make` 128 KiB complex128 buffers (over the 64 KiB
      implicit-alloc limit). Fixed at the complex64 source with explicit
      backing arrays; complex128 8192 radix4then2 got −38%/−34% and
      0 B/op from it. `BenchmarkDITComplex64/128` gained 4096/8192/16384
      cases.)_

### A6. Quick fixes _(independent, land anytime)_

- [x] `cmd/bench_compare` + `cmd/measure_correctness` **don't compile**:
      their `go.mod` says `github.com/cwbudde/algofft` (no dash) vs the
      actual module `algo-fft`; `measure_correctness` also imports
      `internal/reference` across a module boundary (illegal).
      _(Done 2026-07.)_ `measure_correctness` folded into the main module
      (its `go.mod`/`go.sum` deleted). `bench_compare` **kept as a separate
      module** — its names fixed to `algo-fft` — so its gonum benchmarking
      dependency stays out of the main module's graph; `cmd/README.md`'s
      "Why a Separate Module?" section was rewritten (not deleted) to reflect
      that only `bench_compare` is isolated.
- [x] `cmd/README.md` documents 2 of 4 tools — add `gencodelets` and
      `benchkernels`. _(Done 2026-07; also documents the new `genkernels`, so
      all five tools are covered.)_
- [x] Naming drift: `gofft` appears 16× in README.md and throughout
      `goal.md`; standardize on `algofft` (package) / `algo-fft` (module).
      _(Done 2026-07.)_ README.md already used `algofft`; `goal.md` was
      **archived** to `docs/goal.md` with a historical banner (superseded by
      this file), and its `gofft` mentions are kept as a historical artifact.
      References in `AGENTS.md` and this file updated to `docs/goal.md`.
- [x] Extend `just clean` to remove `*.test` binaries, `*.pprof`, `*.o`,
      `dist/`, and stale `coverage_*` variants. _(Done 2026-07.)_
- [x] `Executor.Close` doc says "no-op" but calls `plan.Close()`
      (executor.go:35-42) — make the code and comment agree (A1/A4 may
      delete `Executor` entirely; it is a thin `Clone()` wrapper).
      _(Done 2026-07 with A4: `Executor`/`NewExecutor` deleted; `Clone()`
      is the concurrent-use API.)_
- [x] Inline magic epsilons `1e-4`/`1e-12` in real-inverse spectrum
      validation (plan*real_generic.go:342-353) → named, documented
      constants.*(Done 2026-07: `spectrumImagTol32`/`spectrumImagTol64` in
      `plan_real_generic.go`, used by both the even and odd inverse paths.)\_

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
- [x] **Shape-aware Bluestein/convolution pad selection.** _(2026-07-25)_ The
      follow-up above, done. The single `bluesteinSubFFTPenalty` constant is
      gone: measured against the power-of-two endpoint of its own dyadic
      window, a mixed-radix sub-FFT's cost per m·log2(m) point-pass spans ~7×
      on shape alone (`BenchmarkBluesteinPadShapes`, new, i7-1255U/AVX2,
      complex64: 3072 = 2^10·3 → 0.83, 2560 = 2^9·5 → 0.96, 3584 = 2^9·7 →
      1.39, 2160 = 2^4·3^3·5 → 2.31, 3000 = 2^3·3·5^3 → 2.87, 2250 = 2·3^2·5^3
      → 6.18), so no scalar penalty can be right for all of them — which is why
      2.2 had to disable every candidate. A deep power-of-two part lands the
      schedule in a tuned codelet leaf and each surviving odd stage is overhead
      on top, so the model is now a whitelist of candidate shapes
      (`padShapes`, `plan_padsize.go`), each admitted only above the pad size
      where it wins at **both** precisions.
      `BenchmarkBluesteinPadFamilies` (new) calibrated the thresholds over ten
      windows 2^7…2^16 as candidate ns/op over the window's power-of-two
      endpoint (c64/c128): `3·2^(k-2)` turns over at 2^9 (0.71/0.87) and holds
      to 2^16 (0.41/0.46), 2^8 being a wash for complex128 (0.78/1.00);
      `15·2^(k-4)` turns over at 2^13 (0.80/0.69) and holds to 2^16
      (0.74/0.75), 2^12 still losing 13% on complex64. The third family
      `7·2^(k-3)` is admitted by no threshold — it loses to `15·2^(k-4)` in
      every window where either wins (a full-matrix radix-7 butterfly costs
      more than the radix-3 plus radix-5 pair) and, being the smaller of the
      two, it is reachable only when `15·2^(k-4)` is reachable as well, so it
      is dominated outright. Multi-odd-stage shapes (2^a·3^3·5, 2^a·3·5^3, …)
      lose to `3·2^(k-2)` wherever both are reachable and are not candidates
      either. Both callers of the shared model benefit: `bluesteinPadSize` and
      `fastConvolutionLength` (convLen 257 → 384 instead of 512).
      End-to-end `Plan.Forward` (`BenchmarkBluesteinPadModel`, new; both arms
      interleaved in one process by emptying `padShapes` around `NewPlan`,
      medians of 5, c64/c128 vs the power-of-two pad): n=677 → 1536 0.74/0.85,
      n=2531 → 6144 0.43/0.47, n=3079 → 7680 0.80/0.67, n=4099 → 12288
      0.44/0.47, n=6151 → 15360 0.78/0.70, n=8209 → 24576 0.43/0.46 — i.e.
      −15…−57%, with the unchanged control n=1009 measuring 1.00/1.00. The
      per-size ratios track the sub-FFT calibration to within a point or two,
      so the win reaches the user rather than stopping at the sub-FFT.
      Zero-alloc preserved on the new mixed-radix padded path for both plans
      and `Convolver` (`plan_bluestein_norace_test.go`; the assertions are
      `!race`-tagged for the same reason the Rader/radix-7/11 ones are —
      pooled mixed-radix scratch does not survive race instrumentation);
      verified vs `reference.NaiveDFT` at the new pad sizes, plus an invariant
      sweep over every n ≤ 5000 pinning that the pad covers 2n−1, never
      exceeds the power of two it replaces, and is a length both the raw engine
      and the planner accept. Follow-ups: (a) the calibration is AVX2-only —
      the purego build passes but was not re-measured, and since SIMD
      accelerates the power-of-two baseline more than the mixed-radix engine
      the thresholds are conservative there, so a purego pass may lower them;
      (b) shapes filling the (0.9375P, P] gap (e.g. `63·2^(k-6)` = 0.984P) are
      unmeasured, as are the 2^17+ windows.
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
      _Update 2026-07-25:_ the follow-up "extend `RaderEligible` to primes
      with 7/11-smooth p−1" is done. Since the radix-7/11 item below landed,
      the mixed-radix engine executes p−1 exactly for those primes too, so
      `RaderEligible` now gates on `IsMixedRadixSmooth(p−1)`; the sub-FFT,
      table, scratch and zero-alloc paths needed no change (verified vs
      `reference.NaiveDFT` + round-trip at 42 primes spanning every p−1
      shape, and `TestRader_ZeroAlloc` extended to 353 → [11, 32] and
      2269 → [7, 4, 3, 3, 3, 3]). Radix-7/11 stages are full-matrix DFT
      butterflies, so they needed their own win gate rather than the
      5-smooth one — `rader7Or11Wins`, fitted on
      `BenchmarkRader7And11VsBluestein` (32 primes × both precisions,
      i7-1255U/AVX2) and consistent with all 32 measured shapes:
      p−1 ≥ 2048 wins whenever its power-of-two part is ≥ 4 (2113, 2269,
      2689, 2801, 4201, 4481, 6337, 7057, 7393, 9857, 9901, 12097, 12601,
      14081, 15121, 30241: 1.1–3.4×), while below 2048 only a single
      radix-7/11 stage — optionally with one radix-3 — on a deep power-of-two
      chain wins (113, 353, 449, 673, 1409: 1.1–2.0×). Everything shallower
      or odd-heavier measured 0.34–1.06× and stays on Bluestein
      (power-of-two part ≤ 2 at any size: 23, 127, 463, 2311, 22051;
      ≤ 4 below 2048: 29, 197, 701; 8 below 2048: 89, 281, 1321; odd part > 33 below 2048: 881, 1009, 2017). Purego spot-check on the thinnest
      newly-eligible margins (rule 4): 1.14–2.05× at 113/449/673/7057/9901/
      15121 across both precisions, with 881 and 2017 as controls measuring
      0.99–1.00× (the gate leaves them on Bluestein, as intended). One
      exception is recorded rather than papered over: 7393 regressed 9% on
      purego complex64 (its complex128 arm won 1.27× and both AVX2 arms won
      1.11–1.36×); tightening the gate to exclude odd-heavy shapes would
      also discard 9901's 2.05× purego win, so it stays in. Still skipped
      for the same reason as before: padded Rader for non-smooth p−1.
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
      _Update 2026-07-25: this auto-rule change has been reverted — see the
      P4.3 re-measure item below. Split-radix itself, the strategy plumbing,
      and its purego wins are unaffected; only the KernelAuto routing for
      power-of-two squares changed._
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
      _Update 2026-07-25 (second pass):_ every AVX2 codelet that the registry
      actually **selects** for its size is now fused. Cross-referencing the
      FMA-less `.s` files against the winning `Priority` per
      (Size, SIMDAVX2, Prec) left 7 files; all were converted, 97 sites total:
      `avx2_f64_size256_radix16.s` (64 sites, `VADDSUBPD` 64→0),
      `avx2_f32_size32_radix32.s` (19, and −95 lines),
      `avx2_f{32,64}_size8_radix4.s` (4+4),
      `avx2_f32_size16_radix16.s` (6), `avx2_f{32,64}_size384_mixed.s` (2+2).
      Trivial twiddles (±1, ±i — done with shuffle+sign-mask, no multiply) and
      the real-scalar 1/n inverse-normalization multiplies were deliberately
      left alone: they have no addend to fuse, so an FMA form would add work.
      `avx2_f32_radix3.s` was likewise left alone — its `half*t1` is a
      real-scalar multiply and its `coef*t2` product feeds both an add and a
      sub, so fusing would cost 2 copies + 2 FMAs to save 1 multiply.
      **Accuracy (deterministic, `cmd/measure_correctness`, fixed seed 42 —
      identical input vectors both arms):** max relative error vs the reference
      DFT improved at exactly the sizes carrying the most fused twiddle work —
      c64 size 16 2.35e-06→1.68e-06 (−28.5%), c64 size 32 2.41e-05→1.49e-05
      (−38.2%); c128 sizes 8/256 moved by +1.4%/+0.5% in the last digits and
      every other size was bit-identical. This is the expected one-rounding-
      instead-of-two effect.
      _(Metric caveat, added with the `measure_correctness` fix in P5.0 below:
      these figures came from that tool's old per-bin max-relative metric, which
      divides each bin by its own magnitude and is unstable. The direction is
      right — one rounding instead of two — and the c128 bit-identity result
      stands, but the absolute values are not comparable to the relative-L2
      figures reported elsewhere in this document.)_
      **Performance: not demonstrated.** Three benchstat attempts on the
      i7-1255U were inconclusive — the package hits 86–98 °C under sustained
      benchmarking and throttles, giving ±30–90% run-to-run variance that
      swamps the effect. A first sequential A/B suggested wins (c64 size-8
      −13%, size-32 forward −6.7%, c128 size-384 −5.4%/−7.3%) but ran the two
      arms under drifting thermal conditions; an interleaved re-run could not
      confirm them. Treat this pass as instruction-count and accuracy work
      until it can be re-measured on a thermally stable machine — it does
      **not** yet satisfy rule 5 of the methodology gate above.
      Correctness is solid: full `internal/kernels` suite green (incl.
      `-race`), `just vet-arch` clean on amd64/arm64/386/purego, zero-alloc
      preserved on every touched codelet.
      Remaining: (a) the non-codelet AVX2 dispatch sites
      (`internal/fft/complex_mul_amd64.go`, `kernels_amd64_asm.go`,
      `scale_amd64.go`, `internal/kernels/radix5_avx2.go`) still gate on
      `HasAVX2` alone and need the `HasAVX2 && HasFMA` sweep; (b) the
      FMA-less files that no size currently selects
      (`avx2_f32_size512_radix16x32.s` 128 muls,
      `avx2_f{32,64}_size1024_radix32x32.s`, `avx2_f32_size256_radix16.s`,
      `avx2_f32_size128_radix2.s`, `avx2_f32_size32_radix4_then2.s`,
      `avx2_f{32,64}_size4_radix4.s`) — fusing those only matters if a
      priority retune would bring them back into play.
      _Correction to the earlier "Remaining" list:_ the generic
      radix-4/Stockham kernels do **not** need a pass — they are already
      fused (`avx2_f32_generic.s` 22 mul/23 fma,
      `avx2_f{32,64}_generic_radix4_{even,odd}.s` ≈1:1,
      `avx2_f64_stockham.s` 6/6).
- [x] **Codelet priority retune (measured).** _(2026-07, i7-1255U)_ New
      `BenchmarkCodeletCandidates64/128` (internal/kernels) times every
      registered codelet per size, exposing systematic mis-selection: the
      priority-favored six-step / radix-32×32 / radix-16 / radix-8 codelets
      lose to the plain radix-4 family at every size where both exist, in
      both the AVX2 and generic tiers. Flipped priorities in
      `cmd/gencodelets/specs.go` (AVX2 c64: 256 radix16→radix2 −26%,
      512 radix8→radix2 −26%, 1024 radix32x32→radix4 −52%,
      4096 sixstep→radix4 −26%, 8192 sixstep→radix4_then2_params −55%,
      16384 sixstep→radix4 −32%; AVX2 c128: 256 radix4→radix16 −8/−24%;
      generic both precisions: radix4/radix4_then2 over sixstep/radix32x32
      at 1024/4096/8192/16384). The c128 1024 radix-32×32 AVX2 codelet is
      disabled (priority −1): its inverse ran 2× slower (9.3 µs) than the
      SSE2 radix-4 (5.1 µs) that now serves, while its forward was within
      4%. End-to-end (vs FFTW bench harness): c64 forward 1024 −52%,
      8192 −57%, 16384 −35%; c128 inverse 1024 −49%, 256 −30%. Six-step
      codelets stay registered for wisdom/measure to pick where they win.
- [x] **Size-32768 codelets (both precisions, generic + AVX2).** _(2026-07,
      i7-1255U)_ 32768 = 2·4^7 had no codelets at all — plans fell back to
      generic Stockham, which for complex128 is scalar Go on amd64
      (no c128 Stockham asm): 618 µs forward vs FFTW's 123 µs, a 5× cliff.
      New radix-4-then-2 codelets (7 radix-4 stages + radix-2 combine):
      generic Go `dit_32768_radix4_then2.go` ping-pongs between scratch and
      dst instead of per-stage stack arrays (which would exceed the 128 KiB
      stack-alloc limit; src is fully consumed by stage 1's bit-reversed
      loads, so dst is safe from stage 2 even in-place), c128 twin
      generated. AVX2 `avx2_f{32,64}_size32768_radix4_then2.s` follow the
      8192 loop-based pattern but take the digit-reversal table as a
      `bitrev []int` argument (an embedded DATA table would add 256 KiB to
      the binary) via thin wrappers binding internal/kernels' shared table.
      Measured end-to-end at 32768: c128 618→237 µs fwd (2.6×),
      601→257 µs inv; c64 225→168 µs fwd (−26%), 221→170 µs inv (the c64
      baseline was the generic AVX2 Stockham asm). Validated per-direction
      vs the naive reference DFT (generic) and vs the generic codelet
      (AVX2), plus round-trip and in-place aliasing tests. Follow-ups:
      SSE3/SSE2 tier for 16384/32768 (SSE tier currently stops at 8192),
      and 65536+ remain Stockham/split-radix territory.
- [x] **AVX2 complex128 Stockham asm.** _(2026-07, i7-1255U)_ Every
      Stockham-resolved complex128 size (>1024, non-square — 65536, 131072,
      524288, 2^21, …; codelets cover ≤32768) previously ran the scalar Go
      Stockham kernel on amd64. New
      `Forward/InverseAVX2StockhamComplex128Asm`
      (`internal/asm/amd64/avx2_f64_stockham.s`) mirror the f32 Stockham
      structure — ping-pong buffers, contiguous fast path for stage 1,
      running-offset strided twiddle gather, scalar tail — with two
      complex128 per YMM and the f64 VMOVDDUP/VPERMILPD/VFMADDSUB231PD
      (VFMSUBADD for inverse) multiply idiom; 4-slice signature, no bitrev
      argument. Wired in `internal/fft/asm_amd64.go` with Go fallback for
      n < 16. Kernel-level (asm vs scalar Go, fwd/inv): 2048 −50/−55%,
      16384 −50/−45%, 65536 −38/−33%, 131072 −24/−32%, 524288 −29/−12%,
      2^21 −16/−27% (large sizes go memory-bound). End-to-end vs FFTW:
      65536 1.44 ms→1.02 ms (gap 4.4×→3.1×), 524288 gap now 2.0×.
      Noticed in the same run: complex64 131072 end-to-end appeared slower
      than complex128 (3.77 ms vs 2.46 ms) — re-measured 2026-07-24 on an
      idle machine and it does not reproduce (c64 ~1.2 ms vs c128 ~2.1 ms,
      3× consistent); the original reading was a contaminated measurement,
      no selection bug.
- [x] **SSE3/SSE2 tier for 16384/32768 (both precisions).** _(2026-07,
      i7-1255U)_ The SSE codelet ladder stopped at 8192, so pre-AVX2 hosts
      ran the generic Go codelets at 16384/32768. Four new kernels —
      `sse3_f32_size16384_radix4.s`, `sse3_f32_size32768_radix4_then2.s`,
      `sse2_f64_size16384_radix4.s`, `sse2_f64_size32768_radix4_then2.s` —
      emitted by a one-off Go generator whose stage templates were
      validated by regenerating the existing 4096/8192 SSE files and
      diffing (instruction-for-instruction identical). 16384 embeds no new
      table (reuses the AVX2 files' `bitrev16384_r4`); 32768 takes
      `bitrev []int` as an argument like its AVX2 twin, bound via wrappers
      in `dit_32768_radix4_then2_amd64_sse.go`. Registered at priority 12
      (SIMD level dominates priority, so these only serve pre-AVX2 hosts);
      case 16384 also added to the forced-DIT SSE dispatch in internal/fft
      (32768 is registry-only, matching AVX2). Codelet-candidates bench vs
      the generic codelets they displace (fwd/inv): c64 16384 −41/−47%,
      32768 −15/−42%; c128 16384 −41/−23%, 32768 ±0/−38%. Noted: c128
      16384 SSE2 forward measured faster than the AVX2 codelet (104 vs
      124 µs) on this machine — the wisdom cache can exploit that; a
      priority flip would need re-measurement on more hardware. Validated
      per-direction vs the generic codelets plus exact in-place aliasing
      tests (`dit_sse_16384_32768_test.go`).
- [x] **Three missing codelets via subagent delegation.** _(2026-07,
      i7-1255U; opus/sonnet/haiku experiment — see AGENTS.md “Delegating
      Codelet Work to Subagents”)_ New hand-written kernels, each verified
      against the reference DFT via the registry-driven test suite:
      `avx2_f64_size1024_radix4.s` (closes the gap where AVX2 hosts fell
      back to SSE2 at c128-1024 because the radix-32×32 entry is disabled;
      priority 35; idle-machine bench fwd/inv 4165/4354 ns vs SSE2
      4802/5368 ns), `avx2_f64_size128_radix4_then2.s` (priority 25, fwd/inv
      413/443 ns — fastest c128-128 candidate, beats the radix-2 AVX2 at
      484/584 ns), and `sse3_f32_size256_radix2.s` (priority 10, below the
      radix-4 SSE3 at 12 as intended; wisdom-selectable alternative). All
      reuse existing bitrev tables (`bitrev1024_r4`, `bitrev128_mixed`,
      `bitrev256_r2`) and core.s scale constants — no new data symbols.
- [x] **Batch 2 subagent codelets: NEON ladder extension + AVX2 c128 512
      radix-8.** _(2026-07, subagent delegation round 2)_ Three NEON kernels
      close the biggest arm64 gap (the size-specific ladder stopped at 256;
      512/1024 ran the priority-1 generic radix-2 fallback):
      `neon_f32_size1024_radix4.s`, `neon_f64_size1024_radix4.s` (both
      priority 28), and `neon_f64_size512_mixed24.s` (priority 24). Each
      embeds its own file-scoped bitrev table (NEON files use `<>` static
      symbols — not shareable across files); the c64 1024 table was verified
      byte-for-byte against amd64's `bitrev1024_r4`. Correctness validated
      under QEMU (roundtrip, in-place, reference DFT; full kernels + asm/arm64 + root packages); priorities are ladder-mirrored, NOT tuned — QEMU
      timing is meaningless, so real-arm64 benchmarking remains open. On the
      amd64 side, `avx2_f64_size512_radix8.s` (priority 30) beats the previous
      best c128-512 codelet `dit512_radix4_then2_avx2` by 21–30% (idle-machine
      fwd/inv 1719/1860 ns vs 2164/2639 ns) using the byte-pointer addressing + FMA idiom lessons from the 1024 radix-4 kernel; it now auto-selects on
      AVX2 hosts (plan_api_test.go updated accordingly).
- [x] **Batch 3 subagent codelets: NEON ladder completed through 4096.**
      _(2026-07, subagent delegation round 3)_ Five NEON kernels, closing every
      remaining size-specific gap up to 4096 on arm64:
      `neon_f32_size512_mixed24.s` (c64 512 previously ran the priority-1
      radix-2 fallback), `neon_f32_size2048_mixed24.s`,
      `neon_f64_size2048_mixed24.s` (all mixed-2/4, priority 24), and
      `neon_f32_size4096_radix4.s`, `neon_f64_size4096_radix4.s` (priority
      28). All QEMU-verified (roundtrip, in-place, reference DFT; full
      kernels + asm/arm64 + root packages) plus native suites; priorities are
      ladder-mirrored, NOT tuned (see the open item below). Permutation
      tables were cross-checked against `internal/math` helpers
      (`ComputeBitReversalIndicesRadix4` / `...Mixed24`) or the
      twin-precision file's table (tables are precision-independent — the one
      bug of the round was a wrongly self-derived c64-512 table, fixed by
      copying the f64 file's table). `cmd/gencodelets/specs.go` hit the
      1500-line revive limit and was split: NEON rows now live in
      `cmd/gencodelets/specs_neon.go` (concatenated in `init()`; generator
      output unchanged).
- [x] **Batch 4 subagent codelets: NEON ladder completed to 32768.**
      _(2026-07, subagent delegation round 4; six sonnet agents, zero bugs
      reaching a test run)_ Six NEON kernels: `neon_f32_size8192_mixed24.s`,
      `neon_f64_size8192_mixed24.s`, `neon_f32_size32768_mixed24.s`,
      `neon_f64_size32768_mixed24.s` (priority 24) and
      `neon_f32_size16384_radix4.s`, `neon_f64_size16384_radix4.s`
      (priority 28). arm64 now has size-specific codelets at every power of
      two 4–32768 in both precisions, matching the amd64 SSE3 tier. All
      agents generated the asm programmatically, first validating their
      generator by byte-reproducing the existing 2048/4096 template, and
      pulled permutation tables straight from
      `internal/math.ComputeBitReversalIndices{Radix4,Mixed24}`.
      QEMU-verified (registry analytic patterns + roundtrip + in-place — the
      naive-DFT random check caps at 2048 by design) plus native suites.
      The dedicated `neon_f64_size_specific_test.go` naive-DFT check now
      skips sizes ≥8192 under QEMU/-short (it costs ~10 min emulated; runs
      fully on real arm64), with measured-error-based tolerances.
- [ ] **Size-384 path cleanup (both precisions).** Found while auditing the
      384 codelet for FMA (2026-07-25), verified independently — 384 is by far
      the worst ns/point in the registry (c128 fwd ~2.2 µs vs ~0.7 µs at
      size 256, and c64 fwd ~2.3 µs is _slower than c128_ at the same size,
      which should not happen for a half-width type). Three separate causes: 1. **The complex64 AVX2 asm for 384 is dead code.**
      `ApplyTwiddle384Complex64Asm`, `Radix3Butterflies384{Forward,Inverse}Complex64Asm`
      and `{Forward,Inverse}AVX2Size384MixedComplex64Asm` are defined in
      `internal/asm/amd64/avx2_f32_size384_mixed.s` and declared in
      `decl.go`, but have **zero callers** outside `decl.go` — confirmed by
      grep, and confirmed empirically (FMA-fusing that file changed the
      size-384 c64 benchmark by nothing). `forwardDIT384MixedComplex64` in
      `internal/kernels/dit_384_decomp_128x3_amd64_asm.go` instead does the
      radix-3 column DFT and the twiddle multiply in **scalar Go**
      (`butterfly3ForwardComplex64`, `work[i] *= twiddle[...]`). The
      complex128 twins _are_ wired up. Either wire the c64 asm in or delete it. 2. **complex128 uses the slower 128-point sub-kernel.** The c128 path
      calls `amd64.ForwardAVX2Size128Radix2Complex128Asm` (plain radix-2,
      7 stages) for its three 128-point sub-FFTs, while the c64 path
      correctly uses `...Size128Radix4Then2Complex64Asm`. The c128
      radix-4-then-2 variant exists and is declared — this looks like a
      straight oversight. 3. **complex64 rebuilds per-call scratch.** The c64 path does
      `make([]complex64, stride)` twice per transform (sub-twiddle and
      sub-scratch); these currently stay on the stack so the zero-alloc
      tests still pass, but the c128 path precomputes the 128-point twiddle
      once at init and pools both buffers via `sync.Pool`. Mirror that.
- [ ] **NEON priority tuning on real arm64 hardware** — the ladder
      priorities added above (batches 2–4: 512–32768, and the 512/1024
      generic-fallback relegation) were mirrored from smaller sizes; needs
      benchmarking on Apple Silicon/Graviton. Above ~8192 the DIT codelets
      also compete with the Go six-step path on real hardware (cache
      behavior differs from QEMU) — measure before trusting the 24/28
      priorities there.
- [ ] **AVX-512 higher-radix / per-size-tuned variants** (carried over from
      P2.4). The shipped AVX-512 tier is generic radix-2; a radix-4 AVX-512
      kernel should widen the 1.2–2.4× gap and could reclaim size 2048 and
      the complex128 sizes where AVX2 codelets still win.

      **No longer blocked on hardware (2026-07-25).** An AVX-512 host is
              now reachable — Intel Xeon Gold 5218 (Cascade Lake; `avx512f`,
              `avx512dq`, `avx512cd`, `avx512bw`, `avx512vl`, `avx512_vnni`) — and
              the AVX-512 assembly ran there for the first time. Until now every
              AVX-512 test had been skipping at runtime via `cpu.DetectFeatures()`,
              so `internal/asm/amd64/avx512_f{32,64}_generic.s` had **never
              executed**, on any machine. Result: the whole AVX-512 test set passes
              with zero skips, and `go test ./...` is green on that host. The
              assembly is correct; what follows is purely a tuning question.

              Measured against the best AVX2 codelet at each registered size
              (complex64, `BenchmarkCodeletCandidates64`, pinned, idle host):

              | size  | AVX-512 fwd | best AVX2 fwd | fwd Δ  | AVX-512 inv | best AVX2 inv | inv Δ      |
              | ----- | ----------- | ------------- | ------ | ----------- | ------------- | ---------- |
              | 1024  | 9151 ns     | 8210 ns       | +11.5% | 10662 ns    | 10141 ns      | +5.1%      |
              | 4096  | 40786 ns    | 39726 ns      | +2.7%  | 45995 ns    | 50651 ns      | **−9.2%**  |
              | 8192  | 89129 ns    | 83269 ns      | +7.0%  | 96315 ns    | 102567 ns     | **−6.1%**  |
              | 16384 | 199838 ns   | 188084 ns     | +6.2%  | 221941 ns   | 233577 ns     | **−5.0%**  |

              Three things follow. (1) The AVX-512 codelets are registered at
              **Priority 10** against 24–28 for AVX2, so the registry never selects
              them even on an AVX-512 CPU — for _forward_ that is currently the
              right call, but it discards a real 5–9% on _inverse_ at ≥ 4096.
              (2) The AVX-512 codelet is **radix-2** while every AVX2 winner here is
              **radix-4**, so this table is measuring an algorithm gap, not a vector
              width gap — which is exactly what the radix-4 work above is for, and
              raises the prior that it will pay off. (3) Coverage is complex64 only,
              at 1024/4096/8192/16384; `cmd/gencodelets/specs.go` has **no
              `Target: "avx512"` rows for complex128** at all.

              Caveat on the host: Cascade Lake downclocks under AVX-512, so this is
              a pessimistic machine for the tier — a client Ice Lake or later part
              would likely flatter it. Do not retune priorities from this host
              alone. It is also a 2-vCPU VM with no gcc, so it cannot build the cgo
              FFTW baseline; comparisons there are algo-fft-internal only.

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
- [ ] **SSE2 tier breadth.** The non-AVX2 tier had tuned kernels only up to
      1024; on SSE-only hardware the hot sizes above that fell back to the
      generic Go size-codelets (which outrank the plain SSE asm loop in the
      registry: SIMD level first, then priority). Sub-tasks:
  - [x] **Profile the gap.** _(2026-07, i7-1255U)_ Established the actual
        SSE-only paths and baselines: at 2048 the generic Go codelet runs
        17.7/20.7µs fwd/inv (complex64) and 11.9/14.9µs (complex128); at
        4096 the generic Go six-step codelet runs ~79/86µs fwd/inv for both
        precisions. Also found `ForwardSSE2Complex128Asm` is a return-false
        stub, so complex128 ≥2048 ran pure Go.
  - [x] **Size-2048 kernels (both precisions).** _(2026-07)_ Radix-4-then-2
        (2048 = 4⁵·2): `sse3_f32_size2048_radix4_then2.s`,
        `sse2_f64_size2048_radix4_then2.s`, registered as
        `dit2048_radix4_then2_sse3/sse2` (priority 12). complex64 −46/−49%
        (1.8–2.0×), complex128 −12/−26% vs the generic codelet.
  - [x] **Size-4096 kernels (both precisions).** _(2026-07)_ Pure radix-4
        (4096 = 4⁶): `sse3_f32_size4096_radix4.s`,
        `sse2_f64_size4096_radix4.s`, registered as
        `dit4096_radix4_sse3/sse2`, reusing the AVX2 `bitrev4096_r4` digit-
        reversal table. vs the generic six-step codelet: complex64
        49.9/53.2µs (1.59/1.64×), complex128 52.3/60.3µs (1.52/1.41×).
  - [x] **Re-check 256 complex64.** _(2026-07)_
        `ForwardSSE3Size256Radix4Complex64Asm` existed and was wired into
        the SSE3 fallback dispatch, but had no registry entry — the
        plan-level codelet lookup on SSE-only hardware served 256 from the
        generic radix-16 Go codelet instead. Benchmarked: asm 854/964 ns
        fwd/inv vs generic radix-16 1421/1938 ns (1.6/2.0×); registered as
        `dit256_radix4_sse3` (priority 12).
  - [x] **Size-8192 kernels (both precisions).** _(2026-07)_ Gate
        measurement first: at 8192 the straight-line `dit8192_radix4_then2`
        generic codelet (~90/107µs c64, ~67/77µs c128 fwd/inv) clearly
        beats the priority-favored `dit8192_sixstep64x128` (~156–380µs c64,
        ~114/131µs c128) — the 128/256 KiB working set does not rescue
        six-step on the i7-1255U, so the DIT asm was justified.
        Implemented radix-4-then-2 (8192 = 4⁶·2):
        `sse3_f32_size8192_radix4_then2.s`,
        `sse2_f64_size8192_radix4_then2.s`, registered as
        `dit8192_radix4_then2_sse3/sse2` (priority 12), reusing the AVX2
        `bitrev8192_m24` digit-reversal table. vs the six-step codelet the
        SSE path previously served: c64 ~74/108µs, c128 ~70/64µs fwd/inv
        (1.5–2×).
  - [ ] **Validate on genuine SSE-only hardware.** All measurements so far
        force the SSE path on an AVX2-capable i7-1255U; spot-check the
        speedups (and the DIT-vs-six-step crossover) on a real pre-AVX2
        machine or a VM with AVX masked before calling the tier done.
        (Not planned: a complex64 tier for SSE2-without-SSE3 hardware — the
        complex multiply idiom needs `ADDSUBPS`, and SSE3 has been
        universal since ~2005; such machines keep the generic Go path.)

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
- [x] **Cache-blocked variants above L2.** _(2026-07)_ `internal/cpu` now
      exposes per-core L1d/L2 sizes (`DetectCaches`: Linux sysfs, conservative
      32K/256K defaults elsewhere, `SetForcedCaches` for tests). New
      `KernelFourStep` strategy (public enum, planner, dispatch, wisdom name
      "fourstep"): the rectangular generalization of six-step over
      `math.TransposeRect` (tiled out-of-place transpose), so any power-of-two
      n ≥ 4 splits as n1×n2 — including the non-square sizes six-step
      declines — with the split chosen by a cache-residency cost model over
      the detected L1d/L2 instead of the fixed √n (`fourStepSplit`,
      `internal/kernels/fourstep.go`); zero-alloc, both precisions
      (generated twin). Wisdom-tuning: Patient/Exhaustive measure modes now
      benchmark four-step, so per-machine wisdom picks it where it wins.
      Measured (i7-1255U, AVX2, complex64, forward): four-step ≈ six-step at
      square sizes (same row lengths), beats split-radix at 2^21…2^23
      (−7…−28%), ties plain Stockham at 2^23; the split sweep
      (`BenchmarkFourStepSplitSweep`) is flat (±7%), with the cache-derived
      choice within noise of the optimum, so the auto rule is unchanged —
      measure/wisdom remains the arbiter. Follow-up: SIMD row FFTs inside
      four-step (rows are contiguous; the row passes still use the scalar
      Stockham butterflies, the main handicap vs the monolithic kernels).
      The auto-rule re-measure this item asked for is done — next item.
- [x] **Auto-rule re-measure for power-of-two squares.** _(2026-07-25)_ The
      contradiction recorded above was real: the `KernelAuto` square branch
      was costing users at every size it could reach. `BenchmarkSquareAutoRule`
      (new, `plan_autosquare_bench_test.go`) measures all candidate strategies
      at the only power-of-two squares the branch reaches — 2^18, 2^20 and
      2^22 — in both directions, both precisions, both builds, arms adjacent
      in one process, medians of 5 (i7-1255U/AVX2). Outcome: **power-of-two
      squares are no longer special-cased at all**; they fall through to the
      plain size heuristic (Stockham). Non-power-of-two squares keep
      six/eight-step, unchanged and unmeasured here.
      Against the incumbent split-radix, Stockham wins or ties every arm on
      both builds bar one (purego 2^18 c64 forward, −3%, inside noise):
      SIMD 2^18 c64 3.39/3.28 ms fwd/inv vs 10.16/7.00, c128 6.71/6.84 vs
      6.87/8.39; SIMD 2^20 c64 31.0/26.2 vs 42.0/50.3; purego 2^20 c128
      20.2/21.7 vs 44.9/46.4. The eight-step branch fell to the same
      measurement at 2^22 c64: Stockham 157/171 ms vs 201/269 (SIMD) and
      102/113 vs 203/247 (purego), so powers of two skip it too.
      One dissenting arm is accepted knowingly rather than papered over:
      2^20 complex128 forward prefers six-step (39.3 ms) to Stockham
      (49.7 ms) on the SIMD build. A precision- and direction-blind rule
      cannot capture it, the size's other three arms favor Stockham, and
      Stockham still beats the split-radix it replaces there by 1.6×.
      Note the SIMD Stockham complex128 numbers are themselves depressed by
      the packed-route dispatch gap below; fixing that only widens the margin
      this rule was chosen on.
- [ ] **Packed Stockham is disabled on SIMD builds above the codelet range.**
      _(found 2026-07-25 while re-measuring the square rule)_
      `internal/transform/stockham_packed_toggle_simd.go` sets
      `stockhamPackedEnabled = false` for amd64/arm64/386, reasoning that "the
      hand-written codelet path is checked first and supersedes it". That
      holds only up to 32768 — the largest registered codelet. Above it a
      Stockham-resolved plan on a SIMD build has no codelet to supersede
      anything and falls through to the generic SIMD Stockham kernel, which is
      slower than the pure-Go packed radix-4 route the toggle disabled.
      Same benchmark, same machine, forced Stockham, SIMD vs purego ns/op:
      2^18 c64 3.39 vs 5.26 (SIMD wins, 0.64×) — but 2^18 c128 6.71 vs 3.84
      (**1.75× loss**), 2^20 c64 31.0 vs 23.5 (1.32×), 2^20 c128 49.7 vs 20.2
      (**2.46×**), 2^22 c64 157 vs 102 (1.54×). So the SIMD kernel only wins
      for complex64 at 2^18; everywhere above that the disabled route is
      faster, on both precisions. Fix is not simply flipping the toggle — the
      executor checks packed _before_ the bound kernel
      (`kernelExecutor.forward`, `plan_exec.go`), so enabling it wholesale
      would regress 2^18 complex64. Wants a measured per-precision/size
      crossover, then either a size gate on the toggle or a reordering of the
      two branches. Affects every Stockham-resolved size above 32768 on
      amd64/arm64/386, which after the rule change above includes the
      power-of-two squares.
- [x] **Twiddle-table bandwidth reduction.** _(2026-07, i7-1255U)_ Survey
      first: on SIMD builds the large-n strategies (Stockham, six-step/
      eight-step, four-step) already share the single n-entry base table
      between directions (kernels conjugate on load), and the per-size
      codelet layouts are small (≤1128 elements) and cached process-wide —
      no meaningful duplication there. The real duplication was the pure-Go
      packed radix-4 Stockham route (purego/WASM builds): plans held
      `packedForward` plus a fully conjugated `packedInverse` copy
      (~n entries each). (a) landed: the radix-4 stages now conjugate
      w1..w3 on load (the radix-2 stage already did),
      `InverseStockhamPacked` takes the shared forward table, and
      `ConjugatePackedTwiddles` is gone — per-plan packed-table memory
      halved, and the change was free or better (new
      `BenchmarkStockhamPacked` fwd/inv/round-trip at 4K/64K/1M, both
      precisions: geomean −3.1%, inverse −2…−6%; zero-alloc preserved).
      (b) quarter-table symmetry evaluated and declined for the scalar
      path: a constant-twiddle upper-bound experiment at 64K/1M showed
      removing _all_ table traffic wins ~20–24%, but an L1-resident
      tiny-table variant with the identical load mix isolated the
      cache-footprint share to only ~4% (64K) to ~8% (1M, noisy) — the
      rest is load-instruction cost that a quarter-table keeps and adds
      octant-decode ALU on top of, so scalar swizzling would be a net
      loss. Revisit only inside SIMD kernels (sign-flip/shuffle is nearly
      free there) if profiles ever show the base table competing with data
      at very large n.
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

## Priority 5 — Comparative Benchmark Findings (2026-07-25)

Everything above was measured against algo-fft's own history: is this build
faster than the last one? This section comes from measuring against **other
libraries**, which asks a different and less comfortable question, and it
found things the internal benchmarks structurally cannot see.

Source: [`go-fft-bench`](https://github.com/CWBudde/go-fft-bench) at
`a1fa607` — a full sweep of algo-fft `v0.7.0` against FFTW3, gonum, go-dsp
and takatoh over powers of two 8–32768 plus 23 hand-picked
non-power-of-two lengths, both precisions, both directions, default and
`purego` builds. Reproduce with `just sweep && just plot`; the raw JSON and
charts are committed there. Machine: i7-1255U, Go 1.26.1, `GOAMD64=v3`.

**Standing caveat for every number below**: one laptop, one sweep. The
harness interleaves libraries per length so thermal drift lands on all of
them equally — treat the _ratios_ as real and the absolute figures as
indicative. Anything acted on here gets re-measured with `benchstat` under
the §3 methodology, before and after.

Headline: at powers of two algo-fft is 0.63× FFTW3 (geomean) and roughly 8×
the rest of the Go field. At non-power-of-two lengths that lead collapses,
and the three items in P5.0 are defects rather than missing optimizations.

### P5.0 Defects — fix before any of the optimization items

- [x] **complex64 is _slower_ than complex128 at 20 of 23 non-power-of-two
      lengths** (ratios 0.68–0.95; worst 12000 at 0.68, then 257 at 0.70,
      704 and 1000 at 0.74). _(2026-07)_ Root cause found, and it is none of
      the three suspected: **Go's compiler does not implement scalar
      `complex64 * complex64` in single precision.** It widens all four
      float32 components to float64, multiplies in double precision and
      rounds the two results back — `CVTSS2SD ×4, MULSD ×3, VFMADD231SD, SUBSD, CVTSD2SS ×2`, twelve instructions against six for the same
      expression on complex128 (verified in the emitted assembly; addition,
      subtraction and conjugation are unaffected — only the multiply
      promotes). So any FFT stage written as scalar Go is _structurally_
      more expensive in complex64 than in complex128. Power-of-two lengths
      hide it by running entirely inside hand-written float32 SIMD codelets;
      the non-power-of-two routes cannot, because their odd-radix stages,
      chirp modulation and pointwise products are scalar Go. That also
      explains the size ordering the sweep shows and the audit reproduced:
      c64 loses worst where scalar stages dominate (12000 `[5,5,5,3,32]`,
      704 `[11,64]`, 1000, 3600) and still _wins_ where an asm leaf does the
      work (96 `[3,32]`, 768 `[3,256]`). The registry was ruled out first —
      c64 and c128 have codelets at exactly the same sizes and schedule
      identically, so there is no dead or unwired complex64 assembly here.
      Fix: `math.MulComplex64` multiplies the components directly (MULSS ×3,
      VFMADD231SS, SUBSS), applied across the mixed-radix driver, the
      radix-3/5/7/11 butterflies, the Bluestein/Rader glue and the pure-Go
      element-wise fallbacks; the chirp and filter products now route through
      the existing SIMD `ComplexMulArray`/`ScaleInPlace` entrypoints. The
      complex128 twins pick up `math.MulComplex128` (the plain operator)
      through genkernels' existing textual rewrite, with no generator change.
      Every c64 function on the non-power-of-two path is now at **zero**
      float32→float64 promotions, measured as an exact instruction-count A/B
      on a cleaned build cache: mixed-radix driver 92 → 0, `butterfly5*` 16
      → 0 each, `butterfly3*` 8 → 0 each, `butterfly7/11` 4 → 0,
      `bluesteinExecutor.forward/inverse` 40/60 → 0, `raderExecutor.inverse`
      30 → 0, `fft.BluesteinConvolution[c64]` 20 → 0. Accuracy measured
      against a float64 naive DFT over the flagged lengths (fixed seed, so
      both arms transform identical vectors) came out neutral-to-better —
      improved at 8 of 12 sizes (257 −23%, 1009 −39%, 2205 −34%, 3600 −37%),
      slightly worse at 3 (1000 +75%, 31 +7%, 768 +4%), all still inside the
      documented ~10⁻⁶ complex64 band, and the Go paths now round the same
      way the SIMD codelets already did. One test tolerance was corrected as
      part of this: `TestBluestein_MatchesReference`'s complex64 arm bounded
      the error at an absolute 1e-3 on a bin that grows to ~9.5e3, i.e. under
      one float32 ulp of headroom, so it was passing on luck; it is now
      relative to the bin magnitude.

                                Measured c64/c128 forward ratio (>1 = c64 slower, i.e. the defect),
                                        median of 14 process runs, ratio taken _within_ each run so thermal
                                        drift cancels — the machine was contended throughout, so treat the
                                        ratios as sound and the absolute times as indicative:

                                        | n     | route                        | before | after    |
                                        | ----- | ---------------------------- | ------ | -------- |
                                        | 704   | mixed-radix `[11,64]`        | 1.27   | **0.98** |
                                        | 1000  | mixed-radix `[5,5,5,8]`      | 1.26   | **0.91** |
                                        | 2205  | mixed-radix `[5,7,7,3,3]`    | 1.18   | **0.96** |
                                        | 3600  | mixed-radix `[5,5,3,3,16]`   | 1.19   | **0.91** |
                                        | 12000 | mixed-radix `[5,5,5,3,32]`   | 1.27   | **0.90** |
                                        | 257   | Rader, power-of-two sub-FFT  | 1.46   | 1.41     |
                                        | 1009  | Bluestein, power-of-two pad  | 1.29   | 1.25     |

                                        In absolute terms the complex64 arm gained 21–32% at 1000, 2205, 3600
                                        and 12000 (p ≤ 0.04, `benchstat`, consistent across two independent
                                        run orderings) while complex128 showed no significant change at any
                                        length. Note what did _not_ move: 257 and 1009 are exactly the two
                                        lengths whose sub-FFT is a power-of-two DIT rather than the mixed-radix
                                        engine, so their residual deficit is not in the glue this item covers —
                                        it is the power-of-two forward path, which is the next item.

                                        _Confirmed off-laptop (2026-07-25)._ Re-run on a 64-core SSE-only host
                                        (no AVX at all, so every codelet resolves to the SSE2/SSE3 or generic
                                        tier), pre-fix and post-fix trees built side by side and run ABBA-
                                        interleaved, four rounds each. complex64 improved at **all ten**
                                        lengths tested (p = 0.029 each, −4% to −31%); complex128 showed **no
                                        significant change at any length**, which is the signature the fix
                                        predicts, since the c128 twin's `MulComplex128` is still the plain
                                        operator. The c64/c128 ratio moved 1.14–1.46 → 0.97–1.04 at 704, 1000,
                                        2205, 3600, 9973, 12000 and 44100. The three that stayed high are 257
                                        (1.46 → 1.41), 1009 (1.45 → 1.36) and 2003 (1.28 → 1.18) — all three
                                        route through a power-of-two sub-FFT, so 2003 is a _third_
                                        reproduction of the next item rather than a miss in this one.

- [x] **The same promotion still costs the pure-Go power-of-two codelets.**
      The fix above deliberately stopped at the paths P5.0 named. The
      generic radix-2/4 DIT codelets in `internal/kernels` are still written
      with the `*` operator and hold most of the library's remaining 5738
      promotion instructions (`forwardDIT64Radix2Complex64` alone has 448).
      They are dead weight on the default build, where the AVX2 codelets win
      selection, but they _are_ the transform on `purego` and WASM — so this
      is the same defect, one build tag over. Mechanical to fix (swap `*`
      for `math.MulComplex64` in the complex64 sources and regenerate), and
      it should be measured on the purego build rather than the default one.

      _Fixed 2026-07-25._ 1378 scalar `complex64` products across 39 codelet
              sources now go through `math.MulComplex64`. The swap was done with a
              throwaway type-driven rewriter (`go/types` via `x/tools/go/packages`:
              rewrite a `*` only where the expression's *type* is `complex64`, splice
              the original file bytes so nothing else moves) rather than by regex —
              index arithmetic and `float32` scaling look identical to a pattern
              match and must not be touched. It was run once per build configuration
              (`amd64`/`arm64` × default/`purego`), which is what turned up the two
              build-tagged sources the default configuration hides:
              `dit_384_decomp_128x3.go` and `dit_8192_sixstep_64x128_amd64_avx2.go`.

              Three sites could _not_ be fixed by swapping the operator, because they
              live in `[T Complex]` bodies where `w * b` has type `T`. Those were
              monomorphized instead, following the existing
              `radix3TransformComplex64` / `radix5TransformComplex64` precedent:

              - `butterfly2` → `butterfly2Complex64`, which is what the 128- and
                512-point radix-2 codelets call. Those two were the largest remaining
                offenders after the swap (32 and 24 promotions each) precisely
                because their multiply was one inlined generic call away.
              - `radix4Transform` → `radix4TransformComplex64` (new file
                `radix4_complex64.go`), the fallback for power-of-4 sizes with no
                codelet. This one also drops the `any()`-typeswitch dispatch in
                `butterfly4Forward`, so its gain is not purely the promotion.
              - `ditForward` → `ditForwardComplex64`, mirroring the
                `ditInverseComplex64` that already existed. Worth noting the
                asymmetry that made this necessary: `inverseRadix4Then2Complex64`
                already delegated to the monomorphized inverse while
                `forwardRadix4Then2Complex64` delegated to the generic forward, so
                the forward fallback was paying a cost its own inverse was not.

              `genkernels` needed no change beyond three `excludedFuncs` entries: its
              existing `Complex64` → `Complex128` textual rewrite maps the new call
              sites onto `math.MulComplex128` (the plain operator) by itself.

              Measured float32→float64 promotion instructions in `internal/kernels`,
              counted as `CVTSS2SD` in `go tool objdump` of the `purego` test binary:
              **4622 → 162**, and zero in every non-test function reachable from a
              `complex64` codelet. What remains is generic `[go.shape.complex64]`
              instantiations (see the follow-up below).

              Plan-level `purego` benchmark, complex64, ABBA-interleaved against HEAD,
              12 runs per arm, `benchstat`:

              | n     | forward         | inverse         |
              | ----- | --------------- | --------------- |
              | 32    | −22.0%          | −25.5%          |
              | 64    | −30.7%          | −33.4%          |
              | 128   | −28.2%          | −31.7%          |
              | 1024  | −22.6%          | −37.0%          |
              | 2048  | −32.0%          | −23.3%          |
              | 4096  | −32.4%          | −29.9%          |
              | 8192  | −21.5%          | −33.8%          |
              | 16384 | −29.9%          | −36.2%          |

              Geomean over the whole 8…16384 ladder, both directions: **−24.4%**
              (p=0.000 at every size above except n=16, which is unchanged, and
              n=8/256, which move by ~2%). Two guard measurements came out as they
              should: the default SIMD build is unchanged at every size (all `~`,
              geomean +1.6% with no significant point), confirming these codelets
              really are dead weight there; and complex128 is unchanged (5 of 6
              `~`, one −8.7%), confirming the regenerated twins are arithmetically
              the same code.

              _Confirmed off-laptop (2026-07-26)._ Repeated on the idle non-throttling
              AVX2+AVX-512 host, same ABBA protocol, prebuilt `purego` binaries shipped
              over so nothing had to be installed. Geomean **−24.1%** against the
              laptop's −24.4%, p=0.000 at 20 of 22 points, with the same two
              non-movers in the same places (n = 16 `~`, n = 256 `~` forward /
              −9.3% inverse). Two machines three generations apart agreeing to within
              0.3% on the geomean, and agreeing on _which_ sizes do not move, is the
              signature of a code change rather than a measurement artifact — the
              laptop's thermal behaviour cannot produce that.

              **Accuracy costs 3–9% more RMS error, and measuring that took two
              attempts.** Dropping the double-rounded intermediate does make each
              complex64 multiply slightly less accurate, so the question is how much.
              Relative L2 error against a float64 naive DFT of the same float32 input
              vector, 200 trials per size, identical inputs in both arms:

              | n    | relRMS before | relRMS after | Δ     | peak-rel before → after |
              | ---- | ------------- | ------------ | ----- | ----------------------- |
              | 32   | 7.114e-08     | 7.562e-08    | +6.3% | 1.55e-07 → 1.68e-07     |
              | 128  | 8.580e-08     | 9.295e-08    | +8.3% | 1.79e-07 → 1.86e-07     |
              | 512  | 9.855e-08     | 1.067e-07    | +8.3% | 1.72e-07 → 2.01e-07     |
              | 1024 | 1.035e-07     | 1.122e-07    | +8.4% | 1.74e-07 → 1.62e-07     |
              | 2048 | 1.097e-07     | 1.196e-07    | +9.0% | 1.83e-07 → 2.09e-07     |

              Everything stays at ~10⁻⁷, i.e. around float32 epsilon (1.19e-07); the
              peak-normalised error is unchanged within trial-to-trial variation and is
              _lower_ after the change at n = 1024. complex128 is bit-identical before
              and after at every size. That is a real but sub-ulp cost for 21–37%.

- [x] **`cmd/measure_correctness` reports a misleading number and should be
      fixed or removed.** Measured with it instead, the change above looks like
      a 3× accuracy regression (n = 128: 3.95e-05 → 1.12e-04) — which is an
      artifact, and it nearly caused the work to be reverted. Two flaws:

      1. It maxes a _per-bin_ relative error `|got-want| / |want|` over every
                 bin of every trial, skipping only bins below 1e-10. Bins whose
                 magnitude is small but not tiny carry an absolute error set by the
                 _peak_ bin's rounding, so the ratio explodes. It is an extreme-value
                 statistic over an unstable quantity, and it is dominated by whichever
                 bin happened to land nearest a zero.
              2. Its "reference" is `reference.NaiveDFT`, which returns **complex64** —
                 a same-precision computation with its own accumulation error, not a
                 reference. So it partly measured the divergence between two
                 implementations rather than either one's error, which is exactly what
                 made a change to one side of the comparison look catastrophic.

              Report relative L2 error (`||got-want||₂ / ||want||₂`) against a float64
              DFT of the same float32 inputs, and report the mean over trials as well
              as the max. `internal/reference` needs a `NaiveDFT128` taking
              `[]complex64` for this; there is currently no float64 reference for a
              complex64 transform anywhere in the tree, which is why the tool ended up
              comparing complex64 against complex64 in the first place.

              _Fixed 2026-07-26._ Both metric and reference replaced. The new reference
              is `reference.NaiveDFTWide(src []complex64) []complex128` — **not**
              `NaiveDFT128`, as this item asked: that name is already taken by the
              `[]complex128`-input form and Go has no overloading. `NaiveDFT` is now a
              narrowing wrapper around it, which deletes a duplicated O(n²) loop; the
              loop moved verbatim, so its ~131 test call sites see bit-identical values,
              and `TestNaiveDFTWide_MatchesNaiveDFT` pins that with an exact `==`
              comparison rather than a tolerance so a future edit to the shared loop
              cannot silently move every reference baseline in the tree.

              The tool now reports, per trial, `relL2 = ||got-want||₂/||want||₂`
              aggregated as mean **and** max over trials, plus a peak-normalized
              `max|got-want| / max|want|` kept as a max. Both normalize by a
              whole-vector quantity, so the `1e-10`/`1e-15` bin-skipping thresholds are
              gone — they were the symptom, the per-bin normalization was the disease.
              The peak-normalized column earns its place because relL2 averages over `n`
              bins and so attenuates a single wrong bin by ~1/√n, which is exactly the
              failure a broken codelet produces (see the `KernelRecursive` twiddle-layout
              bug at the top of the CHANGELOG); it follows the existing
              `maxNormalizedError` idiom in
              `internal/kernels/codelet_reference_all_test.go`.

              Three further flaws turned up while fixing the two this item named:

              - The two precision arms drew from **separate RNG streams** (`Float32` in
                one, `Float64` in the other), so the columns were computed over different
                input vectors and could not be compared to each other. One draw now feeds
                both: rounded to float32 once, then widened back for the complex128 arm,
                which is exact. The complex64 arm is referenced against the *rounded*
                vector — referencing the unrounded draw would fold input quantization
                (~ε₃₂/√12 ≈ 3.4e-08, the same order as the transform's own error) into
                the budget and install an irreducible ~3.4e-08 floor under the very
                3–9% effects the tool exists to resolve.
              - `%.2e` could not resolve those effects either (1.11e-07 and 1.19e-07 both
                print as `1.1e-07`). Now `%.3e`.
              - The tool printed **no build configuration**, which is why this item's own
                `3.95e-05 → 1.12e-04` reads as a contradiction: HEAD still measures
                3.95e-05 at n = 128 on the default build, because the previous item
                changed only the pure-Go codelets that the default build never runs. The
                header now prints `arch`/`simd`/`purego`.

              Flags `-sizes`/`-trials`/`-seed` were added following `benchkernels`'
              conventions, since every accuracy question in P5.0 concerned lengths
              (257, 1009, 2205, 3600, 12000) the hardcoded power-of-two list cannot
              express. Seeding is per size, so probing one length reproduces its row from
              a full run — verified: `-sizes 16 -trials 100` reprints the full ladder's
              n = 16 row to every digit in both precisions, and two successive runs are
              byte-identical.

              **Validation.** complex64 `relL2 mean`, 100 trials, seed 42, on the
              `purego` build — the build the figures at lines 1382–1388 above were
              measured on, since that item's change only affects the pure-Go codelets —
              against those figures, which came from a separately written harness at 200
              trials:

              | n    | measured  | independent | Δ     |
              | ---- | --------- | ----------- | ----- |
              | 32   | 7.644e-08 | 7.562e-08   | +1.1% |
              | 128  | 9.146e-08 | 9.295e-08   | −1.6% |
              | 512  | 1.066e-07 | 1.067e-07   | −0.1% |
              | 1024 | 1.122e-07 | 1.122e-07   | 0.0%  |
              | 2048 | 1.196e-07 | 1.196e-07   | 0.0%  |

              Two independently written harnesses agreeing to within 1.6%, two of them to
              four significant figures, is the acceptance criterion met: the metric is not
              merely different from the old one, it reproduces a number that was arrived
              at another way. A wrong input recipe — referencing the unrounded float64
              draw — would instead show a systematic ~+7% inflation at every size. (The
              default AVX2 build reads 3–5% lower at the same sizes: 7.087e-08, 8.855e-08,
              1.082e-07, 1.074e-07, 1.144e-07. Different codelets, different rounding;
              this is the difference the new header exists to disambiguate.)

              The full default-build ladder runs
              0.41 → 1.00 × float32 ε monotonically from n = 8 to 4096 (one 0.7% dip at
              1024, inside trial noise), against the old metric's 3× zig-zag — and
              `relL2 max / relL2 mean` falls from 1.7 at n = 8 to 1.02 at n = 4096, i.e.
              the statistic tightens as the norm averages more bins, which is what a
              stable metric does and what the old extreme-value one could not.

              **One finding worth recording: the complex128 column measures the
              reference, not the FFT.** It grows as **O(n)** — 2.00× per doubling across
              seven consecutive doublings, 8.6e-16 at n = 8 to 6.7e-13 at n = 4096 —
              because `NaiveDFT128` builds each twiddle from an un-reduced angle
              `-2πkm/n` whose magnitude reaches ~2πn, so the phase argument's own
              rounding grows in proportion. The FFT alone would sit near float64 ε and
              grow like √(log n). This is flaw 2 again, one precision up, and it is
              documented in the tool's own output rather than fixed: a genuinely
              higher-precision reference means `math/big` at O(n²), which is hours, and
              compensated summation inside `NaiveDFT128` would move reference values for
              its existing test callers. It also bounds the complex64 arm's validity —
              the reference contributes ~1e-16·n there, negligible against ε₃₂ until
              n ≈ 10⁸. A possible follow-up, not urgent: an angle-reduced or compensated
              float64 reference would make the complex128 column mean something.

- [x] **Finish the promotion sweep outside `internal/kernels`.** The
      rewriter, run over the whole module, still reports scalar complex64
      products in three places the two P5.0 items never covered. They were
      left out deliberately: each needs its own benchmark to justify and
      validate, and folding them into a 40-file codelet diff would have
      shipped them unmeasured.

      - `internal/fft/real_repack.go` (7) and `real_recombine.go` (1) — the
                real-FFT recombination inner loop, four complex64 multiplies per
                output pair. Likely the highest-value of the three; measure with the
                real-FFT benchmarks.
              - `internal/transform/stockham_packed.go` (5), `combine.go` (2),
                `recursive.go` (1).
              - `internal/fft/mixedradix_avx2.go` (1) — missed by the first P5.0 item
                because it is build-tagged.

              Separately, a production binary still contains generic
              `[go.shape.complex64]` instantiations that promote and that no operator
              swap can reach: `stockhamForward`/`stockhamInverse`,
              `ditForwardBitrev`/`ditInverseBitrev`, `sixStepForward`/`sixStepInverse`,
              `eightStepForward`/`eightStepInverse`, `fourStepTransform` in
              `internal/kernels`, and `combineRadix4`/`combineRadix8`/`combineGeneral`
              in `internal/transform`. Fixing these means monomorphizing, as above —
              but they are _not_ on the default power-of-two plan path: profiling the
              `purego` 8192 and 16384 forward benchmarks puts 100% of samples in the
              monomorphized codelets plus inlined `MulComplex64`. Establish that a
              given one is hot before duplicating a function for it.

              _Done 2026-07-26._ All 17 sites fixed. Module-wide `CVTSS2SD` count
              **997 → 733** (−264, −26.5%); no function anywhere got worse. Complex
              products go through `math.MulComplex64`; real scaling is component-wise
              (`internal/fft` uses the SIMD-backed `ScaleComplex64InPlace` instead,
              being in the same package); `multiplyByI`/`multiplyByNegI` became
              swap-and-negate and now do no multiplying at all. Each `complex128` twin
              was mirrored through `MulComplex128` so the hand-written pairs stay
              line-for-line comparable, following `internal/fft/mixedradix.go`.

              Measured per area, 5 interleaved passes per arm with a cooldown before
              _each_ arm, because this laptop throttles hard under sustained
              benchmarking (the caveat recorded against the AVX2 FMA-fusion round
              above). Every table below has a negative control that came out flat,
              which is what says these are not thermal artifacts.

              **`internal/fft/real_repack.go` was indeed the highest-value of the
              three, by a wide margin.** Isolated on the scalar loop
              (`BenchmarkRepackInverseComplex64Generic`, added — the complex128 twin
              already existed):

              | half | 128    | 512    | 2048   | 8192   |
              | ---- | ------ | ------ | ------ | ------ |
              | c64  | −60.5% | −62.2% | −62.0% | −62.5% |
              | c128 | −4.0%  | −2.9%  | −4.8%  | ~      |

              Six promoting products per bin, so the loop runs **2.6× faster**. The
              complex128 side improves too — small but real (p ≤ 0.032 at three of
              four sizes): `det := 1 - 2*u` was a full complex multiply by `(2+0i)`,
              four products where component-wise doubling needs two. That was worth
              recording because it is the one place in this sweep where a
              `complex128` path was not already optimal.

              `real_recombine.go`, one product per bin, on the same footing
              (`BenchmarkRecombineForwardComplex64Generic`, already present):
              **−24.5% / −24.1% / −22.3% / −23.3%** at half = 128/512/2048/8192, with
              the complex128 twin flat at every size — the control that says the
              complex64 delta is the promotion and nothing else.

              End-to-end on `purego`, where these two loops are the whole
              recombination rather than a tail:

              | `PlanReal*/Real`    | 256    | 1024   | 4096   | 16384  |
              | ------------------- | ------ | ------ | ------ | ------ |
              | Forward (recombine) | ~      | −7.0%  | −7.0%  | ~      |
              | Inverse (repack)    | −34.8% | −30.9% | −26.2% | −26.7% |

              (`PlanRealForward/Complex_N=*`, the complex-FFT comparison arm of the
              same benchmark, is flat at all four sizes — the negative control.)

              **`internal/transform/stockham_packed.go`** — the packed radix-4
              Stockham engine, which is the Stockham route on `purego` and WASM and
              disabled on SIMD builds. Its benchmarks call the engine directly, so
              these are default-build numbers that hold on every build:

              | n              | 4K     | 64K    | 1M     |
              | -------------- | ------ | ------ | ------ |
              | forward c64    | −28.6% | −31.0% | −28.7% |
              | inverse c64    | −38.5% | −35.9% | −33.6% |
              | c128 (control) | ~      | ~      | ~      |

              complex64 geomean **−32.8%** across the six. The inverse gains more
              because it also carries the 1/n scaling loop.

              **`internal/transform/combine.go` + `recursive.go`** — `KernelRecursive`,
              all builds: forward **−13.8%** (2048) and **−12.4%** (8192), inverse
              **−24.5%** and **−28.6%**; geomean −20.1%, still allocation-free.

              **`internal/fft/mixedradix_avx2.go`** — default AVX2 build, complex64
              **−17.0%** at 3584 and **−14.8%** at 7168, flat at 385 and 1155 (odd-radix
              dominated, so the codelet-scaling loop is a smaller share), complex128
              unchanged. This path had **no benchmark at all**:
              `BenchmarkMixedRadix7And11VsBluestein` measures only the forward
              direction, and the loop in question is inverse-only.
              `BenchmarkMixedRadixInverse` was added to cover it, which also gives the
              44100 item below a handle on the inverse direction.

              **Correction to this item's premise: part of the "separately"
              list _was_ reachable by an operator swap.** Three of the 17 sites
              (`multiplyByI`, `multiplyByNegI`, `scaleComplexSlice`) are the concrete
              branches of `any(x).(type)` switches inside generic functions — and Go
              compiles *every* branch of a type switch into *every* shape
              instantiation. So each `complex64` branch was charging its promotion to
              the `complex128` instantiation as well. Fixing those three zeroed four
              functions from the "needs monomorphizing" list outright —
              `combineRadix4[c128]` 20 → 0, `combineRadix4Conj[c128]` 40 → 0,
              `recursiveInverseWithTwiddle[c64]` and `[c128]` 15 → 0 each — and cut
              `combineRadix4Conj[c64]` 100 → 60 and `combineRadix4[c64]` 80 → 60.
              Worth remembering as a general rule: a type switch is a place where
              concrete-typed code hides inside a generic body, so it is reachable
              without monomorphizing.

              What genuinely still needs monomorphizing is the type-parameter
              multiply `twiddle[k] * sub1[k]`, whose operands have type `T`. The
              remaining census, so the next attempt need not rebuild it
              (`CVTSS2SD` per function, default build; all are
              `[go.shape.complex64]` instantiations):

              | function                                          | count |
              | ------------------------------------------------- | ----- |
              | `transform.combineRadix4` / `combineRadix4Conj`   | 60 ea |
              | `fft.ditInverseStrided`                           | 48    |
              | `transform.combineRadix8` / `Conj`                | 40 ea |
              | `transform.combineGeneral` / `Conj`               | 40 ea |
              | `kernels.ditInverseBitrev`                        | 40    |
              | `kernels.Butterfly2` / `butterfly2`               | 24 ea |
              | `fft.ditForwardStrided`                           | 24    |
              | `transform.recursiveForwardWithTwiddle`           | 20    |
              | `transform.combineRadix2` / `Conj`                | 20 ea |
              | `kernels.ditForwardBitrev`                        | 20    |
              | `fft.mixedRadixTransform`                         | 20    |
              | `kernels.BluesteinConvolution`                    | 16    |
              | `fft.ComplexMulArray{,InPlace}` + `*Generic`      | 16 ea |
              | `fft.ScaleInPlace`                                | 16    |
              | `kernels.stockhamInverse`                         | 8     |
              | `kernels.{stockham,sixStep,eightStep,fourStep}*`  | 4 ea  |

              Two of these are free: `fft.ScaleInPlace`'s 16 are in its `default:`
              branch, which is unreachable under the `Complex` constraint, and the
              `ComplexMulArray*` counts are in generic fallbacks whose complex64
              callers already reach the SIMD entrypoints. The rest need the
              `radix3TransformComplex64` treatment.

- [x] **One fully-unrolled codelet was over the inliner's big-function
      threshold and paid a real `CALL` per complex multiply.**
      `internal/kernels.inverseDIT64Radix2Complex64` contained **193 `CALL`
      instructions to `math.MulComplex64`** — every un-inlined `MulComplex64`
      call in the entire module was inside that one function.

      _Done 2026-07-26._ Module-wide un-inlined `MulComplex{64,128}` calls:
          **193 → 0**.

          **Cause.** 64 of the 193 were not complex multiplies at all: stage 6
          scaled each of the 64 outputs by `complex(float32(1.0/64.0), 0)`, a
          _real_ factor, spending two dead products per output. Naming the
          unscaled butterfly results so they could be scaled took a further 32
          `f*` temporaries. Together those pushed the function past the node count
          above which Go's inliner switches to its big-function cost budget, so
          none of its products inlined.

          **Fix.** Apply the 1/n scaling component-wise in one pass over `work`,
          which makes stage 6 structurally identical to the forward codelet's. The
          rewrite is bit-identical, not merely close: `MulComplex64(f, complex(s,
          0))` reduces to `complex(real(f)*s, imag(f)*s)` exactly, confirmed over
          6M random values at s = 1/64, 1/3 and 1/256 including
          denormal-adjacent magnitudes.

          Per-function inline counts from `-gcflags=-m=2`, which pin the threshold
          between 129 and 193 products:

          | function                  | before    | after         |
          | ------------------------- | --------- | ------------- |
          | `inverseDIT64…Complex64`  | 0 / 193   | **129 / 129** |
          | `forwardDIT64…Complex64`  | 129 / 129 | 129 / 129     |

          `forward` compiles byte-identically before and after (34090 B both) — a
          compiler-level control that the change is confined to the inverse.

          Codelet benchmarks, 7 interleaved passes per arm, `benchstat` n = 12:

          | benchmark                    | base     | new      | change               |
          | ---------------------------- | -------- | -------- | -------------------- |
          | `Size64/Radix2/Inverse` c64  | 576.7 ns | 453.1 ns | **−21.4%** (p=0.000) |
          | `Size64/Radix2/Inverse` c128 | 513.3 ns | 371.9 ns | **−27.5%** (p=0.000) |
          | `Size64/Radix2/Forward` c64  | 290.2 ns | 287.4 ns | ~ (p=0.561)          |
          | `Size64/Radix2/Forward` c128 | 290.8 ns | 280.3 ns | ~ (p=0.173)          |

          Both forward arms are unchanged code and serve as controls; all four
          remain allocation-free. The complex128 twin gains _more_ than complex64
          because it never had un-inlined calls — its whole 27.5% is the removed
          dead arithmetic, which complex64 gets on top of the 193 eliminated
          calls.

          **The end-to-end gain is nevertheless zero, because this codelet is
          never selected.** At n = 64 the registry prefers
          `dit64_radix4_generic` (`Priority: 20`) over `dit64_radix2_generic`
          (`Priority: 0`), and radix-4 is still ~1.8× faster after the fix (c64
          inverse 252.7 ns vs 453.9 ns, forward 189.3 ns vs 288.5 ns), so the
          priority table is right and no selection change is warranted.
          `BenchmarkPlanInverse_64` on `purego` is flat (255.6 ns → 258.9 ns,
          p=0.558), and its 255.6 ns matches radix-4's 252.7 ns, confirming what
          the plan actually runs. What the fix buys is therefore a registered
          tuning candidate that no longer reads ~2× slower than it should —
          which is what `BenchmarkCodeletCandidates64` compares when setting these
          priorities — plus two generalizable lessons: scaling by a real factor
          must not be written as a complex multiply, and folding such a scale into
          a fully-unrolled stage can cross the inliner's threshold and silently
          un-inline _every_ helper in the function.

          **The item's own counts were off.** It cited "193 `CALL`s" and "the
          inverse's 161 products" as separate quantities; they are the same thing.
          161 came from a line-based `grep -c`, which undercounts the
          `work[k], work[k+32] = …, …` lines that carry two products each. The
          true figure is 193 products / 193 calls — none inlined, not merely
          some.

          Answering the item's own diagnostic: no other codelet crosses the
          threshold; the module-wide census is now 0. The census did turn up
          real-factor scaling elsewhere — see the next item.

- [x] **39 more real-factor multiplies remain, in codelets that _are_
      selected.** `grep -rn 'MulComplex64([a-z0-9]*, scale)' internal/` finds
      `dit_16_radix16.go` (16), `dit_256_radix16.go` (16),
      `dit_384_decomp_128x3.go` (3), `dit_384_decomp_128x3_amd64_asm.go` (3)
      and `splitradix.go` (1, inside a loop over all n). Each multiplies by
      `complex(scale, 0)`, spending two dead products plus an add and a
      subtract per element.

      _Done 2026-07-26._ All 39 rewritten component-wise, plus the 33
          generated `complex128` twins and the two hand-written `complex128` sites
          in the 384 files, which used the plain `y * scale` operator and carried
          the same dead arithmetic. The win is real but confined to two of the four
          site groups, and it is _not_ where this item predicted.

          **The dead arithmetic is genuinely emitted.** Every site scales by a
          compile-time constant, so the first question was whether the compiler
          already folds the zero imaginary part. It does not — a per-`STEXT`
          multiply/add-subtract census over `go build -gcflags=-S` for the whole
          module:

          | function                              | before  | after      |
          | ------------------------------------- | ------- | ---------- |
          | `inverseDIT16Radix16` c64 _and_ c128  | 64 / 32 | **32 / 0** |
          | `inverseDIT256Radix16` c64 _and_ c128 | 124 / 62| **92 / 30**|
          | `inverseDIT384Mixed` c64              | 20 / 10 | **14 / 4** |
          | `inverseDIT384Mixed` c128             | 12 / 6  | **10 / 4** |
          | `InverseSplitRadix` c64 _and_ c128    | 4 / 2   | **2 / 0**  |
          | `forwardDIT256Radix16`, `forwardDIT384Mixed` | 60/30, 8/4 | unchanged |

          At n = 16 that was the codelet's _entire_ float-op count: 96 → 32. The
          forward rows are unchanged code and are the compiler-level control. Both
          precisions had identical counts before the change, which is why the
          instruction-count claims in the new comments survive `genkernels`' blanket
          `Complex64`→`Complex128` rewrite honestly — the trap the previous item hit.

          Benchmarks, 6 interleaved passes per arm with arm order alternating per
          pass, `benchstat` n = 6:

          | benchmark                              | base     | new      | change               |
          | -------------------------------------- | -------- | -------- | -------------------- |
          | `Size16/Radix16/Inverse` c64           | 51.6 ns  | 43.1 ns  | **−16.5%** (p=0.041) |
          | same, `purego` binary                  | 54.7 ns  | 43.9 ns  | **−19.7%** (p=0.002) |
          | `Size16/Radix16/Inverse` c128          | 54.5 ns  | 48.6 ns  | **−10.8%** (p=0.002) |
          | `Size256/Radix16/Inverse` c128         | 2.005 µs | 1.761 µs | **−12.2%** (p=0.009) |
          | `Size256/Radix16/Inverse` c64          | 1.912 µs | 1.745 µs | −8.7%, n.s. (p=0.485)|
          | `InverseDIT384Mixed` c64 / c128        | 2.13 / 2.98 µs | 2.22 / 2.93 µs | flat (p=0.394 / 1.000) |
          | `SplitRadix*/Inverse`, 256…65536, both | —        | —        | flat, all 8 cells    |

          All eight forward directions are unchanged code and came out flat, which
          is what makes the four significant cells credible. The n = 16 complex64
          result replicates across two independently built binaries. Zero
          allocations preserved throughout.

          **The two flat groups are flat for structural reasons, not noise.**
          Split-radix applies its 1/n scaling in a separate `for i, v := range work`
          sweep over the output, which is memory-bound — arithmetic there is free,
          at every size from 256 (L1-resident) to 65536. The 384 codelet removes 6
          multiplies and 6 add/subtracts per iteration across 128 iterations, but
          that is invisible against three AVX2 128-point sub-IFFTs plus the
          de-interleave and copy passes. **Arithmetic removal only pays where the
          scaling is a large fraction of a small compute-bound kernel** — which is
          exactly the two fully-unrolled radix-16 codelets and was exactly the n =
          64 case above. That is the generalizable lesson; the raw site count (39)
          was a poor predictor of value.

          **This item's reachability claims were half right.** Verified by probing
          `registry.Lookup` per SIMD level for both precisions rather than reading
          priorities off the source:

          - n = 16 / 256 complex64: `dit{16,256}_radix16_generic` at `Priority: 30`
            are indeed selected on `purego`/WASM/non-amd64, as claimed. On the
            default build AVX2 asm wins at both sizes, so the Go codelets are
            candidates only.
          - **n = 256 complex128 is dead on every build.** The complex128 registry
            gives `dit256_radix16_generic` `Priority: 15`, not 30, so
            `dit256_radix4_generic` (20) wins on `purego` too. Those 16 generated
            sites are unreachable — a deliberate precision-specific tuning, not an
            inconsistency, but it means the item's "both register at `Priority: 30`"
            holds for complex64 only.
          - **n = 384 is selected on every build and both precisions** —
            `dit384_mixed` is the _only_ registered candidate at its size
            (`_avx2` 25, `_generic` 20). This looked like the one site with a
            default-build user-facing path, and it is; the measurement above is why
            that did not translate into a gain.

          **The end-to-end purego numbers this item asked for are not yet
          conclusive.** `BenchmarkPlanInverse_16` (72.3 → 57.2 ns, p=0.065) and
          `_256` (1.865 → 1.653 µs, p=0.093) both moved the right way by roughly the
          codelet margin, and the forward controls did not, but variance in the new
          arm reached ±110% on some cells and nothing cleared significance. A
          focused re-run of just those four cells is queued; treat the codelet
          numbers above as the established result and these as suggestive.

          Benchmark gaps closed along the way, all three of which are why these
          sites went unmeasured for so long: the size-16 radix-16 Go codelet had
          **no benchmark at all** despite being the selected `purego` codelet at
          n = 16; split-radix had none at kernel level; and a plain `NewPlan(384)`
          had none, only `BenchmarkBluestein_*_384` forcing the other route. Added
          `Size16/Radix16` to both DIT tables, `BenchmarkSplitRadixComplex{64,128}`
          over 256…65536, and `BenchmarkPlan{Forward,Inverse}_384`.

- [x] **The power-of-two complex64 _forward_ path underperforms its own
      inverse.** _Not reproduced — closed 2026-07-26; the effect was a
      measurement artifact. See the resolution at the end of this entry._
      Splitting the c64/c128 ratio by direction on the same run:
      at 128, forward gains 1.23× where inverse gains 3.36×; at 256, 1.67×
      vs 4.19×; at 4096, 1.14× vs 2.12×; at 8192, 1.07× vs 1.81×. The
      inverse codelets are extracting the width benefit the forward ones are
      not, at the same sizes, from the same registry. Whatever the inverse
      does differently is the fix — a like-for-like comparison inside one
      build should be cheap to localize. The first item's measurements
      sharpen the target: after the scalar-multiply fix, the only lengths
      where complex64 is still slower than complex128 are 257 (1.41×) and
      1009 (1.25×) — the two whose sub-FFT is a power-of-two DIT rather than
      the mixed-radix engine. They are this item seen through the
      arbitrary-length routes, and give it two extra reproductions (2003
      makes a third; see the first item).

      **It does not reproduce on any other machine (2026-07-25).** Checked on
              two further hosts: a 64-core SSE-only host (no AVX, so the SSE2/SSE3
              tier is what dispatch actually selects) and an idle Xeon Gold 5218
              (AVX2 + AVX-512). On both, forward is _faster_ than its own inverse at
              every power-of-two size — plan-level and codelet-level, complex64 and
              complex128 — which is the expected ordering, since the inverse carries
              the 1/N scaling. The SSE host measured c64 forward/inverse at
              0.84–0.97 across 8…8192. Since the algorithm and the Go code are
              identical across all three machines and only the selected SIMD tier
              differs, the anomaly localizes to **the laptop's AVX2 codelets**, not
              to the forward path in general. Re-confirm it on the laptop before
              spending time in the Go layer, and treat "which AVX2 codelet wins
              selection for forward vs inverse at these sizes" as the first place to
              look — the registry can pick a different codelet per direction.

          **Selection cannot differ by direction; that lead is closed
          (2026-07-26).** Settled structurally rather than by measurement, because
          the answer is in the type: `registry.CodeletEntry` holds `Forward` and
          `Inverse` in one struct and `Lookup` returns one entry per (size,
          features, precision), so both directions always come from the same
          codelet. No registration anywhere leaves one direction `nil` — the only
          way a direction could fall through to a different layer — the per-size
          ladders in `internal/fft/kernels_amd64_size_specific.go` have identical
          `case` sets for forward and inverse in both precisions, and
          `kernelExecutor.forward`/`.inverse` are mirror images. Confirmed at
          runtime for 128…8192: every size binds both directions from one
          signature. The registry cannot "pick a different codelet per direction".

          What _does_ differ is **precision**, at two of the four sizes: n = 256
          binds `dit256_radix2_avx2` for complex64 but `dit256_radix16_avx2` for
          complex128, and n = 8192 binds `dit8192_radix4_then2_params_avx2` against
          plain `dit8192_radix4_then2_avx2`. The complex64 priorities at 256
          (135 / 130 / 120) are outliers from an old tuning round, so **"is each
          incumbent still the fastest candidate, per direction?" replaces the
          original lead** as the first thing to measure.

          **Nothing structural makes the forward slower.** Static instruction
          census of the four incumbents' asm (data directives excluded — counting
          them inflated one inverse to 4496 against its forward's 386, an artifact
          of the inverse being last in its file): at 7 of 8 (size, precision) pairs
          the inverse carries 8–13 _more_ instructions than its forward, being the
          same code plus the 1/n scaling. That predicts the ordering the other two
          machines measured. The lone exception is n = 256 complex128 radix-16
          (forward 764, inverse 601).

          **Measured on the laptop, and it does not reproduce here either.**
          Canary-gated sweep, 36 accepted groups against 4 rejected; plan-level and
          codelet-level figures agree to within ~1% at every size, which is what
          makes them trustworthy. Forward against its own inverse (< 1 = forward
          faster = the expected ordering, since the inverse carries the 1/n scaling):

          | n    | c64 plan | c64 codelet | c128 plan | c128 codelet |
          | ---- | -------- | ----------- | --------- | ------------ |
          | 128  | 0.89     | 0.89        | 0.93      | 0.94         |
          | 256  | 0.92     | 0.85        | 1.08      | 0.95         |
          | 4096 | 1.03     | 0.97        | 0.94      | 0.95         |
          | 8192 | 0.95     | 0.96        | 0.94      | 0.99         |

          Forward is faster than its own inverse at every size in both precisions.
          The only two cells above 1.0 are marginal and each is contradicted by its
          own codelet-level twin. The laptop now agrees with the SSE-only host and
          the Xeon: there is no forward-path defect on any machine tested.

          **The original numbers were half right, and the wrong half was the
          inverse.** Re-measuring the c64/c128 gain per direction:

          | n    | item fwd | measured | item inv | measured |
          | ---- | -------- | -------- | -------- | -------- |
          | 128  | 1.23     | **1.28** | 3.36     | **1.17** |
          | 256  | 1.67     | **1.69** | 4.19     | **1.39** |
          | 4096 | 1.14     | 1.36     | 2.12     | **1.47** |
          | 8192 | 1.07     | 1.51     | 1.81     | **1.54** |

          The forward column reproduces almost exactly at 128 and 256; the inverse
          column collapses to roughly a third of what was claimed. "The inverse
          codelets are extracting the width benefit the forward ones are not" was
          reading a contended _inverse_ arm as signal — consistent with the source
          sweep's own standing caveat that the machine was contended throughout.
          What survives is that the c64/c128 gain is a fairly uniform 1.2–1.7×
          across both directions rather than the ~2× the register width suggests,
          and that belongs to the "complex64 buys nothing between 1024 and 16384"
          item below, not here.

          **Consequence for the arbitrary-length items.** This item claimed 257
          (1.41×) and 1009 (1.25×) were "this item seen through the arbitrary-length
          routes", with 2003 a third reproduction. That link is void: there is no
          forward-path defect for them to be a reproduction of. Their residual
          complex64 deficit needs its own cause — the one property they still share
          is a power-of-two DIT sub-FFT rather than the mixed-radix engine.

          **Two measurement lessons, each of which cost a discarded run.** Ordering
          benchmark cells with `sort -u` puts every `Candidates128` cell ahead of
          every `Candidates64` cell, which ran the whole complex128 arm inside the
          hot window left by a long test run and inflated it ~13× (one cell read
          26730 ns there against 2031 ns cooled) — a fake precision asymmetry,
          caught only because the figure disagreed with a value the previous item
          had already measured for that same function. And **a once-per-pass canary
          is not enough**: a 94-cell pass takes 5–13 minutes, so contention arriving
          mid-pass goes unseen, and 3 of 5 nominally-clean passes were contaminated,
          one by 50×. Only per-group bracketing — a canary cell with a known
          quiet-machine value timed before _and after_ each group, as in scratchpad
          `p47/gated.sh` — rejects that. It is how the tables above were obtained,
          with each group running both precisions and both directions back-to-back
          so every ratio comes from a single thermal window.

          Benchmark gap closed:
          `BenchmarkPlan{Forward,Inverse}_{256,4096}_Complex128_Focus`. complex128
          previously had plan benchmarks at 128, 512 and 8192 only, so two of this
          item's four sizes could not be measured in-tree at all — which is why its
          ratios came from an out-of-tree sweep to begin with. Also added
          `TestCodeletsRegisterBothDirections{64,128}`, which locks in the
          direction-symmetry property established above: a half-registered codelet
          would send one direction down the generic kernel ladder while the other
          kept the codelet, pass every correctness test, and surface only as an
          unexplained forward/inverse asymmetry — exactly the shape of this item.

- [x] **Audit whether each power-of-two incumbent is still the fastest
      registered candidate, per direction.** _Done 2026-07-26 for n = 256, 512
      and 8192 — all six incumbents confirmed, in both directions. The only
      code change is a cosmetic priority normalisation; see the resolution at
      the end of this entry._ Fell out of the item above, which
      ruled out its own stated lead and left this as the live question. The
      complex64 priorities at n = 256 are 135 / 130 / 120 — outliers from an
      earlier tuning round, an order of magnitude above the 10–40 used
      everywhere else — and selection differs by _precision_ at two sizes:
      n = 256 binds `dit256_radix2_avx2` for complex64 but `dit256_radix16_avx2`
      for complex128, n = 8192 binds `dit8192_radix4_then2_params_avx2` against
      plain `dit8192_radix4_then2_avx2`. Neither difference is wrong on its face;
      neither has been re-measured since the priorities were set.

      `BenchmarkCodeletCandidates{64,128}` already times every registered
          candidate per size and direction, so this is a measurement job rather than
          a code one — but two things must be fixed before its output can be
          trusted:

          - the 94-cell sweep needs **per-group canary bracketing** (see the item
            above: a once-per-pass canary let 3 of 5 passes through contaminated,
            one of them by 50×, and the resulting candidate ordering was garbage);
          - the harness drives its input from
            `complex(float32(i%7)-3, float32(i%5)-2)`, a period-35 pattern whose
            spectrum is almost entirely zero, so it partly times cancellation and
            denormal behaviour that differs per strategy. Switch it to random input
            before trusting any per-candidate ordering.

          Run on an idle machine, and re-check `internal/registry`'s descending
          priority order against the result rather than the other way round.

          **Both preconditions fixed first.** `BenchmarkCodeletCandidates{64,128}`
          now fills from a seeded RNG, generated once as `float64` and narrowed for
          the complex64 arm so both precisions see numerically identical input and
          their ratio stays like-for-like; it also enumerates sizes through the
          sorted `GetAvailableSizes` rather than unsorted map keys. Any
          `BenchmarkCodeletCandidates` number recorded before this is no longer
          comparable. And the canary-gated protocol, which existed only as a
          scratchpad script the previous item cited, is now in-tree as
          `scripts/bench_gated.sh` + `scripts/bench_gated_analyze.sh`, with a
          `just bench-gated` recipe and a BENCHMARKS.md section — the numbers below
          are reproducible from the repo. One design change against the scratchpad
          version: a group is one **(precision, size) with all of its candidates
          back-to-back**, not one signature, so the whole ranking is taken inside a
          single verified-quiet window and drift cancels within the comparison.

          **Measured: 85 accepted groups against 11 rejected** (9 over gate, 2 on
          drift), 12–15 groups behind every cell, `-benchtime=0.5s`, 16 passes,
          i7-1255U / AVX2+FMA / no AVX-512. `GOOD` recalibrated to 1650 ns from the
          1810 the previous round used. Ratios are to the incumbent, taken within
          each group and then medianed. Incumbents were read off the runtime
          registry, not off the priority table.

          | n    | prec | incumbent (all confirmed)          | runner-up                     | fwd rel | inv rel |
          | ---- | ---- | ---------------------------------- | ----------------------------- | ------- | ------- |
          | 256  | c64  | `dit256_radix2_avx2`               | `dit256_radix16_avx2`         | 1.303   | 1.488   |
          | 256  | c128 | `dit256_radix16_avx2`              | `dit256_radix4_avx2`          | 1.144   | 1.445   |
          | 512  | c64  | `dit512_radix2_avx2`               | `dit512_radix8_avx2`          | 1.256   | 1.317   |
          | 512  | c128 | `dit512_radix8_avx2`               | `dit512_radix4_then2_avx2`    | 1.277   | 1.284   |
          | 8192 | c64  | `dit8192_radix4_then2_params_avx2` | `dit8192_radix4_then2_avx2`   | 0.993   | 1.009   |
          | 8192 | c128 | `dit8192_radix4_then2_avx2`        | `dit8192_radix4_then2_sse2`   | 0.990   | 1.000   |

          At 256 and 512 the incumbent wins both directions by 14–30 %, so those
          four are settled. The two at 8192 are dead heats, and the pre-registered
          rule was that anything inside ±3 % is a tie and the incumbent stays.

          **The 135 / 130 / 120 outliers encoded the right order.** Measured
          forward and inverse ranking at n = 256 complex64 is radix2 < radix16 <
          radix4 — exactly the order those priorities expressed. Only their
          magnitudes were wrong, so they are now 35 / 30 / 25, inside the band used
          everywhere else. Verified order-preserving: the bound codelet is
          unchanged at all 14 power-of-two sizes in both precisions, and the
          generated diff is three priority lines.

          **Selection differing by precision is correct at both sizes.** At n = 256
          the complex128 winner `dit256_radix16_avx2` is 1.30× faster than radix16
          is for complex64, where radix2 wins instead — the split is real, not a
          tuning accident. At n = 8192 the split is a coin-flip rather than a
          preference (below).

          **Two findings that the priority mechanism cannot express, recorded
          rather than acted on:**

          - At n = 8192 complex64 the `params` incumbent and the plain
            `dit8192_radix4_then2_avx2` are within ±1 % and **swap by direction**
            (plain is 0.7 % faster forward, 0.9 % slower inverse). The `params`
            entry is the only AVX2 complex64 codelet carrying a custom twiddle
            layout (`TwiddleSize` / `PrepareTwiddle`), so it costs extra plan
            memory and prepare-time work for no measurable transform gain at this
            size. A simplification candidate, not a performance one. One priority
            governs both directions — `CodeletEntry` holds `Forward` and `Inverse`
            in one struct — so a per-direction split like this is not expressible
            even in principle.
          - At n = 8192 complex128 the SSE2 codelet **matches** the AVX2 one
            (0.990 forward, 1.000 inverse): AVX2 buys nothing at this size in this
            precision. This echoes the standing c128-16384 suspicion recorded
            earlier. It is also not a priority question — `Register` sorts
            SIMD-level-major (`internal/registry/registry.go:65-72`), so an SSE
            entry can never outrank an AVX2 one; the only lever is a negative
            priority disabling the AVX2 entry, which a 1 % margin does not justify.

          Also noted: at n = 512 complex64 the second and third candidates
          (`dit512_radix8_avx2`, `dit512_radix4_then2_avx2`) swap rank between
          directions. Irrelevant to selection, since the incumbent wins both.

          **A methodological data point for the host itself.** During canary
          calibration one reading blew up to 24 µs against a ~1.7 µs floor — 13× —
          while package temperature *fell* from 92 °C to 61 °C. The cause was
          another process (`trufflehog`) at 111 % CPU. Contention and heat are
          independent failure modes, they move the canary in the same direction,
          and cooling does not address the first; a protocol that only waits for a
          temperature threshold would have accepted that window.

          Still unaudited: 4–128, 1024, 2048, 4096, 16384, 32768. Nothing in this
          round suggests a mis-tuned incumbent is likely there, but nothing rules
          it out either — the sweep is now a `just bench-gated <sizes>` away.

- [x] **algo-fft loses to gonum at n = 44100** — 4.00 ms against gonum's
      2.59 ms (FFTW3: 236 µs). That is the canonical audio sample rate, in
      the FFT library of a DSP suite, and it is the worst result in the
      sweep at 0.06× FFTW3. 44100 = 2²·3²·5²·7² is fully factorable by the
      mixed-radix engine, but `MixedRadixEligible` routes it to Bluestein
      because its power-of-two part is 4, below the `mixedRadix7And11Wins`
      threshold (`internal/planner/utils.go`). That gate was fitted on
      shapes up to ~14080; 44100 is 3× larger and the Bluestein pad it falls
      back to is ≥ 88199, so the extrapolation is well outside the data it
      was derived from. Measure both routes at 44100 before touching the
      gate — and note that 2205 (= 3²·5·7²) _does_ take the mixed-radix
      route and still only reaches 0.07× FFTW3, so neither path is currently
      good at these shapes and the gate may not be the whole answer.

      _Worse than the laptop sweep showed (2026-07-25)._ On the SSE-only
              host, where FFTW loses its AVX2 too and the comparison is therefore
              closer to like-for-like, 44100 measured **7.49 ms against FFTW's
              311 µs (24×) and gonum's 2.12 ms (3.5×)**. The same sweep put
              algo-fft at only 1.1–1.9× FFTW across the whole power-of-two ladder
              (8…8192) and 6.9–13.3× at the 5-smooth lengths. A deficit that
              survives removing SIMD from _both_ sides is algorithmic, not a
              missing-codelet problem — which is the strongest evidence yet that
              P5.1, not more assembly, is where the non-power-of-two work belongs.

          _Fixed 2026-07-26/27._ The gate was **not** the whole answer, as the item
          suspected — but the reason is better than expected: the pow2-part rule was
          fitted on a mixed-radix **driver defect**, not on the algorithm. Measuring
          both routes first (as the item instructed) is what exposed it.

          1. **The driver dispatched 4-point sub-transforms to codelets.** The AVX2
             mixed-radix drivers guarded their codelet-dispatch hook with `n > 1`,
             while the scheduler has always required `n > 5` before emitting a
             codelet-backed composite radix. Any schedule ending in radix 2/3/4/5
             therefore sent every leaf through a full codelet call — a strided
             twiddle gather, two `sync.Pool` round-trips and a `defer` — to do a
             handful of butterflies. At n = 4900 = `[5,5,7,7,4]` that is 1225
             dispatches per transform, and a CPU profile put **32% of complex64 time
             inside `ForwardAVX2Size4Radix4Complex64Asm`** alone. Both drivers now
             use the scheduler's own bound (`mixedRadixCodeletMinSize`), which keeps
             dispatch a superset of what the schedule can emit.

             The symptom that led there was a precision asymmetry: complex64 ran
             **1.6× slower than complex128 on the identical route** at 308, 1100,
             2156 and 4900, and only there. Those are exactly the schedules ending
             in radix 4 — complex64's size-4 codelet is an assembly call where
             complex128's is a Go function. (The tempting correlation, that all four
             lack a factor 3, was a coincidence of this size set.)

             Mixed-radix got 18–58% faster at every length measured, including ones
             that were **already** eligible and therefore never in scope for the
             gate: 693 −29%, 1155 −28/−31%, 1920 −37%, 21 −20/−31%, 33 −31/−26%.
             Sizes whose schedules contain no radix ≤ 5 (3584, 7168, 11264, 49, 77,
             96, 385) were unaffected and served as the noise control. `purego` is
             untouched — the hooks exist only in the AVX2 drivers.

          2. **The win gate collapses to one criterion.** With the pathology gone,
             re-measurement across the gate's whole excluded set (7, 14, 22, 28, 44,
             55, 63, 105, 121, 231, 308, 462, 847, 924, plus 1100 … 44100; both
             precisions; forward and inverse; 8–10 counts) showed the existing
             **odd-length rule generalizes to every parity**: mixed-radix wins where
             Bluestein's pad is ≥ ~2.5n and washes or loses below it. Predicted vs
             measured agrees at 19 of 21 shapes; the two misses are conservative
             (28 and 22, both tiny). So the `pow2 ∈ {2,4} → lose` branch is simply
             deleted — and because `(n+1)/2 == n/2` for even n, the surviving
             expression needed no change at all. The `pow2 ≥ 8 → win` branch stays:
             it encodes a tuned power-of-two codelet leaf that the pad ratio cannot
             see (448, 3584, 7168 and 14080 pad to only ~2.3n yet win 1.3–6×).

          Plan-level result at 44100 (public API, forward): complex128 **3.40 ms →
          1.89 ms (−44%)**, complex64 2.41 ms → 1.88 ms (−22%); inverse −56% and
          −6%. Against the figures that opened this item, algo-fft now sits **ahead
          of gonum's 2.59 ms** at 44100 rather than 1.5× behind it. Newly rerouted
          with it: 308, 1100, 2156, 4900, 6300, 8820, 22050 and 44 (+23…+102%
          forward).

          One documented cost: at 22050 complex64 the forward is a tie and the
          **inverse regresses ~11%**, while complex128 gains 24–36%. The gate sees
          neither precision nor direction, and carving out one size is exactly the
          overfitting that produced the rule being replaced, so 22050 stays
          eligible on the pad-ratio criterion.

          Still open, and unchanged by this: FFTW3 is at 236 µs. Closing 1.89 ms →
          236 µs is P5.1 (mixed-radix engine quality), not routing — the item's own
          2205 observation stands.

- [x] **`KernelRecursive` falls off a cliff above 2048, and allocates.**
      Found incidentally while benchmarking on an idle AVX2/AVX-512 host
      (2026-07-25), but it is not machine-specific and should reproduce
      anywhere. Plan-level complex64, `BenchmarkPlanForward_*_Recursive`:

      | n     | Recursive  | allocs/op | default path | ratio     |
              | ----- | ---------- | --------- | ------------ | --------- |
              | 1024  | 16.9 µs    | 6         | 8.97 µs      | 1.9×      |
              | 2048  | 35.1 µs    | 6         | 15.9 µs      | 2.2×      |
              | 4096  | **1.39 ms** | 11       | 38.6 µs      | **36×**   |
              | 8192  | **5.41 ms** | 531      | 87.3 µs      | **62×**   |
              | 16384 | **23.2 ms** | 547      | 198 µs       | **117×**  |

              Two separate problems in one signature. The 40× jump between 2048 and
              4096 says the recursion stops finding codelet leaves and falls back to
              something quadratic-ish; the allocation count going 6 → 531 at 8192
              says it also starts allocating _per call_, in a library whose stated
              contract is zero allocations after plan creation. The inverse
              direction is identical, so it is in the shared decomposition, not the
              direction-specific glue. `KernelRecursive` is opt-in via
              `PlanOptions.Strategy` and `KernelAuto` never selects it, which is why
              this has stayed invisible — but it is reachable from the public API,
              so it is a defect rather than a tuning gap. Start at the leaf-size cut
              in `internal/transform` and check what happens when the remaining
              factor stops matching a registered codelet.

              _Fixed 2026-07-25._ The investigation turned up **a silent wrong-answer
              bug underneath the performance one**, which was the more serious of the
              two. Three findings:

              1. **Leaf codelets were fed the wrong twiddle table.** A
                 `registry.CodeletEntry` may declare a SIMD-friendly twiddle layout
                 via `TwiddleSize`/`PrepareTwiddle`; the normal plan path materializes
                 those tables (`plan_alloc.go`), but `internal/transform/recursive.go`
                 always handed leaves the standard length-n DIT table.
                 `dit256_radix16_avx2` asks for 748 elements and got 256, so
                 **complex128 `KernelRecursive` at n = 1024 returned a wrong
                 spectrum** — max abs error 2.6e5 against the reference, and
                 `Inverse(Forward(x)) != x`. Leaf lookup now goes through
                 `leafCodelet`, which only binds codelets using the standard layout
                 and otherwise falls back to the generic DIT.

                 This escaped the test suite because every recursive correctness test
                 transformed an **impulse** (`input[0] = 1`), whose spectrum is
                 all-ones — an input that is blind to twiddle errors, because every
                 twiddle multiplies a zero. Parseval and linearity are likewise
                 insensitive. Plan-level coverage of `KernelRecursive` also stopped at
                 n = 64. `plan_recursive_test.go` now cross-checks against the default
                 plan with a real signal at 1024…16384 in both precisions, plus a
                 round-trip.

              2. **The decomposition chose radices with no butterfly.** The scorer's
                 +10000 "sub-size has a codelet" bonus outweighed its penalty on wide
                 radices, so 8192 split 16-way and 16384 split 32-way to reach a
                 512-point leaf in one level. Radix 16 and 32 have no butterfly: they
                 land in `combineGeneral`, a naive size-radix DFT costing O(radix²)
                 complex multiplies **per output element**, with `sin`/`cos`
                 recomputed inside the innermost loop. That is the entire cliff — cost
                 scaled exactly as radix²·subSize (1.39 ms → 5.41 ms → 23.2 ms is 4×
                 per step, matching 8² → 16² → 32²). `combineRadices` now restricts
                 splits to radix 4 and 2, the only two with a real butterfly, so the
                 tree goes deep instead of wide (16384 = 4×4096 → 4×1024 → 4×256).
                 Radix 8 is excluded too: `combineRadix8` is a direct 8-point DFT, not
                 a butterfly, and measured 34–44% slower than reaching the same size
                 through two radix-4 levels.

              3. **The allocations were slice plumbing, not data.** Every recursion
                 node built `[][]T` views and one `make([]T, subSize)` input buffer
                 per sub-FFT, and `combineGeneral` allocated a temporary per output
                 element — 512 of the 531 allocs at 8192. The scratch and twiddle
                 blocks were *already* flat in `[r][k]` order, so the views were pure
                 overhead: the combine functions now index the flat blocks directly,
                 one reused decimation buffer serves all sub-FFTs at a level, and the
                 DIT fallback takes a leaf bit-reversal table precomputed at plan time
                 (`LeafBitrev`) instead of rebuilding it per call.

              Result, same benchmark, ABBA-interleaved against HEAD (n≥4096 all
              p=0.000; 1024/2048 are unchanged-tree controls and read `~`):

              | n     | before      | after   | Δ          | allocs/op | vs default |
              | ----- | ----------- | ------- | ---------- | --------- | ---------- |
              | 1024  | 12.5 µs     | 9.4 µs  | ~          | 6 → **0** | 1.02×      |
              | 2048  | 23.7 µs     | 19.8 µs | ~          | 6 → **0** | 1.20×      |
              | 4096  | 506 µs      | 46.8 µs | **−90.8%** | 11 → **0** | 2.03×     |
              | 8192  | 2.23 ms     | 76.5 µs | **−96.6%** | 531 → **0** | 1.72×    |
              | 16384 | 8.87 ms     | 204 µs  | **−97.7%** | 547 → **0** | 2.69×    |

              Inverse tracks forward (−88.0%, −94.7%, −96.3%). The 36×/62×/117×
              penalty against the default path is now a flat 1.7–2.7×, which is the
              expected cost of 256/512-point codelet leaves versus a full-size tuned
              kernel — a tuning gap, not a cliff.

- [ ] **Let recursive leaves use prepared-twiddle codelets.** Follow-up to the
      fix above: `leafCodelet` currently declines any codelet declaring
      `TwiddleSize`/`PrepareTwiddle` and falls back to the generic DIT, which
      costs the best leaf on some size/precision pairs — on this laptop,
      complex128 at n = 256 (`dit256_radix16_avx2`). Binding them needs
      per-leaf forward _and_ inverse tables built at plan time, since
      `PrepareTwiddle` takes an `inverse` flag while the recursive executor
      currently shares one table across both directions. Worth measuring
      before building: the leaf is one of two levels, so the ceiling is
      modest, and it only matters for the sizes where a prepared-twiddle
      codelet wins.

- [ ] **Audit the other recursive tests for permutation/twiddle-blind inputs.**
      The impulse-input problem in `internal/transform/recursive_integration_test.go`
      hid a wrong-answer bug at every size ≥ 1024 for an entire precision.
      An impulse cannot detect a wrong twiddle (they all multiply zeros) and
      cannot detect a wrong output ordering (its spectrum is all-ones);
      Parseval and linearity are insensitive to both. Those tests are still
      worth keeping, but each needs a companion case driven by a broadband
      signal and compared bin-by-bin against `internal/reference`. The same
      question should be asked of the Bluestein, Rader and mixed-radix test
      vectors, since the reflex to test with an impulse is not specific to
      this file.

### P5.1 The mixed-radix engine is now the weak link

Rader, the newest algorithm work, reaches 0.65× FFTW3 (geomean) and 0.93×
at n = 12289 — near parity. The mixed-radix engine, extended with radix-7/11
this cycle but never retuned, sits at 0.20× for 5-smooth and 0.13× for
7/11-smooth lengths. That ordering is the reverse of the naive expectation
and says plainly where the next round of work goes.

- [ ] **Give the mixed-radix leaves a tuned path.** Six of the weakest
      lengths in the sweep resolve to `dit_fallback` — 96 (0.12× FFTW3),
      448 (0.10×), 480 (0.20×), 704 (0.12×), 768 (0.23×) and 1000 (0.11×) —
      the generic driver with no size-specific leaf. Their SIMD-over-purego
      ratios are correspondingly poor (1.39–3.32× against 4–6× for
      well-served power-of-two sizes), which is the same signal from the
      other side: there is little assembly on these paths to accelerate.
      The odd-first schedule (P4.1) already arranges a power-of-two codelet
      leaf where one is reachable; these are the shapes where it is not.
      Candidates: SIMD radix-3/5/7/11 butterflies, or per-size codelets for
      the most common composite lengths.

      _Partly addressed 2026-07-27 — and the premise above was wrong._ Every
          one of the six lengths **does** reach an assembly codelet leaf; the
          odd-first schedule places one at all of them:

          | n    | schedule    | n    | schedule      |
          | ---- | ----------- | ---- | ------------- |
          | 96   | `[3 32]`    | 704  | `[11 64]`     |
          | 448  | `[7 64]`    | 768  | `[3 256]`     |
          | 480  | `[5 3 32]`  | 1000 | `[5 5 5 8]`   |

          What was slow is the *dispatch wrapper* around the codelet, which the
          driver re-paid at every recursion node. A CPU profile at n = 1000
          (complex64) put only **1.9%** of runtime in the codelet assembly and
          ~40% in dispatch overhead: `cpu.DetectFeatures` took an RWMutex *and* an
          exclusive Mutex on every call (13%), `registry.Lookup` an RWMutex (12%),
          and each leaf gathered a twiddle table into a `sync.Pool` buffer (15% in
          pool traffic) that the codelet then discarded whenever it declared a
          prepared layout. Three fixes landed:

          - `internal/cpu`: features cached in an `atomic.Pointer`, so the steady
            state is two atomic loads and no lock.
          - `internal/registry`: the size map is copy-on-write behind an
            `atomic.Pointer` (writers are init-time only), making `Lookup`
            lock-free. This also makes the returned `*CodeletEntry` stable — the
            previous version handed out a pointer into a slice a later `Register`
            could sort or reallocate in place.
          - `internal/fft`: the leaf twiddle gather is gone. The recursion keeps
            `n*step == len(twiddle)`, so `twiddle[i*step] == W_n^i` — the gather
            always rebuilt the standard size-n table, which is now cached by size
            (`mixedradix_leaf_twiddle.go`). The prepared-layout check moved ahead
            of it so codelets that ignore the standard table build nothing.

          Measured on the i7-1255U (AVX2), interleaved arms, 6 rounds, forward,
          both precisions — geomean **−15.0%**: 96 −27.6/−23.5% (c64/c128),
          448 −6.1/−12.1%, 480 −11.8/−9.6%, 704 ~/−4.2%, 768 −10.6/−14.8%,
          1000 −22.8/−24.4%. All p ≤ 0.015 except 704 c64 (p = 0.18, same sign).

          One hypothesis was tested and **rejected**: raising
          `mixedRadixCodeletMinSize` from 5 to 8, so the 125 size-8 leaves of
          n = 1000 use the inline Go radix-8 butterfly instead of dispatching,
          costs **+25.6%** at n = 1000 and is neutral elsewhere. Even with the
          dispatch overhead the size-8 assembly codelet is the better leaf; the
          threshold stays at 5.

          What remains is the real leaf work the item name asks for. After the
          above, n = 1000 profiles as 34% `math.MulComplex64` + 15%
          `butterfly5ForwardComplex64` + 17% recursion driver — i.e. the scalar
          odd-radix stages, not dispatch. Subtasks, in order:

          - [x] **Vectorise the odd-radix butterfly stages.** _Done 2026-07-27, with a
            smaller payoff than the profile suggested._ The stride-`j*step` twiddle
            gather that blocked vectorisation is removable: the same invariant
            `n*step == len(twiddle)` gives `twiddle[j*k*step] == W_n^(j*k)`, so a
            stage's twiddles are a permutation of the size-n table and can be
            materialised in the data's own layout (entry `j*span+k`), keyed by
            stage shape rather than by plan — which is what lets the recursion
            reach them without plan context. The stage then becomes one in-place
            `ComplexMulArrayInPlace` over `input[span:n]` plus a twiddle-free
            butterfly loop with the radix switch hoisted out of it
            (`internal/fft/mixedradix_stage_twiddle.go`).

            Two gates had to be measured in, both as interleaved sweeps against
            the same binary with the path disabled:

            - **`n - span >= 64`.** Ungated, deep schedules over small factors
              collapse — n = 2205 = `[5 7 7 3 3]` ends in 245 span-3 and 735
              span-1 stages and ran **+80%** (complex64). The scalar stage is not
              slow in absolute terms: its strided twiddle operand stays inside an
              L1-resident table, so the vectorised form must win on issue width
              alone and needs enough elements to do it.
            - **Radix 7 excluded.** It lost at every threshold from 16 to 256:
              n = 448 = `[7 64]` has one radix-7 stage with 384 multiplies and ran
              +6…+8% slower vectorised. Radix 11 is the opposite (n = 704 =
              `[11 64]`, −7%) and is kept.

            Net over the mixed-radix benchmark set (i7-1255U, AVX2, interleaved
            arms, 10 rounds, both precisions): geomean **−4.8%**, no size
            significantly slower. 480 −12.4%, 704 −10.1%, 768 −9.2%/−8.8%
            (c64/c128), 3600 c128 −9.4%, 12000 −7.8%/−5.6%, 1000 c64 −4.0%;
            96, 448 and 2205 neutral. (The gate sweep, which toggles only the new
            path inside one binary, put the path itself at −3.2%; the rest comes
            from splitting the scalar radix-7/8/11 stages out of the driver.)
            Even so this is far short of the ~34% the profile attributed to
            `math.MulComplex64`, which says the scalar twiddle multiply was mostly
            overlapping with the butterfly rather than serialising behind it.
            Beating it properly needs a radix-r stage kernel in assembly that
            keeps the r streams in registers across the multiply and the
            butterfly, not two passes over memory.
          - [x] **Hoist the leaf codelet resolution out of the recursion.**
            _Done 2026-07-27._ The dispatch fires exactly when the node's
            remaining schedule is a single composite radix: the scheduler checks
            the registry for the remaining size at every step and returns as soon
            as it emits a composite radix, so a codelet can only ever match at a
            leaf, and every leaf of one transform has the same size. The entry is
            therefore resolved once per transform from `radices[stageCount-1]`
            (`leafCodelet64/128`) and threaded through the four recursion hooks as
            a trailing parameter; the AVX2 drivers dispatch what they are handed,
            gated on `len(radices) == 1`, instead of running
            `cpu.DetectFeatures()` + `registry.Lookup` + a priority scan per node.
            At n = 1000 = [5 5 5 8] that is 156 lookups per transform down to 1.

            Measured (i7-1255U, AVX2, interleaved arms, 14 rounds, clean builds
            both sides, vs 8298983): **geomean -1.9%**, and the win tracks the
            leaf count exactly as the mechanism predicts -- n = 1000 -9.4%/-6.6%
            (125 leaves), n = 3600 -7.6%/-4.4% (225 leaves), 2205 -4.2%/-3.0%,
            448 -4.4% (c64), 480 -4.1% (c64), 704 -3.0% (c64).

            Two caveats, both honest:

            - **n = 768 regresses +6.8%/+4.4% and is unexplained.** 768 = [3 256]
              has only 3 leaves, so there is no win available there, but the loss
              is real: it reproduced across three independent builds. It is not
              allocations (0 B/op both sides) and it is not the added parameter --
              a variant carrying the signature change but keeping the per-node
              lookup measures neutral at 768. That leaves the guard change or
              code layout, and the two could not be separated on this machine.
            - The measurement machine could not be quiesced (85-100 C, competing
              load), so these are ±1-4% numbers from interleaved arms with
              alternating order, not quiet-machine numbers.

            A same-binary knob sweep during this work also put the *previous*
            subtask's vectorised stage at only about -0.9% once this hoist is in
            place, well under the -4.8% recorded for it above. That figure was
            taken against a different baseline and is not directly comparable, but
            the vectorised stage's value is worth re-deriving before more is built
            on it.

          - [x] **Re-derive the vectorised odd-radix stage's value on top of the
                hoist.** _Done 2026-07-27 — the −0.9% estimate was right and the
                −4.8% is gone. The two-pass form no longer earns anything on
                amd64; the fused kernels are the whole win._

                Enumerating the stage shapes first narrowed the question sharply:
                across all ten mixed-radix bench lengths, **exactly one stage
                still takes the two-pass path** — n = 704's radix-11 span-64
                stage. Every other vectorised stage is fused, everything else is
                scalar. So the −4.8% had already been absorbed by the fused
                kernels before this was measured.

                Three arms from one binary — `full` (fused where available),
                `nofused` (same stage set, every stage forced two-pass) and `off`
                (no vectorised stage at all) — over ten lengths × both
                precisions, 8 canary-gated interleaved rounds with rotating arm
                order. Round 7 was rejected (post-canary 9669 ns against an
                884 ns floor, a 10× disturbance); the other seven bracketed
                inside the 1.20× gate.

                | comparison           | geomean | reading                             |
                | -------------------- | ------- | ----------------------------------- |
                | `off` vs `full`      | +50.9%  | the vectorised stage is worth −34%  |
                | `nofused` vs `full`  | +49.7%  | the fused kernels carry ~all of it  |
                | `off` vs `nofused`   | +1.5%   | two-pass alone is worth −1.5%       |

                **n = 704 is a built-in null control**: it has no fused stage, so
                its `full` and `nofused` arms are the same code. That cell reads
                −2.2%/+0.2%, which pins the run's noise floor at ±2% — and the
                two-pass path's −1.5% does not clear it. Per-cell it straddles
                zero: it loses 7.5% at 448/c128 and 6.6% at 96/c128, wins 9.9% at
                768/c64.

                A second gated sweep (8 rounds, all accepted) forced the binary to
                the SSE2 tier via `cpu.SetForcedFeatures`, where no fused kernel
                exists and the two-pass form is the only vectorised path. There it
                is a net **loss**: `off` vs `nofused` = **−1.3%** geomean, i.e.
                the two-pass stage costs time, best case +1.8% (704/c128), worst
                −5.3% (12000/c64).

                So the gates the item worried about are fine, but for a different
                reason than they were fitted for: `n - span >= 64` was already
                re-derived for the fused path (see above, 64 is a local optimum
                there too), and "radix 7 excluded from two-pass" turns out to be
                the general rule rather than a radix-7 quirk. Nothing was changed
                — the two-pass code is retained because it is the only vectorised
                stage on NEON/WASM/purego, tiers this machine cannot measure, and
                deleting it on an amd64 wash would regress them untested. The
                stale −3.2%/−7% figures in `mixedRadixStageMinMuls` are now marked
                as pre-hoist.

          - [x] **Fused radix-11 stage kernel.** _Done 2026-07-27 — the largest
                single win the fused kernels have produced. n = 704 dropped
                10244 → 2609 ns (complex64, −74.5%) and 11597 → 4191 ns
                (complex128, −63.9%)._

                Fallout from the re-derivation above. n = 704 = `[11 64]` is one
                of the six weakest lengths in the sweep (0.12× FFTW3) and its L0
                is a radix-11 span-64 stage with 640 twiddle multiplies — the
                single largest stage in the set with no fused kernel. Admitting
                it to the two-pass path bought +0.2%/−0.3%, i.e. nothing, so 704
                was running that stage at scalar-equivalent speed.

                `avx2_f32_mixedradix_stage11.s` and its complex128 twin follow
                the radix-7 kernels, but radix 11 no longer fits the register
                file, and both departures follow from that:

                - `t1..t5` and `u1..u5` are live across the whole output half,
                  which spends ten YMM registers before any constant. So the ten
                  butterfly constants sit pre-broadcast in RODATA and are read as
                  FMA memory operands — 320 bytes, L1-hot after the first block,
                  and an FMA with a memory source stays one fused uop.
                - Ten row offsets would need ten index registers; the SIB scale
                  supplies the even multiples, so holding 1×, 3×, 5×, 7× and 9×
                  the row stride reaches all ten rows. That also shortens the
                  prologue, which is where the break-even sits at these spans.

                Verified before any assembly was written: the conjugate-pair
                decomposition was checked against `kernels.Butterfly11*` (the
                11×11 matrix form) in both directions, then the kernels against
                the definition-based stage reference at spans 4, 7, 16, 19, 64,
                65 and 253 — so both the vector body and the Go tail — in both
                precisions, both directions, and with `dst == input` aliasing.

                Measured with the same single-binary env-knob protocol, 8
                canary-gated rounds, all 8 accepted (canary 975–1144 against a
                994.7 floor). The eighteen cells with no radix-11 stage spanned
                −2.2%…+5.4%, which is this run's noise floor; the n = 704 effect
                is 30–60× that.

                Two things did _not_ go the way this item predicted:

                - **No gate change.** The item expected
                  `mixedRadixStageVectorizable` to start treating radix 11 like
                  radix 7. It should not. Radix 7's two-pass form was a measured
                  regression, which is why it is admitted only where the fused
                  kernel runs; radix 11's is a wash on amd64 and is the only
                  vectorised stage on NEON/WASM/purego. Gating it on the fused
                  kernel would drop those tiers to scalar to buy nothing here.
                  The size threshold already admitted radix 11, so the fused
                  kernel simply takes over where AVX2 is present.
                - **The fusion is not the whole win.** About 1.6× of the 3.9× is
                  the conjugate-pair butterfly, not the fusion:
                  `kernels.Butterfly11*` is the full 11×11 matrix — 100 complex
                  multiplies against the pair form's 50 real-by-complex ones —
                  and the same swap written in plain Go benchmarks 113.6 → 72.1
                  ns. See the next item.

          - [ ] **Give `Butterfly11` the conjugate-pair form.** Raised by the
                subtask above. `kernels.Butterfly11ForwardComplex64` and its
                three siblings evaluate the full 11×11 DFT matrix, which is the
                only radix in the set still doing O(r²) complex multiplies —
                radix 3/5/7 all have hand-written butterflies. The fused AVX2
                kernel now sidesteps it, so this is dead weight on amd64, but it
                is still what runs every radix-11 stage on SSE2, NEON, WASM and
                purego, and what the fused kernels' own Go tails call. A
                throwaway pair-form implementation measured 113.6 → 72.1 ns
                against it (−37%) without being tuned; the derivation and the
                index tables are written out in the header of
                `avx2_f32_mixedradix_stage11.s`. Cheap, arch-independent, and the
                registry-driven reference tests already cover it.

          - [ ] **Explain the +6.8%/+4.4% regression at n = 768.** Left open by
                the hoist subtask. 768 = `[3 256]` has 3 leaves so no win was
                available, but the loss reproduced across three independent
                builds. Ruled out: allocations (0 B/op both sides) and the added
                hook parameter (a variant carrying the signature change but
                keeping the per-node lookup is neutral there). Remaining
                candidates are the `len(radices) == 1` guard and code layout, and
                they could not be separated because each variant is a different
                binary. A `perf stat` comparison (branch misses, I-cache) on a
                quiet machine would settle it without needing a third build.

          - [x] **Write a radix-r stage kernel in assembly.** _Done 2026-07-27._
                Fused AVX2 stage kernels for radix 3 and 5, both precisions
                (`internal/asm/amd64/avx2_f{32,64}_mixedradix_stage{3,5}.s`,
                dispatched from `internal/fft/mixedradix_stage_asm_amd64.go`).
                One pass: rows 1..r-1 are multiplied by the stage table and stay
                in registers through the butterfly, so they are never written
                back. The k index is the vector axis — every YMM lane is a
                different k running the same butterfly — so the butterfly needs
                no cross-lane movement at all; the only shuffles are inside the
                complex multiply and the two multiply-by-i steps. Direction costs
                one register: forward's `-i` and inverse's `+i` are the same pair
                swap with a different XOR mask.

                Canary-gated A/B, 8 interleaved rounds from one binary, all 160
                paired cells negative (per-cell CV 1–3%): **geomean −30%**
                (c64 −32%, c128 −28%). Per length: 12000 −47…−58%, 3600
                −36…−45%, 1000 −24…−28%, 44100 −15…−17%, 2205 −8…−10%.

                The spread tracks coverage exactly, which is the check that the
                number means what it says: 12000 = [5 5 5 3 32] gets 156 fused
                calls per transform, 44100 = [5 5 7 7 4 3 3] gets 6 (its radix-7
                levels are excluded from the vectorised path and its radix-3
                levels fall under the `n - span >= 64` gate), and 2205 gets 1.

                The win is larger than the saved memory pass alone would give,
                because the two-pass form only ever vectorised its *multiply* —
                `ComplexMulArrayInPlace` took the SIMD path while the butterfly
                loop that followed stayed fully scalar, one complex per
                iteration. The fused kernel vectorises the butterfly too. That
                also resolves the puzzle recorded above: the profile's ~34% in
                `math.MulComplex64` never converted into a matching win because
                the multiply was not the serialising part.

                Remaining upside is in the follow-ups below; 44100 is now
                1.27 ms against FFTW3's 236 µs (and 781 µs once the radix-7
                kernel below landed).

          - [x] **Extend the fused stage kernel to radix 7.** _Done 2026-07-27._
                `avx2_f32_mixedradix_stage7.s` / `avx2_f64_mixedradix_stage7.s`,
                with the same fused shape as radix 3/5: three conjugate output
                pairs, cosine rows the `c[j*m mod 7]` index map and sine rows
                the same map with `s[7-k] = -s[k]` folded in. Derived and
                checked against the direct DFT (max error 3.6e-15, both
                directions) before any assembly was written.

                Two things differ from the radix-5 kernel. Six constants plus
                the sign mask leave nine YMMs for nine live values plus scratch,
                one short of also holding `a0`, so `a0` stays in memory and is
                re-read four times and the complex multiply uses its own
                destination as the third scratch register. That in turn forces
                the store order: `dst` row 0 is written last, after the final
                read of input row 0, or the documented `dst == input` aliasing
                would corrupt it.

                The gate is now conditional, as this item anticipated:
                `mixedRadixStageFused(span, radix)` (build-tagged, false in the
                stub) is what `mixedRadixStageVectorizable` consults for radix
                7, so the +6…+8% two-pass regression cannot reach a non-AVX2 or
                purego build. `case 7` was added to the two-pass switch in both
                precisions so the fallback executes the stage rather than
                hitting the `default:` panic.

                Coverage is what the win is made of. Fused calls per forward
                transform went 6 → 206 at 44100 (`5:6, 7:200`) and 1 → 6 at
                2205 — the two lengths that were the weakest cells of the
                radix-3/5 round for exactly this reason.

                Measured over 7 canary-gated interleaved rounds from one binary
                (round 6 rejected: its post-round canary came back at 1.59x the
                floor). Median-of-rounds, with min-of-rounds agreeing to within
                a point:

                | length | c64 fwd | c64 inv | c128 fwd | c128 inv |
                | ------ | ------- | ------- | -------- | -------- |
                | 44100  | -41.7%  | -38.3%  | -38.3%   | -38.3%   |
                | 2205   | -21.9%  | -21.8%  | -20.4%   | -20.4%   |

                Geomean over the radix-7 lengths -30.7%; every one of the 28
                paired cells improved in all 7 rounds, on-arm CV 1-3%. The
                controls (1000, 3600, 12000 — no radix-7 stage at any level)
                came back at -0.3% geomean with a +-3% spread and no consistent
                sign, which is the check that the number is the kernel and not
                the machine.

                44100 is now 781 µs, from 1.34 ms in this sweep's baseline and
                1.89 ms before the fused path existed. FFTW3's 236 µs is still
                3.3x away, but the gap has closed from 8.0x.

          - [x] **Re-derive `mixedRadixStageMinMuls` for the fused path.**
                _Done 2026-07-27 — the hypothesis is refuted and 64 stays._ The
                item expected a kernel that pays none of the two-pass form's
                fixed costs to break even sooner. It does not: 64 is a local
                optimum for the fused path as well, and one threshold serves
                both. Single-binary env-knob sweep, 6 canary-gated interleaved
                rounds (floor 3275 ns, every pre/post bracket within 1.15×, no
                round rejected), five lengths × both precisions × both
                directions, geomean against 64 — **32: +8.9%, 48: +1.5%,
                128: +2.7%**. Nothing beat it in either direction.

                The attribution is what makes that a result rather than noise:
                each arm regressed exactly the lengths whose stage set it
                changes and left every other length inside ±2.5%.

                | arm | stages it changes | what moved |
                | --- | ----------------- | ---------- |
                | 48  | +2205 `r=7 span=9` | 2205 only, +2.9…+7.4% |
                | 32  | + 1000 `r=5 span=8`, 3600 `r=3 span=16` | 1000 +20…+38%, 3600 +7.5…+14% |
                | 128 | −3600 (96 muls), −12000 (64 muls) | those two only, +6.4…+11.7% |

                Two corrections to the item's own premise, from enumerating the
                schedules rather than assuming them. **44100's radix-3 levels
                cannot be admitted at any threshold** — they are span 3 and
                span 1, below the fused kernel's span ≥ 4 floor, so
                `mixedRadixStageFused` rejects them before the multiply count is
                consulted; the only 44100 stage the threshold excludes is level
                4, radix 4, which has no fused kernel at all. And 1000's third
                radix-5 level, the item's other example, is precisely the stage
                whose admission costs +37% forward.

                The mechanism: the fused kernel did not shed the fixed cost so
                much as replace it. Its prologue broadcasts up to six constants
                and derives up to six row offsets before the vector loop starts,
                and at span 8 that loop runs twice — so the break-even sits
                where the prologue amortises, not where a second pass over
                memory would have. Recorded in the constant's doc comment so the
                next reader does not re-run it.

                Behaviour is unchanged; `mixedRadixStageVectorizable` was
                restructured to hoist the shared size test out of the radix
                switch, which is the shape the split threshold would have needed
                and reads better without it.

- [x] **Add practical DSP lengths to the internal benchmark set.** The
      lengths where algo-fft's lead over gonum nearly vanishes are exactly
      the ones a DSP user picks: 44100 (loses), 2205 (1.49×), 1000 (1.54×),
      3600 (1.51×), 12000 (1.68×) — against ~8× at powers of two. None of
      them appear in `plan_bench_test.go`, which is why the internal numbers
      looked healthy throughout. Add them so a regression here is visible
      without an external harness.

      _Done 2026-07-27._ All five lengths are in `plan_bench_test.go`, forward
          and inverse, both precisions — 20 benchmarks. The complex128 side reuses
          the `*Complex128Focus` helpers, so each logs the plan it resolved to,
          which is what makes a change of *route* visible and not just a change of
          speed.

          Adding them surfaced a reporting defect worth fixing separately: **the
          logged strategy/algorithm is wrong at all five lengths.** 1000 reports
          `dit_fallback` and 2205/3600/12000/44100 report
          `strategy=Stockham algorithm=stockham`, but none of them execute either.
          In `autoKernelComplex64`/`128` (`internal/fft/kernels_fallback.go`) the
          `!IsPowerOf2 && IsMixedRadixSmooth` branch runs *before* the strategy
          switch and takes mixed-radix unconditionally; the label is whatever
          `ResolveKernelStrategy` would have picked for a power of two of that
          size, and is never consulted. Verified directly: all five are
          `IsMixedRadixSmooth` and non-power-of-two. So `plan.algorithm` cannot be
          used to identify the route of a non-power-of-two length — which matters
          for the gate re-derivation below, and for anyone bisecting a routing
          regression with exactly the log these benchmarks emit.

          - [ ] **Report the route actually taken for non-power-of-two lengths.**
                The defect above. In `autoKernelComplex64`/`128`
                (`internal/fft/kernels_fallback.go`) the mixed-radix branch runs
                ahead of the strategy switch and takes it unconditionally, so
                `plan.algorithm` reports a power-of-two strategy that never
                executes. Either label the mixed-radix route honestly or have the
                plan record the branch it took; the current string is worse than
                no string, because it reads as authoritative. Blocks trusting the
                `*Complex128` benchmark logs, and the gate re-derivation below
                needs to know which route each length took.

          Steady-state allocations were checked while adding them and the
          zero-allocation contract holds at all five: 0 allocs/op in both
          directions and both precisions. (At `-benchtime=1x` they report 8–33
          allocs/op, which is plan warm-up amortised over a single iteration, not a
          per-transform allocation — worth knowing before anyone reads a 1x run as
          a regression.)

- [ ] **Re-derive the radix-7/11 win gates over a wider range.**
      `mixedRadix7And11Wins` and `rader7Or11Wins` were both fitted on the
      shapes measured at the time; the 44100 result shows at least one
      extrapolation failing outside that range. Re-run
      `BenchmarkMixedRadix7And11VsBluestein` with the practical lengths
      included and check whether the "power-of-two part ≥ 8" rule holds at
      large n or needs an n-dependent term.

### P5.2 Power-of-two soft spots

The curve against FFTW3 is not smooth. Two dips are large enough to be
structural rather than noise, and both reproduce in each direction:

- [x] **The n = 64 cliff.** 0.97× FFTW3 at n = 32 drops to 0.36× at n = 64
      (0.33× inverse), and no larger size recovers the n = 32 level.
      `dit64_radix4_avx2` also has the _lowest_ SIMD-over-purego ratio in
      the entire power-of-two ladder — 2.19× against 4–6× for its
      neighbours — so the codelet is barely beating pure Go. Two suspects to
      separate: the codelet itself, and the decomposition it uses. n = 32
      runs `dit32_radix4_then2_avx2` and is fine, so a radix-4-then-2
      variant at 64 is the obvious first experiment.

      _Done 2026-07-27._ It was the codelet, not the decomposition — and no new
          codelet was needed. Two defects, both found by reading the asm rather
          than by benchmarking variants:

          1. **`dit64_radix4_avx2` is not vectorised.** Every load and store in
             all three stages is `VMOVSD` — one complex64 per 128-bit register.
             It is AVX2 by name only, which is exactly the 2.19× ratio the item
             reports. Its complex128 twins are the same story: 4 `Y` register
             references each, all in the scale loop.
          2. **`dit64_radix2_avx2` already was a full-width 256-bit kernel** and
             its forward ran 2.2× faster than the incumbent, but it sat at
             priority 19 behind radix-4's 25 — because its *inverse* measured
             179 ns against the forward's 63 ns and dragged the pair down.

          The inverse asymmetry is the interesting part, because the two
          functions are byte-identical bar 48 `VFMADDSUB231PS`↔`VFMSUBADD231PS`
          sites. It was **one legacy-SSE instruction**: `MOVD AX, X8` in the
          `1/n` scale prologue, which Go assembles as the non-VEX `movq
          %rax,%xmm8`. Executing it with the upper YMM state dirty triggers the
          AVX↔SSE transition penalty, and that single instruction cost ~100 ns —
          three times the rest of the kernel. `VBROADCASTSS ·sixtyFourth32(SB),
          Y8` (broadcast straight from memory, no GPR round trip) removes it.

          Ruled out along the way, each with a measurement rather than an
          argument: the instruction pair itself (swapped `VFMSUBADD` for
          `VFMADDSUB` — no change), the input data (fed the inverse the
          forward's own output — no change), the memory layout (carved src/dst/
          scratch from one slab at fixed offsets — no change), and code
          placement (padded the inverse's start address in 16-byte steps, and
          benchmarked a byte-identical *copy* of the fast forward placed after
          the slow inverse — the copy was fast). The decisive run put all four
          arms in one process: forward 55 ns, inverse-with-`VBROADCASTSS` 55 ns,
          inverse-with-`MOVD` 157 ns.

          Worth knowing for anyone repeating this: the gap does **not** appear on
          the E-cores of this hybrid part (both directions ~150 ns on CPU 4),
          because the penalty is a P-core frontend/state effect. An unpinned
          benchmark can therefore land on either side of it. Pin with `taskset`.

          Landed:

          - `avx2_f32_size64_radix2.s` — folded the `1/64` scale into the
            stage-6 stores (stage 6 writes all 64 outputs exactly once, so the
            separate pass only re-read what it had just stored) and replaced the
            `MOVL`/`MOVD`/`VBROADCASTSS` prologue with a memory broadcast.
          - `avx2_f32_size64_radix4.s` — same prologue fix.
          - `avx2_f64_size64_radix2.s`, `avx2_f64_size64_radix4.s` — 374 legacy
            `MOVUPD` → `VMOVUPD` (1:1, same operand order).
          - Priorities: complex64 `dit64_radix2_avx2` 19 → **26** (now the
            winner); complex128 `dit64_radix2_avx2` 20 → 14 and
            `dit64_radix4_avx2` 25 → 15, with `dit64_radix4_sse2` 18 → 19, so
            the real SSE2 codelet wins over the SSE-width "AVX2" ones. AVX-512
            hosts are unaffected (their size-64 entry stays at 25).

          Measured on the i7-1255U, pinned to CPU 0/1, `-count=6`
          (`BenchmarkCodeletCandidates`):

          | codelet | before fwd/inv | after fwd/inv |
          | --- | --- | --- |
          | c64 `dit64_radix2_avx2` | 63.2 / 179.5 ns | **54.6 / 56.5 ns** |
          | c64 `dit64_radix4_avx2` (was incumbent) | 137.0 / 143.0 ns | 124.6 / 133.0 ns |
          | c128 winner at 64 | `dit64_radix4_avx2` 226 / 235 ns | `dit64_radix4_sse2` **174 / 193 ns** |

          End to end, `BenchmarkPlanForward_64` / `_64` inverse over three
          interleaved before/after rounds: **348–408 → 147–170 ns** forward and
          **369–401 → 153–166 ns** inverse, i.e. ~2.3–2.4× in both directions.

          Two follow-ups this opened, both filed below: the legacy-SSE sweep
          across the rest of the AVX2 tree, and the ~100 ns of plan-level
          overhead that now dominates a 55 ns size-64 codelet.

- [x] **Sweep legacy-SSE encodings out of the AVX2/AVX-512 asm tree.**
      _Done 2026-07-28._ **4089 legacy instructions converted to VEX across 59
      files**; 10 functions deliberately left untouched (see the all-or-nothing
      rule below). The sweep is **end-to-end performance-neutral** — the
      hypothesis that it would fix the 1024–16384 soft spots was **wrong** —
      but it removes a latent hazard class and it uncovered six badly
      mis-measured codelets.

      What the sweep is actually worth:

      | complex64 codelet         |  before fwd/inv |  after fwd/inv |
      | ------------------------- | --------------: | -------------: |
      | `dit4_radix4_avx2`        | 102.9 / 204.9   |    5.0 / 4.6   |
      | `dit8_radix2_avx2`        |   6.0 / 109.2   |    6.0 / 6.0   |
      | `dit8_radix8_avx2`        |   7.0 / 111.6   |    7.3 / 7.6   |
      | `dit16_radix2_avx2`       |  10.8 / 115.8   |   10.3 / 11.2  |
      | `dit32_radix2_avx2`       |  23.7 / 130.1   |   24.6 / 24.3  |

      Note the signature: before the fix every one of those sat at 102–205 ns
      **regardless of transform size**. That is a _fixed_ per-call cost, not a
      size-dependent one — the same ~100 ns `MOVL`/`MOVD`/`VBROADCASTSS` 1/n
      prologue that caused the n = 64 cliff. Everything else measured neutral:
      over 116 AVX2 codelet benchmarks the before/after ratio had median 1.00,
      p25 0.97, p75 1.03, **max 1.14, and zero regressions above 1.3×**.
      Plan-level benchmarks at 8–128 moved 0.95–1.14× (noise). The bulk of the
      edits (3648 `MOVUPD`, 89% of the total) are genuinely free: the
      MOVUPD-heavy c128 kernels measured 2569.5 → 2572.2 ns at n = 512.

      **The rule that matters: convert a function completely or not at all.**
      A partial conversion is catastrophic. Converting only the arithmetic in
      `ForwardAVX2Size1024Radix32x32Complex64Asm` took it from **7.1 µs to
      1.08 ms (152×)**; converting everything the liveness pass allowed still
      left it at **97 µs (14×)**, reproducible to 1% and confirmed through two
      independent measurement paths. Interleaving VEX and legacy encodings in a
      hot loop is far worse than leaving the loop uniformly legacy. The sweep
      therefore skips any function it cannot convert entirely.

      Method (the tooling is worth rebuilding if this is ever revisited):

      - A **liveness pass** decides safety. A VEX write zeroes bits [255:128]
        of its destination where the legacy form preserves them, so conversion
        is legal exactly when the upper half of that register number is dead.
        A backward fixpoint over a per-`TEXT` CFG found 68 instructions (in 10
        functions) where it is live — e.g. `Size1024Radix32x32` builds `Y8` via
        `VINSERTF128 $0, X8, Y8, Y8`, so its `MOVSS …, X8` really is a merge.
        Guard the analysis with an unresolved-jump-target check: two labels can
        share one instruction (an empty label body falling through), and
        silently dropping that edge makes the verdict unsound.
      - **Verify at the machine-code level, not the source level.** Disassemble
        before and after with binutils `objdump` (not `go tool objdump`, which
        misdecodes AVX), normalize away the `v` prefix, the VEX merge operand,
        and addresses that shift because VEX encodings are longer, then diff per
        symbol. All **457 asm symbols decoded to an identical instruction
        stream**. This caught two real bugs a passing test suite did not:
        Go's `MOVD AX, X12` assembles as a **64-bit** `movq %rax,%xmm12` (Go's
        `AX` is RAX), so its faithful VEX form is `VMOVQ`, not the 32-bit
        `VMOVD`; and inter-function `int3` padding is attributed to the
        preceding symbol and must be filtered before comparing.
      - Conversion classes: `MOVUPD`/`MOVUPS`/`MOVAPS`/`MOVDQU`/`MOVQ`/
        `MOVDDUP`/`MOVS[HL]DUP` are 1:1; `MOVSS`/`MOVSD` are 1:1 for mem↔reg
        but need the 3-operand merge form for reg,reg; `MOVHPS`/`MOVLPS` need it
        for the load form only; all two-operand arithmetic becomes
        `VOP src, dst, dst`; `SHUFPS $imm` becomes 4-operand; `CVTSQ2SD` becomes
        `VCVTSI2SDQ src, dst, dst`.
      - Scope the rewrite to functions that **already contain VEX**. They
        would fault on a non-AVX CPU, so adding VEX cannot change which
        hardware can run them — which is the airtight argument for not touching
        the real `sse2_*.s`/`sse3_*.s` files.

- [x] **Retune complex64 priorities at sizes 8/16/32 after the sweep.** The six
      codelets were mis-ranked _because_ of the penalty they carried, the same
      way `dit64_radix2_avx2` was at n = 64. Re-measured on an idle, cooled
      machine (load 0.14, package 62 °C, pinned to a P-core, median of 41
      interleaved rounds with the order rotated each round; p25/p75 within ~3%
      of the median on every arm), which confirmed the contended numbers:

      | n   | old winner                     | new winner (fwd/inv ns) | gain    |
      | --- | ------------------------------ | ----------------------- | ------- |
      | 8   | `dit8_radix4_avx2` (7.34/8.41)  | `dit8_radix2_avx2` 5.51/5.67  | 25/33 % |
      | 16  | `dit16_radix16_avx2` (13.16/13.82) | `dit16_radix2_avx2` 10.12/10.57 | 23/24 % |
      | 32  | `dit32_radix32_avx2` (30.06/33.37) | `dit32_radix2_avx2` 24.46/23.56 | 19/29 % |

      Priorities raised (siblings untouched): `dit8_radix2_avx2` 7 → 12,
      `dit8_radix8_avx2` 9 → 11 (it also beat `dit8_radix4_avx2`, 6.74/7.17 vs
      7.34/8.41, and had been ranked below it), `dit16_radix2_avx2` 20 → 55,
      `dit32_radix2_avx2` 20 → 30. complex128 was confirmed unaffected: its
      AVX2 winners at 8/16/32 (`radix4`, `radix4`, `radix4_then2`) still measure
      best or tied, and they served as the run's canary arms.

      The "raise `dit16_radix16_avx512` in step" caveat turned out to be
      unnecessary: `CodeletRegistry.Register` sorts by **SIMD level first** and
      priority only within a level, so an AVX-512 entry always outranks any
      AVX2 entry on an AVX-512 host regardless of priority. AVX2 priorities can
      be tuned in isolation.

- [x] **A few complex128 SSE2 codelets beat their AVX2 siblings but can never
      be selected.** Fallout of the same SIMD-level-dominates-priority rule
      above. Re-measured on an idle, cooled machine (same protocol as the
      retune: single process, pinned to a P-core, median of 41 interleaved
      rounds with the order rotated), which **refuted the two cases originally
      cited** — they were contended-run artifacts:

      | c128 | selected AVX2 (fwd/inv ns)          | best SSE2 (fwd/inv ns)         | verdict          |
      | ---- | ----------------------------------- | ------------------------------ | ---------------- |
      | 16   | `dit16_radix4_avx2` 22.41/28.79     | `dit16_radix2_sse2` 22.65/28.36 | tie (±1 %)       |
      | 32   | `dit32_radix4_then2_avx2` 65.7/69.4 | `dit32_radix2_sse2` 70.4/80.2   | AVX2 wins        |
      | 64   | `dit64_radix4_avx2` 198.0/217.6     | `dit64_radix4_sse2` 149.5/164.2 | **SSE2 by 25 %** |

      But the underlying gap is real and the n = 64 instance is far bigger than
      the ones the item was filed for. Better still, the intent was **already
      written down and silently ignored**: both complex128 n = 64 AVX2 rows in
      `cmd/gencodelets/specs.go` carried the comment _"SSE-width in practice;
      loses to dit64_radix4_sse2 -> stay below it"_ and had been given
      priorities 14/15 against the SSE2 codelet's 19 — which does nothing,
      because priority is only compared within a SIMD level. A hand-tuned
      decision had been inert since it was made.

      Fixed by splitting the two jobs `SIMDLevel` was doing. It stays the
      **eligibility** gate; a new optional `RankLevel` on `CodeletEntry` sets
      the level used for **ordering** only (unset = rank at `SIMDLevel`, so
      every other codelet is unaffected). The two complex128 n = 64 AVX2 rows
      now carry `RankLevel: SIMDSSE2`, which puts their 14/15 into the same
      ordering group as the SSE2 codelet's 19 and finally makes the comment
      true. `Lookup` for complex128 n = 64 returns `dit64_radix4_sse2`; every
      other size and both precisions are unchanged. Two regression tests pin
      the behaviour (`TestCodeletRegistryRankLevelDemotes` and
      `…UnsetKeepsSIMDOrder`).

      End to end the win survives plan overhead almost intact: `Plan.Forward`
      at complex128 n = 64 measures **158 ns** after the change (four pinned
      plans in one process all resolved to the new winner — see the wisdom item
      below — so they are four samples of the same path, spread 1.8 %). Against
      a 149.5 ns codelet that puts plan overhead at ~9 ns, so the pre-change
      path was ≈ 207 ns: **~24 % faster**.

      Use `RankLevel` to **demote**, not promote: promoting a narrow codelet
      also moves it ahead of its own tier's siblings on CPUs that have nothing
      better, whereas demoting a wide-ISA codelet only affects hosts that could
      already run it.

      Two side findings from the same run:

      - **complex64 needs no change** — the item's suspicion was right. At
        n = 16/32/64 the AVX2 codelets win by 2×+ (e.g. n = 16: 9.70 ns AVX2 vs
        20.68 ns SSE3).
      - **No measurable AVX↔SSE transition penalty** for picking a narrower
        codelet in AVX2 surroundings. An AVX2 n = 64 codelet followed by each
        n = 16 candidate cost 218 ns with the SSE2 codelet vs 228 ns with the
        AVX2 one — the SSE2 arm was, if anything, cheaper. So the outstanding
        `VZEROUPPER` item does not block this kind of cross-tier selection.
      - `dit16_radix4_generic` forward (20.93 ns) beats every SIMD codelet at
        complex128 n = 16, but its inverse (32.29) is the worst of the tuned
        ones. Not acted on — the registry has one priority per entry, not one
        per direction. Worth a look if per-direction ranking ever lands.

- [x] **Wisdom can never override a codelet-covered size.** Found while trying
      to pin a codelet for an end-to-end A/B. `planner.EstimatePlan` tried the
      registry (step 1) _before_ the wisdom cache (step 2), and the registry
      hits for every size with a codelet — i.e. all powers of two from 4 to 4096. A `PlanOptions.Wisdom` entry naming a specific codelet signature was
      silently ignored there; all four pinned plans in the experiment resolved
      to the registry winner. This made `LookupBySignature`'s careful
      "skip disabled codelets so a stale wisdom entry cannot resurrect one"
      logic unreachable for exactly the sizes it was written for.

      Resolved as **neither of the two options the item proposed**: wisdom now
      straddles the registry rather than sitting wholly above or below it,
      because its two kinds of entry carry different evidence.

      - A **signature** entry (`dit64_radix4_sse2`) names the same kind of
        thing the registry does — one specific codelet for one size — but was
        measured on this machine, against a registry order that is a
        compile-time constant. It now runs _before_ the registry
        (`bindWisdomCodelet`), which is what makes signature entries reachable
        at all and the disabled-codelet guard live.
      - A **strategy** entry (`stockham`) now stays _after_ the registry. The
        measurement behind it (`internal/fft.benchmarkStrategy`) times only the
        kernel path via `SelectKernelsWithStrategy` and never the codelet, so
        letting it displace a codelet would act on a comparison that was never
        made. This is also why the naive "swap the order" fix would have been a
        regression: every codelet-covered size that has ever been through
        `PlannerMeasure` has a strategy-name entry in its wisdom, and the swap
        would have routed all of them off the codelet and onto a generic kernel.

      Verified end to end: with `dit64_radix2_avx2` pinned for complex128
      n = 64, `Plan.Algorithm()` reports the pinned codelet while an unpinned
      plan still reports the registry winner `dit64_radix4_sse2`. Three
      regression tests in `internal/planner/planner_test.go` pin the ordering
      in both directions plus the stale-signature case.

      Side fix found on the way: `internal/fft.estimateWithStrategy` (the
      measure-mode and forced-strategy path) built its `PlanEstimate` without
      `TwiddleSize`/`PrepareTwiddle`. A codelet that wants a packed twiddle
      layout then got the plain table, failed its own length check and returned
      false, so the plan silently ran the fallback kernel while still reporting
      the codelet signature. On this CPU that hit
      `dit8192_radix4_then2_params_avx2` under `PlannerMeasure`; 256 and 1024
      escaped only because their top-ranked codelets need no prepared twiddles.
      The magnitude is untimed — the box was under load when it was found.

- [ ] **`PlannerMeasure` can pick a worse plan than `PlannerEstimate`.** Found
      while resolving the item above. `MeasureAndSelect` benchmarks kernel
      _strategies_ only; codelets are never a candidate. It then calls
      `estimateWithStrategy` with the winning strategy, which takes a codelet
      only if the top-ranked codelet's own algorithm happens to match. At
      complex64 n = 1024 the Stockham kernel beats the DIT kernel, so measure
      mode returns `stockham` where `PlannerEstimate` returns
      `dit1024_radix4_avx2` — a codelet that is very likely faster than either
      kernel. The fix is to enter the registry winner as an additional
      candidate in the measurement, and to record what actually won.
- [ ] **Make the 10 remaining mixed functions uniformly VEX.** The sweep left
      `Forward/InverseAVX2Complex64Asm`, both `AVX2Stockham` pairs, and the
      `Size1024Radix32x32` pair in both precisions mixed, because each has a
      legacy write whose upper half is live. Given that partial mixing is worth
      up to 152×, these are worth restructuring (renumber the aliased register
      so `Xn`/`Yn` no longer collide, then convert). Suggestive:
      `dit1024_radix32x32_avx2` measures 7.1 µs against `dit1024_radix4_avx2` at
      3.5 µs — it may already be paying a mixing penalty.
- [ ] **134 YMM/ZMM-using functions never execute `VZEROUPPER`.** Separate from
      the encoding sweep and untested so far: these return with the upper state
      dirty, so the cost lands on the _caller_ — Go's own SSE2-generated float
      code, and any pure-SSE2 codelet selected afterwards (e.g. the c128 n = 64
      winner `dit64_radix4_sse2`). 57 of the 151 amd64 asm files contain no
      `VZEROUPPER` at all. Measure a kernel-then-SSE2-kernel sequence before
      changing anything.
- [ ] **Plan-level overhead now dominates small transforms.** With the size-64
      codelet at 55 ns, `BenchmarkPlanForward_64` still reports ~155 ns — about
      100 ns of dispatch/validation per call that used to hide behind a 137 ns
      codelet. The same overhead sits on every size; it is simply invisible
      above ~1024. Profile the path from `Plan.Forward` to the codelet call.
- [ ] **The n = 2048 local minimum.** 0.29× FFTW3 forward, 0.31× inverse —
      the worst power-of-two point in the sweep, with 1024 (0.43×) and 4096
      (0.45×) either side of it. `dit2048_radix4_then2_avx2` is the
      incumbent. The AVX-512 item in P4.2 mentions reclaiming size 2048;
      this is the AVX2 tier and independent of that.
- [ ] **complex64 buys nothing between 1024 and 16384.** The c64/c128 ratio
      runs 1.10, 1.02, 1.14, 1.07, 1.08 across 1024–16384, against 1.6–2.1×
      at 256/512 and 1.58× at 32768. Part of that band is genuinely
      memory-bound, but 1.02 at n = 2048 is not a bandwidth story — it is
      the forward-path weakness from P5.0, and these five sizes are where it
      costs the most.

### P5.3 The SIMD build is slower than purego at two Bluestein primes

- [ ] **n = 1009 and n = 2003 regress under SIMD** — both at 0.86×, i.e. the
      default build is ~16% _slower_ than `-tags purego` for the same
      transform. n = 9973, the third rough prime measured, goes the right way
      (1.79×), so this is size-dependent rather than a blanket Bluestein
      problem. Both regressing lengths pad to a smaller power of two than
      9973 does; a plausible cause is a codelet selected for the padded
      sub-FFT whose fixed overhead is not amortized at that pad size, but
      that is a guess and wants a profile. Cheap to chase, and it violates
      §3 rule 4 (no regressions on the purego build) in the opposite
      direction from the usual.

### P5.4 Keep the comparison running

- [ ] **Put an external comparison in the release checklist.** Every finding
      in this section was invisible to the internal suite, because "faster
      than last week" and "faster than FFTW3" are different questions, and
      only the second one notices that a whole class of lengths never got
      the attention the power-of-two ladder did. Running `go-fft-bench`
      before a tag — even manually, even on one laptop — is cheap next to
      shipping another release in which 44100 loses to gonum. The harness
      refuses to start on a loaded machine, so the results are at least not
      accidentally measuring a compile storm.

- [ ] **Use the three hardware tiers deliberately.** As of 2026-07 three
      machines are reachable, and they are complementary rather than
      redundant — several findings in this section exist only because a
      result differed between them:

      - **Dev laptop (i7-1255U, AVX2, no AVX-512).** The only one with FFTW
                installed, so the only place the external gap can be measured. It
                thermally throttles hard, so interleave arms and trust ratios over
                absolutes (see the benchmarking protocol in §3).
              - **64-core host with no AVX at all** (SSE4.2 ceiling). Valuable
                _because_ it is limited: it is the only place the SSE2/SSE3 codelet
                tier is what dispatch actually selects. On any AVX2 machine those
                codelets lose the priority ladder and are exercised only by
                forced-strategy tests, so they ship effectively unbenchmarked in
                situ. Also a good proxy for the scalar-Go paths that dominate
                `purego` and WASM. Shared with other tenants — ratios, not
                absolutes.
              - **Xeon Gold 5218 (AVX2 + AVX-512).** The only AVX-512 hardware; see
                the AVX-512 item in P4.2. Doubles as a second, non-throttling AVX2
                reference, which is how the forward-vs-inverse anomaly in P5.0 got
                localized to the laptop. Small (2 vCPU, ~1.5 GB free) and has no
                gcc, so no cgo and no FFTW baseline.

              Access to the two servers is weekend-only, so none of this belongs in
              CI as-is; treat them as periodic validation sweeps, not routine
              iteration. FFTW can be used there without installing anything by
              shipping `libfftw3{,f}.so.3` plus `fftw3.h` from a matching distro
              release and pointing `CGO_CFLAGS`/`CGO_LDFLAGS`/`LD_LIBRARY_PATH` at
              them — but that needs a gcc on the target.

---

## Post-v1.0 Future (unchanged)

**Features**: DCT, Hilbert transform, STFT/spectrograms, audio/image examples,
Gonum ecosystem integration, optional GPU backends (kept out of the pure-Go
core).

**Community**: `CODE_OF_CONDUCT.md`, Dependabot, native ARM64 CI runner
(unblocks the NEON benchmarking items above).

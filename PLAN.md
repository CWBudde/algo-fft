# PLAN.md — algofft Roadmap to a Top-Notch v1.0

This roadmap is the source of truth for status and direction. It was rewritten
after a full architecture review to reflect the library **as it actually builds
and ships today**, not as previously aspired. Work is ordered by priority:
**P0 (correctness & integrity)** must land before performance polish, because
several headline features are currently broken or misleading.

Design philosophy lives in `goal.md`; component inventory in
`docs/IMPLEMENTATION_INVENTORY.md`; the pre-review per-size assembly plan is
preserved in git history (see the commit that replaced this file) and condensed
into the P2 backlog below.

---

## 1. Honest Current Status

### What genuinely works and is well-tested

- **Pure-Go core transforms**: DIT, Stockham, radix-2/3/4/5, mixed-radix,
  Bluestein (arbitrary length), six-step/eight-step large sizes.
- **Real FFT** (1D/2D/3D), **multi-dimensional** (2D/3D/N-D), **batch/strided**,
  **convolution/correlation** (complex & real), **complex64 + complex128**.
- **WebAssembly** target builds and tests under Node.
- **Testing** of the pure-Go paths is broad and passes under `-race`:
  reference-DFT cross-validation, Parseval/linearity/shift properties,
  round-trip, fuzz seed corpus, stress, concurrent, and 1D zero-alloc guards.
- **1D `Plan[T]` is genuinely concurrency-safe** and zero-allocation after
  creation, via an elegant resident-pointer + `sync.Pool` scratch cache.
- **Tooling**: strict `golangci-lint` (`default = all`), `treefmt`, a thorough
  `justfile`, and a multi-OS / multi-arch / WASM CI matrix with the race detector.

### What is broken, dormant, or misleading (the review's headline findings)

> **Update (2026-07, P0 complete):** findings 1–5 and 7 below are fixed — the
> `-tags asm` build compiles and is CI-gated (amd64/386 tested, arm64
> build-only pending P2.2), dead code is deleted, all plan types are genuinely
> concurrency-safe, the mixed-radix driver panics instead of returning garbage,
> every registered codelet is validated per-direction against reference
> spectra, and the docs/module-path/binary issues are resolved. Finding 6
> (global mutable state) is resolved as P1.1 (2026-07): kernel strategy is now
> per-plan via `PlanOptions.Strategy` and the size-only `benchDecisions` cache
> was dropped in favor of the richer Wisdom cache. The list is preserved below
> as the review record.

1. **SIMD is effectively non-functional as shipped.**
   - Default builds (`go build ./...`, `just build`, published module) are
     **100 % pure Go** — every SIMD dispatch file is gated behind
     `//go:build ... && asm && !purego`. A normal `go get` consumer gets **zero
     SIMD**, contradicting README/`goal.md` headline claims.
   - **`go build -tags asm ./...` does not compile** — 11 duplicate-declaration
     errors from committed twin files in `internal/kernels`
     (`dit_size384_mixed.go` ↔ `dit_384_decomp_128x3_amd64_asm.go`,
     `dit_size4096_sixstep_avx2.go` ↔ `dit_4096_sixstep_amd64_avx2.go`,
     `dit_size16384_sixstep_avx2.go` ↔ `dit_16384_sixstep_amd64_avx2.go`).
   - **No CI job builds or tests with `-tags asm`**, which is why the breakage
     went unnoticed. The large, genuine `.s` investment (AVX2/SSE2/SSE3/NEON/x86)
     is currently unreachable and unverified.

2. **Dead code that cannot compile.** The `legacy_radix2` build tag guards
   `dit_{32,64,128}_radix2.go` (~3,347 LOC) which redeclare the symbols already
   provided by the un-tagged `dit_size{32,64,128}_radix2.go` files — so the tag
   fails to build and the files are dead under every configuration. Plus
   ~520 dead lines in `internal/fft/mixedradix.go` (disabled WIP iterative
   drivers + an unused generic ping-pong driver).

3. **False concurrency-safety guarantees.** `Plan2D`, `Plan3D`, `PlanND`, and
   `PlanReal2D` document _"safe for concurrent use during transforms"_ but share
   mutable scratch buffers; two concurrent `Forward` calls on one instance race.
   Only 1D `Plan`/`PlanReal` are actually safe.

4. **Silent wrong-answer paths.** The mixed-radix butterfly `switch` has a
   `default: return` that yields garbage output with `err == nil` when the
   scheduler emits a radix the driver can't execute. The scheduler/driver
   contract is an implicit pair of global func pointers kept in sync by hand
   (the recent `0461cc4` fix patched one instance of exactly this class).

5. **Zero-alloc promise violated off the 1D path.** `PlanND` allocates
   `O(slices × dims)` per transform (`plan_nd.go:336,442,450`), the AVX2
   mixed-radix driver `make()`s per sub-transform, and asm/fallback paths
   recompute bit-reversal / factorization per call.

6. **Process-global mutable state, library-hostile.** Three singletons —
   `kernelStrategy`, `benchDecisions` (keyed by size only, so it crosses
   complex64/complex128), and `DefaultWisdom` — are exported as global mutators
   with no per-consumer isolation. Wisdom files have no version header and import
   non-atomically into the global.

7. **Docs that break users or mislead agents.** README's `go get` /import lines
   use the wrong module path `cwbudde/algofft` (missing dash → won't resolve);
   `CHANGELOG.md` lists shipped features under "Planned"; `GEMINI.md` describes a
   long-gone layout; ~6 MB of compiled binaries (`benchkernels`,
   `cmd/bench_compare/bench_compare`) are committed and not git-ignored.

---

## 2. Guiding Principles for the Road to v1.0

1. **Correctness before speed.** A dormant-but-correct pure-Go library beats a
   fast one that silently returns garbage or fails to build.
2. **The default build is the product.** Whatever `go get` gives a user must be
   the thing we test, benchmark, and document. No headline feature may depend on
   a non-default build tag unless CI builds that tag.
3. **No silent failure.** Every unhandled case fails loudly (error or panic),
   never garbage-with-`nil`-error.
4. **Docs match reality.** Every claim in README/`goal.md` is either true on the
   default build or explicitly scoped.
5. **Prefer generation over duplication.** Regular, size-parameterized code
   (codelets, registration tables, `*128` twins) should be generated.

---

## Priority 0 — Correctness & Build Integrity

**Gate: none of P1–P3 starts until P0 is green.** ✅ **P0 completed 2026-07.**

### P0.1 Repair and CI-gate the `-tags asm` build

- [x] Remove the duplicate declarations so `go build -tags asm ./...` compiles.
      Keep one file per (size, algorithm); delete the redundant twin
      (`dit_384_decomp_128x3_amd64_asm.go`, `dit_4096_sixstep_amd64_avx2.go`,
      `dit_16384_sixstep_amd64_avx2.go`, or their `dit_size*` counterparts).
- [x] Fix the amd64 decl↔TEXT drift: `decl.go` declares
      `Forward/InverseSSE2Size128Radix4Complex128Asm` with no matching `TEXT`
      (only the `...Then2...` symbol exists). Rename or delete the dead wrappers.
- [x] Add CI jobs: `go build -tags asm ./...` and `go test -tags asm ./...`
      (amd64), plus `-tags asm` under QEMU for arm64.
- [x] Add a lint/CI check that every `//go:noescape` decl has a matching `TEXT`
      symbol, to prevent decl↔asm drift recurring.

### P0.2 Delete dead code

- [x] Delete `internal/kernels/dit_{32,64,128}_radix2.go` (`legacy_radix2`,
      ~3,347 LOC — never compiles). Retire the `legacy_radix2` tag.
- [x] Delete the ~520 dead lines in `internal/fft/mixedradix.go`
      (`mixedRadixIterativeComplex64/128`, the unused generic
      `mixedRadixRecursivePingPong[T]`).
- [x] Delete the dead stub `planBitReversal` (`plan.go:261`, always returns
      `nil`) — either implement it (so non-pooled plans get `bitrev` and the
      strided radix-2 fast path can trigger) or remove it and the dead
      `ForwardStrided` fast-path branch. Remove the `// (FIXED)` marker.
- [x] Remove the duplicate `ditAutoThreshold` in
      `internal/fft/kernels_fallback.go:9` (unused shadow of the planner copy).

### P0.3 Fix false concurrency-safety (code or docs)

- [x] Give `Plan2D`/`Plan3D`/`PlanND`/`PlanReal2D` a per-call scratch cache like
      1D `Plan` (preferred), **or** correct their doc comments to
      _"Clone per goroutine; a single instance is not safe for concurrent
      transforms"_ (as `PlanReal` already correctly states).
- [x] Add a `-race` concurrent test per multi-dim plan type to lock in whichever
      guarantee is chosen.

### P0.4 Eliminate silent wrong-answer paths

- [x] Replace every mixed-radix butterfly `default: return` (`mixedradix.go`
      ~364/494/639) with a `panic` (unschedulable radix is a programming error,
      not a runtime input error).
- [x] Make the scheduler/driver contract explicit: have the driver expose its
      executable radix set and validate the emitted schedule against it, instead
      of hand-synced `codeletSchedulable64/128` global hooks.
- [x] Strengthen the registry sweep (`codelet_roundtrip_all_test.go`) from
      round-trip-only to **forward-vs-`reference.NaiveDFT`** for every registered
      codelet, and add a meta-test asserting every `Signature` has reference
      coverage (a compensating forward/inverse bug currently passes).

### P0.5 Fix user- and agent-breaking docs

- [x] Fix README module path everywhere: `cwbudde/algofft` → `cwbudde/algo-fft`
      (lines 3, 5, 49, 59, 174) so `go get` and the examples work.
- [x] Rewrite `CHANGELOG.md` to reflect implemented features (move Core/Real/
      Bluestein/SIMD/multi-dim out of "Planned").
- [x] Point `GEMINI.md` at `AGENTS.md` (like `CLAUDE.md` does) or rewrite it to
      the real `internal/kernels` + `internal/asm` layout and current phase.
- [x] Scope the SIMD claims in README/`goal.md`: state plainly that SIMD requires
      `-tags asm` (until P2 makes it default), or hold the claim until then.
- [x] Fix `justfile:43` `-tags "amd"` → `-tags "asm"` and the matching
      AGENTS.md line ("amd64 uses `-tags amd`" is false).

### P0.6 Repo hygiene

- [x] `git rm` the committed binaries (`benchkernels`,
      `cmd/bench_compare/bench_compare`, ~6 MB) and add `.gitignore` rules for
      extensionless Go binaries (e.g. `/benchkernels`, `cmd/*/bench_compare`, or a
      `/build/` convention).

---

## Priority 1 — Architecture Hardening

### P1.1 Scope the global mutable state ✅ **completed 2026-07.**

- [x] Removed the `kernelStrategy` global and its `SetKernelStrategy`/
      `GetKernelStrategy` accessors entirely; kernel strategy is now chosen
      per-plan via `PlanOptions.Strategy` (mirrors `PlanOptions.Wisdom`'s
      per-instance model). `resolveKernelStrategy` is a pure function of
      `(size, strategy)` with no process-global reads. Migrated
      `examples/recursive_fft` and `cmd/benchkernels` to `PlanOptions.Strategy`.
- [x] Dropped `benchDecisions`/`RecordBenchmarkDecision` in favor of the richer
      Wisdom cache (size + precision + CPU features) — one tuning cache, not two.
      `cmd/benchkernels` records tuning solely through Wisdom export
      (`-wisdom`). (The extra "direction" dimension is deferred: kernel strategy
      is direction-independent here and it would inflate the persisted Wisdom
      format.)
- [x] Fixed the SSE2-complex128 wrappers (`asm_amd64.go`) that re-read the global
      strategy at execution time: they now select the scalar fallback via the
      global-free size heuristic (`ResolveKernelStrategy`), matching their
      complex64 siblings. The genuine 386 wrappers already dispatched by size, so
      no change was needed there.

### P1.2 Harden the Wisdom format ✅ **completed 2026-07.**

- [x] Added a version/magic header (`# algofft-wisdom v2`) written by `Export` and
      required by `Import`; unversioned or unknown-version files are rejected with
      a clear error instead of being mis-parsed (`internal/planner/wisdom.go`).
- [x] Made `Import` atomic: it parses+validates the whole file into a temporary
      map (size range, precision, feature-mask width, algorithm-name charset) and
      merges into the live cache only on full success, so a malformed line leaves
      the cache untouched.
- [x] Widened the CPU-feature mask to distinguish SSE3 from SSE2 (new bit layout
      `SSE2|SSE3|AVX2|AVX512|NEON`; `CPUFeatureMask`/`MakeWisdomKey` take a
      `hasSSE3` argument and all call sites pass `features.HasSSE3`). `Timestamp`
      now drives staleness: added `Wisdom.EvictOlderThan` and
      `ImportWithMaxAge` / public `ImportWisdomWithMaxAge` to drop stale entries.
- [x] Reduced `PlannerMeasure` timing noise: raised iteration counts and added
      multi-trial **median** sampling (outlier rejection) in `benchmarkStrategy`
      (`internal/fft/measure.go`).

### P1.3 Zero-allocation parity across all plan types ✅ **completed 2026-07.**

- [x] `PlanND` is now zero-alloc: the per-dimension slice buffer and the
      `reducedDims`/`coords` index scratch are preallocated in `planNDScratch`
      and threaded through `transformDimension`/`sliceIndexToOffset` (the base
      offset is computed once per slice instead of twice), modeled on the 3D
      path (`plan_nd.go`).
- [x] Hoisted the per-sub-transform `make()`s out of `mixedradix_avx2.go`
      (gathered twiddle + kernel scratch) into pooled leaf buffers
      (`mrScratchPool64/128`); the size-384 complex128 codelet likewise now
      precomputes its 128-point sub-twiddle once and pools its output/scratch
      (`dit_384_decomp_128x3_amd64_asm.go`).
- [x] `ComputeBitReversalIndices` is memoized for the AVX2 Stockham/complex128
      wrappers (`bitrev_cache_amd64.go`, wired in `asm_amd64.go`), and
      `IsHighlyComposite` is now allocation-free (divides out 2/3/5 in place
      instead of materializing the factor slice, `internal/math/factor.go`) so
      it no longer allocates on the per-transform dispatch hot path
      (`kernels_fallback.go`). Two extra default-build leaks found and fixed
      along the way: the mixed-radix schedule buffer (a `[64]int` that escaped
      through the recursion hook) is now pooled (`radixSchedulePool`,
      `mixedradix.go`), and the size-specific AVX2 dispatch no longer builds a
      per-transform strategy closure (`kernels_amd64_size_specific.go`).
- [x] Extended `plan_alloc_test.go` to guard 2D/3D/ND (both precisions) and the
      mixed-radix path (96/768/1536, both precisions); the real 2D variant was
      already covered. All are verified 0-alloc on both the default and
      `-tags asm` builds.

### P1.4 Reduce duplication in the plan layer

- [x] Split `plan.go` (1,425 lines, near the 1,500 cap): extract the ~270 lines
      of triplicated `complex64/complex128/default` alloc type-switches into one
      generic aligned-alloc helper (`mem.AllocAligned[T]` in `plan_alloc.go`), and
      move `Close`/`Reset`/`Clone` to `plan_lifecycle.go`. `plan.go` is now 786
      lines.
- [x] Retire `PlanReal` in favor of the generic `PlanRealT` (the former is a
      verbatim non-generic duplicate kept "for backward compatibility"): it is now
      a type alias for `PlanRealT[float32, complex64]`.
- [x] De-duplicate the `*128` DSP twins (`Convolve128`, `CrossCorrelate128`, …) by
      making the helpers generic (`convolveT[T]`, `crossCorrelateT[T]`,
      `convolveRealT[F,C]`) with thin wrappers, mirroring the already-generic
      2D/3D/ND plans. (The 2D/3D/ND plans were already generic; no template
      generator was introduced.)

### P1.5 Clean up dispatch ✅ **completed 2026-07.**

- [x] Factored the 4×-duplicated `SelectKernels[T]` type-switch/assertion
      boilerplate (`dispatch.go`) into `bridgeKernel[T]`/`bridgeKernels[T]`
      helpers shared by `SelectKernels` and `SelectKernelsWithStrategy`. The
      failed type assertion is no longer discarded with `_`: a mismatch (a
      dispatch bug, impossible under the `Complex` constraint) now panics with a
      descriptive message instead of silently returning a nil kernel. A
      legitimately typed-nil kernel still asserts `ok == true`, so behavior is
      preserved.
- [x] Documented the `stockham_packed_toggle_*.go` inversion instead of removing
      it (no correctness bug behind it): the pure-Go packed radix-4 Stockham path
      is enabled on the default build but disabled under `-tags asm`, where the
      SIMD codelet path in `plan.go` (checked first) supersedes it. Added
      rationale comments to both toggle files and to the two `plan.go` dispatch
      gates (`Forward`/`Inverse`) clarifying it is a dispatch de-duplication, not
      a workaround.

### P1.6 Introduce a codelet generator

- [x] Added a `go:generate` generator (`cmd/gencodelets`) that emits the ~164
      `Register(CodeletEntry{...})` blocks from a single declarative table
      (`cmd/gencodelets/specs.go`). The four hand-written `codelet_init*.go`
      register bodies were replaced by generated `codelet_init_*.gen.go` files
      (generic/avx2/sse2/neon), with the scaffolding (init, `wrapCodelet*`,
      registries) moved to a hand-written `codelet_registry.go`. The table was
      seeded by AST-extracting the originals, so no field was hand-transcribed;
      a registry snapshot confirmed the generated registrations are behaviorally
      identical to the originals (62 generic + 53 AVX2 + 28 SSE2 + 21 NEON), and
      `go generate` is idempotent. **Generating the codelet _bodies_
      (the ~21k lines of hand-tuned unrolled DIT/radix math) was deliberately
      left out of scope** — only the registration boilerplate (the actual
      dead-copy/drift source) is now generated.
- [x] Normalized kernel file naming: `git mv`d the four `dit_size*` fossils to
      the `dit_<size>_<radix>.go` convention (`dit_32_radix2.go`,
      `dit_64_radix2.go`, `dit_128_radix2.go`, `dit_32_mixed24.go`) and renamed
      the 12 off-convention unexported functions to add the `Radix2` marker
      (`forwardDIT32Complex64` → `forwardDIT32Radix2Complex64`, …). The `legacy_radix2`
      tag was already removed by P0.2; the exported size-default aliases
      (`ForwardDIT32Complex64`, matching `ForwardDIT256Complex64`) were kept.
- [x] Fixed the `Lookup`/`lookupUnlocked` divergence in
      `internal/planner/codelet.go`: `lookupUnlocked` now skips disabled
      (`Priority < 0`) codelets like `Lookup` does, so `GetAvailableSizes` can no
      longer advertise a size served only by a disabled codelet. Added a
      regression test (`TestCodeletRegistryGetAvailableSizesDisabled`) asserting
      `GetAvailableSizes` agrees with `Lookup`.

---

## Priority 2 — SIMD That Actually Ships

**Precondition: P0.1 done (asm builds and is CI-tested).**

### P2.1 Make SIMD reachable on the default build ✅ **completed 2026-07.**

- [x] Delivery model decided and implemented: the asm kernels are folded into
      the default build behind runtime CPU detection (the plan's preferred
      option). The `asm` tag was removed from all 225 build constraints —
      SIMD files now gate on `<arch> && !purego`, fallbacks on the negation —
      so a plain `go get` consumer gets SIMD, selected at runtime via
      `cpu.DetectFeatures()`. `-tags purego` is the supported pure-Go opt-out
      (tag was already respected; now it is the _only_ gate). The
      `stockham_packed_toggle` pair now keys on
      `(amd64 || arm64 || 386) && !purego` (formerly the `asm` tag);
      `-tags asm` remains accepted as a harmless no-op for existing
      scripts. Docs updated (README, goal.md,
      AGENTS.md, CHANGELOG) — the SIMD claim is now true on the default build.
- [x] Fallback correctness parity is CI-gated: every `test-arch` matrix leg
      (amd64, 386, arm64/QEMU, darwin/arm64, windows/amd64) and a dedicated
      `test-unit` job build and test `-tags purego` alongside the default
      SIMD build; `just test-purego` runs it locally.

### P2.1a Clean up asmdecl findings ✅ **completed 2026-07.**

- [x] `go vet -tags asm ./...` reported ~1,000 `asmdecl` findings across
      `internal/asm/**.s`; all fixed and vet is now clean on amd64, arm64, and 386. Three classes: (1) 984 slice FP references using the base-pointer
      name at the length-field offset (`src+32(FP)` → `src_len+32(FP)`;
      offsets were numerically correct, only names renamed — codegen is
      unchanged since the assembler resolves by offset); (2) 25 wrong TEXT
      frame sizes (`ScaleComplex64*Asm` declared `$0-32` for a 28-byte
      frame; 22 x86/386 kernels declared `-60`/`-64` for their 61-byte
      5-slices+bool frames); (3) 10 arm64 NEON complex128 size-specific
      kernels (8/32/64/128/256) had `TEXT` symbols but no Go declaration —
      declared in `arm64/decl.go` (wiring into codelet registration stays
      with P2.3). `go vet -tags asm` now runs in CI per architecture
      (`test-arch.yaml`) and locally via `just vet-asm`, closing the reverse
      direction of the P0.1 decl↔TEXT drift check.

### P2.2 Fix known-incorrect kernels

- [x] **ARM64 NEON complex64 kernels produce wrong results** — fixed (2026-07).
      Three bugs: (1) the generic `forwardNEONComplex64Asm`/`inverse` read
      bit-reversal indices from the uninitialized scratch buffer (the `.s` was
      written for a 5-arg `bitrev []int` signature the 4-arg Go decl never
      supplied) — now computed on-the-fly via `CLZ` like the complex128 kernel;
      (2) the size-128 radix-4-then-2 inverse conjugated twice in stage 1
      (sign-flip **and** store swap), cancelling out — now stores X1/X3 in
      forward order; (3) the size-64 radix-4 NEON kernels used a self-consistent
      but reference-wrong convention and were never selected in dispatch (radix-2
      wins for n=64) — removed. `TestAllKernelsCorrectness`,
      `TestNEONSizeSpecificComplex64`, `TestKernelConsistency`, and the full
      `internal/fft` asm suite now pass under QEMU.
- [x] **Remaining arm64 `-tags asm` faults** — fixed (2026-07). The
      `TestInPlaceAllCodelets64` and `TestPlan2DTransformsNoAllocsComplex64`
      segfaults shared one root cause: the size-specific NEON **complex64**
      kernels (4/8/16/32/64/128/256 + the 128/32 mixed variants) corrupted the
      in-place (`dst==src`) copy-back. The loop advanced `R1` to
      `scratch_base + i*8`, then computed the destination as
      `dst_base + R1` = `dst + scratch + i*8` — a wild pointer. Keep the offset
      in `R1` and use a scratch register for the load address. Out-of-place
      transforms skip the copy-back, which is why only the in-place / 2D paths
      faulted. (The complex128 kernels already kept the offset separate.) A
      third, unrelated failure surfaced once the crash was gone —
      `TestPlanAlgorithmSize512Radix4Then2Complex128` asserted an amd64-only
      algorithm; made arch-aware (arm64 has no NEON radix-4-then-2 for 512).
      **The full `go test -tags asm ./...` now passes on arm64 under QEMU (all
      10 packages).** Next: flip `test-arch.yaml` from build-only to test on
      arm64 (was blocked by these faults; see Testing & CI Hardening / P0.1).
- [x] AVX2 Stockham "compiles/runs but produces wrong results" (old Phase
      14.4): **closed 2026-07, not reproduced** — the AVX2 Stockham complex64
      asm kernel matches `reference.NaiveDFT` within float32 tolerance at
      sizes 16–4096 (maxErr ~2e-5 at n=4096) and `TestAllKernelsCorrectness`
      (all strategies) passes on the default (SIMD) build, which CI now runs
      on every push (P2.1). The P0.4 forward-vs-reference sweep gates any
      future changes.
- [x] Size-16 radix-16 on x86/386 — **fixed and re-enabled 2026-07**. Root
      cause: the kernel's work buffer aliases `dst` whenever `dst != src`, so
      the final 4×4 transpose ran in place, and its store order clobbered
      slots 16/48 (writing the transposed rows 2/3) before reading them for
      outputs 8/9/12/13. Fixed by preloading the entire (16,48)↔(64,96) swap
      into registers before storing (forward and inverse). Re-enabled in the
      SSE3 dispatch (`kernels_386_asm.go`, identity bit-reversal) and covered
      by a forward-vs-reference + inverse case in `asm_386_test.go`.

### P2.3 Higher-radix / larger-size kernel backlog (condensed)

Reuse-friendly decompositions, highest benefit/effort first. Each item: add the
`.s`, wire the codelet with priority, add forward-vs-reference + round-trip tests,
benchmark vs the current path with `benchstat`.

| Priority | Size  | Decomposition     | Reuses            | Status         |
| -------- | ----- | ----------------- | ----------------- | -------------- |
| 1        | 4096  | 64×64 (2-stage)   | FFT-64 kernel     | planned        |
| 2        | 1024  | 32×32 (2-stage)   | FFT-32 kernel     | planned        |
| 3        | 256   | 16×16 (2-stage)   | radix-16 kernel   | done (Go+AVX2) |
| 4        | 16384 | 128×128 (2-stage) | needs FFT-128     | partial        |
| 5        | 8192  | 64×128 (2-stage)  | needs FFT-128     | partial        |
| 6        | 512   | radix-8 / 16×32   | radix-8 infra     | done (Go)      |
| 7        | 2048  | 32×64 (2-stage)   | FFT-32/64 kernels | planned        |

- [x] complex128 large-size AVX2 — **completed 2026-07**. 512/1024/2048/8192/
      16384 were already covered (radix-4-then-2, 32×32, radix-4); the last gap,
      size 4096, is closed by a 6-stage radix-4 AVX2 kernel
      (`avx2_f64_size4096_radix4.s`, priority 30, shares the complex64 kernel's
      `bitrev4096_r4` table) — ~1.7× faster than the generic six-step path it
      replaces (36 µs vs 61 µs on Xeon 2.1 GHz).
- [x] SSE2 coverage for 512 mixed-2/4 and 1024 radix-4 (both precisions) — the
      non-AVX2 fallback path. **Completed 2026-07**: four kernels — SSE2
      complex128 + SSE3 complex64 for size 512 radix-4-then-2 (four radix-4
      stages + one radix-2 stage, mixed-digit-reversal permutation) and size
      1024 radix-4 (five stages, base-4 digit reversal). Registered via
      `gencodelets` (priority 12, above the size-512 radix-2 codelets) and
      wired into the SSE3 size-specific dispatch; each is validated
      forward/inverse vs `reference.NaiveDFT` plus round-trip and in-place.
      On the non-AVX2 tier, 1024 complex64 is ~2.8× faster than the generic
      SSE3 kernel; 512 complex128 is ~15 % faster than the radix-2 codelet.
- [x] ARM64 NEON remaining complex128 — **completed 2026-07**. The five
      declared-but-unregistered kernel pairs flagged in P2.1a (8 radix-4,
      32 mixed-2/4, 64 radix-4, 128 mixed-2/4, 256 radix-2) are wired into
      codelet registration via `gencodelets`, alongside the size-specific
      4/8/16/32 kernels that were previously reachable only through the
      `internal/fft` dispatch layer — the complex128 NEON registry now mirrors
      the complex64 one. Every size-specific NEON complex128 kernel is
      validated forward-vs-reference, round-trip, **and in-place** (the P2.2
      copy-back bug class) in `internal/asm/arm64/neon_f64_size_specific_test.go`;
      the P0.4 registry sweep covers the new codelets on top. Full arm64 suite
      passes under QEMU.
- [ ] ARM64 NEON sizes 512+: **evaluation blocked on native ARM64 hardware** —
      QEMU timings are not representative (see `bench-arm64` note), so a
      size-specific 512/1024 kernel can't be benchmark-justified from CI today.
      Sizes 512/1024 are already NEON-served by the generic DIT kernels
      (registered as priority-1 codelets); revisit alongside the post-v1.0
      "native ARM64 CI runner" item.

### P2.4 New instruction sets

- [x] AVX-512 kernels (`HasAVX512` was detected but dead) — **first increment
      completed 2026-07**. Generic radix-2 DIT kernels for both precisions
      (`avx512_f32_generic.s` / `avx512_f64_generic.s`, forward + inverse,
      AVX512F only): ZMM registers process 8 complex64 / 4 complex128
      butterflies per iteration, permutation uses the shared per-size
      bit-reversal table (computing it on the fly costs ~30% at n=1024).
      Wired into `selectKernels*` as a new top tier
      (`kernels_amd64_avx512.go`): tuned AVX2 size-specific codelets keep
      the sizes they cover; the AVX-512 kernel serves the remaining DIT
      sizes and all Stockham-resolved sizes, beating the AVX2 generic
      radix-4/mixed path by 1.2–1.4× and the AVX2/Go Stockham paths by
      1.15–2.4× at every measured size 1024–2²¹ except complex64 2¹⁹ (~5%
      slower; benchmark tables in `kernels_amd64_avx512.go`, Xeon 2.8 GHz).
      Plan-level (`NewPlan32`, non-codelet sizes 32768/65536/131072):
      12–21% faster, still zero-alloc. Validated forward/inverse vs
      `reference.NaiveDFT`, round-trip, and in-place at sizes 16–8192 plus
      a dispatch-level strategy sweep (`asm_amd64_avx512_test.go`).
- [x] Per-size AVX-512 codelets via `gencodelets` — **completed 2026-07**.
      A new `avx512` generator target registers the generic AVX-512 radix-2
      kernel as complex64 codelets at 1024/4096/8192/16384, the sizes where
      it beats the best AVX2 codelet on AVX-512 hardware (Xeon 2.8 GHz):
      plan-level forward 19.9→8.2 µs (1024, 2.4×), 42.8→34.4 µs (4096),
      149→83 µs (8192, 1.8×), 236→195 µs (16384, 1.2×); the inverse
      direction shows the same winners; all still zero-alloc. AVX2 codelets
      keep 2048 (faster) and every complex128 size (faster at ≥2048, tie at
      1024). Codelet selection prefers the higher SIMD level over priority,
      so only benchmark-winning sizes are registered; the measurements and
      rationale live in `internal/kernels/dit_avx512_amd64.go`.
- [ ] AVX-512 follow-ups: higher-radix / per-size-tuned AVX-512 variants
      should widen the gap (and could reclaim 2048 and the complex128
      sizes, where the AVX2 codelets still win). Needs AVX-512 CI/bench
      hardware for regression tracking.
- [ ] Revisit WASM SIMD — **rechecked 2026-07**: Go's `GOEXPERIMENT=simd`
      intrinsics (golang/go#73787) gained amd64 support in Go 1.26 and
      Wasm/ARM64 `archsimd` support in the Go 1.27 RC; still experimental
      and this module targets Go 1.25. Revisit once the experiment
      graduates or the toolchain floor moves.

---

## Priority 3 — API Completeness & Polish

✅ **completed 2026-07.**

- [x] Removed the shipped-but-inert public options: `Radices` (never read) and
      `WorkspacePolicy`/`Workspace` (all three enum values were "not yet
      implemented") are gone from `PlanOptions`, along with their normalization
      code and tests. No dead knobs in the v1.0 API.
- [x] Fixed the pooled-pool API: `NewPlanFromPool`/`NewPlanFromPoolWithOptions`
      (which took an `internal/fft.BufferPool` no external caller can name) are
      unexported; `NewPlanPooled`/`NewPlanPooledWithOptions` remain the public
      surface. Their contract now matches `newPlanWithFeatures`: `opts.Planner`
      is honored (via `selectPlanEstimate`), and Bluestein/recursive sizes are
      served by the regular allocator instead of being rejected with a
      misleading `ErrNotImplemented`/`ErrInvalidLength`.
- [x] Introspection parity (`plan_introspect.go`): `Plan2D/3D/ND` and
      `PlanReal2D/3D` expose `Meta()` plus per-axis `KernelStrategies()`/
      `Algorithms()` (one entry per dimension — a single resolved strategy is
      ill-defined for composite plans); `PlanRealT` and `FastPlanReal32/64`
      delegate singular `Meta()`/`KernelStrategy()`/`Algorithm()` to their
      underlying complex plan; `FastPlan` gained all three plus `Close()`
      (`FastPlanReal32/64` too).
- [x] Resolved the `InPlace` naming outlier: 1D `Plan` and `FastPlan` now have
      `ForwardInPlace` matching the multi-dim plans; the forward-only `InPlace`
      remains as a deprecated alias and internal callers were migrated.
- [x] Added plan-reuse DSP types (`convolver.go`): `Convolver[T]`,
      `Correlator[T]`, and `RealConvolver[F,C]` hold one plan plus pooled
      scratch (residentCache pattern — concurrency-safe, zero allocations in
      steady state, locked in by `AllocsPerRun` tests) so DSP-in-a-loop doesn't
      re-plan/re-allocate; the one-shot helpers' docs point to them.
- [x] Error-handling consistency: 2D/3D and real-2D/3D constructors now wrap
      `ErrInvalidLength` and child-plan failures with dimension context via
      `%w`, matching `plan_nd.go`; transform-time validation keeps bare
      sentinels everywhere (fast path, matches 1D).
- [x] GoDoc audit: AST sweep of the root package found 8 undocumented exported
      consts (`Kernel*` strategy values, `Precision*`), now documented. Enabled
      revive's `exported` doc-comment rule scoped to the public package
      (`internal/`+`cmd/` excluded via `^exported:` text match in
      `.golangci.toml`), so the AGENTS.md requirement is enforced in CI.

---

## Testing & CI Hardening (cross-cutting)

- [x] **SIMD + fallback in CI** (see P0.1/P2.1) — every arch matrix leg builds,
      vets, and tests both the default (SIMD) build and `-tags purego`.
- [x] **Lint gate green** (2026-07): `golangci-lint run` now exits 0 (was
      ~2,800 findings once the P2.1 fold made the SIMD files visible). Three
      prongs: (1) ~110 genuinely dead symbols deleted after per-arch/per-`.s`
      reference triage — unused complex128 SSE2/AVX2 dispatch wrappers in
      `internal/fft/asm_amd64.go`, ~50 dead kernel aliases in
      `internal/fft/kernels.go`, `bitrev_identity.go`, `wrapAsmDIT64/128`,
      stale test helpers (the 9 asm-referenced bitrev tables kept with
      `//nolint:unused`); (2) mechanical fixes — 70 `x = x + y` → `x += y`,
      ~130 auto-fixes (intrange/godot/perfsprint/…), ~110 wrapper closures
      collapsed to direct function references, long lines wrapped, missing
      `b.Helper()`s, `dit_64_radix2.go` and `asm_amd64_avx2_test.go` split to
      respect the 1500-line cap; (3) documented config decisions in
      `.golangci.toml` — disabled linters that fight the codebase's nature
      (varnamelen/wsl_v5/exhaustruct/dupl/paralleltest/gochecknoglobals/…,
      each with a rationale comment), path-scoped exclusions for the
      hand-unrolled kernels, tests, cmd tools, and generic bridge files, and
      cyclop/funlen limits raised to 20/100-80.
- [x] **Coverage gate** (2026-07): `codecov.yml` added — project gate at the
      reconciled target **90 %** (threshold 0.5 %, patch informational;
      `cmd/`, `examples/`, and the `internal/asm` bridge stubs excluded).
      `CONTRIBUTING.md` raised from >80 % to match `AGENTS.md`. Weakest
      packages raised: `internal/transform` 41.0 → 90.6 % (the packed-Stockham
      engine is now tested on every build via a toggle-independent entry
      point, plus forced radix-2/-16 recursive-inverse combine tests),
      `internal/cpu` 78.5 → 100 %, `internal/fft` 74.9 → 80.7 % (forced-SSE3
      dispatch tier now verified against reference on AVX2 machines, generic
      pool/mul/scale helpers covered). Root was already 86.3 %; total 91.4 %+.
- [x] **Benchmark gate** (2026-07): moved off the PR gate to a nightly
      (`schedule`) + `workflow_dispatch` workflow — shared runners are too
      noisy for a reliable threshold. The baseline comparison stays
      informational until a `benchmarks/baseline-<os>.txt` is committed.
- [x] **Pin toolchain** (2026-07): `test-bench.yaml` now uses
      `go-version-file: go.mod` like every other job; `golangci-lint-action`
      pinned to `v2.12.2` instead of `latest`.
- [x] **Continuous fuzzing** (2026-07): `test-fuzz.yaml` runs every fuzz
      target beyond its seed corpus — 20 s each on PRs, 5 min each nightly —
      and uploads new crashers as artifacts. Its first run immediately found
      a divide-by-zero in three fuzz harnesses on empty input (fixed; the
      minimized crashers are committed as the regression corpus under
      `testdata/fuzz/`).
- [x] **Property-test parity** (2026-07): `plan_properties_test.go` applies
      Parseval/linearity/shift to the public 1D `Plan[T]` (both precisions)
      across all dispatch families — powers of two, mixed radix 2/3/5, and
      Bluestein primes — complementing the raw-kernel property suite in
      `internal/fft`.

---

## v1.0 Release — Definition of Done

v1.0 ships only when **all** of the following hold:

- [ ] `go build ./...` and `go build -tags purego ./...` both compile on amd64
      and arm64; both are gated in CI.
- [ ] `go test -race ./...` and `go test -tags purego ./...` pass; 5× repeat run
      is flake-free.
- [ ] No dead build tags, no committed binaries, no false doc guarantees.
- [ ] Every public option and constructor is either implemented or removed —
      no "not yet implemented" in the exported surface.
- [ ] Coverage meets the single agreed target for non-asm code; the asm path has
      forward-vs-reference tests for every registered codelet.
- [ ] README/`goal.md`/CHANGELOG are accurate on the default build; module path
      resolves; `pkg.go.dev` renders.
- [ ] `docs/IMPLEMENTATION_INVENTORY.md` and `BENCHMARKS.md` regenerated with
      committed baselines and the CPU/hardware used.
- [ ] Tag `v1.0.0`, GitHub release notes, `.github/ISSUE_TEMPLATE` +
      `PULL_REQUEST_TEMPLATE.md`.

---

## Post-v1.0 Future

**Performance** (as users request): cache-blocked variants for sizes above L2,
SoA (split real/imag) layout as a v2 API, parallel batch API, AVX-512 breadth.

**Features**: DCT, Hilbert transform, STFT/spectrograms, audio/image examples,
Gonum ecosystem integration, optional GPU backends (kept out of the pure-Go core).

**Community**: `CODE_OF_CONDUCT.md`, Dependabot, native ARM64 CI runner.

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
   `PlanReal2D` document *"safe for concurrent use during transforms"* but share
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

**Gate: none of P1–P3 starts until P0 is green.**

### P0.1 Repair and CI-gate the `-tags asm` build

- [ ] Remove the duplicate declarations so `go build -tags asm ./...` compiles.
      Keep one file per (size, algorithm); delete the redundant twin
      (`dit_384_decomp_128x3_amd64_asm.go`, `dit_4096_sixstep_amd64_avx2.go`,
      `dit_16384_sixstep_amd64_avx2.go`, or their `dit_size*` counterparts).
- [ ] Fix the amd64 decl↔TEXT drift: `decl.go` declares
      `Forward/InverseSSE2Size128Radix4Complex128Asm` with no matching `TEXT`
      (only the `...Then2...` symbol exists). Rename or delete the dead wrappers.
- [ ] Add CI jobs: `go build -tags asm ./...` and `go test -tags asm ./...`
      (amd64), plus `-tags asm` under QEMU for arm64.
- [ ] Add a lint/CI check that every `//go:noescape` decl has a matching `TEXT`
      symbol, to prevent decl↔asm drift recurring.

### P0.2 Delete dead code

- [ ] Delete `internal/kernels/dit_{32,64,128}_radix2.go` (`legacy_radix2`,
      ~3,347 LOC — never compiles). Retire the `legacy_radix2` tag.
- [ ] Delete the ~520 dead lines in `internal/fft/mixedradix.go`
      (`mixedRadixIterativeComplex64/128`, the unused generic
      `mixedRadixRecursivePingPong[T]`).
- [ ] Delete the dead stub `planBitReversal` (`plan.go:261`, always returns
      `nil`) — either implement it (so non-pooled plans get `bitrev` and the
      strided radix-2 fast path can trigger) or remove it and the dead
      `ForwardStrided` fast-path branch. Remove the `// (FIXED)` marker.
- [ ] Remove the duplicate `ditAutoThreshold` in
      `internal/fft/kernels_fallback.go:9` (unused shadow of the planner copy).

### P0.3 Fix false concurrency-safety (code or docs)

- [ ] Give `Plan2D`/`Plan3D`/`PlanND`/`PlanReal2D` a per-call scratch cache like
      1D `Plan` (preferred), **or** correct their doc comments to
      *"Clone per goroutine; a single instance is not safe for concurrent
      transforms"* (as `PlanReal` already correctly states).
- [ ] Add a `-race` concurrent test per multi-dim plan type to lock in whichever
      guarantee is chosen.

### P0.4 Eliminate silent wrong-answer paths

- [ ] Replace every mixed-radix butterfly `default: return` (`mixedradix.go`
      ~364/494/639) with a `panic` (unschedulable radix is a programming error,
      not a runtime input error).
- [ ] Make the scheduler/driver contract explicit: have the driver expose its
      executable radix set and validate the emitted schedule against it, instead
      of hand-synced `codeletSchedulable64/128` global hooks.
- [ ] Strengthen the registry sweep (`codelet_roundtrip_all_test.go`) from
      round-trip-only to **forward-vs-`reference.NaiveDFT`** for every registered
      codelet, and add a meta-test asserting every `Signature` has reference
      coverage (a compensating forward/inverse bug currently passes).

### P0.5 Fix user- and agent-breaking docs

- [ ] Fix README module path everywhere: `cwbudde/algofft` → `cwbudde/algo-fft`
      (lines 3, 5, 49, 59, 174) so `go get` and the examples work.
- [ ] Rewrite `CHANGELOG.md` to reflect implemented features (move Core/Real/
      Bluestein/SIMD/multi-dim out of "Planned").
- [ ] Point `GEMINI.md` at `AGENTS.md` (like `CLAUDE.md` does) or rewrite it to
      the real `internal/kernels` + `internal/asm` layout and current phase.
- [ ] Scope the SIMD claims in README/`goal.md`: state plainly that SIMD requires
      `-tags asm` (until P2 makes it default), or hold the claim until then.
- [ ] Fix `justfile:43` `-tags "amd"` → `-tags "asm"` and the matching
      AGENTS.md line ("amd64 uses `-tags amd`" is false).

### P0.6 Repo hygiene

- [ ] `git rm` the committed binaries (`benchkernels`,
      `cmd/bench_compare/bench_compare`, ~6 MB) and add `.gitignore` rules for
      extensionless Go binaries (e.g. `/benchkernels`, `cmd/*/bench_compare`, or a
      `/build/` convention).

---

## Priority 1 — Architecture Hardening

### P1.1 Scope the global mutable state

- [ ] Move `kernelStrategy` and `benchDecisions` out of package globals into
      `Planner`/`PlanOptions` scope so two library consumers cannot interfere.
      If a process-global default is retained for convenience, document it loudly
      and provide a reset; mirror `PlanOptions.Wisdom`'s per-instance model.
- [ ] Re-key `benchDecisions` like `WisdomKey` (size + precision + CPU features +
      direction) or drop it in favor of the richer Wisdom cache — one tuning
      cache, not two with different keying.
- [ ] Fix the 386 SSE2-complex128 wrappers that re-read the global strategy at
      execution time (`asm_amd64.go:746,761`), breaking the plan snapshot
      invariant; remove their amd64 dead duplicates.

### P1.2 Harden the Wisdom format

- [ ] Add a version/magic header; reject unknown versions instead of
      mis-parsing.
- [ ] Make `Import` atomic: parse+validate into a temp map (known algorithm
      names, supported sizes, feature-mask width) and swap in only on full
      success.
- [ ] Widen the CPU-feature mask to distinguish SSE3 from SSE2. Decide whether
      `Timestamp` drives staleness/eviction or drop it from the format.
- [ ] Increase `PlannerMeasure` iteration counts / add outlier rejection so
      recorded wisdom isn't dominated by timing noise at small sizes.

### P1.3 Zero-allocation parity across all plan types

- [ ] Make `PlanND` zero-alloc: preallocate the per-dimension slice buffer and
      precompute stride math instead of `make()` twice per slice
      (`plan_nd.go:336,440`). Model it on the 3D path.
- [ ] Hoist per-sub-transform `make()`s out of `mixedradix_avx2.go` (twiddle/
      scratch) into plan-owned buffers.
- [ ] Cache bit-reversal indices and `IsHighlyComposite`/factorization at plan
      creation, not per transform (`asm_amd64.go:56-68`, `kernels_fallback.go`).
- [ ] Extend `plan_alloc_test.go` to guard 2D/3D/ND and real variants (currently
      1D-only), so the promise is enforced everywhere it is made.

### P1.4 Reduce duplication in the plan layer

- [ ] Split `plan.go` (1,425 lines, near the 1,500 cap): extract the ~270 lines
      of triplicated `complex64/complex128/default` alloc type-switches into one
      generic aligned-alloc helper (`plan_alloc.go`), and move `Close`/`Reset`/
      `Clone` to `plan_lifecycle.go`.
- [ ] Retire `PlanReal` in favor of the generic `PlanRealT` (the former is a
      verbatim non-generic duplicate kept "for backward compatibility"), or
      generate it.
- [ ] Generate the `*128` DSP twins (`Convolve128`, `CrossCorrelate128`, …) and
      the 2D/3D/ND boilerplate from a single template.

### P1.5 Clean up dispatch

- [ ] Factor the 4×-duplicated `SelectKernels[T]` type-switch/assertion
      boilerplate (`dispatch.go`) into one helper; stop silently ignoring failed
      type assertions.
- [ ] Document the `stockham_packed_toggle_asm.go` inversion (packed Stockham is
      *disabled* exactly when `asm` is enabled) or remove it.

### P1.6 Introduce a codelet generator

- [ ] Add a `go:generate` generator that emits the size-parameterized DIT/radix
      codelets and the ~164 `Register(CodeletEntry{...})` blocks
      (`codelet_init*.go`). This is the root cause of both the duplication and the
      dead-copy class of bug; generation makes new sizes cheap and structurally
      prevents drift.
- [ ] Normalize kernel file naming to one convention (`dit_<size>_<radix>.go`),
      removing the `dit_size*` fossils from the incomplete `legacy_radix2`
      migration.
- [ ] Fix the `Lookup`/`lookupUnlocked` divergence in
      `internal/planner/codelet.go` so `GetAvailableSizes` can't advertise a size
      served only by a disabled (`Priority < 0`) codelet.

---

## Priority 2 — SIMD That Actually Ships

**Precondition: P0.1 done (asm builds and is CI-tested).**

### P2.1 Make SIMD reachable on the default build

- [ ] Decide the delivery model: either fold the asm kernels into the default
      build behind runtime CPU detection (preferred — remove the `asm` tag as a
      *build* gate and select at runtime), or keep `-tags asm` but document it as
      required for SIMD and publish guidance/benchmarks for both.
- [ ] Ensure `ForceGeneric`/fallback correctness parity is what CI benchmarks,
      not just the fast path.

### P2.2 Fix known-incorrect kernels

- [ ] AVX2 Stockham "compiles/runs but produces wrong results" (old Phase 14.4):
      diff intermediate buffers against pure-Go per stage, fix buffer-swap /
      twiddle indexing, gate behind the P0.4 forward-vs-reference sweep.
- [ ] Re-enable size-16 radix-16 on x86/386 once corrected
      (`kernels_386_asm.go:163,182` `TODO(386)`).

### P2.3 Higher-radix / larger-size kernel backlog (condensed)

Reuse-friendly decompositions, highest benefit/effort first. Each item: add the
`.s`, wire the codelet with priority, add forward-vs-reference + round-trip tests,
benchmark vs the current path with `benchstat`.

| Priority | Size  | Decomposition        | Reuses            | Status  |
| -------- | ----- | -------------------- | ----------------- | ------- |
| 1        | 4096  | 64×64 (2-stage)      | FFT-64 kernel     | planned |
| 2        | 1024  | 32×32 (2-stage)      | FFT-32 kernel     | planned |
| 3        | 256   | 16×16 (2-stage)      | radix-16 kernel   | done (Go+AVX2) |
| 4        | 16384 | 128×128 (2-stage)    | needs FFT-128     | partial |
| 5        | 8192  | 64×128 (2-stage)     | needs FFT-128     | partial |
| 6        | 512   | radix-8 / 16×32      | radix-8 infra     | done (Go) |
| 7        | 2048  | 32×64 (2-stage)      | FFT-32/64 kernels | planned |

- [ ] complex128 large-size AVX2 (512 done; 1024/2048/4096/8192/16384 pending).
- [ ] SSE2 coverage for 512 mixed-2/4 and 1024 radix-4 (both precisions) — the
      non-AVX2 fallback path.
- [ ] ARM64 NEON: sizes 512+ (evaluate benefit first), remaining complex128.

### P2.4 New instruction sets

- [ ] AVX-512 kernels (`HasAVX512` is detected today but dead).
- [ ] Revisit WASM SIMD when Go's toolchain supports it.

---

## Priority 3 — API Completeness & Polish

- [ ] Remove or implement the shipped-but-inert public options: `Radices` and
      `WorkspacePolicy`/`Workspace` (all three enum values are "not yet
      implemented"). Don't ship dead knobs in a v1.0 API.
- [ ] Fix the pooled-pool API: `NewPlanFromPool`/`NewPlanFromPoolWithOptions`
      take an `internal/fft.BufferPool` no external caller can name — re-export a
      public pool type + default, or remove these from the public surface. Align
      their length/planner contract with `newPlanWithFeatures` (currently they
      reject Bluestein sizes with a misleading `ErrNotImplemented`).
- [ ] Introspection parity: give `Plan2D/3D/ND`, `PlanReal*`, and `FastPlan` the
      `Meta()`/`KernelStrategy()`/`Algorithm()` accessors that only 1D `Plan` has;
      give `FastPlan` a `Close()`.
- [ ] Resolve the `InPlace` naming outlier (1D `InPlace` = forward-only vs
      `ForwardInPlace`/`InverseInPlace` elsewhere).
- [ ] Add plan-reuse variants for `Convolve`/`Correlate` so DSP-in-a-loop doesn't
      re-plan and re-allocate every call.
- [ ] Error-handling consistency: uniform wrapping (`plan_nd.go` adds
      dimension-indexed context via `%w`; 2D/3D return bare sentinels).
- [ ] `go doc -all` audit: verify every exported symbol has GoDoc; consider
      enabling a doc-comment linter (AGENTS requires it but nothing enforces it).

---

## Testing & CI Hardening (cross-cutting)

- [ ] **asm in CI** (see P0.1) — build + test both tags on amd64 and arm64/QEMU.
- [ ] **Coverage gate**: add `codecov.yml` with a threshold and **reconcile the
      target** — `AGENTS.md` says >90 %, `CONTRIBUTING.md` says >80 %. Pick one.
      Raise the weakest non-asm packages toward it: `internal/fft` (61.9 %),
      root (79.0 %), `internal/cpu` (78.5 %), `internal/planner` (81.6 %).
- [ ] **Make the benchmark gate real or remove it**: commit
      `benchmarks/baseline-<os>.txt` so `scripts/bench_compare.sh` actually
      compares (today it `exit 0`s on the missing baseline), or move benchmarks to
      a manual/nightly job off noisy shared runners.
- [ ] **Pin toolchain**: `test-bench.yaml` pins `go-version: 1.23` while every
      other job uses `go.mod` (1.25). Unify to `go-version-file: go.mod`. Pin
      `golangci-lint-action` to a release instead of `latest`.
- [ ] **Continuous fuzzing**: add a time-budgeted CI fuzz job (currently
      seed-corpus-only) for round-trip and no-panic properties.
- [ ] **Property-test parity**: apply Parseval/linearity/shift to the core 1D
      complex path (today unevenly spread across 2D/3D/real files).

---

## v1.0 Release — Definition of Done

v1.0 ships only when **all** of the following hold:

- [ ] `go build ./...` and `go build -tags asm ./...` both compile on amd64 and
      arm64; both are gated in CI.
- [ ] `go test -race ./...` and `go test -tags asm ./...` pass; 5× repeat run is
      flake-free.
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

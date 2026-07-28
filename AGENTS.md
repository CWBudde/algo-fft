# Repository Guidelines

## Project Overview

**algofft** (module `github.com/cwbudde/algo-fft`) is a high-performance FFT (Fast Fourier Transform) library for Go, targeting production-ready performance with SIMD acceleration, zero-allocation transforms, and support for both complex64 and complex128 precision.

**Current Status**: Early development (pre-v1.0). API may change before stable release.

## Project Structure & Module Organization

### Public API (`/`)

The root package `algofft` exposes the user-facing API, grouped roughly by file:

- **Core plans** (`plan.go` construction, `plan_transform.go` transform methods, `plan_exec*.go` per-strategy executors, `plan_options.go`): `Plan[T Complex]` with generic constructor `NewPlan[T]()` (plus `NewPlan32()`/`NewPlan64()` sugar); transform methods `Forward()`, `Inverse()`, `ForwardInPlace()`, `InverseInPlace()`, `Transform()`; each plan holds one internal `planExecutor` bound to its strategy family (use `Clone()` for concurrent transforms)
- **Real FFT** (`plan_real*.go`): real-input transforms including 2D/3D variants
- **Multi-dimensional** (`plan_2d.go`, `plan_3d.go`, `plan_nd.go`): 2D/3D/N-D transforms
- **Arbitrary lengths** (`plan_bluestein.go`): Bluestein algorithm for non-power-of-2 sizes
- **Batch & strided** (`plan_batch.go`, `plan_strided.go`): multiple transforms per call, custom layouts
- **Wisdom** (`wisdom.go`): persist and reuse plan-tuning decisions
- **DSP helpers** (`convolve.go`, `convolve_real.go`, `correlate.go`): `Convolve`, `ConvolveReal`, `Correlate`, `CrossCorrelate`, `AutoCorrelate` plus `*128` variants
- **Foundations** (`types.go`, `errors.go`, `doc.go`): `Complex`/`Float` constraints, sentinel errors, package docs

### Internal Packages (`/internal/`)

- `internal/kernels`: All FFT kernel implementations — DIT, Stockham, radix-2/3/4/5, six-step/eight-step, Bluestein, per-size codelets and their registration (`codelet_init*.go`); `types.go` defines `Kernel[T]`. The `complex128` twins of the monomorphized `*Complex64` kernels are generated (`*_c128.gen.go`, via `cmd/genkernels`) — edit only the `complex64` sources and regenerate
- `internal/planner`: Strategy selection and wisdom (`selection.go`: `ResolveKernelStrategy`, `ditAutoThreshold`; `utils.go`: the strategy↔algorithm-name table). Kernel strategy is chosen per-plan (no process-global state); tuning decisions are persisted via the Wisdom cache.
- `internal/registry`: Leaf codelet registry — `kernels` registers into it at init, `planner`/`transform` read from it
- `internal/fft`: Architecture dispatch and engine glue (`dispatch.go`: `SelectKernels[T]`; mixed-radix engine, Rader/Bluestein glue, pooling, SIMD helper dispatch). Not a re-export façade: the root imports `planner`/`kernels`/`transform`/`fftypes` directly for their own symbols.
- `internal/fftypes`: Shared types — `Complex`, `Float`, `KernelStrategy`, `SIMDLevel`, `CodeletFunc`
- `internal/cpu`: CPU feature detection (`DetectFeatures()`)
- `internal/asm`: Architecture-specific assembly under `amd64/`, `arm64/`, `x86/` with Go declaration/stub bridges
- `internal/math`: Twiddle factors, bit-reversal, factorization, transpose helpers
- `internal/memory`: SIMD-aligned buffer allocation
- `internal/transform`: Recursive decomposition, packed twiddles
- `internal/reference`: Naive O(n²) DFT (plus 2D/3D and real variants) for testing and validation

### Other Directories

- `cmd/`: Developer tools — `bench_compare`, `benchkernels`, `gencodelets`, `genkernels`, `measure_correctness`
- `examples/`: Usage examples including `wasm-demo`
- `scripts/`: Benchmark, profiling, and WASM build scripts
- `docs/`: Implementation notes (`IMPLEMENTATION_INVENTORY.md`, `PRECISION.md`, `WASM_SIMD.md`, …)

### Supporting Documentation

- `README.md`: User-facing documentation and quick start
- `PLAN.md`: Phased implementation roadmap through v1.0 — the source of truth for current status
- `BENCHMARKS.md`: Performance results
- `CHANGELOG.md`: Release notes
- `CONTRIBUTING.md`: Contribution guidelines
- `docs/goal.md`: High-level design philosophy (archived historical design doc; `PLAN.md` is the source of truth)

## Build, Test, and Development Commands

### Common Just Recipes

Use the `just` recipes defined in `justfile`:

- `just build` — compile all packages.
- `just test` — run unit tests with race detector.
- `just bench` — run benchmarks only.
- `just lint` / `just lint-fix` — run `golangci-lint` (optionally fix).
- `just fmt` — run `treefmt` (Go via `gofumpt` + `gci`, Markdown via `markdownlint` + `prettier`).
- `just fmt-check` — verify formatting without changing files.
- `just cover` — generate `coverage.html` from `coverage.txt`.
- `just check` — run test + lint + cover.
- `just fix` — run lint-fix + fmt.

### Cross-Architecture & WASM

- `just build-amd64` / `just build-arm64` / `just build-all` — cross-compile (SIMD is part of the default build).
- `just test-arm64` / `just bench-arm64` — run ARM64 tests/benchmarks via QEMU (requires `qemu-user-static`; benchmarks are correctness-only, not representative of performance).
- `just test-all` / `just check-all` — run tests/checks on amd64 and arm64.
- `just build-wasm` / `just test-wasm` / `just test-wasm-pkg <pkg>` — build and test the `js/wasm` target (tests run in Node.js).
- `just build-wasm-demo` / `just run-wasm-demo` — build and serve the WASM demo.

### SIMD, Stress, and Profiling

- `just test-purego` — run tests with the pure-Go fallback (`-tags purego`).
- `just vet-arch` — run `go vet` (asmdecl frame checks) on amd64, arm64, and 386.
- `just test-simd-verify` / `just test-arch` — verify SIMD implementations against Go fallbacks.
- `just test-stress` — long-running stress tests (30m timeout).
- `just profile-cpu` / `just profile-mem` — collect and view pprof profiles.

### Running Specific Tests

```bash
# Run a single test
go test -v -run TestName ./...

# Run tests in a specific package
go test -v ./internal/kernels

# Run benchmarks for specific sizes
go test -bench=BenchmarkPlanForward_1024 -benchmem ./...

# Run tests with verbose output
go test -v -count=1 ./...
```

## Coding Style & Naming Conventions

- Follow standard Go style; format with `gofumpt` and import ordering via `gci` (use `just fmt`).
- Use clear, descriptive names; keep functions focused and small.
- Short variable names (`i`, `j`, `k`, `n`, `w`, `x0`, …) are allowed via the `varnamelen` ignore list in `.golangci.toml`.
- Files are capped at 1500 lines (revive `file-length-limit`); split large files rather than growing them.
- Add GoDoc comments for all exported symbols.
- File naming: tests as `*_test.go`; architecture-specific files use `*_amd64.go`, `*_arm64.go`, and `.s` for assembly.
- Default to `complex64` for performance; provide `complex128` for precision-critical applications.
- Benchmarks must report allocations with `b.ReportAllocs()` and throughput with `b.SetBytes()`.

## Testing Guidelines

### Testing Strategy

The library uses multiple testing layers:

1. **Unit tests**: Verify individual components (twiddle generation, bit-reversal)
2. **Correctness tests**: Cross-validate against naive O(n²) DFT in `internal/reference`
3. **Property tests**: Verify mathematical properties (Parseval's theorem, linearity, shift theorems)
4. **Round-trip tests**: `Inverse(Forward(x)) ≈ x` for random inputs
5. **Benchmarks**: Performance regression detection

### Requirements

- Framework: Go `testing` package; tests are colocated with sources.
- Coverage target: aim for >90% on non-assembly code.
- Run tests with `just test`; coverage via `just cover`.
- Always test both `complex64` and `complex128` variants when adding features.
- Test that assembly and Go implementations produce identical results (`just test-simd-verify`).

## Development Workflow

### Adding a New Feature

1. Check `PLAN.md` for the detailed implementation roadmap
2. Read `docs/goal.md` for high-level design philosophy
3. Implement feature with tests first (TDD approach recommended)
4. Run `just lint-fix` to auto-format and fix linter issues
5. Verify `just check` passes (test + lint + coverage)
6. Update documentation in code comments and README if needed

### Performance Optimization

1. **Profile first**: Use `go test -bench -cpuprofile` to identify bottlenecks
2. **Measure baseline**: Run benchmarks before changes
3. **Optimize**: Implement SIMD, algorithmic, or cache improvements
4. **Verify correctness**: Ensure optimized path matches reference
5. **Benchmark**: Confirm speedup with `benchstat`
6. **Document**: Update comments with performance characteristics

### Before Committing

```bash
just fmt       # Format code
just lint      # Check for issues
just test      # Run all tests
just bench     # Verify no performance regressions (optional but recommended)
```

**NEVER revert uncommitted changes you didn't create** — they cannot be recovered and discarding them is data loss.

## Commit & Pull Request Guidelines

- Use Conventional Commits: `feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `docs:` prefixes.
  - Keep the summary concise (~50 chars), blank line, optional details wrapped at 72 chars.
  - Reference issues as needed (e.g., `#123`).
- CI (`.github/workflows/`) gates PRs on unit tests, lint, formatting, benchmarks, cross-architecture tests, and the WASM demo build.
- PRs should include:
  - Clear description of changes and motivation.
  - Linked issues (if any).
  - Test results; include benchmarks for performance changes.

## Architecture & Implementation Details

### Key Design Patterns

#### 1. Generic Kernel Dispatch

The library uses a type-driven dispatch system to select optimized kernels:

```
Plan[T] → internal/planner (strategy) → internal/fft.SelectKernels[T]() → internal/kernels / internal/asm
```

Kernels are selected at plan creation based on:

- CPU features (AVX2, NEON, etc.) detected via `internal/cpu.DetectFeatures()`
- Transform size and strategy (auto-selected or user-specified)
- Wisdom cache (per-instance, keyed by size + precision + CPU features) for empirically-determined best kernel

#### 2. Strategy Selection

The library supports multiple FFT algorithms via `KernelStrategy` (defined in `internal/fftypes`):

- `KernelAuto`: Automatically select based on size (DIT for ≤1024, Stockham for larger; see `ditAutoThreshold` in `internal/planner/selection.go`)
- `KernelDIT`: Force Decimation-in-Time algorithm
- `KernelStockham`: Force Stockham autosort algorithm
- `KernelSixStep` / `KernelEightStep`: Cache-oblivious large-size algorithms
- `KernelFourStep`: Rectangular six-step with the n1×n2 split chosen from detected L1d/L2 cache sizes (any power-of-two length)
- `KernelBluestein`: Arbitrary-length transforms
- `KernelRecursive`: Recursive decomposition with codelet leaves
- `KernelMixedRadix`: The mixed-radix engine (factors 2/3/5/7/11) — the route every non-power-of-two length outside Bluestein takes. Planner-resolved rather than forced: the kernel dispatch checks the length before the strategy, so a plan reports this whenever it is what runs, and forcing it at a length it is not the route for falls back to the size heuristic.

Force a strategy per-plan via `PlanOptions.Strategy` (default `KernelAuto` lets the planner choose by size). The reported `KernelStrategy()`/`Algorithm()` always name the route that executes: a forced strategy the dispatch cannot honor at that length (six-step on a non-square, four-step on a non-power-of-two, anything power-of-two on a smooth length) is resolved to the route that runs, not echoed back. There is no process-global strategy override; empirically-tuned per-size/precision choices are persisted and reused via the Wisdom cache (`PlanOptions.Wisdom`).

#### 3. Zero-Allocation Transforms

After plan creation, transforms perform zero allocations:

- Twiddle factors precomputed and stored in Plan
- Scratch buffers pre-allocated during plan creation (`NewPlan`/`NewPlan32`/`NewPlan64`)
- Bit-reversal indices precomputed
- Packed twiddle tables for SIMD kernels prepared upfront

#### 4. Type Safety via Generics

The `Complex` constraint ensures type safety:

```go
type Complex interface {
    ~complex64 | ~complex128
}
```

Generic implementations are instantiated for both precisions, with type-specific optimizations dispatched at compile time.

### When Adding New Kernels

1. **Define the kernel function** in `internal/kernels`, matching the `Kernel[T]` signature (`internal/kernels/types.go`):

   ```go
   type Kernel[T Complex] func(dst, src, twiddle, scratch []T) bool
   ```

2. **Register the kernel**:
   - Implement the `complex64` version only; the `complex128` twin is generated by `go generate ./internal/kernels/...` (`cmd/genkernels`). If the two precisions must genuinely differ, add the function to `excludedFuncs` in `cmd/genkernels/main.go` and hand-write the `complex128` side
   - Mind the compiler's stack-allocation limits (128 KiB for `var x [n]T`, 64 KiB for `make`): a stage buffer that fits as `complex64` may silently heap-allocate as `complex128` — the codelet zero-alloc sweep (`codelet_alloc_norace_test.go`) catches this
   - Return `true` if the kernel handled the transform, `false` to fall back
   - Size-specific codelets register via `codelet_init*.go` in `internal/kernels`

3. **Update dispatch**: extend `selectKernels*()` in `internal/fft/kernels_*.go` if the kernel needs architecture-gated selection

4. **Test with reference implementation**: Compare against naive DFT in `internal/reference`

5. **Benchmark**: Add benchmarks and potentially update auto-selection thresholds

### When Working with Assembly

- Assembly lives in `internal/asm/{amd64,arm64,x86}/` (`.s` files) with Go declaration/stub bridges alongside
- Use build tags for architecture-specific files: `//go:build amd64` etc.
- Always provide a pure-Go fallback (see `kernels_generic.go` / `kernels_fallback.go` in `internal/fft`)
- Test that assembly and Go implementations produce identical results (`just test-simd-verify`, `just test-purego`)
- Use `go:noescape` pragma for performance-critical functions
- Remember Plan9/Go asm uses src, dst operand order (opposite of Intel’s dst, src)
- Subtractions like VSUBPS b, a, dst → dst = a - b
- add comments after instructions for clarity
- In `avx*.s`, use VEX encodings throughout. **Never mix VEX and legacy-SSE
  vector instructions in one function** — a partially converted hot loop
  measured 152× slower than the same loop left uniformly legacy. Convert a
  function completely or not at all.
- Watch the `1/n` inverse prologue: `MOVL`/`MOVD`/`VBROADCASTSS` costs a fixed
  ~100 ns. Use `VBROADCASTSS ·const(SB), Yn` (broadcast from memory) instead.
  Because the cost is per call, not per instruction, it only shows up on small
  kernels — and it silently mis-ranks them in the codelet registry.
- Go's `MOVD AX, X0` assembles as a **64-bit** `movq` (Go's `AX` is RAX), so its
  VEX form is `VMOVQ`, not `VMOVD`. VEX writes also zero bits [255:128], which
  legacy forms preserve — only safe where the upper half is dead.
- Gate any bulk asm rewrite on a disassembly diff: `objdump -d` from binutils
  (Go's `go tool objdump` misdecodes AVX), normalized for the `v` prefix, the
  VEX merge operand, shifted addresses, and `int3` padding. It catches encoding
  bugs the reference tests pass straight over.

### Delegating Codelet Work to Subagents (Model Choice)

Model choice, from three delegation rounds (2026-07, parallel worktree agents,
NEON/arm64 verified under QEMU):

- **Haiku**: only for near-mechanical template scaling where every data table
  can be copied, not derived (its one failure across three rounds was a
  self-derived permutation table; the asm around it was correct). Spell out
  the known failure mode and the exact table source in the brief.
- **Sonnet**: default for codelet ports and new precision×size combinations,
  including cross-arch (arm64/QEMU) work and debugging another agent's
  kernel. Seven kernels across three rounds with zero asm bugs reaching a
  test run.
- **Opus**: when a performance bar must be beaten, not just correctness —
  e.g. it redesigned the addressing/butterfly form to beat SSE2 after a
  template-faithful port lost, then beat the prior best by 21–30%. Also use
  Opus (or the main session) for tuning and novel idioms.

Process notes that mattered more than model tier:

- Give each agent an isolated git worktree — they all edit the specs table
  (`cmd/gencodelets/specs.go`; NEON rows live in `specs_neon.go`) and
  regenerate `.gen.go` files, which races in a shared tree.
- Write a tight brief: exact template files to copy idioms from, reusable bitrev
  tables/scale constants (cross-file symbol reuse within `internal/asm/<arch>`;
  NEON `<>` symbols are file-scoped, so each arm64 file embeds its own copies),
  frame layout (`$0-97` for the 4-slice+bool Kernel signature), the specs
  registration steps, and the verification ladder (build → vet → full
  `internal/kernels` tests → benchmark).
- Permutation tables are precision-independent: the complex64 and complex128
  files for the same size/algorithm must contain the SAME index table. Tell
  agents to copy the twin file's table (or cross-check against
  `internal/math`'s `ComputeBitReversalIndices*` helpers) instead of deriving
  their own — a self-derived-but-wrong table was the only correctness bug to
  reach testing across rounds 1–3.
- Agents tend to background long QEMU runs and then stop to "wait" for them,
  which stalls the round; instruct them to run all verification in the
  foreground with generous timeouts.
- Have agents generate repetitive asm with a throwaway Go generator and
  validate it by byte-reproducing the existing template size first — this
  caught every would-be bug in rounds 3–4 before a single test ran. Tell
  them to use unique scratch paths (PID/random suffix): two parallel agents
  once collided on the same "obvious" generator filename in the shared
  scratchpad.
- Correctness is guarded by the registry-driven reference tests, so a cheaper
  model is safe to try: a wrong kernel fails tests rather than landing silently.
  Verify benchmark-based Priority claims yourself on an idle machine — subagent
  runs contend with each other and skew timings.

### Error Handling

Custom errors defined in `errors.go`:

- `ErrInvalidLength`: Invalid FFT size (supported: powers of two and lengths factored by 2/3/5; mixed-radix and Bluestein extend further)
- `ErrNilSlice`: Nil input/output slices
- `ErrLengthMismatch`: Slice length doesn't match plan size
- `ErrInvalidStride`: Invalid stride for the given data layout
- `ErrInvalidSpectrum`: Real-FFT spectrum violates symmetry constraints (e.g., non-real DC or Nyquist bins)
- `ErrNotImplemented`: Feature not yet implemented

Validate inputs at the Plan API boundary, not in internal kernels.

## Design Philosophy

From `docs/goal.md` and `README.md`:

1. **Pure Go**: No cgo, WebAssembly-compatible
2. **Performance**: SIMD acceleration, zero-allocation transforms
3. **Correctness**: Extensive testing, reference validation
4. **Clean API**: Hide complexity (SIMD/assembly) from users
5. **Flexibility**: Support complex64/128, arbitrary lengths (via Bluestein), real FFT
6. **Extensibility**: Pluggable kernels, architecture-specific optimizations

## Current Implementation Status

See `PLAN.md` for the roadmap and current status. In short: core transforms (DIT, Stockham, mixed-radix, Bluestein, six-step/eight-step), real FFT, 2D/3D/N-D transforms, complex128 support, convolution/correlation, and the WASM target are implemented; remaining work focuses on per-size tuning, broader SIMD coverage (SSE2, NEON), and v1.0 polish.

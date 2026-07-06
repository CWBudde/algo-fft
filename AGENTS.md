# Repository Guidelines

## Project Overview

**algofft** (module `github.com/cwbudde/algo-fft`) is a high-performance FFT (Fast Fourier Transform) library for Go, targeting production-ready performance with SIMD acceleration, zero-allocation transforms, and support for both complex64 and complex128 precision.

**Current Status**: Early development (pre-v1.0). API may change before stable release.

## Project Structure & Module Organization

### Public API (`/`)

The root package `algofft` exposes the user-facing API, grouped roughly by file:

- **Core plans** (`plan.go`, `planner.go`, `plan_options.go`, `executor.go`): `Plan[T Complex]` with constructors `NewPlanT[T]()`, `NewPlan()`, `NewPlan32()`, `NewPlan64()`; transform methods `Forward()`, `Inverse()`, `InPlace()`, `Transform()`
- **Real FFT** (`plan_real*.go`): real-input transforms including 2D/3D variants
- **Multi-dimensional** (`plan_2d.go`, `plan_3d.go`, `plan_nd.go`): 2D/3D/N-D transforms
- **Arbitrary lengths** (`plan_bluestein.go`): Bluestein algorithm for non-power-of-2 sizes
- **Batch & strided** (`plan_batch.go`, `plan_strided.go`): multiple transforms per call, custom layouts
- **Wisdom** (`wisdom.go`): persist and reuse plan-tuning decisions
- **DSP helpers** (`convolve.go`, `convolve_real.go`, `correlate.go`): `Convolve`, `ConvolveReal`, `Correlate`, `CrossCorrelate`, `AutoCorrelate` plus `*128` variants
- **Foundations** (`types.go`, `errors.go`, `doc.go`): `Complex`/`Float` constraints (aliases into `internal/fftypes`), sentinel errors, package docs

### Internal Packages (`/internal/`)

- `internal/kernels`: All FFT kernel implementations — DIT, Stockham, radix-2/3/4/5, six-step/eight-step, Bluestein, per-size codelets and their registration (`codelet_init*.go`); `types.go` defines `Kernel[T]`
- `internal/planner`: Strategy selection, wisdom, and benchmark decisions (`selection.go`: `SetKernelStrategy`, `RecordBenchmarkDecision`, `ditAutoThreshold`)
- `internal/fft`: Dispatch and re-export layer bridging the public API to kernels (`dispatch.go`: `SelectKernels[T]`)
- `internal/fftypes`: Shared types — `Complex`, `Float`, `KernelStrategy`, `SIMDLevel`
- `internal/cpu`: CPU feature detection (`DetectFeatures()`)
- `internal/asm`: Architecture-specific assembly under `amd64/`, `arm64/`, `x86/` with Go declaration/stub bridges
- `internal/math`: Twiddle factors, bit-reversal, factorization, transpose helpers
- `internal/memory`: SIMD-aligned buffer allocation
- `internal/transform`: Recursive decomposition, packed twiddles
- `internal/reference`: Naive O(n²) DFT (plus 2D/3D and real variants) for testing and validation

### Other Directories

- `cmd/`: Developer tools — `bench_compare`, `benchkernels`, `measure_correctness`
- `examples/`: Usage examples including `wasm-demo`
- `scripts/`: Benchmark, profiling, and WASM build scripts
- `docs/`: Implementation notes (`IMPLEMENTATION_INVENTORY.md`, `PRECISION.md`, `WASM_SIMD.md`, …)

### Supporting Documentation

- `README.md`: User-facing documentation and quick start
- `PLAN.md`: Phased implementation roadmap through v1.0 — the source of truth for current status
- `BENCHMARKS.md`: Performance results
- `CHANGELOG.md`: Release notes
- `CONTRIBUTING.md`: Contribution guidelines
- `goal.md`: High-level design philosophy

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

- `just build-amd64` / `just build-arm64` / `just build-all` — cross-compile (amd64 uses `-tags "asm"`).
- `just test-arm64` / `just bench-arm64` — run ARM64 tests/benchmarks via QEMU (requires `qemu-user-static`; benchmarks are correctness-only, not representative of performance).
- `just test-all` / `just check-all` — run tests/checks on amd64 and arm64.
- `just build-wasm` / `just test-wasm` / `just test-wasm-pkg <pkg>` — build and test the `js/wasm` target (tests run in Node.js).
- `just build-wasm-demo` / `just run-wasm-demo` — build and serve the WASM demo.

### SIMD, Stress, and Profiling

- `just test-asm` — run tests with the `asm` build tag.
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
2. Read `goal.md` for high-level design philosophy
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
- Benchmark cache for empirically-determined best kernel per size

#### 2. Strategy Selection

The library supports multiple FFT algorithms via `KernelStrategy` (defined in `internal/fftypes`):

- `KernelAuto`: Automatically select based on size (DIT for ≤1024, Stockham for larger; see `ditAutoThreshold` in `internal/planner/selection.go`)
- `KernelDIT`: Force Decimation-in-Time algorithm
- `KernelStockham`: Force Stockham autosort algorithm
- `KernelSixStep` / `KernelEightStep`: Cache-oblivious large-size algorithms
- `KernelBluestein`: Arbitrary-length transforms
- `KernelRecursive`: Recursive decomposition with codelet leaves

Set globally via `SetKernelStrategy()` or override per-size via `RecordBenchmarkDecision()` (both in `internal/planner/selection.go`).

#### 3. Zero-Allocation Transforms

After plan creation, transforms perform zero allocations:

- Twiddle factors precomputed and stored in Plan
- Scratch buffers pre-allocated during plan creation (`NewPlanT`/`NewPlan32`/`NewPlan64`)
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
   - Implement for both `complex64` and `complex128`
   - Return `true` if the kernel handled the transform, `false` to fall back
   - Size-specific codelets register via `codelet_init*.go` in `internal/kernels`

3. **Update dispatch**: extend `selectKernels*()` in `internal/fft/kernels_*.go` if the kernel needs architecture-gated selection

4. **Test with reference implementation**: Compare against naive DFT in `internal/reference`

5. **Benchmark**: Add benchmarks and potentially update auto-selection thresholds

### When Working with Assembly

- Assembly lives in `internal/asm/{amd64,arm64,x86}/` (`.s` files) with Go declaration/stub bridges alongside
- Use build tags for architecture-specific files: `//go:build amd64` etc.
- Always provide a pure-Go fallback (see `kernels_generic.go` / `kernels_fallback.go` in `internal/fft`)
- Test that assembly and Go implementations produce identical results (`just test-simd-verify`, `just test-asm`)
- Use `go:noescape` pragma for performance-critical functions
- Remember Plan9/Go asm uses src, dst operand order (opposite of Intel’s dst, src)
- Subtractions like VSUBPS b, a, dst → dst = a - b
- add comments after instructions for clarity

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

From `goal.md` and `README.md`:

1. **Pure Go**: No cgo, WebAssembly-compatible
2. **Performance**: SIMD acceleration, zero-allocation transforms
3. **Correctness**: Extensive testing, reference validation
4. **Clean API**: Hide complexity (SIMD/assembly) from users
5. **Flexibility**: Support complex64/128, arbitrary lengths (via Bluestein), real FFT
6. **Extensibility**: Pluggable kernels, architecture-specific optimizations

## Current Implementation Status

See `PLAN.md` for the roadmap and current status. In short: core transforms (DIT, Stockham, mixed-radix, Bluestein, six-step/eight-step), real FFT, 2D/3D/N-D transforms, complex128 support, convolution/correlation, and the WASM target are implemented; remaining work focuses on per-size tuning, broader SIMD coverage (SSE2, NEON), and v1.0 polish.

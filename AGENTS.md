# Repository Guidelines

## Project Overview

**algoforge** is a high-performance FFT (Fast Fourier Transform) library for Go, targeting production-ready performance with SIMD acceleration, zero-allocation transforms, and support for both complex64 and complex128 precision.

**Current Status**: Early development (pre-v1.0). API may change before stable release.

## Project Structure & Module Organization

**Public API (`/`)**: The root package `algoforge` exposes the user-facing API:

- `Plan[T Complex]`: Generic FFT plan supporting complex64 and complex128
- `NewPlan[T]()`, `NewPlan32()`, `NewPlan64()`: Plan constructors
- `Plan.Forward()`, `Plan.Inverse()`: Transform methods
- Error types and validation

**Internal Implementation (`/internal/fft/`)**: Core FFT algorithms and optimizations:

- **Kernel System**: Pluggable FFT kernels with runtime dispatch
- **Algorithm Implementations**:
  - **Stockham autosort**: Cache-friendly, no explicit bit-reversal (default for larger sizes)
  - **DIT (Decimation-in-Time)**: Traditional Cooley-Tukey with bit-reversal (default for smaller sizes)
- **SIMD Optimization**: Architecture-specific assembly (amd64, arm64) with pure-Go fallbacks
- **Precomputation**: Twiddle factors, bit-reversal indices, packed twiddle tables for SIMD

**Reference Implementation (`/internal/reference/`)**: O(n²) naive DFT for testing and validation

**Supporting Documentation**:

- `README.md`: User-facing documentation and quick start
- `PLAN.md`: Detailed 28-phase implementation roadmap
- `CONTRIBUTING.md`: Contribution guidelines
- `goal.md`: High-level design philosophy

## Build, Test, and Development Commands

### Common Just Recipes

Use the `just` recipes defined in `justfile`:

- `just build` — compile all packages.
- `just test` — run unit tests with race detector.
- `just bench` — run benchmarks only.
- `just lint` / `just lint-fix` — run `golangci-lint` (optionally fix).
- `just fmt` — run `treefmt` (Go via `gofumpt` + `gci`, Markdown via `prettier`).
- `just cover` — generate `coverage.html` from `coverage.txt`.
- `just check` — run test + lint + cover.

### Running Specific Tests

```bash
# Run a single test
go test -v -run TestName ./...

# Run tests in a specific package
go test -v ./internal/fft

# Run benchmarks for specific sizes
go test -bench=BenchmarkPlanForward_1024 -benchmem ./...

# Run tests with verbose output
go test -v -count=1 ./...
```

## Coding Style & Naming Conventions

- Follow standard Go style; format with `gofumpt` and import ordering via `gci` (use `just fmt`).
- Use clear, descriptive names; keep functions focused and small.
- Single letters (`i`, `j`, `k`, `n`) allowed for loop indices (configured in `.golangci.toml`).
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
- Test that assembly and Go implementations produce identical results.

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

## Commit & Pull Request Guidelines

- Commit history is minimal; follow `CONTRIBUTING.md` style:
  - Short summary (<=50 chars), blank line, optional details wrapped at 72 chars.
  - Reference issues as needed (e.g., `#123`).
- PRs should include:
  - Clear description of changes and motivation.
  - Linked issues (if any).
  - Test results; include benchmarks for performance changes.

## Architecture & Implementation Details

### Key Design Patterns

#### 1. Generic Kernel Dispatch

The library uses a type-driven dispatch system to select optimized kernels:

```
Plan[T] → SelectKernels[T]() → selectKernelsComplex64/128() → Architecture-specific kernel
```

Kernels are selected at plan creation based on:

- CPU features (AVX2, NEON, etc.) detected via `DetectFeatures()`
- Transform size and strategy (auto-selected or user-specified)
- Benchmark cache for empirically-determined best kernel per size

#### 2. Strategy Selection

The library supports multiple FFT algorithms via `KernelStrategy`:

- `KernelAuto`: Automatically select based on size (DIT for ≤1024, Stockham for larger)
- `KernelDIT`: Force Decimation-in-Time algorithm
- `KernelStockham`: Force Stockham autosort algorithm

Set globally via `SetKernelStrategy()` or override per-size via `RecordBenchmarkDecision()`.

#### 3. Zero-Allocation Transforms

After plan creation, transforms perform zero allocations:

- Twiddle factors precomputed and stored in Plan
- Scratch buffers pre-allocated during `NewPlan()`
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

### File Organization

**Root Package**:

- `plan.go`: Public Plan API and constructors
- `types.go`: Type constraints (Complex, Float)
- `errors.go`: Error definitions
- `doc.go`: Package documentation

**internal/fft**:

- `dispatch.go`: Kernel selection and type dispatch
- `selection.go`: Strategy resolution (Auto/DIT/Stockham)
- `fft.go`: Core FFT utilities (twiddle generation, bit-reversal)
- `stockham.go`: Stockham autosort implementation
- `dit.go`: Decimation-in-Time implementation
- `features.go`: CPU feature detection
- `kernels_*.go`: Architecture-specific kernel implementations
- `asm_*.go`, `kernels_*_asm.go`: Assembly optimizations

### When Adding New Kernels

1. **Define the kernel function** matching the `Kernel[T]` signature:

   ```go
   func(dst, src, twiddle, scratch []T, bitrev []int) bool
   ```

2. **Add to dispatch in `kernels_*.go`**:
   - Implement for both `complex64` and `complex128`
   - Return `true` if kernel handled the transform, `false` to fall back

3. **Update `selectKernels*()` functions** in dispatch logic to include the new kernel

4. **Test with reference implementation**: Compare against naive DFT in `internal/reference`

5. **Benchmark**: Add benchmarks and potentially update auto-selection thresholds

### When Working with Assembly

- Assembly kernels live in `kernels_*_asm.go` and `asm_*.go`
- Use build tags for architecture-specific files: `//go:build amd64` etc.
- Always provide a pure-Go fallback in `kernels_generic.go` or `kernels_fallback.go`
- Test that assembly and Go implementations produce identical results
- Use `go:noescape` pragma for performance-critical functions

### Error Handling

Custom errors defined in `errors.go`:

- `ErrInvalidLength`: Non-power-of-2 or invalid size
- `ErrNilSlice`: Nil input/output slices
- `ErrLengthMismatch`: Slice length doesn't match plan size
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

See `PLAN.md` for the complete 28-phase roadmap. Key completed phases:

- ✅ Phase 1-3: Project setup, core types, mathematical foundations
- ✅ Phase 4 (partial): Stockham autosort implementation, DIT kernels, kernel selection
- ✅ Phase 5: Testing infrastructure with naive DFT reference
- ✅ Phase 6 (partial): Basic benchmarking suite
- 🚧 Phase 4-22: Mixed-radix, real FFT, SIMD optimization, complex128 support (in progress)

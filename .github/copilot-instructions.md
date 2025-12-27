# algofft AI Coding Instructions

You are an expert Go developer working on `algofft`, a high-performance FFT library.

## Big Picture Architecture

- **Public API**: Root package (`/`) defines `Plan[T Complex]`. Users create a plan once and reuse it for zero-allocation transforms.
- **Internal Core**: `/internal/fft/` contains the actual FFT algorithms (Stockham, DIT), SIMD kernels, and dispatch logic.
- **Zero-Allocation**: All necessary memory (twiddles, scratch, bit-reversal) is pre-allocated in `Plan[T]` during `NewPlan`.
- **Type Dispatch**: The library uses Go generics with a `Complex` constraint (`complex64 | complex128`). Runtime dispatch in `internal/fft/dispatch.go` selects optimized kernels based on CPU features and precision.

## Critical Workflows

- **Formatting**: Always run `just fmt` after changes. It uses `gofumpt` and `gci` for strict formatting.
- **Testing**: Run `just test`. For new features, cross-validate against the naive DFT in `internal/reference`.
- **Benchmarking**: Use `just bench`. Performance is a primary goal; always check `b.ReportAllocs()` to ensure zero allocations in the hot path.
- **Linting**: Run `just lint` (`golangci-lint`) to ensure code quality.

## Project Patterns & Conventions

- **Generics**: Use `[T Complex]` for all FFT-related functions. Use `complexFromFloat64[T](re, im)` to create complex values generically.
- **Kernel Signature**: Kernels must match `func(dst, src, twiddle, scratch []T, bitrev []int) bool`.
- **SIMD**: Architecture-specific optimizations go in `*_amd64.go`, `*_arm64.go`, and `.s` files. Always provide a pure-Go fallback in `kernels_generic.go`.
- **Error Handling**: Validate inputs (lengths, nil checks) at the `Plan` API boundary in `plan.go`. Internal kernels assume valid inputs for performance.
- **Naming**: Use `complex64` as the default precision in examples and benchmarks unless `complex128` is specifically required.

## Common Tasks

- **Adding a Kernel**:
  1. Implement the kernel in `internal/fft/kernels_generic.go` (or arch-specific file).
  2. Register it in `internal/fft/dispatch.go` within `selectKernels*` functions.
  3. Add a test case in `internal/fft/fft_test.go` or a new test file.
- **Performance Tuning**:
  1. Run `just bench` to get a baseline.
  2. Use `go test -bench . -cpuprofile cpu.prof` to profile.
  3. Compare results using `benchstat`.

## Key Files to Reference

- `plan.go`: Public API and `Plan` struct definition.
- `internal/fft/dispatch.go`: Kernel selection and feature detection logic.
- `internal/fft/fft.go`: Core mathematical utilities (twiddles, bit-reversal).
- `internal/fft/kernels_generic.go`: Reference for implementing new FFT kernels.
- `AGENTS.md`: Detailed repository guidelines and roadmap.

# algofft Context

## Project Overview

**algofft** is a high-performance, production-ready Fast Fourier Transform (FFT) library for Go. It aims to provide a comprehensive suite of FFT algorithms with a focus on correctness, speed (via SIMD optimizations), and flexibility.

**Key Features:**

- **Core FFT:** Radix-2 DIT, Stockham autosort, Mixed-Radix (2, 3, 4, 5), and Bluestein's algorithm for arbitrary sizes.
- **Precision:** Supports both `complex64` (single) and `complex128` (double) precision via generics.
- **Performance:** SIMD acceleration (AVX2 on amd64, NEON on ARM64), zero-allocation steady state, runtime CPU feature detection.
- **Advanced:** Multi-dimensional (2D, 3D, N-D), Real-to-Complex, Batch processing, Strided access, Convolution/Correlation.

**Current Status:**

- Phase 10 of the implementation plan (`PLAN.md`) is active.
- **Recent Work:** Implemented Bluestein's algorithm to support prime and arbitrary length FFTs, effectively removing power-of-2 size restrictions.

## Architecture

- **Public API (`algofft` package):** Defines the `Plan[T]` structure and user-facing methods (`NewPlan`, `Forward`, `Inverse`, `NewPlanReal`, `NewPlan2D`, etc.).
- **Internal Implementation (`internal/fft`):** Contains the core logic, including:
  - `fft.go`: Twiddle factor computation, bit reversal.
  - `dit.go`, `stockham.go`: Specific FFT kernels.
  - `bluestein.go`: Chirp-Z transform implementation for arbitrary sizes.
  - `dispatch.go` & `kernels_*.go`: CPU feature detection and dynamic kernel dispatch.
  - `pool.go`: Buffer pooling for efficient memory usage.

## Build & Test

The project uses standard Go tooling, augmented by `just` for task management.

**Key Commands:**

- `just build`: Compile the library (`go build -v ./...`).
- `just test`: Run all tests with race detection (`go test -v -race -count=1 ./...`).
- `just bench`: Run benchmarks (`go test -bench=. -benchmem ...`).
- `just lint`: Run `golangci-lint`.
- `just fmt`: Format code using `treefmt`.

**Manual Commands (if `just` is unavailable):**

- **Test:** `go test -v ./...`
- **Test specific package:** `go test -v ./internal/fft`
- **Benchmark:** `go test -bench=. ./...`

## Development Conventions

- **Generics:** Extensive use of Go 1.18+ generics (`T Complex`) to support `complex64` and `complex128` with a single codebase.
- **Safety:** `Plan` objects are designed to be safe for concurrent use (internal scratch buffers are managed carefully or cloned).
- **Testing:**
  - Unit tests in `*_test.go` files.
  - Reference implementations in `internal/reference` for cross-validation.
  - Fuzz testing for robustness.
- **Code Style:** Standard Go fmt, vetted by `golangci-lint`.

### When Working with Assembly

- Assembly kernels live in `kernels_*_asm.go` and `asm_*.go`
- Use build tags for architecture-specific files: `//go:build amd64` etc.
- Always provide a pure-Go fallback in `kernels_generic.go` or `kernels_fallback.go`
- Test that assembly and Go implementations produce identical results
- Use `go:noescape` pragma for performance-critical functions
- We want one line per instruction and preferably with a brief comment after the instruction
- Remember Plan9/Go asm uses src, dst operand order (opposite of Intel’s dst, src)
- Subtractions like VSUBPS b, a, dst → dst = a - b
- add comments after instructions for clarity

## Key Files

- `PLAN.md`: Detailed implementation roadmap and status.
- `plan.go`: Core `Plan` struct definition and public API entry points.
- `plan_bluestein.go`: Bluestein-specific methods for the `Plan` struct.
- `internal/fft/fft.go`: Fundamental FFT math (twiddles, bit-reversal).
- `internal/fft/dispatch.go`: Kernel selection logic.
- `justfile`: Command runner configuration.

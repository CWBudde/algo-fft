# algo-fft - High-Performance Go FFT Library

[![Tests](https://github.com/MeKo-Christian/algofft/actions/workflows/test.yaml/badge.svg)](https://github.com/MeKo-Christian/algo-fft/actions/workflows/test.yaml)
[![codecov](https://codecov.io/gh/MeKo-Christian/algo-fft/branch/main/graph/badge.svg)](https://codecov.io/gh/MeKo-Christian/algo-fft)
[![Go Reference](https://pkg.go.dev/badge/github.com/MeKo-Christian/algofft.svg)](https://pkg.go.dev/github.com/MeKo-Christian/algo-fft)

A new FFT (Fast Fourier Transform) library for Go, designed for high performance, numerical accuracy, and flexibility.

## Try It Online

Experience algo-fft in your browser: **[Interactive FFT Demo](https://meko-christian.github.io/algo-fft/)**

The demo runs the library compiled to WebAssembly, allowing you to visualize FFT transforms in real-time.

## Features

- **Core FFT Algorithms**

  - Radix-2 Decimation-in-Time (DIT) FFT
  - Complex-to-complex forward and inverse transforms
  - Both in-place and out-of-place variants
  - Power-of-2 and arbitrary-length transform support via Bluestein's algorithm

- **Real FFT Support**

  - Specialized real-to-complex forward transforms
  - Complex-to-real inverse transforms
  - Optimized for real-valued signals

- **Multi-Dimensional Transforms**

  - 1D, 2D, 3D, and N-dimensional FFT support
  - Efficient row-column algorithms

- **Advanced Features**

  - Batch processing with optional parallelization
  - Strided data access for efficient matrix operations
  - Convolution and correlation via FFT
  - Both complex64 and complex128 precision

- **Performance**
  - Zero-dispatch codelets for common sizes (8, 16, 32, 64, 128)
  - SIMD acceleration (AVX2 on amd64, NEON on ARM64)
  - Zero-allocation transforms with pre-allocated Plans
  - CPU feature detection and runtime dispatch
  - Wisdom system for caching optimal planning decisions
  - Comprehensive benchmarking infrastructure

## Installation

```bash
go get github.com/MeKo-Christian/algofft
```

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/MeKo-Christian/algofft"
)

func main() {
    // Create a plan for FFT of length 8
    plan, err := algofft.NewPlan(8)
    if err != nil {
        panic(err)
    }

    // Prepare input data
    input := make([]complex64, 8)
    input[0] = 1 // impulse at index 0

    // Perform FFT
    output := make([]complex64, 8)
    err = plan.Forward(output, input)
    if err != nil {
        panic(err)
    }

    fmt.Println("FFT output:", output)
}
```

## API Overview

### Basic Transforms

```go
// Create a plan
plan, err := algofft.NewPlan(n)

// Forward FFT (out-of-place)
err = plan.Forward(dst, src)

// Inverse FFT
err = plan.Inverse(dst, src)

// In-place transforms
err = plan.ForwardInPlace(data)
err = plan.InverseInPlace(data)
```

### Real FFT

```go
// Float32 precision (single-precision)
planReal32, err := algofft.NewPlanReal32(n)  // or NewPlanReal(n) for backward compatibility
if err != nil {
    // handle error
}

input32 := make([]float32, n)
output32 := make([]complex64, n/2+1)  // Half-spectrum: N/2+1 bins
err = planReal32.Forward(output32, input32)

// Float64 precision (double-precision) - for high-precision applications
planReal64, err := algofft.NewPlanReal64(n)
if err != nil {
    // handle error
}

input64 := make([]float64, n)
output64 := make([]complex128, n/2+1)  // Half-spectrum: N/2+1 bins
err = planReal64.Forward(output64, input64)

// Generic API (type-safe)
plan, err := algofft.NewPlanRealT[float64, complex128](n)
```

The real FFT returns the non-redundant half-spectrum with length N/2+1.
For real inputs, the spectrum is conjugate-symmetric:
`X[k] = conj(X[N-k])` for `k = 1..N/2-1`.

**Precision comparison:**

- `float32` → `complex64`: ~7 decimal digits, round-trip error < 1e-6
- `float64` → `complex128`: ~15 decimal digits, round-trip error < 1e-12

### Strided Transforms

```go
// Transform a column in a row-major matrix.
cols := 256
stride := cols
col := 7
err = plan.ForwardStrided(dst[col:], src[col:], stride)
```

Strided transforms operate directly on non-contiguous data for power-of-two sizes,
which is typically faster than copying when stride is moderate. For very large
strides or cache-unfriendly layouts, explicitly copying to a contiguous buffer
can be faster.

### Batch Processing

```go
// Process multiple FFTs efficiently
plan, _ := algofft.NewPlan(1024)
count := 16
src := make([]complex64, 1024*count)
dst := make([]complex64, 1024*count)

// All FFTs stored sequentially: [FFT0, FFT1, FFT2, ...]
err := plan.ForwardBatch(dst, src, count)
```

Batch processing uses an interleaved/sequential memory layout where FFT `i` occupies `data[i*n:(i+1)*n]`. This layout is cache-friendly and maintains zero allocations during transforms.

### Wisdom System (Plan Caching)

The wisdom system caches optimal planning decisions for reuse across program runs:

```go
import "github.com/MeKo-Christian/algofft"

// Plans are automatically optimized using built-in wisdom

// Export wisdom to a file for reuse
err := algofft.ExportWisdom("fft_wisdom.txt")
if err != nil {
    // handle error
}

// Import wisdom in a future run
err = algofft.ImportWisdom("fft_wisdom.txt")
if err != nil {
    // handle error
}

// Embed wisdom in your binary
const embeddedWisdom = `64:0:3:dit64_avx2:1234567890
128:0:3:dit128_avx2:1234567890`
err = algofft.ImportWisdomFromString(embeddedWisdom)
```

The wisdom format is text-based and portable across platforms with the same CPU features. Each line contains:

- FFT size
- Precision (0=complex64, 1=complex128)
- CPU feature bitmask
- Algorithm name
- Timestamp

Benefits:

- Skip planning overhead on subsequent runs
- Consistent algorithm selection across program restarts
- Portable wisdom files for deployment

## Performance Characteristics

- **Time Complexity**: O(n log n) for power-of-2 sizes
- **Memory**: Single Plan object with pre-allocated workspace
- **Allocations**: Zero steady-state allocations during transforms
- **Codelets**: Sizes 8, 16, 32, 64, 128 use zero-dispatch codelets for maximum performance
  - Size 8: ~30 ns/op
  - Size 16: ~58 ns/op
  - Size 32: ~198 ns/op
  - Size 64: ~490 ns/op
  - Size 128: ~1028 ns/op

For detailed performance numbers, see [BENCHMARKS.md](BENCHMARKS.md).

### Performance Comparison: algofft vs gonum

The table below shows performance comparison with gonum's FFT implementation for power-of-2 sizes:

| Size | algofft  | gonum     | Speedup |
| ---- | -------- | --------- | ------- |
| 8    | 30 ns    | 516 ns    | 17.2x   |
| 16   | 75 ns    | 994 ns    | 13.3x   |
| 32   | 207 ns   | 766 ns    | 3.7x    |
| 64   | 436 ns   | 1.55 µs   | 3.6x    |
| 128  | 976 ns   | 5.73 µs   | 5.9x    |
| 256  | 2.98 µs  | 10.11 µs  | 3.4x    |
| 512  | 5.70 µs  | 17.00 µs  | 3.0x    |
| 1024 | 14.71 µs | 38.28 µs  | 2.6x    |
| 2048 | 30.68 µs | 93.93 µs  | 3.1x    |
| 4096 | 95.84 µs | 310.53 µs | 3.2x    |

_Benchmarked on Linux/amd64 with AVX2 acceleration_

## Correctness

algofft is validated against a reference O(n²) DFT implementation for mathematical correctness. The table below shows maximum relative error across 100 random test vectors per size:

| Size | complex64 Max Error | complex128 Max Error |
| ---- | ------------------- | -------------------- |
| 8    | 5.34e-07            | 2.12e-14             |
| 16   | 1.68e-06            | 1.31e-13             |
| 32   | 6.98e-06            | 5.09e-13             |
| 64   | 4.86e-06            | 4.63e-13             |
| 128  | 3.54e-05            | 1.26e-11             |
| 256  | 3.83e-05            | 8.32e-12             |
| 512  | 2.27e-05            | 8.54e-12             |
| 1024 | 2.63e-05            | 9.90e-11             |
| 2048 | 5.60e-05            | 4.49e-10             |

Errors are well within expected numerical precision limits for IEEE 754 floating-point arithmetic with accumulated rounding errors across O(n log n) operations.

## Development

### Building

```bash
just build      # Compile the library
just test       # Run all tests
just bench      # Run benchmarks
just lint       # Run linters
just fmt        # Format code
```

### Testing

The library includes comprehensive test coverage:

- Unit tests for all core algorithms
- Property-based tests (linearity, Parseval's theorem)
- Fuzz tests for robustness
- Cross-validation with reference DFT implementation

### WebAssembly

Build and run WASM tests in Node.js:

```bash
just build-wasm
just test-wasm
```

Run WASM tests for a specific package:

```bash
just test-wasm-pkg ./internal/fft
```

The WASM test runner uses a minimal Node.js environment to avoid the
argv+env size limit in `wasm_exec.js`.

WASM demo (browser):

```bash
just build-wasm-demo
python3 -m http.server 8080 --directory dist
```

See `examples/wasm-demo/README.md` for details.

For a browser smoke test, build a WASM test binary and serve it with the Go
runtime JavaScript:

```bash
GOOS=js GOARCH=wasm go test -c -o test.wasm .
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" .
cp "$(go env GOROOT)/misc/wasm/wasm_exec.html" .
python3 -m http.server 8080
```

Then open `http://localhost:8080/wasm_exec.html` and click "Run".

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to algofft.

## Supported Platforms

- Linux (amd64, arm64, 386)
- macOS (amd64, arm64)
- Windows (amd64, 386)
- WebAssembly (via GOOS=js GOARCH=wasm)

## Goals & Design

- **Correctness**: Extensive testing and mathematical precision
- **Performance**: SIMD optimization across architectures
- **Usability**: Clean, ergonomic Go API
- **Maintainability**: Well-documented, modular codebase

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Related Resources

- FFT Algorithm Overview: [Cooley-Tukey FFT](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)
- Bluestein's Algorithm: [Chirp-Z Transform](https://en.wikipedia.org/wiki/Bluestein%27s_FFT_algorithm)
- Real FFT: [Real FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform#Real_FFT)

## Status

Early development - API subject to change before v1.0 release.

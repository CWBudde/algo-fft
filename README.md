# algoforge - High-Performance Go FFT Library

A production-ready FFT (Fast Fourier Transform) library for Go, designed for high performance, numerical accuracy, and flexibility.

## Try It Online

Experience algoforge in your browser: **[Interactive FFT Demo](https://meko-christian.github.io/algoforge/)**

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
  - SIMD acceleration (AVX2 on amd64, NEON on ARM64)
  - Zero-allocation transforms with pre-allocated Plans
  - CPU feature detection and runtime dispatch
  - Comprehensive benchmarking infrastructure

## Installation

```bash
go get github.com/MeKo-Christian/algoforge
```

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/MeKo-Christian/algoforge"
)

func main() {
    // Create a plan for FFT of length 8
    plan, err := algoforge.NewPlan(8)
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
plan, err := algoforge.NewPlan(n)

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
// Real-to-complex forward transform
planReal, err := algoforge.NewPlanReal(n)
if err != nil {
    // handle error
}
err = planReal.Forward(complexDst, realSrc)

// Complex-to-real inverse transform
err = planReal.Inverse(realDst, complexSrc)
```

The real FFT returns the non-redundant half-spectrum with length N/2+1.
For real inputs, the spectrum is conjugate-symmetric:
`X[k] = conj(X[N-k])` for `k = 1..N/2-1`.

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
err = plan.ForwardBatch(dst, src, count, stride)
```

## Performance Characteristics

- **Time Complexity**: O(n log n) for power-of-2 sizes
- **Memory**: Single Plan object with pre-allocated workspace
- **Allocations**: Zero steady-state allocations during transforms

For detailed performance numbers, see [BENCHMARKS.md](BENCHMARKS.md).

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to algoforge.

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

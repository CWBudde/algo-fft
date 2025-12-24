# algoforge - High-Performance Go FFT Library

A production-ready FFT (Fast Fourier Transform) library for Go, designed for high performance, numerical accuracy, and flexibility.

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
planReal := algoforge.NewPlanReal(n)
err = planReal.Forward(complexDst, realSrc)

// Complex-to-real inverse transform
err = planReal.Inverse(realDst, complexSrc)
```

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

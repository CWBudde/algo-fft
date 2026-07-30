# algo-fft - High-Performance Go FFT Library

[![Tests](https://github.com/cwbudde/algo-fft/actions/workflows/test.yaml/badge.svg)](https://github.com/cwbudde/algo-fft/actions/workflows/test.yaml)
[![codecov](https://codecov.io/gh/cwbudde/algo-fft/branch/main/graph/badge.svg)](https://codecov.io/gh/cwbudde/algo-fft)
[![Go Reference](https://pkg.go.dev/badge/github.com/cwbudde/algo-fft.svg)](https://pkg.go.dev/github.com/cwbudde/algo-fft)

A new FFT (Fast Fourier Transform) library for Go, designed for high performance, numerical accuracy, and flexibility.

## Try It Online

Experience algo-fft in your browser: **[Interactive FFT Demo](https://cwbudde.github.io/algo-fft/)**

The demo runs the library compiled to WebAssembly, allowing you to visualize FFT transforms in real-time.

## Features

- **Core FFT Algorithms**
  - Radix-2 Decimation-in-Time (DIT) FFT
  - Complex-to-complex forward and inverse transforms
  - Both in-place and out-of-place variants
  - Power-of-2 and arbitrary-length transform support via Bluestein's algorithm
  - Rader's algorithm for prime lengths with 5-smooth n-1 (e.g. 17, 257, 65537)

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
  - SIMD acceleration (AVX-512/AVX2/SSE3/SSE2 on amd64, NEON on arm64) included in the
    default build and selected at runtime via CPU detection (`-tags purego`
    opts out to pure Go)
  - Zero-allocation transforms with pre-allocated Plans
  - CPU feature detection and runtime dispatch
  - Wisdom system for caching optimal planning decisions
  - Comprehensive benchmarking infrastructure

## Installation

```bash
go get github.com/cwbudde/algo-fft
```

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/cwbudde/algo-fft"
)

func main() {
    // Create a plan for FFT of length 8
    plan, err := algofft.NewPlan[complex64](8)
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
plan, err := algofft.NewPlan[complex64](n)

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
planReal32, err := algofft.NewPlanReal32(n)  // sugar for NewPlanReal[float32, complex64](n)
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
plan, err := algofft.NewPlanReal[float64, complex128](n)
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
plan, _ := algofft.NewPlan[complex64](1024)
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
import "github.com/cwbudde/algo-fft"

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

// Embed wisdom in your binary (the first line must be the version header)
const embeddedWisdom = `# algofft-wisdom v3
64:0:5:dit64_avx2:1234567890
128:0:5:dit128_avx2:1234567890`
err = algofft.ImportWisdomFromString(embeddedWisdom)
```

The wisdom format is text-based and portable across platforms with the same CPU features. The first line is a version header (`# algofft-wisdom v3`); files without a recognized header are rejected rather than mis-parsed. Each subsequent line contains:

- FFT size
- Precision (0=complex64, 1=complex128)
- CPU feature bitmask (bit0=SSE2, bit1=SSE3, bit2=AVX2, bit3=AVX512, bit4=NEON)
- Algorithm name
- Timestamp

An entry overrides the built-in preference order for its size, precision and CPU feature set: an algorithm field naming a codelet signature pins that codelet, and one naming a kernel strategy selects that strategy even where a codelet exists. Entries come from the measuring planner modes (`PlannerMeasure` and up), which benchmark the size's codelets alongside the kernel strategies and record whichever won.

The header moved from v2 to v3 when wisdom gained that override. The syntax is unchanged, but a v2 entry was recorded without ever being compared against the codelet it would now displace, so v2 files are rejected rather than reinterpreted — re-measure to regenerate them.

Import can also evict stale entries by age via `algofft.ImportWisdomWithMaxAge(path, maxAge)`.

Benefits:

- Skip planning overhead on subsequent runs
- Consistent algorithm selection across program restarts
- Portable wisdom files for deployment

## Performance Characteristics

- **Time Complexity**: O(n log n) for power-of-2 sizes
- **Memory**: Single Plan object with pre-allocated workspace
- **Allocations**: Zero steady-state allocations during transforms
- **Codelets**: sizes 4 … 65536 have size-specific zero-dispatch codelets. Kernel-level
  complex64 forward times on an i7-1255U (AVX2), measured with the canary-gated
  sweep described in [docs/CODELET_BENCHMARKS.md](docs/CODELET_BENCHMARKS.md):

  | Size  |   8 |   16 |  32 |  64 | 128 | 512 | 1024 | 4096 |
  | ----- | --: | ---: | --: | --: | --: | --: | ---: | ---: |
  | ns/op | 5.8 | 10.5 |  24 |  54 |  89 | 409 |  914 | 5293 |

  Add roughly 100 ns of per-call plan dispatch and validation to reach
  `Plan.Forward` timings — a fixed cost, so it dominates below ~1024 and is
  invisible above it.

For detailed performance numbers, see [BENCHMARKS.md](BENCHMARKS.md).

### Performance Comparison

Against FFTW3 and the rest of the Go field, from the cross-library harness in
[go-fft-bench](https://github.com/cwbudde/go-fft-bench) (i7-1255U, complex128,
**algofft v0.7.4**):

| n                 |    8 |   16 |   32 |   64 |   128 |   256 |   512 |  1024 |  2048 |  4096 |  8192 | 16384 | 32768 |
| ----------------- | ---: | ---: | ---: | ---: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| vs FFTW3, forward | 5.3× | 2.4× | 1.9× | 1.3× | 1.03× | 1.07× | 0.97× | 0.97× | 0.91× | 1.16× | 1.24× | 1.19× | 1.03× |

Geomean over that range is **1.36× FFTW3 forward, 1.34× inverse** — and roughly
8–15× the other Go FFT libraries (gonum, go-dsp, takatoh), which sit at
0.02–0.12× FFTW3.

Non-power-of-two lengths are the weak half at **0.60×** by geomean. Rader-routed
primes are healthy (0.78–1.58×, with outright wins), Bluestein sits at
0.42–0.62×, and composite smooth lengths with a shallow power-of-two part are
the worst case (44100 at 0.25×). Closing that is the main open work; see
[PLAN.md](PLAN.md) §5.

_Quote these figures with the tag. They are measured in a separate repository
against a pinned release, so they lag the tip of this one._

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
- **Performance**: SIMD optimization across architectures, on by default with runtime CPU detection
- **Usability**: Clean, ergonomic Go API
- **Maintainability**: Well-documented, modular codebase

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Related Resources

- FFT Algorithm Overview: [Cooley-Tukey FFT](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)
- Bluestein's Algorithm: [Chirp-Z Transform](https://en.wikipedia.org/wiki/Bluestein%27s_FFT_algorithm)
- Rader's Algorithm: [Rader's FFT Algorithm](https://en.wikipedia.org/wiki/Rader%27s_FFT_algorithm)
- Real FFT: [Real FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform#Real_FFT)

## Status

Pre-v1.0, but the API is settled: the v1.0 engineering work is complete and
nothing in the remaining backlog changes a signature. What the tag waits on is
performance and coverage work — chiefly the non-power-of-two gap above. See
[PLAN.md](PLAN.md) for the roadmap and current status.

# Command-Line Tools

This directory contains standalone tools for benchmarking and testing. Each tool has its own `go.mod` to keep dependencies isolated from the main module.

## Tools

### bench_compare

Compares algofft FFT performance against gonum's implementation.

```bash
go run ./cmd/bench_compare/main.go
```

**Dependencies**: Uses a separate module with gonum.org/v1/gonum to avoid polluting the main module.

### measure_correctness

Measures maximum relative error vs reference DFT implementation across multiple random test vectors.

```bash
go run ./cmd/measure_correctness/main.go
```

**Output**: Shows max relative error for both complex64 and complex128 across various FFT sizes.

## Why Separate Modules?

These tools use their own `go.mod` files with `replace` directives to:

- Keep the main module dependency-free from benchmarking libraries
- Allow independent versioning of tools
- Maintain clean production dependencies

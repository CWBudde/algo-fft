# Command-Line Tools

This directory contains standalone developer tools for benchmarking, code
generation, and testing. Most are part of the main module; only `bench_compare`
keeps its own `go.mod` (see [Why a Separate Module?](#why-a-separate-module)).

## Tools

### bench_compare

Compares algofft FFT performance against gonum's implementation.

```bash
cd cmd/bench_compare && go run .
```

**Dependencies**: Uses a separate module with `gonum.org/v1/gonum` to avoid
polluting the main module.

### measure_correctness

Measures maximum relative error vs the reference DFT implementation across
multiple random test vectors.

```bash
go run ./cmd/measure_correctness
```

**Output**: Shows max relative error for both complex64 and complex128 across
various FFT sizes.

### benchkernels

Micro-benchmarks the kernel strategies for a set of sizes and can export the
resulting best-per-size choices as a portable wisdom file.

```bash
go run ./cmd/benchkernels -sizes 1024,4096,16384 -mode all
go run ./cmd/benchkernels -sizes 1024,4096 -wisdom wisdom.json
```

**Flags**: `-sizes` (comma-separated), `-mode` (`forward`/`inverse`/`roundtrip`/`all`),
`-iters`, `-warmup`, `-wisdom` (export path), `-seed`.

### gencodelets

Generates the built-in codelet registration functions in `internal/kernels`
(one file per build target) from the declarative table in `specs.go`. With
`-inventory`, it renders `docs/IMPLEMENTATION_INVENTORY.md` from the same table
instead.

```bash
go generate ./internal/kernels/...
```

### genkernels

Generates the `complex128` kernel twins (`*_c128.gen.go`) in `internal/kernels`
from their hand-written `complex64` sources. Edit the `complex64` files only,
then regenerate. Deliberately-different twins are hand-written and listed in
`excludedFuncs` in `cmd/genkernels/main.go`.

```bash
go generate ./internal/kernels/...
```

## Why a Separate Module?

Only `bench_compare` keeps its own `go.mod` (with a `replace` directive back to
the repo root). This isolates its `gonum.org/v1/gonum` benchmarking dependency
so it never enters the main module's dependency graph. The other tools
(`measure_correctness`, `benchkernels`, `gencodelets`, `genkernels`) depend only
on the library and its `internal/` packages, so they live in the main module.

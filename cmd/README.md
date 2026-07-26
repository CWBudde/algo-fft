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

Measures FFT accuracy against a float64 naive DFT, in both precisions, over many
random vectors per size. Reports relative L2 error (`||got-want||₂ / ||want||₂`)
as a mean and a max over trials, plus a peak-normalized max-per-bin error
(`max|got-want| / max|want|`).

The complex64 reference is `reference.NaiveDFTWide`, a float64 DFT of the same
float32-rounded input vector, so the reported error is the transform's own and
does not fold in input quantization. The complex128 column is compared against
`reference.NaiveDFT128`, which is also float64 — above n ≈ 64 the reference's own
accumulation dominates, so read that column as a fixed-reference regression
tripwire rather than as a measurement of the FFT's error.

Both metrics normalize by a whole-vector quantity. Neither is the per-bin
`|got-want| / |want|` this tool reported before: that divided each bin by its own
magnitude, so it was decided by whichever bin landed nearest a zero and swung 3×
on changes that moved the true error by 8%. Numbers from the old metric run 2–3
orders of magnitude higher and are not comparable.

```bash
go run ./cmd/measure_correctness
go run ./cmd/measure_correctness -sizes 257,1009,2205 -trials 20 -seed 7
```

**Flags**: `-sizes` (comma-separated, any length the planner accepts), `-trials`,
`-seed` (re-applied per size, so a single `-sizes` value reproduces its row from a
full run).

**Output**: One block per precision, one row per size, with `relL2 mean`,
`relL2 max` and `peak max` columns, headed by the build configuration
(`arch`/`simd`/`purego`) — the same tree reports different complex64 numbers on
the default and `purego` builds because different codelets run, so a pasted table
is not interpretable without it. Runtime is dominated by the O(n²) reference; use
`-trials` to keep large sizes tractable.

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

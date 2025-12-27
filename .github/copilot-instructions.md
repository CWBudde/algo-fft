# algo-fft (algofft) AI Coding Instructions

## Orientation (where things live)

- **Public API (root package `algofft/`)**: `Plan[T Complex]`, options, executors, N-D/real FFTs, convolution/correlation.
  - Start here: `plan.go`, `plan_options.go`, `plan_real*.go`, `plan_nd.go`, `convolve*.go`, `correlate.go`.
- **Core engine (`internal/fft/`)**: DIT/Stockham/mixed-radix/split-radix FFTs, Bluestein (arbitrary lengths), six/eight-step, twiddles/transpose/strided.
  - Entry points: `internal/fft/dit.go`, `internal/fft/stockham.go`, `internal/fft/bluestein.go`, `internal/fft/fft.go`.
- **CPU/SIMD + asm**: feature detection in `internal/cpu/`; arch dispatch + wrappers in `internal/fft/kernels_*` and `internal/fft/asm_*` / `.s`.
  - Size-specific code paths often live in `internal/fft/dit_size*.go` and are bound via the codelet registry.

## Developer workflow (what to run)

- Format with `just fmt` (treefmt: Go via gofumpt+gci; Markdown via prettier).
- Test with `just test` (race enabled). For portability/regressions: `just test-wasm`, `just test-arm64` (QEMU correctness only).
- Benchmark with `just bench` and keep steady-state transforms allocation-free.

## Planning/dispatch model (critical for changes)

- Plan creation precomputes `twiddle`, `bitrev`, and aligned scratch so transforms are zero-allocation.
- Planning order: **codelet registry → wisdom cache → heuristic strategy** (`internal/fft/planner.go`, `internal/fft/selection.go`).
- Strategy selection: DIT/Stockham threshold, six/eight-step for large square sizes, Bluestein for lengths that aren’t power-of-two and not “highly composite”.

## Conventions for kernels/codelets

- Validate at the API boundary (e.g., `Plan.Forward`/`Inverse` in `plan.go`); internal kernels assume validated inputs for performance.
- Kernel signature (fallback path): `func(dst, src, twiddle, scratch []T, bitrev []int) bool` (`internal/fft/dispatch.go`).
- Codelets are fixed-size, no-check fast paths: `type CodeletFunc[T] func(dst, src, twiddle, scratch []T, bitrev []int)`.
  - Register/choose codelets via `internal/fft/codelet*.go` + `internal/fft/codelet_init.go` using signatures like `dit512_avx2`.
- If adding SIMD/asm, keep a correct pure-Go fallback (`internal/fft/kernels_generic.go` / `internal/fft/kernels_fallback.go`).

## Tests to lean on

- Correctness vs O(n²) reference: `internal/reference/` and tests like `plan_reference_test.go` and `internal/fft/*_test.go`.
- SIMD equivalence: `simd_verify_test.go` and `just test-simd-verify`.

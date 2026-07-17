# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Bluestein plans run their padded sub-FFT through the size-dispatched DIT
  kernels (radix-4 and size-specific codelets, SIMD where available) instead
  of the generic radix-2 path: prime-size transforms measure 25–64% faster
  (geomean −39%) on the default build and 1.2–1.4× faster on `purego`
- Bluestein pad sizes are chosen per size at plan time via a cost model that
  can also select 5-smooth (2^a·3^b·5^c) padded lengths executed by the
  mixed-radix engine; with the current kernels the measured crossover always
  favors the next power of two, so behavior is unchanged (see PLAN.md P4.1)

### Added

- Rader's algorithm for prime-size transforms: primes whose p−1 is 5-smooth
  and passes a measured cost gate (e.g. 17, 257, 401, 641, 1601, 4001,
  12289, 40961, 65537) now run an exact length-(p−1) cyclic convolution
  instead of Bluestein's power-of-two pad to ≥ 2p−1, measuring 1.3–5.2×
  faster on both precisions (still zero-alloc); other primes keep Bluestein,
  and forcing `PlanOptions.Strategy = KernelBluestein` opts out
- Plan-reuse DSP types `Convolver`, `Correlator`, and `RealConvolver`:
  reusable, concurrency-safe, zero-allocation convolution/correlation for
  loops (the one-shot `Convolve`/`CrossCorrelate`/`ConvolveReal` helpers
  re-plan on every call)
- Introspection parity: `Plan2D`/`Plan3D`/`PlanND`/`PlanReal2D`/`PlanReal3D`
  expose `Meta()` and per-axis `KernelStrategies()`/`Algorithms()`;
  `PlanRealT`, `FastPlan`, and `FastPlanReal32/64` expose
  `Meta()`/`KernelStrategy()`/`Algorithm()`; `FastPlan` and
  `FastPlanReal32/64` gained `Close()`
- `Plan.ForwardInPlace` and `FastPlan.ForwardInPlace`, matching the
  multi-dimensional plans' `ForwardInPlace`/`InverseInPlace` naming
- Core FFT implementation: DIT, Stockham, radix-2/3/4/5, mixed-radix,
  six-step/eight-step algorithms with per-size codelets
- Bluestein's algorithm for arbitrary-length transforms
- Real FFT support (1D, 2D, 3D) for float32 and float64 input
- Multi-dimensional transforms (2D, 3D, N-D)
- Batch and strided transform APIs
- Convolution and correlation helpers (complex and real, both precisions)
- complex64 and complex128 precision throughout
- SIMD kernels (AVX-512/AVX2/SSE2/SSE3 on amd64, NEON on arm64) in the
  default build, selected at runtime via CPU detection (`purego` build tag
  opts out); on AVX-512 CPUs the generic AVX-512 kernel also serves as the
  complex64 codelet at sizes 1024/4096/8192/16384 (1.2–2.4× over the AVX2
  codelets it replaces)
- Wisdom: persist and reuse plan-tuning decisions
- WebAssembly (js/wasm) target support
- Concurrency-safe plans: a single plan instance may run transforms from
  multiple goroutines
- Comprehensive testing infrastructure: reference-DFT cross-validation,
  property tests, round-trip, fuzz, stress, race, and zero-allocation guards
- Performance benchmarking suite
- Development tooling (justfile, golangci-lint, treefmt, pre-commit hooks)
- GitHub Actions CI/CD workflow (multi-OS, multi-arch, WASM)
- Project documentation (README, CONTRIBUTING, CHANGELOG)

### Changed

- Multi-dimensional plan constructors now wrap `ErrInvalidLength` and child
  plan failures with dimension context (matching `PlanND`); match errors with
  `errors.Is`
- `NewPlanPooled`/`NewPlanPooledWithOptions` accept the same lengths and
  planner options as `NewPlanT`: Bluestein sizes are served by the regular
  allocator instead of being rejected, and `PlanOptions.Planner` measure
  modes are honored

### Deprecated

- `Plan.InPlace` and `FastPlan.InPlace` (forward-only): use `ForwardInPlace`

### Removed

- Inert `PlanOptions.Radices` and `PlanOptions.Workspace` options and the
  `WorkspacePolicy` type (all were documented "not yet implemented" and never
  read)
- `NewPlanFromPool`/`NewPlanFromPoolWithOptions` (took an internal pool type
  no external caller could name; use `NewPlanPooled`/`NewPlanPooledWithOptions`)

### Planned

- Higher-radix / per-size-tuned AVX-512 kernel variants
- Broader SSE2/NEON size coverage

## [0.0.1] - 2025-12-24

### Added

- Project initialization
- Basic project structure

---

## Notes

For breaking changes, feature requests, or bug reports, please open an issue on GitHub.

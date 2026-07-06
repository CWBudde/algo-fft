# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Core FFT implementation: DIT, Stockham, radix-2/3/4/5, mixed-radix,
  six-step/eight-step algorithms with per-size codelets
- Bluestein's algorithm for arbitrary-length transforms
- Real FFT support (1D, 2D, 3D) for float32 and float64 input
- Multi-dimensional transforms (2D, 3D, N-D)
- Batch and strided transform APIs
- Convolution and correlation helpers (complex and real, both precisions)
- complex64 and complex128 precision throughout
- SIMD kernels (AVX2/SSE2/SSE3 on amd64, NEON on arm64), currently behind
  the `asm` build tag; default builds use pure Go
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

### Planned

- SIMD enabled on default builds via runtime CPU detection (removing the
  `asm` build-tag gate)
- AVX-512 kernels
- Broader SSE2/NEON size coverage

## [0.0.1] - 2025-12-24

### Added

- Project initialization
- Basic project structure

---

## Notes

For breaking changes, feature requests, or bug reports, please open an issue on GitHub.

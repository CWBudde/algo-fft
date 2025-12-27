# PLAN.md - algofft Implementation Roadmap

## Completed (Summary)

✅ **Phases 1-13**: Project setup, types, API, errors, twiddles, bit-reversal, radix-2/3/4/5 FFT, Stockham autosort, DIT, mixed-radix, Bluestein, six-step/eight-step large FFT, SIMD infrastructure, CPU detection
✅ **Real FFT**: Forward/inverse, generic float32/float64, 2D/3D real FFT with compact/full output
✅ **Multi-dimensional**: 2D, 3D, N-D FFT with generic support
✅ **Testing**: Reference DFT, correctness, property-based, fuzz, precision, stress, concurrent tests
✅ **Benchmarking**: Full suite with regression tracking, BENCHMARKS.md
✅ **Batch/Strided**: Sequential batch API, strided transforms
✅ **Convolution**: FFT-based convolution/correlation for complex and real
✅ **complex128**: Full generic API with explicit constructors
✅ **WebAssembly**: Build, test (Node.js), examples
✅ **Cross-arch CI**: amd64, arm64, 386, WASM matrix

---

## Phase 14: AVX2 SIMD (AMD64) - Remaining Work

### 14.5 Size-Specific Fully Unrolled AVX2 Kernels

**Status**: Dispatch mechanism complete (14.5.1 ✅), Size-16 kernel implemented (14.5.2 ✅), Size-32 kernel implemented (14.5.3 ✅)

**Motivation**: Combine size-specific unrolling with SIMD for maximum performance on critical sizes (16, 32, 64, 128).

#### 14.5.2 Implement AVX2 Size-16 kernel (complex64) ✅

**File**: `internal/fft/asm_amd64.s`

- [x] Create `forwardAVX2Size16Complex64Asm`
- [x] Fully unroll 4 FFT stages (size=2, 4, 8, 16)
- [x] Hardcode bit-reversal indices (no loop): `[0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]`
- [x] Hardcode twiddle indices for each stage
- [x] Vectorize butterflies using AVX2 (4 complex64 per YMM register)
- [x] Test correctness vs reference and generic AVX2
- [x] Benchmark speedup: **2.1x faster** than pure Go, **88% faster** than generic AVX2 (25ns vs 48ns vs 53ns @ i7-1255U)

#### 14.5.3 Implement AVX2 Size-32 kernel (complex64) ✅

**File**: `internal/fft/asm_amd64.s`

- [x] Create `forwardAVX2Size32Complex64Asm`
- [x] Fully unroll 5 FFT stages (size=2, 4, 8, 16, 32)
- [x] Hardcode bit-reversal indices: `[0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31]`
- [x] Optimize for L1 cache locality (all data fits in 256 bytes)
- [x] Test correctness vs reference and generic AVX2
- [x] Benchmark speedup: **2.5x faster** than generic AVX2 (70ns vs 180ns @ i7-1255U)

#### 14.5.4 Implement AVX2 Size-64 kernel (complex64)

- [ ] Create `forwardAVX2Size64Complex64Asm`
- [ ] Fully unroll 6 FFT stages
- [ ] Consider radix-4 decomposition to limit code size
- [ ] Test and benchmark

#### 14.5.5 Implement AVX2 Size-128 kernel (complex64)

- [ ] Create `forwardAVX2Size128Complex64Asm`
- [ ] Fully unroll 7 FFT stages or use radix-4
- [ ] Balance code size (~500-1000 lines) vs performance
- [ ] Test and benchmark

#### 14.5.6 Add inverse transform variants

- [ ] Implement inverse versions for sizes 16, 32, 64, 128
- [ ] Conjugate twiddles and add 1/n scaling
- [ ] Test round-trip accuracy

#### 14.5.7 Add complex128 size-specific kernels (optional)

- [ ] Implement for sizes 16, 32 (higher priority, smaller code)
- [ ] AVX2 processes 2 complex128, so expect ~2x speedup vs pure-Go
- [ ] Test and benchmark

**Success Criteria**:

- 5-20% speedup over generic AVX2 for sizes 16-128
- All tests pass, code size <1000 lines per kernel

---

### 14.6 Fix Stockham AVX2 Correctness

**Status**: Compiles ✅, segfault fixed ✅, **produces wrong results** ⚠️

**Location**: `internal/fft/asm_amd64.s` lines 789-854 (forward), ~1625-1690 (inverse)

#### 14.6.2 Fix Stockham runtime correctness

- [ ] Debug why Stockham transforms differ from pure-Go
  - [ ] Add debug logging to identify which stage diverges
  - [ ] Compare intermediate buffer states step by step
  - [ ] Check buffer swap logic (dst ↔ scratch)
- [ ] Run full test suite with `-tags=fft_asm`
- [ ] Benchmark Stockham vs DIT with AVX2

**Debugging approach**:

```bash
# Build with fft_asm tag
go test -tags=fft_asm -v -run TestStockham ./internal/fft/

# Compare output
go test -tags=fft_asm -v -run TestAVX2MatchesPureGo ./internal/fft/
```

**Priority**: HIGH (blocks accurate Stockham benchmarking)

---

## Phase 15: ARM64 NEON - Remaining Work

### 15.4 Production Testing on Real Hardware

**Status**: QEMU testing complete ✅, real hardware pending

- [ ] **Test on physical ARM64 device**
  - [ ] Raspberry Pi 4/5
  - [ ] AWS Graviton (t4g.micro free tier)
  - [ ] Apple Silicon Mac (if available)

- [ ] **Benchmark actual performance**

  ```bash
  # On ARM64 hardware:
  just bench | tee benchmarks/arm64_native.txt
  ```

  - [ ] Compare QEMU vs native performance
  - [ ] Document realistic speedup numbers

- [ ] **Add ARM64 to CI**
  - [ ] Set up GitHub Actions ARM64 runner (or use `runs-on: macos-14`)
  - [ ] Ensure cross-architecture tests pass
  - [ ] Verify SIMD paths produce same results as pure-Go

- [ ] **Update documentation**
  - [ ] Add ARM64 section to BENCHMARKS.md
  - [ ] Document NEON performance characteristics
  - [ ] Compare NEON vs AVX2 speedup ratios

---

### 15.5 Size-Specific Unrolled NEON Kernels

**Prerequisite**: 15.4 (need real hardware for meaningful benchmarks)

**Approach**: Mirror AVX2 Phase 14.5 for ARM64

#### 15.5.1 Design size-specific dispatch ✅

- [x] Add dispatch layer in `kernels_arm64_asm.go` routing by size
- [x] Define function signatures for size-specific kernels in `asm_arm64.go`
- [x] Create fallback chain: size-specific NEON → generic NEON → pure-Go
- [x] Add benchmarks in `kernels_arm64_size_specific_bench_test.go`

#### 15.5.2 Implement NEON Size-16 kernel (complex64) ✅

**File**: `internal/fft/asm_arm64.s`

- [x] Create `forwardNEONSize16Complex64Asm`
- [x] Fully unroll 4 FFT stages (size=2, 4, 8, 16)
- [x] Hardcode bit-reversal indices: `[0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]`
- [x] Hardcode twiddle indices for each stage
- [x] Vectorize butterflies using NEON (2 complex64 per 128-bit register)
- [x] Test correctness vs reference and generic NEON - PASS

**Implementation notes**:

- Go ARM64 assembler lacks VFADD/VFSUB; used VFMLA/VFMLS with ones vector
- QEMU benchmarks not representative; real hardware needed (see 15.4)
- All correctness tests pass under QEMU emulation

#### 15.5.3-5 Implement NEON kernels for sizes 16, 32, 64, 128

For each size:

- [ ] Create `forwardNEONSize{N}Complex64Asm` in `asm_arm64.s`
- [ ] Fully unroll all FFT stages
- [ ] Hardcode bit-reversal and twiddle indices
- [ ] Vectorize using NEON (2 complex64 per 128-bit register)
- [ ] Test and benchmark

**Architecture notes**:

- NEON 128-bit (2 complex64) vs AVX2 256-bit (4 complex64)
- Expect different unrolling patterns due to register width

#### 15.5.6 Inverse transforms

- [ ] Implement inverse versions for sizes 16, 32, 64, 128
- [ ] Conjugate twiddles, add 1/n scaling

#### 15.5.7 complex128 kernels (optional)

- [ ] Implement for sizes 16, 32
- [ ] NEON processes 1 complex128 per register

---

## Phase 16: Cache & Loop Optimization

### 16.1 Cache Optimization

- [ ] **Profile cache behavior**
  ```bash
  perf stat -e cache-references,cache-misses go test -bench=BenchmarkPlan -benchtime=5s
  ```
- [ ] Implement cache-oblivious or cache-aware strategies
- [ ] Optimize memory access patterns in butterfly loops
- [ ] Test performance impact

### 16.2 Loop Unrolling

- [ ] Identify critical inner loops via profiling
- [ ] Manually unroll by 2x or 4x
- [ ] Benchmark unrolled vs original
- [ ] Balance code size vs performance

### 16.3 Bounds Check Elimination

- [ ] Profile to find bounds check hotspots:
  ```bash
  go build -gcflags="-d=ssa/check_bce/debug=1" ./...
  ```
- [ ] Restructure loops to eliminate checks
- [ ] Use `_ = slice[len-1]` pattern where needed
- [ ] Verify no safety regressions

---

## Phase 19: Batch Processing - Remaining

### 19.3 Parallel Batch Processing

- [ ] **Implement `Plan.ForwardBatchParallel`**

  ```go
  func (p *Plan[T]) ForwardBatchParallel(dst, src []T, count int) error
  ```

  - [ ] Use `sync.WaitGroup` and goroutines
  - [ ] Implement worker pool for large batch counts
  - [ ] Ensure Plan is safe for concurrent read-only use

- [ ] **Tune parallelism**
  - [ ] Add `GOMAXPROCS` awareness
  - [ ] Find optimal batch-per-goroutine threshold
  - [ ] Benchmark parallel speedup for batch sizes 4, 16, 64, 256

---

## Phase 22: complex128 - Remaining

### 22.3 Precision Profiling

- [ ] **Profile precision differences**
  - [ ] Measure error accumulation for FFT sizes 64 → 65536
  - [ ] Compare complex64 vs complex128 round-trip error
  - [ ] Document precision guarantees in PRECISION.md

---

## Phase 23: WebAssembly - Remaining

### 23.1 Browser Testing

- [ ] **Test in browser environment**
  - [ ] Create test HTML page with wasm_exec.js
  - [ ] Run FFT in browser, verify results
  - [ ] Document browser-specific considerations

### 23.2 WASM SIMD (experimental)

- [ ] **Prototype WASM SIMD butterfly** (if Go 1.24+ adds support)
- [ ] **Benchmark WASM vs native**

  ```bash
  # Native
  go test -bench=BenchmarkPlan -benchtime=5s

  # WASM via Node
  GOOS=js GOARCH=wasm go test -bench=BenchmarkPlan -benchtime=5s
  ```

---

## Phase 24: Documentation & Examples

### 24.1 GoDoc Completion

- [ ] Audit all exported symbols for GoDoc comments
- [ ] Add runnable examples in `example_test.go`:
  - [ ] `ExampleNewPlan`
  - [ ] `ExamplePlan_Forward`
  - [ ] `ExampleNewPlan2D`
  - [ ] `ExampleConvolve`
- [ ] Document all error conditions
- [ ] Add package-level overview in `doc.go`

### 24.2 README Enhancement

- [ ] **Installation instructions**
  ```bash
  go get github.com/MeKo-Tech/algo-fft
  ```
- [ ] **Quick start examples** (copy-paste ready)
- [ ] **API overview table**
- [ ] **Performance characteristics** (link to BENCHMARKS.md)
- [ ] **Comparison to other libraries** (gonum, go-fft)
- [ ] **Badges**: Go Report Card, CI status, coverage, pkg.go.dev

### 24.3 Tutorial Examples

Create these directories with working code + README:

- [ ] `examples/basic/` - simple 1D FFT usage
- [ ] `examples/audio/` - audio spectrum analysis with real FFT
- [ ] `examples/image/` - 2D FFT for image processing
- [ ] `examples/benchmark/` - performance comparison tool

---

## Phase 26: Profiling & Tuning

### 26.1 Comprehensive Profiling

- [ ] **CPU profiling**
  ```bash
  go test -bench=BenchmarkPlan_1024 -cpuprofile=cpu.prof
  go tool pprof -http=:8080 cpu.prof
  ```
- [ ] **Memory profiling**
  ```bash
  go test -bench=BenchmarkPlan_1024 -memprofile=mem.prof
  ```
- [ ] Identify remaining optimization opportunities
- [ ] Document profiling results

### 26.2 Auto-Tuning (optional)

- [ ] Research auto-tuning approaches (FFTW-style planner)
- [ ] Prototype runtime algorithm selection based on size
- [ ] Evaluate complexity vs benefit

### 26.3 Final Optimization Pass

- [ ] Address remaining performance hotspots
- [ ] Ensure consistent performance across sizes
- [ ] Final benchmark comparison to goals
- [ ] Update BENCHMARKS.md

---

## Phase 27: v1.0 Preparation

### 27.1 API Review

- [ ] Review all public APIs for consistency
- [ ] Ensure backward compatibility patterns
- [ ] Document any breaking changes from pre-release
- [ ] Create migration guide if needed

### 27.2 Stability Testing

- [ ] Run full test suite 10+ times (check for flakes)
- [ ] Test on Go versions: 1.21, 1.22, 1.23, 1.24
- [ ] Verify all CI checks pass
- [ ] Address any flaky tests

### 27.3 Release Preparation

- [ ] Update CHANGELOG.md with all changes since v0.1.0
- [ ] Write v1.0.0 release notes
- [ ] Tag release: `git tag v1.0.0`
- [ ] Push tag: `git push origin v1.0.0`
- [ ] Verify on pkg.go.dev

---

## Phase 28: Community & Maintenance

### 28.1 Community Setup

- [ ] Create `.github/ISSUE_TEMPLATE/bug_report.md`
- [ ] Create `.github/ISSUE_TEMPLATE/feature_request.md`
- [ ] Create `.github/PULL_REQUEST_TEMPLATE.md`
- [ ] Set up GitHub Discussions for Q&A
- [ ] Add `CODE_OF_CONDUCT.md`

### 28.2 Contributor Experience

- [ ] Document development setup in CONTRIBUTING.md
- [ ] Add "good first issue" labels to starter tasks
- [ ] Set up Dependabot for dependency updates

### 28.3 Ongoing Maintenance

- [ ] Document maintenance schedule
- [ ] Set up security vulnerability monitoring (govulncheck)
- [ ] Plan for future Go version compatibility
- [ ] Create roadmap for post-1.0 features

---

## Future (Post v1.0)

- AVX-512 support (when Go supports it better)
- GPU acceleration (OpenCL/CUDA via cgo, optional)
- Distributed FFT for very large datasets
- Pruned FFT for sparse inputs/outputs
- DCT (Discrete Cosine Transform)
- Hilbert transform
- Short-time FFT (STFT) for spectrograms
- Gonum ecosystem integration

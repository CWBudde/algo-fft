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

## Phase 14: FFT Size Optimizations (Target: Sizes 4-16384)

**Status**: In Progress
**Priority**: complex64 first, then complex128
**Target**: Blazingly fast FFTs for sizes up to 16K

---

### Current Implementation Status

Based on `docs/IMPLEMENTATION_INVENTORY.md` (updated 2026-01-01):

| Size  | Go complex64 | AVX2 complex64 | Go complex128 | AVX2 complex128 |
| ----- | ------------ | -------------- | ------------- | --------------- |
| 4     | ✅ radix-4   | ✅ radix-4     | ✅ radix-4    | ✅ radix-4      |
| 8     | ✅ r2/r8/mix | ✅ r2/r8/mix¹  | ✅ r2/r8/mix  | ✅ r2/mix       |
| 16    | ✅ r2/r4     | ✅ r2/r4       | ✅ r2/r4      | ✅ r2/r4        |
| 32    | ✅ r2/mix24  | ✅ r2/mix24    | ✅ r2/mix24   | ✅ r2/mix24     |
| 64    | ✅ r2/r4     | ✅ r2/r4       | ✅ r2/r4      | ✅ r2/r4        |
| 128   | ✅ r2/mix24  | ✅ r2/mix24    | ✅ r2/mix24   | ✅ r2/mix24     |
| 256   | ✅ r2/r4     | ✅ r2/r4       | ✅ r2/r4      | ✅ radix-2      |
| 512   | ✅ mix24     | ❌ generic     | ✅ mix24      | ❌              |
| 1024  | ✅ radix-4   | ❌ generic     | ✅ radix-4    | ❌              |
| 2048  | ✅ mix24     | ❌ generic     | ✅ mix24      | ❌              |
| 4096  | ✅ radix-4   | ❌ generic     | ✅ radix-4    | ❌              |
| 8192  | ✅ mix24     | ❌ generic     | ✅ mix24      | ❌              |
| 16384 | ✅ radix-4   | ❌ generic     | ✅ radix-4    | ❌              |

**Legend**: r2=radix-2, r4=radix-4, r8=radix-8, mix=mixed-radix, mix24=mixed-radix-2/4
¹ Size-8 AVX2 exists but disabled (slower than Go radix-8)

---

### 14.1 Pure Go Size-Specific Kernels (complex64)

**Goal**: Complete coverage for all sizes up to 16K with optimal algorithm choice.

#### 14.1.1 Size 512 - Mixed-Radix-2/4 Optimization ✅ COMPLETE

**Current**: radix-2 only (9 stages)
**Target**: mixed-radix-2/4 (4 r4 + 1 r2 = 5 stages, ~40% faster)
**Status**: Complete

**Implementation (2024-12-31):**

Successfully implemented mixed-radix-2/4 for size 512 = 2 × 4⁴:

1. **Algorithm structure**: 4 radix-4 stages FIRST, then 1 radix-2 stage LAST
2. **Mixed-radix bit-reversal**: Custom `ComputeBitReversalIndicesMixed24()` function
   - Interprets index as 1 binary digit (MSB) + 4 quaternary digits
   - Reverses quaternary part, keeps binary bit in LSB position
3. **Stage structure**:
   - Stage 1: 128 radix-4 butterflies (groups of 4)
   - Stage 2: 32 radix-4 groups × 4 butterflies
   - Stage 3: 8 radix-4 groups × 16 butterflies
   - Stage 4: 2 radix-4 groups × 64 butterflies
   - Stage 5: 1 radix-2 stage (combines two 256-point halves)

- [x] Create `dit_size512_mixed24.go` ✅
- [x] Create `ComputeBitReversalIndicesMixed24()` for mixed-radix ✅
- [x] Implement `forwardDIT512Mixed24Complex64()` ✅
- [x] Implement `inverseDIT512Mixed24Complex64()` ✅
- [x] Implement `forwardDIT512Mixed24Complex128()` ✅
- [x] Implement `inverseDIT512Mixed24Complex128()` ✅
- [x] Tests passing ✅

#### 14.1.2 Size 1024 - Pure Radix-4 ✅ COMPLETE

**Current**: generic Stockham fallback
**Target**: size-specific radix-4 (5 stages)

- [x] Create `dit_size1024_radix4.go` ✅
- [x] Implement `forwardDIT1024Radix4Complex64()` ✅
- [x] Implement `inverseDIT1024Radix4Complex64()` ✅
- [x] Implement `forwardDIT1024Radix4Complex128()` ✅
- [x] Implement `inverseDIT1024Radix4Complex128()` ✅
- [x] Uses existing `ComputeBitReversalIndicesRadix4()` ✅
- [x] Register in `codelet_init.go` with priority 15 ✅
- [x] All tests passing (6/6 tests) ✅
- [ ] Benchmark vs generic Stockham

#### 14.1.3 Size 2048 - Mixed-Radix-2/4 ✅ COMPLETE

**Current**: Size-specific mixed-radix-2/4 (5 r4 + 1 r2 = 6 stages)
**Status**: Fully implemented and tested

**Implementation (2024-12-31):**

For 2048 = 2 × 4⁵, uses same algorithm structure as size 512:

- Stages 1-5: radix-4 butterflies (512, 128, 32, 8, 2 groups)
- Stage 6: radix-2 final stage (combines two 1024-point halves)

- [x] Create `dit_size2048_mixed24.go` ✅
- [x] Uses `ComputeBitReversalIndicesMixed24()` from size 512 ✅
- [x] Implement `forwardDIT2048Mixed24Complex64()` ✅
- [x] Implement `inverseDIT2048Mixed24Complex64()` ✅
- [x] Implement `forwardDIT2048Mixed24Complex128()` ✅
- [x] Implement `inverseDIT2048Mixed24Complex128()` ✅
- [x] All tests passing (6/6 tests) ✅
- [ ] Benchmark vs generic Stockham

#### 14.1.4 Size 4096 - Pure Radix-4 ✅ COMPLETE

**Current**: size-specific radix-4 (6 stages)
**Status**: Fully implemented and tested

- [x] Create `dit_size4096_radix4.go` ✅
- [x] Implement `forwardDIT4096Radix4Complex64()` ✅
- [x] Implement `inverseDIT4096Radix4Complex64()` ✅
- [x] Implement `forwardDIT4096Radix4Complex128()` ✅
- [x] Implement `inverseDIT4096Radix4Complex128()` ✅
- [x] Uses existing `ComputeBitReversalIndicesRadix4()` ✅
- [x] Register in `codelet_init.go` with priority 15 ✅
- [x] All tests passing (6/6 tests) ✅
- [ ] Benchmark vs generic Stockham

#### 14.1.5 Size 8192 - Mixed-Radix-2/4 ✅ COMPLETE

**Current**: Size-specific mixed-radix-2/4 (6 r4 + 1 r2 = 7 stages)
**Status**: Fully implemented and tested

**Implementation (2024-12-31):**

For 8192 = 2 × 4⁶, uses same algorithm structure as sizes 512 and 2048:

- Stages 1-6: radix-4 butterflies (2048, 512, 128, 32, 8, 2 groups)
- Stage 7: radix-2 final stage (combines two 4096-point halves)

- [x] Create `dit_size8192_mixed24.go` ✅
- [x] Uses `ComputeBitReversalIndicesMixed24()` from size 512 ✅
- [x] Implement `forwardDIT8192Mixed24Complex64()` ✅
- [x] Implement `inverseDIT8192Mixed24Complex64()` ✅
- [x] Implement `forwardDIT8192Mixed24Complex128()` ✅
- [x] Implement `inverseDIT8192Mixed24Complex128()` ✅
- [x] All tests passing (6/6 tests) ✅
- [ ] Benchmark vs generic Stockham

#### 14.1.6 Size 16384 - Pure Radix-4 ✅ COMPLETE

**Current**: size-specific radix-4 (7 stages)
**Status**: Fully implemented and tested

- [x] Create `dit_size16384_radix4.go` ✅
- [x] Implement `forwardDIT16384Radix4Complex64()` ✅
- [x] Implement `inverseDIT16384Radix4Complex64()` ✅
- [x] Implement `forwardDIT16384Radix4Complex128()` ✅
- [x] Implement `inverseDIT16384Radix4Complex128()` ✅
- [x] Uses existing `ComputeBitReversalIndicesRadix4()` ✅
- [x] Register in `codelet_init.go` with priority 15 ✅
- [x] All tests passing (6/6 tests) ✅
- [ ] Benchmark vs generic Stockham

---

### 14.2 AVX2 Assembly - Size-Specific Kernels (complex64)

**Goal**: AVX2 acceleration for all sizes up to 16K.

#### 14.2.1 Size 512 - AVX2 Mixed-Radix-2/4

**File**: `internal/kernels/asm/amd64/avx2_size512_mixed24.s`

- [ ] Implement `forwardAVX2Size512Mixed24Complex64Asm`
  - Stage 1: 256 radix-2 butterflies (AVX2 vectorized)
  - Stages 2-5: Radix-4 stages (reuse patterns from size-64/256)
- [ ] Implement `inverseAVX2Size512Mixed24Complex64Asm`
- [ ] Add declarations in `amd64/dispatch.go`
- [ ] Register in `codelet_init_avx2.go` with priority 25
- [ ] Test correctness vs pure-Go
- [ ] Benchmark (expect 1.8-2.2x over Go)

#### 14.2.2 Size 1024 - AVX2 Pure Radix-4

**File**: `internal/kernels/asm/amd64/avx2_size1024_radix4.s`

- [ ] Implement `forwardAVX2Size1024Radix4Complex64Asm`
  - 5 fully unrolled radix-4 stages
  - Hardcoded radix-4 bit-reversal
- [ ] Implement `inverseAVX2Size1024Radix4Complex64Asm`
- [ ] Add declarations and register
- [ ] Test and benchmark (expect 2.0-2.5x over Go)

#### 14.2.3 Size 2048 - AVX2 Mixed-Radix-2/4

**File**: `internal/kernels/asm/amd64/avx2_size2048_mixed24.s`

- [ ] Implement `forwardAVX2Size2048Mixed24Complex64Asm`
  - 1 radix-2 stage + 5 radix-4 stages
- [ ] Implement inverse
- [ ] Register and test
- [ ] Benchmark

#### 14.2.4 Size 4096 - AVX2 Pure Radix-4

**File**: `internal/kernels/asm/amd64/avx2_size4096_radix4.s`

- [ ] Implement `forwardAVX2Size4096Radix4Complex64Asm`
  - 6 radix-4 stages
- [ ] Implement inverse
- [ ] Register and test
- [ ] Benchmark

#### 14.2.5 Size 8192 - AVX2 Mixed-Radix-2/4

**File**: `internal/kernels/asm/amd64/avx2_size8192_mixed24.s`

- [ ] Implement `forwardAVX2Size8192Mixed24Complex64Asm`
  - 1 radix-2 stage + 6 radix-4 stages
- [ ] Implement inverse
- [ ] Register and test
- [ ] Benchmark

#### 14.2.6 Size 16384 - AVX2 Pure Radix-4

**File**: `internal/kernels/asm/amd64/avx2_size16384_radix4.s`

- [ ] Implement `forwardAVX2Size16384Radix4Complex64Asm`
  - 7 radix-4 stages
- [ ] Implement inverse
- [ ] Register and test
- [ ] Benchmark

---

### 14.3 Complete Existing AVX2 Gaps (complex64)

**Goal**: Fill gaps in existing small-size AVX2 implementations.

#### 14.3.1 Size 4 - Verify Complete

- [x] Forward AVX2 radix-4 ✅
- [ ] Verify inverse AVX2 exists and works
- [ ] Test round-trip accuracy

#### 14.3.2 Size 8 - Re-evaluate AVX2 Performance

**Current**: AVX2 disabled because Go radix-8 is faster

- [ ] Benchmark AVX2 vs Go radix-8 on modern CPUs
- [ ] If AVX2 can be improved, optimize assembly
- [ ] Otherwise, document decision to prefer Go

#### 14.3.3 Size 64 - Verify Radix-4 AVX2

- [x] Forward radix-4 AVX2 ✅
- [ ] Verify inverse radix-4 AVX2 exists
- [ ] Test round-trip accuracy

#### 14.3.4 Size 128 - Add Radix-4 AVX2 Variant

**Current**: Only radix-2 and mixed-2/4 (wrapper)

- [ ] Create `asm_amd64_avx2_size128_radix4.s`
- [ ] Implement radix-4 variant (4 stages vs 7 for radix-2)
- [ ] Compare performance vs mixed-2/4 wrapper
- [ ] Register best performer

#### 14.3.5 Size 256 - Verify Complete

- [x] Forward radix-2 AVX2 ✅
- [x] Forward radix-4 AVX2 ✅
- [ ] Verify inverse variants exist
- [ ] Test round-trip accuracy

---

### 14.4 Fix AVX2 Stockham Correctness

**Status**: Compiles ✅, segfault fixed ✅, **produces wrong results** ⚠️
**Priority**: HIGH (blocks Stockham AVX2 usage)

**Location**: `internal/kernels/asm/amd64/` (AVX2 assembly files)

- [ ] Debug why Stockham transforms differ from pure-Go
  - [ ] Add debug logging to identify which stage diverges
  - [ ] Compare intermediate buffer states step by step
  - [ ] Check buffer swap logic (dst ↔ scratch)
- [ ] Fix identified bugs
- [ ] Run full test suite with `-tags=fft_asm`
- [ ] Benchmark Stockham AVX2 vs DIT AVX2

**Debugging approach**:

```bash
go test -tags=fft_asm -v -run TestStockham ./internal/fft/
go test -tags=fft_asm -v -run TestAVX2MatchesPureGo ./internal/fft/
```

---

### 14.5 Pure Go Size-Specific Kernels (complex128) ✅ COMPLETE

**Goal**: Match complex64 coverage for complex128.
**Status**: All sizes 512-16384 implemented (2024-12-31)

#### 14.5.1 Size 512 - Mixed-Radix-2/4 ✅

- [x] Implement `forwardDIT512Mixed24Complex128()` ✅
- [x] Implement `inverseDIT512Mixed24Complex128()` ✅
- [x] Test and benchmark ✅

#### 14.5.2 Size 1024 - Pure Radix-4 ✅

- [x] Implement `forwardDIT1024Radix4Complex128()` ✅
- [x] Implement `inverseDIT1024Radix4Complex128()` ✅
- [x] Test and benchmark ✅

#### 14.5.3 Size 2048 - Mixed-Radix-2/4 ✅

- [x] Implement `forwardDIT2048Mixed24Complex128()` ✅
- [x] Implement `inverseDIT2048Mixed24Complex128()` ✅
- [x] Test and benchmark ✅

#### 14.5.4 Size 4096 - Pure Radix-4 ✅

- [x] Implement `forwardDIT4096Radix4Complex128()` ✅
- [x] Implement `inverseDIT4096Radix4Complex128()` ✅
- [x] Test and benchmark ✅

#### 14.5.5 Size 8192 - Mixed-Radix-2/4 ✅

- [x] Implement `forwardDIT8192Mixed24Complex128()` ✅
- [x] Implement `inverseDIT8192Mixed24Complex128()` ✅
- [x] Test and benchmark ✅

#### 14.5.6 Size 16384 - Pure Radix-4 ✅

- [x] Implement `forwardDIT16384Radix4Complex128()` ✅
- [x] Implement `inverseDIT16384Radix4Complex128()` ✅
- [x] Test and benchmark ✅

---

### 14.6 AVX2 Assembly (complex128)

**Goal**: AVX2 acceleration for complex128 (2 values per YMM register).
**Status**: Small sizes (4-256) complete, large sizes pending

#### 14.6.1 Complete Small Sizes ✅ COMPLETE

**Implementation (2026-01-01)**:

- [x] Size 4: `asm_amd64_avx2_size4_complex128.s` (radix-4) ✅
- [x] Size 8: `asm_amd64_avx2_size8_complex128.s` (radix-2, mixed-radix) ✅
- [x] Size 64: `asm_amd64_avx2_size64_complex128.s` (radix-2 and radix-4) ✅
- [x] Size 128: `asm_amd64_avx2_size128_complex128.s` (radix-2) ✅
- [x] Size 256: `asm_amd64_avx2_size256_complex128.s` (radix-2) ✅
- [x] Register all in `codelet_init_avx2.go` ✅
- [x] All 1D kernel tests passing ✅

#### 14.6.2 Large Sizes (512-16384)

- [ ] Size 512: `asm_amd64_avx2_size512_complex128.s`
- [ ] Size 1024: `asm_amd64_avx2_size1024_complex128.s`
- [ ] Size 2048: `asm_amd64_avx2_size2048_complex128.s`
- [ ] Size 4096: `asm_amd64_avx2_size4096_complex128.s`
- [ ] Size 8192: `asm_amd64_avx2_size8192_complex128.s`
- [ ] Size 16384: `asm_amd64_avx2_size16384_complex128.s`

---

### 14.7 Stockham Optimizations (Optional)

**Goal**: Size-specific Stockham for cache-sensitive workloads.
**Priority**: LOW (DIT with proper SIMD likely sufficient)

#### 14.7.1 Evaluate Stockham vs DIT Performance

- [ ] Benchmark generic Stockham vs optimized DIT for sizes 512-16384
- [ ] Identify if Stockham provides benefit for any size range
- [ ] Document findings

#### 14.7.2 Size-Specific Stockham (if beneficial)

Only if 14.7.1 shows benefit:

- [ ] `stockham_size512.go`
- [ ] `stockham_size1024.go`
- [ ] `stockham_size2048.go`

---

### 14.8 Testing & Benchmarking

#### 14.8.1 Comprehensive Test Suite

- [ ] Correctness tests for all new kernels vs reference DFT
- [ ] Round-trip tests (Forward → Inverse ≈ identity)
- [ ] Property tests (Parseval, linearity, shift theorems)
- [ ] Cross-validation: AVX2 vs pure-Go for each size
- [ ] Edge case tests (DC component, Nyquist, pure tones)

#### 14.8.2 Performance Benchmarks

- [ ] Create `benchmarks/phase14_results.txt`
- [ ] Benchmark all sizes 4-16384 for:
  - Pure Go baseline
  - Optimized Go (radix-4/mixed-radix)
  - AVX2 assembly
- [ ] Use `benchstat` for statistical comparison
- [ ] Document speedup ratios

#### 14.8.3 Update Documentation

- [ ] Update `docs/IMPLEMENTATION_INVENTORY.md` with new implementations
- [ ] Update `BENCHMARKS.md` with performance data
- [ ] Add performance notes to README

---

### Success Criteria

**Phase 14 Complete When**:

1. **complex64 Go**: All sizes 4-16384 have optimal algorithm (radix-4 or mixed-2/4)
2. **complex64 AVX2**: All sizes 4-16384 have size-specific AVX2 kernels
3. **complex128 Go**: All sizes 4-16384 have optimal algorithm
4. **complex128 AVX2**: Key sizes (8, 16, 32, 64, 128, 256) have AVX2
5. **Performance**: 2-3x speedup over baseline for all sizes
6. **Correctness**: All tests pass, round-trip error < 1e-6 (complex64) / 1e-14 (complex128)
7. **Documentation**: IMPLEMENTATION_INVENTORY.md fully updated

**Stretch Goals**:

- Size 8 AVX2 faster than Go radix-8
- Stockham AVX2 working correctly
- complex128 AVX2 for sizes 512-16384

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

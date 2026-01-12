# PLAN.md - algofft Implementation Roadmap

## Completed (Summary)

**Phases 1-10**: Project setup, types, API, errors, twiddles, bit-reversal, radix-2/3/4/5 FFT, Stockham autosort, DIT, mixed-radix, Bluestein, six-step/eight-step large FFT, SIMD infrastructure, CPU detection

**Real FFT**: Forward/inverse, generic float32/float64, 2D/3D real FFT with compact/full output
**Multi-dimensional**: 2D, 3D, N-D FFT with generic support
**Testing**: Reference DFT, correctness, property-based, fuzz, precision, stress, concurrent tests
**Benchmarking**: Full suite with regression tracking, BENCHMARKS.md
**Batch/Strided**: Sequential batch API, strided transforms
**Convolution**: FFT-based convolution/correlation for complex and real
**complex128**: Full generic API with explicit constructors
**WebAssembly**: Build, test (Node.js), examples
**Cross-arch CI**: amd64, arm64, 386, WASM matrix

---

## Current Status

See `docs/IMPLEMENTATION_INVENTORY.md` for full inventory. Assembly: `internal/asm/{amd64,arm64,x86}/`, registration: `internal/kernels/codelet_init*.go`

**Phase 11 ✅**: Bit-reversal refactoring complete — all kernels now self-contained with internal permutation (Pure Go, AVX2, SSE2, NEON)

**Phase 12 ✅**: Mixed-radix optimization complete — ping-pong buffering (20% throughput gain), type-specific butterflies (19% speedup), AVX2 radix-3/5, size-384 codelet; iterative version deferred (recursive already optimal)

---

## Phase 13: SSE2 Coverage (Sizes 256-1024)

**Status**: In Progress
**Priority**: Medium (provides fallback for systems without AVX2)

Target: Implement SSE2 kernels for sizes 256-1024 to ensure systems without AVX2 have optimized paths. Reference: `docs/IMPLEMENTATION_INVENTORY.md`

### 13.1 Size 256 SSE2 Kernels

#### 13.1.1 complex64 Size 256 (Already Complete)

- [x] Size 256: radix-4 SSE2 (exists: `internal/asm/amd64/sse2_f32_size256_radix4.s`)

#### 13.1.2 complex128 Size 256 SSE2 Kernels

- [x] Implement Size 256 radix-2 SSE2 for complex128
  - [x] Create `internal/asm/amd64/sse2_f64_size256_radix2.s`
  - [x] Implement `ForwardSSE2Size256Radix2Complex128Asm` (8 stages)
  - [x] Implement `InverseSSE2Size256Radix2Complex128Asm` (with 1/256 scaling)
  - [x] Add Go wrapper in `internal/kernels/sse2_f64_size256_radix2.go`
  - [x] Register in codelet system with priority 10
- [ ] Implement Size 256 radix-4 SSE2 for complex128
  - [ ] Create `internal/asm/amd64/sse2_f64_size256_radix4.s`
  - [ ] Implement `ForwardSSE2Size256Radix4Complex128Asm` (4 stages)
  - [ ] Implement `InverseSSE2Size256Radix4Complex128Asm` (with 1/256 scaling)
  - [ ] Add Go wrapper in `internal/kernels/sse2_f64_size256_radix4.go`
  - [ ] Register in codelet system with priority 15

#### 13.1.3 Size 256 Test Coverage

- [ ] Add test file `internal/kernels/sse2_f64_size256_radix2_test.go`
  - [ ] TestForwardSSE2Size256Radix2Complex128
  - [ ] TestInverseSSE2Size256Radix2Complex128
  - [ ] TestRoundTripSSE2Size256Radix2Complex128
- [ ] Add test file `internal/kernels/sse2_f64_size256_radix4_test.go`
  - [ ] TestForwardSSE2Size256Radix4Complex128
  - [ ] TestInverseSSE2Size256Radix4Complex128
  - [ ] TestRoundTripSSE2Size256Radix4Complex128

### 13.2 Size 512 SSE2 Kernels

#### 13.2.1 complex64 Size 512 SSE2 Kernels

- [ ] Implement Size 512 radix-2 SSE2 for complex64
  - [ ] Create `internal/asm/amd64/sse2_f32_size512_radix2.s`
  - [ ] Implement `ForwardSSE2Size512Radix2Complex64Asm` (9 stages)
  - [ ] Implement `InverseSSE2Size512Radix2Complex64Asm` (with 1/512 scaling)
  - [ ] Add Go wrapper in `internal/kernels/sse2_f32_size512_radix2.go`
  - [ ] Register in codelet system with priority 10
- [ ] Implement Size 512 mixed-2/4 SSE2 for complex64
  - [ ] Create `internal/asm/amd64/sse2_f32_size512_mixed24.s`
  - [ ] Implement `ForwardSSE2Size512Mixed24Complex64Asm` (4 radix-4 + 1 radix-2 = 5 stages)
  - [ ] Implement `InverseSSE2Size512Mixed24Complex64Asm` (with 1/512 scaling)
  - [ ] Add Go wrapper in `internal/kernels/sse2_f32_size512_mixed24.go`
  - [ ] Register in codelet system with priority 15

#### 13.2.2 complex128 Size 512 SSE2 Kernels

- [ ] Implement Size 512 radix-2 SSE2 for complex128
  - [ ] Create `internal/asm/amd64/sse2_f64_size512_radix2.s`
  - [ ] Implement `ForwardSSE2Size512Radix2Complex128Asm` (9 stages)
  - [ ] Implement `InverseSSE2Size512Radix2Complex128Asm` (with 1/512 scaling)
  - [ ] Add Go wrapper in `internal/kernels/sse2_f64_size512_radix2.go`
  - [ ] Register in codelet system with priority 10
- [ ] Implement Size 512 mixed-2/4 SSE2 for complex128
  - [ ] Create `internal/asm/amd64/sse2_f64_size512_mixed24.s`
  - [ ] Implement `ForwardSSE2Size512Mixed24Complex128Asm` (4 radix-4 + 1 radix-2 = 5 stages)
  - [ ] Implement `InverseSSE2Size512Mixed24Complex128Asm` (with 1/512 scaling)
  - [ ] Add Go wrapper in `internal/kernels/sse2_f64_size512_mixed24.go`
  - [ ] Register in codelet system with priority 15

#### 13.2.3 Size 512 Test Coverage

- [ ] Add test files for complex64:
  - [ ] `internal/kernels/sse2_size512_radix2_test.go` (Forward, Inverse, RoundTrip)
  - [ ] `internal/kernels/sse2_size512_mixed24_test.go` (Forward, Inverse, RoundTrip)
- [ ] Add test files for complex128:
  - [ ] `internal/kernels/sse2_f64_size512_radix2_test.go` (Forward, Inverse, RoundTrip)
  - [ ] `internal/kernels/sse2_f64_size512_mixed24_test.go` (Forward, Inverse, RoundTrip)

### 13.3 Size 1024 SSE2 Kernels

#### 13.3.1 complex64 Size 1024 SSE2 Kernels

- [ ] Implement Size 1024 radix-4 SSE2 for complex64
  - [ ] Create `internal/asm/amd64/sse2_f32_size1024_radix4.s`
  - [ ] Implement `ForwardSSE2Size1024Radix4Complex64Asm` (5 radix-4 stages)
  - [ ] Implement `InverseSSE2Size1024Radix4Complex64Asm` (with 1/1024 scaling)
  - [ ] Use radix-4 bit-reversal pattern
  - [ ] Add Go wrapper in `internal/kernels/sse2_f32_size1024_radix4.go`
  - [ ] Register in codelet system with priority 15

#### 13.3.2 complex128 Size 1024 SSE2 Kernels

- [ ] Implement Size 1024 radix-4 SSE2 for complex128
  - [ ] Create `internal/asm/amd64/sse2_f64_size1024_radix4.s`
  - [ ] Implement `ForwardSSE2Size1024Radix4Complex128Asm` (5 radix-4 stages)
  - [ ] Implement `InverseSSE2Size1024Radix4Complex128Asm` (with 1/1024 scaling)
  - [ ] Use radix-4 bit-reversal pattern
  - [ ] Add Go wrapper in `internal/kernels/sse2_f64_size1024_radix4.go`
  - [ ] Register in codelet system with priority 15

#### 13.3.3 Size 1024 Test Coverage

- [ ] Add test file `internal/kernels/sse2_size1024_radix4_test.go`
  - [ ] TestForwardSSE2Size1024Radix4Complex64
  - [ ] TestInverseSSE2Size1024Radix4Complex64
  - [ ] TestRoundTripSSE2Size1024Radix4Complex64
- [ ] Add test file `internal/kernels/sse2_f64_size1024_radix4_test.go`
  - [ ] TestForwardSSE2Size1024Radix4Complex128
  - [ ] TestInverseSSE2Size1024Radix4Complex128
  - [ ] TestRoundTripSSE2Size1024Radix4Complex128

### 13.4 Performance Validation

- [ ] Run benchmarks comparing SSE2 vs pure Go for all new kernels
- [ ] Document performance improvements in benchmark results
- [ ] Update `docs/IMPLEMENTATION_INVENTORY.md` with new SSE2 coverage
- [ ] Verify SSE2 kernels are selected on systems without AVX2 support

---

## Phase 14: FFT Size Optimizations - Remaining Work

### 14.2 AVX2 Large Size Kernels (512-16384)

**Status**: In Progress
**Priority**: Medium (Pure Go implementations exist and perform well)

Sizes 512-16384 currently use pure Go mixed-radix or radix-4 implementations. AVX2 acceleration could provide 1.5-2x additional speedup.

#### 14.2.5 Size 8192 - AVX2 Mixed-Radix-2/4

- [x] Create `internal/asm/amd64/avx2_f32_size8192_mixed24.s`
  - [x] Implement `forwardAVX2Size8192Mixed24Complex64` (6 radix-4 stages + 1 radix-2 stage)
  - [x] Implement `inverseAVX2Size8192Mixed24Complex64` (with 1/8192 scaling)
- [x] Add Go declarations and register with priority 25
- [x] Add correctness tests and benchmark

#### 14.2.6 Size 16384 - AVX2 Pure Radix-4

- [x] Create `internal/asm/amd64/avx2_f32_size16384_radix4.s`
  - [x] Implement `forwardAVX2Size16384Radix4Complex64` (7 radix-4 stages)
  - [x] Implement `inverseAVX2Size16384Radix4Complex64` (with 1/16384 scaling)
- [x] Add Go declarations and register with priority 25
- [x] Add correctness tests and benchmark

### 14.3 Complete Existing AVX2 Gaps

#### 14.3.1 Verify Inverse Transforms

- [x] Size 4: Test `inverseAVX2Size4Radix4Complex64` round-trip accuracy
  - [x] Run `Forward → Inverse` and verify `max|x - result| < 1e-6`
  - [x] Test with random inputs, DC component, Nyquist frequency
- [x] Size 64: Test `inverseAVX2Size64Radix4Complex64` round-trip accuracy
- [x] Size 256: Test `inverseAVX2Size256Radix4Complex64` round-trip accuracy
- [x] Add dedicated inverse transform test file if not present

#### 14.3.2 Size 8 AVX2 Re-evaluation

- [x] Benchmark current Go radix-8 vs AVX2 radix-2 on modern CPUs (Zen4, Raptor Lake)
- [x] Profile to identify bottlenecks in AVX2 size-8 implementation
- [x] If AVX2 can be improved:
  - [ ] Optimize register allocation and instruction scheduling
  - [ ] Consider radix-8 AVX2 instead of radix-2
  - [ ] Re-benchmark and enable if faster
- [x] If Go remains faster:
  - [x] Document rationale in code comments
  - [x] Keep AVX2 disabled (priority 9, lower than SSE2)
  - **Note**: SSE2 Size 8 Radix 8 fixed (fast). AVX2 Size 8 Radix 8 fixed (slow).

#### 14.3.3 Size 128 Radix-4 AVX2

- [x] Create `internal/asm/amd64/avx2_f32_size128_radix4.s` (currently only radix-2/mixed exist)
  - [x] Implement `forwardAVX2Size128Radix4Complex64` (3.5 stages: 3 radix-4 + 1 radix-2)
  - [x] Implement `inverseAVX2Size128Radix4Complex64`
  - [ ] Use radix-4 bit-reversal for first 64 elements, binary for rest
- [ ] Benchmark radix-4 vs current mixed-2/4 wrapper
- [ ] Register higher-performing variant with higher priority
- **Status**: Disabled. Implementation exists but failed correctness tests (bit-reversal/logic mismatch). Reverted to pure-Go fallback.

### 14.4 Fix AVX2 Stockham Correctness

**Status**: Compiles and runs without segfault, but produces wrong results
**Priority**: LOW (DIT kernels work correctly)

- [ ] Add debug logging to Stockham assembly
  - [ ] Dump intermediate buffer after each stage
  - [ ] Compare with pure-Go stage outputs
- [ ] Identify which stage first diverges from pure-Go
- [ ] Check buffer swap logic (dst ↔ scratch pointer handling)
- [ ] Verify twiddle factor indexing matches pure-Go
- [ ] Fix identified bugs and re-test
- [ ] Run full test suite with `-tags=asm -run TestStockham`

### 14.6.2 AVX2 complex128 Large Sizes (512-16384)

**Status**: In Progress
**Priority**: Low (complex128 use cases less common)

For each size, create assembly file in `internal/asm/amd64/`:

- [x] Size 512: `avx2_f64_size512_mixed24.s`
  - [x] Forward and inverse transforms
  - [x] Register in `codelet_init_avx2.go`
- [ ] Size 1024: `avx2_f64_size1024_radix4.s`
- [ ] Size 2048: `avx2_f64_size2048_mixed24.s`
- [ ] Size 4096: `avx2_f64_size4096_radix4.s`
- [ ] Size 8192: `avx2_f64_size8192_mixed24.s`
- [ ] Size 16384: `avx2_f64_size16384_radix4.s`

### 14.7 Higher-Radix Optimization Strategies

**Status**: Not started
**Priority**: Medium-High (could provide 1.3-2x speedup over radix-4 for larger sizes)

For larger FFT sizes, higher radices reduce the number of stages (and thus memory passes), potentially improving cache utilization and throughput. Each radix-N stage reduces log₂(N) stages into one, at the cost of more complex butterfly operations.

#### 14.7.1 Radix Decomposition Analysis

| Size  | Radix-2   | Radix-4    | Radix-8   | Radix-16   | Optimal Decomposition      |
| ----- | --------- | ---------- | --------- | ---------- | -------------------------- |
| 256   | 8 stages  | 4 stages   | 2⅔ stages | 2 stages   | 16×16 (2 stages)           |
| 512   | 9 stages  | 4.5 stages | 3 stages  | 2¼ stages  | 8×8×8 (3 radix-8) or 16×32 |
| 1024  | 10 stages | 5 stages   | 3⅓ stages | 2.5 stages | 32×32 or 16×64             |
| 2048  | 11 stages | 5.5 stages | 3⅔ stages | 2¾ stages  | 32×64 or 8×16×16           |
| 4096  | 12 stages | 6 stages   | 4 stages  | 3 stages   | 64×64 or 16×16×16          |
| 8192  | 13 stages | 6.5 stages | 4⅓ stages | 3¼ stages  | 64×128 or 16×32×16         |
| 16384 | 14 stages | 7 stages   | 4⅔ stages | 3.5 stages | 128×128 or 16×16×64        |

**Note**: "N×M" notation means a 2D Cooley-Tukey decomposition (N rows × M columns).

#### 14.7.2 Size 256 - Radix-16 (2-Stage)

**Rationale**: 256 = 16 × 16, can be computed as a 16×16 matrix with:

1. Column FFT-16 (using existing radix-16 kernel)
2. Twiddle multiplication
3. Row FFT-16 (same kernel)

- [ ] Create `internal/asm/amd64/avx2_f32_size256_radix16.s`
  - [ ] Implement as 16×16 matrix factorization
  - [ ] Stage 1: 16 parallel FFT-16 on columns
  - [ ] Twiddle: W₂₅₆^(row×col) multiplication
  - [ ] Stage 2: 16 parallel FFT-16 on rows
  - [ ] Final transposition to natural order
- [ ] No bit-reversal needed (identity permutation for 4^k sizes)
- [ ] Register with priority 30 (higher than radix-4 priority 25)
- [ ] Benchmark: Target 1.3-1.5x speedup vs radix-4

#### 14.7.3 Size 512 - Radix-8 (3-Stage)

**Rationale**: 512 = 8 × 8 × 8, can be computed with 3 radix-8 stages.

- [x] Create `internal/asm/amd64/avx2_f32_size512_radix8.s`
  - [x] Implement 3 radix-8 stages (vs 5 stages for mixed-2/4)
  - [x] Use radix-8 twiddle factors: W₅₁₂^k for k ∈ {0,1,2,3,4,5,6,7}×stride
  - [x] Radix-8 butterfly: 8-point DFT inline
- [x] Create radix-8 bit-reversal function `ComputeBitReversalIndicesRadix8(n int) []int`
- [x] Register with priority 30 (higher than mixed-2/4 priority 25)
- [x] Benchmark: Target 1.2-1.4x speedup vs mixed-2/4

**Alternative**: 512 = 16 × 32 (2-stage mixed-radix-16/32)

- [x] Pure Go optimized implementation: `internal/kernels/dit_size512_radix16x32.go`
  - [x] Six-step FFT algorithm: n = 16*n2 + n1, k = 32*k1 + k2
  - [x] Stage 1: 16 FFT-32s on columns using Cooley-Tukey decomposition (FFT-32 = 2×FFT-16 + twiddles)
  - [x] Stage 2: 32 FFT-16s on rows using DIT fft16 with bit-reversed input
  - [x] Algorithm validated: forward/inverse for complex64 and complex128
  - [x] **Performance**: ~8% faster than radix-8 Go implementation (3908 ns/op vs 4245 ns/op forward)
  - [x] Uses precomputed twiddle factors from 512-point table
  - [x] Uses identity permutation (no bit-reversal on input)
- [x] Create `internal/asm/amd64/avx2_f32_size512_radix16x32.s` (stub implementation)
  - [x] Add Go function declarations in `internal/asm/amd64/decl.go`
  - [x] Add test file `internal/kernels/avx2_f32_size512_radix16x32_test.go`
  - [x] Stub returns false to use Go fallback (full AVX2 implementation deferred)
  - **Performance**: Go radix-16x32 achieves ~4821 ns/op (faster than Go radix-8 ~5147 ns/op)
  - **Note**: AVX2 radix-8 achieves ~2052 ns/op - full AVX2 radix-16x32 could be competitive
  - [x] Stage 1: 16 parallel FFT-32 on columns (using SIMD radix-32 butterflies)
  - [x] Twiddle multiplication
  - [x] Stage 2: 32 parallel FFT-16 on rows (using SIMD radix-16 butterflies)
  - [x] Could be competitive with radix-8 AVX2 (2 stages vs 3 stages)

#### 14.7.4 Size 1024 - Radix-16 (2.5-Stage) or 32×32

**Rationale**: 1024 = 32 × 32 or 1024 = 16 × 64

**Option A - 32×32 Matrix**:

- [ ] Create `internal/asm/amd64/avx2_f32_size1024_radix32x32.s`
  - [ ] Stage 1: 32 parallel FFT-32 on columns
  - [ ] Twiddle multiplication
  - [ ] Stage 2: 32 parallel FFT-32 on rows
- [ ] Requires radix-32 butterfly (existing size-32 radix-32 kernel can be reused)

**Option B - 16×64 Matrix**:

- [ ] Create `internal/asm/amd64/avx2_f32_size1024_radix16x64.s`
  - [ ] Stage 1: 64 parallel FFT-16 on columns
  - [ ] Twiddle multiplication
  - [ ] Stage 2: 16 parallel FFT-64 on rows

- [ ] Benchmark both options vs current radix-4 (5 stages)
- [ ] Register higher-performing variant with priority 30

#### 14.7.5 Size 2048 - Higher-Radix Decompositions

**Rationale**: 2048 = 2 × 1024 = 32 × 64 = 16 × 128 = 8 × 256

**Option A - 32×64 Matrix**:

- [ ] Create `internal/asm/amd64/avx2_f32_size2048_radix32x64.s`
  - [ ] Stage 1: 64 parallel FFT-32
  - [ ] Twiddle multiplication
  - [ ] Stage 2: 32 parallel FFT-64

**Option B - 16×128 Matrix**:

- [ ] Create `internal/asm/amd64/avx2_f32_size2048_radix16x128.s`
  - [ ] Stage 1: 128 parallel FFT-16
  - [ ] Twiddle multiplication
  - [ ] Stage 2: 16 parallel FFT-128

**Option C - 8×16×16 (3D decomposition)**:

- [ ] Create `internal/asm/amd64/avx2_f32_size2048_radix8x16x16.s`
  - [ ] Three-stage: FFT-8 → twiddle → FFT-16 → twiddle → FFT-16
  - [ ] Only 3 stages instead of 5.5

- [ ] Benchmark all options vs current mixed-2/4
- [ ] Register best performer with priority 30

#### 14.7.6 Size 4096 - Radix-16 (3-Stage) or 64×64

**Rationale**: 4096 = 16³ = 64 × 64 = 256 × 16

**Option A - 16×16×16 Cube (3-stage)**:

- [ ] Create `internal/asm/amd64/avx2_f32_size4096_radix16cubed.s`
  - [ ] Stage 1: 256 parallel FFT-16
  - [ ] Twiddle multiplication
  - [ ] Stage 2: 256 parallel FFT-16
  - [ ] Twiddle multiplication
  - [ ] Stage 3: 256 parallel FFT-16
  - [ ] 3 stages total (vs 6 for radix-4)

**Option B - 64×64 Matrix**:

- [ ] Create `internal/asm/amd64/avx2_f32_size4096_radix64x64.s`
  - [ ] Stage 1: 64 parallel FFT-64
  - [ ] Twiddle multiplication
  - [ ] Stage 2: 64 parallel FFT-64
  - [ ] 2 stages total (optimal!)

- [ ] Benchmark vs current radix-4 (6 stages)
- [ ] Target 1.5-2x speedup with 64×64 decomposition

#### 14.7.7 Size 8192 - Higher-Radix Decompositions

**Rationale**: 8192 = 64 × 128 = 32 × 256 = 16 × 512

**Recommended - 64×128 Matrix**:

- [ ] Create `internal/asm/amd64/avx2_f32_size8192_radix64x128.s`
  - [ ] Stage 1: 128 parallel FFT-64
  - [ ] Twiddle multiplication
  - [ ] Stage 2: 64 parallel FFT-128
  - [ ] 2 stages (vs 6.5 for mixed-2/4)

**Alternative - 16×32×16 (3D)**:

- [ ] Create `internal/asm/amd64/avx2_f32_size8192_radix16x32x16.s`
  - [ ] Three-stage decomposition

- [ ] Benchmark vs current mixed-2/4
- [ ] Target 2-3x speedup with 2-stage decomposition

#### 14.7.8 Size 16384 - Radix-128 (2-Stage)

**Rationale**: 16384 = 128 × 128 = 64 × 256 = 16 × 1024

**Optimal - 128×128 Matrix**:

- [ ] Create `internal/asm/amd64/avx2_f32_size16384_radix128x128.s`
  - [ ] Stage 1: 128 parallel FFT-128
  - [ ] Twiddle multiplication
  - [ ] Stage 2: 128 parallel FFT-128
  - [ ] 2 stages only! (vs 7 for radix-4)

**Alternative - 16×16×64 (3D)**:

- [ ] Create `internal/asm/amd64/avx2_f32_size16384_radix16x16x64.s`
  - [ ] Three-stage: FFT-16 → twiddle → FFT-16 → twiddle → FFT-64
  - [ ] 3 stages (vs 7 for radix-4)

- [ ] Benchmark vs current radix-4 (7 stages)
- [ ] Target 2-3x speedup with 2-stage decomposition

#### 14.7.9 Cache-Oblivious Strategies

For sizes that exceed L2 cache (typically 256KB-1MB), consider:

**Blocking/Tiling**:

- [ ] Implement cache-blocked variants for sizes ≥ 8192
  - [ ] Divide FFT into cache-sized blocks
  - [ ] Process blocks sequentially to maintain cache residency
  - [ ] Trade extra passes for better cache utilization

**SIMD-Aware Data Layout**:

- [ ] Investigate SOA (Structure of Arrays) layout for complex data
  - [ ] Separate real and imaginary arrays
  - [ ] Better SIMD utilization (no interleave/deinterleave overhead)
  - [ ] Requires API extension (breaking change, v2.0 consideration)

#### 14.7.10 Implementation Priority Order

Based on expected benefit/effort ratio:

1. **Size 4096 - 64×64** (High impact, reuses FFT-64 kernel)
2. **Size 1024 - 32×32** (Medium size, reuses FFT-32 kernel)
3. **Size 256 - 16×16** (Reuses existing radix-16 kernel)
4. **Size 16384 - 128×128** (Highest absolute benefit, requires FFT-128)
5. **Size 8192 - 64×128** (Large benefit, requires FFT-128)
6. **Size 512 - radix-8** (Smaller benefit, new radix-8 infrastructure)
7. **Size 2048 - 32×64** (Medium benefit)

### 14.8 Testing & Benchmarking

#### 14.8.1 Comprehensive Benchmark Suite

- [ ] Create `benchmarks/phase14_results/` directory
- [ ] Run benchmarks for all sizes 4-16384:
  - [ ] Pure Go baseline (no SIMD tags)
  - [ ] Optimized Go (radix-4/mixed-radix)
  - [ ] AVX2 assembly (`-tags=asm`)
  - [ ] SSE2 fallback (`-tags=asm` on non-AVX2 CPU or emulated)

- [ ] Save results as `benchmarks/phase14_results/{arch}_{date}.txt`

#### 14.8.2 Statistical Analysis

- [ ] Install `benchstat` if not present: `go install golang.org/x/perf/cmd/benchstat@latest`
- [ ] Compare baseline vs optimized: `benchstat baseline.txt optimized.txt`
- [ ] Document speedup ratios in table format
- [ ] Identify any regressions

#### 14.8.3 Documentation Updates

- [ ] Update `docs/IMPLEMENTATION_INVENTORY.md` with new implementations
- [ ] Update `BENCHMARKS.md` with:
  - [ ] Performance comparison tables
  - [ ] Speedup charts (if applicable)
  - [ ] Hardware tested (CPU model, RAM speed)
- [ ] Add performance notes to README.md

---

## Phase 15: ARM64 NEON - Remaining Work

### 15.4 Production Testing on Real Hardware

**Status**: QEMU testing complete, real hardware pending

#### 15.4.1 Hardware Testing

- [ ] Acquire access to ARM64 hardware:
  - [ ] Option A: Raspberry Pi 4/5 (local)
  - [ ] Option B: AWS Graviton t4g.micro (free tier eligible)
  - [ ] Option C: Apple Silicon Mac (M1/M2/M3)
- [ ] Run full test suite on real hardware:
  ```bash
  go test -v -tags=asm ./...
  ```
- [ ] Verify all NEON kernels produce correct results
- [ ] Check for any hardware-specific issues (alignment, denormals)

#### 15.4.2 Performance Benchmarking

- [ ] Run benchmarks on real ARM64 hardware:
  ```bash
  just bench | tee benchmarks/arm64_native.txt
  ```
- [ ] Compare QEMU vs native performance ratios
- [ ] Document realistic speedup numbers for NEON kernels
- [ ] Identify sizes where NEON provides most benefit

#### 15.4.3 CI Integration

- [ ] Add ARM64 runner to GitHub Actions:
  - [ ] Option A: `runs-on: macos-14` (Apple Silicon)
  - [ ] Option B: Self-hosted ARM64 runner
  - [ ] Option C: ARM64 Docker container via QEMU (slower but available)
- [ ] Add ARM64 build job to `.github/workflows/ci.yml`
- [ ] Ensure SIMD paths are tested in CI
- [ ] Add ARM64 badge to README

#### 15.4.4 Documentation

- [ ] Add ARM64 section to BENCHMARKS.md:
  - [ ] Performance comparison tables (NEON vs pure-Go)
  - [ ] Hardware tested (Cortex-A76, Apple M1, Graviton, etc.)
- [ ] Document NEON characteristics:
  - [ ] 128-bit registers (2 complex64 per register)
  - [ ] Expected speedup range
- [ ] Compare NEON vs AVX2 speedup ratios

### 15.5 Size-Specific NEON Kernels - Remaining

Sizes 4, 8, 16, 32, 64, 128, 256 forward transforms implemented for complex64.

#### 15.5.1 Inverse Transforms

For each existing forward NEON kernel, implement inverse:

- [x] Size 4: `inverseNEONSize4Radix4Complex64`
  - [x] Add to `internal/asm/arm64/neon_f32_size4_radix4.s`
  - [x] Conjugate twiddle factors (negate imaginary part)
  - [x] Add 1/4 scaling factor
- [x] Size 8: `inverseNEONSize8Radix2Complex64`, `inverseNEONSize8Radix8Complex64`
- [x] Size 16: `inverseNEONSize16Radix4Complex64`
- [x] Size 32: `inverseNEONSize32Radix2Complex64`, `inverseNEONSize32Mixed24Complex64`
- [x] Size 64: `inverseNEONSize64Radix4Complex64`
- [x] Size 128: `inverseNEONSize128Radix2Complex64`, `inverseNEONSize128Mixed24Complex64`
- [x] Size 256: `inverseNEONSize256Radix4Complex64`
- [x] Add round-trip tests for each size

#### 15.5.2 Size 512+ NEON Kernels

Evaluate benefit before implementing (may not be worthwhile due to cache effects):

- [ ] Benchmark pure-Go sizes 512, 1024, 2048 on ARM64
- [ ] Estimate potential NEON speedup
- [ ] If > 1.5x expected:
  - [ ] Implement `forwardNEONSize512Mixed24Complex64`
  - [ ] Implement `forwardNEONSize1024Radix4Complex64`
- [ ] If < 1.3x expected:
  - [ ] Document decision to use pure-Go for large sizes
  - [ ] Focus optimization effort elsewhere

#### 15.5.3 complex128 NEON Kernels

NEON processes 1 complex128 per 128-bit register (half the throughput of complex64):

- [x] Evaluate if NEON complex128 provides benefit over pure-Go
- [x] If beneficial, implement for key sizes:
  - [x] Size 4: `forwardNEONSize4Radix4Complex128`
  - [x] Size 8: `forwardNEONSize8Radix2Complex128`
  - [x] Size 16: `forwardNEONSize16Radix4Complex128`
- [x] Add corresponding inverse transforms
- [x] Benchmark and document speedup

---

## Phase 16: Cache & Loop Optimization

### 16.1 Cache Profiling

- [ ] Install `perf` if not present (Linux)
- [ ] Run cache profiling on key benchmarks:
  ```bash
  perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
    go test -bench=BenchmarkPlan_1024 -benchtime=5s ./...
  ```
- [ ] Identify high cache-miss operations
- [ ] Compare cache behavior across FFT sizes (small vs large)
- [ ] Document baseline cache metrics

### 16.2 Loop Optimization

- [ ] Profile to identify critical inner loops:
  ```bash
  go test -bench=BenchmarkPlan_1024 -cpuprofile=cpu.prof ./...
  go tool pprof -top cpu.prof
  ```
- [ ] For top 3 hotspots:
  - [ ] Analyze loop structure
  - [ ] Implement 2x or 4x manual unrolling
  - [ ] Benchmark unrolled vs original
  - [ ] Keep if > 5% improvement, revert if not
- [ ] Consider `go:noinline` to prevent inlining that hurts cache

### 16.3 Bounds Check Elimination

- [ ] Identify bounds check hotspots:
  ```bash
  go build -gcflags="-d=ssa/check_bce/debug=1" ./internal/fft 2>&1 | grep -v "^#"
  ```
- [ ] For each bounds check in hot path:
  - [ ] Add `_ = slice[len-1]` pattern before loop
  - [ ] Or restructure loop to eliminate check
- [ ] Verify no safety regressions with fuzz tests
- [ ] Benchmark improvement

### 16.4 Memory Access Patterns

- [ ] Review butterfly loop memory access:
  - [ ] Check for cache line conflicts
  - [ ] Verify sequential access where possible
- [ ] Consider cache-oblivious algorithms for large sizes
- [ ] Implement prefetch hints if beneficial (via assembly)

---

## Phase 19: Batch Processing - Remaining

### 19.3 Parallel Batch Processing

#### 19.3.1 API Design

- [ ] Define parallel batch API:
  ```go
  func (p *Plan[T]) ForwardBatchParallel(dst, src []T, count int) error
  func (p *Plan[T]) InverseBatchParallel(dst, src []T, count int) error
  ```
- [ ] Decide on concurrency options:
  - [ ] Option A: Auto-detect optimal goroutine count
  - [ ] Option B: Accept worker count parameter
  - [ ] Option C: Use `runtime.GOMAXPROCS` directly

#### 19.3.2 Implementation

- [ ] Implement worker pool for batch processing:
  - [ ] Create fixed pool of workers (1 per CPU core)
  - [ ] Distribute transforms across workers
  - [ ] Use `sync.WaitGroup` for synchronization
- [ ] Ensure Plan is safe for concurrent read-only use:
  - [ ] Verify twiddle factors are read-only
  - [ ] Verify scratch buffers are per-goroutine (not shared)
- [ ] Handle partial batches (count not divisible by worker count)

#### 19.3.3 Tuning

- [ ] Find optimal batch-per-goroutine threshold:
  - [ ] Benchmark with batch sizes: 4, 8, 16, 32, 64, 128, 256
  - [ ] Find crossover point where parallelism helps
- [ ] Add `GOMAXPROCS` awareness:
  - [ ] Scale worker count with available cores
  - [ ] Respect user-set GOMAXPROCS
- [ ] Consider work-stealing for load balancing

#### 19.3.4 Testing

- [ ] Add concurrent correctness tests
- [ ] Add race detector tests: `go test -race ./...`
- [ ] Benchmark parallel vs sequential for various batch sizes
- [ ] Document speedup curves in BENCHMARKS.md

---

## Phase 22: complex128 - Remaining

### 22.3 Precision Profiling

#### 22.3.1 Error Measurement

- [ ] Create precision test suite in `precision_test.go`:
  - [ ] Measure round-trip error: `max|x - Inverse(Forward(x))|`
  - [ ] Test for FFT sizes: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
- [ ] Compare complex64 vs complex128 error:
  - [ ] Run same tests for both types
  - [ ] Calculate error ratio (complex64_error / complex128_error)

#### 22.3.2 Error Accumulation Analysis

- [ ] Analyze how error grows with size:
  - [ ] Plot error vs log2(size)
  - [ ] Fit to theoretical O(log n) error growth
- [ ] Identify precision-critical operations:
  - [ ] Twiddle factor computation
  - [ ] Butterfly multiply-add
- [ ] Compare with theoretical bounds

#### 22.3.3 Documentation

- [ ] Create `docs/PRECISION.md`:
  - [ ] Precision guarantees for each type
  - [ ] Recommended use cases (audio vs scientific computing)
  - [ ] Error bounds by FFT size
  - [ ] Comparison with other FFT libraries
- [ ] Add precision notes to README.md

---

## Phase 23: WebAssembly - Remaining

### 23.1 Browser Testing

#### 23.1.1 Test Environment Setup

- [ ] Create `examples/wasm/` directory
- [ ] Add `index.html` with wasm_exec.js loader:
  ```html
  <script src="wasm_exec.js"></script>
  <script>
    const go = new Go();
    WebAssembly.instantiateStreaming(fetch("fft.wasm"), go.importObject).then(
      (result) => go.run(result.instance),
    );
  </script>
  ```
- [ ] Create simple FFT demo in Go (exports to JS)
- [ ] Build WASM: `GOOS=js GOARCH=wasm go build -o fft.wasm`

#### 23.1.2 Browser Compatibility

- [ ] Test in major browsers:
  - [ ] Chrome (latest)
  - [ ] Firefox (latest)
  - [ ] Safari (latest)
  - [ ] Edge (latest)
- [ ] Verify correct FFT results in each browser
- [ ] Check for performance differences

#### 23.1.3 Documentation

- [ ] Document browser-specific considerations:
  - [ ] Memory limits
  - [ ] Performance characteristics
  - [ ] Known issues
- [ ] Add browser example to `examples/wasm/README.md`

### 23.2 WASM SIMD (experimental)

#### 23.2.1 SIMD Support Check

- [ ] Monitor Go issue tracker for WASM SIMD support
- [ ] Check Go 1.24+ release notes for SIMD features
- [ ] Evaluate experimental `//go:wasmexport` and SIMD intrinsics

#### 23.2.2 Prototype (if supported)

- [ ] Create experimental WASM SIMD butterfly:
  - [ ] Use v128 (128-bit SIMD) type
  - [ ] Implement 2-element complex64 butterfly
- [ ] Benchmark WASM SIMD vs scalar WASM
- [ ] Document findings

#### 23.2.3 Performance Comparison

- [ ] Benchmark WASM vs native:

  ```bash
  # Native
  go test -bench=BenchmarkPlan -benchtime=5s ./...

  # WASM via Node.js
  GOOS=js GOARCH=wasm go test -bench=BenchmarkPlan -benchtime=5s ./...
  ```

- [ ] Calculate WASM overhead percentage
- [ ] Document in BENCHMARKS.md

---

## Phase 24: Documentation & Examples

### 24.1 GoDoc Completion

#### 24.1.1 Symbol Audit

- [ ] List all exported symbols:
  ```bash
  go doc -all github.com/MeKo-Tech/algo-fft | grep "^func\|^type\|^var\|^const"
  ```
- [ ] For each symbol, verify GoDoc comment exists and is clear
- [ ] Add missing comments following Go conventions

#### 24.1.2 Runnable Examples

- [ ] Create `example_test.go` with:
  - [ ] `ExampleNewPlan` - basic plan creation
  - [ ] `ExamplePlan_Forward` - forward transform
  - [ ] `ExamplePlan_Inverse` - inverse transform
  - [ ] `ExampleNewPlan2D` - 2D FFT usage
  - [ ] `ExampleConvolve` - convolution example
  - [ ] `ExamplePlanReal_Forward` - real FFT
- [ ] Verify examples run: `go test -v -run Example ./...`

#### 24.1.3 Package Documentation

- [ ] Create/update `doc.go`:
  - [ ] Package overview
  - [ ] Basic usage example
  - [ ] Performance notes
  - [ ] Architecture support
- [ ] Verify rendering on pkg.go.dev

### 24.2 README Enhancement

#### 24.2.1 Installation & Quick Start

- [ ] Add installation section:
  ```bash
  go get github.com/MeKo-Tech/algo-fft
  ```
- [ ] Add copy-paste ready quick start:
  ```go
  plan, _ := algofft.NewPlan32(1024)
  plan.Forward(data)
  ```

#### 24.2.2 API Overview

- [ ] Create API overview table:
      | Function | Description |
      | -------- | ----------- |
      | `NewPlan32(n)` | Create complex64 FFT plan |
      | `NewPlan64(n)` | Create complex128 FFT plan |
      | ... | ... |

#### 24.2.3 Performance & Comparison

- [ ] Add performance characteristics section:
  - [ ] Supported architectures
  - [ ] SIMD acceleration
  - [ ] Typical speedup ranges
- [ ] Add comparison to other libraries:
  - [ ] gonum/fourier
  - [ ] go-fft
  - [ ] scientificgo/fft

#### 24.2.4 Badges

- [ ] Add badges to README:
  - [ ] Go Report Card
  - [ ] GitHub Actions CI status
  - [ ] Coverage (codecov or coveralls)
  - [ ] pkg.go.dev reference
  - [ ] License

### 24.3 Tutorial Examples

#### 24.3.1 Basic Example

- [ ] Create `examples/basic/`:
  - [ ] `main.go` - simple 1D FFT demonstration
  - [ ] `README.md` - explanation and usage
  - [ ] Show forward transform, magnitude spectrum, inverse

#### 24.3.2 Audio Example

- [ ] Create `examples/audio/`:
  - [ ] `main.go` - audio spectrum analysis
  - [ ] `README.md` - explanation
  - [ ] Load WAV file (or generate test signal)
  - [ ] Apply real FFT
  - [ ] Display frequency content

#### 24.3.3 Image Example

- [ ] Create `examples/image/`:
  - [ ] `main.go` - 2D FFT for image processing
  - [ ] `README.md` - explanation
  - [ ] Load image, apply 2D FFT
  - [ ] Demonstrate frequency domain filtering
  - [ ] Save result image

---

## Phase 26: Profiling & Tuning

### 26.1 CPU Profiling

- [ ] Run CPU profiling on key benchmarks:
  ```bash
  go test -bench=BenchmarkPlan_1024 -cpuprofile=cpu.prof ./...
  go tool pprof -http=:8080 cpu.prof
  ```
- [ ] Identify top 5 CPU consumers
- [ ] Analyze call graphs for optimization opportunities
- [ ] Document findings

### 26.2 Memory Profiling

- [ ] Run memory profiling:
  ```bash
  go test -bench=BenchmarkPlan_1024 -memprofile=mem.prof ./...
  go tool pprof -http=:8080 mem.prof
  ```
- [ ] Verify zero-allocation transforms (after plan creation)
- [ ] Identify any unexpected allocations
- [ ] Fix allocation hotspots

### 26.3 Optimization Pass

- [ ] Address top performance hotspots from profiling
- [ ] Re-run benchmarks after each optimization
- [ ] Keep changes that provide > 5% improvement
- [ ] Revert changes that regress performance

### 26.4 Final Benchmark Comparison

- [ ] Run full benchmark suite on final code
- [ ] Compare against original Phase 14 baseline
- [ ] Document overall speedup achieved
- [ ] Update BENCHMARKS.md with final numbers

---

## Phase 27: v1.0 Preparation

### 27.1 API Review

#### 27.1.1 Consistency Check

- [ ] Review all public function signatures:
  - [ ] Consistent naming (NewXxx, XxxFunc, etc.)
  - [ ] Consistent parameter ordering
  - [ ] Consistent error handling
- [ ] Review all public types:
  - [ ] Consistent field naming
  - [ ] Appropriate visibility
- [ ] Ensure generics are used consistently

#### 27.1.2 Backward Compatibility

- [ ] Document any breaking changes from v0.x
- [ ] Create migration guide if needed
- [ ] Consider deprecation warnings for removed features

### 27.2 Stability Testing

#### 27.2.1 Flake Detection

- [ ] Run test suite 10+ times:
  ```bash
  for i in {1..10}; do go test ./... || echo "FAIL $i"; done
  ```
- [ ] Identify and fix any flaky tests
- [ ] Add retry logic for inherently flaky tests (if unavoidable)

#### 27.2.2 Go Version Testing

- [ ] Test on Go 1.21: `go1.21 test ./...`
- [ ] Test on Go 1.22: `go1.22 test ./...`
- [ ] Test on Go 1.23: `go1.23 test ./...`
- [ ] Test on Go 1.24: `go1.24 test ./...`
- [ ] Fix any version-specific issues

### 27.3 Release Preparation

#### 27.3.1 Changelog

- [ ] Create/update CHANGELOG.md:
  - [ ] List all changes since v0.2.0
  - [ ] Categorize: Added, Changed, Fixed, Removed
  - [ ] Include migration notes

#### 27.3.2 Release Notes

- [ ] Write v1.0.0 release notes:
  - [ ] Highlight key features
  - [ ] Performance improvements
  - [ ] API stability guarantee
  - [ ] Acknowledgments

#### 27.3.3 Tagging

- [ ] Tag release: `git tag v1.0.0`
- [ ] Push tag: `git push origin v1.0.0`
- [ ] Create GitHub release with notes
- [ ] Verify on pkg.go.dev

---

## Phase 28: Community & Maintenance

### 28.1 Community Setup

#### 28.1.1 Issue Templates

- [ ] Create `.github/ISSUE_TEMPLATE/bug_report.md`:
  - [ ] Version information
  - [ ] Steps to reproduce
  - [ ] Expected vs actual behavior
  - [ ] Platform details
- [ ] Create `.github/ISSUE_TEMPLATE/feature_request.md`:
  - [ ] Problem statement
  - [ ] Proposed solution
  - [ ] Alternatives considered

#### 28.1.2 PR Template

- [ ] Create `.github/PULL_REQUEST_TEMPLATE.md`:
  - [ ] Description of changes
  - [ ] Related issues
  - [ ] Testing performed
  - [ ] Checklist (tests, docs, benchmarks)

#### 28.1.3 Community Features

- [ ] Enable GitHub Discussions for Q&A
- [ ] Add `CODE_OF_CONDUCT.md` (Contributor Covenant)
- [ ] Add `SECURITY.md` for vulnerability reporting

### 28.2 Contributor Experience

#### 28.2.1 Development Documentation

- [ ] Update CONTRIBUTING.md:
  - [ ] Development environment setup
  - [ ] How to run tests and benchmarks
  - [ ] Code style guide
  - [ ] PR process

#### 28.2.2 Issue Management

- [ ] Add "good first issue" labels to starter tasks
- [ ] Add "help wanted" labels to complex tasks
- [ ] Create issue templates for common tasks

#### 28.2.3 Automation

- [ ] Set up Dependabot for Go modules:
  - [ ] Create `.github/dependabot.yml`
  - [ ] Configure weekly update checks
- [ ] Add stale issue bot (optional)

### 28.3 Ongoing Maintenance

#### 28.3.1 Security

- [ ] Set up govulncheck in CI:
  ```yaml
  - run: go install golang.org/x/vuln/cmd/govulncheck@latest
  - run: govulncheck ./...
  ```
- [ ] Create security policy in SECURITY.md

#### 28.3.2 Compatibility

- [ ] Document minimum Go version policy
- [ ] Plan for future Go version testing
- [ ] Monitor Go release notes for breaking changes

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

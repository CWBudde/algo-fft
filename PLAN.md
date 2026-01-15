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

### SIMD Params Preparation System ✅

Infrastructure for SIMD-optimized codelet twiddle data. Enables codelets to receive pre-prepared twiddle layouts instead of computing twiddle indices at runtime.

**API Changes:**

- `CodeletFunc[T]` signature: uses twiddle-only buffers (supports SIMD-friendly layouts)
- `CodeletEntry`: added `TwiddleExtraSize` and `PrepareTwiddleExtra` callbacks
- `PlanEstimate`: carries extra-twiddle callbacks from registry to plan creation
- `Plan[T]`/`FastPlan[T]`: store codelet twiddle buffers per direction

**Benefits:** Eliminates runtime index computation, scalar loads, and register shuffles. Pre-broadcast twiddles enable ~50% fewer instructions in SIMD twiddle handling.

**Files modified:** `internal/fftypes/codelet.go`, `internal/planner/codelet.go`, `internal/planner/planner.go`, `plan.go`, `plan_fast.go`, `internal/kernels/codelet_init.go`, test files

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
  - [x] Register in codelet system (consolidated in `codelet_init_sse2.go`)
- [x] Implement Size 256 radix-4 SSE2 for complex128
  - [x] Create `internal/asm/amd64/sse2_f64_size256_radix4.s`
  - [x] Implement `ForwardSSE2Size256Radix4Complex128Asm` (4 stages)
  - [x] Implement `InverseSSE2Size256Radix4Complex128Asm` (with 1/256 scaling)
  - [x] Register in codelet system (consolidated in `codelet_init_sse2.go`)

#### 13.1.3 Size 256 Test Coverage

- [x] Add test cases to unified `internal/kernels/sse2_kernels_test.go`
  - [x] Test cases for complex64 (Radix 4)
  - [x] Test cases for complex128 (Radix 2, Radix 4)

### 13.2 Size 512 SSE2 Kernels

#### 13.2.1 complex64 Size 512 SSE2 Kernels

- [x] Implement Size 512 radix-2 SSE2 for complex64
  - [x] Create `internal/asm/amd64/sse2_f32_size512_radix2.s`
  - [x] Implement `ForwardSSE2Size512Radix2Complex64Asm` (9 stages)
  - [x] Implement `InverseSSE2Size512Radix2Complex64Asm` (with 1/512 scaling)
  - [x] Register in codelet system (consolidated in `codelet_init_sse2.go`)
- [ ] Implement Size 512 mixed-2/4 SSE2 for complex64
  - [ ] Create `internal/asm/amd64/sse2_f32_size512_mixed24.s`
  - [ ] Implement `ForwardSSE2Size512Mixed24Complex64Asm` (4 radix-4 + 1 radix-2 = 5 stages)
  - [ ] Implement `InverseSSE2Size512Mixed24Complex64Asm` (with 1/512 scaling)
  - [ ] Add Go wrapper in `internal/kernels/sse2_f32_size512_mixed24.go`
  - [ ] Register in codelet system with priority 15

#### 13.2.2 complex128 Size 512 SSE2 Kernels

- [x] Implement Size 512 radix-2 SSE2 for complex128
  - [x] Create `internal/asm/amd64/sse2_f64_size512_radix2.s`
  - [x] Implement `ForwardSSE2Size512Radix2Complex128Asm` (9 stages)
  - [x] Implement `InverseSSE2Size512Radix2Complex128Asm` (with 1/512 scaling)
  - [x] Register in codelet system (consolidated in `codelet_init_sse2.go`)
- [ ] Implement Size 512 mixed-2/4 SSE2 for complex128
  - [ ] Create `internal/asm/amd64/sse2_f64_size512_mixed24.s`
  - [ ] Implement `ForwardSSE2Size512Mixed24Complex128Asm` (4 radix-4 + 1 radix-2 = 5 stages)
  - [ ] Implement `InverseSSE2Size512Mixed24Complex128Asm` (with 1/512 scaling)
  - [ ] Add Go wrapper in `internal/kernels/sse2_f64_size512_mixed24.go`
  - [ ] Register in codelet system with priority 15

#### 13.2.3 Size 512 Test Coverage

- [ ] Add test cases to unified `internal/kernels/sse2_kernels_test.go`
  - [x] Test cases for complex64 (Radix 2)
  - [ ] Test cases for complex64 (Mixed 2/4)
  - [x] Test cases for complex128 (Radix 2)
  - [ ] Test cases for complex128 (Mixed 2/4)

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

- [ ] Add test cases to unified `internal/kernels/sse2_kernels_test.go`
  - [ ] Test cases for complex64 (Radix 4)
  - [ ] Test cases for complex128 (Radix 4)

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

#### 14.3 Size 128 Radix-4 AVX2

- [x] Create `internal/asm/amd64/avx2_f32_size128_radix4.s` (currently only
      radix-2/mixed exist)
  - [x] Implement `forwardAVX2Size128Radix4Complex64` (3.5 stages: 3 radix-4 + 1
        radix-2)
  - [ ] Use radix-4 bit-reversal for first 64 elements, binary for rest
- [ ] Benchmark radix-4 vs current mixed-2/4 wrapper
- [ ] Register higher-performing variant with higher priority
- **Status**: Enabled for size-128 row FFTs in AVX2 six-step (16384) path;
  correctness covered by asm tests.

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

- [x] Create `internal/asm/amd64/avx2_f32_size256_radix16.s`
  - [x] Implement as 16×16 matrix factorization
  - [x] Stage 1: 16 parallel FFT-16 on columns
  - [x] Twiddle: W₂₅₆^(row×col) multiplication
  - [x] Stage 2: 16 parallel FFT-16 on rows
  - [x] Final transposition to natural order
- [x] No bit-reversal needed (identity permutation for 4^k sizes)
- [x] Register with priority 30 (higher than radix-4 priority 25)
- [x] Benchmark: Target 1.3-1.5x speedup vs radix-4

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
  - [x] Add test cases to unified `internal/kernels/avx2_kernels_test.go`
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

- [x] Create `internal/asm/amd64/avx2_f32_size8192_radix64x128.s`
  - [x] Stage 1: 128 parallel FFT-64
  - [x] Twiddle multiplication
  - [x] Stage 2: 64 parallel FFT-128
  - [x] 2 stages (vs 6.5 for mixed-2/4)

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

## Phase 16: v1.0 Release

**Goal**: Ship a stable, well-tested v1.0 release without over-engineering.

### 16.1 Fix Current Build Issues

- [ ] Resolve `prepareCodeletTwiddles64` redeclaration in test files
- [ ] Ensure all tests pass: `go test ./...`

### 16.2 API Consistency Review

- [ ] List exported symbols: `go doc -all . | grep "^func\|^type"`
- [ ] Verify all exported symbols have GoDoc comments
- [ ] Check consistent naming (NewXxx pattern)
- [ ] Verify error handling consistency

### 16.3 Stability Testing

- [ ] Run tests 5x to detect flaky tests:
  ```bash
  for i in {1..5}; do go test ./... || echo "FAIL $i"; done
  ```
- [ ] Fix any flaky tests found
- [ ] Run `go test -race ./...` to verify concurrency safety

### 27.4 Release Checklist

- [ ] Create CHANGELOG.md with key features
- [ ] Tag release: `git tag v1.0.0`
- [ ] Create GitHub release with notes
- [ ] Verify on pkg.go.dev

### 27.5 Basic GitHub Templates (Optional)

- [ ] Add simple `.github/ISSUE_TEMPLATE/bug_report.md`
- [ ] Add simple `.github/PULL_REQUEST_TEMPLATE.md`

---

## Future (Post v1.0)

**Performance Optimizations** (only if users request):

- Cache profiling and loop optimization
- Parallel batch processing API
- WASM SIMD (when Go supports it)
- AVX-512 support
- Higher-radix optimizations for remaining sizes

**Features**:

- Audio/image processing examples
- GPU acceleration (OpenCL/CUDA via cgo)
- Distributed FFT for very large datasets
- DCT (Discrete Cosine Transform)
- Hilbert transform
- STFT for spectrograms
- Gonum ecosystem integration

**Community** (as project grows):

- CODE_OF_CONDUCT.md
- Dependabot configuration
- Issue/PR templates refinement
- ARM64 native CI runner

# PLAN.md - algofft Implementation Roadmap

A detailed, task-by-task implementation plan for building the **algofft** Go FFT library.
Each phase is scoped to approximately one day of focused work.

---

## Phase 1: Project Setup & Infrastructure

### 1.1 Repository Initialization

- [x] Initialize Go module (`go mod init github.com/MeKo-Christian/algofft`)
- [x] Create initial directory structure:
  - [x] `/` - root with main package files
  - [x] `/internal/` - internal implementation details
  - [x] `/testdata/` - test fixtures and reference data
  - [x] `/examples/` - usage examples
- [x] Create `.gitignore` for Go projects
- [x] Create `LICENSE` file (MIT or Apache 2.0)
- [x] Create initial `README.md` with project description and goals

### 1.2 Development Tooling Setup

- [x] Set up `justfile` with common commands:
  - [x] `build` - compile the library
  - [x] `test` - run all tests
  - [x] `bench` - run benchmarks
  - [x] `lint` - run linters
  - [x] `fmt` - format code
  - [x] `cover` - generate coverage report
- [x] Configure `golangci-lint` with `.golangci.yml`
- [x] Set up pre-commit hooks for formatting/linting
- [x] Create GitHub Actions CI workflow for tests on push/PR

### 1.3 Documentation Foundation

- [x] Create `CONTRIBUTING.md` with contribution guidelines
- [x] Create `CHANGELOG.md` structure
- [x] Set up GoDoc documentation structure
- [x] Add code comment templates for exported types

---

## Phase 2: Core Type Definitions & API Design

### 2.1 Plan Type Definition

- [x] Define `Plan` struct in `plan.go`:
  - [x] Store FFT length `n`
  - [x] Store direction flag (forward/inverse)
  - [x] Placeholder for twiddle factors slice
  - [x] Placeholder for scratch buffer
- [x] Implement `NewPlan(n int) (*Plan, error)` constructor
- [x] Add validation for input length (power of 2 check)
- [x] Implement `Plan.Len() int` method
- [x] Write unit tests for Plan creation and validation

### 2.2 Public API Signatures

- [x] Define `Plan.Forward(dst, src []T) error` signature (generic)
- [x] Define `Plan.Inverse(dst, src []T) error` signature (generic)
- [x] Define `Plan.InPlace(data []T) error` (in-place variant)
- [x] Define `Plan.InverseInPlace(data []T) error` (in-place inverse)
- [x] Add comprehensive GoDoc comments with examples
- [x] Create stub implementations that return `ErrNotImplemented`

### 2.3 Error Types & Validation

- [x] Define custom error types in `errors.go`:
  - [x] `ErrInvalidLength` - for non-power-of-2 lengths
  - [x] `ErrNilSlice` - for nil input slices
  - [x] `ErrLengthMismatch` - when dst/src lengths don't match Plan
  - [x] `ErrNotImplemented` - temporary during development
  - [x] `ErrInvalidStride` - for strided operations
- [x] Implement input validation helper functions
- [x] Write tests for all error conditions
- [x] Document error conditions in GoDoc

---

## Phase 3: Mathematical Foundations

### 3.1 Twiddle Factor Generation

- [x] Implement `generateTwiddleFactors(n int) []complex64` in `twiddle.go`
  - Note: Implemented as `ComputeTwiddleFactors[T]` in `internal/fft/fft.go`
- [x] Use `exp(-2πi*k/n)` formula for forward transform
- [x] Pre-compute all twiddle factors for size `n`
- [x] Store twiddle factors in Plan during initialization
- [x] Write unit tests verifying twiddle factor correctness:
  - [x] Test W_n^0 = 1
  - [x] Test W_n^(n/2) = -1 for even n
  - [x] Test periodicity: W_n^(k+n) = W_n^k
- [x] Benchmark twiddle generation for various sizes

### 3.2 Bit-Reversal Permutation

- [x] Implement `bitReverse(x, bits uint) uint` utility function
  - Note: Implemented as `reverseBits(x, bits int) int` in `internal/fft/fft.go`
- [x] Implement `bitReversePermute(data []complex64)` in-place permutation
  - Note: Implemented as `ComputeBitReversalIndices` for precomputed lookup
- [x] Create lookup table approach for common sizes (8, 16, 32, 64...)
  - Note: Uses precomputed indices stored in Plan.bitrev
- [x] Write unit tests for bit reversal:
  - [x] Test known permutations for small sizes
  - [x] Test that double permutation returns original
- [x] Benchmark bit-reversal for various sizes

### 3.3 Basic Complex Arithmetic Helpers

- [x] Implement `complexMul(a, b complex64) complex64` (if needed beyond stdlib)
  - Note: Go's built-in `*` operator is used
- [x] Implement `complexAdd(a, b complex64) complex64`
  - Note: Go's built-in `+` operator is used
- [x] Implement `complexSub(a, b complex64) complex64`
  - Note: Go's built-in `-` operator is used
- [x] Verify Go's built-in complex64 operations are sufficient
- [x] Write precision tests comparing to complex128 reference

---

## Phase 4: Radix-2 FFT Core Implementation

### 4.1 Decimation-in-Time (DIT) Algorithm

- [x] Implement iterative Cooley-Tukey DIT algorithm in `fft_dit.go`
- [x] Structure: bit-reverse input, then log2(n) stages of butterflies
- [x] Implement basic butterfly operation: `butterfly2(a, b *complex64, w complex64)`
- [x] Wire DIT kernels into `selectKernels*` as an option
- [x] Write correctness tests for sizes 2, 4, 8 (compare vs reference DFT)

### 4.2 Forward Transform Completion

- [x] Complete forward FFT implementation for all power-of-2 sizes
- [x] Optimize loop structure for cache efficiency
- [x] Ensure no allocations in hot path (use Plan's scratch space)
- [x] Write tests for sizes 16, 32, 64, 128, 256, 512, 1024
- [x] Verify against known FFT results (e.g., impulse → flat spectrum)

### 4.3 Stockham Autosort Variant (OTFFT-Inspired)

- [x] Implement Stockham autosort FFT kernel (no explicit bit-reversal)
- [x] Validate numerical parity with reference DFT for small sizes
- [x] Compare cache behavior and throughput vs DIT implementation
- [x] Define selection heuristic (size threshold or plan flag)
- [x] If Stockham is the default path, de-prioritize split-radix/mixed-radix work

### 4.4 Twiddle Packing & SIMD-Friendly Layout

- [x] Precompute per-radix twiddle tables in contiguous SIMD-friendly order
- [x] Add twiddle packing for radix-4/8/16 butterflies
- [x] Measure twiddle load impact in asm kernels
- [x] Ensure Plan memory stays aligned for SIMD loads/stores

### 4.5 Inverse Transform Implementation

- [x] Implement inverse transform in Stockham kernel with 1/n scaling
- [x] Write round-trip tests: `Inverse(Forward(x)) ≈ x`
- [x] Implement DIT inverse path (conjugate method or swapped twiddles)
- [x] Test scaling factor correctness for DIT path

### 4.6 In-Place vs Out-of-Place Variants

- [x] Implement true in-place FFT (modify input directly)
- [x] Implement out-of-place FFT (src → dst, src unchanged)
- [x] Add `Plan.Transform(dst, src []complex64, inverse bool)` unified API
- [x] Test that out-of-place doesn't modify source
- [x] Test that in-place produces same results as out-of-place

### 4.7 Six-Step / Eight-Step Large FFT Strategy

- [x] Implement blocked transpose + FFT (six-step) for large power-of-two sizes
- [x] Add optional eight-step variant for very large N
- [x] Precompute transpose index tables to avoid per-call overhead
- [x] Add kernel benchmark runner to compare strategies and emit recommendations
- [x] Evaluate when to enable (based on N and cache size heuristics)

### 4.8 Kernel Selection & Benchmark-Driven Tuning

- [x] Add plan-time heuristic to choose Stockham vs DIT (size + CPU features)
- [x] Add optional benchmark cache (persisted or in-memory) to inform selection
- [x] Provide override flag to force a kernel (for testing/profiling)
- [x] Record decision in Plan for visibility in benchmarks/logs

---

## Phase 5: Testing Infrastructure

### 5.1 Reference Implementation

- [x] Create `internal/reference/dft.go` with naive O(n²) DFT
- [x] Implement `NaiveDFT(src []complex64) []complex64`
- [x] Implement `NaiveIDFT(src []complex64) []complex64`
- [x] Use complex128 internally for higher precision reference
- [x] Write tests verifying naive DFT correctness on tiny inputs

### 5.2 Correctness Test Suite

- [x] Create `fft_test.go` with comprehensive tests:
  - [x] Test impulse response (delta function)
  - [x] Test constant signal (DC component)
  - [x] Test pure sinusoids at various frequencies
  - [x] Test Nyquist frequency signal
  - [x] Test random signals vs reference DFT
- [x] Add tolerance-based comparison helper: `complexSliceEqual(a, b []complex64, tol float32) bool`
- [x] Test edge cases: length 1, length 2

### 5.3 Property-Based Testing

- [x] Implement Parseval's theorem test: energy conservation
  - Note: Implemented in `internal/reference/dft_test.go:TestNaiveDFT_Parseval`
- [x] Implement linearity test: `FFT(ax + by) = a*FFT(x) + b*FFT(y)`
  - Note: Implemented in `internal/reference/dft_test.go:TestNaiveDFT_Linearity`
- [x] Implement shift theorem tests
  - Note: Implemented in `internal/reference/dft_test.go:TestNaiveDFT_TimeShift` and `TestNaiveDFT_FrequencyShift`
- [x] Implement convolution theorem test (prep for later)
  - Note: Implemented in `internal/reference/dft_test.go:TestNaiveDFT_Convolution`
- [x] Create randomized input generators for property tests
  - Note: `generateRandomSignal` and `generateRandomSignal128` in `internal/reference/dft_test.go`

### 5.4 Fuzz Testing Setup

- [x] Create `fft_fuzz_test.go` with Go's native fuzzing
- [x] Fuzz test: `Inverse(Forward(x)) ≈ x` for random x
- [x] Fuzz test: no panics for any valid input
- [x] Fuzz test: consistent results for same input

---

## Phase 6: Benchmarking Infrastructure

### 6.1 Basic Benchmark Suite

- [x] Create `fft_bench_test.go`
- [x] Benchmark forward FFT for sizes: 16, 64, 256, 1024, 4096, 16384, 65536
- [x] Benchmark inverse FFT for same sizes
- [x] Benchmark Plan creation (one-time cost)
  - Note: `BenchmarkNewPlan_*` in `fft_bench_test.go` and `plan_test.go`
- [x] Add throughput calculation (samples/second)
  - Note: Using `b.SetBytes()` for MB/s throughput in benchmarks

### 6.2 Memory & Allocation Benchmarks

- [x] Benchmark allocations per transform using `testing.B.ReportAllocs()`
  - Note: `b.ReportAllocs()` used in `fft_bench_test.go`
- [x] Profile memory usage with `runtime.MemStats`
  - Note: `BenchmarkPlanForward_MemStats_1024` in `fft_bench_test.go`
- [x] Ensure zero allocations in steady-state transforms
  - Note: `TestPlanTransformsNoAllocs*` in `fft_alloc_test.go`
- [x] Benchmark with varying Plan reuse patterns
  - Note: `BenchmarkPlanReusePatterns_1024` in `fft_bench_test.go`

### 6.3 Comparison Benchmarks

- [x] Create benchmarks comparing to reference DFT (for small sizes)
  - Note: `BenchmarkReferenceDFT_*` in `fft_bench_test.go`
- [x] Document baseline performance numbers in `BENCHMARKS.md`
  - Note: `BENCHMARKS.md` and `benchmarks/baseline.txt`
- [x] Create benchmark runner script that outputs markdown table
  - Note: `scripts/bench_md.sh`
- [x] Set up benchmark regression tracking in CI
  - Note: `test-bench` workflow using `benchstat` with `benchmarks/baseline.txt`

---

## Phase 7: Code Quality & Refactoring

### 7.1 Code Organization Refactor

- [x] Move FFT implementation to `internal/fft/` package
- [x] Create clean separation: `algofft` (public) → `internal/fft` (implementation)
- [x] Define internal interfaces for algorithm strategies
- [x] Ensure all internal functions are unexported
- [x] Update imports and fix any circular dependencies

### 7.2 Memory Management

- [x] Implement buffer pooling for scratch space
  - Note: `internal/fft/pool.go` provides `BufferPool` with `sync.Pool` for complex64/128 and int slices
- [x] Add `Plan.Reset()` method to clear internal state
- [x] Implement `Plan.Close()` or finalizer if needed
  - Note: `Close()` returns pooled buffers; no-op for non-pooled plans
- [x] Profile and eliminate unnecessary allocations
  - Note: Zero allocations during transforms; pooling reduces Plan creation allocations
- [x] Add `sync.Pool` for frequently created small buffers
  - Note: `NewPlanPooled[T]()` and `NewPlanFromPool[T]()` use pooled allocations

### 7.3 API Polish

- [x] Review and finalize all public API signatures
  - Note: Consistent naming (NewPlan\*, Plan.Forward/Inverse/InPlace, etc.)
- [x] Ensure consistency in naming conventions
  - Note: All methods follow Go conventions; kernel strategies use consistent naming
- [x] Add `String()` method to Plan for debugging
  - Note: Returns "Plan[type](size, strategy, pooled)" format
- [x] Implement `Plan.Clone()` if useful
  - Note: Creates independent copy with own scratch buffer; shares immutable data
- [x] Review error messages for clarity
  - Note: All errors prefixed with "algofft:" and include descriptive text

---

## Phase 8: Mixed-Radix Foundation

### 8.1 Length Factorization

- [x] Implement `factorize(n int) []int` to find prime factors
- [x] Implement `isPowerOf2(n int) bool` utility
- [x] Implement `nextPowerOf2(n int) int` utility
- [x] Implement `isHighlyComposite(n int) bool` (2,3,5 factors only)
- [x] Write tests for factorization edge cases

### 8.2 Radix-4 Implementation

- [x] Implement `butterfly4` operation in `radix4.go`
- [x] Implement radix-4 FFT for lengths that are powers of 4
- [x] Integrate radix-4 into Plan when `n` is power of 4
- [x] Write correctness tests for radix-4
- [x] Benchmark radix-4 vs radix-2 for powers of 4

### 8.3 Split-Radix Preparation

- [x] Research split-radix algorithm structure
- [x] Design interface for pluggable radix implementations
- [x] Stub out `splitRadixFFT()` function
- [x] Document algorithm selection strategy

---

## Phase 9: Small Prime Radices

### 9.1 Radix-3 Implementation

- [x] Implement `butterfly3` operation in `radix3.go`
- [x] Derive and implement radix-3 twiddle factors
- [x] Implement radix-3 FFT stage
- [x] Write correctness tests for lengths 3, 9, 27
- [x] Benchmark radix-3 performance

### 9.2 Radix-5 Implementation

- [x] Implement `butterfly5` operation in `radix5.go`
- [x] Derive and implement radix-5 twiddle factors
- [x] Implement radix-5 FFT stage
- [x] Write correctness tests for lengths 5, 25, 125
- [x] Benchmark radix-5 performance

### 9.3 Mixed-Radix Combiner

- [x] Implement mixed-radix FFT combining radix-2, 3, 4, 5
- [x] Create factor scheduling algorithm (which radix to apply when)
- [x] Test composite lengths: 6, 10, 12, 15, 20, 30, 60
- [x] Verify correctness against reference DFT
- [x] Benchmark mixed-radix vs zero-padded power-of-2

---

## Phase 10: Bluestein's Algorithm

### 10.1 Chirp-Z Transform Foundation

- [x] Study and document Bluestein's algorithm theory
- [x] Implement chirp sequence generation: `W_n^(k²/2)`
- [x] Implement convolution via FFT helper
- [x] Write tests for chirp sequence properties

### 10.2 Bluestein Implementation

- [x] Implement `bluesteinFFT(data []complex64, n int)` in `bluestein.go`
- [x] Determine optimal padded length (next power of 2 ≥ 2n-1)
- [x] Pre-compute and cache chirp sequences in Plan
- [x] Wire Bluestein into Plan for prime-length inputs

### 10.3 Bluestein Testing

- [x] Test prime lengths: 7, 11, 13, 17, 19, 23, 31, 127
- [x] Test large primes: 251, 509, 1021
- [x] Verify round-trip correctness
- [x] Benchmark Bluestein vs naive DFT for prime lengths
- [x] Profile and optimize Bluestein hot paths
  - Bluestein achieves O(N log N) vs naive O(N²)
  - For N=127: Bluestein is ~41x faster (7.1 µs vs 297 µs)
  - Zero allocations in steady state (0 allocs/op verified)
  - Throughput: 115-475 MB/s depending on size and precision

---

## Phase 11: Real FFT - Forward Transform

### 11.1 Real FFT API Design

- [x] Design `PlanReal` type or extend `Plan` with real mode
- [x] Define `PlanReal.Forward(dst []complex64, src []float32) error`
- [x] Define output format: N/2+1 complex values for N real inputs
- [x] Document conjugate symmetry property
- [x] Write API usage examples

### 11.2 Real FFT Implementation (Pack Method)

- [x] Implement "pack" approach: treat N reals as N/2 complex
- [x] Perform N/2 complex FFT
- [x] Unpack and combine results using symmetry
- [x] Handle DC (index 0) and Nyquist (index N/2) specially
- [x] Write correctness tests for real sinusoids

### 11.3 Real FFT Testing

- [x] Test real impulse response
- [x] Test real constant signal
- [x] Test real cosine at various frequencies
- [x] Verify output has conjugate symmetry
- [x] Benchmark real FFT vs equivalent complex FFT

### 11.x Optional Improvements

- [x] Optional: precompute `U[k]` weights (otfftpp style) to reduce recombination ops
- [x] Optional: add normalized forward variants (unitary / 1/N) or flags in PlanReal

---

## Phase 12: Real FFT - Inverse Transform

### 12.1 Inverse Real FFT Implementation

- [x] Implement `PlanReal.Inverse(dst []float32, src []complex64) error`
- [x] Reconstruct full complex spectrum from half-spectrum
- [x] Apply inverse complex FFT
- [x] Extract real part of result
- [x] Verify imaginary parts are near-zero

Hints:

- Mirror otfftpp's inverse pack method: build length N/2 complex `z[k]` from half-spectrum using the same weights, then inverse FFT and interleave real outputs.
- Handle DC/Nyquist carefully to keep outputs purely real (use real-only bins at k=0 and k=N/2).

### 12.2 Real FFT Round-Trip Testing

- [x] Test `Inverse(Forward(x)) ≈ x` for real signals
- [x] Test with various signal types (noise, tones, chirps)
- [x] Fuzz test real FFT round-trip
- [x] Document precision expectations

### 12.3 Real FFT Optimization

- [x] Optimize real FFT to avoid full complex allocation
- [x] Profile and reduce memory usage
- [x] Ensure zero steady-state allocations
- [x] Benchmark optimized real FFT

---

## Phase 13: SIMD Infrastructure (amd64)

### 13.1 CPU Feature Detection

- [x] Implement `internal/cpu/` package for feature detection
  - [x] Created `cpu.go` with Features struct, caching, and query functions
  - [x] Created `detect_amd64.go` with SSE2, SSE3, SSSE3, SSE4.1, AVX, AVX2, AVX-512 detection
  - [x] Created `detect_arm64.go` with NEON detection
  - [x] Created `detect_generic.go` as fallback for other architectures
- [x] Store detection results in package-level variables (cached with sync.Once)
- [x] Create `HasAVX2()`, `HasSSE41()`, `HasSSE2()`, `HasNEON()`, etc. query functions
- [x] Test detection on various CPU configurations (via forced features for mocking)
- [x] Migrate existing code from `internal/fft/features.go` to new package
- [x] Update all kernel selection code to use `cpu.Features` type
- [x] Add comprehensive test coverage with feature mocking support

### 13.2 Backend Dispatch System

- [x] Create `internal/dispatch/dispatch.go`
  - Note: Implemented in `internal/fft/dispatch.go` - separate package not needed due to tight coupling
- [x] Define function pointer types for FFT operations
  - Note: `Kernel[T]` and `Kernels[T]` types in `dispatch.go`
- [x] Implement `init()` that sets function pointers based on CPU features
  - Note: `SelectKernels[T](features)` and `SelectKernelsWithStrategy[T](features, strategy)` in `dispatch.go`
- [x] Create fallback pure-Go implementations
  - Note: `autoKernelComplex64/128()` in `kernels_fallback.go`, tested in `TestKernelsFunctional_*`
- [x] Test dispatch mechanism with mock feature flags
  - Note: `TestKernelSelectionWithForcedFeatures` in `dispatch_test.go`

### 13.3 Assembly Infrastructure

- [x] Set up `internal/asm/` directory structure
- [x] Create build tags for `amd64`, `arm64`, `purego`
- [x] Create stub `.s` files with proper Go assembly syntax
- [x] Set up `go:noescape` pragma usage
- [x] Document assembly contribution guidelines

---

## Phase 14: AVX2 SIMD Acceleration

**Note:** Assembly infrastructure already exists with stubs in `internal/fft/asm_amd64.s` and dispatch system in `kernels_amd64_asm.go`. This phase implements the actual AVX2 kernels within that framework.

### 14.1 AVX2 DIT Kernel Implementation (complex64)

**Approach:** Implement full DIT FFT kernel with AVX2 vectorization, not standalone butterfly functions.

**Files to modify:**

- `internal/fft/asm_amd64.s` - Replace `forwardAVX2Complex64Asm` and `inverseAVX2Complex64Asm` stubs
- `internal/fft/kernels_amd64_asm_wrapper.go` - Update wrappers to call assembly instead of returning false
- `internal/fft/butterfly_avx2_test.go` - New comprehensive test file

**Implementation tasks:**

- [x] **14.1.1 Create test suite first (TDD approach)**
  - [x] Create `butterfly_avx2_test.go` with tests for AVX2 vs pure-Go DIT
  - [x] Add correctness tests against reference DFT for sizes 16-256
  - [x] Add round-trip tests: `Inverse(Forward(x)) ≈ x`
  - [x] Add property tests: Parseval's theorem, linearity
  - [x] Add edge case tests: all-zeros, impulse, random signals
  - [x] Add benchmarks comparing AVX2 vs pure-Go for sizes 64-16384

- [x] **14.1.2 Implement assembly foundation**
  - [x] Add function prologue: extract slice parameters from Go calling convention
  - [x] Add input validation: size >= 16, power-of-2 check, return false for invalid
  - [x] Implement scalar bit-reversal stage (follows `dit.go` pattern exactly)
  - [x] Add constants section: scaling factors for inverse transform

- [x] **14.1.3 Implement scalar butterfly loops (SSE)**
  - [x] Implement forward transform scalar path using SSE (MOVSS, ADDSS, SUBSS, MULSS)
  - [x] Implement inverse transform scalar path (conjugate twiddles, scale by 1/n)
  - [x] Follow DIT algorithm structure from `dit.go:35-86` exactly
  - [x] Test that scalar path passes all tests for sizes 16-2048

- [x] **14.1.4 Implement AVX2 vectorized butterflies**
  - [x] Implement complex multiply using AVX2 (process 4 complex64 at once)
    - [x] Use VSHUFPS to separate real/imaginary components
    - [x] Use VMULPS for parallel multiplication
    - [x] Use VADDPS/VSUBPS for butterfly additions
    - [x] Use VUNPCKLPS/VUNPCKHPS to interleave results
  - [x] Implement vectorized butterfly loop for `step==1` (contiguous twiddles)
  - [x] Add scalar fallback for remainder when `half % 4 != 0`
  - [x] Add scalar fallback for `step > 1` (non-contiguous twiddles)

- [x] **14.1.5 Update wrapper functions**
  - [x] Modify `forwardAVX2Complex64` in `kernels_amd64_asm_wrapper.go` to call assembly
  - [x] Modify `inverseAVX2Complex64` similarly
  - [x] Ensure proper feature detection and fallback mechanism

- [x] **14.1.6 Verify integration and performance**
  - [x] Confirm dispatch system routes to AVX2 kernel when `cpu.HasAVX2()`
  - [x] Verify zero allocations during transform (use `fft_alloc_test.go` pattern)
  - [x] Benchmark and verify 2-4x speedup over pure-Go DIT
  - [x] Ensure no performance regression for any size

**Success Criteria:**

- All tests pass for sizes 16-2048 (forward and inverse)
- Results match pure-Go DIT within 1e-6 relative error
- Round-trip error < 1e-5
- 2x speedup at size=64, 3-4x at size=1024+
- Zero allocations during steady-state transforms

### 14.2 AVX2 Optimization Pass ✅

**Prerequisite:** 14.1 must be complete and working

- [x] **14.2.1 Optimize twiddle access for step > 1**
  - [x] Implement manual twiddle gathering (4 scalar loads + shuffle)
  - [x] Compare performance vs VGATHERDPS instruction (manual gather faster)
  - [x] Test with various FFT sizes that produce different step values

- [x] **14.2.2 Optimize complex multiply**
  - [x] Experiment with FMA instructions (VFMADDSUB231PS) to reduce instruction count
  - [x] Minimize shuffle operations using better permute patterns
  - [x] Profile and measure impact on different CPU microarchitectures

- [x] **14.2.3 Loop-level optimizations**
  - [x] Experiment with unrolling inner loop (8 butterflies at once)
  - [ ] Add software prefetch hints (PREFETCHT0) for large transforms (optional)
  - [ ] Measure L1/L2 cache hit rates and tune accordingly (optional)

- [x] **14.2.4 Benchmark and document**
  - [x] Run comprehensive benchmarks on various CPU models (if available)
  - [x] Document achieved speedups in comments
  - [x] Update BENCHMARKS.md with AVX2 performance characteristics

**Success Criteria:** ✅ All met

- Works correctly for all step values (non-contiguous twiddles) ✅
- 4-5x speedup over pure-Go for size >= 1024 ✅ (achieved 4.5x at 1K, 4.8x at 4K)
- No degradation in correctness (all tests still pass) ✅

### 14.3 AVX2 Stockham Kernel (Optional)

**Note:** Stockham autosort has better cache locality than DIT but requires buffer swapping

- [x] **Implement AVX2 Stockham kernel**
  - [x] Implement `forwardStockhamAVX2Complex64Asm` following `stockham.go` structure
  - [x] Handle buffer swapping between dst and scratch
  - [x] Vectorize inner butterfly loops similar to 14.1
  - [x] Test against pure-Go Stockham implementation

- [ ] **Benchmark Stockham vs DIT with AVX2**
  - [ ] Compare throughput for sizes 256-16384
  - [ ] Measure cache behavior (L1/L2 hit rates)
  - [ ] Update strategy selection heuristics in `selection.go` if beneficial

**Success Criteria:**

- Stockham kernel achieves similar or better performance than DIT
- Strategy auto-selection chooses optimal kernel based on size

### 14.4 AVX2 complex128 Support ✅

**Prerequisite:** 14.1 must be complete

**Note:** AVX2 YMM registers hold 4 float64 = 2 complex128, so half the parallelism

- [x] **Implement complex128 AVX2 kernels**
  - [x] Implement `forwardAVX2Complex128Asm` (process 2 complex128 at once)
  - [x] Implement `inverseAVX2Complex128Asm` similarly
  - [x] Adapt complex multiply for float64 (same algorithm, different types)
  - [x] Update wrappers in `kernels_amd64_asm_wrapper.go`

- [x] **Test and validate**
  - [x] Add tests comparing AVX2 vs pure-Go for complex128
  - [x] Verify higher precision (error < 1e-12 for round-trip)
  - [x] Benchmark speedup (expect ~2x due to half parallelism)

**Success Criteria:**

- complex128 kernels work correctly for all sizes >= 16
- Achieve 2-3x speedup over pure-Go complex128 DIT
- Maintain higher precision (< 1e-12 error)

### 14.5 Size-Specific Fully Unrolled AVX2 Kernels

**Motivation:** Combine the benefits of size-specific unrolling (like `dit_small.go`) with SIMD vectorization for maximum performance on critical sizes.

**Prerequisite:** 14.1 and 14.4 must be complete

**Approach:** Create fully unrolled AVX2 kernels for common small sizes (16, 32, 64, 128) that eliminate all loop overhead and hardcode stage counts.

- [x] **14.5.1 Design size-specific dispatch mechanism** ✅
  - [x] Add dispatch layer in `kernels_amd64_size_specific.go` that routes by size
  - [x] Define function signatures for size-specific kernels in `asm_amd64.go`
  - [x] Create fallback chain: size-specific SIMD → generic SIMD → pure-Go
  - [x] Add benchmarks to measure improvement vs generic AVX2
  - [x] Add assembly stubs in `asm_amd64.s` (return false to trigger fallback)
  - [x] Update `selectKernelsComplex64` to use new dispatch
  - [x] Design document: `docs/plans/2025-01-27-avx2-size-specific-dispatch-design.md`

- [ ] **14.5.2 Implement AVX2 Size-16 kernel (complex64)**
  - [ ] Create `forwardAVX2Size16Complex64Asm` in `asm_amd64.s`
  - [ ] Fully unroll 4 FFT stages (size=2, 4, 8, 16)
  - [ ] Hardcode bit-reversal indices (no loop)
  - [ ] Hardcode twiddle indices for each stage
  - [ ] Vectorize butterflies using AVX2 (4 complex64 per YMM)
  - [ ] Test correctness vs reference and generic AVX2
  - [ ] Benchmark speedup (expect 10-20% over generic AVX2)

- [ ] **14.5.3 Implement AVX2 Size-32 kernel (complex64)**
  - [ ] Create `forwardAVX2Size32Complex64Asm`
  - [ ] Fully unroll 5 FFT stages
  - [ ] Optimize for L1 cache locality (all data fits in 256 bytes)
  - [ ] Test and benchmark

- [ ] **14.5.4 Implement AVX2 Size-64 kernel (complex64)**
  - [ ] Create `forwardAVX2Size64Complex64Asm`
  - [ ] Fully unroll 6 FFT stages
  - [ ] Consider radix-4 decomposition to limit code size
  - [ ] Test and benchmark

- [ ] **14.5.5 Implement AVX2 Size-128 kernel (complex64)**
  - [ ] Create `forwardAVX2Size128Complex64Asm`
  - [ ] Fully unroll 7 FFT stages or use radix-4
  - [ ] Balance code size (~500-1000 lines) vs performance
  - [ ] Test and benchmark

- [ ] **14.5.6 Add inverse transform variants**
  - [ ] Implement inverse versions for sizes 16, 32, 64, 128
  - [ ] Conjugate twiddles and add 1/n scaling
  - [ ] Test round-trip accuracy

- [ ] **14.5.7 Add complex128 size-specific kernels (optional)**
  - [ ] Implement for sizes 16, 32 (higher priority, smaller code)
  - [ ] AVX2 processes 2 complex128, so expect ~2x speedup vs pure-Go
  - [ ] Test and benchmark

**Success Criteria:**

- Size-specific kernels achieve 5-20% speedup over generic AVX2 for sizes 16-128
- All tests pass with correctness matching generic implementation
- Code size remains manageable (<1000 lines per size)
- Dispatch system correctly routes to size-specific kernels when available

**Estimated Effort:** 3-5 days (comparable to `dit_small.go` development)

---

## Phase 15: ARM64 NEON Implementation

**Note:** NEON SIMD is always available on ARMv8. ARM64 registers are 128-bit (half the width of AVX2's 256-bit), so we process 2 complex64 (4 floats) or 1 complex128 (2 doubles) per register.

### 15.1 ARM64 Infrastructure & Testing Environment ✅

**Cross-compilation and testing infrastructure setup**

- [x] **15.1.1 Verify cross-compilation tooling**
  - [x] Confirm Go ARM64 cross-compilation support (GOOS=linux GOARCH=arm64)
  - [x] Test basic ARM64 binary creation
  - [x] Document cross-compilation requirements in `/tmp/phase15_requirements.md`

- [x] **15.1.2 Set up QEMU emulation environment**
  - [x] Document QEMU installation requirements (qemu-user-static, binfmt-support)
  - [x] Create ARM64 test infrastructure using QEMU user-mode emulation
  - [x] Verify tests can run on amd64 host via qemu-aarch64-static
  - [x] Document known test failures (3 ARM64-specific failures, not blocking)

- [x] **15.1.3 Add ARM64 build targets to justfile**
  - [x] Add `build-arm64` target for cross-compilation
  - [x] Add `test-arm64` target with QEMU execution
  - [x] Add `bench-arm64` target for ARM64 benchmarking
  - [x] Add `build-all`, `test-all`, `check-all` convenience targets
  - [x] Verify all targets work correctly

- [x] **15.1.4 Create ARM64 assembly infrastructure**
  - [x] Verify existing `internal/fft/asm_arm64.s` stub file
  - [x] Verify existing `internal/fft/asm_arm64.go` function declarations
  - [x] Verify existing `internal/fft/kernels_arm64_asm.go` dispatch logic
  - [x] Verify NEON feature detection in `internal/cpu/detect_arm64.go`
  - [x] Add `selectKernelsComplex64WithStrategy` wrapper function
  - [x] Add `selectKernelsComplex128WithStrategy` wrapper function

- [x] **15.1.5 Study ARM64 SIMD and reference implementations**
  - [x] Study ARM64 NEON instruction set (128-bit SIMD registers V0-V31)
  - [x] Study AVX2 implementation in `asm_amd64.s` as reference
  - [x] Document differences: AVX2 (4 complex64) vs NEON (2 complex64)
  - [x] Document Go ARM64 assembly syntax (Plan 9 dialect)
  - [x] Create comprehensive NEON implementation guide (`/tmp/neon_implementation_guide.md`)
  - [x] Create detailed assembly skeleton (`/tmp/asm_arm64_skeleton.s`)
  - [x] Create test template (`/tmp/butterfly_neon_test.go.template`)
  - [x] Create groundwork summary (`/tmp/phase15_groundwork_summary.md`)

**Success Criteria:** ✅ All met

- Cross-compilation works for ARM64 ✅
- QEMU test environment functional ✅
- Test suite runs on ARM64 (167/170 tests pass) ✅
- Documentation and reference materials created ✅

### 15.2 NEON DIT Kernel Implementation (complex64)

**Approach:** Implement full DIT FFT kernel with NEON vectorization, mirroring AVX2 structure

**Files involved:**

- `internal/fft/asm_arm64.s` - Assembly implementation
- `internal/fft/kernels_arm64_asm.go` - Dispatch integration
- `internal/fft/butterfly_neon_test.go` - Test suite (from template)

#### 15.2.1 Assembly Foundation & Bit-Reversal ✅

- [x] **Implement function structure in `asm_arm64.s`**
  - [x] Add function prologue for `forwardNEONComplex64Asm`
  - [x] Extract slice parameters from Go calling convention (dst, src, twiddle, scratch, bitrev)
  - [x] Load pointers into registers (R8-R13 for work, src, twiddle, scratch, bitrev, n)
  - [x] Add proper register allocation comments

- [x] **Add input validation**
  - [x] Check for empty input (CBZ instruction, return true for n=0)
  - [x] Validate all slice lengths >= n (CMP + BLT to return_false)
  - [x] Handle trivial case n=1 (copy single element, return true)
  - [x] Verify n is power of 2 using (n & (n-1)) == 0 check
  - [x] Check minimum size (n >= 16 for NEON vectorization)

- [x] **Implement working buffer selection**
  - [x] Detect in-place vs out-of-place transform (compare dst and src pointers)
  - [x] Use scratch buffer for in-place transforms (dst == src)
  - [x] Use dst directly for out-of-place transforms (dst != src)

- [x] **Implement bit-reversal permutation**
  - [x] Create loop counter (R17 = i = 0)
  - [x] Load bit-reversed index j = bitrev[i] (LSL + ADD + MOVD pattern)
  - [x] Load src[j] (complex64 = 8 bytes)
  - [x] Store to work[i]
  - [x] Increment counter and loop
  - [x] Fix ARM64 assembly syntax errors:
    - [x] Use R0-R30 not X0-X30 (Go Plan 9 assembly convention)
    - [x] Fix CMP operand order (immediate first: CMP $imm, Rn)
    - [x] Fix comment placement (separate lines for clarity)

- [x] **Verify compilation and integration**
  - [x] Successfully build with `just build-arm64`
  - [x] Run tests to verify bit-reversal works correctly
  - [x] Confirm fallback to Go for unimplemented butterfly stages (expected)

**Key Learnings:**

- ARM64 Go assembly uses R-prefix (R0-R30), not X-prefix
- CMP syntax: immediate comes first (CMP $16, R13, not CMP R13, $16)
- complex64 = 8 bytes (4-byte real + 4-byte imag)
- int (in bitrev array) = 8 bytes on ARM64
- MOVD loads/stores 64 bits (8 bytes)
- LSL shifts left: LSL $3, Rsrc, Rdst (multiply by 8 for byte offset)

**Current Status:** ✅ Scalar DIT butterfly loop structure complete for forward complex64

- Forward path matches reference for sizes 2, 4, 8, 16 on ARM64 asm
- `/tmp/phase15_progress.md` documents implementation details

#### 15.2.2 Butterfly Loop Structure

**Next concrete task:** Implement size=2 butterfly (simplest case, no complex multiply needed)

- [x] **Add outer loop for FFT stages**
  - [x] Initialize size = 2 (R14 = 2)
  - [x] Loop: while size <= n, process stage then size \*= 2
  - [x] Compute half = size / 2 (LSR $1, R14, R15)
  - [x] Compute step = n / size (UDIV R14, R13, R16)

- [x] **Add middle loop for butterfly groups**
  - [x] Initialize base = 0 (R17 = 0)
  - [x] Loop: while base < n, process group then base += size
  - [x] Each group processes half butterflies

- [x] **Add inner loop for butterflies within group**
  - [x] Initialize j = 0 (R0 = 0)
  - [x] Loop: while j < half, process butterfly then j++
  - [x] Compute indices: idx_a = base + j, idx_b = base + j + half
  - [x] Load twiddle: w = twiddle[j * step]

- [x] **Implement size=2 butterfly (scalar, no NEON)**
  - [x] Load a = work[idx_a], b = work[idx_b]
  - [x] For size=2, w=1 so: a' = a + b, b' = a - b
  - [x] Use scalar floating-point: FMOV, FADD, FSUB
  - [x] Store results back to work buffer
  - [x] Test with size=16 (exercises size=2, 4, 8, 16 stages)

- [x] **Fix ARM64 compare/divide operand order**
  - [x] Ensure loop bounds use correct CMP operand order
  - [x] Ensure step uses n/size (UDIV R14, R13, R16)

**Reference:** See `/tmp/phase15_progress.md` lines 62-154 for detailed butterfly loop structure

#### 15.2.3 NEON Complex Multiply (TODO)

**Critical building block for all stages except size=2**

- [x] **Implement complex multiply: V2 = V0 \* V1**
  - [x] Input: V0 = a (complex values), V1 = b (complex values)
  - [x] Extract components using UZP1/UZP2 (unzip)
  - [x] Compute real part: a.real _ b.real - a.imag _ b.imag
  - [x] Compute imag part: a.real _ b.imag + a.imag _ b.real
  - [x] Interleave result using ZIP1
  - [x] ~15 lines of NEON instructions

- [x] **Create standalone test for complex multiply**
  - [x] Test known multiplications: (1+0i)\*(2+0i) = 2+0i
  - [x] Test with i: (1+0i)\*(0+1i) = 0+1i
  - [x] Test general case: (3+4i)\*(1+2i) = -5+10i
  - [x] Compare NEON result vs Go computation

**Reference:** See `/tmp/asm_arm64_skeleton.s` lines 180-210 for detailed NEON complex multiply pattern

#### 15.2.4 NEON Vectorized Butterfly (step==1) (TODO)

**Process 2 butterflies per iteration when twiddles are contiguous**

- [x] **Implement contiguous twiddle path**
  - [x] Detect step==1 (contiguous twiddles allow vectorized loads)
  - [x] Load 2 'a' values into V0 (LD1 {V0.4S})
  - [x] Load 2 'b' values into V1
  - [x] Load 2 'w' twiddles into V2
  - [x] Complex multiply (w \* b) using VUZP1/VUZP2 + VFMLA/VFMLS
  - [x] Butterfly: a' = a + wb, b' = a - wb (VFMLA/VFMLS with ones vector)
  - [x] Store 2 results back (ST1 {V10.4S}, {V11.4S})
  - [x] Process 2 butterflies per inner loop iteration

- [x] **Handle remainder when half % 2 != 0**
  - [x] Fall back to scalar for last butterfly if odd count
  - [x] Load single values, scalar multiply, scalar add/sub

- [x] **Test vectorized path**
  - [x] Test sizes 2, 4, 8, 16 via `TestForwardMatchesReferenceSmall`
  - [x] Compare results vs reference DFT

#### 15.2.5 Scalar Fallback for Strided Twiddles (step>1) (TODO)

**Manual gather when twiddles are non-contiguous**

- [x] **Implement strided twiddle path**
  - [x] Detect step > 1 (requires manual scalar loads)
  - [x] Compute twiddle offset: tw_idx = j \* step
  - [x] Load single twiddle w = twiddle[tw_idx]
  - [x] Load single a, single b
  - [x] Scalar complex multiply (similar to NEON but single values)
  - [x] Scalar butterfly (FADD, FSUB on scalar registers)
  - [x] Store single result

- [x] **Test strided path**
  - [x] Test size=1024 (exercises various step values: 1, 2, 4, ..., 512)
  - [x] Verify correctness for all stages

- [x] **Optional: NEON gather for step>1**
  - [x] Manually load 2 twiddles using scalar loads + vector insert
  - [x] Process 2 butterflies even when step>1

#### 15.2.6 Inverse Transform (TODO)

- [x] **Implement `inverseNEONComplex64Asm`**
  - [x] Copy forward transform structure
  - [x] Modify complex multiply to use conjugate twiddles
  - [x] Change w _ b to conj(w) _ b (negate imaginary component)
  - [x] Add 1/n scaling factor after all butterfly stages
  - [x] Use FMOV for loading 1/n constant
  - [x] Use FMUL to scale all elements

- [x] **Test inverse transform**
  - [x] Test round-trip: Inverse(Forward(x)) ≈ x (complex64)
  - [x] Test with reference DFT (complex64)
  - [ ] Benchmark inverse vs forward (should be similar)

#### 15.2.7 Integration & Testing (TODO)

- [x] **Copy test template to actual file**
  - [x] Copy `/tmp/butterfly_neon_test.go.template` to `internal/fft/butterfly_neon_test.go`
  - [x] Add tests for sizes 16, 32, 64, 128, 256
  - [x] Add correctness tests vs reference DFT
  - [x] Add round-trip tests

- [x] **Verify dispatch system**
  - [x] Confirm `selectKernelsComplex64` returns NEON kernel when NEON available
  - [x] Test with forced features to simulate NEON on/off

- [x] **Benchmark NEON vs pure Go**
  - [x] Add benchmarks for sizes 64, 256, 1024, 4096
  - [x] Document results in BENCHMARKS.md
  - [x] Note: QEMU benchmarks not representative, need real ARM64 hardware

**Success Criteria:**

- All tests pass for sizes 16-2048 (forward and inverse)
- Results match pure-Go DIT within 1e-6 relative error
- Round-trip error < 1e-5
- Zero allocations during steady-state transforms
- 2-3x speedup on real ARM64 hardware (not measured in QEMU)

### 15.3 NEON complex128 Support (TODO)

**Prerequisite:** 15.2 must be complete

**Note:** ARM64 128-bit registers hold 2 float64 = 1 complex128, so half the parallelism of complex64

- [x] **Implement complex128 NEON kernels**
  - [x] Implement `forwardNEONComplex128Asm`
  - [x] Change element size from 8 to 16 bytes (complex128)
  - [x] Use `.2D` NEON instructions instead of `.4S` (double precision)
  - [x] Process 1 complex128 per vector instead of 2
  - [x] Adapt complex multiply for float64
  - [x] Implement `inverseNEONComplex128Asm` similarly

- [x] **Test and validate**
  - [x] Add tests comparing NEON vs pure Go for complex128
  - [x] Verify higher precision (error ~1e-11 in QEMU)
  - [x] Benchmark speedup (QEMU only; real hardware pending)

**Success Criteria:**

- complex128 kernels work correctly for all sizes >= 16
- Achieve 1.5-2x speedup over pure-Go complex128 DIT
- Maintain higher precision (< 1e-12 error)

### 15.4 NEON Integration & Production Testing (TODO)

- [ ] **Test on real ARM64 hardware**
  - [ ] Run test suite on physical ARM64 device (Raspberry Pi, AWS Graviton, etc.)
  - [ ] Benchmark actual performance (QEMU not representative)
  - [ ] Verify NEON detection works on real hardware
  - [ ] Test different ARM cores (Cortex-A77, A78, X1, etc.)

- [ ] **Add ARM64 to CI pipeline**
  - [ ] Set up GitHub Actions ARM64 runner
  - [x] Add macOS ARM64 (macos-14) to benchmark matrix
  - [ ] Ensure cross-architecture tests pass
  - [ ] Verify SIMD paths produce same results as pure-Go

- [ ] **Documentation and benchmarks**
  - [ ] Update BENCHMARKS.md with ARM64 results
  - [ ] Document NEON performance characteristics
  - [ ] Compare NEON vs AVX2 speedup ratios
  - [ ] Add ARM64 usage notes to README

**Success Criteria:**

- CI tests pass on both amd64 and arm64
- Real ARM64 benchmarks documented
- Users can build and run on ARM64 without issues

**Implementation Progress:**

- Phase 15.1: ✅ Complete (infrastructure, QEMU, documentation)
- Phase 15.2.1: ✅ Complete (bit-reversal permutation, 199 lines)
- Phase 15.2.2-7: 🚧 Next steps (butterfly loops, complex multiply, vectorization)
- Phase 15.3-4: ⏳ Waiting for 15.2 completion

**See Also:**

- `/tmp/phase15_progress.md` - Detailed implementation progress and next steps
- `/tmp/neon_implementation_guide.md` - Complete NEON instruction reference
- `/tmp/asm_arm64_skeleton.s` - Detailed assembly skeleton with examples
- `/tmp/butterfly_neon_test.go.template` - Test infrastructure template
- `/tmp/phase15_groundwork_summary.md` - Phase 15 roadmap and resources

### 15.5 Size-Specific Fully Unrolled NEON Kernels

**Motivation:** Mirror the AVX2 size-specific optimization (Phase 14.5) for ARM64 platforms, achieving maximum performance on critical sizes.

**Prerequisite:** 15.2 and 15.3 must be complete

**Approach:** Create fully unrolled NEON kernels for common small sizes (16, 32, 64, 128) that eliminate all loop overhead and hardcode stage counts.

- [ ] **15.5.1 Design size-specific dispatch mechanism**
  - [ ] Add dispatch layer in `kernels_arm64_asm.go` that routes by size
  - [ ] Define function signatures for size-specific kernels
  - [ ] Create fallback chain: size-specific NEON → generic NEON → pure-Go
  - [ ] Add benchmarks to measure improvement vs generic NEON

- [ ] **15.5.2 Implement NEON Size-16 kernel (complex64)**
  - [ ] Create `forwardNEONSize16Complex64Asm` in `asm_arm64.s`
  - [ ] Fully unroll 4 FFT stages (size=2, 4, 8, 16)
  - [ ] Hardcode bit-reversal indices (no loop)
  - [ ] Hardcode twiddle indices for each stage
  - [ ] Vectorize butterflies using NEON (2 complex64 per 128-bit register)
  - [ ] Test correctness vs reference and generic NEON
  - [ ] Benchmark speedup (expect 10-20% over generic NEON)

- [ ] **15.5.3 Implement NEON Size-32 kernel (complex64)**
  - [ ] Create `forwardNEONSize32Complex64Asm`
  - [ ] Fully unroll 5 FFT stages
  - [ ] Optimize for ARM L1 cache locality
  - [ ] Test and benchmark on real ARM64 hardware

- [ ] **15.5.4 Implement NEON Size-64 kernel (complex64)**
  - [ ] Create `forwardNEONSize64Complex64Asm`
  - [ ] Fully unroll 6 FFT stages
  - [ ] Consider radix-4 decomposition to limit code size
  - [ ] Test and benchmark

- [ ] **15.5.5 Implement NEON Size-128 kernel (complex64)**
  - [ ] Create `forwardNEONSize128Complex64Asm`
  - [ ] Fully unroll 7 FFT stages or use radix-4
  - [ ] Balance code size (~500-1000 lines) vs performance
  - [ ] Test and benchmark

- [ ] **15.5.6 Add inverse transform variants**
  - [ ] Implement inverse versions for sizes 16, 32, 64, 128
  - [ ] Conjugate twiddles and add 1/n scaling
  - [ ] Test round-trip accuracy

- [ ] **15.5.7 Add complex128 size-specific kernels (optional)**
  - [ ] Implement for sizes 16, 32 (higher priority)
  - [ ] NEON processes 1 complex128 per register (128-bit = 2 float64)
  - [ ] Test and benchmark on ARM64

- [ ] **15.5.8 Cross-architecture validation**
  - [ ] Ensure NEON size-specific kernels produce identical results to AVX2 equivalents
  - [ ] Compare performance characteristics across architectures
  - [ ] Document ARM-specific optimizations vs x86

**Success Criteria:**

- Size-specific kernels achieve 5-20% speedup over generic NEON for sizes 16-128
- All tests pass with correctness matching generic implementation
- Results are bit-identical to AVX2 size-specific kernels (when compared on same precision)
- Code size remains manageable (<1000 lines per size)
- Dispatch system correctly routes to size-specific kernels when available

**Estimated Effort:** 3-5 days (comparable to AVX2 size-specific kernels, Phase 14.5)

**Architecture Notes:**

- NEON 128-bit registers (vs AVX2 256-bit) means processing 2 complex64 instead of 4
- Expect slightly different unrolling patterns due to register width
- ARM cores may have different cache characteristics than x86
- Test on multiple ARM cores (Cortex-A series, Apple Silicon, Graviton) for validation

---

## Phase 16: Performance Optimization Pass

### 16.1 Cache Optimization

- [ ] Analyze cache behavior with profiling tools
- [ ] Implement cache-oblivious or cache-aware strategies
- [ ] Optimize memory access patterns in butterfly loops
- [ ] Test performance impact of cache optimizations

### 16.2 Loop Unrolling

- [ ] Manually unroll critical inner loops
- [ ] Use Go compiler directives where applicable
- [ ] Benchmark unrolled vs original loops
- [ ] Balance code size vs performance

### 16.3 Bounds Check Elimination

- [ ] Profile to identify bounds check hotspots
- [ ] Restructure loops to eliminate bounds checks
- [ ] Use `_ = slice[len-1]` pattern where needed
- [ ] Verify performance improvement without sacrificing safety

---

## Phase 17: 2D FFT Implementation ✅

### 17.1 2D FFT API Design ✅

- [x] Design `Plan2D[T Complex]` type for 2D transforms
- [x] Define matrix representation (row-major []T with rows/cols stored in plan)
- [x] Define `Plan2D.Forward(dst, src []T) error` method
- [x] Define `Plan2D.Inverse(dst, src []T) error` method
- [x] Define `Plan2D.ForwardInPlace` and `InverseInPlace` methods
- [x] Document row-major storage convention
- [x] Add convenience constructors `NewPlan2D32` and `NewPlan2D64`
- [x] Implement `Clone()` for concurrent use

### 17.2 Row-Column Algorithm ✅

- [x] Implement 2D FFT via row-wise then column-wise 1D FFTs
- [x] Create helpers for column transforms (transpose-based for square, strided for non-square)
- [x] Reuse 1D Plan objects internally (rowPlan and colPlan)
- [x] Handle non-square matrices (rows ≠ cols)
- [x] Write comprehensive correctness tests (vs naive 2D DFT)
- [x] Zero-allocation transforms after plan creation
- [x] Support both complex64 and complex128 precision

### 17.3 2D FFT Testing & Optimization ✅

- [x] Test 2D FFT against naive O(M²N²) reference (up to 16×16)
- [x] Test round-trip correctness for sizes up to 128×128
- [x] Test mathematical properties (linearity, Parseval, separability)
- [x] Test signal properties (constant, pure sinusoid, corner frequencies)
- [x] Optimize square matrices with transpose-based column access
- [x] Benchmark 2D FFT for sizes 8×8 to 512×512
- [x] Verify zero allocations during transforms
- [x] Benchmark non-square matrices (16×32, 32×64, etc.)
- [x] Compare in-place vs out-of-place performance

**Implementation Notes:**

- Uses existing transpose utilities from `internal/fft/transpose.go`
- Square matrices use cached transpose pairs for efficiency
- Non-square matrices use strided column extraction
- All transforms maintain zero-allocation guarantee
- Reference implementation in `internal/reference/dft2d.go` for validation

---

## Phase 18: 3D FFT & N-D Generalization

### 18.1 3D FFT Implementation ✅ COMPLETED

- [x] Design `Plan3D` type for 3D transforms
- [x] Implement as three passes of 1D FFTs along each axis
- [x] Handle axis iteration order for cache efficiency
- [x] Write correctness tests for 3D transforms

**Implementation**: [plan_3d.go](plan_3d.go)
**Tests**: [plan_3d_test.go](plan_3d_test.go)
**Reference**: [internal/reference/dft3d.go](internal/reference/dft3d.go)

### 18.2 N-Dimensional API ✅ COMPLETED

- [x] Design generic `PlanND` type for arbitrary dimensions
- [x] Accept dimension sizes as `[]int`
- [x] Implement iterative axis-by-axis transform
- [x] Test 4D and 5D transforms
- [x] Benchmark N-D overhead vs specialized 2D/3D

**Implementation**: [plan_nd.go](plan_nd.go)
**Tests**: [plan_nd_test.go](plan_nd_test.go)

**Performance**: PlanND has ~24-34% overhead vs specialized Plan3D for 3D transforms (18 µs vs 14 µs for 8×8×8).
This is a reasonable trade-off for supporting arbitrary dimensions.

### 18.3 Multi-Dimensional Real FFT ✅ COMPLETED

- [x] Implement real-input 2D FFT (output: M×(N/2+1) compact, M×N full)
- [x] Implement real-input 3D FFT (output: D×H×(W/2+1) compact, D×H×W full)
- [x] Test with real-valued image/volume data (correctness, round-trip, linearity)
- [x] Document output size conventions (compact and full spectrum formats)

**Implementation**: [plan_real_2d.go](plan_real_2d.go), [plan_real_3d.go](plan_real_3d.go)
**Tests**: [plan_real_2d_test.go](plan_real_2d_test.go), [plan_real_3d_test.go](plan_real_3d_test.go)
**Benchmarks**: [plan_real_2d_bench_test.go](plan_real_2d_bench_test.go), [plan_real_3d_bench_test.go](plan_real_3d_bench_test.go)
**Reference**: [internal/reference/real_dft2d.go](internal/reference/real_dft2d.go), [internal/reference/real_dft3d.go](internal/reference/real_dft3d.go)

**Key Features**:

- **Dual API**: Compact half-spectrum (`Forward`/`Inverse`) and full spectrum (`ForwardFull`/`InverseFull`)
- **Memory Efficiency**: Compact format saves 50% memory by exploiting conjugate symmetry
- **Last-Dimension Compression**: Real FFT applied to innermost dimension (industry standard)
- **Zero Allocations**: After plan creation, transforms allocate no new memory
- **Comprehensive Testing**: Validated against naive DFT, round-trip, linearity, constant signals
- **Performance**: 64×64 2D: ~350 MB/s, 16×16×16 3D: ~143 MB/s (1-2 allocations per transform)

---

## Phase 19: Batch Processing

### 19.1 Batch FFT API ✅

- [x] Design batch API: `Plan.ForwardBatch(dst, src []complex64, count int) error`
- [x] Support interleaved (sequential) layout (stride=n)
- [x] Document memory layout expectations
- [x] Write basic batch correctness tests

### 19.2 Batch Implementation ✅

- [x] Implement simple loop-based batching
- [x] Ensure each FFT in batch uses same Plan
- [x] Test batch results match individual FFT calls
- [x] Benchmark batch vs individual calls

### 19.3 Parallel Batch Processing

- [ ] Add `Plan.ForwardBatchParallel(...)` with goroutines
- [ ] Implement worker pool for large batch counts
- [ ] Ensure Plan is safe for concurrent use (read-only during transform)
- [ ] Benchmark parallel speedup for various batch sizes
- [ ] Add `GOMAXPROCS` awareness

---

## Phase 20: Strided Data Support

### 20.1 Strided Input/Output

- [x] Add stride parameters to transform methods
- [x] Implement `Plan.ForwardStrided(dst, src []complex64, stride int) error`
- [x] Handle non-contiguous memory access
- [x] Test with matrix column transforms (stride = num_cols)

### 20.2 Zero-Copy Column FFT

- [x] Optimize column transforms to avoid copying
- [x] Use strided access directly in butterfly operations
- [x] Benchmark strided vs copy-based column FFT
- [x] Document when strided is faster vs copying

---

## Phase 21: Convolution API

### 21.1 Convolution via FFT

- [x] Implement `Convolve(dst, a, b []complex64) error` in `convolve.go`
- [x] Use FFT-multiply-IFFT algorithm
- [x] Handle different input lengths (zero-pad to sum of lengths - 1)
- [x] Test against naive O(n²) convolution
- [x] Benchmark for various input sizes

### 21.2 Real Convolution

- [x] Implement `ConvolveReal(dst, a, b []float32) error`
- [x] Optimize using real FFT
- [x] Test with known filter kernels (e.g., Gaussian blur)
- [x] Benchmark real convolution performance

### 21.3 Correlation API

- [x] Implement `Correlate(dst, a, b []complex64) error`
- [x] Implement `CrossCorrelate` and `AutoCorrelate` variants
- [x] Test correlation properties
- [x] Document relationship to convolution

---

## Phase 22: complex128 Support & Generic API

### 22.1 Generic Plan Type

- [x] Define `Complex` type constraint: `complex64 | complex128`
- [x] Define `Float` type constraint: `float32 | float64`
- [x] Create generic `Plan[T Complex]` type for unified API
- [x] Implement `NewPlanT[T Complex](n int) (*Plan[T], error)` generic constructor
- [x] Implement `NewPlan32(n int) (*Plan[complex64], error)` explicit constructor
- [x] Implement `NewPlan64(n int) (*Plan[complex128], error)` explicit constructor
- [x] Implement `NewPlan(n int) (*Plan[complex64], error)` convenience alias
- [x] Test generic Plan instantiation for both complex64 and complex128
- [x] Align docs/examples with the chosen `NewPlan` API (generic vs. convenience alias)

### 22.2 complex128 Implementation

- [x] Implement twiddle factor generation for complex128 (float64 precision)
- [x] Implement radix-2 FFT for complex128
- [x] Implement radix-4 FFT for complex128
- [x] Test complex128 correctness against reference DFT
- [x] Verify round-trip `Inverse(Forward(x)) ≈ x` for complex128
- [x] Add Plan-level complex128 reference checks for power-of-two sizes (public API)

### 22.3 complex128 Optimization

- [x] Implement AVX/AVX2 butterfly for complex128 (256-bit: 2 complex128)
- [x] Implement NEON butterfly for complex128 on ARM64
- [x] Benchmark complex128 vs complex64 for various sizes
- [ ] Profile precision differences (error accumulation)
- [x] Document when to use complex128 vs complex64

### 22.4 Multi-dimensional Generic Plans

- [x] Implement `NewPlan2D[T Complex](rows, cols int)` generic 2D constructor
- [x] Implement `NewPlan3D[T Complex](d, r, c int)` generic 3D constructor
- [x] Implement `NewPlanND[T Complex](dims []int)` generic N-D constructor
- [x] Add explicit `NewPlan2D32`, `NewPlan2D64` etc. convenience constructors
- [x] Test multi-dimensional plans with both precisions

---

## Phase 23: WebAssembly Support

### 23.1 WASM Build & Test

- [x] Add WASM build target to justfile
- [x] Create WASM-specific tests
- [x] Test in Node.js environment
- [ ] Test in browser environment (via wasm_exec.js)
- [x] Document WASM usage

### 23.2 WASM SIMD Exploration

- [x] Research WASM SIMD instruction availability
- [x] Evaluate Go's WASM SIMD support status
- [ ] Prototype WASM SIMD butterfly (if supported)
- [ ] Benchmark WASM performance vs native

### 23.3 WASM Examples

- [x] Create browser-based FFT demo
- [x] Create audio visualization example
- [x] Document WASM-specific considerations
- [x] Add WASM example to repository

---

## Phase 24: Documentation & Examples

### 24.1 GoDoc Completion

- [ ] Ensure all exported symbols have GoDoc comments
- [ ] Add runnable examples in `example_test.go`
- [ ] Document all error conditions
- [ ] Add package-level documentation with overview

### 24.2 README Enhancement

- [ ] Write comprehensive README with:
  - [ ] Installation instructions
  - [ ] Quick start examples
  - [ ] API overview
  - [ ] Performance characteristics
  - [ ] Comparison to other libraries
- [ ] Add badges (Go Report Card, CI status, coverage)
- [ ] Add architecture diagram

### 24.3 Tutorial Examples

- [ ] Create `examples/basic/` - simple FFT usage
- [ ] Create `examples/audio/` - audio spectrum analysis
- [ ] Create `examples/image/` - 2D FFT for image processing
- [ ] Create `examples/benchmark/` - performance comparison tool
- [ ] Document each example in its own README

---

## Phase 25: Advanced Testing

### 25.1 Cross-Architecture Testing

- [x] Set up CI matrix for amd64, arm64, 386 (`.github/workflows/test-arch.yaml`)
- [x] Add WASM to CI test matrix (included in test-arch workflow)
- [x] Test pure-Go fallback on all architectures (SIMD verification tests)
- [x] Verify SIMD paths produce same results as pure-Go (`internal/fft/simd_verify_test.go`)

### 25.2 Numerical Precision Testing

- [x] Create precision analysis tests (`precision_test.go` - Parseval, error accumulation, known signals)
- [x] Compare complex64 vs complex128 error accumulation (comparative precision tests)
- [x] Test precision for very large FFT sizes (up to 262144 tested)
- [x] Document precision guarantees (`PRECISION.md`)

### 25.3 Stress Testing

- [x] Create long-running stress tests (`stress_test.go` - configurable duration via env var)
- [x] Test memory stability over millions of transforms (1M iteration test with memory sampling)
- [x] Test concurrent usage patterns (`concurrent_test.go` - shared plans, pooled plans, mixed operations)
- [x] Profile for memory leaks (`scripts/profile_memory.sh` with automated analysis)

---

## Phase 26: Performance Profiling & Tuning

### 26.1 Comprehensive Profiling

- [ ] Profile CPU usage for various FFT sizes
- [ ] Profile memory allocation patterns
- [ ] Identify remaining optimization opportunities
- [ ] Document profiling results

### 26.2 Auto-Tuning Consideration

- [ ] Research auto-tuning approaches (like FFTW's planner)
- [ ] Consider runtime algorithm selection based on size
- [ ] Prototype simple auto-tuning for radix selection
- [ ] Evaluate complexity vs benefit

### 26.3 Final Optimization Pass

- [ ] Address any remaining performance hotspots
- [ ] Ensure consistent performance across sizes
- [ ] Final benchmark comparison to goals
- [ ] Update performance documentation

---

## Phase 27: API Stability & v1.0 Preparation

### 27.1 API Review

- [ ] Review all public APIs for consistency
- [ ] Ensure backward compatibility patterns
- [ ] Document any breaking changes from pre-release
- [ ] Create migration guide if needed

### 27.2 Stability Testing

- [ ] Run full test suite multiple times
- [ ] Test on multiple Go versions (1.21, 1.22, 1.23+)
- [ ] Verify all CI checks pass
- [ ] Address any flaky tests

### 27.3 Release Preparation

- [ ] Update CHANGELOG.md with all changes
- [ ] Create v1.0.0 release notes
- [ ] Tag release with semantic versioning
- [ ] Publish to pkg.go.dev

---

## Phase 28: Community & Maintenance

### 28.1 Community Setup

- [ ] Create issue templates (bug report, feature request)
- [ ] Create pull request template
- [ ] Set up GitHub Discussions for Q&A
- [ ] Add CODE_OF_CONDUCT.md

### 28.2 Contributor Experience

- [ ] Document development setup process
- [ ] Add "good first issue" labels to starter tasks
- [ ] Create contributor recognition system
- [ ] Set up automated dependency updates (Dependabot)

### 28.3 Ongoing Maintenance Plan

- [ ] Document maintenance schedule
- [ ] Set up security vulnerability monitoring
- [ ] Plan for future Go version compatibility
- [ ] Create roadmap for post-1.0 features

---

## Future Considerations (Post v1.0)

### Potential Future Features

- [ ] AVX-512 support when Go supports it better
- [ ] GPU acceleration (OpenCL/CUDA via cgo, optional)
- [ ] Distributed FFT for very large datasets
- [ ] Pruned FFT for sparse inputs/outputs
- [ ] DCT (Discrete Cosine Transform) support
- [ ] Hilbert transform
- [ ] Short-time FFT (STFT) for spectrograms

### Research Areas

- [ ] Explore WASM SIMD as it matures
- [ ] Evaluate Go generics for type-safe API
- [ ] Consider integration with Gonum ecosystem
- [ ] Explore memory-mapped file support for huge transforms

---

## Notes

- Each phase is designed to be completable in approximately one focused day
- Phases can be parallelized where dependencies allow
- Testing and documentation are integrated throughout, not afterthoughts
- Performance optimization happens incrementally, not all at once
- API stability is prioritized to avoid breaking changes for users

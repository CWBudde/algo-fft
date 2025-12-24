# PLAN.md - algoforge Implementation Roadmap

A detailed, task-by-task implementation plan for building the **algoforge** Go FFT library.
Each phase is scoped to approximately one day of focused work.

---

## Phase 1: Project Setup & Infrastructure

### 1.1 Repository Initialization

- [x] Initialize Go module (`go mod init github.com/MeKo-Christian/algoforge`)
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
- [x] Create clean separation: `algoforge` (public) → `internal/fft` (implementation)
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
  - Note: All errors prefixed with "algoforge:" and include descriptive text

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
- [ ] Benchmark Bluestein vs naive DFT for prime lengths
- [ ] Profile and optimize Bluestein hot paths

---

## Phase 11: Real FFT - Forward Transform

### 11.1 Real FFT API Design

- [ ] Design `PlanReal` type or extend `Plan` with real mode
- [ ] Define `PlanReal.Forward(dst []complex64, src []float32) error`
- [ ] Define output format: N/2+1 complex values for N real inputs
- [ ] Document conjugate symmetry property
- [ ] Write API usage examples

### 11.2 Real FFT Implementation (Pack Method)

- [ ] Implement "pack" approach: treat N reals as N/2 complex
- [ ] Perform N/2 complex FFT
- [ ] Unpack and combine results using symmetry
- [ ] Handle DC (index 0) and Nyquist (index N/2) specially
- [ ] Write correctness tests for real sinusoids

### 11.3 Real FFT Testing

- [ ] Test real impulse response
- [ ] Test real constant signal
- [ ] Test real cosine at various frequencies
- [ ] Verify output has conjugate symmetry
- [ ] Benchmark real FFT vs equivalent complex FFT

---

## Phase 12: Real FFT - Inverse Transform

### 12.1 Inverse Real FFT Implementation

- [ ] Implement `PlanReal.Inverse(dst []float32, src []complex64) error`
- [ ] Reconstruct full complex spectrum from half-spectrum
- [ ] Apply inverse complex FFT
- [ ] Extract real part of result
- [ ] Verify imaginary parts are near-zero

### 12.2 Real FFT Round-Trip Testing

- [ ] Test `Inverse(Forward(x)) ≈ x` for real signals
- [ ] Test with various signal types (noise, tones, chirps)
- [ ] Fuzz test real FFT round-trip
- [ ] Document precision expectations

### 12.3 Real FFT Optimization

- [ ] Optimize real FFT to avoid full complex allocation
- [ ] Profile and reduce memory usage
- [ ] Ensure zero steady-state allocations
- [ ] Benchmark optimized real FFT

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

- [ ] Set up `internal/asm/` directory structure
- [ ] Create build tags for `amd64`, `arm64`, `purego`
- [ ] Create stub `.s` files with proper Go assembly syntax
- [ ] Set up `go:noescape` pragma usage
- [ ] Document assembly contribution guidelines

---

## Phase 14: AVX2 Butterfly Implementation

### 14.1 AVX2 Radix-2 Butterfly

- [ ] Implement `butterfly2_avx2` in `butterfly_amd64.s`
- [ ] Process 4 complex64 values (8 floats) per iteration
- [ ] Handle memory alignment requirements
- [ ] Write Go wrapper function with `go:noescape`
- [ ] Test AVX2 butterfly against Go reference

### 14.2 AVX2 Radix-4 Butterfly

- [ ] Implement `butterfly4_avx2` in assembly
- [ ] Optimize register usage for radix-4 structure
- [ ] Handle twiddle factor loading efficiently
- [ ] Test AVX2 radix-4 against Go reference
- [ ] Benchmark AVX2 vs pure Go for radix-4

### 14.3 AVX2 Integration

- [ ] Wire AVX2 butterflies into dispatch system
- [ ] Implement hybrid approach: AVX2 for bulk, Go for edges
- [ ] Test full FFT with AVX2 enabled
- [ ] Benchmark end-to-end FFT speedup from AVX2

---

## Phase 15: ARM64 NEON Implementation

### 15.1 ARM64 Infrastructure

- [ ] Detect NEON availability (always available on ARMv8)
- [ ] Set up `butterfly_arm64.s` assembly file
- [ ] Study ARM64 SIMD register layout (128-bit vectors)
- [ ] Create test environment (use ARM CI runner or emulator)

### 15.2 NEON Radix-2 Butterfly

- [ ] Implement `butterfly2_neon` in ARM64 assembly
- [ ] Process 2 complex64 values (4 floats) per iteration
- [ ] Use FMLA/FMLS instructions for multiply-accumulate
- [ ] Test NEON butterfly against Go reference

### 15.3 NEON Integration & Testing

- [ ] Wire NEON into dispatch system
- [ ] Test full FFT with NEON enabled on ARM64
- [ ] Benchmark NEON vs pure Go on ARM64 hardware
- [ ] Ensure CI tests on both amd64 and arm64

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

## Phase 17: 2D FFT Implementation

### 17.1 2D FFT API Design

- [ ] Design `Plan2D` type for 2D transforms
- [ ] Define matrix representation (row-major []complex64 with stride)
- [ ] Define `Plan2D.Forward(dst, src []complex64, rows, cols int) error`
- [ ] Define `Plan2D.Inverse(...)` method
- [ ] Document row-major storage convention

### 17.2 Row-Column Algorithm

- [ ] Implement 2D FFT via row-wise then column-wise 1D FFTs
- [ ] Create helper for extracting/inserting columns with stride
- [ ] Reuse 1D Plan objects internally
- [ ] Handle non-square matrices (rows ≠ cols)
- [ ] Write basic 2D correctness tests

### 17.3 2D FFT Testing & Optimization

- [ ] Test 2D FFT on known images/patterns
- [ ] Test round-trip correctness
- [ ] Optimize column extraction (avoid copies if possible)
- [ ] Benchmark 2D FFT for various matrix sizes
- [ ] Compare to theoretical O(MN log(MN)) complexity

---

## Phase 18: 3D FFT & N-D Generalization

### 18.1 3D FFT Implementation

- [ ] Design `Plan3D` type for 3D transforms
- [ ] Implement as three passes of 1D FFTs along each axis
- [ ] Handle axis iteration order for cache efficiency
- [ ] Write correctness tests for 3D transforms

### 18.2 N-Dimensional API

- [ ] Design generic `PlanND` type for arbitrary dimensions
- [ ] Accept dimension sizes as `[]int`
- [ ] Implement iterative axis-by-axis transform
- [ ] Test 4D and 5D transforms
- [ ] Benchmark N-D overhead vs specialized 2D/3D

### 18.3 Multi-Dimensional Real FFT

- [ ] Implement real-input 2D FFT (output: (N/2+1) × M)
- [ ] Implement real-input 3D FFT
- [ ] Test with real-valued image/volume data
- [ ] Document output size conventions

---

## Phase 19: Batch Processing

### 19.1 Batch FFT API

- [ ] Design batch API: `Plan.ForwardBatch(dst, src []complex64, count, stride int) error`
- [ ] Support interleaved (stride=n) and sequential (stride=1) layouts
- [ ] Document memory layout expectations
- [ ] Write basic batch correctness tests

### 19.2 Batch Implementation

- [ ] Implement simple loop-based batching
- [ ] Ensure each FFT in batch uses same Plan
- [ ] Test batch results match individual FFT calls
- [ ] Benchmark batch vs individual calls

### 19.3 Parallel Batch Processing

- [ ] Add `Plan.ForwardBatchParallel(...)` with goroutines
- [ ] Implement worker pool for large batch counts
- [ ] Ensure Plan is safe for concurrent use (read-only during transform)
- [ ] Benchmark parallel speedup for various batch sizes
- [ ] Add `GOMAXPROCS` awareness

---

## Phase 20: Strided Data Support

### 20.1 Strided Input/Output

- [ ] Add stride parameters to transform methods
- [ ] Implement `Plan.ForwardStrided(dst, src []complex64, stride int) error`
- [ ] Handle non-contiguous memory access
- [ ] Test with matrix column transforms (stride = num_cols)

### 20.2 Zero-Copy Column FFT

- [ ] Optimize column transforms to avoid copying
- [ ] Use strided access directly in butterfly operations
- [ ] Benchmark strided vs copy-based column FFT
- [ ] Document when strided is faster vs copying

---

## Phase 21: Convolution API

### 21.1 Convolution via FFT

- [ ] Implement `Convolve(dst, a, b []complex64) error` in `convolve.go`
- [ ] Use FFT-multiply-IFFT algorithm
- [ ] Handle different input lengths (zero-pad to sum of lengths - 1)
- [ ] Test against naive O(n²) convolution
- [ ] Benchmark for various input sizes

### 21.2 Real Convolution

- [ ] Implement `ConvolveReal(dst, a, b []float32) error`
- [ ] Optimize using real FFT
- [ ] Test with known filter kernels (e.g., Gaussian blur)
- [ ] Benchmark real convolution performance

### 21.3 Correlation API

- [ ] Implement `Correlate(dst, a, b []complex64) error`
- [ ] Implement `CrossCorrelate` and `AutoCorrelate` variants
- [ ] Test correlation properties
- [ ] Document relationship to convolution

---

## Phase 22: complex128 Support & Generic API

### 22.1 Generic Plan Type

- [ ] Define `Complex` type constraint: `complex64 | complex128`
- [ ] Define `Float` type constraint: `float32 | float64`
- [ ] Create generic `Plan[T Complex]` type for unified API
- [ ] Implement `NewPlan[T Complex](n int) (*Plan[T], error)` generic constructor
- [ ] Implement `NewPlan32(n int) (*Plan[complex64], error)` explicit constructor
- [ ] Implement `NewPlan64(n int) (*Plan[complex128], error)` explicit constructor
- [ ] Implement `NewPlan(n int) (*Plan[complex64], error)` convenience alias
- [ ] Test generic Plan instantiation for both complex64 and complex128

### 22.2 complex128 Implementation

- [ ] Implement twiddle factor generation for complex128 (float64 precision)
- [ ] Implement radix-2 FFT for complex128
- [ ] Implement radix-4 FFT for complex128
- [ ] Test complex128 correctness against reference DFT
- [ ] Verify round-trip `Inverse(Forward(x)) ≈ x` for complex128

### 22.3 complex128 Optimization

- [ ] Implement AVX/AVX2 butterfly for complex128 (256-bit: 2 complex128)
- [ ] Implement NEON butterfly for complex128 on ARM64
- [ ] Benchmark complex128 vs complex64 for various sizes
- [ ] Profile precision differences (error accumulation)
- [ ] Document when to use complex128 vs complex64

### 22.4 Multi-dimensional Generic Plans

- [ ] Implement `NewPlan2D[T Complex](rows, cols int)` generic 2D constructor
- [ ] Implement `NewPlan3D[T Complex](d, r, c int)` generic 3D constructor
- [ ] Implement `NewPlanND[T Complex](dims []int)` generic N-D constructor
- [ ] Add explicit `NewPlan2D32`, `NewPlan2D64` etc. convenience constructors
- [ ] Test multi-dimensional plans with both precisions

---

## Phase 23: WebAssembly Support

### 23.1 WASM Build & Test

- [ ] Add WASM build target to Makefile/justfile
- [ ] Create WASM-specific tests
- [ ] Test in Node.js environment
- [ ] Test in browser environment (via wasm_exec.js)
- [ ] Document WASM usage

### 23.2 WASM SIMD Exploration

- [ ] Research WASM SIMD instruction availability
- [ ] Evaluate Go's WASM SIMD support status
- [ ] Prototype WASM SIMD butterfly (if supported)
- [ ] Benchmark WASM performance vs native

### 23.3 WASM Examples

- [ ] Create browser-based FFT demo
- [ ] Create audio visualization example
- [ ] Document WASM-specific considerations
- [ ] Add WASM example to repository

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

- [ ] Set up CI matrix for amd64, arm64, 386
- [ ] Add WASM to CI test matrix
- [ ] Test pure-Go fallback on all architectures
- [ ] Verify SIMD paths produce same results as pure-Go

### 25.2 Numerical Precision Testing

- [ ] Create precision analysis tests
- [ ] Compare complex64 vs complex128 error accumulation
- [ ] Test precision for very large FFT sizes
- [ ] Document precision guarantees

### 25.3 Stress Testing

- [ ] Create long-running stress tests
- [ ] Test memory stability over millions of transforms
- [ ] Test concurrent usage patterns
- [ ] Profile for memory leaks

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

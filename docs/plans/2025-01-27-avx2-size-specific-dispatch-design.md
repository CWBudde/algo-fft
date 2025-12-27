# Size-Specific AVX2 Dispatch Design

**Date**: 2025-01-27
**Phase**: 14.5.1 - Design size-specific dispatch mechanism
**Status**: Approved for implementation

## Overview

This design adds a size-specific dispatch layer for AVX2 SIMD kernels, targeting common FFT sizes (16, 32, 64, 128) with fully unrolled implementations. This combines the benefits of size-specific unrolling (like `dit_small.go`) with SIMD vectorization for maximum performance.

## Motivation

The codebase already has:

- **Size-specific pure-Go kernels** (`dit_small.go`): Fully unrolled for sizes 8-128
- **Generic AVX2 kernels** (`asm_amd64.s`): SIMD with 2× loop unrolling for any size ≥16

Size-specific AVX2 kernels will eliminate all loop overhead while using SIMD, providing 5-20% additional speedup for the most common FFT sizes.

## Architecture

### Component Overview

```
User Code
    ↓
Plan.Forward/Inverse
    ↓
selectKernelsComplex64 (kernels_amd64_asm.go)
    ↓
fallbackKernel(
    avx2SizeSpecificOrGenericDITComplex64,  ← NEW wrapper
    auto.Forward                             ← Pure-Go fallback
)
    ↓
Size-specific wrapper (NEW FILE: kernels_amd64_size_specific.go)
    ↓
    ├─ size 16 → forwardAVX2Size16Complex64Asm → (fallback) → forwardAVX2Complex64Asm
    ├─ size 32 → forwardAVX2Size32Complex64Asm → (fallback) → forwardAVX2Complex64Asm
    ├─ size 64 → forwardAVX2Size64Complex64Asm → (fallback) → forwardAVX2Complex64Asm
    ├─ size 128 → forwardAVX2Size128Complex64Asm → (fallback) → forwardAVX2Complex64Asm
    └─ other → forwardAVX2Complex64Asm (generic)
```

### Fallback Chain (Three Levels)

1. **Size-specific AVX2**: Try fully unrolled SIMD for exact sizes (16, 32, 64, 128)
2. **Generic AVX2**: Fall back to loop-based SIMD for any size ≥16
3. **Pure-Go DIT**: Final fallback (handled by existing `fallbackKernel`)

The first two levels are encapsulated in the new size-specific wrapper.

## File Structure

### New File: `internal/fft/kernels_amd64_size_specific.go`

**Purpose**: Size-specific dispatch for AVX2 kernels

**Build tags**: `//go:build amd64 && fft_asm && !purego`

**Contents**:

- `avx2SizeSpecificOrGenericDITComplex64(strategy KernelStrategy) Kernel[complex64]`
- `avx2SizeSpecificOrGenericDITInverseComplex64(strategy KernelStrategy) Kernel[complex64]`
- `avx2SizeSpecificOrGenericStockhamComplex64(strategy KernelStrategy) Kernel[complex64]` (optional)
- `avx2SizeSpecificOrGenericStockhamInverseComplex64(strategy KernelStrategy) Kernel[complex64]` (optional)

Each function returns a `Kernel[complex64]` that:

1. Checks `len(src)` via switch statement
2. For sizes 16, 32, 64, 128: calls size-specific assembly, falls back to generic AVX2
3. For other sizes: calls generic AVX2 directly

### Assembly Function Declarations

**File**: `internal/fft/asm_amd64.go` (or new `kernels_amd64_size_specific_asm.go`)

**New function declarations** (8 for complex64):

```go
//go:noescape
func forwardAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// Inverse variants
//go:noescape
func inverseAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
// ... (3 more inverse)
```

**Implementation**: Initially, these will be **stub functions** in `asm_amd64.s` that immediately return false, allowing the fallback to generic AVX2. Actual unrolled implementations will be added in phases 14.5.2-14.5.5.

### Modified File: `internal/fft/kernels_amd64_asm.go`

**Change in `selectKernelsComplex64`**:

**Before**:

```go
return Kernels[complex64]{
    Forward: fallbackKernel(
        avx2KernelComplex64(KernelAuto, forwardAVX2Complex64, forwardAVX2StockhamComplex64),
        auto.Forward,
    ),
    Inverse: fallbackKernel(
        avx2KernelComplex64(KernelAuto, inverseAVX2Complex64, inverseAVX2StockhamComplex64),
        auto.Inverse,
    ),
}
```

**After**:

```go
return Kernels[complex64]{
    Forward: fallbackKernel(
        avx2SizeSpecificOrGenericComplex64(KernelAuto),  // NEW
        auto.Forward,
    ),
    Inverse: fallbackKernel(
        avx2SizeSpecificOrGenericInverseComplex64(KernelAuto),  // NEW
        auto.Inverse,
    ),
}
```

Similar changes for `selectKernelsComplex64WithStrategy`.

## Implementation Details

### Size-Specific Wrapper Function

```go
// kernels_amd64_size_specific.go
package fft

func avx2SizeSpecificOrGenericDITComplex64(strategy KernelStrategy) Kernel[complex64] {
    return func(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
        n := len(src)

        // Determine which algorithm (DIT vs Stockham) based on strategy
        resolved := resolveKernelStrategy(n, strategy)
        if resolved != KernelDIT {
            // For non-DIT strategies, use generic implementation
            return avx2KernelComplex64(strategy, forwardAVX2Complex64, forwardAVX2StockhamComplex64)(
                dst, src, twiddle, scratch, bitrev,
            )
        }

        // DIT strategy: try size-specific, fall back to generic
        switch n {
        case 16:
            if forwardAVX2Size16Complex64Asm(dst, src, twiddle, scratch, bitrev) {
                return true
            }
            return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

        case 32:
            if forwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch, bitrev) {
                return true
            }
            return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

        case 64:
            if forwardAVX2Size64Complex64Asm(dst, src, twiddle, scratch, bitrev) {
                return true
            }
            return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

        case 128:
            if forwardAVX2Size128Complex64Asm(dst, src, twiddle, scratch, bitrev) {
                return true
            }
            return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

        default:
            // For other sizes, use generic AVX2
            return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
        }
    }
}
```

### Assembly Stubs (Initial Implementation)

```asm
// asm_amd64.s

// Size-16 forward transform (stub - returns false to fall back)
TEXT ·forwardAVX2Size16Complex64Asm(SB), NOSPLIT, $0-121
    MOVB $0, ret+120(FP)  // Return false
    RET

// Size-32 forward transform (stub)
TEXT ·forwardAVX2Size32Complex64Asm(SB), NOSPLIT, $0-121
    MOVB $0, ret+120(FP)  // Return false
    RET

// ... (6 more stubs for sizes 64, 128 and inverse variants)
```

These stubs allow the dispatch mechanism to work immediately, with all size-specific calls falling back to the generic AVX2 implementation. Phases 14.5.2-14.5.5 will replace these with actual unrolled kernels.

## Testing Strategy

### Phase 14.5.1 (This Phase)

**Goal**: Verify dispatch mechanism works correctly with stubs

1. **Unit tests**: Verify wrapper routes to correct functions
2. **Integration tests**: Ensure transforms still produce correct results (via fallback)
3. **Benchmark tests**: Baseline performance (should match generic AVX2 exactly)

### Future Phases (14.5.2-14.5.5)

Once unrolled kernels are implemented:

1. **Correctness tests**: Compare size-specific vs generic AVX2 vs reference DFT
2. **Performance tests**: Measure 5-20% speedup over generic AVX2
3. **Fallback tests**: Verify graceful degradation if size-specific kernel fails

## Benchmarking

### New Benchmarks

Add to existing benchmark file or create `kernels_amd64_size_specific_bench_test.go`:

```go
func BenchmarkAVX2SizeSpecific_vs_Generic_16(b *testing.B) {
    benchmarkSizeSpecificVsGeneric(b, 16)
}

func BenchmarkAVX2SizeSpecific_vs_Generic_32(b *testing.B) {
    benchmarkSizeSpecificVsGeneric(b, 32)
}

func BenchmarkAVX2SizeSpecific_vs_Generic_64(b *testing.B) {
    benchmarkSizeSpecificVsGeneric(b, 64)
}

func BenchmarkAVX2SizeSpecific_vs_Generic_128(b *testing.B) {
    benchmarkSizeSpecificVsGeneric(b, 128)
}

func benchmarkSizeSpecificVsGeneric(b *testing.B, n int) {
    // Benchmark both size-specific and generic paths
    // Initially should show identical performance (stubs)
    // Later should show 5-20% speedup for size-specific
}
```

## Success Criteria

### Phase 14.5.1 Completion

- ✅ `kernels_amd64_size_specific.go` created with wrapper functions
- ✅ Assembly stub declarations added
- ✅ `selectKernelsComplex64` updated to use new wrapper
- ✅ All existing tests pass (no regressions)
- ✅ Benchmarks show no performance change (stubs fall back to generic)
- ✅ Code compiles with build tags: `amd64`, `fft_asm`

### Overall Phase 14.5 Completion (14.5.2-14.5.7)

- Size-specific kernels achieve 5-20% speedup over generic AVX2
- Zero correctness regressions
- Graceful fallback if size-specific kernel returns false
- Comprehensive benchmarks documenting improvements

## Future Extensions

### Phase 14.5.7: complex128 Support

Similar structure, but:

- AVX2 processes 2 complex128 instead of 4 complex64
- Expect ~2× speedup vs pure-Go complex128 (vs 4× for complex64)
- Add `avx2SizeSpecificOrGenericDITComplex128` wrapper

### Phase 15.5: NEON Size-Specific Kernels

Mirror this design for ARM64:

- File: `kernels_arm64_size_specific.go`
- NEON processes 2 complex64 per 128-bit register
- Same dispatch pattern, different assembly

## References

- **Existing patterns**:
  - `dit.go` (lines 3-22): Size dispatch for pure-Go kernels
  - `kernels_amd64_asm.go`: Feature-based dispatch
  - `kernels_avx2_strategy.go`: Strategy-based dispatch

- **Related files**:
  - `dit_small.go`: Pure-Go size-specific implementations (model for unrolling)
  - `asm_amd64.s`: Generic AVX2 implementations (base for size-specific versions)

## Implementation Order

1. ✅ Design document (this file)
2. Create `kernels_amd64_size_specific.go` with wrapper functions
3. Add assembly stub declarations
4. Update `selectKernelsComplex64` integration point
5. Add benchmarks for baseline measurement
6. Test compilation and correctness (all tests pass via fallback)
7. Document in `PLAN.md` as complete (14.5.1 ✅)
8. Proceed to 14.5.2 (implement actual size-16 unrolled kernel)

# FFT Implementation Inventory

This document provides a comprehensive overview of all specialized FFT implementations in the `algo-fft` library.

## Quick Reference Grid

| Size | Algorithm | Complex64 (Go) | Complex64 (AVX2) | Complex128 (Go) | Complex128 (AVX2) |
| ---- | --------- | -------------- | ---------------- | --------------- | ----------------- |
| 4    | Radix-4   | ✓              | ✓                | ✓               | -                 |
| 8    | Radix-2   | ✓              | ✓                | ✓               | ✓                 |
| 8    | Radix-8   | ✓              | ✓                | ✓               | ✓                 |
| 8    | Mixed¹    | ✓              | ✓                | ✓               | -                 |
| 16   | Radix-2   | ✓              | ✓                | ✓               | ✓                 |
| 16   | Radix-4   | ✓              | ✓                | ✓               | ✓                 |
| 32   | Radix-2   | ✓              | ✓                | ✓               | ✓                 |
| 64   | Radix-2   | ✓              | ✓                | ✓               | -                 |
| 64   | Radix-4   | ✓              | ✓                | ✓               | -                 |
| 128  | Radix-2   | ✓              | ✓                | ✓               | -                 |
| 256  | Radix-2   | ✓              | ✓                | ✓               | -                 |
| 256  | Radix-4   | ✓              | ✓                | ✓               | -                 |
| 512  | Radix-2   | ✓              | -                | ✓               | -                 |
| 2048 | Mixed³    | ✓              | -                | -               | -                 |
| 8192 | Mixed³    | ✓              | -                | -               | -                 |

**Legend:**

- ✓ = Fully implemented (forward + inverse)
- ⚠ = Partially implemented
- \- = Not implemented
- ¹ Mixed = 1x radix-4 + 1x radix-2 stage
- ² Size 256 radix-4 AVX2: Fully implemented
- ³ Mixed = 1x radix-2 + N radix-4 stages (currently delegates to radix-2)

## Detailed Breakdown

### Size 4

| Type       | Algorithm | SIMD | Source | Status | Files                    |
| ---------- | --------- | ---- | ------ | ------ | ------------------------ |
| complex64  | radix-4   | none | Go     | ✓      | `dit_size4.go`           |
| complex64  | radix-4   | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size4.s` |
| complex128 | radix-4   | none | Go     | ✓      | `dit_size4.go`           |

**Notes:**

- No bit-reversal needed (size is power of 4)
- AVX2 uses scalar-style SIMD pattern

### Size 8

| Type       | Algorithm           | SIMD | Source | Status | Files                               |
| ---------- | ------------------- | ---- | ------ | ------ | ----------------------------------- |
| complex64  | radix-2 (3 stages)  | none | Go     | ✓      | `dit_size8.go`                      |
| complex64  | radix-2 (3 stages)  | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size8.s`            |
| complex64  | radix-8 (1 stage)   | none | Go     | ✓      | `dit_size8.go`                      |
| complex64  | radix-8 (1 stage)   | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size8.s`            |
| complex64  | mixed-radix (2 st.) | none | Go     | ✓      | `dit_size8.go`                      |
| complex64  | mixed-radix (2 st.) | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size8.s`            |
| complex128 | radix-2 (3 stages)  | none | Go     | ✓      | `dit_size8.go`                      |
| complex128 | radix-2 (3 stages)  | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size8_complex128.s` |
| complex128 | radix-8 (1 stage)   | none | Go     | ✓      | `dit_size8.go`                      |
| complex128 | mixed-radix (2 st.) | none | Go     | ✓      | `dit_size8.go`                      |

**Notes:**

- Mixed-radix: 1 radix-4 stage + 1 radix-2 stage (reduces from 3 to 2 stages)
- AVX2 mixed-radix uses scalar-style SIMD pattern from size-4
- AVX2 size-8 codelets exist but are not registered by default because they are slower than Go
- Binary bit-reversal used (8 is not a power of 4)

### Size 16

| Type       | Algorithm | SIMD | Source | Status | Files                            |
| ---------- | --------- | ---- | ------ | ------ | -------------------------------- |
| complex64  | radix-2   | none | Go     | ✓      | `dit_size16_radix2.go`           |
| complex64  | radix-2   | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size16.s`        |
| complex64  | radix-4   | none | Go     | ✓      | `dit_size16_radix4.go`           |
| complex64  | radix-4   | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size16_radix4.s` |
| complex128 | radix-2   | none | Go     | ✓      | `dit_size16_radix2.go`           |
| complex128 | radix-2   | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size16.s`        |
| complex128 | radix-4   | none | Go     | ✓      | `dit_size16_radix4.go`           |
| complex128 | radix-4   | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size16_radix4.s` |

**Notes:**

- Radix-4 variant is 12-15% faster (per benchmarks)
- Radix-4 uses radix-4 bit-reversal indices
- AVX2 radix-4 variant implemented for both precisions

### Size 32

| Type       | Algorithm | SIMD | Source | Status | Files                                |
| ---------- | --------- | ---- | ------ | ------ | ------------------------------------ |
| complex64  | radix-2   | none | Go     | ✓      | `dit_size32.go`                      |
| complex64  | radix-2   | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size32.s`            |
| complex128 | radix-2   | none | Go     | ✓      | `dit_size32.go`                      |
| complex128 | radix-2   | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size32_complex128.s` |

**Notes:**

- Only radix-2 variant implemented
- AVX2 complex128 variant uses scalar-style SIMD pattern

### Size 64

| Type       | Algorithm | SIMD | Source | Status | Files                            |
| ---------- | --------- | ---- | ------ | ------ | -------------------------------- |
| complex64  | radix-2   | none | Go     | ✓      | `dit_size64.go`                  |
| complex64  | radix-2   | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size64.s`        |
| complex64  | radix-4   | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size64_radix4.s` |
| complex64  | radix-4   | none | Go     | ✓      | `dit_size64_radix4.go`           |
| complex128 | radix-2   | none | Go     | ✓      | `dit_size64.go`                  |
| complex128 | radix-4   | none | Go     | ✓      | `dit_size64_radix4.go`           |

**Notes:**

- Radix-4 uses radix-4 bit-reversal indices

### Size 128

| Type       | Algorithm | SIMD | Source | Status | Files                      |
| ---------- | --------- | ---- | ------ | ------ | -------------------------- |
| complex64  | radix-2   | none | Go     | ✓      | `dit_size128.go`           |
| complex64  | radix-2   | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size128.s` |
| complex128 | radix-2   | none | Go     | ✓      | `dit_size128.go`           |

**Notes:**

- Only radix-2 variant implemented
- Potential optimization: Add radix-4 variant

### Size 256

| Type       | Algorithm | SIMD | Source | Status | Files                             |
| ---------- | --------- | ---- | ------ | ------ | --------------------------------- |
| complex64  | radix-2   | none | Go     | ✓      | `dit_size256_radix2.go`           |
| complex64  | radix-2   | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size256_radix2.s` |
| complex64  | radix-4   | none | Go     | ✓      | `dit_size256_radix4.go`           |
| complex64  | radix-4   | AVX2 | Asm    | ✓      | `asm_amd64_avx2_size256_radix4.s` |
| complex128 | radix-2   | none | Go     | ✓      | `dit_size256_radix2.go`           |
| complex128 | radix-4   | none | Go     | ✓      | `dit_size256_radix4.go`           |

**Notes:**

- Radix-4 variant potentially faster (higher priority in registry)
- AVX2 radix-4: Fully implemented (forward + inverse)
- Radix-4 uses radix-4 bit-reversal indices

### Size 512

| Type       | Algorithm | SIMD | Source | Status | Files            |
| ---------- | --------- | ---- | ------ | ------ | ---------------- |
| complex64  | radix-2   | none | Go     | ✓      | `dit_size512.go` |
| complex128 | radix-2   | none | Go     | ✓      | `dit_size512.go` |

### Size 2048 / 8192

| Type      | Algorithm | SIMD | Source | Status | Files                 |
| --------- | --------- | ---- | ------ | ------ | --------------------- |
| complex64 | mixed³    | none | Go     | ✓      | `dit_mixedradix24.go` |

**Notes:**

- ³ Mixed = 1x radix-2 + N radix-4 stages (currently delegates to radix-2 for correctness)

## Coverage Summary

### Pure Go Implementations

All sizes have complete Go implementations for both `complex64` and `complex128` (except for very large mixed-radix sizes):

- **Size 4**: 1 variant each (radix-4)
- **Size 8**: 3 variants each (radix-2, radix-8, mixed-radix)
- **Size 16**: 2 variants each (radix-2, radix-4)
- **Size 32**: 1 variant each (radix-2)
- **Size 64**: 2 variants each (radix-2, radix-4)
- **Size 128**: 1 variant each (radix-2)
- **Size 256**: 2 variants each (radix-2, radix-4)
- **Size 512**: 1 variant each (radix-2)
- **Size 2048 / 8192**: 1 variant (mixed-radix complex64)

**Total:** 31 implementations (16 complex64 + 15 complex128)

### AVX2 Assembly Implementations

AVX2 optimizations exist for both `complex64` and `complex128`:

- **Size 4**: 1 variant (radix-4 complex64)
- **Size 8**: 3 variants (radix-2, radix-8, mixed-radix complex64)
- **Size 8**: 1 variant (radix-2 complex128)
- **Size 16**: 2 variants each (radix-2, radix-4 for both 64/128)
- **Size 32**: 1 variant each (radix-2 for both 64/128)
- **Size 64**: 2 variants (radix-2, radix-4 complex64)
- **Size 128**: 1 variant (radix-2 complex64)
- **Size 256**: 2 variants (radix-2, radix-4 complex64)

**Total:** 19 complete implementations

## Missing Implementations

### Critical Missing Features

1. **Complex128 AVX2 for Sizes 64+**
   - Currently only implemented up to size 32
   - Larger sizes fall back to generic kernels or pure Go

2. **Size 512+ AVX2**
   - No size-specific assembly for 512, 1024, etc.
   - These use the generic AVX2 kernel

### Potential Optimizations

The following sizes could benefit from additional variants:

1. **Size 32 AVX2 Radix-4** (`complex64`/`complex128`)
   - Could reduce stages and improve performance

2. **Size 128 AVX2 Radix-4** (`complex64`)
   - Could reduce stages and improve performance

3. **Size 128 AVX2 Radix-4** (`complex64`)
   - Could reduce stages and improve performance

## Implementation Patterns

### Scalar-Style SIMD (Size 4, 8)

Used for size-4 and size-8 radix-4 AVX2 implementations:

- Extract pairs to 128-bit XMM registers
- Process with scalar-like operations
- Use duplicate/shuffle for parallel add/subtract
- Recombine results
- **Advantage:** Cleaner, more maintainable than full vectorization

### Full Vectorization (Size 16+)

Used for larger sizes:

- Full YMM register utilization
- Vectorized butterfly operations
- Optimized for cache efficiency

## Registration System

Implementations are registered via the codelet system:

1. **Generic Go** - `codelet_init.go`:
   - `registerDITCodelets64()` - complex64 variants
   - `registerDITCodelets128()` - complex128 variants

2. **AVX2 Assembly** - `codelet_init_avx2.go`:
   - `registerAVX2DITCodelets64()` - complex64 AVX2 variants
   - `registerAVX2DITCodelets128()` - complex128 AVX2 variants (TODO)

Priority determines which implementation is selected when multiple exist for the same size.

## Build Tags

Assembly implementations require:

- Build tag: `fft_asm`
- Architecture: `amd64`
- Excluded by: `purego`

Build constraint: `//go:build amd64 && fft_asm && !purego`

## Testing

All implementations are validated against:

- Naive O(n²) reference DFT
- Round-trip tests: `Inverse(Forward(x)) ≈ x`
- Property tests: Parseval's theorem, linearity, shift theorems
- Cross-validation: Assembly vs Pure Go

---

_Generated: 2025-12-30_
_See: `internal/fft/inventory_check.go` for automated inventory generation_

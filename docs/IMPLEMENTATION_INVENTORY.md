# FFT Implementation Inventory

This document provides a comprehensive overview of all specialized FFT implementations in the `algo-fft` library.

## Quick Reference Grid

### Complex64

| Size  | Algorithm | Go  | AVX2 | SSE2 | NEON |
| ----- | --------- | --- | ---- | ---- | ---- |
| 4     | Radix-4   | ✓   | ✓    | ✓    | ✓    |
| 8     | Radix-2   | ✓   | ✓    | -    | ✓    |
| 8     | Radix-8   | ✓   | ✓    | -    | ✓    |
| 8     | Mixed¹    | ✓   | ✓    | -    | ✓    |
| 16    | Radix-2   | ✓   | ✓    | -    | ✓    |
| 16    | Radix-4   | ✓   | ✓    | ✓    | ✓    |
| 32    | Radix-2   | ✓   | ✓    | -    | ✓    |
| 32    | Mixed²    | ✓   | ✓    | -    | ✓    |
| 64    | Radix-2   | ✓   | ✓    | -    | ✓    |
| 64    | Radix-4   | ✓   | ✓    | ✓    | ✓    |
| 128   | Radix-2   | ✓   | ✓    | -    | ✓    |
| 128   | Mixed²    | ✓   | ✓    | -    | ✓    |
| 256   | Radix-2   | ✓   | ✓    | -    | ✓    |
| 256   | Radix-4   | ✓   | ✓    | -    | ✓    |
| 512   | Radix-2   | ✓   | ✓    | -    | ✓    |
| 512   | Mixed⁴    | ✓   | ✓    | -    | -    |
| 1024  | Radix-4   | ✓   | -    | -    | -    |
| 2048  | Mixed⁴    | ✓   | -    | -    | -    |
| 4096  | Radix-4   | ✓   | -    | -    | -    |
| 8192  | Mixed⁴    | ✓   | -    | -    | -    |
| 16384 | Radix-4   | ✓   | -    | -    | -    |

### Complex128

| Size  | Algorithm | Go  | AVX2 | SSE2 | NEON |
| ----- | --------- | --- | ---- | ---- | ---- |
| 4     | Radix-4   | ✓   | ✓    | ✓    | -    |
| 8     | Radix-2   | ✓   | ✓    | -    | -    |
| 8     | Radix-8   | ✓   | ✓    | -    | -    |
| 8     | Mixed¹    | ✓   | ✓    | -    | -    |
| 16    | Radix-2   | ✓   | ✓    | -    | ✓    |
| 16    | Radix-4   | ✓   | ✓    | -    | -    |
| 32    | Radix-2   | ✓   | ✓    | -    | ✓    |
| 32    | Mixed²    | ✓   | ✓    | -    | -    |
| 64    | Radix-2   | ✓   | ✓    | -    | ✓    |
| 64    | Radix-4   | ✓   | ✓    | -    | -    |
| 128   | Radix-2   | ✓   | ✓    | -    | ✓    |
| 128   | Mixed²    | ✓   | ✓    | -    | -    |
| 256   | Radix-2   | ✓   | ✓    | -    | ✓    |
| 256   | Radix-4   | ✓   | -    | -    | -    |
| 512   | Radix-2   | ✓   | -    | -    | ✓    |
| 512   | Mixed⁴    | ✓   | ✓    | -    | -    |
| 1024  | Radix-4   | ✓   | -    | -    | -    |
| 2048  | Mixed⁴    | ✓   | -    | -    | -    |
| 4096  | Radix-4   | ✓   | -    | -    | -    |
| 8192  | Mixed⁴    | ✓   | -    | -    | -    |
| 16384 | Radix-4   | ✓   | -    | -    | -    |

**Legend:**

- ✓ = Fully implemented (forward + inverse)
- ⚠ = Partially implemented
- \- = Not implemented
- ¹ Mixed = 1x radix-4 + 1x radix-2 stage
- ² Mixed = 2x radix-4 + 1x radix-2 stages (delegates to proven radix-2 for correctness)
- ⁴ Mixed = N radix-4 stages + 1x radix-2 final stage (for sizes 2×4ᵏ)

## Detailed Breakdown

### Size 4

| Type       | Algorithm | SIMD | Source | Status | Files                                             |
| ---------- | --------- | ---- | ------ | ------ | ------------------------------------------------- |
| complex64  | radix-4   | none | Go     | ✓      | `dit_size4.go`                                    |
| complex64  | radix-4   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f32_size4_radix4.s`      |
| complex64  | radix-4   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f32_size4_radix4.s`      |
| complex128 | radix-4   | none | Go     | ✓      | `dit_size4.go`                                    |
| complex128 | radix-4   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f64_size4_radix4.s`      |
| complex128 | radix-4   | SSE2 | Asm    | ✓      | `internal/asm/amd64/sse2_f64_size4_radix4.s`      |
| complex128 | radix-4   | NEON | Wrap   | ✓      | `asm_arm64.go`                                    |

**Notes:**

- No bit-reversal needed (size is power of 4)
- AVX2 uses scalar-style SIMD pattern (complex64 and complex128 have assembly)
- SSE2 complex128 uses a scalar-style radix-4 kernel for size 4
- NEON complex128 implementation wraps pure Go implementation

### Size 8

| Type       | Algorithm           | SIMD | Source | Status | Files                                        |
| ---------- | ------------------- | ---- | ------ | ------ | -------------------------------------------- |
| complex64  | radix-2 (3 stages)  | none | Go     | ✓      | `dit_size8.go`                               |
| complex64  | radix-2 (3 stages)  | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f32_size8_radix2.s` |
| complex64  | radix-2 (3 stages)  | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f32_size8_radix2.s` |
| complex64  | radix-8 (1 stage)   | none | Go     | ✓      | `dit_size8.go`                               |
| complex64  | radix-8 (1 stage)   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f32_size8_radix8.s` |
| complex64  | radix-8 (1 stage)   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f32_size8_radix8.s` |
| complex64  | mixed-radix (2 st.) | none | Go     | ✓      | `dit_size8.go`                               |
| complex64  | mixed-radix (2 st.) | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f32_size8_radix4.s` |
| complex64  | mixed-radix (2 st.) | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f32_size8_radix4.s` |
| complex128 | radix-2 (3 stages)  | none | Go     | ✓      | `dit_size8.go`                               |
| complex128 | radix-2 (3 stages)  | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f64_size8_radix2.s` |
| complex128 | radix-8 (1 stage)   | none | Go     | ✓      | `dit_size8.go`                               |
| complex128 | radix-8 (1 stage)   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f64_size8_radix8.s` |
| complex128 | mixed-radix (2 st.) | none | Go     | ✓      | `dit_size8.go`                               |
| complex128 | mixed-radix (2 st.) | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f64_size8_radix4.s` |

**Notes:**

- Mixed-radix: 1 radix-4 stage + 1 radix-2 stage (reduces from 3 to 2 stages)
- AVX2 mixed-radix uses scalar-style SIMD pattern from size-4
- AVX2 size-8 codelets exist but are not registered by default because they are slower than Go
- Binary bit-reversal used (8 is not a power of 4)

### Size 16

| Type       | Algorithm | SIMD | Source | Status | Files                                         |
| ---------- | --------- | ---- | ------ | ------ | --------------------------------------------- |
| complex64  | radix-2   | none | Go     | ✓      | `dit_size16_radix2.go`                        |
| complex64  | radix-2   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f32_size16_radix2.s` |
| complex64  | radix-4   | none | Go     | ✓      | `dit_size16_radix4.go`                        |
| complex64  | radix-4   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f32_size16_radix4.s` |
| complex64  | radix-4   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f32_size16_radix4.s` |
| complex64  | radix-2   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f32_size16_radix2.s` |
| complex128 | radix-2   | none | Go     | ✓      | `dit_size16_radix2.go`                        |
| complex128 | radix-2   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f64_size16_radix2.s` |
| complex128 | radix-4   | none | Go     | ✓      | `dit_size16_radix4.go`                        |
| complex128 | radix-4   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f64_size16_radix4.s` |
| complex128 | radix-2   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f64_generic.s`       |

**Notes:**

- Radix-4 variant is 12-15% faster (per benchmarks)
- Radix-4 uses radix-4 bit-reversal indices
- AVX2 radix-4 variant implemented for both precisions

### Size 32

| Type       | Algorithm | SIMD | Source | Status | Files                                         |
| ---------- | --------- | ---- | ------ | ------ | --------------------------------------------- |
| complex64  | radix-2   | none | Go     | ✓      | `dit_size32.go`                               |
| complex64  | radix-2   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f32_size32_radix2.s` |
| complex64  | radix-2   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f32_size32_radix2.s` |
| complex64  | mixed-2/4 | none | Go     | ✓      | `dit_size32_mixed24.go`                       |
| complex64  | mixed-2/4 | AVX2 | Wrap   | ✓      | `dit_size32_mixed24_avx2.go`                  |
| complex64  | mixed-2/4 | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f32_size32_mixed24.s`|
| complex128 | radix-2   | none | Go     | ✓      | `dit_size32.go`                               |
| complex128 | radix-2   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f64_size32_radix2.s` |
| complex128 | mixed-2/4 | none | Go     | ✓      | `dit_size32_mixed24.go`                       |
| complex128 | mixed-2/4 | AVX2 | Wrap   | ✓      | `dit_size32_mixed24_avx2.go`                  |

**Notes:**

- Radix-2 variant: 5 stages (2^5 = 2×2×2×2×2)
- Mixed-2/4 variant: 3 stages (4×4×2); Go implementation delegates to radix-2 for correctness
- Mixed-2/4 AVX2: Wraps existing AVX2 radix-2 assembly with same guarantees
- Mixed-2/4 NEON: Dedicated mixed-radix assembly (uses mixed-2/4 bit-reversal)
- Standard binary bit-reversal indices (not radix-4 reversal)
- Mixed-radix priority: Generic (15) > Radix-2 (0), AVX2 mixed (25) > AVX2 radix-2 (20)

### Size 64

| Type       | Algorithm | SIMD | Source | Status | Files                                         |
| ---------- | --------- | ---- | ------ | ------ | --------------------------------------------- |
| complex64  | radix-2   | none | Go     | ✓      | `dit_size64.go`                               |
| complex64  | radix-2   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f32_size64_radix2.s` |
| complex64  | radix-2   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f32_size64_radix2.s` |
| complex64  | radix-4   | none | Go     | ✓      | `dit_size64_radix4.go`                        |
| complex64  | radix-4   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f32_size64_radix4.s` |
| complex64  | radix-4   | SSE2 | Asm    | ✓      | `internal/asm/amd64/sse2_f32_size64_radix4.s` |
| complex64  | radix-4   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f32_size64_radix4.s` |
| complex128 | radix-2   | none | Go     | ✓      | `dit_size64.go`                               |
| complex128 | radix-2   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f64_size64_radix2.s` |
| complex128 | radix-2   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f64_generic.s`       |
| complex128 | radix-4   | none | Go     | ✓      | `dit_size64_radix4.go`                        |
| complex128 | radix-4   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f64_size64_radix4.s` |

**Notes:**

- Radix-4 uses radix-4 bit-reversal indices
- SSE2 size-64 radix-4 newly added for systems without AVX2
- Complex128 AVX2 implements both radix-2 (6 stages) and radix-4 (3 stages)

### Size 128

| Type       | Algorithm | SIMD | Source | Status | Files                                          |
| ---------- | --------- | ---- | ------ | ------ | ---------------------------------------------- |
| complex64  | radix-2   | none | Go     | ✓      | `dit_size128.go`                               |
| complex64  | radix-2   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f32_size128_radix4.s` |
| complex64  | radix-2   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f32_size128_radix2.s` |
| complex64  | mixed-2/4 | none | Go     | ✓      | `dit_size128_mixed24.go`                       |
| complex64  | mixed-2/4 | AVX2 | Wrap   | ✓      | `dit_size128_mixed24_avx2.go`                  |
| complex64  | mixed-2/4 | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f32_size128_mixed24.s`|
| complex128 | radix-2   | none | Go     | ✓      | `dit_size128.go`                               |
| complex128 | radix-2   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f64_size128_radix4.s` |
| complex128 | radix-2   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f64_generic.s`        |
| complex128 | mixed-2/4 | none | Go     | ✓      | `dit_size128_mixed24.go`                       |
| complex128 | mixed-2/4 | AVX2 | Wrap   | ✓      | `dit_size128_mixed24_avx2.go`                  |

**Notes:**

- Radix-2 variant: 7 stages (2^7 = 2×2×2×2×2×2×2)
- Mixed-2/4 variant: 3 stages (4×4×2) - delegates to proven radix-2 for guaranteed correctness
- Mixed-2/4 AVX2: Wraps existing AVX2 radix-2 assembly with same guarantees
- Mixed-2/4 NEON: Dedicated mixed-radix assembly
- Complex128 AVX2 radix-2: Fully optimized 7-stage assembly
- Standard binary bit-reversal indices (not radix-4 reversal)
- Mixed-radix priority: Generic (15) > Radix-2 (0), AVX2 mixed (25) > AVX2 radix-2 (20)

### Size 256

| Type       | Algorithm | SIMD | Source | Status | Files                                          |
| ---------- | --------- | ---- | ------ | ------ | ---------------------------------------------- |
| complex64  | radix-2   | none | Go     | ✓      | `dit_size256_radix2.go`                        |
| complex64  | radix-2   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f32_size256_radix2.s` |
| complex64  | radix-2   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f32_size256_radix2.s` |
| complex64  | radix-4   | none | Go     | ✓      | `dit_size256_radix4.go`                        |
| complex64  | radix-4   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f32_size256_radix4.s` |
| complex64  | radix-4   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f32_size256_radix4.s` |
| complex128 | radix-2   | none | Go     | ✓      | `dit_size256_radix2.go`                        |
| complex128 | radix-2   | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f64_size256_radix2.s` |
| complex128 | radix-2   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f64_generic.s`        |
| complex128 | radix-4   | none | Go     | ✓      | `dit_size256_radix4.go`                        |

**Notes:**

- Radix-4 variant potentially faster (higher priority in registry)
- AVX2 radix-4: Fully implemented for complex64 (forward + inverse)
- Complex128 AVX2 radix-2: Fully optimized 8-stage assembly
- Radix-4 uses radix-4 bit-reversal indices

### Size 512

| Type       | Algorithm | SIMD | Source | Status | Files                                          |
| ---------- | --------- | ---- | ------ | ------ | ---------------------------------------------- |
| complex64  | radix-2   | none | Go     | ✓      | `dit_size512.go`                               |
| complex64  | radix-2   | AVX2 | Wrap   | ✓      | `internal/asm/amd64/avx2_f32_size512_radix2.s` |
| complex64  | radix-2   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f32_size512_radix2.s` |
| complex64  | mixed⁴    | none | Go     | ✓      | `dit_size512_mixed24.go`                       |
| complex64  | mixed⁴    | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f32_size512_mixed24.s`|
| complex128 | radix-2   | none | Go     | ✓      | `dit_size512.go`                               |
| complex128 | radix-2   | NEON | Asm    | ✓      | `internal/asm/arm64/neon_f64_generic.s`        |
| complex128 | mixed⁴    | none | Go     | ✓      | `dit_size512_mixed24.go`                       |
| complex128 | mixed⁴    | AVX2 | Asm    | ✓      | `internal/asm/amd64/avx2_f64_size512_mixed24.s`|

**Notes:**

- Mixed⁴ variant: 4 radix-4 stages + 1 radix-2 stage (5 total vs 9 for pure radix-2)
- Uses `ComputeBitReversalIndicesMixed24()` for mixed-radix bit-reversal
- AVX2 mixed⁴ uses dedicated assembly; AVX2 radix-2 wrapper delegates to the generic AVX2 kernel

### Size 1024

| Type       | Algorithm | SIMD | Source | Status | Files                    |
| ---------- | --------- | ---- | ------ | ------ | ------------------------ |
| complex64  | radix-4   | none | Go     | ✓      | `dit_size1024_radix4.go` |
| complex128 | radix-4   | none | Go     | ✓      | `dit_size1024_radix4.go` |

**Notes:**

- Pure radix-4: 5 stages (4⁵ = 1024)
- Uses `ComputeBitReversalIndicesRadix4()` for radix-4 bit-reversal

### Size 2048

| Type       | Algorithm | SIMD | Source | Status | Files                     |
| ---------- | --------- | ---- | ------ | ------ | ------------------------- |
| complex64  | mixed⁴    | none | Go     | ✓      | `dit_size2048_mixed24.go` |
| complex128 | mixed⁴    | none | Go     | ✓      | `dit_size2048_mixed24.go` |

**Notes:**

- Mixed⁴ variant: 5 radix-4 stages + 1 radix-2 stage (6 total vs 11 for pure radix-2)
- Uses `ComputeBitReversalIndicesMixed24()` for mixed-radix bit-reversal

### Size 4096

| Type       | Algorithm | SIMD | Source | Status | Files                    |
| ---------- | --------- | ---- | ------ | ------ | ------------------------ |
| complex64  | radix-4   | none | Go     | ✓      | `dit_size4096_radix4.go` |
| complex128 | radix-4   | none | Go     | ✓      | `dit_size4096_radix4.go` |

**Notes:**

- Pure radix-4: 6 stages (4⁶ = 4096)
- Uses `ComputeBitReversalIndicesRadix4()` for radix-4 bit-reversal

### Size 8192

| Type       | Algorithm | SIMD | Source | Status | Files                     |
| ---------- | --------- | ---- | ------ | ------ | ------------------------- |
| complex64  | mixed⁴    | none | Go     | ✓      | `dit_size8192_mixed24.go` |
| complex128 | mixed⁴    | none | Go     | ✓      | `dit_size8192_mixed24.go` |

**Notes:**

- Mixed⁴ variant: 6 radix-4 stages + 1 radix-2 stage (7 total vs 13 for pure radix-2)
- Uses `ComputeBitReversalIndicesMixed24()` for mixed-radix bit-reversal

### Size 16384

| Type       | Algorithm | SIMD | Source | Status | Files                     |
| ---------- | --------- | ---- | ------ | ------ | ------------------------- |
| complex64  | radix-4   | none | Go     | ✓      | `dit_size16384_radix4.go` |
| complex128 | radix-4   | none | Go     | ✓      | `dit_size16384_radix4.go` |

**Notes:**

- Pure radix-4: 7 stages (4⁷ = 16384)
- Uses `ComputeBitReversalIndicesRadix4()` for radix-4 bit-reversal

## Coverage Summary

### Pure Go Implementations

All sizes have complete Go implementations for both `complex64` and `complex128`:

- **Size 4**: 1 variant each (radix-4)
- **Size 8**: 3 variants each (radix-2, radix-8, mixed-radix)
- **Size 16**: 2 variants each (radix-2, radix-4)
- **Size 32**: 2 variants each (radix-2, mixed-radix)
- **Size 64**: 2 variants each (radix-2, radix-4)
- **Size 128**: 2 variants each (radix-2, mixed-radix)
- **Size 256**: 2 variants each (radix-2, radix-4)
- **Size 512**: 2 variants each (radix-2, mixed⁴)
- **Size 1024**: 1 variant each (radix-4)
- **Size 2048**: 1 variant each (mixed⁴)
- **Size 4096**: 1 variant each (radix-4)
- **Size 8192**: 1 variant each (mixed⁴)
- **Size 16384**: 1 variant each (radix-4)

**Total:** 48 implementations (24 complex64 + 24 complex128)

### AVX2 Assembly Implementations

AVX2 optimizations exist for both `complex64` and `complex128`:

- **Size 4**: 2 variants (radix-4 complex64, radix-4 complex128)
- **Size 8**: 3 variants complex64 (radix-2, radix-8, mixed-radix) + 1 variant complex128 (radix-2)
- **Size 16**: 2 variants each (radix-2, radix-4 for both 64/128)
- **Size 32**: 2 variants each (radix-2, mixed-radix for both 64/128)
- **Size 64**: 2 variants complex64 (radix-2, radix-4) + 2 variants complex128 (radix-2, radix-4) + SSE2 fallback
- **Size 128**: 2 variants complex64 (radix-2, mixed-radix) + 2 variants complex128 (radix-2, mixed-radix)
- **Size 256**: 2 variants complex64 (radix-2, radix-4) + 1 variant complex128 (radix-2)
- **Size 512**: 3 variants (radix-2 complex64 wrapper, mixed⁴ complex64, mixed⁴ complex128)

**Total:** 33 complete implementations

## Missing Implementations

### Critical Missing Features

1. **Size 1024+ AVX2**
   - Sizes 1024, 2048, 4096, 8192, 16384 use Pure Go implementations

2. **Complex128 AVX2 Radix-4 for Sizes 64+**
   - Size 64: Radix-4 AVX2 implemented ✓
   - Sizes 128, 256: Radix-4 AVX2 not yet implemented (radix-2 available)

### Potential Optimizations

The following sizes could benefit from additional variants:

1. **Size 32 AVX2 Radix-4** (`complex64`/`complex128`)
   - Could reduce stages and improve performance

2. **Size 128 AVX2 Radix-4** (`complex64`/`complex128`)
   - Could reduce stages and improve performance

3. **Size 256 AVX2 Radix-4** (`complex128`)
   - Could reduce stages and improve performance (complex64 already has this)

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

- Build tag: `asm`
- Architecture: `amd64`
- Excluded by: `purego`

Build constraint: `//go:build amd64 && asm && !purego`

## Testing

All implementations are validated against:

- Naive O(n²) reference DFT
- Round-trip tests: `Inverse(Forward(x)) ≈ x`
- Property tests: Parseval's theorem, linearity, shift theorems
- Cross-validation: Assembly vs Pure Go

---

_Generated: 2025-12-31_
_See: `internal/fft/inventory_check.go` for automated inventory generation_

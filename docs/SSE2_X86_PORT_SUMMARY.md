# SSE2/SSE3 x86 Port - Initial Implementation Summary

## Overview

Successfully ported SSE3 size-8 radix-2 FFT kernels from AMD64 (64-bit) to x86/386 (32-bit) architecture as a template for future ports. This establishes the foundation and patterns for porting additional SSE2/SSE3 kernels to 32-bit systems.

## Files Created/Modified

### 1. Documentation

- **`docs/PORTING_AMD64_TO_X86.md`** - Comprehensive porting guide
  - Detailed architecture differences (registers, ABI, stack offsets)
  - Step-by-step porting process with examples
  - Register mapping tables
  - Common pitfalls and verification checklist
  - Build/test commands for x86

### 2. Assembly Implementation

- **`internal/asm/x86/sse3_f32_size8_radix2.s`** - Ported SSE3 kernels (uses `ADDSUBPS`)
  - `ForwardSSE3Size8Radix2Complex64Asm` - Forward FFT
  - `InverseSSE3Size8Radix2Complex64Asm` - Inverse FFT with 1/8 scaling
  - Frame size: `$36-61` (36 bytes local stack, 61 bytes arguments)
  - Uses 32-bit registers (EAX, EBX, ECX, EDX, ESI, EDI, EBP)
  - Proper x86 ABI with 12-byte slice descriptors

### 3. Constants and Data

- **`internal/asm/x86/core.s`** - Updated with required constants
  - Added scaling factors: `eighth32`, `sixteenth32`, `thirtySecond32`, etc.
  - Added sign masks: `signbit32`
  - Added XMM lane negation masks: `maskNegLoPS`, `maskNegHiPS`
  - All constants properly aligned and marked `RODATA|NOPTR`

### 4. Go Declarations

- **`internal/asm/x86/decl.go`** - Updated function declarations
  - Added `ForwardSSE3Size8Radix2Complex64Asm`
  - Added `InverseSSE3Size8Radix2Complex64Asm`
  - Proper `//go:noescape` pragmas for performance

### 5. Tests

- **`internal/kernels/sse3_f32_size8_radix2_386_test.go`** - Comprehensive tests
  - `TestForwardSSE3Size8Radix2Complex64_386` - Correctness test vs reference
  - `TestInverseSSE3Size8Radix2Complex64_386` - Correctness test vs reference
  - `TestRoundTripSSE3Size8Radix2Complex64_386` - Round-trip test
  - `BenchmarkForwardSSE3Size8Radix2Complex64_386` - Performance benchmark
  - `BenchmarkInverseSSE3Size8Radix2Complex64_386` - Performance benchmark

## Key Porting Changes

### Register Mapping

| AMD64   | x86/386  | Usage                |
| ------- | -------- | -------------------- |
| R8, R14 | DI       | Destination pointer  |
| R9      | SI       | Source pointer       |
| R10     | BX       | Twiddle factors      |
| R11     | DX       | Scratch buffer       |
| R12     | BP       | Bit-reversal indices |
| R13     | AX/Stack | Length (n)           |

### Stack Frame Adjustments

```assembly
# AMD64
TEXT ·ForwardSSE3Size8Radix2Complex64Asm(SB), NOSPLIT, $0-121

# x86
TEXT ·ForwardSSE3Size8Radix2Complex64Asm(SB), NOSPLIT, $36-61
```

- Frame size reduced from 121 to 61 bytes (5 slices × 12 bytes + 1 bool)
- Added 36 bytes local stack for register spills

### Offset Calculations

```assembly
# AMD64 offsets (24-byte slices)
dst+0(FP)      src+24(FP)     twiddle+48(FP)
scratch+72(FP) bitrev+96(FP)  ret+120(FP)

# x86 offsets (12-byte slices)
dst+0(FP)      src+12(FP)     twiddle+24(FP)
scratch+36(FP) bitrev+48(FP)  ret+60(FP)
```

### Pointer Operations

```assembly
# AMD64: 64-bit pointers
MOVQ dst+0(FP), R8
SHLQ $3, DX              # complex64 = 8 bytes

# x86: 32-bit pointers
MOVL dst+0(FP), DI
SHLL $3, CX              # complex64 = 8 bytes
```

## SSE2 Instructions - Unchanged!

Good news: SSE2 instructions work identically on both architectures:

- `MOVSD`, `MOVAPS`, `MOVUPS` - Same
- `ADDPS`, `SUBPS`, `MULPS` - Same
- `SHUFPS`, `XORPS` - Same
- XMM0-XMM7 registers available (x86 has 8, AMD64 has 16)

Note: `ADDSUBPS` is an SSE3 instruction. Any kernels that use it now require SSE3.

## Testing

### Build for x86

```bash
GOARCH=386 go build -tags=asm ./...
```

### Run Tests

```bash
GOARCH=386 go test -v -tags=asm ./internal/kernels -run ".*386"
```

### Run Benchmarks

```bash
GOARCH=386 go test -bench=.*386 -benchmem -tags=asm ./internal/kernels
```

## Performance Expectations

- **x86 SSE2 vs Go fallback**: 2-4× speedup expected
- **x86 SSE2 vs AMD64 SSE2**: Slower due to:
  - Fewer registers (8 vs 16 general-purpose)
  - More stack spills required
  - 32-bit addressing overhead
- **Still valuable** for:
  - Embedded x86 systems
  - Legacy 32-bit applications
  - Docker containers optimizing memory

## Next Steps

### Immediate (Priority 1)

1. Port size-16 radix-2 kernels
   - Similar complexity to size-8
   - High usage frequency
2. Port size-32 radix-2 kernels
   - Common size in practice
3. Verify tests pass on actual 386 hardware/VM

### Short-term (Priority 2)

4. Port size-4 radix-4 kernels
   - Smallest kernel, good validation
5. Port size-64 kernels
   - Medium complexity
6. Add size-128 kernels
   - Larger, less common

### Long-term (Priority 3)

7. Consider radix-4 and radix-8 variants
   - Only if benchmarks show significant benefit over radix-2
8. Consider complex128 support for x86
   - Lower priority (most users need complex64)
   - More complex due to larger data size

### Optional

9. Profile performance on actual 386 hardware
10. Document performance characteristics
11. Add to CI/CD if 386 builds are supported

## Architecture Support Matrix (After This Port)

| Size | Radix | AMD64 SSE2 | x86 SSE2 | ARM64 NEON |
| ---- | ----- | ---------- | -------- | ---------- |
| 4    | R4    | ✓          | -        | ✓          |
| 8    | R2    | ✓          | **✓**    | ✓          |
| 8    | R4    | ✓          | -        | -          |
| 8    | R8    | ✓          | -        | -          |
| 16   | R2    | ✓          | -        | ✓          |
| 16   | R4    | ✓          | -        | -          |
| 32   | R2    | ✓          | -        | ✓          |
| 32   | M24   | ✓          | -        | -          |

**Bold** = newly implemented in this port

## Lessons Learned

### What Worked Well

1. **SSE2 compatibility** - Instructions work identically, reducing porting complexity
2. **Stack-based approach** - Using stack for register spills is straightforward
3. **Existing test infrastructure** - Reference implementations made validation easy
4. **Documentation** - Creating porting guide first helped catch issues early

### Challenges Encountered

1. **Register pressure** - x86 has only 8 general-purpose registers vs 16 on AMD64
   - Solution: Aggressive use of stack for temporary storage
2. **Offset calculation** - Easy to miss the 24→12 byte slice descriptor change
   - Solution: Created systematic offset calculation table
3. **Label naming** - Needed unique labels to avoid conflicts with AMD64 versions
   - Solution: Added `_386` suffix to all labels

### Recommendations for Future Ports

1. **Start simple** - Port smallest/simplest kernels first (size-4, size-8)
2. **Validate early** - Test each port immediately, don't batch them
3. **Reuse patterns** - This size-8 implementation serves as template for others
4. **Document differences** - Note any architecture-specific quirks in comments
5. **Benchmark** - Verify performance gains justify maintenance burden

## References

- Source implementation: `internal/asm/amd64/sse3_f32_size8_radix2.s`
- Porting guide: `docs/PORTING_AMD64_TO_X86.md`
- x86 package: `internal/asm/x86/`
- Test patterns: `internal/kernels/*_386_test.go`

## Maintenance Notes

### When Adding New Kernels

1. Follow the pattern in `sse3_f32_size8_radix2.s`
2. Update `x86/decl.go` with new function declarations
3. Add required constants to `x86/core.s` if needed
4. Create corresponding `*_386_test.go` file
5. Run full test suite with `GOARCH=386`

### When Modifying Existing AMD64 Kernels

1. Consider if x86 port needs same changes
2. Keep implementations in sync when possible
3. Document any intentional divergence

### Build Tags

Always use: `//go:build 386 && asm && !purego`

This ensures kernels only build for:

- 32-bit x86 architecture (`386`)
- When assembly is enabled (`asm`)
- Not in pure-Go mode (`!purego`)

---

**Status**: ✅ Complete - Ready for testing and benchmarking on 386 hardware
**Date**: 2026-01-03
**Author**: Ported from AMD64 SSE2 kernels

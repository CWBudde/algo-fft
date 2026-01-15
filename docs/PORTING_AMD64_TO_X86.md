# Porting SSE2 Kernels from AMD64 to x86 (386)

## Overview

This guide describes how to port SSE2-optimized FFT kernels from 64-bit AMD64 to 32-bit x86 (386) architecture. While SSE2 instructions are largely compatible, the different ABIs and register sets require careful adaptation.

## Architecture Differences Summary

| Aspect                   | AMD64 (64-bit)                                               | x86/386 (32-bit)                                    |
| ------------------------ | ------------------------------------------------------------ | --------------------------------------------------- |
| **General Registers**    | 16 × 64-bit (RAX, RBX, RCX, RDX, RSI, RDI, RBP, RSP, R8-R15) | 8 × 32-bit (EAX, EBX, ECX, EDX, ESI, EDI, EBP, ESP) |
| **Pointer Size**         | 8 bytes                                                      | 4 bytes                                             |
| **Pointer Instructions** | MOVQ, LEAQ                                                   | MOVL, LEAL                                          |
| **Slice Descriptor**     | 24 bytes (ptr:8 + len:8 + cap:8)                             | 12 bytes (ptr:4 + len:4 + cap:4)                    |
| **SSE2 Registers**       | XMM0-XMM15 (128-bit)                                         | XMM0-XMM7 (128-bit)                                 |
| **Stack Alignment**      | 16-byte                                                      | 16-byte (for SSE2)                                  |

## Step-by-Step Porting Process

### 1. Update Build Tag

**AMD64:**

```assembly
//go:build amd64 && asm && !purego
```

**x86:**

```assembly
//go:build 386 && asm && !purego
```

### 2. Adjust Stack Frame Size

The frame size calculation changes due to different slice layouts:

**AMD64 Example:**

```assembly
TEXT ·ForwardSSE3Size8Radix2Complex64Asm(SB), NOSPLIT, $0-121
```

- Frame size: 121 bytes
- Calculation: 5 slices × 24 bytes + 1 bool = 120 + 1 = 121

**x86 Equivalent:**

```assembly
TEXT ·ForwardSSE3Size8Radix2Complex64Asm(SB), NOSPLIT, $0-61
```

- Frame size: 61 bytes
- Calculation: 5 slices × 12 bytes + 1 bool = 60 + 1 = 61

### 3. Calculate Stack Offsets

**AMD64 Offsets:**

```assembly
// func kernel(dst, src, twiddle, scratch []complex64, bitrev []int) bool
dst+0(FP)       // ptr at 0,   len at 8,   cap at 16
src+24(FP)      // ptr at 24,  len at 32,  cap at 40
twiddle+48(FP)  // ptr at 48,  len at 56,  cap at 64
scratch+72(FP)  // ptr at 72,  len at 80,  cap at 88
bitrev+96(FP)   // ptr at 96,  len at 104, cap at 112
ret+120(FP)     // bool at 120
```

**x86 Offsets:**

```assembly
// func kernel(dst, src, twiddle, scratch []complex64, bitrev []int) bool
dst+0(FP)       // ptr at 0,  len at 4,  cap at 8
src+12(FP)      // ptr at 12, len at 16, cap at 20
twiddle+24(FP)  // ptr at 24, len at 28, cap at 32
scratch+36(FP)  // ptr at 36, len at 40, cap at 44
bitrev+48(FP)   // ptr at 48, len at 52, cap at 56
ret+60(FP)      // bool at 60
```

**Quick Formula:**

```
x86_offset = (amd64_offset / 24) * 12  // for slice parameters
```

### 4. Register Mapping

Map AMD64 64-bit registers to x86 32-bit equivalents:

| AMD64 Usage     | AMD64 Register | x86 Register | Notes                   |
| --------------- | -------------- | ------------ | ----------------------- |
| dst pointer     | R8, R14        | DI           | Destination buffer      |
| src pointer     | R9             | SI           | Source buffer           |
| twiddle pointer | R10            | BX           | Twiddle factors         |
| scratch pointer | R11            | DX           | Scratch buffer          |
| bitrev pointer  | R12            | BP           | Bit-reversal indices    |
| length          | R13            | AX or stack  | Often saved to stack    |
| temp index      | DX             | CX or stack  | Loop counters           |
| Extra temps     | R8-R15         | Stack spills | x86 has fewer registers |

**Critical:** x86 has only 8 general-purpose registers vs AMD64's 16. You'll need to use stack more often.

### 5. Convert Pointer Operations

**Loading slice pointers and lengths:**

AMD64:

```assembly
MOVQ dst+0(FP), R8      // dst.ptr
MOVQ src+24(FP), R9     // src.ptr
MOVQ src+32(FP), R13    // src.len (n)
```

x86:

```assembly
MOVL dst+0(FP), DI      // dst.ptr
MOVL src+12(FP), SI     // src.ptr
MOVL src+16(FP), AX     // src.len (n)
```

**Pointer arithmetic:**

AMD64:

```assembly
SHLQ $4, DX             // DX *= 16 (complex128 size)
MOVUPD (R9)(DX*1), X0   // Load from src[index]
```

x86:

```assembly
SHLL $3, DX             // DX *= 8 (complex64 size)
MOVUPS (SI)(DX*1), X0   // Load from src[index]
```

### 6. Stack Usage for Register Pressure

When you run out of registers, save values to stack:

**Allocate stack space:**

```assembly
// AMD64: $0-121 means 0 bytes local stack
// x86 with 36 bytes local stack:
TEXT ·ForwardSSE3Size8Radix2Complex64Asm(SB), NOSPLIT, $36-61
```

**Save/restore pattern:**

```assembly
// Save parameters to stack locals
MOVL DI, 0(SP)      // dst pointer at SP+0
MOVL SI, 4(SP)      // src pointer at SP+4
MOVL BX, 8(SP)      // twiddle pointer at SP+8
MOVL DX, 12(SP)     // scratch pointer at SP+12
MOVL BP, 16(SP)     // bitrev pointer at SP+16
MOVL AX, 20(SP)     // n at SP+20

// Later, reload when needed
MOVL 8(SP), BX      // Reload twiddle pointer
```

### 7. SSE2 Instructions - Same on Both Architectures!

Good news: SSE2 instructions work identically:

```assembly
// These are the same on amd64 and x86:
MOVUPS (SI), X0         // Load 16 bytes unaligned
MOVAPS X0, X1           // Move aligned packed single
ADDPS X1, X0            // Add packed single
MULPS X2, X1            // Multiply packed single
SHUFPS $0xB1, X0, X0    // Shuffle
XORPS X3, X0            // XOR (for sign flips)

// For complex64: use PS instructions (packed single-precision)
// For complex128: use PD instructions (packed double-precision)
```

**XMM register differences:**

- AMD64: XMM0-XMM15 available
- x86: XMM0-XMM7 available (8 registers, not 16)

### 8. XMM Register Constraint - CRITICAL!

**IMPORTANT:** x86 (32-bit) only has XMM0-XMM7, not XMM8-XMM15!

This is a major difference from AMD64:

- AMD64: XMM0-XMM15 (16 registers)
- x86: XMM0-XMM7 (8 registers)

**Impact:**

- Cannot use X8, X9, X10, X11, X12, X13, X14, X15
- Must reorganize code to use only X0-X7
- May require more memory operations for temporary storage
- Memory-based approach often necessary for complex calculations

**Workarounds:**

1. **Store to memory between stages** - Write intermediate results to stack/buffer
2. **Reuse registers cleverly** - Overwrite values no longer needed
3. **Process one butterfly at a time** - Load, compute, store, repeat

See `internal/asm/x86/sse3_f32_size8_radix2.s` for a working example.

### 9. Data Size Differences

Remember to adjust for data type sizes:

**complex64 (8 bytes):**

```assembly
SHLL $3, index_reg      // Multiply index by 8
MOVUPS (ptr)(index*1), X0
```

**complex128 (16 bytes):**

```assembly
SHLL $4, index_reg      // Multiply index by 16
MOVUPD (ptr)(index*1), X0
```

### 9. Label Naming Convention

Use different label prefixes to avoid conflicts with AMD64 versions:

AMD64:

```assembly
fwd_err:
fwd_use_dst:
fwd_stage1:
```

x86:

```assembly
fwd32_err:          // Add '32' suffix for 32-bit
fwd32_use_dst:
fwd32_stage1:
```

Or use more descriptive prefixes:

```assembly
sse2_386_fwd_err:
sse2_386_use_dst:
```

### 10. Constants and Shared Data

Constants need to be defined in `x86/core.s` separately from `amd64/core.s`.

**For complex64, add to x86/core.s:**

```assembly
// Sign bit masks for negation via XOR
DATA ·maskNegLoPS+0(SB)/4, $0x80000000
DATA ·maskNegLoPS+4(SB)/4, $0x00000000
DATA ·maskNegLoPS+8(SB)/4, $0x80000000
DATA ·maskNegLoPS+12(SB)/4, $0x00000000
GLOBL ·maskNegLoPS(SB), RODATA|NOPTR, $16

DATA ·maskNegHiPS+0(SB)/4, $0x00000000
DATA ·maskNegHiPS+4(SB)/4, $0x80000000
DATA ·maskNegHiPS+8(SB)/4, $0x00000000
DATA ·maskNegHiPS+12(SB)/4, $0x80000000
GLOBL ·maskNegHiPS(SB), RODATA|NOPTR, $16
```

**Scale factors for inverse FFT:**

```assembly
DATA ·eighth32+0(SB)/4, $0x3e000000  // 0.125f = 1/8
GLOBL ·eighth32(SB), RODATA|NOPTR, $4

DATA ·sixteenth32+0(SB)/4, $0x3d800000  // 0.0625f = 1/16
GLOBL ·sixteenth32(SB), RODATA|NOPTR, $4
```

### 11. Testing the Port

After porting, add corresponding test files:

**File:** `internal/kernels/sse3_f32_size8_radix2_386_test.go`

```go
//go:build 386 && asm && !purego

package kernels

import "testing"

func TestSSE3Size8Radix2Complex64_386(t *testing.T) {
    // Test implementation
}
```

## Complete Example: Size-8 Radix-2 Complex64

See the actual ported implementation in:

- Source: `internal/asm/x86/sse3_f32_size8_radix2.s`
- Tests: `internal/kernels/sse3_f32_size8_radix2_386_test.go`

## Common Pitfalls

### ❌ Pitfall 1: Forgetting to adjust offsets

```assembly
// WRONG - using amd64 offset on x86
MOVL src+24(FP), SI    // Should be src+12(FP)!
```

### ❌ Pitfall 2: Using 64-bit instructions

```assembly
// WRONG - MOVQ doesn't work for pointers on x86
MOVQ dst+0(FP), R8     // Should use MOVL with 32-bit register

// CORRECT
MOVL dst+0(FP), DI
```

### ❌ Pitfall 3: Register name typos

```assembly
// WRONG - R8 doesn't exist on x86
MOVL (R8), EAX

// CORRECT - use one of the 8 available registers
MOVL (DI), EAX
```

### ❌ Pitfall 4: Wrong frame size

```assembly
// WRONG - using amd64 frame size
TEXT ·ForwardSSE3Size8Radix2Complex64Asm(SB), NOSPLIT, $0-121

// CORRECT for x86
TEXT ·ForwardSSE3Size8Radix2Complex64Asm(SB), NOSPLIT, $36-61
```

## Verification Checklist

Before committing a ported kernel:

- [ ] Build tag is `//go:build 386 && asm && !purego`
- [ ] Frame size calculated correctly (slices × 12 + return)
- [ ] All stack offsets adjusted to x86 ABI
- [ ] Only 32-bit registers used (EAX, EBX, ECX, EDX, ESI, EDI, EBP, ESP)
- [ ] Pointer operations use MOVL/LEAL (not MOVQ/LEAQ)
- [ ] Data size multipliers correct (×8 for complex64, not ×16)
- [ ] Required constants added to `x86/core.s`
- [ ] Function declared in `x86/decl.go` with `//go:noescape`
- [ ] Test file created with same build tags
- [ ] Labels renamed to avoid conflicts
- [ ] Tests pass with `-tags=asm` on 386 architecture

## Building and Testing for x86

```bash
# Build for x86 (32-bit)
GOARCH=386 go build -tags=asm ./...

# Run tests for x86
GOARCH=386 go test -v -tags=asm ./internal/kernels/...

# Benchmark on x86
GOARCH=386 go test -bench=. -benchmem -tags=asm ./...
```

## Performance Considerations

**Why port to x86 when most systems are 64-bit?**

1. **Embedded systems**: Many embedded x86 devices still run 32-bit
2. **Legacy support**: Some industrial/scientific applications require 32-bit
3. **Docker containers**: 32-bit containers on 64-bit hosts for memory savings
4. **Completeness**: Ensure the library works across all Go-supported architectures

**Performance expectations:**

- x86 SSE2 kernels will be slower than AMD64 due to:
  - Fewer registers → more stack spills
  - 32-bit pointers still require same memory bandwidth
  - Only 8 XMM registers vs 16
- But still **much faster** than pure Go fallback
- Expected speedup: 2-4× over Go on x86 (vs 4-8× on AMD64)

## Next Steps

1. Port size-8, size-16, size-32 kernels (most common sizes)
2. Focus on radix-2 first (simplest algorithm)
3. Add complex64 support before complex128 (more widely used)
4. Consider radix-4 only if benchmarks show significant gains
5. Document performance characteristics in comments

## References

- [Go Assembly Documentation](https://go.dev/doc/asm)
- [Intel SSE2 Instruction Reference](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [x86 Calling Conventions](https://en.wikipedia.org/wiki/X86_calling_conventions)
- This codebase: `internal/asm/amd64/` (source material)
- This codebase: `internal/asm/x86/` (ported implementations)

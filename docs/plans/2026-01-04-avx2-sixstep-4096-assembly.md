# AVX2 Six-Step 4096 FFT Assembly Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a fully AVX2-optimized 4096-point FFT using the six-step (64x64 matrix) algorithm entirely in Plan9 assembly for maximum performance.

**Architecture:** The six-step algorithm decomposes 4096 = 64×64 into matrix operations: transpose → row FFTs → transpose+twiddle → row FFTs → transpose. By implementing the transpose operations with AVX2 8×8 block transposes using VUNPCK/VPERM instructions and calling the existing size-64 AVX2 FFT kernel for row transforms, we can achieve significant parallelism and cache efficiency.

**Tech Stack:** Plan9 assembly (Go asm), AVX2 SIMD intrinsics (YMM registers), existing `ForwardAVX2Size64Radix4Complex64Asm` kernel.

---

## Background

### Current Go Implementation Performance

- ~38µs per transform (~870 MB/s)
- Calls existing AVX2 size-64 kernel 128 times (64 row FFTs × 2 passes)
- Transpose and twiddle multiply done in scalar Go loops

### Expected Improvements

- **AVX2 Transpose**: Process 8 complex64 values (32 bytes) per iteration
- **Fused Transpose+Twiddle**: Avoid separate memory passes
- **Inlined Row FFTs**: Eliminate function call overhead for 128 calls
- **Register Blocking**: Keep intermediate values in YMM registers

### Six-Step Algorithm Structure

```
Input[4096] viewed as 64×64 matrix

Step 1: Transpose src → work (column-major to row-major)
Step 2: 64× Row FFTs on work (each row is size-64)
Step 3: Transpose work → dst
Step 4: Twiddle multiply dst[i,j] *= W_4096^(i*j)
Step 5: 64× Row FFTs on dst
Step 6: Transpose dst → work → dst (final output)
```

---

## Task 1: Create Assembly File Structure

**Files:**

- Create: `internal/asm/amd64/avx2_f32_size4096_sixstep.s`

**Step 1: Create the assembly file with header and basic structure**

```asm
//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-4096 Six-Step FFT Kernel for AMD64
// ===========================================================================
//
// Algorithm: Six-Step (64×64 matrix decomposition) FFT
//
// The 4096-point FFT is computed as:
//   1. Transpose: View input as 64×64 matrix, transpose to work buffer
//   2. Row FFTs: 64 × 64-point FFTs on rows (using AVX2 radix-4 kernel)
//   3. Transpose + Twiddle: Transpose and multiply by W_4096^(i*j)
//   4. Row FFTs: 64 × 64-point FFTs on rows
//   5. Final Transpose: Output result
//
// Performance advantages:
//   - Cache-friendly access patterns (row-wise processing)
//   - AVX2 8×8 block transpose (processes 8 complex values at once)
//   - Fused transpose+twiddle eliminates memory pass
//   - All row FFTs use proven AVX2 size-64 kernel
//
// ===========================================================================

#include "textflag.h"

// Constants
// m = 64 (matrix dimension)
// n = 4096 (total size)
// complex64 = 8 bytes (4 bytes real + 4 bytes imag)
// Row size = 64 * 8 = 512 bytes
// Total size = 4096 * 8 = 32768 bytes

TEXT ·ForwardAVX2Size4096SixStepComplex64Asm(SB), NOSPLIT, $0-121
    // Parameters (same layout as other FFT kernels):
    // dst+0(FP)     = dst slice (ptr, len, cap)
    // src+24(FP)    = src slice (ptr, len, cap)
    // twiddle+48(FP)= twiddle slice (ptr, len, cap)
    // scratch+72(FP)= scratch slice (ptr, len, cap)
    // bitrev+96(FP) = bitrev slice (ptr, len, cap) - unused for six-step
    // ret+120(FP)   = return bool

    // TODO: Implement forward transform
    MOVB $0, ret+120(FP)
    RET

TEXT ·InverseAVX2Size4096SixStepComplex64Asm(SB), NOSPLIT, $0-121
    // TODO: Implement inverse transform
    MOVB $0, ret+120(FP)
    RET
```

**Step 2: Verify the file builds**

Run: `go build ./internal/asm/amd64/`
Expected: Build succeeds (stubs return false)

**Step 3: Commit**

```bash
git add internal/asm/amd64/avx2_f32_size4096_sixstep.s
git commit -m "feat: add skeleton for AVX2 six-step 4096 FFT assembly"
```

---

## Task 2: Add Function Declarations

**Files:**

- Modify: `internal/asm/amd64/decl.go` (after line 246)

**Step 1: Add the function declarations**

Add after the Size4096Radix4 declarations:

```go
//go:noescape
func ForwardAVX2Size4096SixStepComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Size4096SixStepComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
```

**Step 2: Verify build**

Run: `go build ./internal/asm/amd64/`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add internal/asm/amd64/decl.go
git commit -m "feat: add declarations for AVX2 six-step 4096 FFT"
```

---

## Task 3: Implement Parameter Loading and Validation

**Files:**

- Modify: `internal/asm/amd64/avx2_f32_size4096_sixstep.s`

**Step 1: Add parameter loading to forward transform**

Replace the TODO section:

```asm
TEXT ·ForwardAVX2Size4096SixStepComplex64Asm(SB), NOSPLIT, $0-121
    // Load parameters
    MOVQ dst+0(FP), R8       // R8  = dst pointer
    MOVQ src+24(FP), R9      // R9  = src pointer
    MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
    MOVQ scratch+72(FP), R11 // R11 = scratch pointer (work buffer)
    MOVQ src+32(FP), R13     // R13 = src length (should be 4096)

    // Verify length == 4096
    CMPQ R13, $4096
    JNE  sixstep_fwd_return_false

    // Validate slice capacities
    MOVQ dst+8(FP), AX
    CMPQ AX, $4096
    JL   sixstep_fwd_return_false

    MOVQ twiddle+56(FP), AX
    CMPQ AX, $4096
    JL   sixstep_fwd_return_false

    MOVQ scratch+80(FP), AX
    CMPQ AX, $4096
    JL   sixstep_fwd_return_false

    // Select working buffer (handle in-place case)
    // If dst == src, we must use scratch as intermediate
    CMPQ R8, R9
    JNE  sixstep_fwd_use_dst
    MOVQ R11, R8             // In-place: use scratch for work

sixstep_fwd_use_dst:
    // R8 = work buffer (either dst or scratch)
    // R9 = src (input)
    // R10 = twiddle factors
    // R11 = scratch (for row FFT scratch space)

    // TODO: Step 1 - Transpose
    JMP sixstep_fwd_done

sixstep_fwd_done:
    // Copy results to dst if we used scratch
    MOVQ dst+0(FP), R9
    CMPQ R8, R9
    JE   sixstep_fwd_ret

    // Copy work → dst (32KB)
    XORQ CX, CX
sixstep_fwd_copy_loop:
    VMOVUPS (R8)(CX*1), Y0
    VMOVUPS 32(R8)(CX*1), Y1
    VMOVUPS Y0, (R9)(CX*1)
    VMOVUPS Y1, 32(R9)(CX*1)
    ADDQ $64, CX
    CMPQ CX, $32768          // 4096 * 8 bytes
    JL   sixstep_fwd_copy_loop

sixstep_fwd_ret:
    VZEROUPPER
    MOVB $1, ret+120(FP)
    RET

sixstep_fwd_return_false:
    VZEROUPPER
    MOVB $0, ret+120(FP)
    RET
```

**Step 2: Verify build**

Run: `go build ./internal/asm/amd64/`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add internal/asm/amd64/avx2_f32_size4096_sixstep.s
git commit -m "feat: add parameter loading for six-step FFT"
```

---

## Task 4: Implement AVX2 64×64 Matrix Transpose

**Files:**

- Modify: `internal/asm/amd64/avx2_f32_size4096_sixstep.s`

The 64×64 matrix transpose processes the data in 8×8 blocks. Each block:

- Loads 8 rows of 8 complex64 values (8 YMM loads = 64 complex values)
- Transposes using VUNPCKLPS/VUNPCKHPS + VPERM2F128
- Stores 8 columns as rows in destination

**Step 1: Add the transpose macro/subroutine**

Add before the main function:

```asm
// ===========================================================================
// 64×64 Matrix Transpose (complex64)
// ===========================================================================
// Transposes a 64×64 matrix of complex64 values using 8×8 block processing.
// Each complex64 = 8 bytes (re, im as float32).
// AVX2 YMM = 32 bytes = 4 complex64 values.
//
// Input:  R8  = destination pointer
//         R9  = source pointer
//         Matrix is 64×64 complex64 = 4096 elements = 32KB
// Clobbers: Y0-Y15, AX, BX, CX, DX, SI, DI, R12, R13, R14, R15

// For a 64×64 matrix with row stride 64:
// Block (bi, bj) starts at src[bi*8*64 + bj*8] = src[bi*512 + bj*8]
// After transpose, it goes to dst[bj*8*64 + bi*8] = dst[bj*512 + bi*8]
// We process 8×8 blocks = 64 complex64 per block
// 64/8 = 8 blocks per dimension → 64 total blocks

sixstep_transpose_64x64:
    // Outer loop: block row (bi = 0..7)
    XORQ R12, R12            // R12 = bi (block row index)

sixstep_transpose_bi_loop:
    CMPQ R12, $8
    JGE  sixstep_transpose_done

    // Inner loop: block col (bj = 0..7)
    XORQ R13, R13            // R13 = bj (block col index)

sixstep_transpose_bj_loop:
    CMPQ R13, $8
    JGE  sixstep_transpose_bi_next

    // Calculate source block address: src + (bi*8*64 + bj*8) * 8
    // = src + (bi*512 + bj*8) * 8 = src + bi*4096 + bj*64
    MOVQ R12, AX
    SHLQ $12, AX             // bi * 4096
    MOVQ R13, BX
    SHLQ $6, BX              // bj * 64
    ADDQ BX, AX
    LEAQ (R9)(AX*1), SI      // SI = src block ptr

    // Calculate dest block address: dst + (bj*8*64 + bi*8) * 8
    // = dst + (bj*512 + bi*8) * 8 = dst + bj*4096 + bi*64
    MOVQ R13, AX
    SHLQ $12, AX             // bj * 4096
    MOVQ R12, BX
    SHLQ $6, BX              // bi * 64
    ADDQ BX, AX
    LEAQ (R8)(AX*1), DI      // DI = dst block ptr

    // Load 8 rows of the 8×8 block (each row = 8 complex64 = 64 bytes = 2 YMM)
    // But we process 4 complex64 at a time (1 YMM), so 16 YMM loads total
    // Row stride = 64 * 8 = 512 bytes

    // Load rows 0-3 (first 4 complex values each)
    VMOVUPS 0(SI), Y0              // row0[0:4]
    VMOVUPS 512(SI), Y1            // row1[0:4]
    VMOVUPS 1024(SI), Y2           // row2[0:4]
    VMOVUPS 1536(SI), Y3           // row3[0:4]

    // 4×4 complex transpose in-register
    // Each YMM has 4 complex = [r0,i0, r1,i1, r2,i2, r3,i3]
    // We want columns: col0 = [row0[0], row1[0], row2[0], row3[0]]

    // Interleave pairs
    VUNPCKLPS Y1, Y0, Y4     // [r0_0,r1_0, i0_0,i1_0, r0_2,r1_2, i0_2,i1_2]
    VUNPCKHPS Y1, Y0, Y5     // [r0_1,r1_1, i0_1,i1_1, r0_3,r1_3, i0_3,i1_3]
    VUNPCKLPS Y3, Y2, Y6
    VUNPCKHPS Y3, Y2, Y7

    // Interleave quads using 128-bit lane permutation
    VPERM2F128 $0x20, Y6, Y4, Y8   // low halves
    VPERM2F128 $0x31, Y6, Y4, Y9   // high halves
    VPERM2F128 $0x20, Y7, Y5, Y10
    VPERM2F128 $0x31, Y7, Y5, Y11

    // Store transposed columns as rows in destination
    VMOVUPS Y8, 0(DI)              // col0 → dst row0
    VMOVUPS Y10, 512(DI)           // col1 → dst row1
    VMOVUPS Y9, 1024(DI)           // col2 → dst row2
    VMOVUPS Y11, 1536(DI)          // col3 → dst row3

    // Load rows 0-3, columns 4-7 (second 4 complex values each)
    VMOVUPS 32(SI), Y0             // row0[4:8]
    VMOVUPS 544(SI), Y1            // row1[4:8]  (512+32)
    VMOVUPS 1056(SI), Y2           // row2[4:8]  (1024+32)
    VMOVUPS 1568(SI), Y3           // row3[4:8]  (1536+32)

    VUNPCKLPS Y1, Y0, Y4
    VUNPCKHPS Y1, Y0, Y5
    VUNPCKLPS Y3, Y2, Y6
    VUNPCKHPS Y3, Y2, Y7

    VPERM2F128 $0x20, Y6, Y4, Y8
    VPERM2F128 $0x31, Y6, Y4, Y9
    VPERM2F128 $0x20, Y7, Y5, Y10
    VPERM2F128 $0x31, Y7, Y5, Y11

    VMOVUPS Y8, 2048(DI)           // col4 → dst row4 (4*512)
    VMOVUPS Y10, 2560(DI)          // col5 → dst row5 (5*512)
    VMOVUPS Y9, 3072(DI)           // col6 → dst row6 (6*512)
    VMOVUPS Y11, 3584(DI)          // col7 → dst row7 (7*512)

    // Load rows 4-7, columns 0-3
    VMOVUPS 2048(SI), Y0           // row4[0:4]
    VMOVUPS 2560(SI), Y1           // row5[0:4]
    VMOVUPS 3072(SI), Y2           // row6[0:4]
    VMOVUPS 3584(SI), Y3           // row7[0:4]

    VUNPCKLPS Y1, Y0, Y4
    VUNPCKHPS Y1, Y0, Y5
    VUNPCKLPS Y3, Y2, Y6
    VUNPCKHPS Y3, Y2, Y7

    VPERM2F128 $0x20, Y6, Y4, Y8
    VPERM2F128 $0x31, Y6, Y4, Y9
    VPERM2F128 $0x20, Y7, Y5, Y10
    VPERM2F128 $0x31, Y7, Y5, Y11

    VMOVUPS Y8, 32(DI)             // Goes to dst[row0, col4-7] = offset 32
    VMOVUPS Y10, 544(DI)
    VMOVUPS Y9, 1056(DI)
    VMOVUPS Y11, 1568(DI)

    // Load rows 4-7, columns 4-7
    VMOVUPS 2080(SI), Y0           // row4[4:8] (2048+32)
    VMOVUPS 2592(SI), Y1           // row5[4:8] (2560+32)
    VMOVUPS 3104(SI), Y2           // row6[4:8] (3072+32)
    VMOVUPS 3616(SI), Y3           // row7[4:8] (3584+32)

    VUNPCKLPS Y1, Y0, Y4
    VUNPCKHPS Y1, Y0, Y5
    VUNPCKLPS Y3, Y2, Y6
    VUNPCKHPS Y3, Y2, Y7

    VPERM2F128 $0x20, Y6, Y4, Y8
    VPERM2F128 $0x31, Y6, Y4, Y9
    VPERM2F128 $0x20, Y7, Y5, Y10
    VPERM2F128 $0x31, Y7, Y5, Y11

    VMOVUPS Y8, 2080(DI)
    VMOVUPS Y10, 2592(DI)
    VMOVUPS Y9, 3104(DI)
    VMOVUPS Y11, 3616(DI)

    INCQ R13
    JMP  sixstep_transpose_bj_loop

sixstep_transpose_bi_next:
    INCQ R12
    JMP  sixstep_transpose_bi_loop

sixstep_transpose_done:
    RET
```

**Step 2: Call transpose in the forward transform**

Replace the TODO line:

```asm
    // Step 1: Transpose src → work
    // Save registers before subroutine call
    PUSHQ R8
    PUSHQ R9
    PUSHQ R10
    PUSHQ R11

    // R8 = work (destination), R9 = src (source)
    CALL sixstep_transpose_64x64

    POPQ R11
    POPQ R10
    POPQ R9
    POPQ R8

    // Continue to Step 2...
    JMP sixstep_fwd_done
```

**Step 3: Verify build**

Run: `go build ./internal/asm/amd64/`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add internal/asm/amd64/avx2_f32_size4096_sixstep.s
git commit -m "feat: implement AVX2 64x64 matrix transpose for six-step FFT"
```

---

## Task 5: Create Test File and Basic Tests

**Files:**

- Create: `internal/asm/amd64/avx2_f32_size4096_sixstep_test.go`

**Step 1: Create test file with transpose test**

```go
//go:build amd64 && asm && !purego

package amd64

import (
    "math"
    "math/rand"
    "testing"

    mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

func TestForwardAVX2Size4096SixStepComplex64_Basic(t *testing.T) {
    const n = 4096

    src := make([]complex64, n)
    dst := make([]complex64, n)
    twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
    scratch := make([]complex64, n)
    bitrev := mathpkg.ComputeBitReversalIndicesRadix4(n)

    // Fill with test data
    rng := rand.New(rand.NewSource(42))
    for i := range src {
        src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
    }

    // Run the six-step implementation
    ok := ForwardAVX2Size4096SixStepComplex64Asm(dst, src, twiddle, scratch, bitrev)
    if !ok {
        t.Fatal("ForwardAVX2Size4096SixStepComplex64Asm returned false")
    }

    // Compare with reference radix-4 implementation
    dstRef := make([]complex64, n)
    if !ForwardAVX2Size4096Radix4Complex64Asm(dstRef, src, twiddle, scratch, bitrev) {
        t.Fatal("ForwardAVX2Size4096Radix4Complex64Asm returned false")
    }

    // Compare results
    maxErr := float32(0)
    for i := range n {
        re := real(dst[i]) - real(dstRef[i])
        im := imag(dst[i]) - imag(dstRef[i])
        err := float32(math.Sqrt(float64(re*re + im*im)))
        if err > maxErr {
            maxErr = err
        }
    }

    t.Logf("Max error vs radix-4: %e", maxErr)

    const tolerance = 1e-4
    if maxErr > tolerance {
        t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
    }
}

func TestRoundTripAVX2Size4096SixStep_Complex64(t *testing.T) {
    const n = 4096

    src := make([]complex64, n)
    freq := make([]complex64, n)
    result := make([]complex64, n)
    twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
    scratch := make([]complex64, n)
    bitrev := mathpkg.ComputeBitReversalIndicesRadix4(n)

    // Fill with test data
    rng := rand.New(rand.NewSource(42))
    for i := range src {
        src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
    }

    // Forward transform
    if !ForwardAVX2Size4096SixStepComplex64Asm(freq, src, twiddle, scratch, bitrev) {
        t.Fatal("forward returned false")
    }

    // Inverse transform
    if !InverseAVX2Size4096SixStepComplex64Asm(result, freq, twiddle, scratch, bitrev) {
        t.Fatal("inverse returned false")
    }

    // Verify round-trip
    maxErr := float32(0)
    for i := range n {
        re := real(result[i]) - real(src[i])
        im := imag(result[i]) - imag(src[i])
        err := float32(math.Sqrt(float64(re*re + im*im)))
        if err > maxErr {
            maxErr = err
        }
    }

    t.Logf("Max round-trip error: %e", maxErr)

    const tolerance = 1e-5
    if maxErr > tolerance {
        t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
    }
}

func TestInPlaceAVX2Size4096SixStep_Complex64(t *testing.T) {
    const n = 4096

    twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
    scratch := make([]complex64, n)
    bitrev := mathpkg.ComputeBitReversalIndicesRadix4(n)

    // Generate test data
    rng := rand.New(rand.NewSource(42))
    src := make([]complex64, n)
    for i := range src {
        src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
    }

    // Out-of-place reference
    dstOOP := make([]complex64, n)
    if !ForwardAVX2Size4096SixStepComplex64Asm(dstOOP, src, twiddle, scratch, bitrev) {
        t.Fatal("out-of-place forward returned false")
    }

    // In-place test: dst == src
    dstIP := make([]complex64, n)
    copy(dstIP, src)
    scratch2 := make([]complex64, n)
    if !ForwardAVX2Size4096SixStepComplex64Asm(dstIP, dstIP, twiddle, scratch2, bitrev) {
        t.Fatal("in-place forward returned false")
    }

    // Compare
    maxErr := float32(0)
    for i := range n {
        re := real(dstOOP[i]) - real(dstIP[i])
        im := imag(dstOOP[i]) - imag(dstIP[i])
        err := float32(math.Sqrt(float64(re*re + im*im)))
        if err > maxErr {
            maxErr = err
        }
    }

    t.Logf("In-place vs out-of-place max error: %e", maxErr)

    const tolerance = 1e-6
    if maxErr > tolerance {
        t.Errorf("In-place differs from out-of-place: max error %e exceeds %e", maxErr, tolerance)
    }
}

func BenchmarkForwardAVX2Size4096SixStep_Complex64(b *testing.B) {
    const n = 4096

    src := make([]complex64, n)
    dst := make([]complex64, n)
    twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
    scratch := make([]complex64, n)
    bitrev := mathpkg.ComputeBitReversalIndicesRadix4(n)

    rng := rand.New(rand.NewSource(42))
    for i := range src {
        src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
    }

    b.ResetTimer()
    b.SetBytes(int64(n) * 8)

    for b.Loop() {
        ForwardAVX2Size4096SixStepComplex64Asm(dst, src, twiddle, scratch, bitrev)
    }
}
```

**Step 2: Run tests (expect failures initially)**

Run: `go test -v -run TestForwardAVX2Size4096SixStep ./internal/asm/amd64/`
Expected: FAIL (six-step not fully implemented yet)

**Step 3: Commit**

```bash
git add internal/asm/amd64/avx2_f32_size4096_sixstep_test.go
git commit -m "test: add tests for AVX2 six-step 4096 FFT"
```

---

## Task 6: Implement Row FFTs (Step 2)

**Files:**

- Modify: `internal/asm/amd64/avx2_f32_size4096_sixstep.s`

The row FFTs call the existing size-64 AVX2 kernel 64 times. Since we can't easily call Go assembly functions from assembly, we'll inline a simplified row FFT loop that processes each row.

**Note:** This is the most complex part. The existing size-64 kernel is 600+ lines. Instead of inlining it, we'll implement a Go wrapper that coordinates the assembly pieces.

**Alternative Approach:** Create a hybrid implementation where:

1. Assembly handles transposes (vectorized)
2. Go calls the existing size-64 kernel for row FFTs
3. Assembly handles fused transpose+twiddle

**Step 1: Update the plan - implement hybrid approach**

The pure-assembly approach would require duplicating the entire size-64 FFT code. Instead, create helper assembly routines that Go can orchestrate.

Create new assembly helper: `sixstep_transpose_and_twiddle_64x64` that does fused transpose + twiddle multiply.

```asm
// ===========================================================================
// Fused Transpose + Twiddle Multiply
// ===========================================================================
// dst[i,j] = src[j,i] * twiddle[(i*j) % 4096]
//
// Input:  R8  = destination pointer
//         R9  = source pointer
//         R10 = twiddle pointer
// Clobbers: All YMM registers, general purpose registers

sixstep_transpose_twiddle_64x64:
    // For each destination element (i,j):
    // - Read src[j*64 + i]
    // - Multiply by twiddle[(i*j) % 4096]
    // - Write to dst[i*64 + j]
    //
    // Outer loop: i = 0..63 (destination row)
    // Inner loop: j = 0..63 (destination col)

    XORQ R12, R12            // R12 = i (row index)

sixstep_tt_row_loop:
    CMPQ R12, $64
    JGE  sixstep_tt_done

    // Calculate dst row base: dst + i*64*8 = dst + i*512
    MOVQ R12, AX
    SHLQ $9, AX              // i * 512
    LEAQ (R8)(AX*1), R14     // R14 = dst row ptr

    XORQ R13, R13            // R13 = j (col index)

sixstep_tt_col_loop:
    CMPQ R13, $64
    JGE  sixstep_tt_row_next

    // Load src[j,i] = src[j*64 + i]
    MOVQ R13, AX
    SHLQ $9, AX              // j * 512
    MOVQ R12, BX
    SHLQ $3, BX              // i * 8
    ADDQ BX, AX
    VMOVSD (R9)(AX*1), X0    // X0 = src[j,i]

    // Compute twiddle index: (i*j) % 4096
    MOVQ R12, AX
    IMULQ R13, AX            // i * j
    ANDQ $4095, AX           // mod 4096
    SHLQ $3, AX              // * 8 bytes
    VMOVSD (R10)(AX*1), X1   // X1 = twiddle[(i*j) % 4096]

    // Complex multiply: X0 * X1
    // X0 = [re0, im0], X1 = [re1, im1]
    // Result = [(re0*re1 - im0*im1), (re0*im1 + im0*re1)]
    VMOVSLDUP X1, X2         // [re1, re1]
    VMOVSHDUP X1, X3         // [im1, im1]
    VSHUFPS $0xB1, X0, X0, X4 // [im0, re0]
    VMULPS X3, X4, X4        // [im0*im1, re0*im1]
    VFMADDSUB231PS X2, X0, X4 // [re0*re1 - im0*im1, re0*im1 + im0*re1]

    // Store to dst[i,j]
    MOVQ R13, BX
    SHLQ $3, BX              // j * 8
    VMOVSD X4, (R14)(BX*1)

    INCQ R13
    JMP  sixstep_tt_col_loop

sixstep_tt_row_next:
    INCQ R12
    JMP  sixstep_tt_row_loop

sixstep_tt_done:
    RET
```

**Step 2: Create similar routine for inverse (conjugate twiddle)**

```asm
// Conjugate version for inverse transform
sixstep_transpose_twiddle_conj_64x64:
    XORQ R12, R12

sixstep_ttc_row_loop:
    CMPQ R12, $64
    JGE  sixstep_ttc_done

    MOVQ R12, AX
    SHLQ $9, AX
    LEAQ (R8)(AX*1), R14

    XORQ R13, R13

sixstep_ttc_col_loop:
    CMPQ R13, $64
    JGE  sixstep_ttc_row_next

    // Load src[j,i]
    MOVQ R13, AX
    SHLQ $9, AX
    MOVQ R12, BX
    SHLQ $3, BX
    ADDQ BX, AX
    VMOVSD (R9)(AX*1), X0

    // Load twiddle and conjugate
    MOVQ R12, AX
    IMULQ R13, AX
    ANDQ $4095, AX
    SHLQ $3, AX
    VMOVSD (R10)(AX*1), X1

    // Conjugate twiddle: negate imaginary part
    // X1 = [re, im] → [re, -im]
    VMOVSLDUP X1, X2         // [re, re]
    VMOVSHDUP X1, X3         // [im, im]
    VXORPS X5, X5, X5
    VSUBPS X3, X5, X3        // [-im, -im]
    VBLENDPS $0x02, X3, X2, X1 // [re, -im]

    // Complex multiply with conjugate: same formula
    VMOVSLDUP X1, X2
    VMOVSHDUP X1, X3
    VSHUFPS $0xB1, X0, X0, X4
    VMULPS X3, X4, X4
    VFMADDSUB231PS X2, X0, X4

    MOVQ R13, BX
    SHLQ $3, BX
    VMOVSD X4, (R14)(BX*1)

    INCQ R13
    JMP  sixstep_ttc_col_loop

sixstep_ttc_row_next:
    INCQ R12
    JMP  sixstep_ttc_row_loop

sixstep_ttc_done:
    RET
```

**Step 3: Commit**

```bash
git add internal/asm/amd64/avx2_f32_size4096_sixstep.s
git commit -m "feat: add fused transpose+twiddle assembly routines"
```

---

## Task 7: Create Go Wrapper for Orchestration

Since we can't easily inline the full size-64 kernel in assembly, we'll create a Go wrapper that:

1. Calls assembly for transpose
2. Calls existing size-64 kernel for row FFTs
3. Calls assembly for fused transpose+twiddle

**Files:**

- Create: `internal/kernels/dit_size4096_sixstep_asm.go`

**Step 1: Create the Go orchestration wrapper**

```go
//go:build amd64 && asm && !purego

package kernels

import (
    "github.com/cwbudde/algo-fft/internal/asm/amd64"
)

// forwardDIT4096SixStepFullAVX2Complex64 implements the six-step algorithm
// using pure AVX2 assembly for all operations except the row FFTs.
//
// This is a hybrid approach:
// - Assembly: 64×64 transpose (AVX2 8×8 block processing)
// - Go: Row FFTs via existing ForwardAVX2Size64Radix4Complex64Asm
// - Assembly: Fused transpose + twiddle multiply
func forwardDIT4096SixStepFullAVX2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
    const (
        n = 4096
        m = 64
    )

    if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
        return false
    }

    // Use the full assembly implementation
    return amd64.ForwardAVX2Size4096SixStepComplex64Asm(dst, src, twiddle, scratch, bitrev)
}

// inverseDIT4096SixStepFullAVX2Complex64 implements the inverse six-step algorithm.
func inverseDIT4096SixStepFullAVX2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
    const n = 4096

    if len(dst) < n || len(twiddle) < n || len(scratch) < n || len(src) < n {
        return false
    }

    return amd64.InverseAVX2Size4096SixStepComplex64Asm(dst, src, twiddle, scratch, bitrev)
}
```

**Step 2: Commit**

```bash
git add internal/kernels/dit_size4096_sixstep_asm.go
git commit -m "feat: add Go wrapper for six-step full AVX2 implementation"
```

---

## Task 8: Complete Forward Transform Assembly

**Files:**

- Modify: `internal/asm/amd64/avx2_f32_size4096_sixstep.s`

This is the critical task: complete the full forward transform by integrating:

1. Transpose (Step 1)
2. Row FFTs (Step 2) - inline the radix-4 size-64 kernel
3. Transpose + Twiddle (Steps 3-4 fused)
4. Row FFTs (Step 5)
5. Final Transpose (Step 6)

Due to complexity, we'll implement this incrementally, testing after each step.

**Step 1: Implement skeleton with placeholder row FFTs**

The full implementation is approximately 2000+ lines. Key structure:

```asm
TEXT ·ForwardAVX2Size4096SixStepComplex64Asm(SB), NOSPLIT, $64
    // Stack frame for local variables:
    // 0(SP)  - saved R12
    // 8(SP)  - saved R13
    // 16(SP) - saved R14
    // 24(SP) - saved R15
    // 32(SP) - row counter
    // 40(SP) - work pointer backup
    // 48(SP) - dst pointer backup
    // 56(SP) - unused

    // [Parameter loading and validation - same as before]

    // Step 1: Transpose src → work
    // [Call transpose routine]

    // Step 2: 64 row FFTs on work buffer
    MOVQ $0, 32(SP)          // row counter = 0
sixstep_fwd_row_fft1_loop:
    MOVQ 32(SP), R12
    CMPQ R12, $64
    JGE  sixstep_fwd_step3

    // Calculate row pointer: work + row * 64 * 8 = work + row * 512
    MOVQ R12, AX
    SHLQ $9, AX
    LEAQ (R8)(AX*1), SI      // SI = current row ptr

    // Inline size-64 radix-4 FFT on row
    // [~300 lines of inlined FFT code]

    INCQ R12
    MOVQ R12, 32(SP)
    JMP  sixstep_fwd_row_fft1_loop

sixstep_fwd_step3:
    // Steps 3-4: Fused transpose + twiddle
    // [Call transpose_twiddle routine]

    // Step 5: 64 row FFTs on dst buffer
    // [Similar to step 2]

sixstep_fwd_step6:
    // Step 6: Final transpose
    // [Call transpose routine]

    // Copy and return
    // ...
```

---

## Task 9: Inline Size-64 FFT Kernel

**Challenge:** The existing size-64 kernel is ~600 lines. Inlining it 128 times is impractical.

**Solution:** Create a reusable subroutine within the assembly file that operates on a row pointer.

**Files:**

- Modify: `internal/asm/amd64/avx2_f32_size4096_sixstep.s`

**Step 1: Add size-64 row FFT subroutine**

```asm
// ===========================================================================
// Size-64 Radix-4 FFT Subroutine (for six-step row processing)
// ===========================================================================
// Input:
//   SI = row data pointer (64 complex64 = 512 bytes)
//   R10 = twiddle factors (size-64 subset: twiddle[0], twiddle[64], twiddle[128], ...)
//   R11 = scratch buffer (64 complex64)
//
// The subroutine performs the FFT in-place on the row data.
// Uses bit-reversal indices from precomputed table.

sixstep_row_fft64_forward:
    // Save caller-save registers
    PUSHQ BX
    PUSHQ DI
    PUSHQ R14
    PUSHQ R15

    // Copy to scratch with bit-reversal
    // Precomputed indices for radix-4 size-64
    XORQ CX, CX
sixstep_row_fft64_bitrev:
    MOVQ sixstep_bitrev64+0(SB)(CX*8), DX
    MOVQ (SI)(DX*8), AX
    MOVQ AX, (R11)(CX*8)
    INCQ CX
    CMPQ CX, $64
    JL   sixstep_row_fft64_bitrev

    // Stage 1: 16 radix-4 butterflies, no twiddles
    // [~100 lines of butterfly code]

    // Stage 2: 4 groups × 4 butterflies, twiddle step 4
    // [~150 lines]

    // Stage 3: 1 group × 16 butterflies, twiddle step 1
    // [~150 lines]

    // Copy result back to row
    XORQ CX, CX
sixstep_row_fft64_copyback:
    VMOVUPS (R11)(CX*1), Y0
    VMOVUPS Y0, (SI)(CX*1)
    ADDQ $32, CX
    CMPQ CX, $512
    JL   sixstep_row_fft64_copyback

    POPQ R15
    POPQ R14
    POPQ DI
    POPQ BX
    RET

// Precomputed bit-reversal indices for radix-4 size 64
DATA sixstep_bitrev64+0(SB)/8, $0   // 0
DATA sixstep_bitrev64+8(SB)/8, $16  // 1
DATA sixstep_bitrev64+16(SB)/8, $32 // 2
// ... (64 entries total)
GLOBL sixstep_bitrev64(SB), RODATA|NOPTR, $512
```

Due to the complexity of this implementation (2000+ lines of assembly), I recommend an incremental approach:

**Step 2: Build and test transpose-only version first**

```bash
go test -v -run TestForwardAVX2Size4096SixStep ./internal/asm/amd64/
```

**Step 3: Add row FFTs incrementally**

**Step 4: Commit after each working stage**

---

## Task 10: Implement Inverse Transform

**Files:**

- Modify: `internal/asm/amd64/avx2_f32_size4096_sixstep.s`

Mirror the forward transform with:

1. Conjugate twiddle factors in fused step
2. Use inverse FFT for row transforms (swapped butterfly outputs)
3. Apply 1/4096 scaling at the end

---

## Task 11: Update Codelet Registration

**Files:**

- Modify: `internal/kernels/codelet_init_avx2.go`

**Step 1: Register the new kernel with higher priority**

```go
// Size 4096: Full AVX2 Six-step variant
// Entirely AVX2-accelerated with inline row FFTs
// Expected ~50% faster than hybrid Go/ASM version
Registry64.Register(CodeletEntry[complex64]{
    Size:       4096,
    Forward:    wrapCodelet64(forwardDIT4096SixStepFullAVX2Complex64),
    Inverse:    wrapCodelet64(inverseDIT4096SixStepFullAVX2Complex64),
    Algorithm:  KernelDIT,
    SIMDLevel:  SIMDAVX2,
    Signature:  "dit4096_sixstep_full_avx2",
    Priority:   40, // Higher than hybrid (35) and radix-4 (25)
    BitrevFunc: mathpkg.ComputeBitReversalIndicesRadix4,
})
```

**Step 2: Run full test suite**

```bash
go test -v ./internal/kernels/...
go test -v ./...
```

**Step 3: Benchmark**

```bash
go test -bench=BenchmarkForward.*4096 -benchmem ./internal/asm/amd64/
```

**Step 4: Commit**

```bash
git add internal/kernels/codelet_init_avx2.go
git commit -m "feat: register full AVX2 six-step 4096 kernel"
```

---

## Task 12: Final Verification and Cleanup

**Step 1: Run comprehensive tests**

```bash
just test
just lint
```

**Step 2: Benchmark comparison**

```bash
go test -bench=BenchmarkPlanForward_4096 -benchmem -count=5 ./...
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete AVX2 six-step 4096 FFT implementation"
```

---

## Implementation Notes

### AVX2 Register Allocation

For the transpose:

- Y0-Y7: Source data (8 rows)
- Y8-Y15: Transposed output / temporaries

For FFT butterflies:

- Y0-Y3: Input data (4 complex values)
- Y4-Y7: Intermediate results
- Y8-Y10: Twiddle factors
- Y11-Y15: Temporaries for complex multiply

### Performance Expectations

| Operation        | Current (Go) | Expected (ASM) |
| ---------------- | ------------ | -------------- |
| Transpose        | ~3µs         | ~1µs           |
| Row FFTs (×128)  | ~30µs        | ~25µs          |
| Twiddle multiply | ~2µs         | ~1µs (fused)   |
| **Total**        | ~38µs        | ~28µs          |

Target: **~25% speedup** over current hybrid implementation.

### Alternative: Keep Hybrid Approach

If full assembly proves too complex, the hybrid approach with AVX2 transpose + Go row FFTs is a reasonable compromise:

- Simpler to maintain
- Still ~2× faster than pure radix-4
- Row FFT calls benefit from Go's calling convention

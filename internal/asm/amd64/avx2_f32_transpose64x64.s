//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 64×64 Matrix Transpose for Complex64 (Six-Step FFT Support)
// ===========================================================================
//
// These routines provide optimized matrix transpose operations for the
// six-step FFT algorithm which decomposes 4096 = 64×64.
//
// Memory layout:
//   - Matrix is 64×64 complex64 values
//   - Each complex64 = 8 bytes (4 bytes real + 4 bytes imag)
//   - Row stride = 64 * 8 = 512 bytes
//   - Total size = 4096 * 8 = 32768 bytes
//
// For transpose of complex64, each complex is treated as a 64-bit unit.
// We use VUNPCKLPD/VUNPCKHPD (64-bit operations) not VUNPCKLPS/VUNPCKHPS.
//
// ===========================================================================

#include "textflag.h"

// Transpose64x64Complex64AVX2Asm transposes a 64×64 matrix of complex64 values.
//
// Algorithm: Process 4×4 blocks of complex64 values.
// Each YMM register holds 4 complex64 = 32 bytes.
// A 4×4 block = 16 complex values.
//
// For 4×4 transpose of 64-bit elements (complex64):
//   Input rows:  r0=[a,b,c,d], r1=[e,f,g,h], r2=[i,j,k,l], r3=[m,n,o,p]
//   Output rows: c0=[a,e,i,m], c1=[b,f,j,n], c2=[c,g,k,o], c3=[d,h,l,p]
//
// Using VUNPCKLPD/VUNPCKHPD for 64-bit interleave, VPERM2F128 for lane crossing.
//
// func Transpose64x64Complex64AVX2Asm(dst, src []complex64) bool
TEXT ·Transpose64x64Complex64AVX2Asm(SB), NOSPLIT, $0-49
    // Load parameters
    MOVQ dst+0(FP), R8       // R8 = dst pointer
    MOVQ dst+8(FP), R9       // R9 = dst length
    MOVQ src+24(FP), R10     // R10 = src pointer
    MOVQ src+32(FP), R11     // R11 = src length

    // Validate lengths >= 4096
    CMPQ R9, $4096
    JL   transpose_return_false
    CMPQ R11, $4096
    JL   transpose_return_false

    // Process 4×4 blocks of complex64: 16 block rows × 16 block cols = 256 blocks
    // Block (bi, bj) at src: src + (bi*4*64 + bj*4) * 8 = src + bi*2048 + bj*32
    // After transpose:       dst + (bj*4*64 + bi*4) * 8 = dst + bj*2048 + bi*32

    XORQ CX, CX              // CX = bi (block row index, 0..15)

transpose_bi_loop:
    CMPQ CX, $16
    JGE  transpose_done

    XORQ DX, DX              // DX = bj (block col index, 0..15)

transpose_bj_loop:
    CMPQ DX, $16
    JGE  transpose_bi_next

    // Calculate source block address: src + bi*2048 + bj*32
    MOVQ CX, AX
    SHLQ $11, AX             // bi * 2048
    MOVQ DX, BX
    SHLQ $5, BX              // bj * 32
    ADDQ BX, AX
    LEAQ (R10)(AX*1), SI     // SI = src block ptr

    // Calculate dest block address: dst + bj*2048 + bi*32
    MOVQ DX, AX
    SHLQ $11, AX             // bj * 2048
    MOVQ CX, BX
    SHLQ $5, BX              // bi * 32
    ADDQ BX, AX
    LEAQ (R8)(AX*1), DI      // DI = dst block ptr

    // =========================================================
    // Transpose 4×4 complex64 block using 64-bit operations
    // =========================================================
    // Row stride = 512 bytes (64 complex64 * 8 bytes each)
    //
    // Load 4 rows, each containing 4 complex64 values:
    //   Y0 = [c00, c01, c02, c03]  (row 0)
    //   Y1 = [c10, c11, c12, c13]  (row 1)
    //   Y2 = [c20, c21, c22, c23]  (row 2)
    //   Y3 = [c30, c31, c32, c33]  (row 3)
    //
    // After transpose:
    //   [c00, c10, c20, c30]  (col 0 becomes row 0)
    //   [c01, c11, c21, c31]  (col 1 becomes row 1)
    //   [c02, c12, c22, c32]  (col 2 becomes row 2)
    //   [c03, c13, c23, c33]  (col 3 becomes row 3)

    VMOVUPS 0(SI), Y0              // row0
    VMOVUPS 512(SI), Y1            // row1 (stride = 512)
    VMOVUPS 1024(SI), Y2           // row2
    VMOVUPS 1536(SI), Y3           // row3

    // Step 1: 64-bit interleave within 128-bit lanes
    // VUNPCKLPD: interleave low 64-bit elements
    // VUNPCKHPD: interleave high 64-bit elements
    //
    // Y0=[c00,c01,c02,c03], Y1=[c10,c11,c12,c13]
    // VUNPCKLPD Y1,Y0 → [c00,c10, c02,c12]
    // VUNPCKHPD Y1,Y0 → [c01,c11, c03,c13]
    VUNPCKLPD Y1, Y0, Y4           // Y4 = [c00, c10, c02, c12]
    VUNPCKHPD Y1, Y0, Y5           // Y5 = [c01, c11, c03, c13]
    VUNPCKLPD Y3, Y2, Y6           // Y6 = [c20, c30, c22, c32]
    VUNPCKHPD Y3, Y2, Y7           // Y7 = [c21, c31, c23, c33]

    // Step 2: Cross 128-bit lanes with VPERM2F128
    // Y4=[c00,c10,c02,c12], Y6=[c20,c30,c22,c32]
    // VPERM2F128 $0x20 → [low128(Y4), low128(Y6)] = [c00,c10,c20,c30] = col0
    // VPERM2F128 $0x31 → [high128(Y4), high128(Y6)] = [c02,c12,c22,c32] = col2
    VPERM2F128 $0x20, Y6, Y4, Y8   // Y8 = [c00, c10, c20, c30] = transposed row 0
    VPERM2F128 $0x31, Y6, Y4, Y9   // Y9 = [c02, c12, c22, c32] = transposed row 2
    VPERM2F128 $0x20, Y7, Y5, Y10  // Y10 = [c01, c11, c21, c31] = transposed row 1
    VPERM2F128 $0x31, Y7, Y5, Y11  // Y11 = [c03, c13, c23, c33] = transposed row 3

    // Store transposed rows
    VMOVUPS Y8, 0(DI)              // transposed col0 → dst row0
    VMOVUPS Y10, 512(DI)           // transposed col1 → dst row1
    VMOVUPS Y9, 1024(DI)           // transposed col2 → dst row2
    VMOVUPS Y11, 1536(DI)          // transposed col3 → dst row3

    // Next block column
    INCQ DX
    JMP  transpose_bj_loop

transpose_bi_next:
    INCQ CX
    JMP  transpose_bi_loop

transpose_done:
    VZEROUPPER
    MOVB $1, ret+48(FP)
    RET

transpose_return_false:
    VZEROUPPER
    MOVB $0, ret+48(FP)
    RET


// ===========================================================================
// Fused Transpose + Twiddle Multiply (Forward)
// ===========================================================================
//
// TransposeTwiddle64x64Complex64AVX2Asm performs:
//   dst[i,j] = src[j,i] * twiddle[(i*j) % 4096]
//
// This fuses the transpose (step 3) and twiddle multiply (step 4) of the
// six-step algorithm into a single pass, reducing memory traffic.
//
// Parameters:
//   dst: destination buffer (64×64 complex64)
//   src: source buffer (64×64 complex64)
//   twiddle: twiddle factors (4096 complex64)
//
// func TransposeTwiddle64x64Complex64AVX2Asm(dst, src, twiddle []complex64) bool
TEXT ·TransposeTwiddle64x64Complex64AVX2Asm(SB), NOSPLIT, $0-73
    // Load parameters
    MOVQ dst+0(FP), R8       // R8 = dst pointer
    MOVQ dst+8(FP), R9       // dst length
    MOVQ src+24(FP), R10     // R10 = src pointer
    MOVQ src+32(FP), R11     // src length
    MOVQ twiddle+48(FP), R12 // R12 = twiddle pointer
    MOVQ twiddle+56(FP), R13 // twiddle length

    // Validate lengths
    CMPQ R9, $4096           // dst length >= 4096?
    JL   tt_return_false     // jump if not
    CMPQ R11, $4096          // src length >= 4096?
    JL   tt_return_false     // jump if not
    CMPQ R13, $4096          // twiddle length >= 4096?
    JL   tt_return_false     // jump if not

    // Process element by element with twiddle multiplication
    // dst[i*64+j] = src[j*64+i] * twiddle[(i*j) % 4096]
    //
    // Outer loop: i = 0..63 (destination row)
    XORQ CX, CX              // CX = i (clear for loop counter)

tt_row_loop:
    CMPQ CX, $64             // i < 64?
    JGE  tt_done             // jump if done

    // R14 = dst row base = dst + i*64*8 = dst + i*512
    MOVQ CX, AX              // AX = i
    SHLQ $9, AX              // AX = i * 512 (row stride)
    LEAQ (R8)(AX*1), R14     // R14 = dst + i*512 (row base address)

    // Inner loop: j = 0..63 (destination column), process 4 at a time
    XORQ DX, DX              // DX = j (clear for loop counter)

tt_col_loop:
    CMPQ DX, $64             // j < 64?
    JGE  tt_row_next         // jump to next row if done

    // Load 4 consecutive src elements: src[j*64+i], src[(j+1)*64+i], ...
    // These are NOT contiguous in memory (stride = 512 bytes)

    // src[j,i] address = src + (j*64 + i)*8 = src + j*512 + i*8
    MOVQ DX, AX              // AX = j
    SHLQ $9, AX              // AX = j * 512 (row stride for src)
    MOVQ CX, BX              // BX = i
    SHLQ $3, BX              // BX = i * 8 (element offset within row)
    ADDQ BX, AX              // AX = j*512 + i*8 (address offset)

    // Load 4 source elements (gather operation, non-contiguous)
    // Each complex64 is 8 bytes = 64 bits, use VMOVSD (scalar double)
    VMOVSD (R10)(AX*1), X0   // X0 = src[j,i] (first element)
    ADDQ $512, AX            // AX += 512 (advance to next row)
    VMOVSD (R10)(AX*1), X1   // X1 = src[j+1,i] (second element)
    ADDQ $512, AX            // AX += 512 (advance to next row)
    VMOVSD (R10)(AX*1), X2   // X2 = src[j+2,i] (third element)
    ADDQ $512, AX            // AX += 512 (advance to next row)
    VMOVSD (R10)(AX*1), X3   // X3 = src[j+3,i] (fourth element)

    // Combine into YMM: build [src[j], src[j+1], src[j+2], src[j+3]]
    // X0 has src[j] in low 64 bits
    // X1 has src[j+1] in low 64 bits
    // First combine X0,X1 into lower 128 bits, X2,X3 into upper 128 bits
    VUNPCKLPD X1, X0, X4     // X4 = [src[j], src[j+1]] (interleave low 64-bit elements)
    VUNPCKLPD X3, X2, X5     // X5 = [src[j+2], src[j+3]] (interleave low 64-bit elements)
    VINSERTF128 $1, X5, Y4, Y4 // Y4 = [src[j], src[j+1], src[j+2], src[j+3]] (insert upper 128 bits)

    // Load twiddle factors: twiddle[(i*j) % 4096], twiddle[(i*(j+1)) % 4096], ...
    MOVQ CX, AX              // AX = i (outer loop counter)
    MOVQ DX, BX              // BX = j (inner loop counter)
    IMULQ BX, AX             // AX = i * j (multiply)
    ANDQ $4095, AX           // AX = (i * j) % 4096 (modulo via bitwise AND)
    SHLQ $3, AX              // AX = ((i*j) % 4096) * 8 (offset in bytes)
    VMOVSD (R12)(AX*1), X8   // X8 = twiddle[i*j] (first twiddle factor)

    MOVQ CX, AX              // AX = i (reload outer counter)
    LEAQ 1(DX), BX           // BX = j+1 (next column)
    IMULQ BX, AX             // AX = i * (j+1)
    ANDQ $4095, AX           // AX = (i*(j+1)) % 4096
    SHLQ $3, AX              // AX = offset in bytes
    VMOVSD (R12)(AX*1), X9   // X9 = twiddle[i*(j+1)] (second twiddle factor)

    MOVQ CX, AX              // AX = i (reload outer counter)
    LEAQ 2(DX), BX           // BX = j+2 (next column)
    IMULQ BX, AX             // AX = i * (j+2)
    ANDQ $4095, AX           // AX = (i*(j+2)) % 4096
    SHLQ $3, AX              // AX = offset in bytes
    VMOVSD (R12)(AX*1), X10  // X10 = twiddle[i*(j+2)] (third twiddle factor)

    MOVQ CX, AX              // AX = i (reload outer counter)
    LEAQ 3(DX), BX           // BX = j+3 (next column)
    IMULQ BX, AX             // AX = i * (j+3)
    ANDQ $4095, AX           // AX = (i*(j+3)) % 4096
    SHLQ $3, AX              // AX = offset in bytes
    VMOVSD (R12)(AX*1), X11  // X11 = twiddle[i*(j+3)] (fourth twiddle factor)

    // Combine twiddles into YMM
    VUNPCKLPD X9, X8, X12    // X12 = [tw0, tw1] (interleave twiddles 0 and 1)
    VUNPCKLPD X11, X10, X13  // X13 = [tw2, tw3] (interleave twiddles 2 and 3)
    VINSERTF128 $1, X13, Y12, Y5 // Y5 = [tw0, tw1, tw2, tw3] (combine into 256-bit)

    // Complex multiply: Y4 * Y5
    // Each complex64 in YMM: [re, im, re, im, re, im, re, im]
    // result.re = a.re*b.re - a.im*b.im
    // result.im = a.re*b.im + a.im*b.re
    //
    // VMOVSLDUP: duplicate low floats (re parts): [re,re,re,re,...]
    // VMOVSHDUP: duplicate high floats (im parts): [im,im,im,im,...]
    // VPERMILPS $0xB1: swap adjacent pairs (re,im)->(im,re)
    VMOVSLDUP Y5, Y6         // Y6 = [tw.re, tw.re, ...] (duplicate real parts)
    VMOVSHDUP Y5, Y7         // Y7 = [tw.im, tw.im, ...] (duplicate imaginary parts)
    VPERMILPS $0xB1, Y4, Y8  // Y8 = [src.im, src.re, ...] (swap adjacent elements)
    VMULPS Y7, Y8, Y8        // Y8 = [src.im*tw.im, src.re*tw.im, ...] (multiply swapped by imaginary)
    // VFMADDSUB231PS: result = Y4*Y6 +/- Y8
    // This gives: [src.re*tw.re - src.im*tw.im, src.im*tw.re + src.re*tw.im, ...]
    VFMADDSUB231PS Y6, Y4, Y8 // Y8 = src*tw (fused multiply-add/subtract for complex multiply)

    // Store 4 results contiguously to dst[i,j..j+3]
    MOVQ DX, AX              // AX = j (column offset)
    SHLQ $3, AX              // AX = j * 8 (offset in bytes for 4 complex values)
    VMOVUPS Y8, (R14)(AX*1)  // store result to dst row at column offset

    ADDQ $4, DX              // DX += 4 (process next 4 columns)
    JMP  tt_col_loop         // continue inner loop

tt_row_next:
    INCQ CX              // CX += 1 (move to next row)
    JMP  tt_row_loop     // continue outer loop

tt_done:
    VZEROUPPER           // clear upper 128 bits of YMM registers (required before RET)
    MOVB $1, ret+72(FP) // set return value to true (success)
    RET                  // return to caller

tt_return_false:
    VZEROUPPER           // clear upper 128 bits of YMM registers (required before RET)
    MOVB $0, ret+72(FP) // set return value to false (validation failed)
    RET                  // return to caller


// ===========================================================================
// Fused Transpose + Conjugate Twiddle Multiply (Inverse)
// ===========================================================================
//
// TransposeTwiddleConj64x64Complex64AVX2Asm performs:
//   dst[i,j] = src[j,i] * conj(twiddle[(i*j) % 4096])
//
// Same as above but uses conjugate twiddle for inverse transform.
//
// func TransposeTwiddleConj64x64Complex64AVX2Asm(dst, src, twiddle []complex64) bool
TEXT ·TransposeTwiddleConj64x64Complex64AVX2Asm(SB), NOSPLIT, $0-73
    // Load parameters
    MOVQ dst+0(FP), R8
    MOVQ dst+8(FP), R9
    MOVQ src+24(FP), R10
    MOVQ src+32(FP), R11
    MOVQ twiddle+48(FP), R12
    MOVQ twiddle+56(FP), R13

    // Validate lengths
    CMPQ R9, $4096           // dst length >= 4096?
    JL   ttc_return_false    // jump if not
    CMPQ R11, $4096          // src length >= 4096?
    JL   ttc_return_false    // jump if not
    CMPQ R13, $4096          // twiddle length >= 4096?
    JL   ttc_return_false    // jump if not

    // Sign mask for conjugation: negate imaginary parts
    // For complex64: [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    MOVL $0x3F800000, AX     // AX = 1.0f (IEEE 754 representation)
    MOVD AX, X0              // X0 = [1.0, ?, ?, ?]
    MOVL $0xBF800000, AX     // AX = -1.0f (IEEE 754 representation)
    MOVD AX, X1              // X1 = [-1.0, ?, ?, ?]
    VPUNPCKLDQ X1, X0, X2    // X2 = [1.0, -1.0, 0, 0] (interleave double words)
    VBROADCASTSD X2, Y15     // Y15 = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0] (broadcast sign mask)

    XORQ CX, CX              // CX = i (clear for outer loop counter)

ttc_row_loop:
    CMPQ CX, $64             // i < 64?
    JGE  ttc_done            // jump if done

    MOVQ CX, AX              // AX = i
    SHLQ $9, AX              // AX = i * 512 (row stride)
    LEAQ (R8)(AX*1), R14     // R14 = dst + i*512 (dst row base address)

    XORQ DX, DX              // DX = j (clear for inner loop counter)

ttc_col_loop:
    CMPQ DX, $64             // j < 64?
    JGE  ttc_row_next        // jump to next row if done

    // Load 4 source elements (non-contiguous, stride 512)
    MOVQ DX, AX              // AX = j
    SHLQ $9, AX              // AX = j * 512 (row stride for src)
    MOVQ CX, BX              // BX = i
    SHLQ $3, BX              // BX = i * 8 (element offset within row)
    ADDQ BX, AX              // AX = j*512 + i*8 (address offset)

    VMOVSD (R10)(AX*1), X0   // X0 = src[j,i] (first element)
    ADDQ $512, AX            // AX += 512 (advance to next row)
    VMOVSD (R10)(AX*1), X1   // X1 = src[j+1,i] (second element)
    ADDQ $512, AX            // AX += 512 (advance to next row)
    VMOVSD (R10)(AX*1), X2   // X2 = src[j+2,i] (third element)
    ADDQ $512, AX            // AX += 512 (advance to next row)
    VMOVSD (R10)(AX*1), X3   // X3 = src[j+3,i] (fourth element)

    // Combine into Y4
    VUNPCKLPD X1, X0, X4     // X4 = [src[j], src[j+1]] (interleave low 64-bit elements)
    VUNPCKLPD X3, X2, X5     // X5 = [src[j+2], src[j+3]] (interleave low 64-bit elements)
    VINSERTF128 $1, X5, Y4, Y4 // Y4 = [src[j], src[j+1], src[j+2], src[j+3]] (insert upper 128 bits)

    // Load twiddle factors: same as forward function
    MOVQ CX, AX              // AX = i (outer loop counter)
    MOVQ DX, BX              // BX = j (inner loop counter)
    IMULQ BX, AX             // AX = i * j (multiply)
    ANDQ $4095, AX           // AX = (i * j) % 4096 (modulo via bitwise AND)
    SHLQ $3, AX              // AX = ((i*j) % 4096) * 8 (offset in bytes)
    VMOVSD (R12)(AX*1), X8   // X8 = twiddle[i*j] (first twiddle factor)

    MOVQ CX, AX              // AX = i (reload outer counter)
    LEAQ 1(DX), BX           // BX = j+1 (next column)
    IMULQ BX, AX             // AX = i * (j+1)
    ANDQ $4095, AX           // AX = (i*(j+1)) % 4096
    SHLQ $3, AX              // AX = offset in bytes
    VMOVSD (R12)(AX*1), X9   // X9 = twiddle[i*(j+1)] (second twiddle factor)

    MOVQ CX, AX              // AX = i (reload outer counter)
    LEAQ 2(DX), BX           // BX = j+2 (next column)
    IMULQ BX, AX             // AX = i * (j+2)
    ANDQ $4095, AX           // AX = (i*(j+2)) % 4096
    SHLQ $3, AX              // AX = offset in bytes
    VMOVSD (R12)(AX*1), X10  // X10 = twiddle[i*(j+2)] (third twiddle factor)

    MOVQ CX, AX              // AX = i (reload outer counter)
    LEAQ 3(DX), BX           // BX = j+3 (next column)
    IMULQ BX, AX             // AX = i * (j+3)
    ANDQ $4095, AX           // AX = (i*(j+3)) % 4096
    SHLQ $3, AX              // AX = offset in bytes
    VMOVSD (R12)(AX*1), X11  // X11 = twiddle[i*(j+3)] (fourth twiddle factor)

    // Combine twiddles into Y5
    VUNPCKLPD X9, X8, X12        // X12 = [tw0, tw1] (interleave twiddles 0 and 1)
    VUNPCKLPD X11, X10, X13      // X13 = [tw2, tw3] (interleave twiddles 2 and 3)
    VINSERTF128 $1, X13, Y12, Y5 // Y5 = [tw0, tw1, tw2, tw3] (combine into 256-bit)

    // Conjugate: multiply by [1, -1, 1, -1, ...] to negate imaginary parts
    VMULPS Y15, Y5, Y5           // Y5 = conj(twiddle) (flip sign of imaginary parts)

    // Complex multiply: Y4 * Y5 (same as forward, but with conjugated twiddle)
    VMOVSLDUP Y5, Y6          // Y6 = [tw.re, tw.re, ...] (duplicate real parts)
    VMOVSHDUP Y5, Y7          // Y7 = [tw.im, tw.im, ...] (duplicate imaginary parts)
    VPERMILPS $0xB1, Y4, Y8   // Y8 = [src.im, src.re, ...] (swap adjacent elements)
    VMULPS Y7, Y8, Y8         // Y8 = [src.im*tw.im, src.re*tw.im, ...] (multiply swapped by imaginary)
    VFMADDSUB231PS Y6, Y4, Y8 // Y8 = src*conj(tw) (fused multiply-add/subtract for complex multiply)

    // Store 4 results contiguously to dst[i,j..j+3]
    MOVQ DX, AX              // AX = j (column offset)
    SHLQ $3, AX              // AX = j * 8 (offset in bytes for 4 complex values)
    VMOVUPS Y8, (R14)(AX*1)  // store result to dst row at column offset

    ADDQ $4, DX              // DX += 4 (process next 4 columns)
    JMP  ttc_col_loop        // continue inner loop

ttc_row_next:
    INCQ CX              // CX += 1 (move to next row)
    JMP  ttc_row_loop    // continue outer loop

ttc_done:
    VZEROUPPER           // clear upper 128 bits of YMM registers (required before RET)
    MOVB $1, ret+72(FP)  // set return value to true (success)
    RET                  // return to caller

ttc_return_false:
    VZEROUPPER           // clear upper 128 bits of YMM registers (required before RET)
    MOVB $0, ret+72(FP)  // set return value to false (validation failed)
    RET                  // return to caller

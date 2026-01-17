//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-32 Radix-4-then-2 FFT Kernels for AMD64
// ===========================================================================
//
// This file contains a radix-4-then-2 DIT FFT optimized for size 32.
// Stages:
//   - Stage 1: radix-4 (Stride 1) - 8 butterflies
//   - Stage 2: radix-4 (Stride 4) - 8 butterflies (2 groups of 4)
//   - Stage 3: radix-2 (Stride 16) - 16 butterflies (1 group)
//
// ===========================================================================

#include "textflag.h"

// Forward transform, size 32, complex64, radix-4-then-2
TEXT ·ForwardSSE3Size32Radix4Then2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	LEAQ ·bitrevSSE2Size32Radix4Then2(SB), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 32)

	// Verify n == 32
	CMPQ R13, $32
	JNE  m24_32_sse2_fwd_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $32
	JL   m24_32_sse2_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $32
	JL   m24_32_sse2_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $32
	JL   m24_32_sse2_fwd_return_false

	MOVQ $32, AX
	CMPQ AX, $32
	JL   m24_32_sse2_fwd_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  m24_32_sse2_fwd_use_dst
	MOVQ R11, R8             // In-place: use scratch

m24_32_sse2_fwd_use_dst:
	// ==================================================================
	// Bit-reversal permutation (Mixed-radix)
	// ==================================================================
	XORQ CX, CX
m24_32_sse2_fwd_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVSD (R9)(DX*8), X0
	MOVSD X0, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $32
	JL   m24_32_sse2_fwd_bitrev_loop

	// ==================================================================
	// Stage 1: 8 radix-4 butterflies (Stride 1)
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer
	MOVQ $8, CX              // 8 butterflies
	MOVUPS ·maskNegHiPS(SB), X15

m24_32_sse2_fwd_stage1_loop:
	MOVUPS (SI), X0          // x0, x1
	MOVUPS 16(SI), X2        // x2, x3
	
	// Split into X0, X1, X2, X3 (each 1 complex)
	MOVAPS X0, X1
	SHUFPS $0xEE, X1, X1     // X1 = x1
	MOVAPS X2, X3
	SHUFPS $0xEE, X3, X3     // X3 = x3
	// X0 = x0, X2 = x2 (already low parts)

	// Radix-4 butterfly (w=1)
	MOVAPS X0, X4
	ADDPS  X2, X4            // t0 = x0 + x2
	MOVAPS X0, X5
	SUBPS  X2, X5            // t1 = x0 - x2
	MOVAPS X1, X6
	ADDPS  X3, X6            // t2 = x1 + x3
	MOVAPS X1, X7
	SUBPS  X3, X7            // t3 = x1 - x3

	// y0 = t0 + t2
	MOVAPS X4, X0
	ADDPS  X6, X0
	// y2 = t0 - t2
	MOVAPS X4, X2
	SUBPS  X6, X2
	
	// y1 = t1 + (-i)*t3
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8     // swap re/im
	XORPS  X15, X8           // negate high float -> (im, -re)
	MOVAPS X5, X1
	ADDPS  X8, X1
	
	// y3 = t1 + i*t3
	MOVAPS X7, X9
	SHUFPS $0xB1, X9, X9
	MOVUPS ·maskNegLoPS(SB), X14
	XORPS  X14, X9           // negate low float -> (-im, re)
	MOVAPS X5, X3
	ADDPS  X9, X3

	// Store y0..y3 back
	UNPCKLPD X1, X0          // X0 = [y0, y1]
	UNPCKLPD X3, X2          // X2 = [y2, y3]
	MOVUPS X0, (SI)
	MOVUPS X2, 16(SI)

	ADDQ $32, SI
	DECQ CX
	JNZ  m24_32_sse2_fwd_stage1_loop

	// ==================================================================
	// Stage 2: 8 radix-4 butterflies (Stride 4)
	// 2 groups of 4 butterflies.
	// ==================================================================
	MOVQ R8, SI
	MOVQ $2, CX              // 2 groups

m24_32_sse2_fwd_stage2_loop:
	// j=0 (w=1)
	MOVSD (SI), X0
	MOVSD 32(SI), X1
	MOVSD 64(SI), X2
	MOVSD 96(SI), X3
	// Butterfly 4 (no twiddles)
	MOVAPS X0, X4
	ADDPS  X2, X4            // t0
	MOVAPS X0, X5
	SUBPS  X2, X5            // t1
	MOVAPS X1, X6
	ADDPS  X3, X6            // t2
	MOVAPS X1, X7
	SUBPS  X3, X7            // t3
	MOVAPS X4, X0
	ADDPS  X6, X0            // y0
	MOVAPS X4, X2
	SUBPS  X6, X2            // y2
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	XORPS  X15, X8           // -i*t3
	MOVAPS X5, X1
	ADDPS  X8, X1            // y1
	MOVAPS X7, X9
	SHUFPS $0xB1, X9, X9
	MOVUPS ·maskNegLoPS(SB), X14
	XORPS  X14, X9           // i*t3
	MOVAPS X5, X3
	ADDPS  X9, X3            // y3
	MOVSD X0, (SI)
	MOVSD X1, 32(SI)
	MOVSD X2, 64(SI)
	MOVSD X3, 96(SI)

	// j=1 (twiddle[2], twiddle[4], twiddle[6])
	MOVSD 8(SI), X0
	MOVSD 40(SI), X1
	MOVSD 72(SI), X2
	MOVSD 104(SI), X3
	// Load twiddles
	MOVSD 16(R10), X10       // twiddle[2]
	MOVSD 32(R10), X11       // twiddle[4]
	MOVSD 48(R10), X12       // twiddle[6]
	// t1 = X1 * w2
	MOVAPS X10, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X10, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X1, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X1
	MULPS X14, X8
	ADDSUBPS X8, X1
	// t2 = X2 * w4
	MOVAPS X11, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X11, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X2, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X2
	MULPS X14, X8
	ADDSUBPS X8, X2
	// t3 = X3 * w6
	MOVAPS X12, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X12, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X3, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X3
	MULPS X14, X8
	ADDSUBPS X8, X3
	// Radix-4 butterfly
	MOVAPS X0, X4
	ADDPS  X2, X4
	MOVAPS X0, X5
	SUBPS  X2, X5
	MOVAPS X1, X6
	ADDPS  X3, X6
	MOVAPS X1, X7
	SUBPS  X3, X7
	MOVAPS X4, X0
	ADDPS  X6, X0
	MOVAPS X4, X2
	SUBPS  X6, X2
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	XORPS  X15, X8
	MOVAPS X5, X1
	ADDPS  X8, X1
	MOVAPS X7, X9
	SHUFPS $0xB1, X9, X9
	MOVUPS ·maskNegLoPS(SB), X14
	XORPS  X14, X9
	MOVAPS X5, X3
	ADDPS  X9, X3
	MOVSD X0, 8(SI)
	MOVSD X1, 40(SI)
	MOVSD X2, 72(SI)
	MOVSD X3, 104(SI)

	// j=2 (twiddle[4], twiddle[8], twiddle[12])
	MOVSD 16(SI), X0
	MOVSD 48(SI), X1
	MOVSD 80(SI), X2
	MOVSD 112(SI), X3
	MOVSD 32(R10), X10      // twiddle[4]
	MOVSD 64(R10), X11      // twiddle[8]
	MOVSD 96(R10), X12      // twiddle[12]
	// t1
	MOVAPS X10, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X10, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X1, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X1
	MULPS X14, X8
	ADDSUBPS X8, X1
	// t2
	MOVAPS X11, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X11, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X2, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X2
	MULPS X14, X8
	ADDSUBPS X8, X2
	// t3
	MOVAPS X12, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X12, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X3, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X3
	MULPS X14, X8
	ADDSUBPS X8, X3
	// Radix-4
	MOVAPS X0, X4
	ADDPS  X2, X4
	MOVAPS X0, X5
	SUBPS  X2, X5
	MOVAPS X1, X6
	ADDPS  X3, X6
	MOVAPS X1, X7
	SUBPS  X3, X7
	MOVAPS X4, X0
	ADDPS  X6, X0
	MOVAPS X4, X2
	SUBPS  X6, X2
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	XORPS  X15, X8
	MOVAPS X5, X1
	ADDPS  X8, X1
	MOVAPS X7, X9
	SHUFPS $0xB1, X9, X9
	MOVUPS ·maskNegLoPS(SB), X14
	XORPS  X14, X9
	MOVAPS X5, X3
	ADDPS  X9, X3
	MOVSD X0, 16(SI)
	MOVSD X1, 48(SI)
	MOVSD X2, 80(SI)
	MOVSD X3, 112(SI)

	// j=3 (twiddle[6], twiddle[12], twiddle[18])
	MOVSD 24(SI), X0
	MOVSD 56(SI), X1
	MOVSD 88(SI), X2
	MOVSD 120(SI), X3
	MOVSD 48(R10), X10      // twiddle[6]
	MOVSD 96(R10), X11      // twiddle[12]
	MOVSD 144(R10), X12     // twiddle[18]
	// t1
	MOVAPS X10, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X10, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X1, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X1
	MULPS X14, X8
	ADDSUBPS X8, X1
	// t2
	MOVAPS X11, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X11, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X2, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X2
	MULPS X14, X8
	ADDSUBPS X8, X2
	// t3
	MOVAPS X12, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X12, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X3, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X3
	MULPS X14, X8
	ADDSUBPS X8, X3
	// Radix-4
	MOVAPS X0, X4
	ADDPS  X2, X4
	MOVAPS X0, X5
	SUBPS  X2, X5
	MOVAPS X1, X6
	ADDPS  X3, X6
	MOVAPS X1, X7
	SUBPS  X3, X7
	MOVAPS X4, X0
	ADDPS  X6, X0
	MOVAPS X4, X2
	SUBPS  X6, X2
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	XORPS  X15, X8
	MOVAPS X5, X1
	ADDPS  X8, X1
	MOVAPS X7, X9
	SHUFPS $0xB1, X9, X9
	MOVUPS ·maskNegLoPS(SB), X14
	XORPS  X14, X9
	MOVAPS X5, X3
	ADDPS  X9, X3
	MOVSD X0, 24(SI)
	MOVSD X1, 56(SI)
	MOVSD X2, 88(SI)
	MOVSD X3, 120(SI)

	ADDQ $128, SI
	DECQ CX
	JNZ  m24_32_sse2_fwd_stage2_loop

	// ==================================================================
	// Stage 3: 1 radix-2 butterfly (Stride 16)
	// ==================================================================
	MOVQ R8, SI
	MOVQ $16, CX             // 16 butterflies

m24_32_sse2_fwd_stage3_loop:
	MOVSD (SI), X0           // a
	MOVSD 128(SI), X1        // b
	// twiddle[CX]
	// CX starts at 16, but we need twiddle[0..15]
	// Actually we need twiddle[16-CX]
	MOVQ $16, AX
	SUBQ CX, AX              // AX = 0, 1, 2...
	MOVSD (R10)(AX*8), X10
	// t = X1 * X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X11, X1
	MULPS X12, X13
	ADDSUBPS X13, X1
	
	MOVAPS X0, X2
	ADDPS  X1, X0
	SUBPS  X1, X2
	
	MOVSD X0, (SI)
	MOVSD X2, 128(SI)
	
	ADDQ $8, SI
	DECQ CX
	JNZ  m24_32_sse2_fwd_stage3_loop

	// Copy to dst if needed
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   m24_32_sse2_fwd_done

	MOVQ $16, CX
	MOVQ R8, SI
m24_32_sse2_fwd_copy_loop:
	MOVUPS (SI), X0
	MOVUPS X0, (R14)
	ADDQ $16, SI
	ADDQ $16, R14
	DECQ CX
	JNZ m24_32_sse2_fwd_copy_loop

m24_32_sse2_fwd_done:
	MOVB $1, ret+96(FP)
	RET

m24_32_sse2_fwd_return_false:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 32, complex64, radix-4-then-2
TEXT ·InverseSSE3Size32Radix4Then2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size32Radix4Then2(SB), R12
	MOVQ src+32(FP), R13

	// Verify n == 32
	CMPQ R13, $32
	JNE  m24_32_sse2_inv_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $32
	JL   m24_32_sse2_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $32
	JL   m24_32_sse2_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $32
	JL   m24_32_sse2_inv_return_false

	MOVQ $32, AX
	CMPQ AX, $32
	JL   m24_32_sse2_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  m24_32_sse2_inv_use_dst
	MOVQ R11, R8

m24_32_sse2_inv_use_dst:
	// Bit-reversal
	XORQ CX, CX
m24_32_sse2_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVSD (R9)(DX*8), X0
	MOVSD X0, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $32
	JL   m24_32_sse2_inv_bitrev_loop

	// Stage 1: 8 radix-4 butterflies (Stride 1)
	MOVQ R8, SI
	MOVQ $8, CX
	MOVUPS ·maskNegHiPS(SB), X15

m24_32_sse2_inv_stage1_loop:
	MOVUPS (SI), X0
	MOVUPS 16(SI), X2
	MOVAPS X0, X1
	SHUFPS $0xEE, X1, X1
	MOVAPS X2, X3
	SHUFPS $0xEE, X3, X3

	// Radix-4 butterfly (Inverse: swap i/-i)
	MOVAPS X0, X4
	ADDPS  X2, X4
	MOVAPS X0, X5
	SUBPS  X2, X5
	MOVAPS X1, X6
	ADDPS  X3, X6
	MOVAPS X1, X7
	SUBPS  X3, X7

	MOVAPS X4, X0
	ADDPS  X6, X0            // y0
	MOVAPS X4, X2
	SUBPS  X6, X2            // y2
	
	// y1 = t1 + i*t3
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	MOVUPS ·maskNegLoPS(SB), X14
	XORPS  X14, X8
	MOVAPS X5, X1
	ADDPS  X8, X1
	
	// y3 = t1 + (-i)*t3
	MOVAPS X7, X9
	SHUFPS $0xB1, X9, X9
	XORPS  X15, X9
	MOVAPS X5, X3
	ADDPS  X9, X3

	UNPCKLPD X1, X0
	UNPCKLPD X3, X2
	MOVUPS X0, (SI)
	MOVUPS X2, 16(SI)

	ADDQ $32, SI
	DECQ CX
	JNZ  m24_32_sse2_inv_stage1_loop

	// Stage 2: 8 radix-4 butterflies (Stride 4)
	MOVQ R8, SI
	MOVQ $2, CX

m24_32_sse2_inv_stage2_loop:
	// j=0
	MOVSD (SI), X0
	MOVSD 32(SI), X1
	MOVSD 64(SI), X2
	MOVSD 96(SI), X3
	MOVAPS X0, X4
	ADDPS  X2, X4
	MOVAPS X0, X5
	SUBPS  X2, X5
	MOVAPS X1, X6
	ADDPS  X3, X6
	MOVAPS X1, X7
	SUBPS  X3, X7
	MOVAPS X4, X0
	ADDPS  X6, X0
	MOVAPS X4, X2
	SUBPS  X6, X2
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	MOVUPS ·maskNegLoPS(SB), X14
	XORPS  X14, X8           // i*t3
	MOVAPS X5, X1
	ADDPS  X8, X1
	MOVAPS X7, X9
	SHUFPS $0xB1, X9, X9
	XORPS  X15, X9           // -i*t3
	MOVAPS X5, X3
	ADDPS  X9, X3
	MOVSD X0, (SI)
	MOVSD X1, 32(SI)
	MOVSD X2, 64(SI)
	MOVSD X3, 96(SI)

	// j=1
	MOVSD 8(SI), X0
	MOVSD 40(SI), X1
	MOVSD 72(SI), X2
	MOVSD 104(SI), X3
	MOVSD 16(R10), X10
	XORPS X15, X10           // conj(w2)
	MOVSD 32(R10), X11
	XORPS X15, X11           // conj(w4)
	MOVSD 48(R10), X12
	XORPS X15, X12           // conj(w6)
	// t1
	MOVAPS X10, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X10, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X1, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X1
	MULPS X14, X8
	ADDSUBPS X8, X1
	// t2
	MOVAPS X11, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X11, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X2, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X2
	MULPS X14, X8
	ADDSUBPS X8, X2
	// t3
	MOVAPS X12, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X12, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X3, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X3
	MULPS X14, X8
	ADDSUBPS X8, X3
	// Radix-4
	MOVAPS X0, X4
	ADDPS  X2, X4
	MOVAPS X0, X5
	SUBPS  X2, X5
	MOVAPS X1, X6
	ADDPS  X3, X6
	MOVAPS X1, X7
	SUBPS  X3, X7
	MOVAPS X4, X0
	ADDPS  X6, X0
	MOVAPS X4, X2
	SUBPS  X6, X2
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	MOVUPS ·maskNegLoPS(SB), X14
	XORPS  X14, X8
	MOVAPS X5, X1
	ADDPS  X8, X1
	MOVAPS X7, X9
	SHUFPS $0xB1, X9, X9
	XORPS  X15, X9
	MOVAPS X5, X3
	ADDPS  X9, X3
	MOVSD X0, 8(SI)
	MOVSD X1, 40(SI)
	MOVSD X2, 72(SI)
	MOVSD X3, 104(SI)

	// j=2
	MOVSD 16(SI), X0
	MOVSD 48(SI), X1
	MOVSD 80(SI), X2
	MOVSD 112(SI), X3
	MOVSD 32(R10), X10
	XORPS X15, X10
	MOVSD 64(R10), X11
	XORPS X15, X11
	MOVSD 96(R10), X12
	XORPS X15, X12
	// t1
	MOVAPS X10, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X10, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X1, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X1
	MULPS X14, X8
	ADDSUBPS X8, X1
	// t2
	MOVAPS X11, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X11, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X2, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X2
	MULPS X14, X8
	ADDSUBPS X8, X2
	// t3
	MOVAPS X12, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X12, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X3, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X3
	MULPS X14, X8
	ADDSUBPS X8, X3
	// Radix-4
	MOVAPS X0, X4
	ADDPS  X2, X4
	MOVAPS X0, X5
	SUBPS  X2, X5
	MOVAPS X1, X6
	ADDPS  X3, X6
	MOVAPS X1, X7
	SUBPS  X3, X7
	MOVAPS X4, X0
	ADDPS  X6, X0
	MOVAPS X4, X2
	SUBPS  X6, X2
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	MOVUPS ·maskNegLoPS(SB), X14
	XORPS  X14, X8
	MOVAPS X5, X1
	ADDPS  X8, X1
	MOVAPS X7, X9
	SHUFPS $0xB1, X9, X9
	XORPS  X15, X9
	MOVAPS X5, X3
	ADDPS  X9, X3
	MOVSD X0, 16(SI)
	MOVSD X1, 48(SI)
	MOVSD X2, 80(SI)
	MOVSD X3, 112(SI)

	// j=3
	MOVSD 24(SI), X0
	MOVSD 56(SI), X1
	MOVSD 88(SI), X2
	MOVSD 120(SI), X3
	MOVSD 48(R10), X10
	XORPS X15, X10
	MOVSD 96(R10), X11
	XORPS X15, X11
	MOVSD 144(R10), X12
	XORPS X15, X12
	// t1
	MOVAPS X10, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X10, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X1, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X1
	MULPS X14, X8
	ADDSUBPS X8, X1
	// t2
	MOVAPS X11, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X11, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X2, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X2
	MULPS X14, X8
	ADDSUBPS X8, X2
	// t3
	MOVAPS X12, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X12, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X3, X8
	SHUFPS $0xB1, X8, X8
	MULPS X13, X3
	MULPS X14, X8
	ADDSUBPS X8, X3
	// Radix-4
	MOVAPS X0, X4
	ADDPS  X2, X4
	MOVAPS X0, X5
	SUBPS  X2, X5
	MOVAPS X1, X6
	ADDPS  X3, X6
	MOVAPS X1, X7
	SUBPS  X3, X7
	MOVAPS X4, X0
	ADDPS  X6, X0
	MOVAPS X4, X2
	SUBPS  X6, X2
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	MOVUPS ·maskNegLoPS(SB), X14
	XORPS  X14, X8
	MOVAPS X5, X1
	ADDPS  X8, X1
	MOVAPS X7, X9
	SHUFPS $0xB1, X9, X9
	XORPS  X15, X9
	MOVAPS X5, X3
	ADDPS  X9, X3
	MOVSD X0, 24(SI)
	MOVSD X1, 56(SI)
	MOVSD X2, 88(SI)
	MOVSD X3, 120(SI)

	ADDQ $128, SI
	DECQ CX
	JNZ  m24_32_sse2_inv_stage2_loop

	// Stage 3: 1 radix-2 butterfly (Stride 16)
	MOVQ R8, SI
	MOVQ $16, CX

m24_32_sse2_inv_stage3_loop:
	MOVSD (SI), X0
	MOVSD 128(SI), X1
	MOVQ $16, AX
	SUBQ CX, AX
	MOVSD (R10)(AX*8), X10
	XORPS X15, X10           // conj
	// t = X1 * X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X11, X1
	MULPS X12, X13
	ADDSUBPS X13, X1
	MOVAPS X0, X2
	ADDPS  X1, X0
	SUBPS  X1, X2
	MOVSD X0, (SI)
	MOVSD X2, 128(SI)
	ADDQ $8, SI
	DECQ CX
	JNZ  m24_32_sse2_inv_stage3_loop

	// Scale by 1/32
	MOVSS ·thirtySecond32(SB), X15
	SHUFPS $0x00, X15, X15
	MOVQ $16, CX
	MOVQ R8, SI
m24_32_sse2_inv_scale_loop:
	MOVUPS (SI), X0
	MULPS X15, X0
	MOVUPS X0, (SI)
	ADDQ $16, SI
	DECQ CX
	JNZ m24_32_sse2_inv_scale_loop

	// Copy to dst if needed
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   m24_32_sse2_inv_done

	MOVQ $16, CX
	MOVQ R8, SI
m24_32_sse2_inv_copy_loop:
	MOVUPS (SI), X0
	MOVUPS X0, (R14)
	ADDQ $16, SI
	ADDQ $16, R14
	DECQ CX
	JNZ m24_32_sse2_inv_copy_loop

m24_32_sse2_inv_done:
	MOVB $1, ret+96(FP)
	RET

m24_32_sse2_inv_return_false:
	MOVB $0, ret+96(FP)
	RET

//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-64 Radix-4 FFT Kernels for AMD64
// ===========================================================================
//
// This file contains radix-4 DIT FFT kernels optimized for size 64 using SSE2.
// Size 64 = 4^3, so the radix-4 algorithm uses 3 stages:
//   Stage 1: 16 butterflies, stride=4, twiddle = 1
//   Stage 2: 4 groups × 4 butterflies, stride=16, twiddle step=4
//   Stage 3: 1 group × 16 butterflies, twiddle step=1
//
// SSE2 provides 128-bit SIMD operations (vs AVX2's 256-bit).
// ===========================================================================

#include "textflag.h"

// Forward transform, size 64, complex64, radix-4 variant
TEXT ·ForwardSSE3Size64Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	LEAQ ·bitrevSSE2Size64Radix4(SB), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 64)

	// Verify n == 64
	CMPQ R13, $64
	JNE  size64_r4_sse2_return_false

	// Validate all slice lengths >= 64
	MOVQ dst+8(FP), AX
	CMPQ AX, $64
	JL   size64_r4_sse2_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $64
	JL   size64_r4_sse2_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $64
	JL   size64_r4_sse2_return_false

	MOVQ $64, AX
	CMPQ AX, $64
	JL   size64_r4_sse2_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size64_r4_sse2_use_dst
	MOVQ R11, R8             // In-place: use scratch

size64_r4_sse2_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 bit-reversal)
	// ==================================================================
	XORQ CX, CX

size64_r4_sse2_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $64
	JL   size64_r4_sse2_bitrev_loop

size64_r4_sse2_stage1:
	// ==================================================================
	// Stage 1: 16 radix-4 butterflies, stride=4
	// ==================================================================
	XORQ CX, CX

size64_r4_sse2_stage1_loop:
	CMPQ CX, $64
	JGE  size64_r4_sse2_stage2

	LEAQ (R8)(CX*8), SI
	MOVSD (SI), X0     // a0
	MOVSD 8(SI), X1    // a1
	MOVSD 16(SI), X2   // a2
	MOVSD 24(SI), X3   // a3

	// Radix-4 butterfly (twiddle = 1)
	MOVAPS X0, X4
	ADDPS X2, X4       // t0 = a0 + a2
	MOVAPS X0, X5
	SUBPS X2, X5       // t1 = a0 - a2
	MOVAPS X1, X6
	ADDPS X3, X6       // t2 = a1 + a3
	MOVAPS X1, X7
	SUBPS X3, X7       // t3 = a1 - a3

	// (-i)*t3 = (im, -re)
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	MOVUPS ·maskNegHiPS(SB), X9
	XORPS X9, X8

	// i*t3 = (-im, re)
	MOVAPS X7, X11
	SHUFPS $0xB1, X11, X11
	MOVUPS ·maskNegLoPS(SB), X9
	XORPS X9, X11

	// Final butterfly outputs
	MOVAPS X4, X0
	ADDPS X6, X0       // a0 = t0 + t2
	MOVAPS X5, X1
	ADDPS X8, X1       // a1 = t1 + (-i)*t3
	MOVAPS X4, X2
	SUBPS X6, X2       // a2 = t0 - t2
	MOVAPS X5, X3
	ADDPS X11, X3      // a3 = t1 + i*t3

	MOVSD X0, (SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)

	ADDQ $4, CX
	JMP  size64_r4_sse2_stage1_loop

size64_r4_sse2_stage2:
	// ==================================================================
	// Stage 2: 4 groups × 4 butterflies, stride=16, twiddle step=4
	// ==================================================================
	XORQ BX, BX

size64_r4_sse2_stage2_outer:
	CMPQ BX, $4
	JGE  size64_r4_sse2_stage3

	XORQ DX, DX

size64_r4_sse2_stage2_loop:
	CMPQ DX, $4
	JGE  size64_r4_sse2_stage2_next_group

	// Calculate indices for group BX, butterfly DX
	MOVQ BX, SI
	IMULQ $16, SI      // group offset = BX * 16
	ADDQ DX, SI        // idx0 = group_offset + DX
	MOVQ SI, DI
	ADDQ $4, DI        // idx1 = idx0 + 4
	MOVQ SI, R14
	ADDQ $8, R14       // idx2 = idx0 + 8
	MOVQ SI, R15
	ADDQ $12, R15      // idx3 = idx0 + 12

	// Load twiddle factors: twiddle[DX*4], twiddle[DX*8], twiddle[DX*12]
	MOVQ DX, CX
	IMULQ $4, CX       // DX * 4
	MOVSD (R10)(CX*8), X8   // w1 = twiddle[DX*4]

	MOVQ DX, CX
	IMULQ $8, CX       // DX * 8
	MOVSD (R10)(CX*8), X9   // w2 = twiddle[DX*8]

	MOVQ DX, CX
	IMULQ $12, CX      // DX * 12
	MOVSD (R10)(CX*8), X10  // w3 = twiddle[DX*12]

	// Load data
	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1
	MOVSD (R8)(R14*8), X2
	MOVSD (R8)(R15*8), X3

	// Complex multiply a1*w1 (SSE2: no FMA, use separate mul/add)
	MOVAPS X8, X11
	SHUFPS $0x00, X11, X11  // broadcast real part
	MOVAPS X8, X12
	SHUFPS $0x55, X12, X12  // broadcast imag part
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13  // swap components
	MULPS X12, X13
	MOVAPS X1, X4
	MULPS X11, X4
	ADDSUBPS X13, X4        // complex multiply result
	MOVAPS X4, X1

	// Complex multiply a2*w2
	MOVAPS X9, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X9, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X2, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X2, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X2

	// Complex multiply a3*w3
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X3, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X3, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X3

	// Radix-4 butterfly
	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	// (-i)*t3 = (im, -re)
	MOVAPS X7, X14
	SHUFPS $0xB1, X14, X14
	MOVUPS ·maskNegHiPS(SB), X15
	XORPS X15, X14

	// i*t3 = (-im, re)
	MOVAPS X7, X12
	SHUFPS $0xB1, X12, X12
	MOVUPS ·maskNegLoPS(SB), X15
	XORPS X15, X12

	// Final outputs
	MOVAPS X4, X0
	ADDPS X6, X0
	MOVAPS X5, X1
	ADDPS X14, X1
	MOVAPS X4, X2
	SUBPS X6, X2
	MOVAPS X5, X3
	ADDPS X12, X3

	MOVSD X0, (R8)(SI*8)
	MOVSD X1, (R8)(DI*8)
	MOVSD X2, (R8)(R14*8)
	MOVSD X3, (R8)(R15*8)

	INCQ DX
	JMP  size64_r4_sse2_stage2_loop

size64_r4_sse2_stage2_next_group:
	INCQ BX
	JMP  size64_r4_sse2_stage2_outer

size64_r4_sse2_stage3:
	// ==================================================================
	// Stage 3: 1 group × 16 butterflies, twiddle step=1
	// ==================================================================
	XORQ DX, DX

size64_r4_sse2_stage3_loop:
	CMPQ DX, $16
	JGE  size64_r4_sse2_done

	// Calculate indices
	MOVQ DX, SI          // idx0 = DX
	MOVQ DX, DI
	ADDQ $16, DI         // idx1 = DX + 16
	MOVQ DX, R14
	ADDQ $32, R14        // idx2 = DX + 32
	MOVQ DX, R15
	ADDQ $48, R15        // idx3 = DX + 48

	// Twiddle factors: twiddle[DX], twiddle[2*DX], twiddle[3*DX]
	MOVQ DX, CX
	MOVSD (R10)(CX*8), X8
	MOVQ DX, CX
	IMULQ $2, CX
	MOVSD (R10)(CX*8), X9
	MOVQ DX, CX
	IMULQ $3, CX
	MOVSD (R10)(CX*8), X10

	// Load data
	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1
	MOVSD (R8)(R14*8), X2
	MOVSD (R8)(R15*8), X3

	// Complex multiply a1*w1
	MOVAPS X8, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X8, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X1, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X1

	// Complex multiply a2*w2
	MOVAPS X9, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X9, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X2, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X2, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X2

	// Complex multiply a3*w3
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X3, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X3, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X3

	// Radix-4 butterfly
	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	// (-i)*t3 = (im, -re)
	MOVAPS X7, X14
	SHUFPS $0xB1, X14, X14
	MOVUPS ·maskNegHiPS(SB), X15
	XORPS X15, X14

	// i*t3 = (-im, re)
	MOVAPS X7, X12
	SHUFPS $0xB1, X12, X12
	MOVUPS ·maskNegLoPS(SB), X15
	XORPS X15, X12

	// Final outputs
	MOVAPS X4, X0
	ADDPS X6, X0
	MOVAPS X5, X1
	ADDPS X14, X1
	MOVAPS X4, X2
	SUBPS X6, X2
	MOVAPS X5, X3
	ADDPS X12, X3

	MOVSD X0, (R8)(SI*8)
	MOVSD X1, (R8)(DI*8)
	MOVSD X2, (R8)(R14*8)
	MOVSD X3, (R8)(R15*8)

	INCQ DX
	JMP  size64_r4_sse2_stage3_loop

size64_r4_sse2_done:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size64_r4_sse2_done_direct

	XORQ CX, CX

size64_r4_sse2_copy_loop:
	MOVUPS (R8)(CX*1), X0
	MOVUPS X0, (R9)(CX*1)
	ADDQ $16, CX
	CMPQ CX, $512
	JL   size64_r4_sse2_copy_loop

size64_r4_sse2_done_direct:
	MOVB $1, ret+96(FP)
	RET

size64_r4_sse2_return_false:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 64, complex64, radix-4 variant
TEXT ·InverseSSE3Size64Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size64Radix4(SB), R12
	MOVQ src+32(FP), R13

	// Verify n == 64
	CMPQ R13, $64
	JNE  size64_r4_sse2_inv_return_false

	// Validate all slice lengths >= 64
	MOVQ dst+8(FP), AX
	CMPQ AX, $64
	JL   size64_r4_sse2_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $64
	JL   size64_r4_sse2_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $64
	JL   size64_r4_sse2_inv_return_false

	MOVQ $64, AX
	CMPQ AX, $64
	JL   size64_r4_sse2_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size64_r4_sse2_inv_use_dst
	MOVQ R11, R8

size64_r4_sse2_inv_use_dst:
	// Bit-reversal
	XORQ CX, CX

size64_r4_sse2_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $64
	JL   size64_r4_sse2_inv_bitrev_loop

	// Stage 1: twiddle = 1
	XORQ CX, CX

size64_r4_sse2_inv_stage1_loop:
	CMPQ CX, $64
	JGE  size64_r4_sse2_inv_stage2

	LEAQ (R8)(CX*8), SI
	MOVSD (SI), X0
	MOVSD 8(SI), X1
	MOVSD 16(SI), X2
	MOVSD 24(SI), X3

	// Radix-4 butterfly (inverse: swap i/-i)
	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	// i*t3 (inverse uses i instead of -i)
	MOVAPS X7, X11
	SHUFPS $0xB1, X11, X11
	MOVUPS ·maskNegLoPS(SB), X10
	XORPS X10, X11

	// (-i)*t3 (inverse uses -i instead of i)
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	MOVUPS ·maskNegHiPS(SB), X9
	XORPS X9, X8

	MOVAPS X4, X0
	ADDPS X6, X0
	MOVAPS X5, X1
	ADDPS X11, X1
	MOVAPS X4, X2
	SUBPS X6, X2
	MOVAPS X5, X3
	ADDPS X8, X3

	MOVSD X0, (SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)

	ADDQ $4, CX
	JMP  size64_r4_sse2_inv_stage1_loop

size64_r4_sse2_inv_stage2:
	// Stage 2: 4 groups × 4 butterflies
	XORQ BX, BX

size64_r4_sse2_inv_stage2_outer:
	CMPQ BX, $4
	JGE  size64_r4_sse2_inv_stage3

	XORQ DX, DX

size64_r4_sse2_inv_stage2_loop:
	CMPQ DX, $4
	JGE  size64_r4_sse2_inv_stage2_next_group

	// Calculate indices
	MOVQ BX, SI
	IMULQ $16, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ $4, DI
	MOVQ SI, R14
	ADDQ $8, R14
	MOVQ SI, R15
	ADDQ $12, R15

	// Load twiddle factors
	MOVQ DX, CX
	IMULQ $4, CX
	MOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	IMULQ $8, CX
	MOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $12, CX
	MOVSD (R10)(CX*8), X10

	// Load data
	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1
	MOVSD (R8)(R14*8), X2
	MOVSD (R8)(R15*8), X3

	// Complex multiply (conjugate twiddles for inverse)
	// Conjugate by negating imaginary part
	MOVAPS X8, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X8, X12
	SHUFPS $0x55, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X1, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X1

	MOVAPS X9, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X9, X12
	SHUFPS $0x55, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X2, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X2, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X2

	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X3, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X3, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X3

	// Radix-4 butterfly (inverse)
	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	// i*t3 = (-im, re)
	MOVAPS X7, X12
	SHUFPS $0xB1, X12, X12
	MOVUPS ·maskNegLoPS(SB), X15
	XORPS X15, X12

	// (-i)*t3 = (im, -re)
	MOVAPS X7, X14
	SHUFPS $0xB1, X14, X14
	MOVUPS ·maskNegHiPS(SB), X15
	XORPS X15, X14

	MOVAPS X4, X0
	ADDPS X6, X0
	MOVAPS X5, X1
	ADDPS X12, X1
	MOVAPS X4, X2
	SUBPS X6, X2
	MOVAPS X5, X3
	ADDPS X14, X3

	MOVSD X0, (R8)(SI*8)
	MOVSD X1, (R8)(DI*8)
	MOVSD X2, (R8)(R14*8)
	MOVSD X3, (R8)(R15*8)

	INCQ DX
	JMP  size64_r4_sse2_inv_stage2_loop

size64_r4_sse2_inv_stage2_next_group:
	INCQ BX
	JMP  size64_r4_sse2_inv_stage2_outer

size64_r4_sse2_inv_stage3:
	// Stage 3: 1 group × 16 butterflies
	XORQ DX, DX

size64_r4_sse2_inv_stage3_loop:
	CMPQ DX, $16
	JGE  size64_r4_sse2_inv_scale

	// Calculate indices
	MOVQ DX, SI
	MOVQ DX, DI
	ADDQ $16, DI
	MOVQ DX, R14
	ADDQ $32, R14
	MOVQ DX, R15
	ADDQ $48, R15

	// Twiddle factors
	MOVQ DX, CX
	MOVSD (R10)(CX*8), X8
	MOVQ DX, CX
	IMULQ $2, CX
	MOVSD (R10)(CX*8), X9
	MOVQ DX, CX
	IMULQ $3, CX
	MOVSD (R10)(CX*8), X10

	// Load data
	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1
	MOVSD (R8)(R14*8), X2
	MOVSD (R8)(R15*8), X3

	// Complex multiply (conjugate twiddles)
	MOVAPS X8, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X8, X12
	SHUFPS $0x55, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X1, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X1

	MOVAPS X9, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X9, X12
	SHUFPS $0x55, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X2, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X2, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X2

	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X3, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X3, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X3

	// Radix-4 butterfly (inverse)
	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	// i*t3 = (-im, re)
	MOVAPS X7, X12
	SHUFPS $0xB1, X12, X12
	MOVUPS ·maskNegLoPS(SB), X15
	XORPS X15, X12

	// (-i)*t3 = (im, -re)
	MOVAPS X7, X14
	SHUFPS $0xB1, X14, X14
	MOVUPS ·maskNegHiPS(SB), X15
	XORPS X15, X14

	MOVAPS X4, X0
	ADDPS X6, X0
	MOVAPS X5, X1
	ADDPS X12, X1
	MOVAPS X4, X2
	SUBPS X6, X2
	MOVAPS X5, X3
	ADDPS X14, X3

	MOVSD X0, (R8)(SI*8)
	MOVSD X1, (R8)(DI*8)
	MOVSD X2, (R8)(R14*8)
	MOVSD X3, (R8)(R15*8)

	INCQ DX
	JMP  size64_r4_sse2_inv_stage3_loop

size64_r4_sse2_inv_scale:
	// Scale by 1/64
	MOVSS $0.015625, X15  // 1/64 = 0.015625
	SHUFPS $0x00, X15, X15
	XORQ CX, CX

size64_r4_sse2_inv_scale_loop:
	MOVSD (R8)(CX*8), X0
	MULPS X15, X0
	MOVSD X0, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $64
	JL   size64_r4_sse2_inv_scale_loop

	// Copy to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size64_r4_sse2_inv_done_direct

	XORQ CX, CX

size64_r4_sse2_inv_copy_loop:
	MOVUPS (R8)(CX*1), X0
	MOVUPS X0, (R9)(CX*1)
	ADDQ $16, CX
	CMPQ CX, $512
	JL   size64_r4_sse2_inv_copy_loop

size64_r4_sse2_inv_done_direct:
	MOVB $1, ret+96(FP)
	RET

size64_r4_sse2_inv_return_false:
	MOVB $0, ret+96(FP)
	RET

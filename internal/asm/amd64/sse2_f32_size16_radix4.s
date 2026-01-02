//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-16 Radix-4 FFT Kernels for AMD64
// ===========================================================================
//
// This file contains radix-4 DIT FFT kernels optimized for size 16 using SSE2.
// Size 16 = 4^2, so the radix-4 algorithm uses 2 stages:
//   Stage 1: 4 butterflies, stride=4, twiddle = 1
//   Stage 2: 1 group, 4 butterflies, twiddle step=1
//
// SSE2 provides 128-bit SIMD operations (vs AVX2's 256-bit).
// ===========================================================================

#include "textflag.h"

// Forward transform, size 16, complex64, radix-4 variant
TEXT ·ForwardSSE2Size16Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 16)

	// Verify n == 16
	CMPQ R13, $16
	JNE  size16_r4_sse2_return_false

	// Validate all slice lengths >= 16
	MOVQ dst+8(FP), AX
	CMPQ AX, $16
	JL   size16_r4_sse2_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $16
	JL   size16_r4_sse2_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $16
	JL   size16_r4_sse2_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $16
	JL   size16_r4_sse2_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size16_r4_sse2_use_dst
	MOVQ R11, R8             // In-place: use scratch

size16_r4_sse2_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 bit-reversal)
	// ==================================================================
	XORQ CX, CX

size16_r4_sse2_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $16
	JL   size16_r4_sse2_bitrev_loop

size16_r4_sse2_stage1:
	// ==================================================================
	// Stage 1: 4 radix-4 butterflies, stride=4
	// ==================================================================
	XORQ CX, CX

size16_r4_sse2_stage1_loop:
	CMPQ CX, $16
	JGE  size16_r4_sse2_stage2

	LEAQ (R8)(CX*8), SI
	MOVSD (SI), X0     // SSE2: use MOVSD instead of VMOVSD
	MOVSD 8(SI), X1
	MOVSD 16(SI), X2
	MOVSD 24(SI), X3

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
	JMP  size16_r4_sse2_stage1_loop

size16_r4_sse2_stage2:
	// ==================================================================
	// Stage 2: 1 group, 4 butterflies
	// Twiddle step = 1
	// ==================================================================
	XORQ DX, DX

size16_r4_sse2_stage2_loop:
	CMPQ DX, $4
	JGE  size16_r4_sse2_done

	MOVQ DX, BX
	LEAQ 4(DX), SI
	LEAQ 8(DX), DI
	LEAQ 12(DX), R14

	// Twiddle factors: twiddle[j], twiddle[2*j], twiddle[3*j]
	MOVQ DX, R15
	MOVSD (R10)(R15*8), X8
	SHLQ $1, R15
	MOVSD (R10)(R15*8), X9
	ADDQ DX, R15
	MOVSD (R10)(R15*8), X10

	MOVSD (R8)(BX*8), X0
	MOVSD (R8)(SI*8), X1
	MOVSD (R8)(DI*8), X2
	MOVSD (R8)(R14*8), X3

	// Complex multiply a1*w1 (SSE2: no FMA, use separate mul/add)
	MOVAPS X8, X11
	SHUFPS $0x00, X11, X11  // broadcast real part
	MOVAPS X8, X12
	SHUFPS $0x55, X12, X12  // broadcast imag part
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13  // swap components
	MULPS X12, X13
	MOVAPS X1, X14
	MULPS X11, X14
	ADDSUBPS X13, X14       // addsubps: a[i] = (i%2==0) ? a[i]+b[i] : a[i]-b[i]
	MOVAPS X14, X1

	// Complex multiply a2*w2
	MOVAPS X9, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X9, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X2, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X2, X14
	MULPS X11, X14
	ADDSUBPS X13, X14
	MOVAPS X14, X2

	// Complex multiply a3*w3
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X3, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X3, X14
	MULPS X11, X14
	ADDSUBPS X13, X14
	MOVAPS X14, X3

	// Radix-4 butterfly
	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	// (-i)*t3
	MOVAPS X7, X14
	SHUFPS $0xB1, X14, X14
	XORPS X15, X15
	MOVAPS X14, X11
	SUBPS X11, X15
	SHUFPS $0x44, X11, X15
	SHUFPS $0x0E, X11, X15
	MOVAPS X15, X14

	// i*t3
	MOVAPS X7, X12
	SHUFPS $0xB1, X12, X12
	XORPS X15, X15
	MOVAPS X12, X11
	SUBPS X11, X15
	SHUFPS $0xEE, X15, X12

	// Final outputs
	MOVAPS X4, X0
	ADDPS X6, X0
	MOVAPS X5, X1
	ADDPS X14, X1
	MOVAPS X4, X2
	SUBPS X6, X2
	MOVAPS X5, X3
	ADDPS X12, X3

	MOVSD X0, (R8)(BX*8)
	MOVSD X1, (R8)(SI*8)
	MOVSD X2, (R8)(DI*8)
	MOVSD X3, (R8)(R14*8)

	INCQ DX
	JMP  size16_r4_sse2_stage2_loop

size16_r4_sse2_done:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size16_r4_sse2_done_direct

	XORQ CX, CX

size16_r4_sse2_copy_loop:
	MOVUPS (R8)(CX*1), X0
	MOVUPS X0, (R9)(CX*1)
	ADDQ $16, CX
	CMPQ CX, $128
	JL   size16_r4_sse2_copy_loop

size16_r4_sse2_done_direct:
	MOVB $1, ret+120(FP)
	RET

size16_r4_sse2_return_false:
	MOVB $0, ret+120(FP)
	RET

// Inverse transform, size 16, complex64, radix-4 variant
TEXT ·InverseSSE2Size16Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	// Verify n == 16
	CMPQ R13, $16
	JNE  size16_r4_sse2_inv_return_false

	// Validate all slice lengths >= 16
	MOVQ dst+8(FP), AX
	CMPQ AX, $16
	JL   size16_r4_sse2_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $16
	JL   size16_r4_sse2_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $16
	JL   size16_r4_sse2_inv_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $16
	JL   size16_r4_sse2_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size16_r4_sse2_inv_use_dst
	MOVQ R11, R8

size16_r4_sse2_inv_use_dst:
	// Bit-reversal
	XORQ CX, CX

size16_r4_sse2_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $16
	JL   size16_r4_sse2_inv_bitrev_loop

	// Stage 1: twiddle = 1
	XORQ CX, CX

size16_r4_sse2_inv_stage1_loop:
	CMPQ CX, $16
	JGE  size16_r4_sse2_inv_stage2

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
	XORPS X10, X10
	MOVAPS X11, X12
	SUBPS X12, X10
	SHUFPS $0xEE, X10, X11

	// (-i)*t3 (inverse uses -i instead of i)
	SHUFPS $0xB1, X7, X7
	XORPS X9, X9
	MOVAPS X7, X8
	SUBPS X8, X9
	SHUFPS $0x44, X8, X9
	SHUFPS $0x0E, X8, X9
	MOVAPS X9, X8

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
	JMP  size16_r4_sse2_inv_stage1_loop

size16_r4_sse2_inv_stage2:
	// Stage 2
	XORQ DX, DX

size16_r4_sse2_inv_stage2_loop:
	CMPQ DX, $4
	JGE  size16_r4_sse2_inv_scale

	MOVQ DX, BX
	LEAQ 4(DX), SI
	LEAQ 8(DX), DI
	LEAQ 12(DX), R14

	// Twiddle factors
	MOVQ DX, R15
	MOVSD (R10)(R15*8), X8
	SHLQ $1, R15
	MOVSD (R10)(R15*8), X9
	ADDQ DX, R15
	MOVSD (R10)(R15*8), X10

	MOVSD (R8)(BX*8), X0
	MOVSD (R8)(SI*8), X1
	MOVSD (R8)(DI*8), X2
	MOVSD (R8)(R14*8), X3

	// Complex multiply (conjugate twiddles for inverse)
	// Conjugate by negating imaginary part
	MOVAPS X8, X11
	SHUFPS $0xA0, X11, X11
	MOVAPS X8, X12
	SHUFPS $0xF5, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X1, X14
	MULPS X11, X14
	ADDSUBPS X13, X14
	MOVAPS X14, X1

	MOVAPS X9, X11
	SHUFPS $0xA0, X11, X11
	MOVAPS X9, X12
	SHUFPS $0xF5, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X2, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X2, X14
	MULPS X11, X14
	ADDSUBPS X13, X14
	MOVAPS X14, X2

	MOVAPS X10, X11
	SHUFPS $0xA0, X11, X11
	MOVAPS X10, X12
	SHUFPS $0xF5, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X3, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X3, X14
	MULPS X11, X14
	ADDSUBPS X13, X14
	MOVAPS X14, X3

	// Radix-4 butterfly (inverse)
	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	// i*t3
	MOVAPS X7, X12
	SHUFPS $0xB1, X12, X12
	XORPS X15, X15
	MOVAPS X12, X11
	SUBPS X11, X15
	SHUFPS $0xEE, X15, X12

	// (-i)*t3
	MOVAPS X7, X14
	SHUFPS $0xB1, X14, X14
	XORPS X15, X15
	MOVAPS X14, X11
	SUBPS X11, X15
	SHUFPS $0x44, X11, X15
	SHUFPS $0x0E, X11, X15
	MOVAPS X15, X14

	MOVAPS X4, X0
	ADDPS X6, X0
	MOVAPS X5, X1
	ADDPS X12, X1
	MOVAPS X4, X2
	SUBPS X6, X2
	MOVAPS X5, X3
	ADDPS X14, X3

	MOVSD X0, (R8)(BX*8)
	MOVSD X1, (R8)(SI*8)
	MOVSD X2, (R8)(DI*8)
	MOVSD X3, (R8)(R14*8)

	INCQ DX
	JMP  size16_r4_sse2_inv_stage2_loop

size16_r4_sse2_inv_scale:
	// Scale by 1/16
	MOVSS $0.0625, X15  // 1/16
	SHUFPS $0x00, X15, X15
	XORQ CX, CX

size16_r4_sse2_inv_scale_loop:
	MOVSD (R8)(CX*8), X0
	MULPS X15, X0
	MOVSD X0, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $16
	JL   size16_r4_sse2_inv_scale_loop

	// Copy to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size16_r4_sse2_inv_done_direct

	XORQ CX, CX

size16_r4_sse2_inv_copy_loop:
	MOVUPS (R8)(CX*1), X0
	MOVUPS X0, (R9)(CX*1)
	ADDQ $16, CX
	CMPQ CX, $128
	JL   size16_r4_sse2_inv_copy_loop

size16_r4_sse2_inv_done_direct:
	MOVB $1, ret+120(FP)
	RET

size16_r4_sse2_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

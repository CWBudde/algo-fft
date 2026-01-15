//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-32 Mixed-Radix-2/4 FFT Kernels for AMD64
// ===========================================================================
//
// This file contains a mixed-radix-2/4 DIT FFT optimized for size 32.
// Stages:
//   - Stage 1: radix-4 (Stride 1) - 8 butterflies
//   - Stage 2: radix-4 (Stride 4) - 8 butterflies (2 groups of 4)
//   - Stage 3: radix-2 (Stride 16) - 16 butterflies (1 group)
//
// ===========================================================================

#include "textflag.h"

// Forward transform, size 32, complex64, mixed-radix
TEXT ·ForwardAVX2Size32Mixed24Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8 // R8  = dst pointer
	MOVQ src+24(FP), R9 // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	LEAQ ·bitrevSSE2Size32Mixed24(SB), R12 // R12 = bitrev pointer
	MOVQ src+32(FP), R13 // R13 = n (should be 32)

	// Verify n == 32
	CMPQ R13, $32
	JNE  m24_32_avx2_fwd_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $32
	JL   m24_32_avx2_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $32
	JL   m24_32_avx2_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $32
	JL   m24_32_avx2_fwd_return_false

	MOVQ $32, AX
	CMPQ AX, $32
	JL   m24_32_avx2_fwd_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  m24_32_avx2_fwd_use_dst
	MOVQ R11, R8 // In-place: use scratch

m24_32_avx2_fwd_use_dst:
	// ==================================================================
	// Bit-reversal permutation (Mixed-radix)
	// ==================================================================
	XORQ CX, CX
m24_32_avx2_fwd_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVSD (R9)(DX*8), X0
	MOVSD X0, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $32
	JL   m24_32_avx2_fwd_bitrev_loop

	// ==================================================================
	// Stage 1: 8 radix-4 butterflies (Stride 1)
	// ==================================================================
	MOVQ R8, SI // SI = work buffer
	MOVQ $8, CX // 8 butterflies
	VMOVUPS ·maskNegHiPS(SB), X15

m24_32_avx2_fwd_stage1_loop:
	VMOVUPS (SI), X0 // x0, x1
	VMOVUPS 16(SI), X2 // x2, x3
	
	// Split into X0, X1, X2, X3 (each 1 complex)
	VMOVAPS X0, X1
	VSHUFPS $0xEE, X1, X1, X1 // X1 = x1
	VMOVAPS X2, X3
	VSHUFPS $0xEE, X3, X3, X3 // X3 = x3
	// X0 = x0, X2 = x2 (already low parts)

	// Radix-4 butterfly (w=1)
	VMOVAPS X0, X4
	VADDPS  X2, X4, X4 // t0 = x0 + x2
	VMOVAPS X0, X5
	VSUBPS  X2, X5, X5 // t1 = x0 - x2
	VMOVAPS X1, X6
	VADDPS  X3, X6, X6 // t2 = x1 + x3
	VMOVAPS X1, X7
	VSUBPS  X3, X7, X7 // t3 = x1 - x3

	// y0 = t0 + t2
	VMOVAPS X4, X0
	VADDPS  X6, X0, X0
	// y2 = t0 - t2
	VMOVAPS X4, X2
	VSUBPS  X6, X2, X2
	
	// y1 = t1 + (-i)*t3
	VMOVAPS X7, X8
	VSHUFPS $0xB1, X8, X8, X8 // swap re/im
	VXORPS  X15, X8, X8 // negate high float -> (im, -re)
	VMOVAPS X5, X1
	VADDPS  X8, X1, X1
	
	// y3 = t1 + i*t3
	VMOVAPS X7, X9
	VSHUFPS $0xB1, X9, X9, X9
	VMOVUPS ·maskNegLoPS(SB), X14
	VXORPS  X14, X9, X9 // negate low float -> (-im, re)
	VMOVAPS X5, X3
	VADDPS  X9, X3, X3

	// Store y0..y3 back
	VUNPCKLPD X1, X0, X0 // X0 = [y0, y1]
	VUNPCKLPD X3, X2, X2 // X2 = [y2, y3]
	VMOVUPS X0, (SI)
	VMOVUPS X2, 16(SI)

	ADDQ $32, SI
	DECQ CX
	JNZ  m24_32_avx2_fwd_stage1_loop

	// ==================================================================
	// Stage 2: 8 radix-4 butterflies (Stride 4)
	// 2 groups of 4 butterflies.
	// ==================================================================
	MOVQ R8, SI
	MOVQ $2, CX // 2 groups

m24_32_avx2_fwd_stage2_loop:
	// j=0 (w=1)
	MOVSD (SI), X0
	MOVSD 32(SI), X1
	MOVSD 64(SI), X2
	MOVSD 96(SI), X3
	// Butterfly 4 (no twiddles)
	VMOVAPS X0, X4
	VADDPS  X2, X4, X4 // t0
	VMOVAPS X0, X5
	VSUBPS  X2, X5, X5 // t1
	VMOVAPS X1, X6
	VADDPS  X3, X6, X6 // t2
	VMOVAPS X1, X7
	VSUBPS  X3, X7, X7 // t3
	VMOVAPS X4, X0
	VADDPS  X6, X0, X0 // y0
	VMOVAPS X4, X2
	VSUBPS  X6, X2, X2 // y2
	VMOVAPS X7, X8
	VSHUFPS $0xB1, X8, X8, X8
	VXORPS  X15, X8, X8 // -i*t3
	VMOVAPS X5, X1
	VADDPS  X8, X1, X1 // y1
	VMOVAPS X7, X9
	VSHUFPS $0xB1, X9, X9, X9
	VMOVUPS ·maskNegLoPS(SB), X14
	VXORPS  X14, X9, X9 // i*t3
	VMOVAPS X5, X3
	VADDPS  X9, X3, X3 // y3
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
	MOVSD 16(R10), X10 // twiddle[2]
	MOVSD 32(R10), X11 // twiddle[4]
	MOVSD 48(R10), X12 // twiddle[6]
	// t1 = X1 * w2
	VMOVAPS X10, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X10, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X1, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X1, X1
	VMULPS X14, X8, X8
	VADDSUBPS X8, X1, X1
	// t2 = X2 * w4
	VMOVAPS X11, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X11, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X2, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X2, X2
	VMULPS X14, X8, X8
	VADDSUBPS X8, X2, X2
	// t3 = X3 * w6
	VMOVAPS X12, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X12, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X3, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X3, X3
	VMULPS X14, X8, X8
	VADDSUBPS X8, X3, X3
	// Radix-4 butterfly
	VMOVAPS X0, X4
	VADDPS  X2, X4, X4
	VMOVAPS X0, X5
	VSUBPS  X2, X5, X5
	VMOVAPS X1, X6
	VADDPS  X3, X6, X6
	VMOVAPS X1, X7
	VSUBPS  X3, X7, X7
	VMOVAPS X4, X0
	VADDPS  X6, X0, X0
	VMOVAPS X4, X2
	VSUBPS  X6, X2, X2
	VMOVAPS X7, X8
	VSHUFPS $0xB1, X8, X8, X8
	VXORPS  X15, X8, X8
	VMOVAPS X5, X1
	VADDPS  X8, X1, X1
	VMOVAPS X7, X9
	VSHUFPS $0xB1, X9, X9, X9
	VMOVUPS ·maskNegLoPS(SB), X14
	VXORPS  X14, X9, X9
	VMOVAPS X5, X3
	VADDPS  X9, X3, X3
	MOVSD X0, 8(SI)
	MOVSD X1, 40(SI)
	MOVSD X2, 72(SI)
	MOVSD X3, 104(SI)

	// j=2 (twiddle[4], twiddle[8], twiddle[12])
	MOVSD 16(SI), X0
	MOVSD 48(SI), X1
	MOVSD 80(SI), X2
	MOVSD 112(SI), X3
	MOVSD 32(R10), X10 // twiddle[4]
	MOVSD 64(R10), X11 // twiddle[8]
	MOVSD 96(R10), X12 // twiddle[12]
	// t1
	VMOVAPS X10, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X10, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X1, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X1, X1
	VMULPS X14, X8, X8
	VADDSUBPS X8, X1, X1
	// t2
	VMOVAPS X11, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X11, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X2, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X2, X2
	VMULPS X14, X8, X8
	VADDSUBPS X8, X2, X2
	// t3
	VMOVAPS X12, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X12, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X3, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X3, X3
	VMULPS X14, X8, X8
	VADDSUBPS X8, X3, X3
	// Radix-4
	VMOVAPS X0, X4
	VADDPS  X2, X4, X4
	VMOVAPS X0, X5
	VSUBPS  X2, X5, X5
	VMOVAPS X1, X6
	VADDPS  X3, X6, X6
	VMOVAPS X1, X7
	VSUBPS  X3, X7, X7
	VMOVAPS X4, X0
	VADDPS  X6, X0, X0
	VMOVAPS X4, X2
	VSUBPS  X6, X2, X2
	VMOVAPS X7, X8
	VSHUFPS $0xB1, X8, X8, X8
	VXORPS  X15, X8, X8
	VMOVAPS X5, X1
	VADDPS  X8, X1, X1
	VMOVAPS X7, X9
	VSHUFPS $0xB1, X9, X9, X9
	VMOVUPS ·maskNegLoPS(SB), X14
	VXORPS  X14, X9, X9
	VMOVAPS X5, X3
	VADDPS  X9, X3, X3
	MOVSD X0, 16(SI)
	MOVSD X1, 48(SI)
	MOVSD X2, 80(SI)
	MOVSD X3, 112(SI)

	// j=3 (twiddle[6], twiddle[12], twiddle[18])
	MOVSD 24(SI), X0
	MOVSD 56(SI), X1
	MOVSD 88(SI), X2
	MOVSD 120(SI), X3
	MOVSD 48(R10), X10 // twiddle[6]
	MOVSD 96(R10), X11 // twiddle[12]
	MOVSD 144(R10), X12 // twiddle[18]
	// t1
	VMOVAPS X10, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X10, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X1, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X1, X1
	VMULPS X14, X8, X8
	VADDSUBPS X8, X1, X1
	// t2
	VMOVAPS X11, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X11, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X2, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X2, X2
	VMULPS X14, X8, X8
	VADDSUBPS X8, X2, X2
	// t3
	VMOVAPS X12, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X12, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X3, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X3, X3
	VMULPS X14, X8, X8
	VADDSUBPS X8, X3, X3
	// Radix-4
	VMOVAPS X0, X4
	VADDPS  X2, X4, X4
	VMOVAPS X0, X5
	VSUBPS  X2, X5, X5
	VMOVAPS X1, X6
	VADDPS  X3, X6, X6
	VMOVAPS X1, X7
	VSUBPS  X3, X7, X7
	VMOVAPS X4, X0
	VADDPS  X6, X0, X0
	VMOVAPS X4, X2
	VSUBPS  X6, X2, X2
	VMOVAPS X7, X8
	VSHUFPS $0xB1, X8, X8, X8
	VXORPS  X15, X8, X8
	VMOVAPS X5, X1
	VADDPS  X8, X1, X1
	VMOVAPS X7, X9
	VSHUFPS $0xB1, X9, X9, X9
	VMOVUPS ·maskNegLoPS(SB), X14
	VXORPS  X14, X9, X9
	VMOVAPS X5, X3
	VADDPS  X9, X3, X3
	MOVSD X0, 24(SI)
	MOVSD X1, 56(SI)
	MOVSD X2, 88(SI)
	MOVSD X3, 120(SI)

	ADDQ $128, SI
	DECQ CX
	JNZ  m24_32_avx2_fwd_stage2_loop

	// ==================================================================
	// Stage 3: 1 radix-2 butterfly (Stride 16)
	// ==================================================================
	MOVQ R8, SI
	MOVQ $16, CX // 16 butterflies

m24_32_avx2_fwd_stage3_loop:
	MOVSD (SI), X0 // a
	MOVSD 128(SI), X1 // b
	// twiddle[CX]
	// CX starts at 16, but we need twiddle[0..15]
	// Actually we need twiddle[16-CX]
	MOVQ $16, AX
	SUBQ CX, AX // AX = 0, 1, 2...
	MOVSD (R10)(AX*8), X10
	// t = X1 * X10
	VMOVAPS X10, X11
	VSHUFPS $0x00, X11, X11, X11
	VMOVAPS X10, X12
	VSHUFPS $0x55, X12, X12, X12
	VMOVAPS X1, X13
	VSHUFPS $0xB1, X13, X13, X13
	VMULPS X11, X1, X1
	VMULPS X12, X13, X13
	VADDSUBPS X13, X1, X1
	
	VMOVAPS X0, X2
	VADDPS  X1, X0, X0
	VSUBPS  X1, X2, X2
	
	MOVSD X0, (SI)
	MOVSD X2, 128(SI)
	
	ADDQ $8, SI
	DECQ CX
	JNZ  m24_32_avx2_fwd_stage3_loop

	// Copy to dst if needed
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   m24_32_avx2_fwd_done

	MOVQ $16, CX
	MOVQ R8, SI
m24_32_avx2_fwd_copy_loop:
	VMOVUPS (SI), X0
	VMOVUPS X0, (R14)
	ADDQ $16, SI
	ADDQ $16, R14
	DECQ CX
	JNZ m24_32_avx2_fwd_copy_loop

m24_32_avx2_fwd_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

m24_32_avx2_fwd_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 32, complex64, mixed-radix
TEXT ·InverseAVX2Size32Mixed24Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size32Mixed24(SB), R12
	MOVQ src+32(FP), R13

	// Verify n == 32
	CMPQ R13, $32
	JNE  m24_32_avx2_inv_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $32
	JL   m24_32_avx2_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $32
	JL   m24_32_avx2_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $32
	JL   m24_32_avx2_inv_return_false

	MOVQ $32, AX
	CMPQ AX, $32
	JL   m24_32_avx2_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  m24_32_avx2_inv_use_dst
	MOVQ R11, R8

m24_32_avx2_inv_use_dst:
	// Bit-reversal
	XORQ CX, CX
m24_32_avx2_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVSD (R9)(DX*8), X0
	MOVSD X0, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $32
	JL   m24_32_avx2_inv_bitrev_loop

	// Stage 1: 8 radix-4 butterflies (Stride 1)
	MOVQ R8, SI
	MOVQ $8, CX
	VMOVUPS ·maskNegHiPS(SB), X15

m24_32_avx2_inv_stage1_loop:
	VMOVUPS (SI), X0
	VMOVUPS 16(SI), X2
	VMOVAPS X0, X1
	VSHUFPS $0xEE, X1, X1, X1
	VMOVAPS X2, X3
	VSHUFPS $0xEE, X3, X3, X3

	// Radix-4 butterfly (Inverse: swap i/-i)
	VMOVAPS X0, X4
	VADDPS  X2, X4, X4
	VMOVAPS X0, X5
	VSUBPS  X2, X5, X5
	VMOVAPS X1, X6
	VADDPS  X3, X6, X6
	VMOVAPS X1, X7
	VSUBPS  X3, X7, X7

	VMOVAPS X4, X0
	VADDPS  X6, X0, X0 // y0
	VMOVAPS X4, X2
	VSUBPS  X6, X2, X2 // y2
	
	// y1 = t1 + i*t3
	VMOVAPS X7, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMOVUPS ·maskNegLoPS(SB), X14
	VXORPS  X14, X8, X8
	VMOVAPS X5, X1
	VADDPS  X8, X1, X1
	
	// y3 = t1 + (-i)*t3
	VMOVAPS X7, X9
	VSHUFPS $0xB1, X9, X9, X9
	VXORPS  X15, X9, X9
	VMOVAPS X5, X3
	VADDPS  X9, X3, X3

	VUNPCKLPD X1, X0, X0
	VUNPCKLPD X3, X2, X2
	VMOVUPS X0, (SI)
	VMOVUPS X2, 16(SI)

	ADDQ $32, SI
	DECQ CX
	JNZ  m24_32_avx2_inv_stage1_loop

	// Stage 2: 8 radix-4 butterflies (Stride 4)
	MOVQ R8, SI
	MOVQ $2, CX

m24_32_avx2_inv_stage2_loop:
	// j=0
	MOVSD (SI), X0
	MOVSD 32(SI), X1
	MOVSD 64(SI), X2
	MOVSD 96(SI), X3
	VMOVAPS X0, X4
	VADDPS  X2, X4, X4
	VMOVAPS X0, X5
	VSUBPS  X2, X5, X5
	VMOVAPS X1, X6
	VADDPS  X3, X6, X6
	VMOVAPS X1, X7
	VSUBPS  X3, X7, X7
	VMOVAPS X4, X0
	VADDPS  X6, X0, X0
	VMOVAPS X4, X2
	VSUBPS  X6, X2, X2
	VMOVAPS X7, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMOVUPS ·maskNegLoPS(SB), X14
	VXORPS  X14, X8, X8 // i*t3
	VMOVAPS X5, X1
	VADDPS  X8, X1, X1
	VMOVAPS X7, X9
	VSHUFPS $0xB1, X9, X9, X9
	VXORPS  X15, X9, X9 // -i*t3
	VMOVAPS X5, X3
	VADDPS  X9, X3, X3
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
	VXORPS X15, X10, X10 // conj(w2)
	MOVSD 32(R10), X11
	VXORPS X15, X11, X11 // conj(w4)
	MOVSD 48(R10), X12
	VXORPS X15, X12, X12 // conj(w6)
	// t1
	VMOVAPS X10, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X10, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X1, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X1, X1
	VMULPS X14, X8, X8
	VADDSUBPS X8, X1, X1
	// t2
	VMOVAPS X11, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X11, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X2, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X2, X2
	VMULPS X14, X8, X8
	VADDSUBPS X8, X2, X2
	// t3
	VMOVAPS X12, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X12, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X3, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X3, X3
	VMULPS X14, X8, X8
	VADDSUBPS X8, X3, X3
	// Radix-4
	VMOVAPS X0, X4
	VADDPS  X2, X4, X4
	VMOVAPS X0, X5
	VSUBPS  X2, X5, X5
	VMOVAPS X1, X6
	VADDPS  X3, X6, X6
	VMOVAPS X1, X7
	VSUBPS  X3, X7, X7
	VMOVAPS X4, X0
	VADDPS  X6, X0, X0
	VMOVAPS X4, X2
	VSUBPS  X6, X2, X2
	VMOVAPS X7, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMOVUPS ·maskNegLoPS(SB), X14
	VXORPS  X14, X8, X8
	VMOVAPS X5, X1
	VADDPS  X8, X1, X1
	VMOVAPS X7, X9
	VSHUFPS $0xB1, X9, X9, X9
	VXORPS  X15, X9, X9
	VMOVAPS X5, X3
	VADDPS  X9, X3, X3
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
	VXORPS X15, X10, X10
	MOVSD 64(R10), X11
	VXORPS X15, X11, X11
	MOVSD 96(R10), X12
	VXORPS X15, X12, X12
	// t1
	VMOVAPS X10, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X10, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X1, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X1, X1
	VMULPS X14, X8, X8
	VADDSUBPS X8, X1, X1
	// t2
	VMOVAPS X11, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X11, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X2, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X2, X2
	VMULPS X14, X8, X8
	VADDSUBPS X8, X2, X2
	// t3
	VMOVAPS X12, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X12, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X3, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X3, X3
	VMULPS X14, X8, X8
	VADDSUBPS X8, X3, X3
	// Radix-4
	VMOVAPS X0, X4
	VADDPS  X2, X4, X4
	VMOVAPS X0, X5
	VSUBPS  X2, X5, X5
	VMOVAPS X1, X6
	VADDPS  X3, X6, X6
	VMOVAPS X1, X7
	VSUBPS  X3, X7, X7
	VMOVAPS X4, X0
	VADDPS  X6, X0, X0
	VMOVAPS X4, X2
	VSUBPS  X6, X2, X2
	VMOVAPS X7, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMOVUPS ·maskNegLoPS(SB), X14
	VXORPS  X14, X8, X8
	VMOVAPS X5, X1
	VADDPS  X8, X1, X1
	VMOVAPS X7, X9
	VSHUFPS $0xB1, X9, X9, X9
	VXORPS  X15, X9, X9
	VMOVAPS X5, X3
	VADDPS  X9, X3, X3
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
	VXORPS X15, X10, X10
	MOVSD 96(R10), X11
	VXORPS X15, X11, X11
	MOVSD 144(R10), X12
	VXORPS X15, X12, X12
	// t1
	VMOVAPS X10, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X10, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X1, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X1, X1
	VMULPS X14, X8, X8
	VADDSUBPS X8, X1, X1
	// t2
	VMOVAPS X11, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X11, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X2, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X2, X2
	VMULPS X14, X8, X8
	VADDSUBPS X8, X2, X2
	// t3
	VMOVAPS X12, X13
	VSHUFPS $0x00, X13, X13, X13
	VMOVAPS X12, X14
	VSHUFPS $0x55, X14, X14, X14
	VMOVAPS X3, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMULPS X13, X3, X3
	VMULPS X14, X8, X8
	VADDSUBPS X8, X3, X3
	// Radix-4
	VMOVAPS X0, X4
	VADDPS  X2, X4, X4
	VMOVAPS X0, X5
	VSUBPS  X2, X5, X5
	VMOVAPS X1, X6
	VADDPS  X3, X6, X6
	VMOVAPS X1, X7
	VSUBPS  X3, X7, X7
	VMOVAPS X4, X0
	VADDPS  X6, X0, X0
	VMOVAPS X4, X2
	VSUBPS  X6, X2, X2
	VMOVAPS X7, X8
	VSHUFPS $0xB1, X8, X8, X8
	VMOVUPS ·maskNegLoPS(SB), X14
	VXORPS  X14, X8, X8
	VMOVAPS X5, X1
	VADDPS  X8, X1, X1
	VMOVAPS X7, X9
	VSHUFPS $0xB1, X9, X9, X9
	VXORPS  X15, X9, X9
	VMOVAPS X5, X3
	VADDPS  X9, X3, X3
	MOVSD X0, 24(SI)
	MOVSD X1, 56(SI)
	MOVSD X2, 88(SI)
	MOVSD X3, 120(SI)

	ADDQ $128, SI
	DECQ CX
	JNZ  m24_32_avx2_inv_stage2_loop

	// Stage 3: 1 radix-2 butterfly (Stride 16)
	MOVQ R8, SI
	MOVQ $16, CX

m24_32_avx2_inv_stage3_loop:
	MOVSD (SI), X0
	MOVSD 128(SI), X1
	MOVQ $16, AX
	SUBQ CX, AX
	MOVSD (R10)(AX*8), X10
	VXORPS X15, X10, X10 // conj
	// t = X1 * X10
	VMOVAPS X10, X11
	VSHUFPS $0x00, X11, X11, X11
	VMOVAPS X10, X12
	VSHUFPS $0x55, X12, X12, X12
	VMOVAPS X1, X13
	VSHUFPS $0xB1, X13, X13, X13
	VMULPS X11, X1, X1
	VMULPS X12, X13, X13
	VADDSUBPS X13, X1, X1
	VMOVAPS X0, X2
	VADDPS  X1, X0, X0
	VSUBPS  X1, X2, X2
	MOVSD X0, (SI)
	MOVSD X2, 128(SI)
	ADDQ $8, SI
	DECQ CX
	JNZ  m24_32_avx2_inv_stage3_loop

	// Scale by 1/32
	MOVSS ·thirtySecond32(SB), X15
	VSHUFPS $0x00, X15, X15, X15
	MOVQ $16, CX
	MOVQ R8, SI
m24_32_avx2_inv_scale_loop:
	VMOVUPS (SI), X0
	VMULPS X15, X0, X0
	VMOVUPS X0, (SI)
	ADDQ $16, SI
	DECQ CX
	JNZ m24_32_avx2_inv_scale_loop

	// Copy to dst if needed
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   m24_32_avx2_inv_done

	MOVQ $16, CX
	MOVQ R8, SI
m24_32_avx2_inv_copy_loop:
	VMOVUPS (SI), X0
	VMOVUPS X0, (R14)
	ADDQ $16, SI
	ADDQ $16, R14
	DECQ CX
	JNZ m24_32_avx2_inv_copy_loop

m24_32_avx2_inv_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

m24_32_avx2_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

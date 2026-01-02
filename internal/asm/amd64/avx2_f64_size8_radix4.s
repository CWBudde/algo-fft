//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-8 Radix-4 (complex128) FFT Kernels for AMD64 (complex128)
// ===========================================================================
//
// This file contains fully-unrolled FFT kernels optimized for size 8.
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 8, complex128, radix-4 (mixed-radix) variant
// ===========================================================================
TEXT ·ForwardAVX2Size8Radix4Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ R8, R14             // R14 = original dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 8)

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_128_r4_fwd_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_fwd_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_fwd_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_128_r4_fwd_use_dst
	MOVQ R11, R8             // In-place: use scratch

size8_128_r4_fwd_use_dst:
	// =======================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =======================================================================
	// complex128 is 16 bytes, use SHLQ $4 for indexing
	MOVQ (R12), DX           // bitrev[0]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X0
	MOVQ 8(R12), DX          // bitrev[1]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X1
	MOVQ 16(R12), DX         // bitrev[2]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X2
	MOVQ 24(R12), DX         // bitrev[3]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X3
	MOVQ 32(R12), DX         // bitrev[4]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X4
	MOVQ 40(R12), DX         // bitrev[5]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X5
	MOVQ 48(R12), DX         // bitrev[6]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X6
	MOVQ 56(R12), DX         // bitrev[7]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X7
	// Now: X0=x0, X1=x1, X2=x2, X3=x3, X4=x4, X5=x5, X6=x6, X7=x7

	// =======================================================================
	// Scalar-style mixed-radix computation (correctness-focused)
	// =======================================================================
	// Build sign masks: X15 = [0, signbit] for -i, X14 = [signbit, 0] for +i
	MOVQ ·signbit64(SB), AX
	VMOVQ AX, X14
	VPERMILPD $1, X14, X15

	// Radix-4 butterfly 1: [x0, x1, x2, x3]
	VADDPD X2, X0, X8        // t0
	VSUBPD X2, X0, X9        // t1
	VADDPD X3, X1, X10       // t2
	VSUBPD X3, X1, X11       // t3
	VPERMILPD $1, X11, X12
	VXORPD X15, X12, X12     // t3 * (-i)
	VADDPD X10, X8, X0       // a0
	VSUBPD X10, X8, X2       // a2
	VADDPD X12, X9, X1       // a1
	VSUBPD X12, X9, X3       // a3

	// Radix-4 butterfly 2: [x4, x5, x6, x7]
	VADDPD X6, X4, X8
	VSUBPD X6, X4, X9
	VADDPD X7, X5, X10
	VSUBPD X7, X5, X11
	VPERMILPD $1, X11, X12
	VXORPD X15, X12, X12     // t3 * (-i)
	VADDPD X10, X8, X4       // a4
	VSUBPD X10, X8, X6       // a6
	VADDPD X12, X9, X5       // a5
	VSUBPD X12, X9, X7       // a7

	// Stage 2: radix-2 with twiddles
	VADDPD X4, X0, X11       // y0
	VSUBPD X4, X0, X12       // y4
	MOVUPD X11, (R8)
	MOVUPD X12, 64(R8)

	// w1 * a5
	MOVUPD 16(R10), X8       // w1
	VPERMILPD $1, X8, X9     // w1 swap
	VMULPD X8, X5, X13       // tmp1
	VMULPD X9, X5, X14       // tmp2
	VPERMILPD $1, X13, X15
	VSUBPD X15, X13, X13     // realvec
	VPERMILPD $1, X14, X15
	VADDPD X15, X14, X14     // imagvec
	VUNPCKLPD X14, X13, X13  // w1*a5
	VADDPD X13, X1, X0       // y1
	VSUBPD X13, X1, X1       // y5
	MOVUPD X0, 16(R8)
	MOVUPD X1, 80(R8)

	// Save a2, a3 before overwriting
	VMOVAPD X2, X10
	VMOVAPD X3, X11

	// w2 * a6
	MOVUPD 32(R10), X8       // w2
	VPERMILPD $1, X8, X9
	VMULPD X8, X6, X13
	VMULPD X9, X6, X14
	VPERMILPD $1, X13, X15
	VSUBPD X15, X13, X13
	VPERMILPD $1, X14, X15
	VADDPD X15, X14, X14
	VUNPCKLPD X14, X13, X13  // w2*a6
	VADDPD X13, X10, X2      // y2
	VSUBPD X13, X10, X3      // y6
	MOVUPD X2, 32(R8)
	MOVUPD X3, 96(R8)

	// w3 * a7
	MOVUPD 48(R10), X8       // w3
	VPERMILPD $1, X8, X9
	VMULPD X8, X7, X13
	VMULPD X9, X7, X14
	VPERMILPD $1, X13, X15
	VSUBPD X15, X13, X13
	VPERMILPD $1, X14, X15
	VADDPD X15, X14, X14
	VUNPCKLPD X14, X13, X13  // w3*a7
	VADDPD X13, X11, X4      // y3
	VSUBPD X13, X11, X5      // y7
	MOVUPD X4, 48(R8)
	MOVUPD X5, 112(R8)

	// Copy to dst if needed
	CMPQ R8, R14
	JE   size8_128_r4_fwd_done

	MOVUPD (R8), X0
	MOVUPD X0, (R14)
	MOVUPD 16(R8), X0
	MOVUPD X0, 16(R14)
	MOVUPD 32(R8), X0
	MOVUPD X0, 32(R14)
	MOVUPD 48(R8), X0
	MOVUPD X0, 48(R14)
	MOVUPD 64(R8), X0
	MOVUPD X0, 64(R14)
	MOVUPD 80(R8), X0
	MOVUPD X0, 80(R14)
	MOVUPD 96(R8), X0
	MOVUPD X0, 96(R14)
	MOVUPD 112(R8), X0
	MOVUPD X0, 112(R14)

size8_128_r4_fwd_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size8_128_r4_fwd_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 8, complex128, radix-4 (mixed-radix) variant
// ===========================================================================
// Uses +i instead of -i, conjugated twiddles, and 1/8 scaling
TEXT ·InverseAVX2Size8Radix4Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_128_r4_inv_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_inv_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_128_r4_inv_use_dst
	MOVQ R11, R8

size8_128_r4_inv_use_dst:
	// Bit-reversal permutation
	MOVQ (R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X0
	MOVQ 8(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X1
	MOVQ 16(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X2
	MOVQ 24(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X3
	MOVQ 32(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X4
	MOVQ 40(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X5
	MOVQ 48(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X6
	MOVQ 56(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X7

	// Scalar-style mixed-radix computation (inverse)
	// Build sign masks: X15 = [0, signbit] for -i, X14 = [signbit, 0] for +i
	MOVQ ·signbit64(SB), AX
	VMOVQ AX, X14
	VPERMILPD $1, X14, X15

	// Radix-4 butterfly 1 (+i)
	VADDPD X2, X0, X8
	VSUBPD X2, X0, X9
	VADDPD X3, X1, X10
	VSUBPD X3, X1, X11
	VPERMILPD $1, X11, X12
	VXORPD X14, X12, X12     // t3 * (+i)
	VADDPD X10, X8, X0       // a0
	VSUBPD X10, X8, X2       // a2
	VADDPD X12, X9, X1       // a1
	VSUBPD X12, X9, X3       // a3

	// Radix-4 butterfly 2 (+i)
	VADDPD X6, X4, X8
	VSUBPD X6, X4, X9
	VADDPD X7, X5, X10
	VSUBPD X7, X5, X11
	VPERMILPD $1, X11, X12
	VXORPD X14, X12, X12
	VADDPD X10, X8, X4       // a4
	VSUBPD X10, X8, X6       // a6
	VADDPD X12, X9, X5       // a5
	VSUBPD X12, X9, X7       // a7

	// Stage 2 with conjugated twiddles
	VADDPD X4, X0, X11       // y0
	VSUBPD X4, X0, X12       // y4

	// Apply 1/8 scaling for y0/y4 and store
	MOVQ ·eighth64(SB), AX
	VMOVQ AX, X8
	VMOVDDUP X8, X8
	VMULPD X8, X11, X11
	VMULPD X8, X12, X12
	MOVUPD X11, (R8)
	MOVUPD X12, 64(R8)

	// conj(w1) * a5
	MOVUPD 16(R10), X8
	MOVQ ·signbit64(SB), AX
	VMOVQ AX, X9
	VPERMILPD $1, X9, X9
	VXORPD X9, X8, X8
	VPERMILPD $1, X8, X9
	VMULPD X8, X5, X13
	VMULPD X9, X5, X14
	VPERMILPD $1, X13, X15
	VSUBPD X15, X13, X13
	VPERMILPD $1, X14, X15
	VADDPD X15, X14, X14
	VUNPCKLPD X14, X13, X13
	VADDPD X13, X1, X0       // y1
	VSUBPD X13, X1, X1       // y5
	MOVQ ·eighth64(SB), AX
	VMOVQ AX, X8
	VMOVDDUP X8, X8
	VMULPD X8, X0, X0
	VMULPD X8, X1, X1
	MOVUPD X0, 16(R8)
	MOVUPD X1, 80(R8)

	// Save a2, a3 before overwriting
	VMOVAPD X2, X10
	VMOVAPD X3, X11

	// conj(w2) * a6
	MOVUPD 32(R10), X8
	MOVQ ·signbit64(SB), AX
	VMOVQ AX, X9
	VPERMILPD $1, X9, X9
	VXORPD X9, X8, X8
	VPERMILPD $1, X8, X9
	VMULPD X8, X6, X13
	VMULPD X9, X6, X14
	VPERMILPD $1, X13, X15
	VSUBPD X15, X13, X13
	VPERMILPD $1, X14, X15
	VADDPD X15, X14, X14
	VUNPCKLPD X14, X13, X13
	VADDPD X13, X10, X2      // y2
	VSUBPD X13, X10, X3      // y6
	MOVQ ·eighth64(SB), AX
	VMOVQ AX, X8
	VMOVDDUP X8, X8
	VMULPD X8, X2, X2
	VMULPD X8, X3, X3
	MOVUPD X2, 32(R8)
	MOVUPD X3, 96(R8)

	// conj(w3) * a7
	MOVUPD 48(R10), X8
	MOVQ ·signbit64(SB), AX
	VMOVQ AX, X9
	VPERMILPD $1, X9, X9
	VXORPD X9, X8, X8
	VPERMILPD $1, X8, X9
	VMULPD X8, X7, X13
	VMULPD X9, X7, X14
	VPERMILPD $1, X13, X15
	VSUBPD X15, X13, X13
	VPERMILPD $1, X14, X15
	VADDPD X15, X14, X14
	VUNPCKLPD X14, X13, X13
	VADDPD X13, X11, X4      // y3
	VSUBPD X13, X11, X5      // y7
	MOVQ ·eighth64(SB), AX
	VMOVQ AX, X8
	VMOVDDUP X8, X8
	VMULPD X8, X4, X4
	VMULPD X8, X5, X5
	MOVUPD X4, 48(R8)
	MOVUPD X5, 112(R8)

	// Copy to dst if needed
	CMPQ R8, R14
	JE   size8_128_r4_inv_done

	MOVUPD (R8), X0
	MOVUPD X0, (R14)
	MOVUPD 16(R8), X0
	MOVUPD X0, 16(R14)
	MOVUPD 32(R8), X0
	MOVUPD X0, 32(R14)
	MOVUPD 48(R8), X0
	MOVUPD X0, 48(R14)
	MOVUPD 64(R8), X0
	MOVUPD X0, 64(R14)
	MOVUPD 80(R8), X0
	MOVUPD X0, 80(R14)
	MOVUPD 96(R8), X0
	MOVUPD X0, 96(R14)
	MOVUPD 112(R8), X0
	MOVUPD X0, 112(R14)

size8_128_r4_inv_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size8_128_r4_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

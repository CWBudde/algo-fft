//go:build amd64 && fft_asm && !purego

// ===========================================================================
// AVX2 Size-4 FFT Kernels for AMD64 (complex128)
// ===========================================================================
//
// This file contains a fully-unrolled radix-4 FFT kernel optimized for size 4.
// For size 4, the radix-4 FFT is a single radix-4 butterfly with no twiddles.
//
// Radix-4 Butterfly Algorithm:
//   Given 4 input points x[0], x[1], x[2], x[3]:
//
//   t0 = x[0] + x[2]
//   t1 = x[0] - x[2]
//   t2 = x[1] + x[3]
//   t3 = x[1] - x[3]
//
//   y[0] = t0 + t2
//   y[1] = t1 + t3*(-i)    // multiply by -i: (r,i) -> (i,-r)
//   y[2] = t0 - t2
//   y[3] = t1 - t3*(-i)
//
// For inverse FFT, replace -i with +i: (r,i) -> (-i,r)
// and apply 1/4 scaling at the end.
//
// ===========================================================================

#include "textflag.h"

DATA ·scaleQuarterPD+0(SB)/8, $0x3fd0000000000000 // 0.25
GLOBL ·scaleQuarterPD(SB), RODATA|NOPTR, $8

// ===========================================================================
// Forward transform, size 4, complex128, radix-4
// ===========================================================================
// Pure radix-4 implementation - no bit-reversal needed for size 4!
TEXT ·forwardAVX2Size4Radix4Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ R8, R14             // R14 = original dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer (unused)
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer (unused)
	MOVQ src+32(FP), R13     // R13 = n (should be 4)

	// Verify n == 4
	CMPQ R13, $4
	JNE  size4_128_fwd_return_false

	// Validate all slice lengths >= 4
	MOVQ dst+8(FP), AX
	CMPQ AX, $4
	JL   size4_128_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $4
	JL   size4_128_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $4
	JL   size4_128_fwd_return_false

	// bitrev is unused for size-4 radix-4; allow nil/empty.

	// Load x0..x3
	MOVUPD (R9), X0
	MOVUPD 16(R9), X1
	MOVUPD 32(R9), X2
	MOVUPD 48(R9), X3

	// t0 = x0 + x2, t1 = x0 - x2
	VADDPD X2, X0, X4
	VSUBPD X2, X0, X5
	// t2 = x1 + x3, t3 = x1 - x3
	VADDPD X3, X1, X6
	VSUBPD X3, X1, X7

	// t3NegI = swap(t3) with sign toggle on high lane -> (im, -re)
	VPERMILPD $1, X7, X8
	VXORPD ·maskSignHiPD(SB), X8, X8

	// y0, y2
	VADDPD X6, X4, X9
	VSUBPD X6, X4, X10
	// y1, y3
	VADDPD X8, X5, X11
	VSUBPD X8, X5, X12

	// Store results
	MOVUPD X9, (R14)
	MOVUPD X11, 16(R14)
	MOVUPD X10, 32(R14)
	MOVUPD X12, 48(R14)

	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size4_128_fwd_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 4, complex128, radix-4
// ===========================================================================
TEXT ·inverseAVX2Size4Radix4Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14             // R14 = original dst pointer
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10 // unused
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12  // unused
	MOVQ src+32(FP), R13

	// Verify n == 4
	CMPQ R13, $4
	JNE  size4_128_inv_return_false

	// Validate all slice lengths >= 4
	MOVQ dst+8(FP), AX
	CMPQ AX, $4
	JL   size4_128_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $4
	JL   size4_128_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $4
	JL   size4_128_inv_return_false

	// Load x0..x3
	MOVUPD (R9), X0
	MOVUPD 16(R9), X1
	MOVUPD 32(R9), X2
	MOVUPD 48(R9), X3

	// t0 = x0 + x2, t1 = x0 - x2
	VADDPD X2, X0, X4
	VSUBPD X2, X0, X5
	// t2 = x1 + x3, t3 = x1 - x3
	VADDPD X3, X1, X6
	VSUBPD X3, X1, X7

	// t3PosI = swap(t3) with sign toggle on low lane -> (-im, re)
	VPERMILPD $1, X7, X8
	VXORPD ·maskSignLoPD(SB), X8, X8

	// y0, y2
	VADDPD X6, X4, X9
	VSUBPD X6, X4, X10
	// y1, y3
	VADDPD X8, X5, X11
	VSUBPD X8, X5, X12

	// Scale by 1/4
	MOVSD ·scaleQuarterPD(SB), X15
	VMOVDDUP X15, X15
	VMULPD X15, X9, X9
	VMULPD X15, X11, X11
	VMULPD X15, X10, X10
	VMULPD X15, X12, X12

	// Store results
	MOVUPD X9, (R14)
	MOVUPD X11, 16(R14)
	MOVUPD X10, 32(R14)
	MOVUPD X12, 48(R14)

	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size4_128_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

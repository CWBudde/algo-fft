//go:build amd64 && fft_asm && !purego

// ===========================================================================
// SSE2 Size-4 FFT Kernels for AMD64 (complex128)
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

DATA ·sse2MaskSignLoPD+0(SB)/8, $0x8000000000000000
DATA ·sse2MaskSignLoPD+8(SB)/8, $0x0000000000000000
GLOBL ·sse2MaskSignLoPD(SB), RODATA|NOPTR, $16

DATA ·sse2MaskSignHiPD+0(SB)/8, $0x0000000000000000
DATA ·sse2MaskSignHiPD+8(SB)/8, $0x8000000000000000
GLOBL ·sse2MaskSignHiPD(SB), RODATA|NOPTR, $16

DATA ·sse2ScaleQuarterPD+0(SB)/8, $0x3fd0000000000000 // 0.25
GLOBL ·sse2ScaleQuarterPD(SB), RODATA|NOPTR, $8

// ===========================================================================
// Forward transform, size 4, complex128, radix-4
// ===========================================================================
TEXT ·forwardSSE2Size4Radix4Complex128Asm(SB), NOSPLIT, $0-121
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
	JNE  size4_sse2_128_fwd_return_false

	// Validate all slice lengths >= 4
	MOVQ dst+8(FP), AX
	CMPQ AX, $4
	JL   size4_sse2_128_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $4
	JL   size4_sse2_128_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $4
	JL   size4_sse2_128_fwd_return_false

	// Load x0..x3
	MOVUPD (R9), X0
	MOVUPD 16(R9), X1
	MOVUPD 32(R9), X2
	MOVUPD 48(R9), X3

	// t0 = x0 + x2, t1 = x0 - x2
	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	// t2 = x1 + x3, t3 = x1 - x3
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// t3NegI = swap(t3) with sign toggle on high lane -> (im, -re)
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD ·sse2MaskSignHiPD(SB), X8

	// y0, y2
	MOVAPD X4, X9
	ADDPD X6, X9
	MOVAPD X4, X10
	SUBPD X6, X10
	// y1, y3
	MOVAPD X5, X11
	ADDPD X8, X11
	MOVAPD X5, X12
	SUBPD X8, X12

	// Store results
	MOVUPD X9, (R14)
	MOVUPD X11, 16(R14)
	MOVUPD X10, 32(R14)
	MOVUPD X12, 48(R14)

	MOVB $1, ret+120(FP)
	RET

size4_sse2_128_fwd_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 4, complex128, radix-4
// ===========================================================================
TEXT ·inverseSSE2Size4Radix4Complex128Asm(SB), NOSPLIT, $0-121
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
	JNE  size4_sse2_128_inv_return_false

	// Validate all slice lengths >= 4
	MOVQ dst+8(FP), AX
	CMPQ AX, $4
	JL   size4_sse2_128_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $4
	JL   size4_sse2_128_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $4
	JL   size4_sse2_128_inv_return_false

	// Load x0..x3
	MOVUPD (R9), X0
	MOVUPD 16(R9), X1
	MOVUPD 32(R9), X2
	MOVUPD 48(R9), X3

	// t0 = x0 + x2, t1 = x0 - x2
	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	// t2 = x1 + x3, t3 = x1 - x3
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// t3PosI = swap(t3) with sign toggle on low lane -> (-im, re)
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD ·sse2MaskSignLoPD(SB), X8

	// y0, y2
	MOVAPD X4, X9
	ADDPD X6, X9
	MOVAPD X4, X10
	SUBPD X6, X10
	// y1, y3
	MOVAPD X5, X11
	ADDPD X8, X11
	MOVAPD X5, X12
	SUBPD X8, X12

	// Scale by 1/4
	MOVSD ·sse2ScaleQuarterPD(SB), X15
	SHUFPD $0, X15, X15
	MULPD X15, X9
	MULPD X15, X11
	MULPD X15, X10
	MULPD X15, X12

	// Store results
	MOVUPD X9, (R14)
	MOVUPD X11, 16(R14)
	MOVUPD X10, 32(R14)
	MOVUPD X12, 48(R14)

	MOVB $1, ret+120(FP)
	RET

size4_sse2_128_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

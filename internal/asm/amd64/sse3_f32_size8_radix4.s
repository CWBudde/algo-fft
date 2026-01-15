//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-8 Radix-4 FFT Kernels for AMD64 (complex64)
// ===========================================================================
//
// Mixed-radix 4×2 FFT kernel for size 8.
//
// Stage 1 (radix-4): Two 4-point butterflies
// Stage 2 (radix-2): Four 2-point butterflies with twiddles
//
// Mixed-radix bit-reversal for n=8: [0, 2, 4, 6, 1, 3, 5, 7]
//
// Radix-4 Butterfly:
//   t0 = x0 + x2
//   t1 = x0 - x2
//   t2 = x1 + x3
//   t3 = x1 - x3
//
//   y0 = t0 + t2
//   y1 = t1 + t3*(-i)   // forward
//   y2 = t0 - t2
//   y3 = t1 - t3*(-i)   // forward
//
// Inverse uses +i and applies 1/8 scaling.
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 8, complex64, radix-4 (mixed-radix) variant
// ===========================================================================
TEXT ·ForwardSSE3Size8Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ R8, R14             // R14 = original dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 8)

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_r4_sse2_fwd_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_r4_sse2_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_r4_sse2_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_r4_sse2_fwd_return_false

	MOVQ $8, AX
	CMPQ AX, $8
	JL   size8_r4_sse2_fwd_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_r4_sse2_fwd_use_dst
	MOVQ R11, R8             // In-place: use scratch

size8_r4_sse2_fwd_use_dst:
	// ==================================================================
	// Bit-reversal load: pattern [0, 2, 4, 6, 1, 3, 5, 7] * 8 bytes
	// ==================================================================
	MOVSD 0(R9), X0     // src[0]
	MOVSD 16(R9), X1    // src[2]
	MOVSD 32(R9), X2    // src[4]
	MOVSD 48(R9), X3    // src[6]
	MOVSD 8(R9), X4     // src[1]
	MOVSD 24(R9), X5    // src[3]
	MOVSD 40(R9), X6    // src[5]
	MOVSD 56(R9), X7    // src[7]

	// ==================================================================
	// Stage 1: Radix-4 butterfly 1 on [x0, x1, x2, x3]
	// ==================================================================
	// t0 = x0 + x2, t1 = x0 - x2
	MOVAPS X0, X8
	ADDPS  X2, X8            // t0
	MOVAPS X0, X9
	SUBPS  X2, X9            // t1

	// t2 = x1 + x3, t3 = x1 - x3
	MOVAPS X1, X10
	ADDPS  X3, X10           // t2
	MOVAPS X1, X11
	SUBPS  X3, X11           // t3

	// t3 * (-i) = (im, -re) via swap and negate high lane
	MOVAPS X11, X12
	SHUFPS $0xB1, X12, X12   // swap: (re, im) -> (im, re)
	MOVUPS ·maskNegHiPS(SB), X13
	XORPS  X13, X12          // negate high lane: (im, -re)

	// Final outputs for butterfly 1
	MOVAPS X8, X0
	ADDPS  X10, X0           // a0 = t0 + t2
	MOVAPS X9, X1
	ADDPS  X12, X1           // a1 = t1 + t3*(-i)
	MOVAPS X8, X2
	SUBPS  X10, X2           // a2 = t0 - t2
	MOVAPS X9, X3
	SUBPS  X12, X3           // a3 = t1 - t3*(-i)

	// ==================================================================
	// Stage 1: Radix-4 butterfly 2 on [x4, x5, x6, x7]
	// ==================================================================
	// t0 = x4 + x6, t1 = x4 - x6
	MOVAPS X4, X8
	ADDPS  X6, X8            // t0
	MOVAPS X4, X9
	SUBPS  X6, X9            // t1

	// t2 = x5 + x7, t3 = x5 - x7
	MOVAPS X5, X10
	ADDPS  X7, X10           // t2
	MOVAPS X5, X11
	SUBPS  X7, X11           // t3

	// t3 * (-i)
	MOVAPS X11, X12
	SHUFPS $0xB1, X12, X12
	XORPS  X13, X12          // X13 still has maskNegHiPS

	// Final outputs for butterfly 2
	MOVAPS X8, X4
	ADDPS  X10, X4           // a4 = t0 + t2
	MOVAPS X9, X5
	ADDPS  X12, X5           // a5 = t1 + t3*(-i)
	MOVAPS X8, X6
	SUBPS  X10, X6           // a6 = t0 - t2
	MOVAPS X9, X7
	SUBPS  X12, X7           // a7 = t1 - t3*(-i)

	// ==================================================================
	// Stage 2: Radix-2 butterflies with twiddles
	// ==================================================================

	// y0 = a0 + a4, y4 = a0 - a4 (twiddle w^0 = 1)
	MOVAPS X0, X8
	ADDPS  X4, X8            // y0
	MOVAPS X0, X9
	SUBPS  X4, X9            // y4

	// w1 * a5: complex multiply
	// Load w1 from twiddle[1]
	MOVSD  8(R10), X10       // w1 = (re, im)
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11   // broadcast w1.re
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12   // broadcast w1.im
	MOVAPS X5, X13
	SHUFPS $0xB1, X13, X13   // a5 swapped: (im, re)
	MOVAPS X5, X14
	MULPS  X11, X14          // a5.re * w1.re, a5.im * w1.re
	MULPS  X12, X13          // a5.im * w1.im, a5.re * w1.im
	ADDSUBPS X13, X14        // (re*re - im*im, im*re + re*im)
	// y1 = a1 + w1*a5, y5 = a1 - w1*a5
	MOVAPS X1, X10
	ADDPS  X14, X10          // y1
	MOVAPS X1, X11
	SUBPS  X14, X11          // y5

	// w2 * a6: complex multiply
	MOVSD  16(R10), X12      // w2
	MOVAPS X12, X13
	SHUFPS $0x00, X13, X13   // broadcast w2.re
	MOVAPS X12, X14
	SHUFPS $0x55, X14, X14   // broadcast w2.im
	MOVAPS X6, X15
	SHUFPS $0xB1, X15, X15   // a6 swapped
	MOVAPS X6, X0
	MULPS  X13, X0           // a6 * w2.re
	MULPS  X14, X15          // a6_swapped * w2.im
	ADDSUBPS X15, X0         // w2 * a6
	// y2 = a2 + w2*a6, y6 = a2 - w2*a6
	MOVAPS X2, X12
	ADDPS  X0, X12           // y2
	MOVAPS X2, X13
	SUBPS  X0, X13           // y6

	// w3 * a7: complex multiply
	MOVSD  24(R10), X14      // w3
	MOVAPS X14, X15
	SHUFPS $0x00, X15, X15   // broadcast w3.re
	MOVAPS X14, X0
	SHUFPS $0x55, X0, X0     // broadcast w3.im
	MOVAPS X7, X1
	SHUFPS $0xB1, X1, X1     // a7 swapped
	MOVAPS X7, X2
	MULPS  X15, X2           // a7 * w3.re
	MULPS  X0, X1            // a7_swapped * w3.im
	ADDSUBPS X1, X2          // w3 * a7
	// y3 = a3 + w3*a7, y7 = a3 - w3*a7
	MOVAPS X3, X14
	ADDPS  X2, X14           // y3
	MOVAPS X3, X15
	SUBPS  X2, X15           // y7

	// ==================================================================
	// Store results
	// ==================================================================
	MOVSD X8, (R8)           // y0
	MOVSD X10, 8(R8)         // y1
	MOVSD X12, 16(R8)        // y2
	MOVSD X14, 24(R8)        // y3
	MOVSD X9, 32(R8)         // y4
	MOVSD X11, 40(R8)        // y5
	MOVSD X13, 48(R8)        // y6
	MOVSD X15, 56(R8)        // y7

	// Copy to dst if needed
	CMPQ R8, R14
	JE   size8_r4_sse2_fwd_done

	MOVQ (R8), AX
	MOVQ AX, (R14)
	MOVQ 8(R8), AX
	MOVQ AX, 8(R14)
	MOVQ 16(R8), AX
	MOVQ AX, 16(R14)
	MOVQ 24(R8), AX
	MOVQ AX, 24(R14)
	MOVQ 32(R8), AX
	MOVQ AX, 32(R14)
	MOVQ 40(R8), AX
	MOVQ AX, 40(R14)
	MOVQ 48(R8), AX
	MOVQ AX, 48(R14)
	MOVQ 56(R8), AX
	MOVQ AX, 56(R14)

size8_r4_sse2_fwd_done:
	MOVB $1, ret+96(FP)
	RET

size8_r4_sse2_fwd_return_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 8, complex64, radix-4 (mixed-radix) variant
// ===========================================================================
// Same as forward but with +i instead of -i for radix-4,
// conjugated twiddles, and 1/8 scaling.
TEXT ·InverseSSE3Size8Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_r4_sse2_inv_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_r4_sse2_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_r4_sse2_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_r4_sse2_inv_return_false

	MOVQ $8, AX
	CMPQ AX, $8
	JL   size8_r4_sse2_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_r4_sse2_inv_use_dst
	MOVQ R11, R8

size8_r4_sse2_inv_use_dst:
	// ==================================================================
	// Bit-reversal load: pattern [0, 2, 4, 6, 1, 3, 5, 7] * 8 bytes
	// ==================================================================
	MOVSD 0(R9), X0     // src[0]
	MOVSD 16(R9), X1    // src[2]
	MOVSD 32(R9), X2    // src[4]
	MOVSD 48(R9), X3    // src[6]
	MOVSD 8(R9), X4     // src[1]
	MOVSD 24(R9), X5    // src[3]
	MOVSD 40(R9), X6    // src[5]
	MOVSD 56(R9), X7    // src[7]

	// ==================================================================
	// Stage 1: Radix-4 butterfly 1 with +i (inverse)
	// ==================================================================
	MOVAPS X0, X8
	ADDPS  X2, X8            // t0
	MOVAPS X0, X9
	SUBPS  X2, X9            // t1

	MOVAPS X1, X10
	ADDPS  X3, X10           // t2
	MOVAPS X1, X11
	SUBPS  X3, X11           // t3

	// t3 * (+i) = (-im, re) via swap and negate low lane
	MOVAPS X11, X12
	SHUFPS $0xB1, X12, X12
	MOVUPS ·maskNegLoPS(SB), X13
	XORPS  X13, X12          // (-im, re)

	MOVAPS X8, X0
	ADDPS  X10, X0           // a0
	MOVAPS X9, X1
	ADDPS  X12, X1           // a1 = t1 + t3*(+i)
	MOVAPS X8, X2
	SUBPS  X10, X2           // a2
	MOVAPS X9, X3
	SUBPS  X12, X3           // a3 = t1 - t3*(+i)

	// ==================================================================
	// Stage 1: Radix-4 butterfly 2 with +i (inverse)
	// ==================================================================
	MOVAPS X4, X8
	ADDPS  X6, X8
	MOVAPS X4, X9
	SUBPS  X6, X9

	MOVAPS X5, X10
	ADDPS  X7, X10
	MOVAPS X5, X11
	SUBPS  X7, X11

	MOVAPS X11, X12
	SHUFPS $0xB1, X12, X12
	XORPS  X13, X12          // X13 still has maskNegLoPS

	MOVAPS X8, X4
	ADDPS  X10, X4
	MOVAPS X9, X5
	ADDPS  X12, X5
	MOVAPS X8, X6
	SUBPS  X10, X6
	MOVAPS X9, X7
	SUBPS  X12, X7

	// ==================================================================
	// Stage 2: Radix-2 with conjugated twiddles
	// ==================================================================

	// y0, y4 (twiddle = 1)
	MOVAPS X0, X8
	ADDPS  X4, X8            // y0
	MOVAPS X0, X9
	SUBPS  X4, X9            // y4

	// Load mask for conjugation (negate imag)
	MOVUPS ·maskNegHiPS(SB), X15

	// conj(w1) * a5
	MOVSD  8(R10), X10       // w1
	XORPS  X15, X10          // conjugate: negate imag
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11   // broadcast conj(w1).re
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12   // broadcast conj(w1).im
	MOVAPS X5, X13
	SHUFPS $0xB1, X13, X13   // a5 swapped
	MOVAPS X5, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14        // conj(w1) * a5
	MOVAPS X1, X10
	ADDPS  X14, X10          // y1
	MOVAPS X1, X11
	SUBPS  X14, X11          // y5

	// conj(w2) * a6
	MOVSD  16(R10), X12
	XORPS  X15, X12          // conjugate
	MOVAPS X12, X13
	SHUFPS $0x00, X13, X13
	MOVAPS X12, X14
	SHUFPS $0x55, X14, X14
	MOVAPS X6, X0
	SHUFPS $0xB1, X0, X0
	MOVAPS X6, X1
	MULPS  X13, X1
	MULPS  X14, X0
	ADDSUBPS X0, X1          // conj(w2) * a6
	MOVAPS X2, X12
	ADDPS  X1, X12           // y2
	MOVAPS X2, X13
	SUBPS  X1, X13           // y6

	// conj(w3) * a7
	MOVSD  24(R10), X14
	XORPS  X15, X14          // conjugate
	MOVAPS X14, X0
	SHUFPS $0x00, X0, X0
	MOVAPS X14, X1
	SHUFPS $0x55, X1, X1
	MOVAPS X7, X2
	SHUFPS $0xB1, X2, X2
	MOVAPS X7, X4
	MULPS  X0, X4
	MULPS  X1, X2
	ADDSUBPS X2, X4          // conj(w3) * a7
	MOVAPS X3, X14
	ADDPS  X4, X14           // y3
	MOVAPS X3, X0
	SUBPS  X4, X0            // y7

	// ==================================================================
	// Apply 1/8 scaling
	// ==================================================================
	XORPS  X1, X1
	MOVSS  ·eighth32(SB), X1
	SHUFPS $0x00, X1, X1     // broadcast 0.125
	MULPS  X1, X8            // y0
	MULPS  X1, X10           // y1
	MULPS  X1, X12           // y2
	MULPS  X1, X14           // y3
	MULPS  X1, X9            // y4
	MULPS  X1, X11           // y5
	MULPS  X1, X13           // y6
	MULPS  X1, X0            // y7

	// ==================================================================
	// Store results
	// ==================================================================
	MOVSD X8, (R8)
	MOVSD X10, 8(R8)
	MOVSD X12, 16(R8)
	MOVSD X14, 24(R8)
	MOVSD X9, 32(R8)
	MOVSD X11, 40(R8)
	MOVSD X13, 48(R8)
	MOVSD X0, 56(R8)

	// Copy to dst if needed
	CMPQ R8, R14
	JE   size8_r4_sse2_inv_done

	MOVQ (R8), AX
	MOVQ AX, (R14)
	MOVQ 8(R8), AX
	MOVQ AX, 8(R14)
	MOVQ 16(R8), AX
	MOVQ AX, 16(R14)
	MOVQ 24(R8), AX
	MOVQ AX, 24(R14)
	MOVQ 32(R8), AX
	MOVQ AX, 32(R14)
	MOVQ 40(R8), AX
	MOVQ AX, 40(R14)
	MOVQ 48(R8), AX
	MOVQ AX, 48(R14)
	MOVQ 56(R8), AX
	MOVQ AX, 56(R14)

size8_r4_sse2_inv_done:
	MOVB $1, ret+96(FP)
	RET

size8_r4_sse2_inv_return_false:
	MOVB $0, ret+96(FP)
	RET

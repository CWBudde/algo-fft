//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-8 Radix-2 FFT Kernels for AMD64 (complex64)
// ===========================================================================
//
// Radix-2 FFT kernel for size 8.
//
// Stage 1 (radix-2): Four 2-point butterflies (stride 1)
// Stage 2 (radix-2): Four 2-point butterflies (stride 2)
// Stage 3 (radix-2): Four 2-point butterflies (stride 4)
//
// Bit-reversal for n=8: [0, 4, 2, 6, 1, 5, 3, 7]
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 8, complex64, radix-2 variant
// ===========================================================================
TEXT ·ForwardSSE3Size8Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ R8, R14             // R14 = original dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 8)

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_r2_sse2_fwd_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_r2_sse2_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_r2_sse2_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_r2_sse2_fwd_return_false

	MOVQ $8, AX
	CMPQ AX, $8
	JL   size8_r2_sse2_fwd_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_r2_sse2_fwd_use_dst
	MOVQ R11, R8             // In-place: use scratch

size8_r2_sse2_fwd_use_dst:
	// ==================================================================
	// Bit-reversal load: pattern [0, 4, 2, 6, 1, 5, 3, 7] * 8 bytes
	// ==================================================================
	MOVSD 0(R9), X0     // src[0]
	MOVSD 32(R9), X1    // src[4]
	MOVSD 16(R9), X2    // src[2]
	MOVSD 48(R9), X3    // src[6]
	MOVSD 8(R9), X4     // src[1]
	MOVSD 40(R9), X5    // src[5]
	MOVSD 24(R9), X6    // src[3]
	MOVSD 56(R9), X7    // src[7]

	// ==================================================================
	// Stage 1: 4 Radix-2 butterflies, stride 1, no twiddles (W^0 = 1)
	// Butterflies on (X0, X1), (X2, X3), (X4, X5), (X6, X7)
	// ==================================================================
	
	// Butterfly 1: X0, X1
	MOVAPS X0, X8
	ADDPS  X1, X8            // a0 = x0 + x1
	MOVAPS X0, X9
	SUBPS  X1, X9            // a1 = x0 - x1
	MOVAPS X8, X0
	MOVAPS X9, X1

	// Butterfly 2: X2, X3
	MOVAPS X2, X8
	ADDPS  X3, X8            // a2 = x2 + x3
	MOVAPS X2, X9
	SUBPS  X3, X9            // a3 = x2 - x3
	MOVAPS X8, X2
	MOVAPS X9, X3

	// Butterfly 3: X4, X5
	MOVAPS X4, X8
	ADDPS  X5, X8            // a4 = x4 + x5
	MOVAPS X4, X9
	SUBPS  X5, X9            // a5 = x4 - x5
	MOVAPS X8, X4
	MOVAPS X9, X5

	// Butterfly 4: X6, X7
	MOVAPS X6, X8
	ADDPS  X7, X8            // a6 = x6 + x7
	MOVAPS X6, X9
	SUBPS  X7, X9            // a7 = x6 - x7
	MOVAPS X8, X6
	MOVAPS X9, X7

	// ==================================================================
	// Stage 2: 4 Radix-2 butterflies, stride 2
	// Butterflies on (X0, X2), (X1, X3), (X4, X6), (X5, X7)
	// ==================================================================

	// Group 1 (twiddle 1): (X0, X2)
	MOVAPS X0, X8
	ADDPS  X2, X8            // b0 = a0 + a2
	MOVAPS X0, X9
	SUBPS  X2, X9            // b2 = a0 - a2
	MOVAPS X8, X0
	MOVAPS X9, X2

	// Group 1 (twiddle -i): (X1, X3)
	// t = a3 * (-i) = (im, -re)
	MOVAPS X3, X10
	SHUFPS $0xB1, X10, X10   // swap: (re, im) -> (im, re)
	MOVUPS ·maskNegHiPS(SB), X11
	XORPS  X11, X10          // negate high: (im, -re)

	MOVAPS X1, X8
	ADDPS  X10, X8           // b1 = a1 + t
	MOVAPS X1, X9
	SUBPS  X10, X9           // b3 = a1 - t
	MOVAPS X8, X1
	MOVAPS X9, X3

	// Group 2 (twiddle 1): (X4, X6)
	MOVAPS X4, X8
	ADDPS  X6, X8            // b4 = a4 + a6
	MOVAPS X4, X9
	SUBPS  X6, X9            // b6 = a4 - a6
	MOVAPS X8, X4
	MOVAPS X9, X6

	// Group 2 (twiddle -i): (X5, X7)
	// t = a7 * (-i)
	MOVAPS X7, X10
	SHUFPS $0xB1, X10, X10
	XORPS  X11, X10          // X11 has maskNegHiPS

	MOVAPS X5, X8
	ADDPS  X10, X8           // b5 = a5 + t
	MOVAPS X5, X9
	SUBPS  X10, X9           // b7 = a5 - t
	MOVAPS X8, X5
	MOVAPS X9, X7

	// ==================================================================
	// Stage 3: 4 Radix-2 butterflies, stride 4
	// Butterflies on (X0, X4), (X1, X5), (X2, X6), (X3, X7)
	// Twiddles: 1, w1, w2, w3
	// ==================================================================

	// Butterfly 1 (twiddle 1): (X0, X4)
	MOVAPS X0, X8
	ADDPS  X4, X8            // y0 = b0 + b4
	MOVAPS X0, X9
	SUBPS  X4, X9            // y4 = b0 - b4
	MOVAPS X8, X0
	MOVAPS X9, X4

	// Butterfly 2 (twiddle w1): (X1, X5)
	// t = w1 * b5
	MOVSD  8(R10), X10       // w1
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11   // w1.re
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12   // w1.im
	MOVAPS X5, X13
	SHUFPS $0xB1, X13, X13   // b5 swapped
	MOVAPS X5, X14
	MULPS  X11, X14          // b5.re*w1.re, b5.im*w1.re
	MULPS  X12, X13          // b5.im*w1.im, b5.re*w1.im
	ADDSUBPS X13, X14        // t = b5 * w1

	MOVAPS X1, X8
	ADDPS  X14, X8           // y1 = b1 + t
	MOVAPS X1, X9
	SUBPS  X14, X9           // y5 = b1 - t
	MOVAPS X8, X1
	MOVAPS X9, X5

	// Butterfly 3 (twiddle w2): (X2, X6)
	// t = w2 * b6
	MOVSD  16(R10), X10      // w2
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11   // w2.re
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12   // w2.im
	MOVAPS X6, X13
	SHUFPS $0xB1, X13, X13   // b6 swapped
	MOVAPS X6, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14        // t = b6 * w2

	MOVAPS X2, X8
	ADDPS  X14, X8           // y2 = b2 + t
	MOVAPS X2, X9
	SUBPS  X14, X9           // y6 = b2 - t
	MOVAPS X8, X2
	MOVAPS X9, X6

	// Butterfly 4 (twiddle w3): (X3, X7)
	// t = w3 * b7
	MOVSD  24(R10), X10      // w3
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11   // w3.re
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12   // w3.im
	MOVAPS X7, X13
	SHUFPS $0xB1, X13, X13   // b7 swapped
	MOVAPS X7, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14        // t = b7 * w3

	MOVAPS X3, X8
	ADDPS  X14, X8           // y3 = b3 + t
	MOVAPS X3, X9
	SUBPS  X14, X9           // y7 = b3 - t
	MOVAPS X8, X3
	MOVAPS X9, X7

	// ==================================================================
	// Store results
	// ==================================================================
	MOVSD X0, (R8)
	MOVSD X1, 8(R8)
	MOVSD X2, 16(R8)
	MOVSD X3, 24(R8)
	MOVSD X4, 32(R8)
	MOVSD X5, 40(R8)
	MOVSD X6, 48(R8)
	MOVSD X7, 56(R8)

	// Copy to dst if needed
	CMPQ R8, R14
	JE   size8_r2_sse2_fwd_done

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

size8_r2_sse2_fwd_done:
	MOVB $1, ret+96(FP)
	RET

size8_r2_sse2_fwd_return_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 8, complex64, radix-2 variant
// ===========================================================================
TEXT ·InverseSSE3Size8Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_r2_sse2_inv_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_r2_sse2_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_r2_sse2_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_r2_sse2_inv_return_false

	MOVQ $8, AX
	CMPQ AX, $8
	JL   size8_r2_sse2_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_r2_sse2_inv_use_dst
	MOVQ R11, R8

size8_r2_sse2_inv_use_dst:
	// ==================================================================
	// Bit-reversal load: pattern [0, 4, 2, 6, 1, 5, 3, 7] * 8 bytes
	// ==================================================================
	MOVSD 0(R9), X0     // src[0]
	MOVSD 32(R9), X1    // src[4]
	MOVSD 16(R9), X2    // src[2]
	MOVSD 48(R9), X3    // src[6]
	MOVSD 8(R9), X4     // src[1]
	MOVSD 40(R9), X5    // src[5]
	MOVSD 24(R9), X6    // src[3]
	MOVSD 56(R9), X7    // src[7]

	// ==================================================================
	// Stage 1: 4 Radix-2 butterflies, stride 1, no twiddles
	// ==================================================================
	MOVAPS X0, X8
	ADDPS  X1, X8
	MOVAPS X0, X9
	SUBPS  X1, X9
	MOVAPS X8, X0
	MOVAPS X9, X1

	MOVAPS X2, X8
	ADDPS  X3, X8
	MOVAPS X2, X9
	SUBPS  X3, X9
	MOVAPS X8, X2
	MOVAPS X9, X3

	MOVAPS X4, X8
	ADDPS  X5, X8
	MOVAPS X4, X9
	SUBPS  X5, X9
	MOVAPS X8, X4
	MOVAPS X9, X5

	MOVAPS X6, X8
	ADDPS  X7, X8
	MOVAPS X6, X9
	SUBPS  X7, X9
	MOVAPS X8, X6
	MOVAPS X9, X7

	// ==================================================================
	// Stage 2: 4 Radix-2 butterflies, stride 2
	// Twiddles: 1, i
	// ==================================================================

	// Group 1 (twiddle 1): (X0, X2)
	MOVAPS X0, X8
	ADDPS  X2, X8
	MOVAPS X0, X9
	SUBPS  X2, X9
	MOVAPS X8, X0
	MOVAPS X9, X2

	// Group 1 (twiddle i): (X1, X3)
	// t = a3 * (i) = (-im, re)
	MOVAPS X3, X10
	SHUFPS $0xB1, X10, X10
	MOVUPS ·maskNegLoPS(SB), X11
	XORPS  X11, X10          // negate low: (-im, re)

	MOVAPS X1, X8
	ADDPS  X10, X8
	MOVAPS X1, X9
	SUBPS  X10, X9
	MOVAPS X8, X1
	MOVAPS X9, X3

	// Group 2 (twiddle 1): (X4, X6)
	MOVAPS X4, X8
	ADDPS  X6, X8
	MOVAPS X4, X9
	SUBPS  X6, X9
	MOVAPS X8, X4
	MOVAPS X9, X6

	// Group 2 (twiddle i): (X5, X7)
	// t = a7 * (i)
	MOVAPS X7, X10
	SHUFPS $0xB1, X10, X10
	XORPS  X11, X10          // X11 has maskNegLoPS

	MOVAPS X5, X8
	ADDPS  X10, X8
	MOVAPS X5, X9
	SUBPS  X10, X9
	MOVAPS X8, X5
	MOVAPS X9, X7

	// ==================================================================
	// Stage 3: 4 Radix-2 butterflies, stride 4
	// Twiddles: 1, conj(w1), conj(w2), conj(w3)
	// ==================================================================

	// Butterfly 1 (twiddle 1): (X0, X4)
	MOVAPS X0, X8
	ADDPS  X4, X8
	MOVAPS X0, X9
	SUBPS  X4, X9
	MOVAPS X8, X0
	MOVAPS X9, X4

	// Load mask for conjugation (negate imag)
	MOVUPS ·maskNegHiPS(SB), X15

	// Butterfly 2 (twiddle conj(w1)): (X1, X5)
	MOVSD  8(R10), X10
	XORPS  X15, X10          // conjugate
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X5, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X5, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14        // t = b5 * conj(w1)

	MOVAPS X1, X8
	ADDPS  X14, X8
	MOVAPS X1, X9
	SUBPS  X14, X9
	MOVAPS X8, X1
	MOVAPS X9, X5

	// Butterfly 3 (twiddle conj(w2)): (X2, X6)
	MOVSD  16(R10), X10
	XORPS  X15, X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X6, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X6, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14

	MOVAPS X2, X8
	ADDPS  X14, X8
	MOVAPS X2, X9
	SUBPS  X14, X9
	MOVAPS X8, X2
	MOVAPS X9, X6

	// Butterfly 4 (twiddle conj(w3)): (X3, X7)
	MOVSD  24(R10), X10
	XORPS  X15, X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X7, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X7, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14

	MOVAPS X3, X8
	ADDPS  X14, X8
	MOVAPS X3, X9
	SUBPS  X14, X9
	MOVAPS X8, X3
	MOVAPS X9, X7

	// ==================================================================
	// Apply 1/8 scaling
	// ==================================================================
	MOVSS  ·eighth32(SB), X15
	SHUFPS $0x00, X15, X15   // broadcast 0.125
	MULPS  X15, X0
	MULPS  X15, X1           // y1
	MULPS  X15, X2           // y2
	MULPS  X15, X3           // y3
	MULPS  X15, X4           // y4
	MULPS  X15, X5           // y5
	MULPS  X15, X6           // y6
	MULPS  X15, X7           // y7

	// ==================================================================
	// Store results
	// ==================================================================
	MOVSD X0, (R8)
	MOVSD X1, 8(R8)
	MOVSD X2, 16(R8)
	MOVSD X3, 24(R8)
	MOVSD X4, 32(R8)
	MOVSD X5, 40(R8)
	MOVSD X6, 48(R8)
	MOVSD X7, 56(R8)

	// Copy to dst if needed
	CMPQ R8, R14
	JE   size8_r2_sse2_inv_done

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

size8_r2_sse2_inv_done:
	MOVB $1, ret+96(FP)
	RET

size8_r2_sse2_inv_return_false:
	MOVB $0, ret+96(FP)
	RET

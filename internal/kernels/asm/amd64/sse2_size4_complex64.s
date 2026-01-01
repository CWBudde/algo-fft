//go:build amd64 && fft_asm && !purego

// ===========================================================================
// SSE2 Size-4 FFT Kernels for AMD64 (complex64)
// ===========================================================================
//
// Fully-unrolled radix-4 FFT kernel for size 4.
//
// Radix-4 Butterfly:
//   t0 = x0 + x2
//   t1 = x0 - x2
//   t2 = x1 + x3
//   t3 = x1 - x3
//
//   y0 = t0 + t2
//   y1 = t1 + t3*(-i)
//   y2 = t0 - t2
//   y3 = t1 - t3*(-i)
//
// Inverse uses +i and applies 1/4 scaling.
//
// ===========================================================================

#include "textflag.h"

DATA ·sse2ScaleQuarterPS+0(SB)/4, $0x3e800000 // 0.25f
GLOBL ·sse2ScaleQuarterPS(SB), RODATA|NOPTR, $4

// ===========================================================================
// Forward transform, size 4, complex64, radix-4
// ===========================================================================
TEXT ·forwardSSE2Size4Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ R8, R14             // R14 = original dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // unused
	MOVQ scratch+72(FP), R11 // scratch pointer
	MOVQ bitrev+96(FP), R12  // unused
	MOVQ src+32(FP), R13     // R13 = n (should be 4)

	// Verify n == 4
	CMPQ R13, $4
	JNE  size4_sse2_64_fwd_return_false

	// Validate all slice lengths >= 4
	MOVQ dst+8(FP), AX
	CMPQ AX, $4
	JL   size4_sse2_64_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $4
	JL   size4_sse2_64_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $4
	JL   size4_sse2_64_fwd_return_false

	// Load x0..x3 (complex64)
	MOVSD (R9), X0
	MOVSD 8(R9), X1
	MOVSD 16(R9), X2
	MOVSD 24(R9), X3

	// t0 = x0 + x2, t1 = x0 - x2
	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	// t2 = x1 + x3, t3 = x1 - x3
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	// t3NegI = swap(t3) with sign toggle on high lane -> (im, -re)
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	MOVUPS ·sse2MaskNegHiPS(SB), X9
	XORPS X9, X8

	// y0, y2
	MOVAPS X4, X9
	ADDPS X6, X9
	MOVAPS X4, X10
	SUBPS X6, X10
	// y1, y3
	MOVAPS X5, X11
	ADDPS X8, X11
	MOVAPS X5, X12
	SUBPS X8, X12

	// Store results
	MOVSD X9, (R14)
	MOVSD X11, 8(R14)
	MOVSD X10, 16(R14)
	MOVSD X12, 24(R14)

	MOVB $1, ret+120(FP)
	RET

size4_sse2_64_fwd_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 4, complex64, radix-4
// ===========================================================================
TEXT ·inverseSSE2Size4Radix4Complex64Asm(SB), NOSPLIT, $0-121
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
	JNE  size4_sse2_64_inv_return_false

	// Validate all slice lengths >= 4
	MOVQ dst+8(FP), AX
	CMPQ AX, $4
	JL   size4_sse2_64_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $4
	JL   size4_sse2_64_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $4
	JL   size4_sse2_64_inv_return_false

	// Load x0..x3
	MOVSD (R9), X0
	MOVSD 8(R9), X1
	MOVSD 16(R9), X2
	MOVSD 24(R9), X3

	// t0 = x0 + x2, t1 = x0 - x2
	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	// t2 = x1 + x3, t3 = x1 - x3
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	// t3PosI = swap(t3) with sign toggle on low lane -> (-im, re)
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	MOVUPS ·sse2MaskNegLoPS(SB), X9
	XORPS X9, X8

	// y0, y2
	MOVAPS X4, X9
	ADDPS X6, X9
	MOVAPS X4, X10
	SUBPS X6, X10
	// y1, y3
	MOVAPS X5, X11
	ADDPS X8, X11
	MOVAPS X5, X12
	SUBPS X8, X12

	// Scale by 1/4
	XORPS X15, X15
	MOVSS ·sse2ScaleQuarterPS(SB), X15
	SHUFPS $0, X15, X15
	MULPS X15, X9
	MULPS X15, X11
	MULPS X15, X10
	MULPS X15, X12

	// Store results
	MOVSD X9, (R14)
	MOVSD X11, 8(R14)
	MOVSD X10, 16(R14)
	MOVSD X12, 24(R14)

	MOVB $1, ret+120(FP)
	RET

size4_sse2_64_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

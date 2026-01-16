//go:build amd64 && asm && !purego

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

// ===========================================================================
// Forward transform, size 4, complex64, radix-4
// ===========================================================================
TEXT ·ForwardSSE2Size4Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ R8, R14             // R14 = original dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // unused
	MOVQ scratch+72(FP), R11 // scratch pointer
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

	// Load packed pairs to keep all lanes initialized for ADDPS/SUBPS.
	// X0=[x0,x1], X1=[x2,x3]
	MOVUPS 0(R9), X0
	MOVUPS 16(R9), X1

	// T02 = [t0,t2] = [x0+x2, x1+x3]
	MOVAPS X0, X2
	ADDPS X1, X2
	// T13 = [t1,t3] = [x0-x2, x1-x3]
	MOVAPS X0, X3
	SUBPS X1, X3

	// y0, y2 from T02
	MOVAPS X2, X4
	SHUFPS $0x4E, X4, X4
	MOVAPS X2, X9
	ADDPS X4, X9
	MOVAPS X2, X10
	SUBPS X4, X10

	// t3NegI from t3: (re,im)->(im,-re)
	MOVAPS X3, X7
	MOVHLPS X3, X7
	SHUFPS $0xB1, X7, X7
	XORPS ·maskNegHiPS(SB), X7

	// Pack [t1, t3NegI]
	MOVAPS X3, X8
	MOVLHPS X7, X8

	// y1, y3 from packed pair
	MOVAPS X8, X11
	SHUFPS $0x4E, X11, X11
	MOVAPS X8, X12
	ADDPS X11, X12
	MOVAPS X8, X13
	SUBPS X11, X13

	// Store results
	MOVSD X9, (R14)
	MOVSD X12, 8(R14)
	MOVSD X10, 16(R14)
	MOVSD X13, 24(R14)

	MOVB $1, ret+96(FP)
	RET

size4_sse2_64_fwd_return_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 4, complex64, radix-4
// ===========================================================================
TEXT ·InverseSSE2Size4Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14             // R14 = original dst pointer
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10 // unused
	MOVQ scratch+72(FP), R11
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

	// Load packed pairs to keep all lanes initialized for ADDPS/SUBPS.
	// X0=[x0,x1], X1=[x2,x3]
	MOVUPS 0(R9), X0
	MOVUPS 16(R9), X1

	// T02 = [t0,t2] = [x0+x2, x1+x3]
	MOVAPS X0, X2
	ADDPS X1, X2
	// T13 = [t1,t3] = [x0-x2, x1-x3]
	MOVAPS X0, X3
	SUBPS X1, X3

	// y0, y2 from T02
	MOVAPS X2, X4
	SHUFPS $0x4E, X4, X4
	MOVAPS X2, X9
	ADDPS X4, X9
	MOVAPS X2, X10
	SUBPS X4, X10

	// t3PosI from t3: (re,im)->(-im,re)
	MOVAPS X3, X7
	MOVHLPS X3, X7
	SHUFPS $0xB1, X7, X7
	XORPS ·maskNegLoPS(SB), X7

	// Pack [t1, t3PosI]
	MOVAPS X3, X8
	MOVLHPS X7, X8

	// y1, y3 from packed pair
	MOVAPS X8, X11
	SHUFPS $0x4E, X11, X11
	MOVAPS X8, X12
	ADDPS X11, X12
	MOVAPS X8, X13
	SUBPS X11, X13

	// Scale by 1/4
	XORPS X15, X15
	MOVSS ·quarter32(SB), X15
	SHUFPS $0, X15, X15
	MULPS X15, X9
	MULPS X15, X12
	MULPS X15, X10
	MULPS X15, X13

	// Store results
	MOVSD X9, (R14)
	MOVSD X12, 8(R14)
	MOVSD X10, 16(R14)
	MOVSD X13, 24(R14)

	MOVB $1, ret+96(FP)
	RET

size4_sse2_64_inv_return_false:
	MOVB $0, ret+96(FP)
	RET

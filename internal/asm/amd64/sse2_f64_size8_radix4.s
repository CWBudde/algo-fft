//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-8 Radix-4 FFT Kernels for AMD64 (complex128)
// ===========================================================================

#include "textflag.h"

// Forward transform, size 8, complex128, radix-4
TEXT ·ForwardSSE2Size8Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	CMPQ R13, $8
	JNE  size8_r4_fwd_err

	CMPQ R8, R9
	JNE  size8_r4_fwd_use_dst
	MOVQ R11, R8

size8_r4_fwd_use_dst:
	// Bit-reversal load: pattern [0, 2, 4, 6, 1, 3, 5, 7] * 16 bytes
	MOVUPD 0(R9), X0    // src[0]
	MOVUPD 32(R9), X1   // src[2]
	MOVUPD 64(R9), X2   // src[4]
	MOVUPD 96(R9), X3   // src[6]
	MOVUPD 16(R9), X4   // src[1]
	MOVUPD 48(R9), X5   // src[3]
	MOVUPD 80(R9), X6   // src[5]
	MOVUPD 112(R9), X7  // src[7]

	// Stage 1: Two Radix-4 butterflies
	MOVUPS ·maskNegHiPD(SB), X15
	
	// Butterfly 1: [x0, x1, x2, x3]
	MOVAPD X0, X8; ADDPD X2, X8; MOVAPD X0, X9; SUBPD X2, X9 // t0, t1
	MOVAPD X1, X10; ADDPD X3, X10; MOVAPD X1, X11; SUBPD X3, X11 // t2, t3
	MOVAPD X11, X12; SHUFPD $1, X12, X12; XORPD X15, X12 // t3 * -i
	MOVAPD X8, X0; ADDPD X10, X0 // a0
	MOVAPD X9, X1; ADDPD X12, X1 // a1
	MOVAPD X8, X2; SUBPD X10, X2 // a2
	MOVAPD X9, X3; SUBPD X12, X3 // a3

	// Butterfly 2: [x4, x5, x6, x7]
	MOVAPD X4, X8; ADDPD X6, X8; MOVAPD X4, X9; SUBPD X6, X9 // t0, t1
	MOVAPD X5, X10; ADDPD X7, X10; MOVAPD X5, X11; SUBPD X7, X11 // t2, t3
	MOVAPD X11, X12; SHUFPD $1, X12, X12; XORPD X15, X12
	MOVAPD X8, X4; ADDPD X10, X4 // a4
	MOVAPD X9, X5; ADDPD X12, X5 // a5
	MOVAPD X8, X6; SUBPD X10, X6 // a6
	MOVAPD X9, X7; SUBPD X12, X7 // a7

	// Stage 2: Radix-2 combine with twiddles
	// y0, y4
	MOVAPD X0, X8; ADDPD X4, X8; MOVAPD X0, X9; SUBPD X4, X9
	MOVUPD X8, (R8)
	MOVUPD X9, 64(R8)

	// y1, y5 (w1 * a5)
	MOVUPD 16(R10), X10 // w1
	MOVAPD X5, X11; UNPCKLPD X11, X11; MULPD X10, X11
	MOVAPD X5, X12; UNPCKHPD X12, X12; MOVAPD X10, X13; SHUFPD $1, X13, X13; MULPD X12, X13
	XORPD ·maskNegLoPD(SB), X13; ADDPD X13, X11 // t1
	MOVAPD X1, X12; ADDPD X11, X1; SUBPD X11, X12
	MOVUPD X1, 16(R8)
	MOVUPD X12, 80(R8)

	// y2, y6 (w2 * a6)
	MOVUPD 32(R10), X10 // w2
	MOVAPD X6, X11; UNPCKLPD X11, X11; MULPD X10, X11
	MOVAPD X6, X12; UNPCKHPD X12, X12; MOVAPD X10, X13; SHUFPD $1, X13, X13; MULPD X12, X13
	XORPD ·maskNegLoPD(SB), X13; ADDPD X13, X11 // t2
	MOVAPD X2, X12; ADDPD X11, X2; SUBPD X11, X12
	MOVUPD X2, 32(R8)
	MOVUPD X12, 96(R8)

	// y3, y7 (w3 * a7)
	MOVUPD 48(R10), X10 // w3
	MOVAPD X7, X11; UNPCKLPD X11, X11; MULPD X10, X11
	MOVAPD X7, X12; UNPCKHPD X12, X12; MOVAPD X10, X13; SHUFPD $1, X13, X13; MULPD X12, X13
	XORPD ·maskNegLoPD(SB), X13; ADDPD X13, X11 // t3
	MOVAPD X3, X12; ADDPD X11, X3; SUBPD X11, X12
	MOVUPD X3, 48(R8)
	MOVUPD X12, 112(R8)

	// Copy to R14 if needed
	CMPQ R8, R14
	JE size8_r4_fwd_done
	MOVQ $8, CX; MOVQ R8, SI; MOVQ R14, DI
size8_r4_fwd_copy:
	MOVUPD (SI), X0; MOVUPD X0, (DI); ADDQ $16, SI; ADDQ $16, DI; DECQ CX; JNZ size8_r4_fwd_copy

size8_r4_fwd_done:
	MOVB $1, ret+96(FP)
	RET
size8_r4_fwd_err:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 8, complex128, radix-4
TEXT ·InverseSSE2Size8Radix4Complex128Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	CMPQ R13, $8
	JNE  size8_r4_inv_err

	CMPQ R8, R9
	JNE  size8_r4_inv_use_dst
	MOVQ R11, R8

size8_r4_inv_use_dst:
	// Bit-reversal load: pattern [0, 2, 4, 6, 1, 3, 5, 7] * 16 bytes
	MOVUPD 0(R9), X0    // src[0]
	MOVUPD 32(R9), X1   // src[2]
	MOVUPD 64(R9), X2   // src[4]
	MOVUPD 96(R9), X3   // src[6]
	MOVUPD 16(R9), X4   // src[1]
	MOVUPD 48(R9), X5   // src[3]
	MOVUPD 80(R9), X6   // src[5]
	MOVUPD 112(R9), X7  // src[7]

	// Stage 1: Two Radix-4 butterflies (+i)
	MOVUPS ·maskNegLoPD(SB), X14
	
	// Butterfly 1
	MOVAPD X0, X8; ADDPD X2, X8; MOVAPD X0, X9; SUBPD X2, X9 // t0, t1
	MOVAPD X1, X10; ADDPD X3, X10; MOVAPD X1, X11; SUBPD X3, X11 // t2, t3
	MOVAPD X11, X12; SHUFPD $1, X12, X12; XORPD X14, X12 // t3 * i
	MOVAPD X8, X0; ADDPD X10, X0 // a0
	MOVAPD X9, X1; ADDPD X12, X1 // a1
	MOVAPD X8, X2; SUBPD X10, X2 // a2
	MOVAPD X9, X3; SUBPD X12, X3 // a3

	// Butterfly 2
	MOVAPD X4, X8; ADDPD X6, X8; MOVAPD X4, X9; SUBPD X6, X9 // t0, t1
	MOVAPD X5, X10; ADDPD X7, X10; MOVAPD X5, X11; SUBPD X7, X11 // t2, t3
	MOVAPD X11, X12; SHUFPD $1, X12, X12; XORPD X14, X12
	MOVAPD X8, X4; ADDPD X10, X4 // a4
	MOVAPD X9, X5; ADDPD X12, X5 // a5
	MOVAPD X8, X6; SUBPD X10, X6 // a6
	MOVAPD X9, X7; SUBPD X12, X7 // a7

	// Stage 2: Radix-2 combine
	MOVUPS ·maskNegHiPD(SB), X15
	
	// y0, y4
	MOVAPD X0, X8; ADDPD X4, X8; MOVAPD X0, X9; SUBPD X4, X9
	MOVUPD X8, 0(R11)
	MOVUPD X9, 64(R11)

	// conj(w1) * a5
	MOVUPD 16(R10), X10; XORPD X15, X10
	MOVAPD X5, X11; UNPCKLPD X11, X11; MULPD X10, X11
	MOVAPD X5, X12; UNPCKHPD X12, X12; MOVAPD X10, X13; SHUFPD $1, X13, X13; MULPD X12, X13
	XORPD ·maskNegLoPD(SB), X13; ADDPD X13, X11
	MOVAPD X1, X12; ADDPD X11, X1; SUBPD X11, X12
	MOVUPD X1, 16(R11)
	MOVUPD X12, 80(R11)

	// conj(w2) * a6
	MOVUPD 32(R10), X10; XORPD X15, X10
	MOVAPD X6, X11; UNPCKLPD X11, X11; MULPD X10, X11
	MOVAPD X6, X12; UNPCKHPD X12, X12; MOVAPD X10, X13; SHUFPD $1, X13, X13; MULPD X12, X13
	XORPD ·maskNegLoPD(SB), X13; ADDPD X13, X11
	MOVAPD X2, X12; ADDPD X11, X2; SUBPD X11, X12
	MOVUPD X2, 32(R11)
	MOVUPD X12, 96(R11)

	// conj(w3) * a7
	MOVUPD 48(R10), X10; XORPD X15, X10
	MOVAPD X7, X11; UNPCKLPD X11, X11; MULPD X10, X11
	MOVAPD X7, X12; UNPCKHPD X12, X12; MOVAPD X10, X13; SHUFPD $1, X13, X13; MULPD X12, X13
	XORPD ·maskNegLoPD(SB), X13; ADDPD X13, X11
	MOVAPD X3, X12; ADDPD X11, X3; SUBPD X11, X12
	MOVUPD X3, 48(R11)
	MOVUPD X12, 112(R11)

	// Scale by 1/8 and Store
	MOVSD ·eighth64(SB), X15; SHUFPD $0, X15, X15
	MOVQ $8, CX; MOVQ R11, SI; MOVQ R14, DI
size8_r4_inv_scale:
	MOVUPD (SI), X0; MULPD X15, X0; MOVUPD X0, (DI); ADDQ $16, SI; ADDQ $16, DI; DECQ CX; JNZ size8_r4_inv_scale

	MOVB $1, ret+96(FP)
	RET
size8_r4_inv_err:
	MOVB $0, ret+96(FP)
	RET

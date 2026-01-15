//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-8 Radix-2 FFT Kernels for AMD64 (complex128)
// ===========================================================================

#include "textflag.h"

// Forward transform, size 8, complex128, radix-2
TEXT ·ForwardSSE2Size8Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	CMPQ R13, $8
	JNE  fwd_err

	CMPQ R8, R9
	JNE  fwd_use_dst
	MOVQ R11, R8

fwd_use_dst:
	// Bit-reversal load: pattern [0, 4, 2, 6, 1, 5, 3, 7] * 16 bytes
	MOVUPD 0(R9), X0    // src[0]
	MOVUPD 64(R9), X1   // src[4]
	MOVUPD 32(R9), X2   // src[2]
	MOVUPD 96(R9), X3   // src[6]
	MOVUPD 16(R9), X4   // src[1]
	MOVUPD 80(R9), X5   // src[5]
	MOVUPD 48(R9), X6   // src[3]
	MOVUPD 112(R9), X7  // src[7]

	// Stage 1: (0,1), (2,3), (4,5), (6,7) - w=1
	MOVAPD X0, X8; ADDPD X1, X0; SUBPD X1, X8 // X0=W0, X8=W1
	MOVAPD X2, X9; ADDPD X3, X2; SUBPD X3, X9 // X2=W2, X9=W3
	MOVAPD X4, X10; ADDPD X5, X4; SUBPD X5, X10 // X4=W4, X10=W5
	MOVAPD X6, X11; ADDPD X7, X6; SUBPD X7, X11 // X6=W6, X11=W7
	
	// Stage 2: (0,2), (1,3), (4,6), (5,7)
	MOVUPS ·maskNegHiPD(SB), X15
	// j=0 (w=1)
	MOVAPD X0, X1; ADDPD X2, X0; SUBPD X2, X1 // X0=y0, X1=y2
	MOVAPD X4, X5; ADDPD X6, X4; SUBPD X6, X5 // X4=y4, X5=y6
	// j=1 (w=-i)
	// y1, y3
	MOVAPD X9, X2; SHUFPD $1, X2, X2; XORPD X15, X2 // t = W3 * -i
	MOVAPD X8, X3; ADDPD X2, X8; SUBPD X2, X3 // X8=y1, X3=y3
	// y5, y7
	MOVAPD X11, X2; SHUFPD $1, X2, X2; XORPD X15, X2 // t = W7 * -i
	MOVAPD X10, X7; ADDPD X2, X10; SUBPD X2, X7 // X10=y5, X7=y7
	
	// Element locations: X0=y0, X8=y1, X1=y2, X3=y3, X4=y4, X10=y5, X5=y6, X7=y7
	
	// Stage 3: (y0, y4), (y1, y5), (y2, y6), (y3, y7)
	// z0, z4
	MOVAPD X0, X2; ADDPD X4, X0; SUBPD X4, X2
	MOVUPD X0, (R14)
	MOVUPD X2, 64(R14)
	
	// z1, z5
	MOVUPD 16(R10), X4 // w1
	MOVAPD X10, X6; UNPCKLPD X6, X6; MULPD X4, X6
	MOVAPD X10, X9; UNPCKHPD X9, X9; MOVAPD X4, X12; SHUFPD $1, X12, X12; MULPD X9, X12
	XORPD ·maskNegLoPD(SB), X12; ADDPD X12, X6 // t = y5 * w1
	MOVAPD X8, X12; ADDPD X6, X8; SUBPD X6, X12
	MOVUPD X8, 16(R14)
	MOVUPD X12, 80(R14)
	
	// z2, z6
	MOVAPD X5, X6; SHUFPD $1, X6, X6; XORPD X15, X6 // t = y6 * -i
	MOVAPD X1, X12; ADDPD X6, X1; SUBPD X6, X12
	MOVUPD X1, 32(R14)
	MOVUPD X12, 96(R14)
	
	// z3, z7
	MOVUPD 48(R10), X4 // w3
	MOVAPD X7, X6; UNPCKLPD X6, X6; MULPD X4, X6
	MOVAPD X7, X9; UNPCKHPD X9, X9; MOVAPD X4, X12; SHUFPD $1, X12, X12; MULPD X9, X12
	XORPD ·maskNegLoPD(SB), X12; ADDPD X12, X6 // t = y7 * w3
	MOVAPD X3, X12; ADDPD X6, X3; SUBPD X6, X12
	MOVUPD X3, 48(R14)
	MOVUPD X12, 112(R14)

	MOVB $1, ret+96(FP)
	RET
fwd_err:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 8, complex128, radix-2
TEXT ·InverseSSE2Size8Radix2Complex128Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	CMPQ R13, $8
	JNE  inv_err

	CMPQ R8, R9
	JNE  inv_use_dst
	MOVQ R11, R8

inv_use_dst:
	// Bit-reversal load: pattern [0, 4, 2, 6, 1, 5, 3, 7] * 16 bytes
	MOVUPD 0(R9), X0    // src[0]
	MOVUPD 64(R9), X1   // src[4]
	MOVUPD 32(R9), X2   // src[2]
	MOVUPD 96(R9), X3   // src[6]
	MOVUPD 16(R9), X4   // src[1]
	MOVUPD 80(R9), X5   // src[5]
	MOVUPD 48(R9), X6   // src[3]
	MOVUPD 112(R9), X7  // src[7]

	// Stage 1
	MOVAPD X0, X8; ADDPD X1, X0; SUBPD X1, X8 
	MOVAPD X2, X9; ADDPD X3, X2; SUBPD X3, X9 
	MOVAPD X4, X10; ADDPD X5, X4; SUBPD X5, X10 
	MOVAPD X6, X11; ADDPD X7, X6; SUBPD X7, X11
	
	// Stage 2
	MOVUPS ·maskNegLoPD(SB), X15 // for i
	// j=0
	MOVAPD X0, X1; ADDPD X2, X0; SUBPD X2, X1 // X0=y0, X1=y2
	MOVAPD X4, X5; ADDPD X6, X4; SUBPD X6, X5 // X4=y4, X5=y6
	// j=1 (w=i)
	MOVAPD X9, X2; SHUFPD $1, X2, X2; XORPD X15, X2 // t = W3 * i
	MOVAPD X8, X3; ADDPD X2, X8; SUBPD X2, X3 // X8=y1, X3=y3
	MOVAPD X11, X2; SHUFPD $1, X2, X2; XORPD X15, X2 // t = W7 * i
	MOVAPD X10, X7; ADDPD X2, X10; SUBPD X2, X7 // X10=y5, X7=y7
	
	MOVUPS ·maskNegHiPD(SB), X14 // for conj

	// Stage 3
	// z0, z4
	MOVAPD X0, X2; ADDPD X4, X0; SUBPD X4, X2
	MOVUPD X0, 0(R11)
	MOVUPD X2, 64(R11)
	
	// z1, z5
	MOVUPD 16(R10), X4; XORPD X14, X4 // conj(w1)
	MOVAPD X10, X6; UNPCKLPD X6, X6; MULPD X4, X6
	MOVAPD X10, X9; UNPCKHPD X9, X9; MOVAPD X4, X12; SHUFPD $1, X12, X12; MULPD X9, X12
	XORPD ·maskNegLoPD(SB), X12; ADDPD X12, X6 // t
	MOVAPD X8, X12; ADDPD X6, X8; SUBPD X6, X12
	MOVUPD X8, 16(R11)
	MOVUPD X12, 80(R11)
	
	// z2, z6 (w=i)
	MOVAPD X5, X6; SHUFPD $1, X6, X6; XORPD X15, X6 // t
	MOVAPD X1, X12; ADDPD X6, X1; SUBPD X6, X12
	MOVUPD X1, 32(R11)
	MOVUPD X12, 96(R11)
	
	// z3, z7
	MOVUPD 48(R10), X4; XORPD X14, X4 // conj(w3)
	MOVAPD X7, X6; UNPCKLPD X6, X6; MULPD X4, X6
	MOVAPD X7, X9; UNPCKHPD X9, X9; MOVAPD X4, X12; SHUFPD $1, X12, X12; MULPD X9, X12
	XORPD ·maskNegLoPD(SB), X12; ADDPD X12, X6 // t
	MOVAPD X3, X12; ADDPD X6, X3; SUBPD X6, X12
	MOVUPD X3, 48(R11)
	MOVUPD X12, 112(R11)

	// Scale and Store
	MOVSD ·eighth64(SB), X15; SHUFPD $0, X15, X15
	MOVQ $8, CX; MOVQ R11, SI; MOVQ R14, DI
scale_loop:
	MOVUPD (SI), X0; MULPD X15, X0; MOVUPD X0, (DI)
	ADDQ $16, SI; ADDQ $16, DI
	DECQ CX; JNZ scale_loop

	MOVB $1, ret+96(FP)
	RET
inv_err:
	MOVB $0, ret+96(FP)
	RET

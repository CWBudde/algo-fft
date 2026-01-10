//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-8 Radix-8 FFT Kernels for AMD64 (complex128)
// ===========================================================================

#include "textflag.h"

// Forward transform, size 8, complex128, radix-8
TEXT ·ForwardSSE2Size8Radix8Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size8Identity(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $8
	JNE  size8_r8_fwd_err

	// Note: bitrev parameter ignored for radix-8 on size-8 (identity permutation)

	// Load masks
	MOVUPS ·maskNegLoPD(SB), X14
	MOVUPS ·maskNegHiPD(SB), X15

	// Load input in natural order (complex128 = 16 bytes each)
	// Radix-8 on size-8 is a single butterfly with identity permutation
	MOVUPD 0(R9), X0    // x0
	MOVUPD 16(R9), X1   // x1
	MOVUPD 32(R9), X2   // x2
	MOVUPD 48(R9), X3   // x3
	MOVUPD 64(R9), X4   // x4
	MOVUPD 80(R9), X5   // x5
	MOVUPD 96(R9), X6   // x6
	MOVUPD 112(R9), X7  // x7

	// Stage 1: a0..a7
	MOVAPD X0, X8; ADDPD X4, X0; SUBPD X4, X8 // X0=a0, X8=a1
	MOVAPD X2, X9; ADDPD X6, X2; SUBPD X6, X9 // X2=a2, X9=a3
	MOVAPD X1, X10; ADDPD X5, X1; SUBPD X5, X10 // X1=a4, X10=a5
	MOVAPD X3, X11; ADDPD X7, X3; SUBPD X7, X11 // X3=a6, X11=a7

	// Stage 2: Even terms (e0..e3)
	MOVAPD X0, X4; ADDPD X2, X0; SUBPD X2, X4 // X0=e0, X4=e2
	// e1 = a1 + mulNegI(a3)
	MOVAPD X9, X5; SHUFPD $1, X5, X5; XORPD X15, X5 // t = a3 * -i
	MOVAPD X8, X6; ADDPD X5, X8; SUBPD X5, X6 // X8=e1, X6=e3 (Wait, e3 is X6)
	
	// Odd terms (o0..o3)
	MOVAPD X1, X5; ADDPD X3, X1; SUBPD X3, X5 // X1=o0, X5=o2
	// o1 = a5 + mulNegI(a7)
	MOVAPD X11, X7; SHUFPD $1, X7, X7; XORPD X15, X7 // t = a7 * -i
	MOVAPD X10, X9; ADDPD X7, X10; SUBPD X7, X9 // X10=o1, X9=o3

	// Element locations: X0=e0, X8=e1, X4=e2, X6=e3, X1=o0, X10=o1, X5=o2, X9=o3
	
	// Stage 3: out0..out7
	// out0, out4
	MOVAPD X0, X2; ADDPD X1, X0; SUBPD X1, X2
	MOVUPD X0, (R8)
	MOVUPD X2, 64(R8)
	
	// out1, out5 (t1 = w1 * o1)
	MOVUPD 16(R10), X0 // w1
	MOVAPD X10, X1; UNPCKLPD X1, X1; MULPD X0, X1
	MOVAPD X10, X3; UNPCKHPD X3, X3; MOVAPD X0, X7; SHUFPD $1, X7, X7; MULPD X3, X7
	XORPD X14, X7; ADDPD X7, X1 // t1
	MOVAPD X8, X3; ADDPD X1, X8; SUBPD X1, X3
	MOVUPD X8, 16(R8)
	MOVUPD X3, 80(R8)
	
	// out2, out6 (t2 = w2 * o2 = -i * o2)
	MOVAPD X5, X1; SHUFPD $1, X1, X1; XORPD X15, X1 // t2
	MOVAPD X4, X3; ADDPD X1, X4; SUBPD X1, X3
	MOVUPD X4, 32(R8)
	MOVUPD X3, 96(R8)
	
	// out3, out7 (t3 = w3 * o3)
	MOVUPD 48(R10), X0 // w3
	MOVAPD X9, X1; UNPCKLPD X1, X1; MULPD X0, X1
	MOVAPD X9, X3; UNPCKHPD X3, X3; MOVAPD X0, X7; SHUFPD $1, X7, X7; MULPD X3, X7
	XORPD X14, X7; ADDPD X7, X1 // t3
	MOVAPD X6, X3; ADDPD X1, X6; SUBPD X1, X3
	MOVUPD X6, 48(R8)
	MOVUPD X3, 112(R8)

	MOVB $1, ret+96(FP)
	RET
size8_r8_fwd_err:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 8, complex128, radix-8
TEXT ·InverseSSE2Size8Radix8Complex128Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size8Identity(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $8
	JNE  size8_r8_inv_err

	// Note: bitrev parameter ignored for radix-8 on size-8 (identity permutation)

	MOVUPS ·maskNegLoPD(SB), X14
	MOVUPS ·maskNegHiPD(SB), X15

	// Load input in natural order (complex128 = 16 bytes each)
	// Radix-8 on size-8 is a single butterfly with identity permutation
	MOVUPD 0(R9), X0    // x0
	MOVUPD 16(R9), X1   // x1
	MOVUPD 32(R9), X2   // x2
	MOVUPD 48(R9), X3   // x3
	MOVUPD 64(R9), X4   // x4
	MOVUPD 80(R9), X5   // x5
	MOVUPD 96(R9), X6   // x6
	MOVUPD 112(R9), X7  // x7

	// Stage 1
	MOVAPD X0, X8; ADDPD X4, X0; SUBPD X4, X8 
	MOVAPD X2, X9; ADDPD X6, X2; SUBPD X6, X9 
	MOVAPD X1, X10; ADDPD X5, X1; SUBPD X5, X10 
	MOVAPD X3, X11; ADDPD X7, X3; SUBPD X7, X11 

	// Stage 2: Even
	MOVAPD X0, X4; ADDPD X2, X0; SUBPD X2, X4 // X0=e0, X4=e2
	// e1 = a1 + mulI(a3)
	MOVAPD X9, X5; SHUFPD $1, X5, X5; XORPD X14, X5 // t = a3 * i
	MOVAPD X8, X6; ADDPD X5, X8; SUBPD X5, X6 // X8=e1, X6=e3
	
	// Stage 2: Odd
	MOVAPD X1, X5; ADDPD X3, X1; SUBPD X3, X5 // X1=o0, X5=o2
	// o1 = a5 + mulI(a7)
	MOVAPD X11, X7; SHUFPD $1, X7, X7; XORPD X14, X7 // t = a7 * i
	MOVAPD X10, X9; ADDPD X7, X10; SUBPD X7, X9 // X10=o1, X9=o3

	// Stage 3
	// out0, out4
	MOVAPD X0, X2; ADDPD X1, X0; SUBPD X1, X2
	MOVUPD X0, 0(R11)
	MOVUPD X2, 64(R11)
	
	// out1, out5 (t1 = conj(w1) * o1)
	MOVUPD 16(R10), X0; XORPD X15, X0 // conj(w1)
	MOVAPD X10, X1; UNPCKLPD X1, X1; MULPD X0, X1
	MOVAPD X10, X3; UNPCKHPD X3, X3; MOVAPD X0, X7; SHUFPD $1, X7, X7; MULPD X3, X7
	XORPD X14, X7; ADDPD X7, X1 // t1
	MOVAPD X8, X3; ADDPD X1, X8; SUBPD X1, X3
	MOVUPD X8, 16(R11)
	MOVUPD X3, 80(R11)
	
	// out2, out6 (t2 = conj(w2) * o2 = i * o2)
	MOVAPD X5, X1; SHUFPD $1, X1, X1; XORPD X14, X1 // t2
	MOVAPD X4, X3; ADDPD X1, X4; SUBPD X1, X3
	MOVUPD X4, 32(R11)
	MOVUPD X3, 96(R11)
	
	// out3, out7 (t3 = conj(w3) * o3)
	MOVUPD 48(R10), X0; XORPD X15, X0 // conj(w3)
	MOVAPD X9, X1; UNPCKLPD X1, X1; MULPD X0, X1
	MOVAPD X9, X3; UNPCKHPD X3, X3; MOVAPD X0, X7; SHUFPD $1, X7, X7; MULPD X3, X7
	XORPD X14, X7; ADDPD X7, X1 // t3
	MOVAPD X6, X3; ADDPD X1, X6; SUBPD X1, X3
	MOVUPD X6, 48(R11)
	MOVUPD X3, 112(R11)

	// Scale by 1/8 and Store
	MOVSD ·eighth64(SB), X15; SHUFPD $0, X15, X15
	MOVQ $8, CX; MOVQ R11, SI; MOVQ R8, DI
size8_r8_inv_scale_loop:
	MOVUPD (SI), X0; MULPD X15, X0; MOVUPD X0, (DI)
	ADDQ $16, SI; ADDQ $16, DI
	DECQ CX; JNZ size8_r8_inv_scale_loop

	MOVB $1, ret+96(FP)
	RET
size8_r8_inv_err:
	MOVB $0, ret+96(FP)
	RET

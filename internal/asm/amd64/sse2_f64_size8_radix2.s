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
	MOVAPD X0, X8   // Copy X0 to X8
  ADDPD X1, X0    // X0 = X0 + X1
  SUBPD X1, X8    // X8 = X8 - X1 → X0=W0, X8=W1
	MOVAPD X2, X9   // Copy X2 to X9
  ADDPD X3, X2    // X2 = X2 + X3
  SUBPD X3, X9    // X9 = X9 - X3 → X2=W2, X9=W3
	MOVAPD X4, X10  // Copy X4 to X10
  ADDPD X5, X4    // X4 = X4 + X5
  SUBPD X5, X10   // X10 = X10 - X5 → X4=W4, X10=W5
	MOVAPD X6, X11  // Copy X6 to X11
  ADDPD X7, X6    // X6 = X6 + X7
  SUBPD X7, X11   // X11 = X11 - X7 → X6=W6, X11=W7
	
	// Stage 2: (0,2), (1,3), (4,6), (5,7)
	MOVUPS ·maskNegHiPD(SB), X15 // Load mask for negating high double
	// j=0 (w=1)
	MOVAPD X0, X1   // Copy X0 to X1
  ADDPD X2, X0    // X0 = X0 + X2
  SUBPD X2, X1    // X1 = X1 - X2 → X0=y0, X1=y2
	MOVAPD X4, X5   // Copy X4 to X5
  ADDPD X6, X4    // X4 = X4 + X6
  SUBPD X6, X5    // X5 = X5 - X6 → X4=y4, X5=y6
	// j=1 (w=-i)
	// y1, y3
	MOVAPD X9, X2   // Copy X9 (W3) to X2
  SHUFPD $1, X2, X2 // Swap real and imaginary parts
  XORPD X15, X2   // Negate imaginary part → t = W3 * -i
	MOVAPD X8, X3   // Copy X8 (W1) to X3
  ADDPD X2, X8    // X8 = X8 + t
  SUBPD X2, X3    // X3 = X3 - t → X8=y1, X3=y3
	// y5, y7
	MOVAPD X11, X2  // Copy X11 (W7) to X2
  SHUFPD $1, X2, X2 // Swap real and imaginary parts
  XORPD X15, X2   // Negate imaginary part → t = W7 * -i
	MOVAPD X10, X7  // Copy X10 (W5) to X7
  ADDPD X2, X10   // X10 = X10 + t
  SUBPD X2, X7    // X7 = X7 - t → X10=y5, X7=y7
	
	// Element locations: X0=y0, X8=y1, X1=y2, X3=y3, X4=y4, X10=y5, X5=y6, X7=y7
	
	// Stage 3: (y0, y4), (y1, y5), (y2, y6), (y3, y7)
	// z0, z4
	MOVAPD X0, X2   // Copy X0 (y0) to X2
  ADDPD X4, X0    // X0 = y0 + y4 → z0
  SUBPD X4, X2    // X2 = y0 - y4 → z4
	MOVUPD X0, (R14)   // Store z0 to dst[0]
	MOVUPD X2, 64(R14) // Store z4 to dst[4]
	
	// z1, z5
	MOVUPD 16(R10), X4  // Load w1 twiddle factor
	MOVAPD X10, X6      // Copy y5 to X6
  UNPCKLPD X6, X6     // Duplicate low double (real part)
  MULPD X4, X6        // X6 = real(y5) * w1
	MOVAPD X10, X9      // Copy y5 to X9
  UNPCKHPD X9, X9     // Duplicate high double (imag part)
  MOVAPD X4, X12      // Copy w1 to X12
  SHUFPD $1, X12, X12 // Swap real and imag of w1
  MULPD X9, X12       // X12 = imag(y5) * swapped(w1)
	XORPD ·maskNegLoPD(SB), X12 // Negate real part for complex mul
  ADDPD X12, X6       // X6 = t = y5 * w1
	MOVAPD X8, X12      // Copy y1 to X12
  ADDPD X6, X8        // X8 = y1 + t → z1
  SUBPD X6, X12       // X12 = y1 - t → z5
	MOVUPD X8, 16(R14)  // Store z1 to dst[1]
	MOVUPD X12, 80(R14) // Store z5 to dst[5]
	
	// z2, z6
	MOVAPD X5, X6       // Copy y6 to X6
  SHUFPD $1, X6, X6   // Swap real and imaginary parts
  XORPD X15, X6       // Negate imaginary part → t = y6 * -i
	MOVAPD X1, X12      // Copy y2 to X12
  ADDPD X6, X1        // X1 = y2 + t → z2
  SUBPD X6, X12       // X12 = y2 - t → z6
	MOVUPD X1, 32(R14)  // Store z2 to dst[2]
	MOVUPD X12, 96(R14) // Store z6 to dst[6]
	
	// z3, z7
	MOVUPD 48(R10), X4  // Load w3 twiddle factor
	MOVAPD X7, X6       // Copy y7 to X6
  UNPCKLPD X6, X6     // Duplicate low double (real part)
  MULPD X4, X6        // X6 = real(y7) * w3
	MOVAPD X7, X9       // Copy y7 to X9
  UNPCKHPD X9, X9     // Duplicate high double (imag part)
  MOVAPD X4, X12      // Copy w3 to X12
  SHUFPD $1, X12, X12 // Swap real and imag of w3
  MULPD X9, X12       // X12 = imag(y7) * swapped(w3)
	XORPD ·maskNegLoPD(SB), X12 // Negate real part for complex mul
  ADDPD X12, X6       // X6 = t = y7 * w3
	MOVAPD X3, X12      // Copy y3 to X12
  ADDPD X6, X3        // X3 = y3 + t → z3
  SUBPD X6, X12       // X12 = y3 - t → z7
	MOVUPD X3, 48(R14)  // Store z3 to dst[3]
	MOVUPD X12, 112(R14) // Store z7 to dst[7]

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

	// Stage 1: (0,1), (2,3), (4,5), (6,7) - w=1
	MOVAPD X0, X8   // Copy X0 to X8
  ADDPD X1, X0    // X0 = X0 + X1
  SUBPD X1, X8    // X8 = X8 - X1 → X0=W0, X8=W1
	MOVAPD X2, X9   // Copy X2 to X9
  ADDPD X3, X2    // X2 = X2 + X3
  SUBPD X3, X9    // X9 = X9 - X3 → X2=W2, X9=W3
	MOVAPD X4, X10  // Copy X4 to X10
  ADDPD X5, X4    // X4 = X4 + X5
  SUBPD X5, X10   // X10 = X10 - X5 → X4=W4, X10=W5
	MOVAPD X6, X11  // Copy X6 to X11
  ADDPD X7, X6    // X6 = X6 + X7
  SUBPD X7, X11   // X11 = X11 - X7 → X6=W6, X11=W7
	
	// Stage 2: (0,2), (1,3), (4,6), (5,7)
	MOVUPS ·maskNegLoPD(SB), X15 // Load mask for negating low double (for i)
	// j=0 (w=1)
	MOVAPD X0, X1   // Copy X0 to X1
  ADDPD X2, X0    // X0 = X0 + X2
  SUBPD X2, X1    // X1 = X1 - X2 → X0=y0, X1=y2
	MOVAPD X4, X5   // Copy X4 to X5
  ADDPD X6, X4    // X4 = X4 + X6
  SUBPD X6, X5    // X5 = X5 - X6 → X4=y4, X5=y6
	// j=1 (w=i)
	MOVAPD X9, X2   // Copy X9 (W3) to X2
  SHUFPD $1, X2, X2 // Swap real and imaginary parts
  XORPD X15, X2   // Negate real part → t = W3 * i
	MOVAPD X8, X3   // Copy X8 (W1) to X3
  ADDPD X2, X8    // X8 = X8 + t
  SUBPD X2, X3    // X3 = X3 - t → X8=y1, X3=y3
	MOVAPD X11, X2  // Copy X11 (W7) to X2
  SHUFPD $1, X2, X2 // Swap real and imaginary parts
  XORPD X15, X2   // Negate real part → t = W7 * i
	MOVAPD X10, X7  // Copy X10 (W5) to X7
  ADDPD X2, X10   // X10 = X10 + t
  SUBPD X2, X7    // X7 = X7 - t → X10=y5, X7=y7
	
	MOVUPS ·maskNegHiPD(SB), X14 // Load mask for conjugate (negate imag)

	// Stage 3: (y0, y4), (y1, y5), (y2, y6), (y3, y7)
	// z0, z4
	MOVAPD X0, X2   // Copy X0 (y0) to X2
  ADDPD X4, X0    // X0 = y0 + y4 → z0
  SUBPD X4, X2    // X2 = y0 - y4 → z4
	MOVUPD X0, 0(R11)  // Store z0 to scratch[0]
	MOVUPD X2, 64(R11) // Store z4 to scratch[4]
	
	// z1, z5
	MOVUPD 16(R10), X4  // Load w1 twiddle factor
  XORPD X14, X4       // Conjugate w1 (negate imag)
	MOVAPD X10, X6      // Copy y5 to X6
  UNPCKLPD X6, X6     // Duplicate low double (real part)
  MULPD X4, X6        // X6 = real(y5) * conj(w1)
	MOVAPD X10, X9      // Copy y5 to X9
  UNPCKHPD X9, X9     // Duplicate high double (imag part)
  MOVAPD X4, X12      // Copy conj(w1) to X12
  SHUFPD $1, X12, X12 // Swap real and imag
  MULPD X9, X12       // X12 = imag(y5) * swapped(conj(w1))
	XORPD ·maskNegLoPD(SB), X12 // Negate real part for complex mul
  ADDPD X12, X6       // X6 = t = y5 * conj(w1)
	MOVAPD X8, X12      // Copy y1 to X12
  ADDPD X6, X8        // X8 = y1 + t → z1
  SUBPD X6, X12       // X12 = y1 - t → z5
	MOVUPD X8, 16(R11)  // Store z1 to scratch[1]
	MOVUPD X12, 80(R11) // Store z5 to scratch[5]
	
	// z2, z6 (w=i)
	MOVAPD X5, X6       // Copy y6 to X6
  SHUFPD $1, X6, X6   // Swap real and imaginary parts
  XORPD X15, X6       // Negate real part → t = y6 * i
	MOVAPD X1, X12      // Copy y2 to X12
  ADDPD X6, X1        // X1 = y2 + t → z2
  SUBPD X6, X12       // X12 = y2 - t → z6
	MOVUPD X1, 32(R11)  // Store z2 to scratch[2]
	MOVUPD X12, 96(R11) // Store z6 to scratch[6]
	
	// z3, z7
	MOVUPD 48(R10), X4  // Load w3 twiddle factor
  XORPD X14, X4       // Conjugate w3 (negate imag)
	MOVAPD X7, X6       // Copy y7 to X6
  UNPCKLPD X6, X6     // Duplicate low double (real part)
  MULPD X4, X6        // X6 = real(y7) * conj(w3)
	MOVAPD X7, X9       // Copy y7 to X9
  UNPCKHPD X9, X9     // Duplicate high double (imag part)
  MOVAPD X4, X12      // Copy conj(w3) to X12
  SHUFPD $1, X12, X12 // Swap real and imag
  MULPD X9, X12       // X12 = imag(y7) * swapped(conj(w3))
	XORPD ·maskNegLoPD(SB), X12 // Negate real part for complex mul
  ADDPD X12, X6       // X6 = t = y7 * conj(w3)
	MOVAPD X3, X12      // Copy y3 to X12
  ADDPD X6, X3        // X3 = y3 + t → z3
  SUBPD X6, X12       // X12 = y3 - t → z7
	MOVUPD X3, 48(R11)  // Store z3 to scratch[3]
	MOVUPD X12, 112(R11) // Store z7 to scratch[7]

	// Scale and Store
	MOVSD ·eighth64(SB), X15 // Load 1/8 scaling factor
  SHUFPD $0, X15, X15      // Broadcast to both doubles
	MOVQ $8, CX     // Loop counter = 8 elements
  MOVQ R11, SI    // Source = scratch buffer
  MOVQ R14, DI    // Destination = dst
scale_loop:
	MOVUPD (SI), X0 // Load complex number from scratch
  MULPD X15, X0   // Scale by 1/8
  MOVUPD X0, (DI) // Store to destination
	ADDQ $16, SI    // Advance source pointer
  ADDQ $16, DI    // Advance destination pointer
	DECQ CX         // Decrement counter
  JNZ scale_loop  // Loop if not zero

	MOVB $1, ret+96(FP)
	RET
inv_err:
	MOVB $0, ret+96(FP)
	RET

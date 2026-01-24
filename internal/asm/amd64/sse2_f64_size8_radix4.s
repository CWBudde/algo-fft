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
	MOVUPS ·maskNegHiPD(SB), X15 // Load mask for negating high double

	// Butterfly 1: [x0, x1, x2, x3]
	MOVAPD X0, X8   // Copy x0 to X8
  ADDPD X2, X8    // X8 = x0 + x2 → t0
  MOVAPD X0, X9   // Copy x0 to X9
  SUBPD X2, X9    // X9 = x0 - x2 → t1
	MOVAPD X1, X10  // Copy x1 to X10
  ADDPD X3, X10   // X10 = x1 + x3 → t2
  MOVAPD X1, X11  // Copy x1 to X11
  SUBPD X3, X11   // X11 = x1 - x3 → t3
	MOVAPD X11, X12 // Copy t3 to X12
  SHUFPD $1, X12, X12 // Swap real and imaginary parts
  XORPD X15, X12  // Negate imaginary part → t3 * -i
	MOVAPD X8, X0   // Copy t0 to X0
  ADDPD X10, X0   // X0 = t0 + t2 → a0
	MOVAPD X9, X1   // Copy t1 to X1
  ADDPD X12, X1   // X1 = t1 + (t3*-i) → a1
	MOVAPD X8, X2   // Copy t0 to X2
  SUBPD X10, X2   // X2 = t0 - t2 → a2
	MOVAPD X9, X3   // Copy t1 to X3
  SUBPD X12, X3   // X3 = t1 - (t3*-i) → a3

	// Butterfly 2: [x4, x5, x6, x7]
	MOVAPD X4, X8   // Copy x4 to X8
  ADDPD X6, X8    // X8 = x4 + x6 → t0
  MOVAPD X4, X9   // Copy x4 to X9
  SUBPD X6, X9    // X9 = x4 - x6 → t1
	MOVAPD X5, X10  // Copy x5 to X10
  ADDPD X7, X10   // X10 = x5 + x7 → t2
  MOVAPD X5, X11  // Copy x5 to X11
  SUBPD X7, X11   // X11 = x5 - x7 → t3
	MOVAPD X11, X12 // Copy t3 to X12
  SHUFPD $1, X12, X12 // Swap real and imaginary parts
  XORPD X15, X12  // Negate imaginary part → t3 * -i
	MOVAPD X8, X4   // Copy t0 to X4
  ADDPD X10, X4   // X4 = t0 + t2 → a4
	MOVAPD X9, X5   // Copy t1 to X5
  ADDPD X12, X5   // X5 = t1 + (t3*-i) → a5
	MOVAPD X8, X6   // Copy t0 to X6
  SUBPD X10, X6   // X6 = t0 - t2 → a6
	MOVAPD X9, X7   // Copy t1 to X7
  SUBPD X12, X7   // X7 = t1 - (t3*-i) → a7

	// Stage 2: Radix-2 combine with twiddles
	// y0, y4
	MOVAPD X0, X8   // Copy a0 to X8
  ADDPD X4, X8    // X8 = a0 + a4 → y0
  MOVAPD X0, X9   // Copy a0 to X9
  SUBPD X4, X9    // X9 = a0 - a4 → y4
	MOVUPD X8, (R8)   // Store y0 to dst[0]
	MOVUPD X9, 64(R8) // Store y4 to dst[4]

	// y1, y5 (w1 * a5)
	MOVUPD 16(R10), X10 // Load w1 twiddle factor
	MOVAPD X5, X11      // Copy a5 to X11
  UNPCKLPD X11, X11   // Duplicate low double (real part)
  MULPD X10, X11      // X11 = real(a5) * w1
	MOVAPD X5, X12      // Copy a5 to X12
  UNPCKHPD X12, X12   // Duplicate high double (imag part)
  MOVAPD X10, X13     // Copy w1 to X13
  SHUFPD $1, X13, X13 // Swap real and imag of w1
  MULPD X12, X13      // X13 = imag(a5) * swapped(w1)
	XORPD ·maskNegLoPD(SB), X13 // Negate real part for complex mul
  ADDPD X13, X11      // X11 = t1 = a5 * w1
	MOVAPD X1, X12      // Copy a1 to X12
  ADDPD X11, X1       // X1 = a1 + t1 → y1
  SUBPD X11, X12      // X12 = a1 - t1 → y5
	MOVUPD X1, 16(R8)   // Store y1 to dst[1]
	MOVUPD X12, 80(R8)  // Store y5 to dst[5]

	// y2, y6 (w2 * a6)
	MOVUPD 32(R10), X10 // Load w2 twiddle factor
	MOVAPD X6, X11      // Copy a6 to X11
  UNPCKLPD X11, X11   // Duplicate low double (real part)
  MULPD X10, X11      // X11 = real(a6) * w2
	MOVAPD X6, X12      // Copy a6 to X12
  UNPCKHPD X12, X12   // Duplicate high double (imag part)
  MOVAPD X10, X13     // Copy w2 to X13
  SHUFPD $1, X13, X13 // Swap real and imag of w2
  MULPD X12, X13      // X13 = imag(a6) * swapped(w2)
	XORPD ·maskNegLoPD(SB), X13 // Negate real part for complex mul
  ADDPD X13, X11      // X11 = t2 = a6 * w2
	MOVAPD X2, X12      // Copy a2 to X12
  ADDPD X11, X2       // X2 = a2 + t2 → y2
  SUBPD X11, X12      // X12 = a2 - t2 → y6
	MOVUPD X2, 32(R8)   // Store y2 to dst[2]
	MOVUPD X12, 96(R8)  // Store y6 to dst[6]

	// y3, y7 (w3 * a7)
	MOVUPD 48(R10), X10 // Load w3 twiddle factor
	MOVAPD X7, X11      // Copy a7 to X11
  UNPCKLPD X11, X11   // Duplicate low double (real part)
  MULPD X10, X11      // X11 = real(a7) * w3
	MOVAPD X7, X12      // Copy a7 to X12
  UNPCKHPD X12, X12   // Duplicate high double (imag part)
  MOVAPD X10, X13     // Copy w3 to X13
  SHUFPD $1, X13, X13 // Swap real and imag of w3
  MULPD X12, X13      // X13 = imag(a7) * swapped(w3)
	XORPD ·maskNegLoPD(SB), X13 // Negate real part for complex mul
  ADDPD X13, X11      // X11 = t3 = a7 * w3
	MOVAPD X3, X12      // Copy a3 to X12
  ADDPD X11, X3       // X3 = a3 + t3 → y3
  SUBPD X11, X12      // X12 = a3 - t3 → y7
	MOVUPD X3, 48(R8)   // Store y3 to dst[3]
	MOVUPD X12, 112(R8) // Store y7 to dst[7]

	// Copy to R14 if needed
	CMPQ R8, R14        // Compare current output with final destination
	JE size8_r4_fwd_done // Skip copy if already in destination
	MOVQ $8, CX         // Loop counter = 8 elements
  MOVQ R8, SI         // Source = current output buffer
  MOVQ R14, DI        // Destination = final output
size8_r4_fwd_copy:
	MOVUPD (SI), X0     // Load complex number from source
  MOVUPD X0, (DI)     // Store to destination
  ADDQ $16, SI        // Advance source pointer
  ADDQ $16, DI        // Advance destination pointer
  DECQ CX             // Decrement counter
  JNZ size8_r4_fwd_copy // Loop if not zero

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
	MOVUPS ·maskNegLoPD(SB), X14 // Load mask for negating low double (for i)

	// Butterfly 1: [x0, x1, x2, x3]
	MOVAPD X0, X8   // Copy x0 to X8
  ADDPD X2, X8    // X8 = x0 + x2 → t0
  MOVAPD X0, X9   // Copy x0 to X9
  SUBPD X2, X9    // X9 = x0 - x2 → t1
	MOVAPD X1, X10  // Copy x1 to X10
  ADDPD X3, X10   // X10 = x1 + x3 → t2
  MOVAPD X1, X11  // Copy x1 to X11
  SUBPD X3, X11   // X11 = x1 - x3 → t3
	MOVAPD X11, X12 // Copy t3 to X12
  SHUFPD $1, X12, X12 // Swap real and imaginary parts
  XORPD X14, X12  // Negate real part → t3 * i
	MOVAPD X8, X0   // Copy t0 to X0
  ADDPD X10, X0   // X0 = t0 + t2 → a0
	MOVAPD X9, X1   // Copy t1 to X1
  ADDPD X12, X1   // X1 = t1 + (t3*i) → a1
	MOVAPD X8, X2   // Copy t0 to X2
  SUBPD X10, X2   // X2 = t0 - t2 → a2
	MOVAPD X9, X3   // Copy t1 to X3
  SUBPD X12, X3   // X3 = t1 - (t3*i) → a3

	// Butterfly 2: [x4, x5, x6, x7]
	MOVAPD X4, X8   // Copy x4 to X8
  ADDPD X6, X8    // X8 = x4 + x6 → t0
  MOVAPD X4, X9   // Copy x4 to X9
  SUBPD X6, X9    // X9 = x4 - x6 → t1
	MOVAPD X5, X10  // Copy x5 to X10
  ADDPD X7, X10   // X10 = x5 + x7 → t2
  MOVAPD X5, X11  // Copy x5 to X11
  SUBPD X7, X11   // X11 = x5 - x7 → t3
	MOVAPD X11, X12 // Copy t3 to X12
  SHUFPD $1, X12, X12 // Swap real and imaginary parts
  XORPD X14, X12  // Negate real part → t3 * i
	MOVAPD X8, X4   // Copy t0 to X4
  ADDPD X10, X4   // X4 = t0 + t2 → a4
	MOVAPD X9, X5   // Copy t1 to X5
  ADDPD X12, X5   // X5 = t1 + (t3*i) → a5
	MOVAPD X8, X6   // Copy t0 to X6
  SUBPD X10, X6   // X6 = t0 - t2 → a6
	MOVAPD X9, X7   // Copy t1 to X7
  SUBPD X12, X7   // X7 = t1 - (t3*i) → a7

	// Stage 2: Radix-2 combine
	MOVUPS ·maskNegHiPD(SB), X15 // Load mask for conjugate (negate imag)

	// y0, y4
	MOVAPD X0, X8   // Copy a0 to X8
  ADDPD X4, X8    // X8 = a0 + a4 → y0
  MOVAPD X0, X9   // Copy a0 to X9
  SUBPD X4, X9    // X9 = a0 - a4 → y4
	MOVUPD X8, 0(R11)  // Store y0 to scratch[0]
	MOVUPD X9, 64(R11) // Store y4 to scratch[4]

	// y1, y5 (conj(w1) * a5)
	MOVUPD 16(R10), X10 // Load w1 twiddle factor
  XORPD X15, X10      // Conjugate w1 (negate imag)
	MOVAPD X5, X11      // Copy a5 to X11
  UNPCKLPD X11, X11   // Duplicate low double (real part)
  MULPD X10, X11      // X11 = real(a5) * conj(w1)
	MOVAPD X5, X12      // Copy a5 to X12
  UNPCKHPD X12, X12   // Duplicate high double (imag part)
  MOVAPD X10, X13     // Copy conj(w1) to X13
  SHUFPD $1, X13, X13 // Swap real and imag
  MULPD X12, X13      // X13 = imag(a5) * swapped(conj(w1))
	XORPD ·maskNegLoPD(SB), X13 // Negate real part for complex mul
  ADDPD X13, X11      // X11 = t1 = a5 * conj(w1)
	MOVAPD X1, X12      // Copy a1 to X12
  ADDPD X11, X1       // X1 = a1 + t1 → y1
  SUBPD X11, X12      // X12 = a1 - t1 → y5
	MOVUPD X1, 16(R11)  // Store y1 to scratch[1]
	MOVUPD X12, 80(R11) // Store y5 to scratch[5]

	// y2, y6 (conj(w2) * a6)
	MOVUPD 32(R10), X10 // Load w2 twiddle factor
  XORPD X15, X10      // Conjugate w2 (negate imag)
	MOVAPD X6, X11      // Copy a6 to X11
  UNPCKLPD X11, X11   // Duplicate low double (real part)
  MULPD X10, X11      // X11 = real(a6) * conj(w2)
	MOVAPD X6, X12      // Copy a6 to X12
  UNPCKHPD X12, X12   // Duplicate high double (imag part)
  MOVAPD X10, X13     // Copy conj(w2) to X13
  SHUFPD $1, X13, X13 // Swap real and imag
  MULPD X12, X13      // X13 = imag(a6) * swapped(conj(w2))
	XORPD ·maskNegLoPD(SB), X13 // Negate real part for complex mul
  ADDPD X13, X11      // X11 = t2 = a6 * conj(w2)
	MOVAPD X2, X12      // Copy a2 to X12
  ADDPD X11, X2       // X2 = a2 + t2 → y2
  SUBPD X11, X12      // X12 = a2 - t2 → y6
	MOVUPD X2, 32(R11)  // Store y2 to scratch[2]
	MOVUPD X12, 96(R11) // Store y6 to scratch[6]

	// y3, y7 (conj(w3) * a7)
	MOVUPD 48(R10), X10 // Load w3 twiddle factor
  XORPD X15, X10      // Conjugate w3 (negate imag)
	MOVAPD X7, X11      // Copy a7 to X11
  UNPCKLPD X11, X11   // Duplicate low double (real part)
  MULPD X10, X11      // X11 = real(a7) * conj(w3)
	MOVAPD X7, X12      // Copy a7 to X12
  UNPCKHPD X12, X12   // Duplicate high double (imag part)
  MOVAPD X10, X13     // Copy conj(w3) to X13
  SHUFPD $1, X13, X13 // Swap real and imag
  MULPD X12, X13      // X13 = imag(a7) * swapped(conj(w3))
	XORPD ·maskNegLoPD(SB), X13 // Negate real part for complex mul
  ADDPD X13, X11      // X11 = t3 = a7 * conj(w3)
	MOVAPD X3, X12      // Copy a3 to X12
  ADDPD X11, X3       // X3 = a3 + t3 → y3
  SUBPD X11, X12      // X12 = a3 - t3 → y7
	MOVUPD X3, 48(R11)  // Store y3 to scratch[3]
	MOVUPD X12, 112(R11) // Store y7 to scratch[7]

	// Scale by 1/8 and Store
	MOVSD ·eighth64(SB), X15 // Load 1/8 scaling factor
  SHUFPD $0, X15, X15      // Broadcast to both doubles
	MOVQ $8, CX     // Loop counter = 8 elements
  MOVQ R11, SI    // Source = scratch buffer
  MOVQ R14, DI    // Destination = dst
size8_r4_inv_scale:
	MOVUPD (SI), X0 // Load complex number from scratch
  MULPD X15, X0   // Scale by 1/8
  MOVUPD X0, (DI) // Store to destination
  ADDQ $16, SI    // Advance source pointer
  ADDQ $16, DI    // Advance destination pointer
  DECQ CX         // Decrement counter
  JNZ size8_r4_inv_scale // Loop if not zero

	MOVB $1, ret+96(FP)
	RET
size8_r4_inv_err:
	MOVB $0, ret+96(FP)
	RET

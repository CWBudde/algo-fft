//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-8 Radix-8 (complex128) FFT Kernels for AMD64 (complex128)
// ===========================================================================
//
// This file contains fully-unrolled FFT kernels optimized for size 8.
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 8, complex128, radix-8 variant
// Single radix-8 butterfly without bit-reversal.
// ===========================================================================
TEXT ·ForwardAVX2Size8Radix8Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // dst
	MOVQ src+24(FP), R9      // src
	MOVQ twiddle+48(FP), R10 // twiddle
	MOVQ scratch+72(FP), R11 // scratch
	MOVQ src+32(FP), R13     // n

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_128_r8_fwd_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_128_r8_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_128_r8_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_128_r8_fwd_return_false

	// Build sign masks
	// X14 = maskNegLo = [signbit, 0]
	// X15 = maskNegHi = [0, signbit]
	MOVQ ·signbit64(SB), AX
	VMOVQ AX, X14
	VPERMILPD $1, X14, X15

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

	// a0..a7
	VADDPD X4, X0, X8   // a0 = x0 + x4
	VSUBPD X4, X0, X9   // a1 = x0 - x4
	VADDPD X6, X2, X10  // a2 = x2 + x6
	VSUBPD X6, X2, X11  // a3 = x2 - x6
	VADDPD X5, X1, X12  // a4 = x1 + x5
	VSUBPD X5, X1, X13  // a5 = x1 - x5
	VADDPD X7, X3, X6   // a6 = x3 + x7  (reuse X6)
	VSUBPD X7, X3, X7   // a7 = x3 - x7  (reuse X7)

	// Even terms
	VADDPD X10, X8, X0  // e0 = a0 + a2
	VSUBPD X10, X8, X2  // e2 = a0 - a2

	// e1 = a1 + mulNegI(a3) = a1 + [im(a3), -re(a3)]
	VPERMILPD $1, X11, X1
	VXORPD X15, X1, X1
	VADDPD X9, X1, X1

	// e3 = a1 + mulI(a3) = a1 + [-im(a3), re(a3)]
	VPERMILPD $1, X11, X3
	VXORPD X14, X3, X3
	VADDPD X9, X3, X3

	// Odd terms
	VADDPD X6, X12, X4  // o0 = a4 + a6
	VSUBPD X6, X12, X6  // o2 = a4 - a6

	// o1 = a5 + mulNegI(a7)
	VPERMILPD $1, X7, X5
	VXORPD X15, X5, X5
	VADDPD X13, X5, X5

	// o3 = a5 + mulI(a7)
	VPERMILPD $1, X7, X7
	VXORPD X14, X7, X7
	VADDPD X13, X7, X7

	// out0/out4
	VADDPD X4, X0, X8   // out0
	VSUBPD X4, X0, X12  // out4

	// t1 = w1 * o1
	MOVUPD 16(R10), X10
	VMOVDDUP X10, X11          // w1_re
	VPERMILPD $1, X10, X10
	VMOVDDUP X10, X10          // w1_im
	VPERMILPD $1, X5, X13      // swap(o1)
	VMULPD X10, X13, X13
	VFMADDSUB231PD X11, X5, X13 // t1
	VADDPD X13, X1, X9         // out1 = e1 + t1
	VSUBPD X13, X1, X13        // out5 = e1 - t1

	// t2 = w2 * o2
	MOVUPD 32(R10), X10
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X10
	VMOVDDUP X10, X10
	VPERMILPD $1, X6, X0       // swap(o2) (reuse X0)
	VMULPD X10, X0, X0
	VFMADDSUB231PD X11, X6, X0 // t2 in X0
	VADDPD X0, X2, X10         // out2
	VSUBPD X0, X2, X14         // out6 (reuse X14)

	// t3 = w3 * o3
	MOVUPD 48(R10), X0
	VMOVDDUP X0, X11
	VPERMILPD $1, X0, X0
	VMOVDDUP X0, X0
	VPERMILPD $1, X7, X2       // swap(o3) (reuse X2)
	VMULPD X0, X2, X2
	VFMADDSUB231PD X11, X7, X2 // t3 in X2
	VADDPD X2, X3, X11         // out3
	VSUBPD X2, X3, X15         // out7 (reuse X15)

	// Store results to dst
	MOVUPD X8, 0(R8)
	MOVUPD X9, 16(R8)
	MOVUPD X10, 32(R8)
	MOVUPD X11, 48(R8)
	MOVUPD X12, 64(R8)
	MOVUPD X13, 80(R8)
	MOVUPD X14, 96(R8)
	MOVUPD X15, 112(R8)

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size8_128_r8_fwd_return_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 8, complex128, radix-8 variant
// Uses conjugated twiddles (via VFMSUBADD) and applies 1/8 scaling.
// ===========================================================================
TEXT ·InverseAVX2Size8Radix8Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_128_r8_inv_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_128_r8_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_128_r8_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_128_r8_inv_return_false

	// Note: bitrev parameter ignored for radix-8 on size-8 (identity permutation)

	// Build sign masks
	MOVQ ·signbit64(SB), AX
	VMOVQ AX, X14            // maskNegLo = [signbit, 0]
	VPERMILPD $1, X14, X15   // maskNegHi = [0, signbit]

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

	// a0..a7
	VADDPD X4, X0, X8
	VSUBPD X4, X0, X9
	VADDPD X6, X2, X10
	VSUBPD X6, X2, X11
	VADDPD X5, X1, X12
	VSUBPD X5, X1, X13
	VADDPD X7, X3, X6
	VSUBPD X7, X3, X7

	// Even terms
	VADDPD X10, X8, X0  // e0
	VSUBPD X10, X8, X2  // e2

	// e1 = a1 + mulI(a3)
	VPERMILPD $1, X11, X1
	VXORPD X14, X1, X1
	VADDPD X9, X1, X1

	// e3 = a1 + mulNegI(a3)
	VPERMILPD $1, X11, X3
	VXORPD X15, X3, X3
	VADDPD X9, X3, X3

	// Odd terms
	VADDPD X6, X12, X4  // o0
	VSUBPD X6, X12, X6  // o2

	// o1 = a5 + mulI(a7)
	VPERMILPD $1, X7, X5
	VXORPD X14, X5, X5
	VADDPD X13, X5, X5

	// o3 = a5 + mulNegI(a7)
	VPERMILPD $1, X7, X7
	VXORPD X15, X7, X7
	VADDPD X13, X7, X7

	// out0/out4
	VADDPD X4, X0, X8
	VSUBPD X4, X0, X12

	// t1 = conj(w1) * o1
	MOVUPD 16(R10), X10
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X10
	VMOVDDUP X10, X10
	VPERMILPD $1, X5, X13
	VMULPD X10, X13, X13
	VFMSUBADD231PD X11, X5, X13
	VADDPD X13, X1, X9
	VSUBPD X13, X1, X13

	// t2 = conj(w2) * o2
	MOVUPD 32(R10), X10
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X10
	VMOVDDUP X10, X10
	VPERMILPD $1, X6, X0
	VMULPD X10, X0, X0
	VFMSUBADD231PD X11, X6, X0
	VADDPD X0, X2, X10
	VSUBPD X0, X2, X14

	// t3 = conj(w3) * o3
	MOVUPD 48(R10), X0
	VMOVDDUP X0, X11
	VPERMILPD $1, X0, X0
	VMOVDDUP X0, X0
	VPERMILPD $1, X7, X2
	VMULPD X0, X2, X2
	VFMSUBADD231PD X11, X7, X2
	VADDPD X2, X3, X11
	VSUBPD X2, X3, X15

	// Apply 1/8 scaling for inverse FFT
	MOVQ ·eighth64(SB), AX  // 0.125 = 1/8
	VMOVQ AX, X0
	VMOVDDUP X0, X0
	VMULPD X0, X8, X8
	VMULPD X0, X9, X9
	VMULPD X0, X10, X10
	VMULPD X0, X11, X11
	VMULPD X0, X12, X12
	VMULPD X0, X13, X13
	VMULPD X0, X14, X14
	VMULPD X0, X15, X15

	// Store results to dst
	MOVUPD X8, 0(R8)
	MOVUPD X9, 16(R8)
	MOVUPD X10, 32(R8)
	MOVUPD X11, 48(R8)
	MOVUPD X12, 64(R8)
	MOVUPD X13, 80(R8)
	MOVUPD X14, 96(R8)
	MOVUPD X15, 112(R8)

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size8_128_r8_inv_return_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Forward transform, size 8, complex128, radix-4 (mixed-radix) variant
// ===========================================================================
// Mixed-radix FFT: 1 radix-4 stage + 1 radix-2 stage (2 stages total)
// More efficient than 3-stage radix-2.
//
// Stage 1 (Radix-4): 2 radix-4 butterflies on [x0,x2,x4,x6] and [x1,x3,x5,x7]
// Stage 2 (Radix-2): 4 radix-2 butterflies combining results with twiddles

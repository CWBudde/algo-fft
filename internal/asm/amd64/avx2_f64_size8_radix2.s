//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-8 Radix-2 (complex128) FFT Kernels for AMD64 (complex128)
// ===========================================================================
//
// This file contains fully-unrolled FFT kernels optimized for size 8.
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 8, complex128, radix-2 variant
// ===========================================================================
TEXT ·ForwardAVX2Size8Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ R8, R14             // R14 = original dst pointer (for in-place safety)
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 8)

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_128_fwd_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_128_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_128_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_128_fwd_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_128_fwd_use_dst
	MOVQ R11, R8             // In-place: use scratch

size8_128_fwd_use_dst:
	// Bit-reversal: work[i] = src[bitrev[i]]
	// complex128 is 16 bytes
	MOVUPD 0(R9), X0         // src[0]
	MOVUPD 64(R9), X1        // src[4]
	MOVUPD 32(R9), X2        // src[2]
	MOVUPD 96(R9), X3        // src[6]
	MOVUPD 16(R9), X4        // src[1]
	MOVUPD 80(R9), X5        // src[5]
	MOVUPD 48(R9), X6        // src[3]
	MOVUPD 112(R9), X7       // src[7]

	// Stage 1: size=2, half=1, step=2
	// (0,1), (2,3), (4,5), (6,7) - all twiddle[0]=1
	VADDPD X1, X0, X8        // W0'
	VSUBPD X1, X0, X9        // W1'
	VADDPD X3, X2, X10       // W2'
	VSUBPD X3, X2, X11       // W3'
	VADDPD X5, X4, X12       // W4'
	VSUBPD X5, X4, X13       // W5'
	VADDPD X7, X6, X14       // W6'
	VSUBPD X7, X6, X15       // W7'

	// Stage 2: size=4, half=2, step=4
	// (W0, W2), (W1, W3), (W4, W6), (W5, W7)
	
	// j=0: twiddle[0]=1
	VADDPD X10, X8, X0       // y0
	VSUBPD X10, X8, X2       // y2
	VADDPD X14, X12, X4      // y4
	VSUBPD X14, X12, X6      // y6
	
	// j=1: twiddle[2]
	MOVUPD 32(R10), X10      // w2
	VMOVDDUP X10, X8         // w2_re
	VPERMILPD $1, X10, X10
	VMOVDDUP X10, X10        // w2_im
	
	// (W1, W3)
	VPERMILPD $1, X11, X12   // [i3, r3]
	VMULPD X10, X12, X12     // [im*i3, im*r3]
	VFMADDSUB231PD X8, X11, X12 // X12 = w2 * W3
	VADDPD X12, X9, X1       // y1
	VSUBPD X12, X9, X3       // y3
	
	// (W5, W7)
	VPERMILPD $1, X15, X12   // [i7, r7]
	VMULPD X10, X12, X12
	VFMADDSUB231PD X8, X15, X12 // X12 = w2 * W7
	VADDPD X12, X13, X5      // y5
	VSUBPD X12, X13, X7      // y7
	
	// Stage 3: size=8, half=4, step=8
	// (y0, y4), (y1, y5), (y2, y6), (y3, y7)
	
	// j=0: twiddle[0]=1
	VADDPD X4, X0, X8        // z0
	VSUBPD X4, X0, X12       // z4
	MOVUPD X12, X4           // save z4 (X12 will be reused as a temp)
	
	// j=1: twiddle[1]
	MOVUPD 16(R10), X10      // w1
	VMOVDDUP X10, X11        // w1_re
	VPERMILPD $1, X10, X10
	VMOVDDUP X10, X10        // w1_im
	VPERMILPD $1, X5, X0     // [i5, r5]
	VMULPD X10, X0, X0
	VFMADDSUB231PD X11, X5, X0 // X0 = w1 * y5
	VADDPD X0, X1, X9        // z1
	VSUBPD X0, X1, X13       // z5
	
	// j=2: twiddle[2]
	MOVUPD 32(R10), X10      // w2
	VMOVDDUP X10, X11        // w2_re
	VPERMILPD $1, X10, X10
	VMOVDDUP X10, X10        // w2_im
	VPERMILPD $1, X6, X0     // [i6, r6]
	VMULPD X10, X0, X0
	VFMADDSUB231PD X11, X6, X0 // X0 = w2 * y6
	VADDPD X0, X2, X10       // z2
	VSUBPD X0, X2, X14       // z6
	
	// j=3: twiddle[3]
	MOVUPD 48(R10), X0       // w3
	VMOVDDUP X0, X11         // w3_re
	VPERMILPD $1, X0, X0
	VMOVDDUP X0, X0          // w3_im
	VPERMILPD $1, X7, X6     // [i7, r7]
	VMULPD X0, X6, X6
	VFMADDSUB231PD X11, X7, X6 // X6 = w3 * y7
	VADDPD X6, X3, X11       // z3
	VSUBPD X6, X3, X15       // z7
	
	// Store results
	MOVUPD X8, (R14)
	MOVUPD X9, 16(R14)
	MOVUPD X10, 32(R14)
	MOVUPD X11, 48(R14)
	MOVUPD X4, 64(R14)
	MOVUPD X13, 80(R14)
	MOVUPD X14, 96(R14)
	MOVUPD X15, 112(R14)

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size8_128_fwd_return_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 8, complex128, radix-2 variant
// ===========================================================================
TEXT ·InverseAVX2Size8Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14             // R14 = original dst pointer (for in-place safety)
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_128_inv_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_128_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_128_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_128_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_128_inv_use_dst
	MOVQ R11, R8

size8_128_inv_use_dst:
	// Bit-reversal
	MOVUPD 0(R9), X0         // src[0]
	MOVUPD 64(R9), X1        // src[4]
	MOVUPD 32(R9), X2        // src[2]
	MOVUPD 96(R9), X3        // src[6]
	MOVUPD 16(R9), X4        // src[1]
	MOVUPD 80(R9), X5        // src[5]
	MOVUPD 48(R9), X6        // src[3]
	MOVUPD 112(R9), X7       // src[7]

	// Stage 1: size=2, half=1, step=2
	VADDPD X1, X0, X8
	VSUBPD X1, X0, X9
	VADDPD X3, X2, X10
	VSUBPD X3, X2, X11
	VADDPD X5, X4, X12
	VSUBPD X5, X4, X13
	VADDPD X7, X6, X14
	VSUBPD X7, X6, X15

	// Stage 2: size=4, half=2, step=4
	// j=0
	VADDPD X10, X8, X0
	VSUBPD X10, X8, X2
	VADDPD X14, X12, X4
	VSUBPD X14, X12, X6
	
	// j=1: twiddle[2]
	MOVUPD 32(R10), X10
	VMOVDDUP X10, X8
	VPERMILPD $1, X10, X10
	VMOVDDUP X10, X10
	
	// (1,3)
	VPERMILPD $1, X11, X12
	VMULPD X10, X12, X12
	VFMSUBADD231PD X8, X11, X12 // conj(w2) * X11
	VADDPD X12, X9, X1
	VSUBPD X12, X9, X3
	
	// (5,7)
	VPERMILPD $1, X15, X12
	VMULPD X10, X12, X12
	VFMSUBADD231PD X8, X15, X12
	VADDPD X12, X13, X5
	VSUBPD X12, X13, X7
	
	// Stage 3: size=8, half=4, step=8
	// j=0
	VADDPD X4, X0, X8
	VSUBPD X4, X0, X12
	MOVUPD X12, X4           // save z4 (X12 will be reused as a temp)
	
	// j=1: twiddle[1]
	MOVUPD 16(R10), X10
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X10
	VMOVDDUP X10, X10
	VPERMILPD $1, X5, X0
	VMULPD X10, X0, X0
	VFMSUBADD231PD X11, X5, X0
	VADDPD X0, X1, X9
	VSUBPD X0, X1, X13
	
	// j=2: twiddle[2]
	MOVUPD 32(R10), X10
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X10
	VMOVDDUP X10, X10
	VPERMILPD $1, X6, X0
	VMULPD X10, X0, X0
	VFMSUBADD231PD X11, X6, X0
	VADDPD X0, X2, X10
	VSUBPD X0, X2, X14
	
	// j=3: twiddle[3]
	MOVUPD 48(R10), X0
	VMOVDDUP X0, X11
	VPERMILPD $1, X0, X0
	VMOVDDUP X0, X0
	VPERMILPD $1, X7, X6
	VMULPD X0, X6, X6
	VFMSUBADD231PD X11, X7, X6
	VADDPD X6, X3, X11
	VSUBPD X6, X3, X15
	
	// Apply 1/8 scaling
	MOVQ ·eighth64(SB), AX // 1/8 = 0.125
	VMOVQ AX, X0
	VMOVDDUP X0, X0
	VMULPD X0, X8, X8
	VMULPD X0, X9, X9
	VMULPD X0, X10, X10
	VMULPD X0, X11, X11
	VMULPD X0, X4, X12
	VMULPD X0, X13, X13
	VMULPD X0, X14, X14
	VMULPD X0, X15, X15

	// Store results
	MOVUPD X8, (R14)
	MOVUPD X9, 16(R14)
	MOVUPD X10, 32(R14)
	MOVUPD X11, 48(R14)
	MOVUPD X12, 64(R14)
	MOVUPD X13, 80(R14)
	MOVUPD X14, 96(R14)
	MOVUPD X15, 112(R14)

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size8_128_inv_return_false:
	MOVB $0, ret+96(FP)
	RET

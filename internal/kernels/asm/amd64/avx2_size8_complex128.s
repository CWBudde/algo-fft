//go:build amd64 && fft_asm && !purego

// ===========================================================================
// AVX2 Size-8 FFT Kernels for AMD64 (complex128)
// ===========================================================================
//
// This file contains fully-unrolled FFT kernels optimized for size 8.
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 8, complex128, radix-2 variant
// ===========================================================================
TEXT ·forwardAVX2Size8Radix2Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ R8, R14             // R14 = original dst pointer (for in-place safety)
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
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

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   size8_128_fwd_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_128_fwd_use_dst
	MOVQ R11, R8             // In-place: use scratch

size8_128_fwd_use_dst:
	// Bit-reversal: work[i] = src[bitrev[i]]
	// complex128 is 16 bytes
	MOVQ (R12), DX           // bitrev[0]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X0
	MOVQ 8(R12), DX          // bitrev[1]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X1
	MOVQ 16(R12), DX         // bitrev[2]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X2
	MOVQ 24(R12), DX         // bitrev[3]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X3
	MOVQ 32(R12), DX         // bitrev[4]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X4
	MOVQ 40(R12), DX         // bitrev[5]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X5
	MOVQ 48(R12), DX         // bitrev[6]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X6
	MOVQ 56(R12), DX         // bitrev[7]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X7

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
	MOVB $1, ret+120(FP)
	RET

size8_128_fwd_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 8, complex128, radix-2 variant
// ===========================================================================
TEXT ·inverseAVX2Size8Radix2Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14             // R14 = original dst pointer (for in-place safety)
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
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

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   size8_128_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_128_inv_use_dst
	MOVQ R11, R8

size8_128_inv_use_dst:
	// Bit-reversal
	MOVQ (R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X0
	MOVQ 8(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X1
	MOVQ 16(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X2
	MOVQ 24(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X3
	MOVQ 32(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X4
	MOVQ 40(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X5
	MOVQ 48(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X6
	MOVQ 56(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X7

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
	MOVQ $0x3fc0000000000000, AX // 1/8 = 0.125
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
	MOVB $1, ret+120(FP)
	RET

size8_128_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Forward transform, size 8, complex128, radix-8 variant
// Single radix-8 butterfly without bit-reversal.
// ===========================================================================
TEXT ·forwardAVX2Size8Radix8Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // dst
	MOVQ src+24(FP), R9      // src
	MOVQ twiddle+48(FP), R10 // twiddle
	MOVQ scratch+72(FP), R11 // scratch
	MOVQ bitrev+96(FP), R12  // bitrev
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

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   size8_128_r8_fwd_return_false

	// Build sign masks
	// X14 = maskNegLo = [signbit, 0]
	// X15 = maskNegHi = [0, signbit]
	MOVQ $0x8000000000000000, AX
	VMOVQ AX, X14
	VPERMILPD $1, X14, X15

	// Load input x0..x7 (complex128 = 16 bytes)
	MOVUPD 0(R9), X0
	MOVUPD 16(R9), X1
	MOVUPD 32(R9), X2
	MOVUPD 48(R9), X3
	MOVUPD 64(R9), X4
	MOVUPD 80(R9), X5
	MOVUPD 96(R9), X6
	MOVUPD 112(R9), X7

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
	MOVB $1, ret+120(FP)
	RET

size8_128_r8_fwd_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 8, complex128, radix-8 variant
// Uses conjugated twiddles (via VFMSUBADD) and applies 1/8 scaling.
// ===========================================================================
TEXT ·inverseAVX2Size8Radix8Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
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

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   size8_128_r8_inv_return_false

	// Build sign masks
	MOVQ $0x8000000000000000, AX
	VMOVQ AX, X14            // maskNegLo = [signbit, 0]
	VPERMILPD $1, X14, X15   // maskNegHi = [0, signbit]

	// Load input x0..x7
	MOVUPD 0(R9), X0
	MOVUPD 16(R9), X1
	MOVUPD 32(R9), X2
	MOVUPD 48(R9), X3
	MOVUPD 64(R9), X4
	MOVUPD 80(R9), X5
	MOVUPD 96(R9), X6
	MOVUPD 112(R9), X7

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
	MOVQ $0x3fc0000000000000, AX  // 0.125 = 1/8
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
	MOVB $1, ret+120(FP)
	RET

size8_128_r8_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Forward transform, size 8, complex128, radix-4 (mixed-radix) variant
// ===========================================================================
// Mixed-radix FFT: 1 radix-4 stage + 1 radix-2 stage (2 stages total)
// More efficient than 3-stage radix-2.
//
// Stage 1 (Radix-4): 2 radix-4 butterflies on [x0,x2,x4,x6] and [x1,x3,x5,x7]
// Stage 2 (Radix-2): 4 radix-2 butterflies combining results with twiddles
TEXT ·forwardAVX2Size8Radix4Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ R8, R14             // R14 = original dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 8)

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_128_r4_fwd_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_fwd_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_fwd_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_128_r4_fwd_use_dst
	MOVQ R11, R8             // In-place: use scratch

size8_128_r4_fwd_use_dst:
	// =======================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =======================================================================
	// complex128 is 16 bytes, use SHLQ $4 for indexing
	MOVQ (R12), DX           // bitrev[0]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X0
	MOVQ 8(R12), DX          // bitrev[1]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X1
	MOVQ 16(R12), DX         // bitrev[2]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X2
	MOVQ 24(R12), DX         // bitrev[3]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X3
	MOVQ 32(R12), DX         // bitrev[4]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X4
	MOVQ 40(R12), DX         // bitrev[5]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X5
	MOVQ 48(R12), DX         // bitrev[6]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X6
	MOVQ 56(R12), DX         // bitrev[7]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X7
	// Now: X0=x0, X1=x1, X2=x2, X3=x3, X4=x4, X5=x5, X6=x6, X7=x7

	// =======================================================================
	// STAGE 1: Two radix-4 butterflies
	// =======================================================================
	// Butterfly 1: [x0, x2, x4, x6] -> [a0, a1, a2, a3]
	// Butterfly 2: [x1, x3, x5, x7] -> [a4, a5, a6, a7]

	// Build sign mask for multiply by -i: [0, signbit]
	MOVQ $0x8000000000000000, AX
	VMOVQ AX, X15
	VPERMILPD $1, X15, X14   // X14 = [signbit, 0] for +i
	                         // X15 = [0, signbit] for -i

	// --- Radix-4 Butterfly 1: [x0, x2, x4, x6] ---
	// t0 = x0 + x4, t1 = x0 - x4
	VADDPD X4, X0, X8        // X8 = t0
	VSUBPD X4, X0, X9        // X9 = t1
	// t2 = x2 + x6, t3 = x2 - x6
	VADDPD X6, X2, X10       // X10 = t2
	VSUBPD X6, X2, X11       // X11 = t3

	// a0 = t0 + t2, a2 = t0 - t2
	VADDPD X10, X8, X0       // X0 = a0
	VSUBPD X10, X8, X2       // X2 = a2

	// Multiply t3 by -i: (r,i) -> (i,-r)
	VPERMILPD $1, X11, X12   // X12 = [i, r] (swap)
	VXORPD X15, X12, X12     // X12 = [i, -r] = t3*(-i)

	// a1 = t1 + t3*(-i), a3 = t1 - t3*(-i)
	VADDPD X12, X9, X1       // X1 = a1
	VSUBPD X12, X9, X3       // X3 = a3

	// --- Radix-4 Butterfly 2: [x1, x3, x5, x7] ---
	// t0 = x1 + x5, t1 = x1 - x5
	VADDPD X5, X1, X8        // Wait, X1 now holds a1, need to save it
	// Oops - register conflict. Let me reorganize.
	// Save a0,a1,a2,a3 to scratch first, then do butterfly 2
	MOVUPD X0, (R8)          // work[0] = a0
	MOVUPD X1, 16(R8)        // work[1] = a1
	MOVUPD X2, 32(R8)        // work[2] = a2
	MOVUPD X3, 48(R8)        // work[3] = a3

	// Reload x1,x3,x5,x7 from original bit-reversed positions
	MOVQ 8(R12), DX          // bitrev[1]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X0    // x1
	MOVQ 24(R12), DX         // bitrev[3]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X2    // x3
	MOVQ 40(R12), DX         // bitrev[5]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X4    // x5
	MOVQ 56(R12), DX         // bitrev[7]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X6    // x7

	// t0 = x1 + x5, t1 = x1 - x5
	VADDPD X4, X0, X8        // X8 = t0
	VSUBPD X4, X0, X9        // X9 = t1
	// t2 = x3 + x7, t3 = x3 - x7
	VADDPD X6, X2, X10       // X10 = t2
	VSUBPD X6, X2, X11       // X11 = t3

	// a4 = t0 + t2, a6 = t0 - t2
	VADDPD X10, X8, X4       // X4 = a4
	VSUBPD X10, X8, X6       // X6 = a6

	// Multiply t3 by -i
	VPERMILPD $1, X11, X12
	VXORPD X15, X12, X12     // X12 = t3*(-i)

	// a5 = t1 + t3*(-i), a7 = t1 - t3*(-i)
	VADDPD X12, X9, X5       // X5 = a5
	VSUBPD X12, X9, X7       // X7 = a7

	// =======================================================================
	// STAGE 2: Radix-2 butterflies with twiddle factors
	// =======================================================================
	// Combine: (a0,a4)*w0, (a1,a5)*w1, (a2,a6)*w2, (a3,a7)*w3
	// Reload a0,a1,a2,a3 from work buffer
	MOVUPD (R8), X0          // a0
	MOVUPD 16(R8), X1        // a1
	MOVUPD 32(R8), X2        // a2
	MOVUPD 48(R8), X3        // a3

	// Now: X0=a0, X1=a1, X2=a2, X3=a3, X4=a4, X5=a5, X6=a6, X7=a7

	// --- (a0, a4) with w0=1 ---
	// y0 = a0 + a4, y4 = a0 - a4
	VADDPD X4, X0, X8        // X8 = y0
	VSUBPD X4, X0, X12       // X12 = y4

	// --- (a1, a5) with w1 ---
	MOVUPD 16(R10), X10      // w1
	VMOVDDUP X10, X11        // X11 = [w1.r, w1.r]
	VPERMILPD $1, X10, X10
	VMOVDDUP X10, X10        // X10 = [w1.i, w1.i]
	VPERMILPD $1, X5, X0     // X0 = [a5.i, a5.r]
	VMULPD X10, X0, X0       // X0 = [a5.i*w1.i, a5.r*w1.i]
	VFMADDSUB231PD X11, X5, X0  // X0 = w1 * a5
	VADDPD X0, X1, X9        // X9 = y1
	VSUBPD X0, X1, X13       // X13 = y5

	// --- (a2, a6) with w2 ---
	MOVUPD 32(R10), X10      // w2
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X10
	VMOVDDUP X10, X10
	VPERMILPD $1, X6, X1
	VMULPD X10, X1, X1
	VFMADDSUB231PD X11, X6, X1  // X1 = w2 * a6
	VADDPD X1, X2, X10       // X10 = y2
	VSUBPD X1, X2, X14       // X14 = y6

	// --- (a3, a7) with w3 ---
	MOVUPD 48(R10), X0       // w3
	VMOVDDUP X0, X11
	VPERMILPD $1, X0, X0
	VMOVDDUP X0, X0
	VPERMILPD $1, X7, X2
	VMULPD X0, X2, X2
	VFMADDSUB231PD X11, X7, X2  // X2 = w3 * a7
	VADDPD X2, X3, X11       // X11 = y3
	VSUBPD X2, X3, X15       // X15 = y7

	// =======================================================================
	// Store results
	// =======================================================================
	MOVUPD X8, (R14)         // y0
	MOVUPD X9, 16(R14)       // y1
	MOVUPD X10, 32(R14)      // y2
	MOVUPD X11, 48(R14)      // y3
	MOVUPD X12, 64(R14)      // y4
	MOVUPD X13, 80(R14)      // y5
	MOVUPD X14, 96(R14)      // y6
	MOVUPD X15, 112(R14)     // y7

	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size8_128_r4_fwd_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 8, complex128, radix-4 (mixed-radix) variant
// ===========================================================================
// Uses +i instead of -i, conjugated twiddles, and 1/8 scaling
TEXT ·inverseAVX2Size8Radix4Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_128_r4_inv_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_inv_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   size8_128_r4_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_128_r4_inv_use_dst
	MOVQ R11, R8

size8_128_r4_inv_use_dst:
	// Bit-reversal permutation
	MOVQ (R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X0
	MOVQ 8(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X1
	MOVQ 16(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X2
	MOVQ 24(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X3
	MOVQ 32(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X4
	MOVQ 40(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X5
	MOVQ 48(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X6
	MOVQ 56(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X7

	// Build sign masks
	MOVQ $0x8000000000000000, AX
	VMOVQ AX, X15
	VPERMILPD $1, X15, X14   // X14 = [signbit, 0] for +i
	                         // X15 = [0, signbit] for -i

	// --- Radix-4 Butterfly 1: [x0, x2, x4, x6] with +i ---
	VADDPD X4, X0, X8
	VSUBPD X4, X0, X9
	VADDPD X6, X2, X10
	VSUBPD X6, X2, X11

	VADDPD X10, X8, X0       // a0
	VSUBPD X10, X8, X2       // a2

	// Multiply t3 by +i: (r,i) -> (-i,r)
	VPERMILPD $1, X11, X12
	VXORPD X14, X12, X12     // X12 = [-i, r] = t3*(+i)

	VADDPD X12, X9, X1       // a1
	VSUBPD X12, X9, X3       // a3

	// Save a0,a1,a2,a3
	MOVUPD X0, (R8)
	MOVUPD X1, 16(R8)
	MOVUPD X2, 32(R8)
	MOVUPD X3, 48(R8)

	// Reload x1,x3,x5,x7
	MOVQ 8(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X0
	MOVQ 24(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X2
	MOVQ 40(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X4
	MOVQ 56(R12), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X6

	// --- Radix-4 Butterfly 2: [x1, x3, x5, x7] with +i ---
	VADDPD X4, X0, X8
	VSUBPD X4, X0, X9
	VADDPD X6, X2, X10
	VSUBPD X6, X2, X11

	VADDPD X10, X8, X4       // a4
	VSUBPD X10, X8, X6       // a6

	// Multiply t3 by +i
	VPERMILPD $1, X11, X12
	VXORPD X14, X12, X12

	VADDPD X12, X9, X5       // a5
	VSUBPD X12, X9, X7       // a7

	// Reload a0,a1,a2,a3
	MOVUPD (R8), X0
	MOVUPD 16(R8), X1
	MOVUPD 32(R8), X2
	MOVUPD 48(R8), X3

	// --- Stage 2: Radix-2 with conjugated twiddles (VFMSUBADD) ---

	// (a0, a4) with w0=1
	VADDPD X4, X0, X8        // y0
	VSUBPD X4, X0, X12       // y4

	// (a1, a5) with conj(w1)
	MOVUPD 16(R10), X10
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X10
	VMOVDDUP X10, X10
	VPERMILPD $1, X5, X0
	VMULPD X10, X0, X0
	VFMSUBADD231PD X11, X5, X0  // conj(w1) * a5
	VADDPD X0, X1, X9        // y1
	VSUBPD X0, X1, X13       // y5

	// (a2, a6) with conj(w2)
	MOVUPD 32(R10), X10
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X10
	VMOVDDUP X10, X10
	VPERMILPD $1, X6, X1
	VMULPD X10, X1, X1
	VFMSUBADD231PD X11, X6, X1
	VADDPD X1, X2, X10       // y2
	VSUBPD X1, X2, X14       // y6

	// (a3, a7) with conj(w3)
	MOVUPD 48(R10), X0
	VMOVDDUP X0, X11
	VPERMILPD $1, X0, X0
	VMOVDDUP X0, X0
	VPERMILPD $1, X7, X2
	VMULPD X0, X2, X2
	VFMSUBADD231PD X11, X7, X2
	VADDPD X2, X3, X11       // y3
	VSUBPD X2, X3, X15       // y7

	// Apply 1/8 scaling
	MOVQ $0x3fc0000000000000, AX  // 0.125
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
	MOVB $1, ret+120(FP)
	RET

size8_128_r4_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

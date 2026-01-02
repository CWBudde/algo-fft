//go:build amd64 && fft_asm && !purego

// ===========================================================================
// AVX2 Size-8 Radix-4 FFT (complex64) Kernels for AMD64
// ===========================================================================
//
// This file contains fully-unrolled FFT kernels optimized for size 8.
// These kernels provide better performance than the generic implementation by:
//   - Eliminating loop overhead
//   - Using hardcoded twiddle factor indices
//   - Optimal register allocation for this size
//
// See asm_amd64_avx2_generic.s for algorithm documentation.
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// SIZE 8 KERNELS
// ===========================================================================
// 8-point FFT: 3 stages, 8 complex64 values = 64 bytes = 2 YMM registers
//
// Bit-reversal for n=8: [0, 4, 2, 6, 1, 5, 3, 7]
//
// Stage 1 (size=2): 4 butterflies with pairs (0,1), (2,3), (4,5), (6,7)
//                   All use twiddle[0] = 1+0i (identity)
// Stage 2 (size=4): 2 groups of 2 butterflies
//                   j=0: twiddle[0], j=1: twiddle[2]
// Stage 3 (size=8): 1 group of 4 butterflies
//                   j=0: twiddle[0], j=1: twiddle[1], j=2: twiddle[2], j=3: twiddle[3]
// ===========================================================================

// Forward transform, size 8, complex64, radix-2 variant
// Fully unrolled 3-stage FFT with AVX2 vectorization
TEXT ·ForwardAVX2Size8Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 8)

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_r4_fwd_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_r4_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_r4_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_r4_fwd_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   size8_r4_fwd_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_r4_fwd_use_dst
	MOVQ R11, R8
	JMP  size8_r4_fwd_bitrev

size8_r4_fwd_use_dst:

size8_r4_fwd_bitrev:
	// =======================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =======================================================================
	MOVQ (R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)

	MOVQ 8(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 8(R8)

	MOVQ 16(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 16(R8)

	MOVQ 24(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 24(R8)

	MOVQ 32(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 32(R8)

	MOVQ 40(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 40(R8)

	MOVQ 48(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 48(R8)

	MOVQ 56(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 56(R8)

	// =======================================================================
	// Load all data into YMM registers
	// =======================================================================
	VMOVUPS (R8), Y0         // Y0 = [x0, x1, x2, x3]
	VMOVUPS 32(R8), Y1       // Y1 = [x4, x5, x6, x7]

	// =======================================================================
	// STAGE 1: Radix-4 butterflies (scalar-style, following size-4 pattern)
	// =======================================================================
	// Process two independent radix-4 butterflies:
	// Butterfly 1: [x0, x2, x4, x6] -> [a0, a1, a2, a3]
	// Butterfly 2: [x1, x3, x5, x7] -> [a4, a5, a6, a7]
	//
	// Strategy: Extract pairs, do radix-4 butterfly in XMM registers

	// --- Radix-4 Butterfly 1: [x0, x2, x4, x6] ---
	// Extract x0, x2 from Y0 and x4, x6 from Y1
	VEXTRACTF128 $0, Y0, X2   // X2 = [x0, x1]
	VEXTRACTF128 $1, Y0, X3   // X3 = [x2, x3]
	VEXTRACTF128 $0, Y1, X4   // X4 = [x4, x5]
	VEXTRACTF128 $1, Y1, X5   // X5 = [x6, x7]

	// Build X6 = [x0, x2] and X7 = [x4, x6]
	VUNPCKLPD X3, X2, X6      // X6 = [x0, x2]
	VUNPCKLPD X5, X4, X7      // X7 = [x4, x6]

	// Compute t0 = x0 + x4, t2 = x2 + x6, t1 = x0 - x4, t3 = x2 - x6
	VADDPS X7, X6, X8         // X8 = [t0, t2]
	VSUBPS X7, X6, X9         // X9 = [t1, t3]

	// Duplicate for radix-4 butterfly
	VMOVDDUP X8, X10          // X10 = [t0, t0]
	VPERMILPD $0x3, X8, X11   // X11 = [t2, t2]

	// Compute a0 = t0 + t2, a2 = t0 - t2
	VADDPS X11, X10, X12      // X12 = [a0, a0]
	VSUBPS X11, X10, X13      // X13 = [a2, a2]

	// Duplicate t1, t3
	VMOVDDUP X9, X14          // X14 = [t1, t1]
	VPERMILPD $0x3, X9, X15   // X15 = [t3, t3]

	// Multiply t3 by -i: (r,i) -> (i,-r)
	VSHUFPS $0xB1, X15, X15, X15  // X15 = [i, r] (swap)
	MOVL ·signbit32(SB), AX
	MOVD AX, X10
	VBROADCASTSS X10, X10
	VXORPD X11, X11, X11
	VBLENDPS $0xAA, X10, X11, X10  // X10 = [0, sign, 0, sign]
	VXORPS X10, X15, X15           // X15 = [i, -r] = t3*(-i)

	// Compute a1 = t1 + t3*(-i), a3 = t1 - t3*(-i)
	VADDPS X15, X14, X0       // X0 = [a1, a1]
	VSUBPS X15, X14, X1       // X1 = [a3, a3]

	// Combine butterfly 1 results: [a0, a1, a2, a3]
	VUNPCKLPD X0, X12, X2     // X2 = [a0, a1]
	VUNPCKLPD X1, X13, X3     // X3 = [a2, a3]

	// --- Radix-4 Butterfly 2: [x1, x3, x5, x7] ---
	VEXTRACTF128 $0, Y0, X4   // X4 = [x0, x1]
	VEXTRACTF128 $1, Y0, X5   // X5 = [x2, x3]
	VEXTRACTF128 $0, Y1, X6   // X6 = [x4, x5]
	VEXTRACTF128 $1, Y1, X7   // X7 = [x6, x7]

	// Build X8 = [x1, x3] and X9 = [x5, x7]
	VUNPCKHPD X5, X4, X8      // X8 = [x1, x3]
	VUNPCKHPD X7, X6, X9      // X9 = [x5, x7]

	// Compute t0 = x1 + x5, t2 = x3 + x7, t1 = x1 - x5, t3 = x3 - x7
	VADDPS X9, X8, X10        // X10 = [t0, t2]
	VSUBPS X9, X8, X11        // X11 = [t1, t3]

	// Duplicate for radix-4 butterfly
	VMOVDDUP X10, X12         // X12 = [t0, t0]
	VPERMILPD $0x3, X10, X13  // X13 = [t2, t2]

	// Compute a4 = t0 + t2, a6 = t0 - t2
	VADDPS X13, X12, X14      // X14 = [a4, a4]
	VSUBPS X13, X12, X15      // X15 = [a6, a6]

	// Duplicate t1, t3
	VMOVDDUP X11, X0          // X0 = [t1, t1]
	VPERMILPD $0x3, X11, X1   // X1 = [t3, t3]

	// Multiply t3 by -i
	VSHUFPS $0xB1, X1, X1, X1  // X1 = [i, r]
	MOVL ·signbit32(SB), AX
	MOVD AX, X4
	VBROADCASTSS X4, X4
	VXORPD X5, X5, X5
	VBLENDPS $0xAA, X4, X5, X4  // X4 = [0, sign, 0, sign]
	VXORPS X4, X1, X1           // X1 = [i, -r] = t3*(-i)

	// Compute a5 = t1 + t3*(-i), a7 = t1 - t3*(-i)
	VADDPS X1, X0, X4         // X4 = [a5, a5]
	VSUBPS X1, X0, X5         // X5 = [a7, a7]

	// Combine butterfly 2 results: [a4, a5, a6, a7]
	VUNPCKLPD X4, X14, X6     // X6 = [a4, a5]
	VUNPCKLPD X5, X15, X7     // X7 = [a6, a7]

	// =======================================================================
	// STAGE 2: Radix-2 butterflies with twiddle factors
	// =======================================================================
	// Combine outputs: (a0,a4) w^0, (a1,a5) w^1, (a2,a6) w^2, (a3,a7) w^3
	// Now we have:
	//   X2 = [a0, a1], X3 = [a2, a3]
	//   X6 = [a4, a5], X7 = [a6, a7]

	// Load twiddle factors: w1, w2, w3 (w0=1)
	VMOVSD 8(R10), X8         // X8 = w1
	VMOVSD 16(R10), X9        // X9 = w2
	VMOVSD 24(R10), X10       // X10 = w3

	// Build twiddle vector: [w0, w1] and [w2, w3]
	VMOVSD (R10), X11         // X11 = w0 = 1
	VPUNPCKLQDQ X8, X11, X11  // X11 = [w0, w1]
	VPUNPCKLQDQ X10, X9, X12  // X12 = [w2, w3]

	// Multiply [a4, a5] by [w0, w1]
	VMOVSLDUP X11, X13        // X13 = [w.r, w.r, ...]
	VMOVSHDUP X11, X14        // X14 = [w.i, w.i, ...]
	VSHUFPS $0xB1, X6, X6, X15  // X15 = [a.i, a.r, ...]
	VMULPS X14, X15, X15
	VFMADDSUB231PS X13, X6, X15  // X15 = [w0*a4, w1*a5]

	// Radix-2: y0 = a0 + w0*a4, y1 = a1 + w1*a5
	//          y4 = a0 - w0*a4, y5 = a1 - w1*a5
	VADDPS X15, X2, X0        // X0 = [y0, y1]
	VSUBPS X15, X2, X1        // X1 = [y4, y5]

	// Multiply [a6, a7] by [w2, w3]
	VMOVSLDUP X12, X13
	VMOVSHDUP X12, X14
	VSHUFPS $0xB1, X7, X7, X15
	VMULPS X14, X15, X15
	VFMADDSUB231PS X13, X7, X15  // X15 = [w2*a6, w3*a7]

	// Radix-2: y2 = a2 + w2*a6, y3 = a3 + w3*a7
	//          y6 = a2 - w2*a6, y7 = a3 - w3*a7
	VADDPS X15, X3, X2        // X2 = [y2, y3]
	VSUBPS X15, X3, X3        // X3 = [y6, y7]

	// Combine final results: Y0 = [y0, y1, y2, y3], Y1 = [y4, y5, y6, y7]
	VINSERTF128 $0, X0, Y0, Y0
	VINSERTF128 $1, X2, Y0, Y0
	VINSERTF128 $0, X1, Y1, Y1
	VINSERTF128 $1, X3, Y1, Y1

size8_r4_fwd_store:
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size8_r4_fwd_store_direct

	VMOVUPS Y0, (R9)
	VMOVUPS Y1, 32(R9)
	JMP size8_r4_fwd_done

size8_r4_fwd_store_direct:
	VMOVUPS Y0, (R8)
	VMOVUPS Y1, 32(R8)

size8_r4_fwd_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size8_r4_fwd_return_false:
	MOVB $0, ret+120(FP)
	RET

// Inverse transform, size 8, complex64, radix-4 variant
// Same as forward but with +i instead of -i for radix-4, conjugated twiddles, and 1/8 scaling
TEXT ·InverseAVX2Size8Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_r4_inv_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_r4_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_r4_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_r4_inv_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   size8_r4_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_r4_inv_use_dst
	MOVQ R11, R8
	JMP  size8_r4_inv_bitrev

size8_r4_inv_use_dst:

size8_r4_inv_bitrev:
	// Bit-reversal permutation
	MOVQ (R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)

	MOVQ 8(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 8(R8)

	MOVQ 16(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 16(R8)

	MOVQ 24(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 24(R8)

	MOVQ 32(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 32(R8)

	MOVQ 40(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 40(R8)

	MOVQ 48(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 48(R8)

	MOVQ 56(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 56(R8)

	// Load data
	VMOVUPS (R8), Y0
	VMOVUPS 32(R8), Y1

	// --- Radix-4 Butterfly 1: [x0, x2, x4, x6] (with +i instead of -i) ---
	VEXTRACTF128 $0, Y0, X2
	VEXTRACTF128 $1, Y0, X3
	VEXTRACTF128 $0, Y1, X4
	VEXTRACTF128 $1, Y1, X5

	VUNPCKLPD X3, X2, X6
	VUNPCKLPD X5, X4, X7

	VADDPS X7, X6, X8
	VSUBPS X7, X6, X9

	VMOVDDUP X8, X10
	VPERMILPD $0x3, X8, X11

	VADDPS X11, X10, X12
	VSUBPS X11, X10, X13

	VMOVDDUP X9, X14
	VPERMILPD $0x3, X9, X15

	// Multiply t3 by +i: (r,i) -> (-i,r) for inverse
	VSHUFPS $0xB1, X15, X15, X15  // swap
	MOVL ·signbit32(SB), AX
	MOVD AX, X10
	VBROADCASTSS X10, X10
	VXORPD X11, X11, X11
	VBLENDPS $0x55, X10, X11, X10  // [sign, 0, sign, 0] for real positions
	VXORPS X10, X15, X15           // X15 = [-i, r] = t3*(+i)

	VADDPS X15, X14, X0
	VSUBPS X15, X14, X1

	VUNPCKLPD X0, X12, X2
	VUNPCKLPD X1, X13, X3

	// --- Radix-4 Butterfly 2: [x1, x3, x5, x7] (with +i) ---
	VEXTRACTF128 $0, Y0, X4
	VEXTRACTF128 $1, Y0, X5
	VEXTRACTF128 $0, Y1, X6
	VEXTRACTF128 $1, Y1, X7

	VUNPCKHPD X5, X4, X8
	VUNPCKHPD X7, X6, X9

	VADDPS X9, X8, X10
	VSUBPS X9, X8, X11

	VMOVDDUP X10, X12
	VPERMILPD $0x3, X10, X13

	VADDPS X13, X12, X14
	VSUBPS X13, X12, X15

	VMOVDDUP X11, X0
	VPERMILPD $0x3, X11, X1

	// Multiply by +i for inverse
	VSHUFPS $0xB1, X1, X1, X1
	MOVL ·signbit32(SB), AX
	MOVD AX, X4
	VBROADCASTSS X4, X4
	VXORPD X5, X5, X5
	VBLENDPS $0x55, X4, X5, X4
	VXORPS X4, X1, X1

	VADDPS X1, X0, X4
	VSUBPS X1, X0, X5

	VUNPCKLPD X4, X14, X6
	VUNPCKLPD X5, X15, X7

	// --- Stage 2: Radix-2 with conjugated twiddles (use VFMSUBADD) ---
	// Load twiddles
	VMOVSD 8(R10), X8
	VMOVSD 16(R10), X9
	VMOVSD 24(R10), X10
	VMOVSD (R10), X11
	VPUNPCKLQDQ X8, X11, X11
	VPUNPCKLQDQ X10, X9, X12

	// Multiply with conjugated twiddles using VFMSUBADD
	VMOVSLDUP X11, X13
	VMOVSHDUP X11, X14
	VSHUFPS $0xB1, X6, X6, X15
	VMULPS X14, X15, X15
	VFMSUBADD231PS X13, X6, X15  // conjugate multiply

	VADDPS X15, X2, X0
	VSUBPS X15, X2, X1

	VMOVSLDUP X12, X13
	VMOVSHDUP X12, X14
	VSHUFPS $0xB1, X7, X7, X15
	VMULPS X14, X15, X15
	VFMSUBADD231PS X13, X7, X15  // conjugate multiply

	VADDPS X15, X3, X2
	VSUBPS X15, X3, X3

	// Combine results
	VINSERTF128 $0, X0, Y0, Y0
	VINSERTF128 $1, X2, Y0, Y0
	VINSERTF128 $0, X1, Y1, Y1
	VINSERTF128 $1, X3, Y1, Y1

	// Apply 1/8 scaling
	MOVL ·eighth32(SB), AX       // 0.125f
	MOVD AX, X2
	VBROADCASTSS X2, Y2
	VMULPS Y2, Y0, Y0
	VMULPS Y2, Y1, Y1

	// Store results
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size8_r4_inv_store_direct

	VMOVUPS Y0, (R9)
	VMOVUPS Y1, 32(R9)
	JMP size8_r4_inv_done

size8_r4_inv_store_direct:
	VMOVUPS Y0, (R8)
	VMOVUPS Y1, 32(R8)

size8_r4_inv_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size8_r4_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

// Forward transform, size 8, complex64, radix-8 variant
// Single radix-8 butterfly without bit-reversal.
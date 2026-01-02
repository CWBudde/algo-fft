//go:build amd64 && asm && !purego

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
// 8-point FFT: mixed-radix 4x2 (radix-4 stage, then radix-2), 8 complex64 values
// = 64 bytes = 2 YMM registers
//
// Mixed-radix bit-reversal for n=8: [0, 2, 4, 6, 1, 3, 5, 7]
//
// Stage 1 (radix-4): 2 butterflies on indices [0,2,4,6] and [1,3,5,7]
// Stage 2 (radix-2): 4 butterflies with twiddles w^0, w^1, w^2, w^3
// ===========================================================================

// Forward transform, size 8, complex64, radix-4 (mixed-radix) variant
// Fully unrolled mixed-radix FFT with AVX2 vectorization
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
	// Scalar-style mixed-radix computation (correctness-focused)
	// =======================================================================
	// Build masks for complex ops
	MOVL ·signbit32(SB), AX
	MOVD AX, X16
	VBROADCASTSS X16, X16
	VXORPS X17, X17, X17
	VBLENDPS $0x55, X16, X17, X18 // maskNegReal  (lanes 0,2)
	VBLENDPS $0xAA, X16, X17, X19 // maskNegImag  (lanes 1,3)

	// Load x0..x7 into X0..X7 (lower 64 bits)
	VXORPS X0, X0, X0
	MOVQ (R8), X0
	VXORPS X1, X1, X1
	MOVQ 8(R8), X1
	VXORPS X2, X2, X2
	MOVQ 16(R8), X2
	VXORPS X3, X3, X3
	MOVQ 24(R8), X3
	VXORPS X4, X4, X4
	MOVQ 32(R8), X4
	VXORPS X5, X5, X5
	MOVQ 40(R8), X5
	VXORPS X6, X6, X6
	MOVQ 48(R8), X6
	VXORPS X7, X7, X7
	MOVQ 56(R8), X7

	// Radix-4 butterfly 1: [x0, x1, x2, x3]
	VADDPS X2, X0, X8    // t0
	VSUBPS X2, X0, X9    // t1
	VADDPS X3, X1, X10   // t2
	VSUBPS X3, X1, X11   // t3
	// t3 * (-i)
	VSHUFPS $0xB1, X11, X11, X12
	VXORPS X19, X12, X12
	VADDPS X10, X8, X13  // a0
	VSUBPS X10, X8, X15  // a2
	VADDPS X12, X9, X14  // a1
	VSUBPS X12, X9, X16  // a3

	// Radix-4 butterfly 2: [x4, x5, x6, x7]
	VADDPS X6, X4, X8    // t0
	VSUBPS X6, X4, X9    // t1
	VADDPS X7, X5, X10   // t2
	VSUBPS X7, X5, X11   // t3
	// t3 * (-i)
	VSHUFPS $0xB1, X11, X11, X12
	VXORPS X19, X12, X12
	VADDPS X10, X8, X4   // a4
	VSUBPS X10, X8, X6   // a6
	VADDPS X12, X9, X5   // a5
	VSUBPS X12, X9, X7   // a7

	// Stage 2: radix-2 with twiddles
	// y0,y4
	VADDPS X4, X13, X0   // y0
	VSUBPS X4, X13, X1   // y4

	// w1 * a5
	VXORPS X2, X2, X2
	MOVQ 8(R10), X2      // w1
	VSHUFPS $0x00, X2, X2, X8  // w1.r
	VSHUFPS $0x55, X2, X2, X9  // w1.i
	VSHUFPS $0xB1, X5, X5, X10 // a5 swapped
	VMULPS X8, X5, X11
	VMULPS X9, X10, X12
	VXORPS X18, X12, X12
	VADDPS X12, X11, X11       // w1*a5
	VADDPS X11, X14, X2        // y1
	VSUBPS X11, X14, X3        // y5

	// w2 * a6
	VXORPS X12, X12, X12
	MOVQ 16(R10), X12     // w2
	VSHUFPS $0x00, X12, X12, X8
	VSHUFPS $0x55, X12, X12, X9
	VSHUFPS $0xB1, X6, X6, X10
	VMULPS X8, X6, X11
	VMULPS X9, X10, X12
	VXORPS X18, X12, X12
	VADDPS X12, X11, X11       // w2*a6
	VADDPS X11, X15, X4        // y2
	VSUBPS X11, X15, X5        // y6

	// w3 * a7
	VXORPS X12, X12, X12
	MOVQ 24(R10), X12     // w3
	VSHUFPS $0x00, X12, X12, X8
	VSHUFPS $0x55, X12, X12, X9
	VSHUFPS $0xB1, X7, X7, X10
	VMULPS X8, X7, X11
	VMULPS X9, X10, X12
	VXORPS X18, X12, X12
	VADDPS X12, X11, X11       // w3*a7
	VADDPS X11, X16, X6        // y3
	VSUBPS X11, X16, X7        // y7

	// Store results to work buffer
	MOVQ X0, (R8)
	MOVQ X2, 8(R8)
	MOVQ X4, 16(R8)
	MOVQ X6, 24(R8)
	MOVQ X1, 32(R8)
	MOVQ X3, 40(R8)
	MOVQ X5, 48(R8)
	MOVQ X7, 56(R8)

	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size8_r4_fwd_done

	// Copy to dst
	MOVQ (R8), AX
	MOVQ AX, (R9)
	MOVQ 8(R8), AX
	MOVQ AX, 8(R9)
	MOVQ 16(R8), AX
	MOVQ AX, 16(R9)
	MOVQ 24(R8), AX
	MOVQ AX, 24(R9)
	MOVQ 32(R8), AX
	MOVQ AX, 32(R9)
	MOVQ 40(R8), AX
	MOVQ AX, 40(R9)
	MOVQ 48(R8), AX
	MOVQ AX, 48(R9)
	MOVQ 56(R8), AX
	MOVQ AX, 56(R9)

size8_r4_fwd_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size8_r4_fwd_return_false:
	MOVB $0, ret+120(FP)
	RET

// Inverse transform, size 8, complex64, radix-4 (mixed-radix) variant
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
	// =======================================================================
	// Scalar-style mixed-radix computation (inverse)
	// =======================================================================
	MOVL ·signbit32(SB), AX
	MOVD AX, X16
	VBROADCASTSS X16, X16
	VXORPS X17, X17, X17
	VBLENDPS $0x55, X16, X17, X18 // maskNegReal  (lanes 0,2)
	VBLENDPS $0xAA, X16, X17, X19 // maskNegImag  (lanes 1,3)

	// Load x0..x7
	VXORPS X0, X0, X0
	MOVQ (R8), X0
	VXORPS X1, X1, X1
	MOVQ 8(R8), X1
	VXORPS X2, X2, X2
	MOVQ 16(R8), X2
	VXORPS X3, X3, X3
	MOVQ 24(R8), X3
	VXORPS X4, X4, X4
	MOVQ 32(R8), X4
	VXORPS X5, X5, X5
	MOVQ 40(R8), X5
	VXORPS X6, X6, X6
	MOVQ 48(R8), X6
	VXORPS X7, X7, X7
	MOVQ 56(R8), X7

	// Radix-4 butterfly 1 (+i)
	VADDPS X2, X0, X8
	VSUBPS X2, X0, X9
	VADDPS X3, X1, X10
	VSUBPS X3, X1, X11
	// t3 * (+i)
	VSHUFPS $0xB1, X11, X11, X12
	VXORPS X18, X12, X12
	VADDPS X10, X8, X13  // a0
	VSUBPS X10, X8, X15  // a2
	VADDPS X12, X9, X14  // a1
	VSUBPS X12, X9, X16  // a3

	// Radix-4 butterfly 2 (+i)
	VADDPS X6, X4, X8
	VSUBPS X6, X4, X9
	VADDPS X7, X5, X10
	VSUBPS X7, X5, X11
	VSHUFPS $0xB1, X11, X11, X12
	VXORPS X18, X12, X12
	VADDPS X10, X8, X4   // a4
	VSUBPS X10, X8, X6   // a6
	VADDPS X12, X9, X5   // a5
	VSUBPS X12, X9, X7   // a7

	// Stage 2 with conjugated twiddles
	// y0,y4
	VADDPS X4, X13, X0
	VSUBPS X4, X13, X1

	// conj(w1) * a5
	VXORPS X2, X2, X2
	MOVQ 8(R10), X2
	VXORPS X19, X2, X2        // conjugate
	VSHUFPS $0x00, X2, X2, X8
	VSHUFPS $0x55, X2, X2, X9
	VSHUFPS $0xB1, X5, X5, X10
	VMULPS X8, X5, X11
	VMULPS X9, X10, X12
	VXORPS X18, X12, X12
	VADDPS X12, X11, X11
	VADDPS X11, X14, X2
	VSUBPS X11, X14, X3

	// conj(w2) * a6
	VXORPS X12, X12, X12
	MOVQ 16(R10), X12
	VXORPS X19, X12, X12
	VSHUFPS $0x00, X12, X12, X8
	VSHUFPS $0x55, X12, X12, X9
	VSHUFPS $0xB1, X6, X6, X10
	VMULPS X8, X6, X11
	VMULPS X9, X10, X12
	VXORPS X18, X12, X12
	VADDPS X12, X11, X11
	VADDPS X11, X15, X4
	VSUBPS X11, X15, X5

	// conj(w3) * a7
	VXORPS X12, X12, X12
	MOVQ 24(R10), X12
	VXORPS X19, X12, X12
	VSHUFPS $0x00, X12, X12, X8
	VSHUFPS $0x55, X12, X12, X9
	VSHUFPS $0xB1, X7, X7, X10
	VMULPS X8, X7, X11
	VMULPS X9, X10, X12
	VXORPS X18, X12, X12
	VADDPS X12, X11, X11
	VADDPS X11, X16, X6
	VSUBPS X11, X16, X7

	// Apply 1/8 scaling
	MOVL ·eighth32(SB), AX
	MOVD AX, X8
	VBROADCASTSS X8, X8
	VMULPS X8, X0, X0
	VMULPS X8, X1, X1
	VMULPS X8, X2, X2
	VMULPS X8, X3, X3
	VMULPS X8, X4, X4
	VMULPS X8, X5, X5
	VMULPS X8, X6, X6
	VMULPS X8, X7, X7

	// Store results to work buffer
	MOVQ X0, (R8)
	MOVQ X2, 8(R8)
	MOVQ X4, 16(R8)
	MOVQ X6, 24(R8)
	MOVQ X1, 32(R8)
	MOVQ X3, 40(R8)
	MOVQ X5, 48(R8)
	MOVQ X7, 56(R8)

	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size8_r4_inv_done

	MOVQ (R8), AX
	MOVQ AX, (R9)
	MOVQ 8(R8), AX
	MOVQ AX, 8(R9)
	MOVQ 16(R8), AX
	MOVQ AX, 16(R9)
	MOVQ 24(R8), AX
	MOVQ AX, 24(R9)
	MOVQ 32(R8), AX
	MOVQ AX, 32(R9)
	MOVQ 40(R8), AX
	MOVQ AX, 40(R9)
	MOVQ 48(R8), AX
	MOVQ AX, 48(R9)
	MOVQ 56(R8), AX
	MOVQ AX, 56(R9)

size8_r4_inv_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size8_r4_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

// Forward transform, size 8, complex64, radix-8 variant
// Single radix-8 butterfly without bit-reversal.

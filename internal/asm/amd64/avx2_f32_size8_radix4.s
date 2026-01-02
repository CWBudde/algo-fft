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
	MOVD AX, X8
	VBROADCASTSS X8, X8
	VXORPS X9, X9, X9
	VBLENDPS $0xAA, X8, X9, X10 // maskNegImag  (lanes 1,3)
	VBLENDPS $0x55, X8, X9, X11 // maskNegReal  (lanes 0,2)

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
	VADDPS X2, X0, X12   // t0
	VSUBPS X2, X0, X13   // t1
	VADDPS X3, X1, X14   // t2
	VSUBPS X3, X1, X15   // t3
	// t3 * (-i)
	VSHUFPS $0xB1, X15, X15, X15
	VXORPS X10, X15, X15
	VADDPS X14, X12, X0  // a0
	VSUBPS X14, X12, X2  // a2
	VADDPS X15, X13, X1  // a1
	VSUBPS X15, X13, X3  // a3

	// Radix-4 butterfly 2: [x4, x5, x6, x7]
	VADDPS X6, X4, X12   // t0
	VSUBPS X6, X4, X13   // t1
	VADDPS X7, X5, X14   // t2
	VSUBPS X7, X5, X15   // t3
	// t3 * (-i)
	VSHUFPS $0xB1, X15, X15, X15
	VXORPS X10, X15, X15
	VADDPS X14, X12, X4  // a4
	VSUBPS X14, X12, X6  // a6
	VADDPS X15, X13, X5  // a5
	VSUBPS X15, X13, X7  // a7

	// Stage 2: radix-2 with twiddles
	// y0,y4
	VADDPS X4, X0, X12   // y0
	VSUBPS X4, X0, X13   // y4

	// w1 * a5
	MOVQ 8(R10), X8
	VSHUFPS $0x00, X8, X8, X9  // w1.r
	VSHUFPS $0x55, X8, X8, X10 // w1.i
	VSHUFPS $0xB1, X5, X5, X11 // a5 swapped
	VMULPS X9, X5, X14
	VMULPS X10, X11, X15
	MOVL ·signbit32(SB), AX
	MOVD AX, X9
	VBROADCASTSS X9, X9
	VXORPS X10, X10, X10
	VBLENDPS $0x55, X9, X10, X9
	VXORPS X9, X15, X15
	VADDPS X15, X14, X14       // w1*a5
	VADDPS X14, X1, X8         // y1
	VSUBPS X14, X1, X9         // y5

	// w2 * a6
	MOVQ 16(R10), X10
	VSHUFPS $0x00, X10, X10, X11
	VSHUFPS $0x55, X10, X10, X14
	VSHUFPS $0xB1, X6, X6, X15
	VMULPS X11, X6, X10
	VMULPS X14, X15, X11
	MOVL ·signbit32(SB), AX
	MOVD AX, X14
	VBROADCASTSS X14, X14
	VXORPS X15, X15, X15
	VBLENDPS $0x55, X14, X15, X14
	VXORPS X14, X11, X11
	VADDPS X11, X10, X14       // w2*a6
	VADDPS X14, X2, X10        // y2
	VSUBPS X14, X2, X11        // y6

	// w3 * a7
	MOVQ 24(R10), X14
	VSHUFPS $0x00, X14, X14, X15
	VSHUFPS $0x55, X14, X14, X4
	VSHUFPS $0xB1, X7, X7, X5
	VMULPS X15, X7, X14
	VMULPS X4, X5, X15
	MOVL ·signbit32(SB), AX
	MOVD AX, X4
	VBROADCASTSS X4, X4
	VXORPS X5, X5, X5
	VBLENDPS $0x55, X4, X5, X4
	VXORPS X4, X15, X15
	VADDPS X15, X14, X14       // w3*a7
	VADDPS X14, X3, X15        // y3
	VSUBPS X14, X3, X4         // y7

	// Store results to work buffer
	MOVQ X12, (R8)
	MOVQ X8, 8(R8)
	MOVQ X10, 16(R8)
	MOVQ X15, 24(R8)
	MOVQ X13, 32(R8)
	MOVQ X9, 40(R8)
	MOVQ X11, 48(R8)
	MOVQ X4, 56(R8)

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
	MOVD AX, X8
	VBROADCASTSS X8, X8
	VXORPS X9, X9, X9
	VBLENDPS $0xAA, X8, X9, X10 // maskNegImag  (lanes 1,3)
	VBLENDPS $0x55, X8, X9, X11 // maskNegReal  (lanes 0,2)

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
	VADDPS X2, X0, X12
	VSUBPS X2, X0, X13
	VADDPS X3, X1, X14
	VSUBPS X3, X1, X15
	// t3 * (+i)
	VSHUFPS $0xB1, X15, X15, X15
	VXORPS X11, X15, X15
	VADDPS X14, X12, X0  // a0
	VSUBPS X14, X12, X2  // a2
	VADDPS X15, X13, X1  // a1
	VSUBPS X15, X13, X3  // a3

	// Radix-4 butterfly 2 (+i)
	VADDPS X6, X4, X12
	VSUBPS X6, X4, X13
	VADDPS X7, X5, X14
	VSUBPS X7, X5, X15
	VSHUFPS $0xB1, X15, X15, X15
	VXORPS X11, X15, X15
	VADDPS X14, X12, X4  // a4
	VSUBPS X14, X12, X6  // a6
	VADDPS X15, X13, X5  // a5
	VSUBPS X15, X13, X7  // a7

	// Stage 2 with conjugated twiddles
	// y0,y4
	VADDPS X4, X0, X12
	VSUBPS X4, X0, X13

	// conj(w1) * a5
	MOVQ 8(R10), X8
	VXORPS X10, X8, X8
	VSHUFPS $0x00, X8, X8, X14
	VSHUFPS $0x55, X8, X8, X15
	VSHUFPS $0xB1, X5, X5, X8
	VMULPS X14, X5, X9
	VMULPS X15, X8, X14
	VXORPS X11, X14, X14
	VADDPS X14, X9, X9        // conj(w1)*a5
	VADDPS X9, X1, X8         // y1
	VSUBPS X9, X1, X9         // y5

	// conj(w2) * a6
	MOVQ 16(R10), X14
	VXORPS X10, X14, X14
	VSHUFPS $0x00, X14, X14, X15
	VSHUFPS $0x55, X14, X14, X1
	VSHUFPS $0xB1, X6, X6, X14
	VMULPS X15, X6, X5
	VMULPS X1, X14, X15
	VXORPS X11, X15, X15
	VADDPS X15, X5, X14       // conj(w2)*a6
	VADDPS X14, X2, X10       // y2
	VSUBPS X14, X2, X11       // y6

	// conj(w3) * a7
	MOVQ 24(R10), X1
	VXORPS X10, X1, X1
	VSHUFPS $0x00, X1, X1, X2
	VSHUFPS $0x55, X1, X1, X6
	VSHUFPS $0xB1, X7, X7, X1
	VMULPS X2, X7, X4
	VMULPS X6, X1, X5
	VXORPS X11, X5, X5
	VADDPS X5, X4, X14        // conj(w3)*a7
	VADDPS X14, X3, X15       // y3
	VSUBPS X14, X3, X4        // y7

	// Apply 1/8 scaling
	MOVL ·eighth32(SB), AX
	MOVD AX, X2
	VBROADCASTSS X2, X2
	VMULPS X2, X12, X12
	VMULPS X2, X8, X8
	VMULPS X2, X10, X10
	VMULPS X2, X15, X15
	VMULPS X2, X13, X13
	VMULPS X2, X9, X9
	VMULPS X2, X11, X11
	VMULPS X2, X4, X4

	// Store results to work buffer
	MOVQ X12, (R8)
	MOVQ X8, 8(R8)
	MOVQ X10, 16(R8)
	MOVQ X15, 24(R8)
	MOVQ X13, 32(R8)
	MOVQ X9, 40(R8)
	MOVQ X11, 48(R8)
	MOVQ X4, 56(R8)

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

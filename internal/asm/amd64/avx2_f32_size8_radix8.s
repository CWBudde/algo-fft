//go:build amd64 && fft_asm && !purego

// ===========================================================================
// AVX2 Size-8 Radix-8 FFT (complex64) Kernels for AMD64
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
TEXT ·ForwardAVX2Size8Radix8Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_r8_fwd_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_r8_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_r8_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_r8_fwd_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   size8_r8_fwd_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_r8_fwd_use_dst
	MOVQ R11, R8

size8_r8_fwd_use_dst:
	// Load input
	VMOVUPS (R9), Y0
	VMOVUPS 32(R9), Y1

	// a0..a7
	VADDPS Y1, Y0, Y2        // Y2 = [a0, a4, a2, a6]
	VSUBPS Y1, Y0, Y3        // Y3 = [a1, a5, a3, a7]

	VEXTRACTF128 $0, Y2, X0  // X0 = [a0, a4]
	VEXTRACTF128 $1, Y2, X1  // X1 = [a2, a6]
	VEXTRACTF128 $0, Y3, X2  // X2 = [a1, a5]
	VEXTRACTF128 $1, Y3, X3  // X3 = [a3, a7]

	VUNPCKLPD X1, X0, X4     // X4 = [a0, a2]
	VUNPCKLPD X3, X2, X5     // X5 = [a1, a3]
	VUNPCKHPD X1, X0, X6     // X6 = [a4, a6]
	VUNPCKHPD X3, X2, X7     // X7 = [a5, a7]

	// Build mask for multiply by -i: [0, sign, 0, sign]
	MOVL ·signbit32(SB), AX
	MOVD AX, X9
	VBROADCASTSS X9, X9
	VXORPD X0, X0, X0
	VBLENDPS $0xAA, X9, X0, X9

	// Even terms
	VMOVDDUP X4, X8
	VPERMILPD $0x3, X4, X10
	VADDPS X10, X8, X11      // e0
	VSUBPS X10, X8, X12      // e2

	VMOVDDUP X5, X13
	VPERMILPD $0x3, X5, X14
	VSHUFPS $0xB1, X14, X14, X14
	VXORPS X9, X14, X14      // a3 * (-i)
	VADDPS X14, X13, X15     // e1
	VSUBPS X14, X13, X2      // e3

	// Odd terms
	VMOVDDUP X6, X8
	VPERMILPD $0x3, X6, X10
	VADDPS X10, X8, X13      // o0
	VSUBPS X10, X8, X14      // o2

	VMOVDDUP X7, X8
	VPERMILPD $0x3, X7, X10
	VSHUFPS $0xB1, X10, X10, X10
	VXORPS X9, X10, X10      // a7 * (-i)
	VADDPS X10, X8, X0       // o1
	VSUBPS X10, X8, X1       // o3

	// Pack even/odd pairs
	VUNPCKLPD X0, X13, X3    // [o0, o1]
	VUNPCKLPD X1, X14, X4    // [o2, o3]
	VUNPCKLPD X15, X11, X5   // [e0, e1]
	VUNPCKLPD X2, X12, X6    // [e2, e3]

	// Load twiddles
	VMOVSD (R10), X7         // w0
	VMOVSD 8(R10), X8        // w1
	VMOVSD 16(R10), X9       // w2
	VMOVSD 24(R10), X10      // w3
	VPUNPCKLQDQ X8, X7, X7   // [w0, w1]
	VPUNPCKLQDQ X10, X9, X8  // [w2, w3]

	// t01 = w * [o0, o1]
	VMOVSLDUP X7, X9
	VMOVSHDUP X7, X10
	VSHUFPS $0xB1, X3, X3, X0
	VMULPS X10, X0, X0
	VFMADDSUB231PS X9, X3, X0
	VADDPS X0, X5, X11       // [y0, y1]
	VSUBPS X0, X5, X12       // [y4, y5]

	// t23 = w * [o2, o3]
	VMOVSLDUP X8, X9
	VMOVSHDUP X8, X10
	VSHUFPS $0xB1, X4, X4, X1
	VMULPS X10, X1, X1
	VFMADDSUB231PS X9, X4, X1
	VADDPS X1, X6, X13       // [y2, y3]
	VSUBPS X1, X6, X14       // [y6, y7]

	// Assemble output
	VINSERTF128 $0, X11, Y0, Y0
	VINSERTF128 $1, X13, Y0, Y0
	VINSERTF128 $0, X12, Y1, Y1
	VINSERTF128 $1, X14, Y1, Y1

	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size8_r8_fwd_store_direct

	VMOVUPS Y0, (R9)
	VMOVUPS Y1, 32(R9)
	JMP size8_r8_fwd_done

size8_r8_fwd_store_direct:
	VMOVUPS Y0, (R8)
	VMOVUPS Y1, 32(R8)

size8_r8_fwd_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size8_r8_fwd_return_false:
	MOVB $0, ret+120(FP)
	RET

// Inverse transform, size 8, complex64, radix-8 variant
// Uses conjugated twiddles and applies 1/8 scaling.
TEXT ·InverseAVX2Size8Radix8Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_r8_inv_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_r8_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_r8_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_r8_inv_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   size8_r8_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_r8_inv_use_dst
	MOVQ R11, R8

size8_r8_inv_use_dst:
	// Load input
	VMOVUPS (R9), Y0
	VMOVUPS 32(R9), Y1

	// a0..a7
	VADDPS Y1, Y0, Y2
	VSUBPS Y1, Y0, Y3

	VEXTRACTF128 $0, Y2, X0
	VEXTRACTF128 $1, Y2, X1
	VEXTRACTF128 $0, Y3, X2
	VEXTRACTF128 $1, Y3, X3

	VUNPCKLPD X1, X0, X4
	VUNPCKLPD X3, X2, X5
	VUNPCKHPD X1, X0, X6
	VUNPCKHPD X3, X2, X7

	// Build mask for multiply by +i: [sign, 0, sign, 0]
	MOVL ·signbit32(SB), AX
	MOVD AX, X9
	VBROADCASTSS X9, X9
	VXORPD X0, X0, X0
	VBLENDPS $0x55, X9, X0, X9

	// Even terms
	VMOVDDUP X4, X8
	VPERMILPD $0x3, X4, X10
	VADDPS X10, X8, X11      // e0
	VSUBPS X10, X8, X12      // e2

	VMOVDDUP X5, X13
	VPERMILPD $0x3, X5, X14
	VSHUFPS $0xB1, X14, X14, X14
	VXORPS X9, X14, X14      // a3 * (+i)
	VADDPS X14, X13, X15     // e1
	VSUBPS X14, X13, X2      // e3

	// Odd terms
	VMOVDDUP X6, X8
	VPERMILPD $0x3, X6, X10
	VADDPS X10, X8, X13      // o0
	VSUBPS X10, X8, X14      // o2

	VMOVDDUP X7, X8
	VPERMILPD $0x3, X7, X10
	VSHUFPS $0xB1, X10, X10, X10
	VXORPS X9, X10, X10      // a7 * (+i)
	VADDPS X10, X8, X0       // o1
	VSUBPS X10, X8, X1       // o3

	// Pack even/odd pairs
	VUNPCKLPD X0, X13, X3    // [o0, o1]
	VUNPCKLPD X1, X14, X4    // [o2, o3]
	VUNPCKLPD X15, X11, X5   // [e0, e1]
	VUNPCKLPD X2, X12, X6    // [e2, e3]

	// Load twiddles
	VMOVSD (R10), X7
	VMOVSD 8(R10), X8
	VMOVSD 16(R10), X9
	VMOVSD 24(R10), X10
	VPUNPCKLQDQ X8, X7, X7
	VPUNPCKLQDQ X10, X9, X8

	// t01 = conj(w) * [o0, o1]
	VMOVSLDUP X7, X9
	VMOVSHDUP X7, X10
	VSHUFPS $0xB1, X3, X3, X0
	VMULPS X10, X0, X0
	VFMSUBADD231PS X9, X3, X0
	VADDPS X0, X5, X11       // [y0, y1]
	VSUBPS X0, X5, X12       // [y4, y5]

	// t23 = conj(w) * [o2, o3]
	VMOVSLDUP X8, X9
	VMOVSHDUP X8, X10
	VSHUFPS $0xB1, X4, X4, X1
	VMULPS X10, X1, X1
	VFMSUBADD231PS X9, X4, X1
	VADDPS X1, X6, X13       // [y2, y3]
	VSUBPS X1, X6, X14       // [y6, y7]

	// Assemble output
	VINSERTF128 $0, X11, Y0, Y0
	VINSERTF128 $1, X13, Y0, Y0
	VINSERTF128 $0, X12, Y1, Y1
	VINSERTF128 $1, X14, Y1, Y1

	// Apply 1/8 scaling
	MOVL ·eighth32(SB), AX
	MOVD AX, X2
	VBROADCASTSS X2, Y2
	VMULPS Y2, Y0, Y0
	VMULPS Y2, Y1, Y1

	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size8_r8_inv_store_direct

	VMOVUPS Y0, (R9)
	VMOVUPS Y1, 32(R9)
	JMP size8_r8_inv_done

size8_r8_inv_store_direct:
	VMOVUPS Y0, (R8)
	VMOVUPS Y1, 32(R8)

size8_r8_inv_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size8_r8_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

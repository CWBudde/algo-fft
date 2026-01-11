//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-16 FFT Kernels for AMD64
// ===========================================================================
//
// This file contains fully-unrolled FFT kernels optimized for size 16.
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
// SIZE 16 KERNELS
// ===========================================================================
// Forward transform, size 16, complex64
// Fully unrolled 4-stage FFT with AVX2 vectorization
//
// This kernel implements a complete radix-2 DIT FFT for exactly 16 complex64 values.
// All 4 stages are fully unrolled with hardcoded twiddle factor indices:
//   Stage 1 (size=2):  8 butterflies, step=8, twiddle index 0 for all
//   Stage 2 (size=4):  8 butterflies in 2 groups, step=4, twiddle indices [0,4]
//   Stage 3 (size=8):  8 butterflies in 1 group, step=2, twiddle indices [0,2,4,6]
//   Stage 4 (size=16): 8 butterflies, step=1, twiddle indices [0,1,2,3,4,5,6,7]
//
// Bit-reversal permutation indices for n=16:
//   [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
//
// Register allocation:
//   R8:  work buffer (dst or scratch for in-place)
//   R9:  src pointer
//   R10: twiddle pointer
//   R11: scratch pointer
//   Y0-Y3: data registers for butterflies
//   Y4-Y7: twiddle and intermediate values
//
TEXT ·ForwardAVX2Size16Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 16)

	// Verify n == 16
	CMPQ R13, $16
	JNE  size16_return_false

	// Validate all slice lengths >= 16
	MOVQ dst+8(FP), AX
	CMPQ AX, $16
	JL   size16_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $16
	JL   size16_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $16
	JL   size16_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size16_use_dst

	// In-place: use scratch
	MOVQ R11, R8
	JMP  size16_bitrev

size16_use_dst:
	// Out-of-place: use dst

size16_bitrev:
	// =======================================================================
	// Bit-reversal + STAGE 1 (fused, identity twiddles)
	// =======================================================================
	// Load bit-reversed data directly into YMM registers and compute Stage 1
	// butterflies in one pass. For size 16:
	//   bitrev = [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
	// Stage 1 pairs adjacent elements: (0,1), (2,3), ... with twiddle[0]=1+0i
	// Result: a' = a + b, b' = a - b (no complex multiply needed)

	// Y0: Load [src[0], src[8], src[4], src[12]] -> butterflies (0,8), (4,12)
	VMOVSD 0(R9), X0         // X0 = src[0]
	VMOVSD 64(R9), X4        // X4 = src[8]
	VPUNPCKLQDQ X4, X0, X0   // X0 = [src[0], src[8]]
	VMOVSD 32(R9), X5        // X5 = src[4]
	VMOVSD 96(R9), X6        // X6 = src[12]
	VPUNPCKLQDQ X6, X5, X5   // X5 = [src[4], src[12]]
	VINSERTF128 $1, X5, Y0, Y0  // Y0 = [src[0], src[8], src[4], src[12]]

	// Stage 1 butterfly for Y0
	VPERMILPD $0x05, Y0, Y4  // Y4 = [src[8], src[0], src[12], src[4]]
	VADDPS Y4, Y0, Y5        // Y5 = [0+8, 8+0, 4+12, 12+4]
	VSUBPS Y0, Y4, Y6        // Y6 = [8-0, 0-8, 12-4, 4-12]
	VBLENDPD $0x0A, Y6, Y5, Y0  // Y0 = [0+8, 0-8, 4+12, 4-12]

	// Y1: Load [src[2], src[10], src[6], src[14]] -> butterflies (2,10), (6,14)
	VMOVSD 16(R9), X1        // X1 = src[2]
	VMOVSD 80(R9), X4        // X4 = src[10]
	VPUNPCKLQDQ X4, X1, X1   // X1 = [src[2], src[10]]
	VMOVSD 48(R9), X5        // X5 = src[6]
	VMOVSD 112(R9), X6       // X6 = src[14]
	VPUNPCKLQDQ X6, X5, X5   // X5 = [src[6], src[14]]
	VINSERTF128 $1, X5, Y1, Y1  // Y1 = [src[2], src[10], src[6], src[14]]

	// Stage 1 butterfly for Y1
	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1

	// Y2: Load [src[1], src[9], src[5], src[13]] -> butterflies (1,9), (5,13)
	VMOVSD 8(R9), X2         // X2 = src[1]
	VMOVSD 72(R9), X4        // X4 = src[9]
	VPUNPCKLQDQ X4, X2, X2   // X2 = [src[1], src[9]]
	VMOVSD 40(R9), X5        // X5 = src[5]
	VMOVSD 104(R9), X6       // X6 = src[13]
	VPUNPCKLQDQ X6, X5, X5   // X5 = [src[5], src[13]]
	VINSERTF128 $1, X5, Y2, Y2  // Y2 = [src[1], src[9], src[5], src[13]]

	// Stage 1 butterfly for Y2
	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y2, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y2

	// Y3: Load [src[3], src[11], src[7], src[15]] -> butterflies (3,11), (7,15)
	VMOVSD 24(R9), X3        // X3 = src[3]
	VMOVSD 88(R9), X4        // X4 = src[11]
	VPUNPCKLQDQ X4, X3, X3   // X3 = [src[3], src[11]]
	VMOVSD 56(R9), X5        // X5 = src[7]
	VMOVSD 120(R9), X6       // X6 = src[15]
	VPUNPCKLQDQ X6, X5, X5   // X5 = [src[7], src[15]]
	VINSERTF128 $1, X5, Y3, Y3  // Y3 = [src[3], src[11], src[7], src[15]]

	// Stage 1 butterfly for Y3
	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y3, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y3

	// =======================================================================
	// STAGE 2: size=4, half=2, step=4
	// =======================================================================
	// 4 groups of 2 butterflies: (0,2), (1,3), (4,6), (5,7), ...
	// Twiddle factors: j=0 uses twiddle[0], j=1 uses twiddle[4]
	// twiddle[0] = (1, 0), twiddle[4] = (0, -1) for n=16
	//
	// After stage 1, Y0-Y3 contain the data. Now pairs are 2 apart.
	// Y0 = [d0, d1, d2, d3] where we need butterflies (d0,d2) and (d1,d3)
	//
	// Reorganize: we need to pair elements that are 2 positions apart
	// Use VPERM2F128 and VSHUFPS to rearrange

	// For each Y register, extract 'a' values (positions 0,1) and 'b' values (positions 2,3)
	// Y0 = [d0, d1, d2, d3]
	// a = [d0, d1], b = [d2, d3]
	// Butterfly: a' = a + w*b, b' = a - w*b where w = twiddle[j*step]

	// Load twiddle factors for stage 2
	// twiddle[0] = exp(0) = (1, 0)
	// twiddle[4] = exp(-2πi*4/16) = exp(-πi/2) = (0, -1)
	// Packed: [tw0, tw4] = [(1,0), (0,-1)]
	VMOVSD (R10), X4         // X4 = twiddle[0] = (1, 0)
	VMOVSD 32(R10), X5       // X5 = twiddle[4] = (0, -1)
	VPUNPCKLQDQ X5, X4, X4   // X4 = [tw0, tw4]
	VINSERTF128 $1, X4, Y4, Y4  // Y4 = [tw0, tw4, tw0, tw4]

	// Pre-split twiddle into real and imag parts (reused for all 4 registers)
	VMOVSLDUP Y4, Y14        // Y14 = [w.r, w.r, ...] (broadcast real parts)
	VMOVSHDUP Y4, Y15        // Y15 = [w.i, w.i, ...] (broadcast imag parts)

	// Y0 = [d0, d1, d2, d3]
	// Extract a = [d0, d1, d0, d1] (low 128 bits duplicated)
	// Extract b = [d2, d3, d2, d3] (high 128 bits duplicated)
	VPERM2F128 $0x00, Y0, Y0, Y5  // Y5 = [d0, d1, d0, d1] (low lane to both)
	VPERM2F128 $0x11, Y0, Y0, Y6  // Y6 = [d2, d3, d2, d3] (high lane to both)

	// Complex multiply: t = w * b (Y4 * Y6)
	VSHUFPS $0xB1, Y6, Y6, Y9  // Y9 = b_swapped = [b.i, b.r, ...]
	VMULPS Y15, Y9, Y9       // Y9 = [b.i*w.i, b.r*w.i, ...]
	VFMADDSUB231PS Y14, Y6, Y9  // Y9 = t = w * b

	// Butterfly: a' = a + t, b' = a - t
	VADDPS Y9, Y5, Y7        // Y7 = a + t
	VSUBPS Y9, Y5, Y8        // Y8 = a - t

	// Recombine: Y0 = [a'0, a'1, b'0, b'1]
	VINSERTF128 $1, X8, Y7, Y0  // Y0 = [a'0, a'1, b'0, b'1]

	// Y1 = [d4, d5, d6, d7] - reuse Y14, Y15
	VPERM2F128 $0x00, Y1, Y1, Y5
	VPERM2F128 $0x11, Y1, Y1, Y6
	VSHUFPS $0xB1, Y6, Y6, Y9
	VMULPS Y15, Y9, Y9
	VFMADDSUB231PS Y14, Y6, Y9
	VADDPS Y9, Y5, Y7
	VSUBPS Y9, Y5, Y8
	VINSERTF128 $1, X8, Y7, Y1

	// Y2 = [d8, d9, d10, d11] - reuse Y14, Y15
	VPERM2F128 $0x00, Y2, Y2, Y5
	VPERM2F128 $0x11, Y2, Y2, Y6
	VSHUFPS $0xB1, Y6, Y6, Y9
	VMULPS Y15, Y9, Y9
	VFMADDSUB231PS Y14, Y6, Y9
	VADDPS Y9, Y5, Y7
	VSUBPS Y9, Y5, Y8
	VINSERTF128 $1, X8, Y7, Y2

	// Y3 = [d12, d13, d14, d15] - reuse Y14, Y15
	VPERM2F128 $0x00, Y3, Y3, Y5
	VPERM2F128 $0x11, Y3, Y3, Y6
	VSHUFPS $0xB1, Y6, Y6, Y9
	VMULPS Y15, Y9, Y9
	VFMADDSUB231PS Y14, Y6, Y9
	VADDPS Y9, Y5, Y7
	VSUBPS Y9, Y5, Y8
	VINSERTF128 $1, X8, Y7, Y3

	// =======================================================================
	// STAGE 3: size=8, half=4, step=2
	// =======================================================================
	// 2 groups of 4 butterflies: indices 0-3 with 4-7, indices 8-11 with 12-15
	// Twiddle factors: twiddle[0], twiddle[2], twiddle[4], twiddle[6]
	// twiddle[0] = (1, 0)
	// twiddle[2] = (cos(-π/4), sin(-π/4)) ≈ (0.707, -0.707)
	// twiddle[4] = (0, -1)
	// twiddle[6] = (-0.707, -0.707)

	// Load twiddle factors for stage 3
	VMOVSD (R10), X4         // twiddle[0]
	VMOVSD 16(R10), X5       // twiddle[2]
	VPUNPCKLQDQ X5, X4, X4   // X4 = [tw0, tw2]
	VMOVSD 32(R10), X5       // twiddle[4]
	VMOVSD 48(R10), X6       // twiddle[6]
	VPUNPCKLQDQ X6, X5, X5   // X5 = [tw4, tw6]
	VINSERTF128 $1, X5, Y4, Y4  // Y4 = [tw0, tw2, tw4, tw6]

	// Pre-split twiddle (used for both groups)
	VMOVSLDUP Y4, Y14        // Y14 = [w.r, w.r, ...]
	VMOVSHDUP Y4, Y15        // Y15 = [w.i, w.i, ...]

	// Group 1: Y0 (indices 0-3) with Y1 (indices 4-7)
	// a = Y0 = [d0, d1, d2, d3]
	// b = Y1 = [d4, d5, d6, d7]
	// t = w * b, a' = a + t, b' = a - t
	VSHUFPS $0xB1, Y1, Y1, Y7  // Y7 = b_swapped
	VMULPS Y15, Y7, Y7       // Y7 = b_swap * w.i
	VFMADDSUB231PS Y14, Y1, Y7  // Y7 = t = w * b

	VADDPS Y7, Y0, Y8        // Y8 = a + t = new indices 0-3
	VSUBPS Y7, Y0, Y9        // Y9 = a - t = new indices 4-7

	// Group 2: Y2 (indices 8-11) with Y3 (indices 12-15)
	VSHUFPS $0xB1, Y3, Y3, Y7
	VMULPS Y15, Y7, Y7
	VFMADDSUB231PS Y14, Y3, Y7

	VADDPS Y7, Y2, Y10       // Y10 = new indices 8-11
	VSUBPS Y7, Y2, Y11       // Y11 = new indices 12-15

	// Results are now:
	// Y8 = indices 0-3, Y9 = indices 4-7, Y10 = indices 8-11, Y11 = indices 12-15

	// =======================================================================
	// STAGE 4: size=16, half=8, step=1
	// =======================================================================
	// 8 butterflies: index i with index i+8, for i=0..7
	// Twiddle factors: twiddle[0..7]

	// Load twiddle factors for stage 4
	VMOVUPS (R10), Y4        // Y4 = [tw0, tw1, tw2, tw3]
	VMOVUPS 32(R10), Y5      // Y5 = [tw4, tw5, tw6, tw7]

	// Group 1: Y8 (indices 0-3) with Y10 (indices 8-11) using Y4 (tw0-3)
	VMOVSLDUP Y4, Y6         // Y6 = [w.r broadcast]
	VMOVSHDUP Y4, Y7         // Y7 = [w.i broadcast]
	VSHUFPS $0xB1, Y10, Y10, Y0  // Y0 = b_swapped
	VMULPS Y7, Y0, Y0        // Y0 = b_swap * w.i
	VFMADDSUB231PS Y6, Y10, Y0  // Y0 = t = w * b

	VADDPS Y0, Y8, Y12       // Y12 = a' (final indices 0-3)
	VSUBPS Y0, Y8, Y13       // Y13 = b' (final indices 8-11)

	// Group 2: Y9 (indices 4-7) with Y11 (indices 12-15) using Y5 (tw4-7)
	VMOVSLDUP Y5, Y6
	VMOVSHDUP Y5, Y7
	VSHUFPS $0xB1, Y11, Y11, Y0
	VMULPS Y7, Y0, Y0
	VFMADDSUB231PS Y6, Y11, Y0

	VADDPS Y0, Y9, Y10       // Y10 = a' (final indices 4-7)
	VSUBPS Y0, Y9, Y11       // Y11 = b' (final indices 12-15)

	// Store final results to dst (not work buffer!)
	// Final register allocation:
	// Y12 = indices 0-3, Y10 = indices 4-7, Y13 = indices 8-11, Y11 = indices 12-15
	MOVQ dst+0(FP), R9       // R9 = dst pointer
	VMOVUPS Y12, (R9)        // dst[0-3]
	VMOVUPS Y10, 32(R9)      // dst[4-7]
	VMOVUPS Y13, 64(R9)      // dst[8-11]
	VMOVUPS Y11, 96(R9)      // dst[12-15]

	VZEROUPPER
	MOVB $1, ret+96(FP)      // Return true (success)
	RET

size16_return_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 16, complex64
// ===========================================================================
// Same as forward but with conjugated twiddle factors and 1/n scaling.
//
// Optimization notes:
// - Stage 1 uses identity twiddle (1+0i), so conjugation has no effect
// - Conjugation is done via VFMSUBADD instead of VFMADDSUB, which naturally
//   produces the conjugate multiply result without explicit sign negation
// - Twiddle factor real/imag splits are hoisted and reused
// - 1/16 = 0.0625 scaling applied at the end
//
TEXT ·InverseAVX2Size16Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 16)

	// Verify n == 16
	CMPQ R13, $16
	JNE  size16_inv_return_false

	// Validate all slice lengths >= 16
	MOVQ dst+8(FP), AX
	CMPQ AX, $16
	JL   size16_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $16
	JL   size16_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $16
	JL   size16_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size16_inv_use_dst

	// In-place: use scratch
	MOVQ R11, R8
	JMP  size16_inv_bitrev

size16_inv_use_dst:
	// Out-of-place: use dst

size16_inv_bitrev:
	// =======================================================================
	// Bit-reversal + STAGE 1 (fused, identity twiddles)
	// =======================================================================
	// Load bit-reversed data directly into YMM registers and compute Stage 1
	// butterflies in one pass. Conjugation has no effect on identity twiddle.

	// Y0: Load [src[0], src[8], src[4], src[12]] -> butterflies (0,8), (4,12)
	VMOVSD 0(R9), X0
	VMOVSD 64(R9), X4
	VPUNPCKLQDQ X4, X0, X0
	VMOVSD 32(R9), X5
	VMOVSD 96(R9), X6
	VPUNPCKLQDQ X6, X5, X5
	VINSERTF128 $1, X5, Y0, Y0

	// Stage 1 butterfly for Y0
	VPERMILPD $0x05, Y0, Y4
	VADDPS Y4, Y0, Y5
	VSUBPS Y0, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y0

	// Y1: Load [src[2], src[10], src[6], src[14]] -> butterflies (2,10), (6,14)
	VMOVSD 16(R9), X1
	VMOVSD 80(R9), X4
	VPUNPCKLQDQ X4, X1, X1
	VMOVSD 48(R9), X5
	VMOVSD 112(R9), X6
	VPUNPCKLQDQ X6, X5, X5
	VINSERTF128 $1, X5, Y1, Y1

	// Stage 1 butterfly for Y1
	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1

	// Y2: Load [src[1], src[9], src[5], src[13]] -> butterflies (1,9), (5,13)
	VMOVSD 8(R9), X2
	VMOVSD 72(R9), X4
	VPUNPCKLQDQ X4, X2, X2
	VMOVSD 40(R9), X5
	VMOVSD 104(R9), X6
	VPUNPCKLQDQ X6, X5, X5
	VINSERTF128 $1, X5, Y2, Y2

	// Stage 1 butterfly for Y2
	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y2, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y2

	// Y3: Load [src[3], src[11], src[7], src[15]] -> butterflies (3,11), (7,15)
	VMOVSD 24(R9), X3
	VMOVSD 88(R9), X4
	VPUNPCKLQDQ X4, X3, X3
	VMOVSD 56(R9), X5
	VMOVSD 120(R9), X6
	VPUNPCKLQDQ X6, X5, X5
	VINSERTF128 $1, X5, Y3, Y3

	// Stage 1 butterfly for Y3
	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y3, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y3

	// =======================================================================
	// STAGE 2: size=4 - use conjugated twiddles via VFMSUBADD
	// =======================================================================
	// VFMSUBADD gives: even=a*b+c, odd=a*b-c -> conjugate multiply result

	// Load twiddle factors for stage 2
	VMOVSD (R10), X4         // twiddle[0]
	VMOVSD 32(R10), X5       // twiddle[4]
	VPUNPCKLQDQ X5, X4, X4
	VINSERTF128 $1, X4, Y4, Y4  // Y4 = [tw0, tw4, tw0, tw4]

	// Pre-split twiddle (reused for all 4 registers)
	VMOVSLDUP Y4, Y14        // Y14 = [w.r, w.r, ...]
	VMOVSHDUP Y4, Y15        // Y15 = [w.i, w.i, ...]

	// Y0
	VPERM2F128 $0x00, Y0, Y0, Y5
	VPERM2F128 $0x11, Y0, Y0, Y6
	VSHUFPS $0xB1, Y6, Y6, Y9
	VMULPS Y15, Y9, Y9
	VFMSUBADD231PS Y14, Y6, Y9  // Conjugate multiply
	VADDPS Y9, Y5, Y7
	VSUBPS Y9, Y5, Y8
	VINSERTF128 $1, X8, Y7, Y0

	// Y1
	VPERM2F128 $0x00, Y1, Y1, Y5
	VPERM2F128 $0x11, Y1, Y1, Y6
	VSHUFPS $0xB1, Y6, Y6, Y9
	VMULPS Y15, Y9, Y9
	VFMSUBADD231PS Y14, Y6, Y9
	VADDPS Y9, Y5, Y7
	VSUBPS Y9, Y5, Y8
	VINSERTF128 $1, X8, Y7, Y1

	// Y2
	VPERM2F128 $0x00, Y2, Y2, Y5
	VPERM2F128 $0x11, Y2, Y2, Y6
	VSHUFPS $0xB1, Y6, Y6, Y9
	VMULPS Y15, Y9, Y9
	VFMSUBADD231PS Y14, Y6, Y9
	VADDPS Y9, Y5, Y7
	VSUBPS Y9, Y5, Y8
	VINSERTF128 $1, X8, Y7, Y2

	// Y3
	VPERM2F128 $0x00, Y3, Y3, Y5
	VPERM2F128 $0x11, Y3, Y3, Y6
	VSHUFPS $0xB1, Y6, Y6, Y9
	VMULPS Y15, Y9, Y9
	VFMSUBADD231PS Y14, Y6, Y9
	VADDPS Y9, Y5, Y7
	VSUBPS Y9, Y5, Y8
	VINSERTF128 $1, X8, Y7, Y3

	// =======================================================================
	// STAGE 3: size=8 - use conjugated twiddles via VFMSUBADD
	// =======================================================================

	// Load twiddle factors for stage 3
	VMOVSD (R10), X4         // twiddle[0]
	VMOVSD 16(R10), X5       // twiddle[2]
	VPUNPCKLQDQ X5, X4, X4
	VMOVSD 32(R10), X5       // twiddle[4]
	VMOVSD 48(R10), X6       // twiddle[6]
	VPUNPCKLQDQ X6, X5, X5
	VINSERTF128 $1, X5, Y4, Y4  // Y4 = [tw0, tw2, tw4, tw6]

	// Pre-split twiddle
	VMOVSLDUP Y4, Y14
	VMOVSHDUP Y4, Y15

	// Group 1: Y0 with Y1
	VSHUFPS $0xB1, Y1, Y1, Y7
	VMULPS Y15, Y7, Y7
	VFMSUBADD231PS Y14, Y1, Y7  // Conjugate multiply

	VADDPS Y7, Y0, Y8        // Y8 = new indices 0-3
	VSUBPS Y7, Y0, Y9        // Y9 = new indices 4-7

	// Group 2: Y2 with Y3
	VSHUFPS $0xB1, Y3, Y3, Y7
	VMULPS Y15, Y7, Y7
	VFMSUBADD231PS Y14, Y3, Y7

	VADDPS Y7, Y2, Y10       // Y10 = new indices 8-11
	VSUBPS Y7, Y2, Y11       // Y11 = new indices 12-15

	// =======================================================================
	// STAGE 4: size=16 - use conjugated twiddles via VFMSUBADD
	// =======================================================================

	// Load twiddle factors for stage 4
	VMOVUPS (R10), Y4        // Y4 = [tw0, tw1, tw2, tw3]
	VMOVUPS 32(R10), Y5      // Y5 = [tw4, tw5, tw6, tw7]

	// Group 1: Y8 (indices 0-3) with Y10 (indices 8-11) using Y4 (tw0-3)
	VMOVSLDUP Y4, Y6
	VMOVSHDUP Y4, Y7
	VSHUFPS $0xB1, Y10, Y10, Y0
	VMULPS Y7, Y0, Y0
	VFMSUBADD231PS Y6, Y10, Y0  // Conjugate multiply

	VADDPS Y0, Y8, Y12       // Y12 = final indices 0-3
	VSUBPS Y0, Y8, Y13       // Y13 = final indices 8-11

	// Group 2: Y9 (indices 4-7) with Y11 (indices 12-15) using Y5 (tw4-7)
	VMOVSLDUP Y5, Y6
	VMOVSHDUP Y5, Y7
	VSHUFPS $0xB1, Y11, Y11, Y0
	VMULPS Y7, Y0, Y0
	VFMSUBADD231PS Y6, Y11, Y0

	VADDPS Y0, Y9, Y10       // Y10 = final indices 4-7
	VSUBPS Y0, Y9, Y11       // Y11 = final indices 12-15

	// =======================================================================
	// Apply 1/n scaling (1/16 = 0.0625)
	// =======================================================================
	MOVL ·sixteenth32(SB), AX     // 0.0625f in IEEE-754
	MOVD AX, X4
	VBROADCASTSS X4, Y4      // Y4 = [0.0625, 0.0625, ...]
	VMULPS Y4, Y12, Y12
	VMULPS Y4, Y10, Y10
	VMULPS Y4, Y13, Y13
	VMULPS Y4, Y11, Y11

	// =======================================================================
	// Store final results to dst
	// =======================================================================
	MOVQ dst+0(FP), R9
	VMOVUPS Y12, (R9)        // dst[0-3]
	VMOVUPS Y10, 32(R9)      // dst[4-7]
	VMOVUPS Y13, 64(R9)      // dst[8-11]
	VMOVUPS Y11, 96(R9)      // dst[12-15]

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size16_inv_return_false:
	MOVB $0, ret+96(FP)
	RET

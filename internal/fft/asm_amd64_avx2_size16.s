//go:build amd64 && fft_asm && !purego

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
TEXT ·forwardAVX2Size16Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
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

	MOVQ bitrev+104(FP), AX
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
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =======================================================================
	// For size 16, bitrev = [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
	// We use precomputed indices from the bitrev slice for correctness.
	// Unrolled into 4 groups of 4 for efficiency.

	// Group 0: indices 0-3
	MOVQ (R12), DX           // bitrev[0]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)            // work[0]

	MOVQ 8(R12), DX          // bitrev[1]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 8(R8)           // work[1]

	MOVQ 16(R12), DX         // bitrev[2]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 16(R8)          // work[2]

	MOVQ 24(R12), DX         // bitrev[3]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 24(R8)          // work[3]

	// Group 1: indices 4-7
	MOVQ 32(R12), DX         // bitrev[4]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 32(R8)          // work[4]

	MOVQ 40(R12), DX         // bitrev[5]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 40(R8)          // work[5]

	MOVQ 48(R12), DX         // bitrev[6]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 48(R8)          // work[6]

	MOVQ 56(R12), DX         // bitrev[7]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 56(R8)          // work[7]

	// Group 2: indices 8-11
	MOVQ 64(R12), DX         // bitrev[8]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 64(R8)          // work[8]

	MOVQ 72(R12), DX         // bitrev[9]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 72(R8)          // work[9]

	MOVQ 80(R12), DX         // bitrev[10]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 80(R8)          // work[10]

	MOVQ 88(R12), DX         // bitrev[11]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 88(R8)          // work[11]

	// Group 3: indices 12-15
	MOVQ 96(R12), DX         // bitrev[12]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 96(R8)          // work[12]

	MOVQ 104(R12), DX        // bitrev[13]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 104(R8)         // work[13]

	MOVQ 112(R12), DX        // bitrev[14]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 112(R8)         // work[14]

	MOVQ 120(R12), DX        // bitrev[15]
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 120(R8)         // work[15]

	// =======================================================================
	// STAGE 1: size=2, half=1, step=8
	// =======================================================================
	// 8 independent butterflies with pairs: (0,1), (2,3), (4,5), (6,7),
	//                                       (8,9), (10,11), (12,13), (14,15)
	// All use twiddle[0] = (1, 0) which is identity multiplication.
	// So: a' = a + b, b' = a - b (no complex multiply needed)

	// Load all 16 complex64 values into 4 YMM registers
	// Y0 = [work[0], work[1], work[2], work[3]]
	// Y1 = [work[4], work[5], work[6], work[7]]
	// Y2 = [work[8], work[9], work[10], work[11]]
	// Y3 = [work[12], work[13], work[14], work[15]]
	VMOVUPS (R8), Y0
	VMOVUPS 32(R8), Y1
	VMOVUPS 64(R8), Y2
	VMOVUPS 96(R8), Y3

	// Stage 1: Butterflies on adjacent pairs within each 128-bit lane
	// For size=2 FFT: out[0] = in[0] + in[1], out[1] = in[0] - in[1]
	// Using twiddle[0] = 1+0i means t = b * 1 = b
	//
	// We need to rearrange: take pairs (0,1), (2,3) etc and compute a+b, a-b
	// VSHUFPD with imm=0b0101 swaps adjacent 64-bit elements in each 128-bit lane
	// But for size-2 butterflies, we need a different approach:
	//
	// Y0 = [a0, b0, a1, b1] where a0=work[0], b0=work[1], a1=work[2], b1=work[3]
	// We want: [a0+b0, a0-b0, a1+b1, a1-b1]
	//
	// Use VPERMILPD to create: [b0, a0, b1, a1]
	// Then add/sub

	// Y0: [w0, w1, w2, w3] -> pairs (w0,w1), (w2,w3)
	VPERMILPD $0x05, Y0, Y4  // Y4 = [w1, w0, w3, w2] (swap within 128-bit lanes)
	VADDPS Y4, Y0, Y5        // Y5 = [w0+w1, w1+w0, w2+w3, w3+w2]
	VSUBPS Y4, Y0, Y6        // Y6 = [w0-w1, w1-w0, w2-w3, w3-w2]
	// Now blend: take even positions from Y5, odd from Y6
	VBLENDPS $0xAA, Y6, Y5, Y0  // Y0 = [w0+w1, w0-w1, w2+w3, w2-w3]

	// Same for Y1: pairs (w4,w5), (w6,w7)
	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y4, Y1, Y6
	VBLENDPS $0xAA, Y6, Y5, Y1

	// Same for Y2: pairs (w8,w9), (w10,w11)
	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y4, Y2, Y6
	VBLENDPS $0xAA, Y6, Y5, Y2

	// Same for Y3: pairs (w12,w13), (w14,w15)
	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y4, Y3, Y6
	VBLENDPS $0xAA, Y6, Y5, Y3

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

	// Y0 = [d0, d1, d2, d3]
	// Extract a = [d0, d1, d0, d1] (low 128 bits duplicated)
	// Extract b = [d2, d3, d2, d3] (high 128 bits duplicated)
	VPERM2F128 $0x00, Y0, Y0, Y5  // Y5 = [d0, d1, d0, d1] (low lane to both)
	VPERM2F128 $0x11, Y0, Y0, Y6  // Y6 = [d2, d3, d2, d3] (high lane to both)

	// Complex multiply: t = w * b (Y4 * Y6)
	VMOVSLDUP Y4, Y7         // Y7 = [w.r, w.r, ...] (broadcast real parts)
	VMOVSHDUP Y4, Y8         // Y8 = [w.i, w.i, ...] (broadcast imag parts)
	VSHUFPS $0xB1, Y6, Y6, Y9  // Y9 = b_swapped = [b.i, b.r, ...]
	VMULPS Y8, Y9, Y9        // Y9 = [b.i*w.i, b.r*w.i, ...]
	VFMADDSUB231PS Y7, Y6, Y9  // Y9 = t = w * b

	// Butterfly: a' = a + t, b' = a - t
	VADDPS Y9, Y5, Y7        // Y7 = a + t
	VSUBPS Y9, Y5, Y8        // Y8 = a - t

	// Recombine: Y0 = [a'0, a'1, b'0, b'1]
	// Y7 has a' in both halves, Y8 has b' in both halves
	// Take low 128 bits of Y7 and high 128 bits (as low) of Y8
	VINSERTF128 $1, X8, Y7, Y0  // Y0 = [a'0, a'1, b'0, b'1]

	// Same for Y1 = [d4, d5, d6, d7]
	VPERM2F128 $0x00, Y1, Y1, Y5
	VPERM2F128 $0x11, Y1, Y1, Y6
	VMOVSLDUP Y4, Y7
	VMOVSHDUP Y4, Y8
	VSHUFPS $0xB1, Y6, Y6, Y9
	VMULPS Y8, Y9, Y9
	VFMADDSUB231PS Y7, Y6, Y9
	VADDPS Y9, Y5, Y7
	VSUBPS Y9, Y5, Y8
	VINSERTF128 $1, X8, Y7, Y1

	// Same for Y2 = [d8, d9, d10, d11]
	VPERM2F128 $0x00, Y2, Y2, Y5
	VPERM2F128 $0x11, Y2, Y2, Y6
	VMOVSLDUP Y4, Y7
	VMOVSHDUP Y4, Y8
	VSHUFPS $0xB1, Y6, Y6, Y9
	VMULPS Y8, Y9, Y9
	VFMADDSUB231PS Y7, Y6, Y9
	VADDPS Y9, Y5, Y7
	VSUBPS Y9, Y5, Y8
	VINSERTF128 $1, X8, Y7, Y2

	// Same for Y3 = [d12, d13, d14, d15]
	VPERM2F128 $0x00, Y3, Y3, Y5
	VPERM2F128 $0x11, Y3, Y3, Y6
	VMOVSLDUP Y4, Y7
	VMOVSHDUP Y4, Y8
	VSHUFPS $0xB1, Y6, Y6, Y9
	VMULPS Y8, Y9, Y9
	VFMADDSUB231PS Y7, Y6, Y9
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

	// Group 1: Y0 (indices 0-3) with Y1 (indices 4-7)
	// a = Y0 = [d0, d1, d2, d3]
	// b = Y1 = [d4, d5, d6, d7]
	// t = w * b, a' = a + t, b' = a - t

	VMOVSLDUP Y4, Y5         // Y5 = [w.r, w.r, ...]
	VMOVSHDUP Y4, Y6         // Y6 = [w.i, w.i, ...]
	VSHUFPS $0xB1, Y1, Y1, Y7  // Y7 = b_swapped
	VMULPS Y6, Y7, Y7        // Y7 = b_swap * w.i
	VFMADDSUB231PS Y5, Y1, Y7  // Y7 = t = w * b

	VADDPS Y7, Y0, Y8        // Y8 = a + t = new a (indices 0-3)
	VSUBPS Y7, Y0, Y9        // Y9 = a - t = new b (indices 4-7)

	// Group 2: Y2 (indices 8-11) with Y3 (indices 12-15)
	VSHUFPS $0xB1, Y3, Y3, Y7
	VMULPS Y6, Y7, Y7
	VFMADDSUB231PS Y5, Y3, Y7

	VADDPS Y7, Y2, Y0        // Y0 = new indices 8-11
	VSUBPS Y7, Y2, Y1        // Y1 = new indices 12-15

	// Move stage 3 results to Y0-Y3
	VMOVAPS Y8, Y2           // Save new 0-3 to Y2 temporarily
	VMOVAPS Y9, Y3           // Save new 4-7 to Y3 temporarily
	// Y0 = new 8-11, Y1 = new 12-15
	// Reorder: Y0=0-3, Y1=4-7, Y2=8-11, Y3=12-15
	VMOVAPS Y2, Y10          // Y10 = new 0-3
	VMOVAPS Y3, Y11          // Y11 = new 4-7
	VMOVAPS Y0, Y2           // Y2 = new 8-11
	VMOVAPS Y1, Y3           // Y3 = new 12-15
	VMOVAPS Y10, Y0          // Y0 = new 0-3
	VMOVAPS Y11, Y1          // Y1 = new 4-7

	// =======================================================================
	// STAGE 4: size=16, half=8, step=1
	// =======================================================================
	// 8 butterflies: index i with index i+8, for i=0..7
	// Twiddle factors: twiddle[0..7]

	// Load twiddle factors for stage 4
	VMOVUPS (R10), Y4        // Y4 = [tw0, tw1, tw2, tw3]
	VMOVUPS 32(R10), Y5      // Y5 = [tw4, tw5, tw6, tw7]

	// Group 1: Y0 (indices 0-3) with Y2 (indices 8-11) using Y4 (tw0-3)
	VMOVSLDUP Y4, Y6         // Y6 = [w.r broadcast]
	VMOVSHDUP Y4, Y7         // Y7 = [w.i broadcast]
	VSHUFPS $0xB1, Y2, Y2, Y8  // Y8 = b_swapped
	VMULPS Y7, Y8, Y8        // Y8 = b_swap * w.i
	VFMADDSUB231PS Y6, Y2, Y8  // Y8 = t = w * b

	VADDPS Y8, Y0, Y10       // Y10 = a' (new indices 0-3)
	VSUBPS Y8, Y0, Y11       // Y11 = b' (new indices 8-11)

	// Group 2: Y1 (indices 4-7) with Y3 (indices 12-15) using Y5 (tw4-7)
	VMOVSLDUP Y5, Y6
	VMOVSHDUP Y5, Y7
	VSHUFPS $0xB1, Y3, Y3, Y8
	VMULPS Y7, Y8, Y8
	VFMADDSUB231PS Y6, Y3, Y8

	VADDPS Y8, Y1, Y12       // Y12 = a' (new indices 4-7)
	VSUBPS Y8, Y1, Y13       // Y13 = b' (new indices 12-15)

	// Store final results to dst (not work buffer!)
	// If we were using scratch, we need to copy to dst
	MOVQ dst+0(FP), R9       // R9 = dst pointer
	VMOVUPS Y10, (R9)        // dst[0-3]
	VMOVUPS Y12, 32(R9)      // dst[4-7]
	VMOVUPS Y11, 64(R9)      // dst[8-11]
	VMOVUPS Y13, 96(R9)      // dst[12-15]

	VZEROUPPER
	MOVB $1, ret+120(FP)     // Return true (success)
	RET

size16_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 16, complex64
// ===========================================================================
// Stub implementation - returns false to fall back to generic implementation
TEXT ·inverseAVX2Size16Complex64Asm(SB), NOSPLIT, $0-121
	MOVB $0, ret+120(FP)        // Return false
	RET

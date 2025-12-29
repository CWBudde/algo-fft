//go:build amd64 && fft_asm && !purego

// ===========================================================================
// AVX2 Size-8 FFT Kernels for AMD64
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
TEXT ·forwardAVX2Size8Radix2Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 8)

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_fwd_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_fwd_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   size8_fwd_return_false

	// Select working buffer: if dst == src, use scratch
	CMPQ R8, R9
	JNE  size8_fwd_use_dst
	MOVQ R11, R8             // In-place: use scratch as work buffer
	JMP  size8_fwd_bitrev

size8_fwd_use_dst:
	// Out-of-place: use dst as work buffer

size8_fwd_bitrev:
	// =======================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =======================================================================
	// For size 8: bitrev = [0, 4, 2, 6, 1, 5, 3, 7]
	// Each complex64 is 8 bytes

	// Load all 8 elements with bit-reversal
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

	// =======================================================================
	// Load all data into YMM registers
	// =======================================================================
	// Y0 = [work[0], work[1], work[2], work[3]] = [x0, x1, x2, x3]
	// Y1 = [work[4], work[5], work[6], work[7]] = [x4, x5, x6, x7]
	VMOVUPS (R8), Y0
	VMOVUPS 32(R8), Y1

	// =======================================================================
	// STAGE 1: size=2, half=1, step=4 (n/size = 8/2 = 4)
	// =======================================================================
	// Butterflies on adjacent pairs: (x0,x1), (x2,x3), (x4,x5), (x6,x7)
	// All use twiddle[0] = 1+0i, so t = b * 1 = b
	// Result: a' = a + b, b' = a - b
	//
	// Y0 = [x0, x1, x2, x3] -> [x0+x1, x0-x1, x2+x3, x2-x3] = [a0, a1, a2, a3]
	// Y1 = [x4, x5, x6, x7] -> [x4+x5, x4-x5, x6+x7, x6-x7] = [a4, a5, a6, a7]

	// For Y0: swap adjacent complex64 elements, add/sub, blend
	// VPERMILPD swaps pairs of 64-bit elements within each 128-bit lane
	// Y0 = [c0, c1, c2, c3] -> Y4 = [c1, c0, c3, c2]
	VPERMILPD $0x05, Y0, Y4
	VADDPS Y4, Y0, Y5        // Y5 = [c0+c1, c1+c0, c2+c3, c3+c2]
	VSUBPS Y0, Y4, Y6        // Y6 = [c1-c0, c0-c1, c3-c2, c2-c3] (reversed order!)
	// Blend at 64-bit (complex64) granularity
	// VBLENDPD $0x0A = 0b1010: positions 1,3 from Y6, positions 0,2 from Y5
	// Y5 = [c0+c1, c1+c0, c2+c3, c3+c2]
	// Y6 = [c1-c0, c0-c1, c3-c2, c2-c3]
	// Result: [c0+c1, c0-c1, c2+c3, c2-c3] as required
	VBLENDPD $0x0A, Y6, Y5, Y0

	// For Y1: same operation
	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1

	// =======================================================================
	// STAGE 2: size=4, half=2, step=2 (n/size = 8/4 = 2)
	// =======================================================================
	// After stage 1:
	//   Y0 = [a0, a1, a2, a3]
	//   Y1 = [a4, a5, a6, a7]
	//
	// Butterflies: (a0,a2) with tw[0], (a1,a3) with tw[2]
	//              (a4,a6) with tw[0], (a5,a7) with tw[2]
	//
	// tw[0] = 1+0i (identity)
	// tw[2] = cos(-π/2) + i*sin(-π/2) = 0 - i = (0, -1)
	//
	// For j=0: t = tw[0] * a2 = a2, so b0 = a0+a2, b2 = a0-a2
	// For j=1: t = tw[2] * a3, so b1 = a1+t, b3 = a1-t

	// Load twiddle factors: tw[0] and tw[2]
	// tw[0] at offset 0, tw[2] at offset 16 (each complex64 is 8 bytes)
	VMOVSD (R10), X4         // X4 = tw[0] = (1, 0)
	VMOVSD 16(R10), X5       // X5 = tw[2] = (0, -1)
	VPUNPCKLQDQ X5, X4, X4   // X4 = [tw[0], tw[2]]
	VINSERTF128 $1, X4, Y4, Y4  // Y4 = [tw[0], tw[2], tw[0], tw[2]]

	// Reorganize Y0: need to pair (a0,a2) and (a1,a3)
	// Y0 = [a0, a1, a2, a3]
	// We need: a_low = [a0, a1, a0, a1], a_high = [a2, a3, a2, a3]
	VPERM2F128 $0x00, Y0, Y0, Y5  // Y5 = [a0, a1, a0, a1]
	VPERM2F128 $0x11, Y0, Y0, Y6  // Y6 = [a2, a3, a2, a3]

	// Complex multiply: t = tw * a_high
	VMOVSLDUP Y4, Y7         // Y7 = [tw.r, tw.r, ...]
	VMOVSHDUP Y4, Y8         // Y8 = [tw.i, tw.i, ...]
	VSHUFPS $0xB1, Y6, Y6, Y9  // Y9 = [a.i, a.r, ...] (swap real/imag)
	VMULPS Y8, Y9, Y9        // Y9 = [a.i*tw.i, a.r*tw.i, ...]
	VFMADDSUB231PS Y7, Y6, Y9  // Y9 = t = tw * a_high

	// Butterfly: a' = a_low + t, b' = a_low - t
	VADDPS Y9, Y5, Y7        // Y7 = a_low + t
	VSUBPS Y9, Y5, Y8        // Y8 = a_low - t

	// Recombine: Y0 = [a'0, a'1, b'0, b'1] = [b0, b1, b2, b3]
	VINSERTF128 $1, X8, Y7, Y0

	// Same for Y1 = [a4, a5, a6, a7]
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

	// =======================================================================
	// STAGE 3: size=8, half=4, step=1 (n/size = 8/8 = 1)
	// =======================================================================
	// After stage 2:
	//   Y0 = [b0, b1, b2, b3]
	//   Y1 = [b4, b5, b6, b7]
	//
	// Butterflies: (b0,b4) with tw[0], (b1,b5) with tw[1],
	//              (b2,b6) with tw[2], (b3,b7) with tw[3]
	//
	// tw[0] = 1+0i, tw[1] = (√2/2, -√2/2), tw[2] = (0,-1), tw[3] = (-√2/2, -√2/2)

	// Load twiddle factors for stage 3: tw[0], tw[1], tw[2], tw[3]
	VMOVUPS (R10), Y4        // Y4 = [tw[0], tw[1], tw[2], tw[3]]

	// Complex multiply: t = tw * Y1 (b4, b5, b6, b7)
	VMOVSLDUP Y4, Y7         // broadcast real parts
	VMOVSHDUP Y4, Y8         // broadcast imag parts
	VSHUFPS $0xB1, Y1, Y1, Y9  // swap real/imag of Y1
	VMULPS Y8, Y9, Y9        // [b.i*tw.i, b.r*tw.i, ...]
	VFMADDSUB231PS Y7, Y1, Y9  // Y9 = t = tw * Y1

	// Butterfly: c = Y0 + t, c' = Y0 - t
	VADDPS Y9, Y0, Y2        // Y2 = [c0, c1, c2, c3]
	VSUBPS Y9, Y0, Y3        // Y3 = [c4, c5, c6, c7]

	// =======================================================================
	// Store results
	// =======================================================================
	// Check if we need to copy back to dst (if we used scratch)
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size8_fwd_store_direct

	// We used scratch, store to dst
	VMOVUPS Y2, (R9)
	VMOVUPS Y3, 32(R9)
	JMP size8_fwd_done

size8_fwd_store_direct:
	// We used dst directly
	VMOVUPS Y2, (R8)
	VMOVUPS Y3, 32(R8)

size8_fwd_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size8_fwd_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 8, complex64, radix-2 variant
// ===========================================================================
// Same as forward but with conjugated twiddle factors and 1/n scaling
//
// Optimization notes:
// - Stage 1 uses identity twiddle (1+0i), so conjugation has no effect
// - Conjugation is done by negating imaginary parts via VFMSUBADD instead
//   of explicit XOR with sign mask, avoiding the mask setup overhead
// - Twiddle factor real/imag splits are hoisted and reused across Y0/Y1
TEXT ·inverseAVX2Size8Radix2Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 8)

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_inv_return_false

	// Validate all slice lengths >= 8
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   size8_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   size8_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   size8_inv_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   size8_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size8_inv_use_dst
	MOVQ R11, R8
	JMP  size8_inv_bitrev

size8_inv_use_dst:

size8_inv_bitrev:
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

	// =======================================================================
	// STAGE 1: size=2 (same as forward - twiddle[0] = 1+0i, conj has no effect)
	// =======================================================================
	VPERMILPD $0x05, Y0, Y4
	VADDPS Y4, Y0, Y5
	VSUBPS Y0, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y0

	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1

	// =======================================================================
	// STAGE 2: size=4 - use conjugated twiddles via VFMSUBADD
	// =======================================================================
	// For inverse: conj(tw) means negate imaginary part
	// Instead of XOR with sign mask, use VFMSUBADD which computes:
	//   result[even] = a[even] - b[even]  (real: r*r - (-i*i) = r*r + i*i... wait, need different approach)
	//
	// Actually, for conjugate multiply: conj(tw) * z = (tr, -ti) * (zr, zi)
	//   real = tr*zr - (-ti)*zi = tr*zr + ti*zi
	//   imag = tr*zi + (-ti)*zr = tr*zi - ti*zr
	//
	// Using VFMSUBADD231PS: dst = a*b +/- dst (alternating sub/add at even/odd positions)
	// For complex64 with [real, imag, real, imag, ...]:
	//   VMULPS:     [zi*ti, zr*ti, ...]
	//   VFMSUBADD:  [tr*zr - zi*ti, tr*zi + zr*ti, ...] <- this is conj(tw)*z when we negate ti!
	//
	// But we can't negate just ti easily. Let's use a different approach:
	// Regular complex multiply: tw * z
	//   real = tr*zr - ti*zi
	//   imag = tr*zi + ti*zr
	// Conjugate multiply: conj(tw) * z
	//   real = tr*zr + ti*zi  (add instead of sub)
	//   imag = tr*zi - ti*zr  (sub instead of add)
	//
	// VFMADDSUB gives: even=a*b-c, odd=a*b+c -> [tr*zr - zi*ti, tr*zi + zr*ti]
	// VFMSUBADD gives: even=a*b+c, odd=a*b-c -> [tr*zr + zi*ti, tr*zi - zr*ti] = conj multiply!

	// Load twiddle factors for stage 2
	VMOVSD (R10), X4         // tw[0]
	VMOVSD 16(R10), X5       // tw[2]
	VPUNPCKLQDQ X5, X4, X4
	VINSERTF128 $1, X4, Y4, Y4  // Y4 = [tw[0], tw[2], tw[0], tw[2]]

	// Split twiddle into real and imag parts (reused for Y0 and Y1)
	VMOVSLDUP Y4, Y10        // Y10 = [tr, tr, ...] (broadcast real parts)
	VMOVSHDUP Y4, Y11        // Y11 = [ti, ti, ...] (broadcast imag parts)

	// Process Y0
	VPERM2F128 $0x00, Y0, Y0, Y5  // Y5 = [a0, a1, a0, a1]
	VPERM2F128 $0x11, Y0, Y0, Y6  // Y6 = [a2, a3, a2, a3]
	VSHUFPS $0xB1, Y6, Y6, Y9     // Y9 = [ai, ar, ...] (swap real/imag)
	VMULPS Y11, Y9, Y9            // Y9 = [ai*ti, ar*ti, ...]
	VFMSUBADD231PS Y10, Y6, Y9    // Y9 = [tr*ar + ai*ti, tr*ai - ar*ti] = conj(tw)*a
	VADDPS Y9, Y5, Y7             // Y7 = a_low + t
	VSUBPS Y9, Y5, Y8             // Y8 = a_low - t
	VINSERTF128 $1, X8, Y7, Y0    // Y0 = [result_low, result_high]

	// Process Y1 (reuse Y10, Y11)
	VPERM2F128 $0x00, Y1, Y1, Y5
	VPERM2F128 $0x11, Y1, Y1, Y6
	VSHUFPS $0xB1, Y6, Y6, Y9
	VMULPS Y11, Y9, Y9
	VFMSUBADD231PS Y10, Y6, Y9
	VADDPS Y9, Y5, Y7
	VSUBPS Y9, Y5, Y8
	VINSERTF128 $1, X8, Y7, Y1

	// =======================================================================
	// STAGE 3: size=8 - use conjugated twiddles via VFMSUBADD
	// =======================================================================
	VMOVUPS (R10), Y4        // Y4 = [tw[0], tw[1], tw[2], tw[3]]

	// Complex multiply with conjugated twiddles using VFMSUBADD
	VMOVSLDUP Y4, Y10        // broadcast real parts
	VMOVSHDUP Y4, Y11        // broadcast imag parts
	VSHUFPS $0xB1, Y1, Y1, Y9  // swap real/imag of Y1
	VMULPS Y11, Y9, Y9       // [bi*ti, br*ti, ...]
	VFMSUBADD231PS Y10, Y1, Y9  // Y9 = conj(tw) * Y1

	// Butterfly
	VADDPS Y9, Y0, Y2
	VSUBPS Y9, Y0, Y3

	// =======================================================================
	// Apply 1/n scaling (1/8 = 0.125)
	// =======================================================================
	MOVL $0x3E000000, AX     // 0.125f in IEEE-754
	MOVD AX, X4
	VBROADCASTSS X4, Y4      // Y4 = [0.125, 0.125, ...]
	VMULPS Y4, Y2, Y2
	VMULPS Y4, Y3, Y3

	// =======================================================================
	// Store results
	// =======================================================================
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size8_inv_store_direct

	VMOVUPS Y2, (R9)
	VMOVUPS Y3, 32(R9)
	JMP size8_inv_done

size8_inv_store_direct:
	VMOVUPS Y2, (R8)
	VMOVUPS Y3, 32(R8)

size8_inv_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size8_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// SIZE 8 RADIX-4 KERNELS
// ===========================================================================
// 8-point FFT using mixed-radix: 1 radix-4 stage + 1 radix-2 stage
//
// This reduces from 3 radix-2 stages to 2 stages total, improving performance.
//
// Stage 1 (Radix-4): 2 radix-4 butterflies processing indices [0,2,4,6] and [1,3,5,7]
// Stage 2 (Radix-2): 4 radix-2 butterflies combining results from stage 1
// ===========================================================================

// Forward transform, size 8, complex64, radix-4 variant
// Mixed-radix FFT: 1 radix-4 stage + 1 radix-2 stage
TEXT ·forwardAVX2Size8Radix4Complex64Asm(SB), NOSPLIT, $0-121
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
	MOVL $0x80000000, AX
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
	MOVL $0x80000000, AX
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
TEXT ·inverseAVX2Size8Radix4Complex64Asm(SB), NOSPLIT, $0-121
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
	MOVL $0x80000000, AX
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
	MOVL $0x80000000, AX
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
	MOVL $0x3E000000, AX       // 0.125f
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

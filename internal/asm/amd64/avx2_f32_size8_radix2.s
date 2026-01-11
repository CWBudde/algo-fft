//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-8 Radix-2 FFT (complex64) Kernels for AMD64
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
TEXT ·ForwardAVX2Size8Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
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

	// Select work buffer: dst for out-of-place, scratch for in-place
	CMPQ R8, R9
	JNE  size8_fwd_start
	MOVQ R11, R8

size8_fwd_start:
	// =======================================================================
	// FUSED: Bit-reversal + STAGE 1 (identity twiddles)
	// =======================================================================
	// Load bit-reversed data [0,4,2,6,1,5,3,7] and compute Stage 1 butterflies
	// Result: Y0 = [c0+c4, c0-c4, c2+c6, c2-c6], Y1 = [c1+c5, c1-c5, c3+c7, c3-c7]

	// Load first half: [c0, c4, c2, c6]
	MOVQ (R9), AX            // src[0] = c0
	MOVQ AX, (R8)            // work[0] = c0

	MOVQ 32(R9), AX          // src[4] = c4
	MOVQ AX, 8(R8)           // work[1] = c4

	MOVQ 16(R9), AX          // src[2] = c2
	MOVQ AX, 16(R8)          // work[2] = c2

	MOVQ 48(R9), AX          // src[6] = c6
	MOVQ AX, 24(R8)          // work[3] = c6

	// Compute Stage 1 butterflies for first half
	VMOVUPS (R8), Y0
	VPERMILPD $0x05, Y0, Y4
	VADDPS Y4, Y0, Y5
	VSUBPS Y0, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y0

	// Load second half: [c1, c5, c3, c7]
	MOVQ 8(R9), AX           // src[1] = c1
	MOVQ AX, 32(R8)          // work[4] = c1

	MOVQ 40(R9), AX          // src[5] = c5
	MOVQ AX, 40(R8)          // work[5] = c5

	MOVQ 24(R9), AX          // src[3] = c3
	MOVQ AX, 48(R8)          // work[6] = c3

	MOVQ 56(R9), AX          // src[7] = c7
	MOVQ AX, 56(R8)          // work[7] = c7

	// Compute Stage 1 butterflies for second half
	VMOVUPS 32(R8), Y1
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
	MOVB $1, ret+96(FP)
	RET

size8_fwd_return_false:
	MOVB $0, ret+96(FP)
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
TEXT ·InverseAVX2Size8Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
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

	// Select work buffer: dst for out-of-place, scratch for in-place
	CMPQ R8, R9
	JNE  size8_inv_start
	MOVQ R11, R8

size8_inv_start:
	// =======================================================================
	// FUSED: Bit-reversal + STAGE 1 (identity twiddles)
	// =======================================================================
	// Load bit-reversed data [0,4,2,6,1,5,3,7] and compute Stage 1 butterflies

	// Load first half: [c0, c4, c2, c6]
	MOVQ (R9), AX
	MOVQ AX, (R8)

	MOVQ 32(R9), AX
	MOVQ AX, 8(R8)

	MOVQ 16(R9), AX
	MOVQ AX, 16(R8)

	MOVQ 48(R9), AX
	MOVQ AX, 24(R8)

	// Compute Stage 1 butterflies for first half
	VMOVUPS (R8), Y0
	VPERMILPD $0x05, Y0, Y4
	VADDPS Y4, Y0, Y5
	VSUBPS Y0, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y0

	// Load second half: [c1, c5, c3, c7]
	MOVQ 8(R9), AX
	MOVQ AX, 32(R8)

	MOVQ 40(R9), AX
	MOVQ AX, 40(R8)

	MOVQ 24(R9), AX
	MOVQ AX, 48(R8)

	MOVQ 56(R9), AX
	MOVQ AX, 56(R8)

	// Compute Stage 1 butterflies for second half
	VMOVUPS 32(R8), Y1
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
	MOVL ·eighth32(SB), AX     // 0.125f in IEEE-754
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
	MOVB $1, ret+96(FP)
	RET

size8_inv_return_false:
	MOVB $0, ret+96(FP)
	RET

//go:build arm64 && fft_asm && !purego

// ===========================================================================
// NEON Size-Specific FFT Kernels for ARM64
// ===========================================================================
//
// This file contains fully-unrolled FFT kernels optimized for specific sizes.
// These kernels provide better performance than the generic implementation by:
//   - Eliminating loop overhead
//   - Using hardcoded twiddle factor indices
//   - Optimal register allocation for each size
//
// Sizes implemented: 16 (full), 32/64/128 (stubs)
//
// See asm_arm64_neon_generic.s for algorithm documentation.
//
// ===========================================================================

#include "textflag.h"

TEXT ·forwardNEONSize16Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVD dst+0(FP), R8           // R8 = dst pointer
	MOVD src+24(FP), R9          // R9 = src pointer
	MOVD twiddle+48(FP), R10     // R10 = twiddle pointer
	MOVD scratch+72(FP), R11     // R11 = scratch pointer
	MOVD bitrev+96(FP), R12      // R12 = bitrev pointer
	MOVD src+32(FP), R13         // R13 = n (should be 16)

	// Verify n == 16
	CMP  $16, R13
	BNE  neon16_return_false

	// Validate all slice lengths >= 16
	MOVD dst+8(FP), R0
	CMP  $16, R0
	BLT  neon16_return_false

	MOVD twiddle+56(FP), R0
	CMP  $16, R0
	BLT  neon16_return_false

	MOVD scratch+80(FP), R0
	CMP  $16, R0
	BLT  neon16_return_false

	MOVD bitrev+104(FP), R0
	CMP  $16, R0
	BLT  neon16_return_false

	// Select working buffer: in-place uses scratch, out-of-place uses dst
	CMP  R8, R9
	BNE  neon16_use_dst

	// In-place: use scratch as working buffer
	MOVD R11, R8
	B    neon16_bitrev

neon16_use_dst:
	// Out-of-place: use dst directly (R8 already set)

neon16_bitrev:
	// =======================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =======================================================================
	// For size 16, bitrev = [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
	// We use the precomputed bitrev slice for correctness.
	// Each complex64 is 8 bytes, each int in bitrev is 8 bytes.

	// Unroll all 16 loads
	MOVD (R12), R0               // R0 = bitrev[0]
	LSL  $3, R0, R0              // byte offset
	ADD  R9, R0, R0              // &src[bitrev[0]]
	MOVD (R0), R1
	MOVD R1, (R8)                // work[0]

	MOVD 8(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 8(R8)               // work[1]

	MOVD 16(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 16(R8)              // work[2]

	MOVD 24(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 24(R8)              // work[3]

	MOVD 32(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 32(R8)              // work[4]

	MOVD 40(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 40(R8)              // work[5]

	MOVD 48(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 48(R8)              // work[6]

	MOVD 56(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 56(R8)              // work[7]

	MOVD 64(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 64(R8)              // work[8]

	MOVD 72(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 72(R8)              // work[9]

	MOVD 80(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 80(R8)              // work[10]

	MOVD 88(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 88(R8)              // work[11]

	MOVD 96(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 96(R8)              // work[12]

	MOVD 104(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 104(R8)             // work[13]

	MOVD 112(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 112(R8)             // work[14]

	MOVD 120(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 120(R8)             // work[15]

	// Load neonOnes constant for butterfly operations
	MOVD $·neonOnes(SB), R0
	VLD1 (R0), [V28.S4]          // V28 = [1.0, 1.0, 1.0, 1.0]

	// =======================================================================
	// STAGE 1: size=2, half=1, step=8
	// =======================================================================
	// 8 butterflies with pairs: (0,1), (2,3), (4,5), (6,7),
	//                           (8,9), (10,11), (12,13), (14,15)
	// All use twiddle[0] = (1, 0), so t = b * 1 = b
	// Butterfly: a' = a + b, b' = a - b (no complex multiply needed)

	// Load all 16 complex64 values into 8 NEON registers (2 complex64 each)
	VLD1 (R8), [V0.S4]           // V0 = [work[0], work[1]] = [a0, b0]
	ADD  $16, R8, R1
	VLD1 (R1), [V1.S4]           // V1 = [work[2], work[3]] = [a1, b1]
	ADD  $16, R1, R1
	VLD1 (R1), [V2.S4]           // V2 = [work[4], work[5]] = [a2, b2]
	ADD  $16, R1, R1
	VLD1 (R1), [V3.S4]           // V3 = [work[6], work[7]] = [a3, b3]
	ADD  $16, R1, R1
	VLD1 (R1), [V4.S4]           // V4 = [work[8], work[9]] = [a4, b4]
	ADD  $16, R1, R1
	VLD1 (R1), [V5.S4]           // V5 = [work[10], work[11]] = [a5, b5]
	ADD  $16, R1, R1
	VLD1 (R1), [V6.S4]           // V6 = [work[12], work[13]] = [a6, b6]
	ADD  $16, R1, R1
	VLD1 (R1), [V7.S4]           // V7 = [work[14], work[15]] = [a7, b7]

	// Stage 1: For each register V, we have [a, b] and need [a+b, a-b]
	// Use VEXT to create [b, a], then add/sub using VFMLA/VFMLS with ones vector
	// (Go assembler doesn't support VFADD/VFSUB directly)

	// V0 = [a0.r, a0.i, b0.r, b0.i] -> [a0+b0, a0-b0]
	VEXT $8, V0.B16, V0.B16, V16.B16  // V16 = [b0, a0]
	// V17 = V0 + V16 via VFMLA: dst = dst + a*1
	VMOV V0.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4      // V17 = V0 + V16*1 = a + b
	// V18 = V0 - V16 via VFMLS: dst = dst - a*1
	VMOV V0.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4      // V18 = V0 - V16*1 = a - b
	// Now extract: low 64 bits of V17 = a', low 64 bits of V18 = b'
	// Combine into V0 = [a', b']
	VMOV V17.D[0], V0.D[0]            // V0.D[0] = a' = a0+b0
	VMOV V18.D[0], V0.D[1]            // V0.D[1] = b' = a0-b0

	// Same pattern for V1-V7
	VEXT $8, V1.B16, V1.B16, V16.B16
	VMOV V1.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4
	VMOV V1.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4
	VMOV V17.D[0], V1.D[0]
	VMOV V18.D[0], V1.D[1]

	VEXT $8, V2.B16, V2.B16, V16.B16
	VMOV V2.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4
	VMOV V2.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4
	VMOV V17.D[0], V2.D[0]
	VMOV V18.D[0], V2.D[1]

	VEXT $8, V3.B16, V3.B16, V16.B16
	VMOV V3.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4
	VMOV V3.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4
	VMOV V17.D[0], V3.D[0]
	VMOV V18.D[0], V3.D[1]

	VEXT $8, V4.B16, V4.B16, V16.B16
	VMOV V4.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4
	VMOV V4.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4
	VMOV V17.D[0], V4.D[0]
	VMOV V18.D[0], V4.D[1]

	VEXT $8, V5.B16, V5.B16, V16.B16
	VMOV V5.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4
	VMOV V5.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4
	VMOV V17.D[0], V5.D[0]
	VMOV V18.D[0], V5.D[1]

	VEXT $8, V6.B16, V6.B16, V16.B16
	VMOV V6.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4
	VMOV V6.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4
	VMOV V17.D[0], V6.D[0]
	VMOV V18.D[0], V6.D[1]

	VEXT $8, V7.B16, V7.B16, V16.B16
	VMOV V7.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4
	VMOV V7.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4
	VMOV V17.D[0], V7.D[0]
	VMOV V18.D[0], V7.D[1]

	// =======================================================================
	// STAGE 2: size=4, half=2, step=4
	// =======================================================================
	// 4 groups of 2 butterflies: (0,2),(1,3), (4,6),(5,7), (8,10),(9,11), (12,14),(13,15)
	// j=0 uses twiddle[0]=(1,0), j=1 uses twiddle[4]=(0,-1)
	//
	// Current layout after stage 1:
	// V0 = [d0, d1], V1 = [d2, d3], V2 = [d4, d5], V3 = [d6, d7]
	// V4 = [d8, d9], V5 = [d10, d11], V6 = [d12, d13], V7 = [d14, d15]
	//
	// Need butterflies: (d0,d2), (d1,d3), (d4,d6), etc.
	// This means pairing V0 with V1, V2 with V3, V4 with V5, V6 with V7

	// Load twiddle factors for stage 2: tw[0] and tw[4]
	// Pack into V24 = [tw0, tw4] for the two butterflies per group
	VLD1.P 8(R10), [V24.D1]      // V24.D[0] = twiddle[0] = (1, 0)
	ADD    $24, R10, R0          // R0 = &twiddle[4] (skip tw[1,2,3])
	VLD1   (R0), [V25.D1]        // V25.D[0] = twiddle[4] = (0, -1)
	VMOV   V24.D[0], V24.D[0]    // tw0 in low
	VMOV   V25.D[0], V24.D[1]    // tw4 in high
	// V24 = [tw0, tw4]

	// Reset R10 to twiddle base for later stages
	MOVD twiddle+48(FP), R10

	// Group 1: V0 = [d0, d1] with V1 = [d2, d3]
	// a = [d0, d1], b = [d2, d3], w = [tw0, tw4]
	// t = w * b, a' = a + t, b' = a - t

	// Complex multiply: t = w * b
	// Deinterleave b into real/imag
	VUZP1 V1.S4, V1.S4, V16.S4   // V16 = [br0, br1, br0, br1]
	VUZP2 V1.S4, V1.S4, V17.S4   // V17 = [bi0, bi1, bi0, bi1]
	VUZP1 V24.S4, V24.S4, V18.S4 // V18 = [wr0, wr1, wr0, wr1]
	VUZP2 V24.S4, V24.S4, V19.S4 // V19 = [wi0, wi1, wi0, wi1]

	// t.real = br*wr - bi*wi
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4

	// t.imag = br*wi + bi*wr
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4

	// Reinterleave t
	VZIP1 V21.S4, V20.S4, V22.S4 // V22 = [t0.r, t0.i, t1.r, t1.i] = t

	// Butterfly: a' = a + t, b' = a - t (using VFMLA/VFMLS with ones vector)
	VMOV V0.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4  // V16 = V0 + V22*1 = a'
	VMOV V0.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4  // V17 = V0 - V22*1 = b'
	VMOV V16.B16, V0.B16          // V0 = new [d0, d1]
	VMOV V17.B16, V1.B16          // V1 = new [d2, d3]

	// Group 2: V2 = [d4, d5] with V3 = [d6, d7]
	VUZP1 V3.S4, V3.S4, V16.S4
	VUZP2 V3.S4, V3.S4, V17.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V2.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4
	VMOV V2.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4
	VMOV V16.B16, V2.B16
	VMOV V17.B16, V3.B16

	// Group 3: V4 = [d8, d9] with V5 = [d10, d11]
	VUZP1 V5.S4, V5.S4, V16.S4
	VUZP2 V5.S4, V5.S4, V17.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V4.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4
	VMOV V4.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4
	VMOV V16.B16, V4.B16
	VMOV V17.B16, V5.B16

	// Group 4: V6 = [d12, d13] with V7 = [d14, d15]
	VUZP1 V7.S4, V7.S4, V16.S4
	VUZP2 V7.S4, V7.S4, V17.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V6.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4
	VMOV V6.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4
	VMOV V16.B16, V6.B16
	VMOV V17.B16, V7.B16

	// =======================================================================
	// STAGE 3: size=8, half=4, step=2
	// =======================================================================
	// 2 groups of 4 butterflies:
	//   Group 1: (d0,d4), (d1,d5), (d2,d6), (d3,d7) with tw[0,2,4,6]
	//   Group 2: (d8,d12), (d9,d13), (d10,d14), (d11,d15) with tw[0,2,4,6]
	//
	// Current layout:
	// V0=[d0,d1], V1=[d2,d3], V2=[d4,d5], V3=[d6,d7]
	// V4=[d8,d9], V5=[d10,d11], V6=[d12,d13], V7=[d14,d15]
	//
	// Need butterflies between: (V0,V2), (V1,V3), (V4,V6), (V5,V7)
	// Twiddles: tw[0], tw[2], tw[4], tw[6] in pairs

	// Load twiddle factors for stage 3
	// V24 = [tw0, tw2], V25 = [tw4, tw6]
	VLD1.P 8(R10), [V24.D1]      // tw[0]
	ADD    $8, R10, R0           // skip tw[1]
	VLD1   (R0), [V25.D1]        // tw[2]
	VMOV   V25.D[0], V24.D[1]    // V24 = [tw0, tw2]
	ADD    $8, R0, R0            // R0 = &tw[3]
	ADD    $8, R0, R0            // R0 = &tw[4]
	VLD1.P 8(R0), [V25.D1]       // load tw[4], R0 = &tw[5]
	ADD    $8, R0, R0            // R0 = &tw[6]
	VLD1   (R0), [V26.D1]        // load tw[6]
	VMOV   V26.D[0], V25.D[1]    // V25 = [tw4, tw6]

	// Reset R10 for stage 4
	MOVD twiddle+48(FP), R10

	// Group 1a: V0 = [d0, d1] with V2 = [d4, d5], using V24 = [tw0, tw2]
	VUZP1 V2.S4, V2.S4, V16.S4
	VUZP2 V2.S4, V2.S4, V17.S4
	VUZP1 V24.S4, V24.S4, V18.S4
	VUZP2 V24.S4, V24.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V0.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4
	VMOV V0.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4
	VMOV V16.B16, V0.B16         // V0 = new [d0, d1]
	VMOV V17.B16, V2.B16         // V2 = new [d4, d5]

	// Group 1b: V1 = [d2, d3] with V3 = [d6, d7], using V25 = [tw4, tw6]
	VUZP1 V3.S4, V3.S4, V16.S4
	VUZP2 V3.S4, V3.S4, V17.S4
	VUZP1 V25.S4, V25.S4, V18.S4
	VUZP2 V25.S4, V25.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V1.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4
	VMOV V1.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4
	VMOV V16.B16, V1.B16         // V1 = new [d2, d3]
	VMOV V17.B16, V3.B16         // V3 = new [d6, d7]

	// Reload twiddles (same pattern for group 2)
	VLD1.P 8(R10), [V24.D1]      // load tw[0], R10 = &tw[1]
	ADD    $8, R10, R0           // R0 = &tw[2]
	VLD1   (R0), [V25.D1]        // load tw[2]
	VMOV   V25.D[0], V24.D[1]    // V24 = [tw[0], tw[2]]
	ADD    $8, R0, R0            // R0 = &tw[3]
	ADD    $8, R0, R0            // R0 = &tw[4]
	VLD1.P 8(R0), [V25.D1]       // load tw[4], R0 = &tw[5]
	ADD    $8, R0, R0            // R0 = &tw[6]
	VLD1   (R0), [V26.D1]        // load tw[6]
	VMOV   V26.D[0], V25.D[1]    // V25 = [tw[4], tw[6]]
	MOVD twiddle+48(FP), R10

	// Group 2a: V4 = [d8, d9] with V6 = [d12, d13], using V24 = [tw0, tw2]
	VUZP1 V6.S4, V6.S4, V16.S4
	VUZP2 V6.S4, V6.S4, V17.S4
	VUZP1 V24.S4, V24.S4, V18.S4
	VUZP2 V24.S4, V24.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V4.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4
	VMOV V4.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4
	VMOV V16.B16, V4.B16         // V4 = new [d8, d9]
	VMOV V17.B16, V6.B16         // V6 = new [d12, d13]

	// Group 2b: V5 = [d10, d11] with V7 = [d14, d15], using V25 = [tw4, tw6]
	VUZP1 V7.S4, V7.S4, V16.S4
	VUZP2 V7.S4, V7.S4, V17.S4
	VUZP1 V25.S4, V25.S4, V18.S4
	VUZP2 V25.S4, V25.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V5.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4
	VMOV V5.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4
	VMOV V16.B16, V5.B16         // V5 = new [d10, d11]
	VMOV V17.B16, V7.B16         // V7 = new [d14, d15]

	// =======================================================================
	// STAGE 4: size=16, half=8, step=1
	// =======================================================================
	// 8 butterflies: (d0,d8), (d1,d9), ..., (d7,d15)
	// Twiddles: tw[0..7]
	//
	// Current layout:
	// V0=[d0,d1], V1=[d2,d3], V2=[d4,d5], V3=[d6,d7]
	// V4=[d8,d9], V5=[d10,d11], V6=[d12,d13], V7=[d14,d15]
	//
	// Need butterflies between: (V0,V4), (V1,V5), (V2,V6), (V3,V7)
	// Twiddles: tw[0,1], tw[2,3], tw[4,5], tw[6,7]

	// Load twiddle factors for stage 4
	VLD1 (R10), [V24.S4]         // V24 = [tw0, tw1]
	ADD  $16, R10, R0
	VLD1 (R0), [V25.S4]          // V25 = [tw2, tw3]
	ADD  $16, R0, R0
	VLD1 (R0), [V26.S4]          // V26 = [tw4, tw5]
	ADD  $16, R0, R0
	VLD1 (R0), [V27.S4]          // V27 = [tw6, tw7]

	// Butterfly 1: V0 = [d0, d1] with V4 = [d8, d9], using V24 = [tw0, tw1]
	VUZP1 V4.S4, V4.S4, V16.S4
	VUZP2 V4.S4, V4.S4, V17.S4
	VUZP1 V24.S4, V24.S4, V18.S4
	VUZP2 V24.S4, V24.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V0.B16, V8.B16
	VFMLA V22.S4, V28.S4, V8.S4   // V8 = V0 + V22 = new [d0, d1]
	VMOV V0.B16, V12.B16
	VFMLS V22.S4, V28.S4, V12.S4  // V12 = V0 - V22 = new [d8, d9]

	// Butterfly 2: V1 = [d2, d3] with V5 = [d10, d11], using V25 = [tw2, tw3]
	VUZP1 V5.S4, V5.S4, V16.S4
	VUZP2 V5.S4, V5.S4, V17.S4
	VUZP1 V25.S4, V25.S4, V18.S4
	VUZP2 V25.S4, V25.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V1.B16, V9.B16
	VFMLA V22.S4, V28.S4, V9.S4   // V9 = V1 + V22 = new [d2, d3]
	VMOV V1.B16, V13.B16
	VFMLS V22.S4, V28.S4, V13.S4  // V13 = V1 - V22 = new [d10, d11]

	// Butterfly 3: V2 = [d4, d5] with V6 = [d12, d13], using V26 = [tw4, tw5]
	VUZP1 V6.S4, V6.S4, V16.S4
	VUZP2 V6.S4, V6.S4, V17.S4
	VUZP1 V26.S4, V26.S4, V18.S4
	VUZP2 V26.S4, V26.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V2.B16, V10.B16
	VFMLA V22.S4, V28.S4, V10.S4  // V10 = V2 + V22 = new [d4, d5]
	VMOV V2.B16, V14.B16
	VFMLS V22.S4, V28.S4, V14.S4  // V14 = V2 - V22 = new [d12, d13]

	// Butterfly 4: V3 = [d6, d7] with V7 = [d14, d15], using V27 = [tw6, tw7]
	VUZP1 V7.S4, V7.S4, V16.S4
	VUZP2 V7.S4, V7.S4, V17.S4
	VUZP1 V27.S4, V27.S4, V18.S4
	VUZP2 V27.S4, V27.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V3.B16, V11.B16
	VFMLA V22.S4, V28.S4, V11.S4  // V11 = V3 + V22 = new [d6, d7]
	VMOV V3.B16, V15.B16
	VFMLS V22.S4, V28.S4, V15.S4  // V15 = V3 - V22 = new [d14, d15]

	// =======================================================================
	// Store final results to dst
	// =======================================================================
	// Results are now in V8-V15:
	// V8=[d0,d1], V9=[d2,d3], V10=[d4,d5], V11=[d6,d7]
	// V12=[d8,d9], V13=[d10,d11], V14=[d12,d13], V15=[d14,d15]

	MOVD dst+0(FP), R9           // R9 = dst pointer
	VST1 [V8.S4], (R9)           // dst[0-1]
	ADD  $16, R9, R9
	VST1 [V9.S4], (R9)           // dst[2-3]
	ADD  $16, R9, R9
	VST1 [V10.S4], (R9)          // dst[4-5]
	ADD  $16, R9, R9
	VST1 [V11.S4], (R9)          // dst[6-7]
	ADD  $16, R9, R9
	VST1 [V12.S4], (R9)          // dst[8-9]
	ADD  $16, R9, R9
	VST1 [V13.S4], (R9)          // dst[10-11]
	ADD  $16, R9, R9
	VST1 [V14.S4], (R9)          // dst[12-13]
	ADD  $16, R9, R9
	VST1 [V15.S4], (R9)          // dst[14-15]

	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon16_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// forwardNEONSize32Complex64Asm - Size-32 forward FFT (stub)
TEXT ·forwardNEONSize32Complex64Asm(SB), NOSPLIT, $0-121
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// forwardNEONSize64Complex64Asm - Size-64 forward FFT (stub)
TEXT ·forwardNEONSize64Complex64Asm(SB), NOSPLIT, $0-121
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// forwardNEONSize128Complex64Asm - Size-128 forward FFT (stub)
TEXT ·forwardNEONSize128Complex64Asm(SB), NOSPLIT, $0-121
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// ===========================================================================
// Inverse Size-Specific Kernels (complex64)
// ===========================================================================

// inverseNEONSize16Complex64Asm - Size-16 inverse FFT (stub)
TEXT ·inverseNEONSize16Complex64Asm(SB), NOSPLIT, $0-121
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// inverseNEONSize32Complex64Asm - Size-32 inverse FFT (stub)
TEXT ·inverseNEONSize32Complex64Asm(SB), NOSPLIT, $0-121
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// inverseNEONSize64Complex64Asm - Size-64 inverse FFT (stub)
TEXT ·inverseNEONSize64Complex64Asm(SB), NOSPLIT, $0-121
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// inverseNEONSize128Complex64Asm - Size-128 inverse FFT (stub)
TEXT ·inverseNEONSize128Complex64Asm(SB), NOSPLIT, $0-121
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

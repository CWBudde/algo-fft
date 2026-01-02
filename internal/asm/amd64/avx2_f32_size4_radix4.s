//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-4 FFT Kernels for AMD64
// ===========================================================================
//
// This file contains a fully-unrolled radix-4 FFT kernel optimized for size 4.
// For size 4, the radix-4 FFT is optimal - just a single radix-4 butterfly
// with no twiddle factors needed (all twiddles are powers of W^0 = 1).
//
// Radix-4 Butterfly Algorithm:
//   Given 4 input points x[0], x[1], x[2], x[3]:
//
//   t0 = x[0] + x[2]
//   t1 = x[0] - x[2]
//   t2 = x[1] + x[3]
//   t3 = x[1] - x[3]
//
//   y[0] = t0 + t2
//   y[1] = t1 + t3*(-i)    // multiply by -i: (r,i) -> (i,-r)
//   y[2] = t0 - t2
//   y[3] = t1 - t3*(-i)
//
// For inverse FFT, replace -i with +i: (r,i) -> (-i,r)
// and apply 1/4 scaling at the end.
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 4, complex64, radix-4
// ===========================================================================
// Pure radix-4 implementation - no bit-reversal needed for size 4!
// Input order for radix-4 is naturally: [0, 1, 2, 3]
TEXT ·ForwardAVX2Size4Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer (not used for size 4)
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer (not used for size 4)
	MOVQ src+32(FP), R13     // R13 = n (should be 4)

	// Verify n == 4
	CMPQ R13, $4
	JNE  size4_r4_fwd_return_false

	// Validate all slice lengths >= 4
	MOVQ dst+8(FP), AX
	CMPQ AX, $4
	JL   size4_r4_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $4
	JL   size4_r4_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $4
	JL   size4_r4_fwd_return_false

	// bitrev is unused for size-4 radix-4; allow nil/empty.

	// Select working buffer: if dst == src, use scratch
	CMPQ R8, R9
	JNE  size4_r4_fwd_use_dst
	MOVQ R11, R8             // In-place: use scratch as work buffer
	JMP  size4_r4_fwd_load

size4_r4_fwd_use_dst:
	// Out-of-place: use dst as work buffer

size4_r4_fwd_load:
	// =======================================================================
	// Load all 4 complex64 values (32 bytes total, fits in 1 YMM register)
	// =======================================================================
	// Y0 = [x0, x1, x2, x3] where each x is complex64 (8 bytes)
	VMOVUPS (R9), Y0

	// =======================================================================
	// Radix-4 Butterfly Computation
	// =======================================================================
	// We need to compute:
	//   t0 = x0 + x2
	//   t1 = x0 - x2
	//   t2 = x1 + x3
	//   t3 = x1 - x3
	//
	// Then:
	//   y0 = t0 + t2
	//   y1 = t1 + t3*(-i)
	//   y2 = t0 - t2
	//   y3 = t1 - t3*(-i)

	// Extract x0, x2 into lower lane; x1, x3 into upper lane
	// Y0 = [x0, x1, x2, x3] (positions 0,1,2,3)
	// We want: Y1 = [x0, x2, x0, x2], Y2 = [x1, x3, x1, x3]

	// VPERMILPS with imm8 = 0xD8 = 0b11011000
	// This permutes within each 128-bit lane independently:
	// Low lane:  positions 0,2,1,3 -> [x0, x2, x1, x3]
	// High lane: positions 0,2,1,3 -> same pattern
	VPERMILPS $0xD8, Y0, Y1  // Y1 = [x0, x2, x1, x3] (within each lane)

	// Now split the lanes properly using VPERM2F128
	// We need [x0, x2, x0, x2] and [x1, x3, x1, x3]
	// Actually, let's use a different approach with VUNPCKLPS/VUNPCKHPS

	// Better approach: Use VSHUFPS to arrange the data
	// VSHUFPS can select elements from two sources
	// Y0 = [x0, x1, x2, x3]

	// Create [x0, x2, x0, x2] and [x1, x3, x1, x3]
	// VSHUFPS $0x88, Y0, Y0 selects bits 10,00,10,00 from low/high
	// Within each 128-bit lane: pos 0,0,2,2 from src1/src2
	VSHUFPS $0x88, Y0, Y0, Y1  // Y1 = [x0, x0, x2, x2] (interleaved)
	VSHUFPS $0xDD, Y0, Y0, Y2  // Y2 = [x1, x1, x3, x3] (interleaved)

	// Now we have pairs, but we need [x0, x2, x0, x2] as complex numbers
	// Let's use VPERM2F128 to rearrange 128-bit lanes
	// Actually, for size 4, we can use scalar-like operations

	// Alternative: Extract individual complex numbers and operate
	// This is simpler for size 4. Let's use VEXTRACTF128 and scalar ops

	// Extract x0, x1 (first 128 bits) and x2, x3 (second 128 bits)
	VEXTRACTF128 $0, Y0, X1   // X1 = [x0, x1]
	VEXTRACTF128 $1, Y0, X2   // X2 = [x2, x3]

	// Compute t0 = x0 + x2, t2 = x1 + x3, t1 = x0 - x2, t3 = x1 - x3
	VADDPS X2, X1, X3         // X3 = [t0, t2]
	VSUBPS X2, X1, X4         // X4 = [t1, t3]

	// Now compute the radix-4 butterfly outputs:
	// y0 = t0 + t2
	// y1 = t1 + t3*(-i)
	// y2 = t0 - t2
	// y3 = t1 - t3*(-i)

	// Compute radix-4 butterfly outputs directly using X3 = [t0, t2] and X4 = [t1, t3]
	// y0 = t0 + t2, y2 = t0 - t2
	// y1 = t1 + t3*(-i), y3 = t1 - t3*(-i)

	// First, compute y0 and y2 from t0 and t2
	// X3 = [t0, t2]
	// Duplicate t0 into both slots
	VMOVDDUP X3, X5            // X5 = [t0, t0] (broadcast low 64 bits)
	// Shuffle to get t2 in both slots
	VPERMILPD $0x3, X3, X6     // X6 = [t2, t2]

	VADDPS X6, X5, X9          // X9 = [t0+t2, t0+t2] = [y0, y0]
	VSUBPS X6, X5, X10         // X10 = [t0-t2, t0-t2] = [y2, y2]

	// Now compute y1 and y3 from t1 and t3
	// X4 = [t1, t3]
	// Extract t3 for multiplication by -i
	VPERMILPD $0x3, X4, X8     // X8 = [t3, t3]

	// Multiply t3 by -i: (r, i) * (-i) = (i, -r)
	VSHUFPS $0xB1, X8, X8, X11 // X11 = [i, r, i, r] (swap real/imag)

	// Negate the "imaginary" part (which after swap is the real part, in positions 1, 3)
	MOVL ·signbit32(SB), AX
	MOVD AX, X12
	VBROADCASTSS X12, X12      // X12 = [sign, sign, sign, sign]
	VXORPD X15, X15, X15       // X15 = [0, 0, 0, 0]
	VBLENDPS $0xAA, X12, X15, X12  // X12 = [0, sign, 0, sign]
	VXORPS X12, X11, X11       // X11 = [i, -r, i, -r] = t3*(-i) duplicated

	// Duplicate t1
	VMOVDDUP X4, X7            // X7 = [t1, t1]

	// Compute y1 and y3
	// SWAP: Try putting y3 in X13 and y1 in X14 to see if that fixes it
	VSUBPS X11, X7, X13        // X13 = [y3, y3] - SWAPPED
	VADDPS X11, X7, X14        // X14 = [y1, y1] - SWAPPED

	// =======================================================================
	// Combine results into output: [y0, y1, y2, y3]
	// =======================================================================
	// Now X13 has y3 and X14 has y1 (swapped)
	// So combine as [y0, y1, y2, y3] using X14 for position 1 and X13 for position 3
	VSHUFPD $0x0, X14, X9, X1  // X1 = [X9[low], X14[low]] = [y0, y1]
	VSHUFPD $0x0, X13, X10, X2 // X2 = [X10[low], X13[low]] = [y2, y3]

	// Combine into full YMM register
	VINSERTF128 $0, X1, Y0, Y0 // Y0[127:0] = X1
	VINSERTF128 $1, X2, Y0, Y0 // Y0[255:128] = X2

	// =======================================================================
	// Store results
	// =======================================================================
	// Check if we need to copy back to dst (if we used scratch)
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size4_r4_fwd_store_direct

	// We used scratch, store to dst
	VMOVUPS Y0, (R9)
	JMP size4_r4_fwd_done

size4_r4_fwd_store_direct:
	// We used dst directly
	VMOVUPS Y0, (R8)

size4_r4_fwd_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size4_r4_fwd_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 4, complex64, radix-4
// ===========================================================================
// Same as forward but with +i instead of -i, and 1/4 scaling at the end
TEXT ·InverseAVX2Size4Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	// Verify n == 4
	CMPQ R13, $4
	JNE  size4_r4_inv_return_false

	// Validate all slice lengths >= 4
	MOVQ dst+8(FP), AX
	CMPQ AX, $4
	JL   size4_r4_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $4
	JL   size4_r4_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $4
	JL   size4_r4_inv_return_false

	// bitrev is unused for size-4 radix-4; allow nil/empty.

	// Select working buffer
	CMPQ R8, R9
	JNE  size4_r4_inv_use_dst
	MOVQ R11, R8
	JMP  size4_r4_inv_load

size4_r4_inv_use_dst:

size4_r4_inv_load:
	// Load all 4 complex64 values
	VMOVUPS (R9), Y0

	// Extract x0, x1 and x2, x3
	VEXTRACTF128 $0, Y0, X1   // X1 = [x0, x1]
	VEXTRACTF128 $1, Y0, X2   // X2 = [x2, x3]

	// Compute t0 = x0 + x2, t2 = x1 + x3, t1 = x0 - x2, t3 = x1 - x3
	VADDPS X2, X1, X3         // X3 = [t0, t2]
	VSUBPS X2, X1, X4         // X4 = [t1, t3]

	// Duplicate t0 and t2 for add/sub
	VSHUFPD $0x0, X3, X3, X5   // X5 = [t0, t0]
	VSHUFPD $0x3, X3, X3, X6   // X6 = [t2, t2]

	// Compute y0 and y2
	VADDPS X6, X5, X9          // X9 = [y0, y0]
	VSUBPS X6, X5, X10         // X10 = [y2, y2]

	// Duplicate t1 and t3
	VSHUFPD $0x0, X4, X4, X7   // X7 = [t1, t1]
	VSHUFPD $0x3, X4, X4, X8   // X8 = [t3, t3]

	// Multiply t3 by +i: (r, i) * (i) = (-i, r)
	// For inverse FFT, we use +i instead of -i
	VSHUFPS $0xB1, X8, X8, X11 // X11 = [i, r, i, r] (swapped)

	// Create mask to negate real parts (which are now in positions 0, 2 after swap)
	MOVL ·signbit32(SB), AX
	MOVD AX, X12
	VBROADCASTSS X12, X12      // X12 = [sign, sign, sign, sign]
	VXORPD X15, X15, X15       // X15 = [0, 0, 0, 0]
	VBLENDPS $0x55, X12, X15, X12  // X12 = [sign, 0, sign, 0]
	VXORPS X12, X11, X11       // X11 = t3*(+i) = [-i, r, -i, r]

	// Compute y1 = t1 + t3*(+i) and y3 = t1 - t3*(+i)
	VADDPS X11, X7, X13        // X13 = X7 + X11 = t1 + t3*(+i) = y1
	VSUBPS X11, X7, X14        // X14 = X7 - X11 = t1 - t3*(+i) = y3

	// Combine results using VUNPCKLPD
	VUNPCKLPD X13, X9, X1      // X1 = [y0, y1]
	VUNPCKLPD X14, X10, X2     // X2 = [y2, y3]
	VINSERTF128 $0, X1, Y0, Y0 // Y0[127:0] = X1
	VINSERTF128 $1, X2, Y0, Y0 // Y0[255:128] = X2, final: Y0 = [y0, y1, y2, y3]

	// =======================================================================
	// Apply 1/4 scaling for inverse transform
	// =======================================================================
	MOVL ·quarter32(SB), AX       // 0.25f in IEEE-754
	MOVD AX, X3
	VBROADCASTSS X3, Y3        // Y3 = [0.25, 0.25, ...]
	VMULPS Y3, Y0, Y0          // Y0 *= 0.25

	// Store results
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size4_r4_inv_store_direct

	VMOVUPS Y0, (R9)
	JMP size4_r4_inv_done

size4_r4_inv_store_direct:
	VMOVUPS Y0, (R8)

size4_r4_inv_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size4_r4_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

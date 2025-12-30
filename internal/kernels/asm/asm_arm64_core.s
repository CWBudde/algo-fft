//go:build arm64 && fft_asm && !purego

// ===========================================================================
// ARM64 NEON FFT Assembly - Core Utilities and Constants
// ===========================================================================
//
// This file contains shared constants and helper functions used by the
// NEON FFT implementations.
//
// See asm_arm64_neon_generic.s for the main FFT algorithm documentation.
//
// ===========================================================================

#include "textflag.h"

DATA ·neonOnes+0(SB)/4, $0x3f800000
DATA ·neonOnes+4(SB)/4, $0x3f800000
DATA ·neonOnes+8(SB)/4, $0x3f800000
DATA ·neonOnes+12(SB)/4, $0x3f800000
GLOBL ·neonOnes(SB), RODATA, $16

DATA ·neonSignImag+0(SB)/4, $0x00000000
DATA ·neonSignImag+4(SB)/4, $0x80000000
DATA ·neonSignImag+8(SB)/4, $0x00000000
DATA ·neonSignImag+12(SB)/4, $0x80000000
GLOBL ·neonSignImag(SB), RODATA, $16

DATA ·neonOne64+0(SB)/8, $0x3ff0000000000000
GLOBL ·neonOne64(SB), RODATA, $8

// ===========================================================================
// func neonComplexMul2Asm(dst, a, b *complex64)
// a and b each point to 2 complex64 values (4 float32 lanes).
// dst receives the 2 complex64 results: dst[i] = a[i] * b[i].
TEXT ·neonComplexMul2Asm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD a+8(FP), R1
	MOVD b+16(FP), R2

	// Load a and b as interleaved complex64 values.
	VLD1 (R1), [V0.S4]          // V0 = [ar0, ai0, ar1, ai1]
	VLD1 (R2), [V1.S4]          // V1 = [br0, bi0, br1, bi1]

	// Deinterleave into real/imag vectors (duplicated lanes).
	VUZP1 V0.S4, V0.S4, V2.S4   // V2 = [ar0, ar1, ar0, ar1]
	VUZP2 V0.S4, V0.S4, V3.S4   // V3 = [ai0, ai1, ai0, ai1]
	VUZP1 V1.S4, V1.S4, V4.S4   // V4 = [br0, br1, br0, br1]
	VUZP2 V1.S4, V1.S4, V5.S4   // V5 = [bi0, bi1, bi0, bi1]

	// real = ar*br - ai*bi
	VEOR V6.B16, V6.B16, V6.B16
	VFMLA V2.S4, V4.S4, V6.S4
	VFMLS V3.S4, V5.S4, V6.S4

	// imag = ar*bi + ai*br
	VEOR V7.B16, V7.B16, V7.B16
	VFMLA V2.S4, V5.S4, V7.S4
	VFMLA V3.S4, V4.S4, V7.S4

	// Re-interleave real/imag into complex lanes.
	VZIP1 V7.S4, V6.S4, V0.S4
	VST1 [V0.S4], (R0)
	RET

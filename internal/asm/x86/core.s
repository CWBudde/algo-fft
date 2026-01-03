//go:build 386 && asm && !purego

// =====================================================================
// 386 FFT Assembly - Core Constants
// =====================================================================
//
// This file contains shared constants used by the SSE2 FFT implementations
// for 32-bit x86 (GOARCH=386).
//
// See asm_386_sse2.s for the FFT implementation.
//
// =====================================================================

#include "textflag.h"

// ===========================================================================
// CONSTANTS: Floating-point scaling factors for complex64
// ===========================================================================
// Single-precision (float32) constants for complex64 operations
DATA ·one32+0(SB)/4, $0x3f800000     // 1.0f
GLOBL ·one32(SB), RODATA|NOPTR, $4

DATA ·half32+0(SB)/4, $0x3f000000    // 0.5f  = 1/2
GLOBL ·half32(SB), RODATA|NOPTR, $4

DATA ·quarter32+0(SB)/4, $0x3e800000 // 0.25f = 1/4
GLOBL ·quarter32(SB), RODATA|NOPTR, $4

DATA ·eighth32+0(SB)/4, $0x3e000000  // 0.125f = 1/8
GLOBL ·eighth32(SB), RODATA|NOPTR, $4

DATA ·sixteenth32+0(SB)/4, $0x3d800000    // 0.0625f = 1/16
GLOBL ·sixteenth32(SB), RODATA|NOPTR, $4

DATA ·thirtySecond32+0(SB)/4, $0x3d000000 // 0.03125f = 1/32
GLOBL ·thirtySecond32(SB), RODATA|NOPTR, $4

DATA ·sixtyFourth32+0(SB)/4, $0x3c800000  // 0.015625f = 1/64
GLOBL ·sixtyFourth32(SB), RODATA|NOPTR, $4

DATA ·oneTwentyEighth32+0(SB)/4, $0x3c000000 // 0.0078125f = 1/128
GLOBL ·oneTwentyEighth32(SB), RODATA|NOPTR, $4

// ===========================================================================
// CONSTANTS: Sign bit masks for complex number negation
// ===========================================================================
DATA ·signbit32+0(SB)/4, $0x80000000     // float32 sign bit mask
GLOBL ·signbit32(SB), RODATA|NOPTR, $4

// ===========================================================================
// CONSTANTS: XMM lane negation masks for complex64
// ===========================================================================
// These 16-byte masks are used for selective lane negation in SSE operations.
// Layout matches XMM register structure for complex numbers.
//
// For complex64 (2 values per XMM): [re0, im0, re1, im1]
//
// Usage: XORPS to negate specific real or imaginary lanes
// ===========================================================================

// Float32 lane negation masks (complex64)
DATA ·maskNegLoPS+0(SB)/4, $0x80000000 // negate lane 0 (re)
DATA ·maskNegLoPS+4(SB)/4, $0x00000000
DATA ·maskNegLoPS+8(SB)/4, $0x80000000 // negate lane 2 (re)
DATA ·maskNegLoPS+12(SB)/4, $0x00000000
GLOBL ·maskNegLoPS(SB), RODATA|NOPTR, $16

DATA ·maskNegHiPS+0(SB)/4, $0x00000000
DATA ·maskNegHiPS+4(SB)/4, $0x80000000 // negate lane 1 (im)
DATA ·maskNegHiPS+8(SB)/4, $0x00000000
DATA ·maskNegHiPS+12(SB)/4, $0x80000000 // negate lane 3 (im)
GLOBL ·maskNegHiPS(SB), RODATA|NOPTR, $16

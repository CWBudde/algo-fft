//go:build 386 && asm && !purego

// ===========================================================================
// X86 (32-bit) FFT Assembly - Core Utilities and Constants
// ===========================================================================
//
// This file contains shared utilities and constants used by the SSE2 FFT 
// implementations for 386 architecture.
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// CONSTANTS: Floating-point scaling factors
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

// Double-precision (float64) constants for complex128 operations
DATA ·one64+0(SB)/8, $0x3ff0000000000000     // 1.0
GLOBL ·one64(SB), RODATA|NOPTR, $8

DATA ·half64+0(SB)/8, $0x3fe0000000000000    // 0.5  = 1/2
GLOBL ·half64(SB), RODATA|NOPTR, $8

DATA ·quarter64+0(SB)/8, $0x3fd0000000000000 // 0.25 = 1/4
GLOBL ·quarter64(SB), RODATA|NOPTR, $8

DATA ·eighth64+0(SB)/8, $0x3fc0000000000000  // 0.125 = 1/8
GLOBL ·eighth64(SB), RODATA|NOPTR, $8

DATA ·sixteenth64+0(SB)/8, $0x3fb0000000000000    // 0.0625 = 1/16
GLOBL ·sixteenth64(SB), RODATA|NOPTR, $8

DATA ·thirtySecond64+0(SB)/8, $0x3fa0000000000000 // 0.03125 = 1/32
GLOBL ·thirtySecond64(SB), RODATA|NOPTR, $8

DATA ·sixtyFourth64+0(SB)/8, $0x3f90000000000000  // 0.015625 = 1/64
GLOBL ·sixtyFourth64(SB), RODATA|NOPTR, $8

DATA ·oneTwentyEighth64+0(SB)/8, $0x3f80000000000000 // 0.0078125 = 1/128
GLOBL ·oneTwentyEighth64(SB), RODATA|NOPTR, $8

// ===========================================================================
// CONSTANTS: Sign bit masks
// ===========================================================================

DATA ·signbit32+0(SB)/4, $0x80000000     // float32 sign bit mask
GLOBL ·signbit32(SB), RODATA|NOPTR, $4

DATA ·signbit64+0(SB)/8, $0x8000000000000000 // float64 sign bit mask
GLOBL ·signbit64(SB), RODATA|NOPTR, $8

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

// Float64 lane negation masks (complex128)
DATA ·maskNegLoPD+0(SB)/8, $0x8000000000000000 // negate lane 0 (re)
DATA ·maskNegLoPD+8(SB)/8, $0x0000000000000000
GLOBL ·maskNegLoPD(SB), RODATA|NOPTR, $16

DATA ·maskNegHiPD+0(SB)/8, $0x0000000000000000
DATA ·maskNegHiPD+8(SB)/8, $0x8000000000000000 // negate lane 1 (im)
GLOBL ·maskNegHiPD(SB), RODATA|NOPTR, $16


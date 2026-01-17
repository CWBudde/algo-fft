//go:build amd64 && asm && !purego

// ===========================================================================
// AMD64 FFT Assembly - Core Utilities and Constants
// ===========================================================================
//
// This file contains shared utilities, constants, and small helpers used by
// the AVX2 and SSE2 FFT implementations.
//
// See asm_amd64_avx2_generic.s for the main FFT algorithm documentation.
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Go Calling Convention - Slice Layout
// ===========================================================================
// Each []T in Go ABI is: ptr (8 bytes) + len (8 bytes) + cap (8 bytes) = 24 bytes
//
// Function signature:
//   func forwardAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
//
// Stack frame layout (offsets from FP):
//   dst:     FP+0   (ptr), FP+8   (len), FP+16  (cap)
//   src:     FP+24  (ptr), FP+32  (len), FP+40  (cap)
//   twiddle: FP+48  (ptr), FP+56  (len), FP+64  (cap)
//   scratch: FP+72  (ptr), FP+80  (len), FP+88  (cap)
//   bitrev:  FP+96  (ptr), FP+104 (len), FP+112 (cap)
//   return:  FP+120 (bool, 1 byte)

// ===========================================================================
// Data Type Sizes
// ===========================================================================
// complex64:  8 bytes  = 4 bytes (float32 real) + 4 bytes (float32 imag)
// complex128: 16 bytes = 8 bytes (float64 real) + 8 bytes (float64 imag)
//
// YMM register (256 bits = 32 bytes):
//   - Holds 4 complex64  values (4 × 8  = 32 bytes)
//   - Holds 2 complex128 values (2 × 16 = 32 bytes)
//
// XMM register (128 bits = 16 bytes):
//   - Holds 2 complex64  values (2 × 8  = 16 bytes)
//   - Holds 1 complex128 value  (1 × 16 = 16 bytes)

// ===========================================================================
// CONSTANTS: Floating-point scaling factors
// ===========================================================================
// These constants are shared across all AMD64 FFT implementations (AVX2, SSE2)
// and are used for inverse FFT normalization and other scaling operations.
//
// IEEE 754 bit patterns:
//   Single-precision (32-bit): sign(1) | exponent(8) | mantissa(23)
//   Double-precision (64-bit): sign(1) | exponent(11) | mantissa(52)
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

// Double-precision (float64) constants for complex128 operations
DATA ·one64+0(SB)/8, $0x3ff0000000000000     // 1.0
GLOBL ·one64(SB), RODATA|NOPTR, $8

DATA ·half64+0(SB)/8, $0x3fe0000000000000    // 0.5  = 1/2
GLOBL ·half64(SB), RODATA|NOPTR, $8

DATA ·quarter64+0(SB)/8, $0x3fd0000000000000 // 0.25 = 1/4
GLOBL ·quarter64(SB), RODATA|NOPTR, $8

DATA ·eighth64+0(SB)/8, $0x3fc0000000000000  // 0.125 = 1/8
GLOBL ·eighth64(SB), RODATA|NOPTR, $8

DATA ·sixteenth32+0(SB)/4, $0x3d800000    // 0.0625f = 1/16
GLOBL ·sixteenth32(SB), RODATA|NOPTR, $4

DATA ·thirtySecond32+0(SB)/4, $0x3d000000 // 0.03125f = 1/32
GLOBL ·thirtySecond32(SB), RODATA|NOPTR, $4

DATA ·sixtyFourth32+0(SB)/4, $0x3c800000  // 0.015625f = 1/64
GLOBL ·sixtyFourth32(SB), RODATA|NOPTR, $4

DATA ·oneTwentyEighth32+0(SB)/4, $0x3c000000 // 0.0078125f = 1/128
GLOBL ·oneTwentyEighth32(SB), RODATA|NOPTR, $4

DATA ·twoFiftySixth32+0(SB)/4, $0x3b800000 // 0.00390625f = 1/256
GLOBL ·twoFiftySixth32(SB), RODATA|NOPTR, $4

DATA ·fiveHundredTwelfth32+0(SB)/4, $0x3b000000 // 0.001953125f = 1/512
GLOBL ·fiveHundredTwelfth32(SB), RODATA|NOPTR, $4

DATA ·oneThousandTwentyFourth32+0(SB)/4, $0x3a800000 // 0.0009765625f = 1/1024
GLOBL ·oneThousandTwentyFourth32(SB), RODATA|NOPTR, $4

DATA ·twoThousandFortyEighth32+0(SB)/4, $0x3a000000 // 0.00048828125f = 1/2048
GLOBL ·twoThousandFortyEighth32(SB), RODATA|NOPTR, $4

DATA ·oneFourThousandNinetySixth32+0(SB)/4, $0x39800000 // 0.000244140625f = 1/4096
GLOBL ·oneFourThousandNinetySixth32(SB), RODATA|NOPTR, $4

DATA ·eightThousandOneHundredThirtySecond32+0(SB)/4, $0x39000000 // 0.0001220703125f = 1/8192
GLOBL ·eightThousandOneHundredThirtySecond32(SB), RODATA|NOPTR, $4


DATA ·oneSixteenThousandThreeHundredEightyFourth32+0(SB)/4, $0x38800000 // 1/16384
GLOBL ·oneSixteenThousandThreeHundredEightyFourth32(SB), RODATA|NOPTR, $4

DATA ·sixteenth64+0(SB)/8, $0x3fb0000000000000    // 0.0625 = 1/16
GLOBL ·sixteenth64(SB), RODATA|NOPTR, $8

DATA ·thirtySecond64+0(SB)/8, $0x3fa0000000000000 // 0.03125 = 1/32
GLOBL ·thirtySecond64(SB), RODATA|NOPTR, $8

DATA ·sixtyFourth64+0(SB)/8, $0x3f90000000000000  // 0.015625 = 1/64
GLOBL ·sixtyFourth64(SB), RODATA|NOPTR, $8

DATA ·oneTwentyEighth64+0(SB)/8, $0x3f80000000000000 // 0.0078125 = 1/128
GLOBL ·oneTwentyEighth64(SB), RODATA|NOPTR, $8

DATA ·twoFiftySixth64+0(SB)/8, $0x3f70000000000000 // 0.00390625 = 1/256
GLOBL ·twoFiftySixth64(SB), RODATA|NOPTR, $8

DATA ·fiveHundredTwelfth64+0(SB)/8, $0x3f60000000000000 // 0.001953125 = 1/512
GLOBL ·fiveHundredTwelfth64(SB), RODATA|NOPTR, $8

DATA ·twoThousandFortyEighth64+0(SB)/8, $0x3f40000000000000 // 0.00048828125 = 1/2048
GLOBL ·twoThousandFortyEighth64(SB), RODATA|NOPTR, $8

DATA ·eightThousandOneHundredThirtySecond64+0(SB)/8, $0x3f20000000000000 // 0.0001220703125 = 1/8192
GLOBL ·eightThousandOneHundredThirtySecond64(SB), RODATA|NOPTR, $8

// ===========================================================================
// CONSTANTS: Sign bit masks for complex number negation
// ===========================================================================
// These masks are used to negate complex numbers via XOR operations.
// XORing with the sign bit flips the sign without affecting the magnitude.
//
// Usage: XORPS/XORPD with these masks to negate real or imaginary components
// ===========================================================================

DATA ·signbit32+0(SB)/4, $0x80000000     // float32 sign bit mask
GLOBL ·signbit32(SB), RODATA|NOPTR, $4

DATA ·signbit64+0(SB)/8, $0x8000000000000000 // float64 sign bit mask
GLOBL ·signbit64(SB), RODATA|NOPTR, $8

// ===========================================================================
// CONSTANTS: XMM/YMM lane negation masks
// ===========================================================================
// These 16-byte masks are used for selective lane negation in SIMD operations.
// Layout matches XMM register structure for complex numbers.
//
// For complex64 (2 values per XMM): [re0, im0, re1, im1]
// For complex128 (1 value per XMM): [re, im]
//
// Usage: XORPS/XORPD/VXORPS/VXORPD to negate specific real or imaginary lanes
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

// Float64 lane negation masks (complex128)
DATA ·maskNegLoPD+0(SB)/8, $0x8000000000000000 // negate lane 0 (re)
DATA ·maskNegLoPD+8(SB)/8, $0x0000000000000000
GLOBL ·maskNegLoPD(SB), RODATA|NOPTR, $16

DATA ·maskNegHiPD+0(SB)/8, $0x0000000000000000
DATA ·maskNegHiPD+8(SB)/8, $0x8000000000000000 // negate lane 1 (im)
GLOBL ·maskNegHiPD(SB), RODATA|NOPTR, $16

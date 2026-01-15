//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-8 Radix-4 FFT Kernels for ARM64 (complex128)
// ===========================================================================
//
// Size 8 = 4 * 2, mixed-radix (radix-4 + radix-2) algorithm:
//   Stage 1: 2 radix-4 butterflies (no twiddles)
//   Stage 2: radix-2 butterflies with twiddles
//
// Each complex128 element is 16 bytes (real f64 + imag f64).
//
// Register allocation:
//   R8  = working dst pointer (may point to scratch if src==dst)
//   R9  = src pointer
//   R10 = twiddle pointer
//   R11 = scratch pointer
//   R12 = bit-reversal table pointer
//   R13 = n (size, should be 8)
//   R20 = original dst pointer (preserved)
//
// ===========================================================================

#include "textflag.h"

// Note: neonInv8F64 is defined in neon_f64_size8_radix2.s to avoid duplicate symbols

// Forward transform, size 8, complex128, radix-4 variant
// func ForwardNEONSize8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool
TEXT ·ForwardNEONSize8Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load slice headers from stack
	MOVD dst+0(FP), R8          // R8 = dst.data
	MOVD src+24(FP), R9         // R9 = src.data
	MOVD twiddle+48(FP), R10    // R10 = twiddle.data
	MOVD scratch+72(FP), R11    // R11 = scratch.data
	MOVD src+32(FP), R13        // R13 = src.len (n)

	// Validate n == 8
	CMP  $8, R13                // n must be exactly 8
	BNE  neon8r4f64_return_false

	// Validate dst capacity >= 8
	MOVD dst+8(FP), R0          // R0 = dst.len
	CMP  $8, R0
	BLT  neon8r4f64_return_false

	// Validate twiddle capacity >= 8
	MOVD twiddle+56(FP), R0     // R0 = twiddle.len
	CMP  $8, R0
	BLT  neon8r4f64_return_false

	// Validate scratch capacity >= 8
	MOVD scratch+80(FP), R0     // R0 = scratch.len
	CMP  $8, R0
	BLT  neon8r4f64_return_false

	// Load bit-reversal table address
	MOVD $bitrev_size8_radix4_f64<>(SB), R12

	// Save original dst pointer
	MOVD R8, R20                // R20 = original dst

	// If src == dst, use scratch as working buffer to avoid aliasing
	CMP  R8, R9                 // compare dst and src pointers
	BNE  neon8r4f64_use_dst     // if different, use dst directly
	MOVD R11, R8                // src==dst: use scratch as working buffer

neon8r4f64_use_dst:
	// =========================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =========================================================================
	MOVD $0, R0                 // R0 = loop counter i

neon8r4f64_bitrev_loop:
	CMP  $8, R0                 // while i < 8
	BGE  neon8r4f64_stage1

	LSL  $3, R0, R1             // R1 = i * 8 (offset into bitrev table)
	ADD  R12, R1, R1            // R1 = &bitrev[i]
	MOVD (R1), R2               // R2 = bitrev[i] (source index)

	LSL  $4, R2, R3             // R3 = bitrev[i] * 16 (complex128 is 16 bytes)
	ADD  R9, R3, R3             // R3 = &src[bitrev[i]]
	MOVD (R3), R4               // R4 = src[bitrev[i]].real (as bits)
	MOVD 8(R3), R5              // R5 = src[bitrev[i]].imag (as bits)

	LSL  $4, R0, R3             // R3 = i * 16
	ADD  R8, R3, R3             // R3 = &work[i]
	MOVD R4, (R3)               // work[i].real = src[bitrev[i]].real
	MOVD R5, 8(R3)              // work[i].imag = src[bitrev[i]].imag

	ADD  $1, R0, R0             // i++
	B    neon8r4f64_bitrev_loop

neon8r4f64_stage1:
	// =========================================================================
	// Stage 1: Two radix-4 butterflies (no twiddles)
	// Process elements [0,1,2,3] and [4,5,6,7]
	// =========================================================================

	// --- First radix-4 butterfly: elements 0,1,2,3 ---
	// Load x0,x1,x2,x3 from work[0..3]
	FMOVD 0(R8), F0             // x0.real
	FMOVD 8(R8), F1             // x0.imag
	FMOVD 16(R8), F2            // x1.real
	FMOVD 24(R8), F3            // x1.imag
	FMOVD 32(R8), F4            // x2.real
	FMOVD 40(R8), F5            // x2.imag
	FMOVD 48(R8), F6            // x3.real
	FMOVD 56(R8), F7            // x3.imag

	// t0 = x0 + x2, t1 = x0 - x2
	FADDD F4, F0, F8            // t0.real = x0.r + x2.r
	FADDD F5, F1, F9            // t0.imag = x0.i + x2.i
	FSUBD F4, F0, F10           // t1.real = x0.r - x2.r
	FSUBD F5, F1, F11           // t1.imag = x0.i - x2.i

	// t2 = x1 + x3, t3 = x1 - x3
	FADDD F6, F2, F12           // t2.real = x1.r + x3.r
	FADDD F7, F3, F13           // t2.imag = x1.i + x3.i
	FSUBD F6, F2, F14           // t3.real = x1.r - x3.r
	FSUBD F7, F3, F15           // t3.imag = x1.i - x3.i

	// a0 = t0 + t2, a2 = t0 - t2
	FADDD F12, F8, F16          // a0.real = t0.r + t2.r
	FADDD F13, F9, F17          // a0.imag = t0.i + t2.i
	FSUBD F12, F8, F18          // a2.real = t0.r - t2.r
	FSUBD F13, F9, F19          // a2.imag = t0.i - t2.i

	// (-i) * t3 => (t3.imag, -t3.real)
	FMOVD F15, F20              // (-i)*t3.real = t3.imag
	FNEGD F14, F21              // (-i)*t3.imag = -t3.real

	// a1 = t1 + (-i)*t3
	FADDD F20, F10, F22         // a1.real = t1.r + t3.i
	FADDD F21, F11, F23         // a1.imag = t1.i - t3.r

	// i * t3 => (-t3.imag, t3.real)
	FNEGD F15, F24              // i*t3.real = -t3.imag
	FMOVD F14, F25              // i*t3.imag = t3.real

	// a3 = t1 + i*t3
	FADDD F24, F10, F26         // a3.real = t1.r - t3.i
	FADDD F25, F11, F27         // a3.imag = t1.i + t3.r

	// Store a0,a1,a2,a3 to work[0..3]
	FMOVD F16, 0(R8)            // work[0].real = a0.r
	FMOVD F17, 8(R8)            // work[0].imag = a0.i
	FMOVD F22, 16(R8)           // work[1].real = a1.r
	FMOVD F23, 24(R8)           // work[1].imag = a1.i
	FMOVD F18, 32(R8)           // work[2].real = a2.r
	FMOVD F19, 40(R8)           // work[2].imag = a2.i
	FMOVD F26, 48(R8)           // work[3].real = a3.r
	FMOVD F27, 56(R8)           // work[3].imag = a3.i

	// --- Second radix-4 butterfly: elements 4,5,6,7 ---
	// Load x4,x5,x6,x7 from work[4..7]
	FMOVD 64(R8), F0            // x4.real
	FMOVD 72(R8), F1            // x4.imag
	FMOVD 80(R8), F2            // x5.real
	FMOVD 88(R8), F3            // x5.imag
	FMOVD 96(R8), F4            // x6.real
	FMOVD 104(R8), F5           // x6.imag
	FMOVD 112(R8), F6           // x7.real
	FMOVD 120(R8), F7           // x7.imag

	// t0 = x4 + x6, t1 = x4 - x6
	FADDD F4, F0, F8            // t0.real
	FADDD F5, F1, F9            // t0.imag
	FSUBD F4, F0, F10           // t1.real
	FSUBD F5, F1, F11           // t1.imag

	// t2 = x5 + x7, t3 = x5 - x7
	FADDD F6, F2, F12           // t2.real
	FADDD F7, F3, F13           // t2.imag
	FSUBD F6, F2, F14           // t3.real
	FSUBD F7, F3, F15           // t3.imag

	// a4 = t0 + t2, a6 = t0 - t2
	FADDD F12, F8, F16          // a4.real
	FADDD F13, F9, F17          // a4.imag
	FSUBD F12, F8, F18          // a6.real
	FSUBD F13, F9, F19          // a6.imag

	// (-i) * t3 => (t3.imag, -t3.real)
	FMOVD F15, F20              // (-i)*t3.real = t3.imag
	FNEGD F14, F21              // (-i)*t3.imag = -t3.real

	// a5 = t1 + (-i)*t3
	FADDD F20, F10, F22         // a5.real
	FADDD F21, F11, F23         // a5.imag

	// i * t3 => (-t3.imag, t3.real)
	FNEGD F15, F24              // i*t3.real = -t3.imag
	FMOVD F14, F25              // i*t3.imag = t3.real

	// a7 = t1 + i*t3
	FADDD F24, F10, F26         // a7.real
	FADDD F25, F11, F27         // a7.imag

	// Store a4,a5,a6,a7 to work[4..7]
	FMOVD F16, 64(R8)           // work[4]
	FMOVD F17, 72(R8)
	FMOVD F22, 80(R8)           // work[5]
	FMOVD F23, 88(R8)
	FMOVD F18, 96(R8)           // work[6]
	FMOVD F19, 104(R8)
	FMOVD F26, 112(R8)          // work[7]
	FMOVD F27, 120(R8)

neon8r4f64_stage2:
	// =========================================================================
	// Stage 2: radix-2 butterflies with twiddles
	// Pairs: (0,4), (1,5), (2,6), (3,7) with twiddles w0=1, w1, w2, w3
	// =========================================================================

	// Load twiddles w1, w2, w3 from twiddle[1], twiddle[2], twiddle[3]
	FMOVD 16(R10), F0           // w1.real (twiddle[1])
	FMOVD 24(R10), F1           // w1.imag
	FMOVD 32(R10), F2           // w2.real (twiddle[2])
	FMOVD 40(R10), F3           // w2.imag
	FMOVD 48(R10), F4           // w3.real (twiddle[3])
	FMOVD 56(R10), F5           // w3.imag

	// --- Butterfly (0, 4): w0 = 1, no multiplication needed ---
	FMOVD 0(R8), F6             // a0.real
	FMOVD 8(R8), F7             // a0.imag
	FMOVD 64(R8), F8            // a4.real
	FMOVD 72(R8), F9            // a4.imag

	FADDD F8, F6, F10           // out0.real = a0.r + a4.r
	FADDD F9, F7, F11           // out0.imag = a0.i + a4.i
	FSUBD F8, F6, F12           // out4.real = a0.r - a4.r
	FSUBD F9, F7, F13           // out4.imag = a0.i - a4.i

	FMOVD F10, 0(R8)            // store out0
	FMOVD F11, 8(R8)
	FMOVD F12, 64(R8)           // store out4
	FMOVD F13, 72(R8)

	// --- Butterfly (1, 5): multiply a5 by w1 ---
	FMOVD 16(R8), F6            // a1.real
	FMOVD 24(R8), F7            // a1.imag
	FMOVD 80(R8), F8            // a5.real
	FMOVD 88(R8), F9            // a5.imag

	// w1 * a5 = (w1.r*a5.r - w1.i*a5.i, w1.r*a5.i + w1.i*a5.r)
	FMULD F0, F8, F14           // w1.r * a5.r
	FMULD F1, F9, F15           // w1.i * a5.i
	FSUBD F15, F14, F14         // wb.real = w1.r*a5.r - w1.i*a5.i
	FMULD F0, F9, F15           // w1.r * a5.i
	FMULD F1, F8, F16           // w1.i * a5.r
	FADDD F16, F15, F15         // wb.imag = w1.r*a5.i + w1.i*a5.r

	FADDD F14, F6, F10          // out1.real = a1.r + wb.r
	FADDD F15, F7, F11          // out1.imag = a1.i + wb.i
	FSUBD F14, F6, F12          // out5.real = a1.r - wb.r
	FSUBD F15, F7, F13          // out5.imag = a1.i - wb.i

	FMOVD F10, 16(R8)           // store out1
	FMOVD F11, 24(R8)
	FMOVD F12, 80(R8)           // store out5
	FMOVD F13, 88(R8)

	// --- Butterfly (2, 6): multiply a6 by w2 ---
	FMOVD 32(R8), F6            // a2.real
	FMOVD 40(R8), F7            // a2.imag
	FMOVD 96(R8), F8            // a6.real
	FMOVD 104(R8), F9           // a6.imag

	// w2 * a6
	FMULD F2, F8, F14           // w2.r * a6.r
	FMULD F3, F9, F15           // w2.i * a6.i
	FSUBD F15, F14, F14         // wb.real
	FMULD F2, F9, F15           // w2.r * a6.i
	FMULD F3, F8, F16           // w2.i * a6.r
	FADDD F16, F15, F15         // wb.imag

	FADDD F14, F6, F10          // out2.real
	FADDD F15, F7, F11          // out2.imag
	FSUBD F14, F6, F12          // out6.real
	FSUBD F15, F7, F13          // out6.imag

	FMOVD F10, 32(R8)           // store out2
	FMOVD F11, 40(R8)
	FMOVD F12, 96(R8)           // store out6
	FMOVD F13, 104(R8)

	// --- Butterfly (3, 7): multiply a7 by w3 ---
	FMOVD 48(R8), F6            // a3.real
	FMOVD 56(R8), F7            // a3.imag
	FMOVD 112(R8), F8           // a7.real
	FMOVD 120(R8), F9           // a7.imag

	// w3 * a7
	FMULD F4, F8, F14           // w3.r * a7.r
	FMULD F5, F9, F15           // w3.i * a7.i
	FSUBD F15, F14, F14         // wb.real
	FMULD F4, F9, F15           // w3.r * a7.i
	FMULD F5, F8, F16           // w3.i * a7.r
	FADDD F16, F15, F15         // wb.imag

	FADDD F14, F6, F10          // out3.real
	FADDD F15, F7, F11          // out3.imag
	FSUBD F14, F6, F12          // out7.real
	FSUBD F15, F7, F13          // out7.imag

	FMOVD F10, 48(R8)           // store out3
	FMOVD F11, 56(R8)
	FMOVD F12, 112(R8)          // store out7
	FMOVD F13, 120(R8)

neon8r4f64_done:
	// =========================================================================
	// Copy back from scratch to dst if we used scratch buffer
	// =========================================================================
	CMP  R8, R20                // did we use scratch? (R8 != R20)
	BEQ  neon8r4f64_return_true // no, result already in dst

	MOVD $0, R0                 // R0 = loop counter
neon8r4f64_copy_loop:
	CMP  $8, R0                 // while i < 8
	BGE  neon8r4f64_return_true
	LSL  $4, R0, R1             // R1 = i * 16
	ADD  R8, R1, R2             // R2 = &scratch[i]
	MOVD (R2), R3               // R3 = scratch[i].real (as bits)
	MOVD 8(R2), R4              // R4 = scratch[i].imag (as bits)
	ADD  R20, R1, R5            // R5 = &dst[i]
	MOVD R3, (R5)               // dst[i].real = scratch[i].real
	MOVD R4, 8(R5)              // dst[i].imag = scratch[i].imag
	ADD  $1, R0, R0             // i++
	B    neon8r4f64_copy_loop

neon8r4f64_return_true:
	MOVD $1, R0                 // return true
	MOVB R0, ret+96(FP)
	RET

neon8r4f64_return_false:
	MOVD $0, R0                 // return false
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 8, complex128, radix-4 variant
// Same as forward but with conjugated twiddles and 1/N scaling
// ===========================================================================
TEXT ·InverseNEONSize8Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load slice headers from stack
	MOVD dst+0(FP), R8          // R8 = dst.data
	MOVD src+24(FP), R9         // R9 = src.data
	MOVD twiddle+48(FP), R10    // R10 = twiddle.data
	MOVD scratch+72(FP), R11    // R11 = scratch.data
	MOVD src+32(FP), R13        // R13 = src.len (n)

	// Validate n == 8
	CMP  $8, R13
	BNE  neon8r4f64_inv_return_false

	// Validate dst capacity >= 8
	MOVD dst+8(FP), R0
	CMP  $8, R0
	BLT  neon8r4f64_inv_return_false

	// Validate twiddle capacity >= 8
	MOVD twiddle+56(FP), R0
	CMP  $8, R0
	BLT  neon8r4f64_inv_return_false

	// Validate scratch capacity >= 8
	MOVD scratch+80(FP), R0
	CMP  $8, R0
	BLT  neon8r4f64_inv_return_false

	// Load bit-reversal table address
	MOVD $bitrev_size8_radix4_f64<>(SB), R12

	// Save original dst pointer
	MOVD R8, R20

	// If src == dst, use scratch as working buffer
	CMP  R8, R9
	BNE  neon8r4f64_inv_use_dst
	MOVD R11, R8

neon8r4f64_inv_use_dst:
	// =========================================================================
	// Bit-reversal permutation
	// =========================================================================
	MOVD $0, R0

neon8r4f64_inv_bitrev_loop:
	CMP  $8, R0
	BGE  neon8r4f64_inv_stage1

	LSL  $3, R0, R1             // offset into bitrev table
	ADD  R12, R1, R1
	MOVD (R1), R2               // R2 = bitrev[i]

	LSL  $4, R2, R3             // source offset (16 bytes per complex128)
	ADD  R9, R3, R3
	MOVD (R3), R4               // load real part
	MOVD 8(R3), R5              // load imag part

	LSL  $4, R0, R3             // dest offset
	ADD  R8, R3, R3
	MOVD R4, (R3)               // store real part
	MOVD R5, 8(R3)              // store imag part

	ADD  $1, R0, R0
	B    neon8r4f64_inv_bitrev_loop

neon8r4f64_inv_stage1:
	// =========================================================================
	// Stage 1: Two radix-4 butterflies (same as forward)
	// =========================================================================

	// --- First radix-4 butterfly: elements 0,1,2,3 ---
	FMOVD 0(R8), F0
	FMOVD 8(R8), F1
	FMOVD 16(R8), F2
	FMOVD 24(R8), F3
	FMOVD 32(R8), F4
	FMOVD 40(R8), F5
	FMOVD 48(R8), F6
	FMOVD 56(R8), F7

	FADDD F4, F0, F8
	FADDD F5, F1, F9
	FSUBD F4, F0, F10
	FSUBD F5, F1, F11

	FADDD F6, F2, F12
	FADDD F7, F3, F13
	FSUBD F6, F2, F14
	FSUBD F7, F3, F15

	FADDD F12, F8, F16
	FADDD F13, F9, F17
	FSUBD F12, F8, F18
	FSUBD F13, F9, F19

	// For inverse: i * t3 => (-t3.imag, t3.real) for a1
	FNEGD F15, F20              // i*t3.real = -t3.imag
	FMOVD F14, F21              // i*t3.imag = t3.real

	FADDD F20, F10, F22         // a1 = t1 + i*t3
	FADDD F21, F11, F23

	// For inverse: (-i) * t3 => (t3.imag, -t3.real) for a3
	FMOVD F15, F24              // (-i)*t3.real = t3.imag
	FNEGD F14, F25              // (-i)*t3.imag = -t3.real

	FADDD F24, F10, F26         // a3 = t1 + (-i)*t3
	FADDD F25, F11, F27

	FMOVD F16, 0(R8)
	FMOVD F17, 8(R8)
	FMOVD F22, 16(R8)
	FMOVD F23, 24(R8)
	FMOVD F18, 32(R8)
	FMOVD F19, 40(R8)
	FMOVD F26, 48(R8)
	FMOVD F27, 56(R8)

	// --- Second radix-4 butterfly: elements 4,5,6,7 ---
	FMOVD 64(R8), F0
	FMOVD 72(R8), F1
	FMOVD 80(R8), F2
	FMOVD 88(R8), F3
	FMOVD 96(R8), F4
	FMOVD 104(R8), F5
	FMOVD 112(R8), F6
	FMOVD 120(R8), F7

	FADDD F4, F0, F8
	FADDD F5, F1, F9
	FSUBD F4, F0, F10
	FSUBD F5, F1, F11

	FADDD F6, F2, F12
	FADDD F7, F3, F13
	FSUBD F6, F2, F14
	FSUBD F7, F3, F15

	FADDD F12, F8, F16
	FADDD F13, F9, F17
	FSUBD F12, F8, F18
	FSUBD F13, F9, F19

	// For inverse: i * t3
	FNEGD F15, F20
	FMOVD F14, F21

	FADDD F20, F10, F22
	FADDD F21, F11, F23

	// For inverse: (-i) * t3
	FMOVD F15, F24
	FNEGD F14, F25

	FADDD F24, F10, F26
	FADDD F25, F11, F27

	FMOVD F16, 64(R8)
	FMOVD F17, 72(R8)
	FMOVD F22, 80(R8)
	FMOVD F23, 88(R8)
	FMOVD F18, 96(R8)
	FMOVD F19, 104(R8)
	FMOVD F26, 112(R8)
	FMOVD F27, 120(R8)

neon8r4f64_inv_stage2:
	// =========================================================================
	// Stage 2: radix-2 butterflies with conjugated twiddles
	// =========================================================================

	// Load twiddles w1, w2, w3 and conjugate (negate imaginary)
	FMOVD 16(R10), F0           // w1.real
	FMOVD 24(R10), F1           // w1.imag
	FNEGD F1, F1                // conjugate
	FMOVD 32(R10), F2           // w2.real
	FMOVD 40(R10), F3           // w2.imag
	FNEGD F3, F3                // conjugate
	FMOVD 48(R10), F4           // w3.real
	FMOVD 56(R10), F5           // w3.imag
	FNEGD F5, F5                // conjugate

	// --- Butterfly (0, 4): w0 = 1 ---
	FMOVD 0(R8), F6
	FMOVD 8(R8), F7
	FMOVD 64(R8), F8
	FMOVD 72(R8), F9

	FADDD F8, F6, F10
	FADDD F9, F7, F11
	FSUBD F8, F6, F12
	FSUBD F9, F7, F13

	FMOVD F10, 0(R8)
	FMOVD F11, 8(R8)
	FMOVD F12, 64(R8)
	FMOVD F13, 72(R8)

	// --- Butterfly (1, 5): multiply a5 by w1* ---
	FMOVD 16(R8), F6
	FMOVD 24(R8), F7
	FMOVD 80(R8), F8
	FMOVD 88(R8), F9

	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15

	FADDD F14, F6, F10
	FADDD F15, F7, F11
	FSUBD F14, F6, F12
	FSUBD F15, F7, F13

	FMOVD F10, 16(R8)
	FMOVD F11, 24(R8)
	FMOVD F12, 80(R8)
	FMOVD F13, 88(R8)

	// --- Butterfly (2, 6): multiply a6 by w2* ---
	FMOVD 32(R8), F6
	FMOVD 40(R8), F7
	FMOVD 96(R8), F8
	FMOVD 104(R8), F9

	FMULD F2, F8, F14
	FMULD F3, F9, F15
	FSUBD F15, F14, F14
	FMULD F2, F9, F15
	FMULD F3, F8, F16
	FADDD F16, F15, F15

	FADDD F14, F6, F10
	FADDD F15, F7, F11
	FSUBD F14, F6, F12
	FSUBD F15, F7, F13

	FMOVD F10, 32(R8)
	FMOVD F11, 40(R8)
	FMOVD F12, 96(R8)
	FMOVD F13, 104(R8)

	// --- Butterfly (3, 7): multiply a7 by w3* ---
	FMOVD 48(R8), F6
	FMOVD 56(R8), F7
	FMOVD 112(R8), F8
	FMOVD 120(R8), F9

	FMULD F4, F8, F14
	FMULD F5, F9, F15
	FSUBD F15, F14, F14
	FMULD F4, F9, F15
	FMULD F5, F8, F16
	FADDD F16, F15, F15

	FADDD F14, F6, F10
	FADDD F15, F7, F11
	FSUBD F14, F6, F12
	FSUBD F15, F7, F13

	FMOVD F10, 48(R8)
	FMOVD F11, 56(R8)
	FMOVD F12, 112(R8)
	FMOVD F13, 120(R8)

neon8r4f64_inv_done:
	// =========================================================================
	// Copy back if we used scratch
	// =========================================================================
	CMP  R8, R20
	BEQ  neon8r4f64_inv_scale

	MOVD $0, R0
neon8r4f64_inv_copy_loop:
	CMP  $8, R0
	BGE  neon8r4f64_inv_scale
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon8r4f64_inv_copy_loop

neon8r4f64_inv_scale:
	// =========================================================================
	// Apply 1/N scaling for inverse transform
	// =========================================================================
	MOVD $·neonInv8F64(SB), R1  // R1 = &(1/8)
	FMOVD (R1), F0              // F0 = 1/8 = 0.125
	MOVD $0, R0                 // R0 = loop counter

neon8r4f64_inv_scale_loop:
	CMP  $8, R0
	BGE  neon8r4f64_inv_return_true
	LSL  $4, R0, R1             // R1 = i * 16
	ADD  R20, R1, R1            // R1 = &dst[i]
	FMOVD 0(R1), F2             // F2 = dst[i].real
	FMOVD 8(R1), F3             // F3 = dst[i].imag
	FMULD F0, F2, F2            // F2 = dst[i].real / 8
	FMULD F0, F3, F3            // F3 = dst[i].imag / 8
	FMOVD F2, 0(R1)             // store scaled real
	FMOVD F3, 8(R1)             // store scaled imag
	ADD  $1, R0, R0             // i++
	B    neon8r4f64_inv_scale_loop

neon8r4f64_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon8r4f64_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Data Section - Bit-reversal permutation table for size 8 radix-4
// ===========================================================================
// Each entry is 8 bytes (uint64), total 8 entries = 64 bytes
// For radix-4 decomposition of size 8: [0,2,4,6,1,3,5,7]
// This groups even indices first (for first radix-4), then odd (for second)

DATA bitrev_size8_radix4_f64<>+0x000(SB)/8, $0   // bitrev[0] = 0
DATA bitrev_size8_radix4_f64<>+0x008(SB)/8, $2   // bitrev[1] = 2
DATA bitrev_size8_radix4_f64<>+0x010(SB)/8, $4   // bitrev[2] = 4
DATA bitrev_size8_radix4_f64<>+0x018(SB)/8, $6   // bitrev[3] = 6
DATA bitrev_size8_radix4_f64<>+0x020(SB)/8, $1   // bitrev[4] = 1
DATA bitrev_size8_radix4_f64<>+0x028(SB)/8, $3   // bitrev[5] = 3
DATA bitrev_size8_radix4_f64<>+0x030(SB)/8, $5   // bitrev[6] = 5
DATA bitrev_size8_radix4_f64<>+0x038(SB)/8, $7   // bitrev[7] = 7
GLOBL bitrev_size8_radix4_f64<>(SB), RODATA, $64

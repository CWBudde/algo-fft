//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-16 Radix-4 FFT Kernels for ARM64 (complex128)
// ===========================================================================
//
// Size 16 = 4^2, radix-4 algorithm uses 2 stages:
//   Stage 1: 4 radix-4 butterflies, stride=4
//   Stage 2: 1 group, 4 butterflies, twiddle step=1
//
// ===========================================================================

#include "textflag.h"

DATA ·neonInv16F64+0(SB)/8, $0x3fb0000000000000 // 1/16
GLOBL ·neonInv16F64(SB), RODATA, $8

// Forward transform, size 16, complex128, radix-4 variant
TEXT ·ForwardNEONSize16Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVD dst+0(FP), R8           // R8  = dst pointer
	MOVD src+24(FP), R9          // R9  = src pointer
	MOVD twiddle+48(FP), R10     // R10 = twiddle pointer
	MOVD scratch+72(FP), R11     // R11 = scratch pointer
	MOVD src+32(FP), R13         // R13 = n (should be 16)

	// Verify n == 16
	CMP  $16, R13
	BNE  neon16r4f64_return_false

	// Validate all slice lengths >= 16
	MOVD dst+8(FP), R0
	CMP  $16, R0
	BLT  neon16r4f64_return_false

	MOVD twiddle+56(FP), R0
	CMP  $16, R0
	BLT  neon16r4f64_return_false

	MOVD scratch+80(FP), R0
	CMP  $16, R0
	BLT  neon16r4f64_return_false

	// Load static bit-reversal table
	MOVD $bitrev_size16_radix4<>(SB), R12

	// Preserve dst pointer
	MOVD R8, R20

	// Select working buffer
	CMP  R8, R9
	BNE  neon16r4f64_use_dst
	MOVD R11, R8

neon16r4f64_use_dst:
	// =======================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =======================================================================
	MOVD $0, R0

neon16r4f64_bitrev_loop:
	CMP  $16, R0
	BGE  neon16r4f64_stage1

	LSL  $3, R0, R1
	ADD  R12, R1, R1
	MOVD (R1), R2

	LSL  $4, R2, R3
	ADD  R9, R3, R3
	MOVD (R3), R4
	MOVD 8(R3), R5

	LSL  $4, R0, R3
	ADD  R8, R3, R3
	MOVD R4, (R3)
	MOVD R5, 8(R3)

	ADD  $1, R0, R0
	B    neon16r4f64_bitrev_loop

neon16r4f64_stage1:
	// =======================================================================
	// Stage 1: 4 radix-4 butterflies, stride=4
	// =======================================================================
	MOVD $0, R0

neon16r4f64_stage1_loop:
	CMP  $16, R0
	BGE  neon16r4f64_stage2

	// addr = &work[base]
	LSL  $4, R0, R1
	ADD  R8, R1, R1

	// Load a0..a3
	FMOVD 0(R1), F0
	FMOVD 8(R1), F1
	FMOVD 16(R1), F2
	FMOVD 24(R1), F3
	FMOVD 32(R1), F4
	FMOVD 40(R1), F5
	FMOVD 48(R1), F6
	FMOVD 56(R1), F7

	// t0 = a0 + a2, t1 = a0 - a2
	FADDD F4, F0, F8
	FADDD F5, F1, F9
	FSUBD F4, F0, F10
	FSUBD F5, F1, F11

	// t2 = a1 + a3, t3 = a1 - a3
	FADDD F6, F2, F12
	FADDD F7, F3, F13
	FSUBD F6, F2, F14
	FSUBD F7, F3, F15

	// b0 = t0 + t2, b2 = t0 - t2
	FADDD F12, F8, F16
	FADDD F13, F9, F17
	FSUBD F12, F8, F18
	FSUBD F13, F9, F19

	// (-i) * t3 => (t3.imag, -t3.real)
	FMOVD F15, F20
	FNEGD F14, F21

	// b1 = t1 + (-i)*t3
	FADDD F20, F10, F22
	FADDD F21, F11, F23

	// i * t3 => (-t3.imag, t3.real)
	FNEGD F15, F24
	FMOVD F14, F25

	// b3 = t1 + i*t3
	FADDD F24, F10, F26
	FADDD F25, F11, F27

	// Store results
	FMOVD F16, 0(R1)
	FMOVD F17, 8(R1)
	FMOVD F22, 16(R1)
	FMOVD F23, 24(R1)
	FMOVD F18, 32(R1)
	FMOVD F19, 40(R1)
	FMOVD F26, 48(R1)
	FMOVD F27, 56(R1)

	ADD  $4, R0, R0
	B    neon16r4f64_stage1_loop

neon16r4f64_stage2:
	// =======================================================================
	// Stage 2: 1 group, 4 butterflies, twiddle step=1
	// =======================================================================
	MOVD $0, R0

neon16r4f64_stage2_loop:
	CMP  $4, R0
	BGE  neon16r4f64_done

	// idx0=j, idx1=j+4, idx2=j+8, idx3=j+12
	MOVD R0, R1                // R1 = j
	ADD  $4, R1, R2            // R2 = j+4
	ADD  $8, R1, R3            // R3 = j+8
	ADD  $12, R1, R4           // R4 = j+12

	// Load twiddles: w1=tw[j], w2=tw[2j], w3=tw[3j]
	LSL  $4, R1, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1

	LSL  $1, R1, R6
	ADD  R6, R1, R7
	LSL  $4, R6, R6
	ADD  R10, R6, R6
	FMOVD 0(R6), F2
	FMOVD 8(R6), F3

	LSL  $4, R7, R7
	ADD  R10, R7, R7            // R7 = &tw[3j]
	FMOVD 0(R7), F4
	FMOVD 8(R7), F5

	// Load a0
	LSL  $4, R1, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F6
	FMOVD 8(R5), F7

	// Load a1
	LSL  $4, R2, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F8
	FMOVD 8(R5), F9

	// Load a2
	LSL  $4, R3, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F10
	FMOVD 8(R5), F11

	// Load a3
	LSL  $4, R4, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F12
	FMOVD 8(R5), F13

	// a1 = w1 * a1
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	// a2 = w2 * a2
	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	// a3 = w3 * a3
	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// t0 = a0 + a2, t1 = a0 - a2
	FADDD F10, F6, F14
	FADDD F11, F7, F15
	FSUBD F10, F6, F16
	FSUBD F11, F7, F17

	// t2 = a1 + a3, t3 = a1 - a3
	FADDD F12, F8, F18
	FADDD F13, F9, F19
	FSUBD F12, F8, F20
	FSUBD F13, F9, F21

	// b0 = t0 + t2, b2 = t0 - t2
	FADDD F18, F14, F22
	FADDD F19, F15, F23
	FSUBD F18, F14, F24
	FSUBD F19, F15, F25

	// (-i) * t3
	FMOVD F21, F26
	FNEGD F20, F27

	// b1 = t1 + (-i)*t3
	FADDD F26, F16, F28
	FADDD F27, F17, F29

	// i * t3
	FNEGD F21, F30
	FMOVD F20, F31

	// b3 = t1 + i*t3
	FADDD F30, F16, F20
	FADDD F31, F17, F21

	// Store results
	LSL  $4, R1, R5
	ADD  R8, R5, R5
	FMOVD F22, 0(R5)
	FMOVD F23, 8(R5)

	LSL  $4, R2, R5
	ADD  R8, R5, R5
	FMOVD F28, 0(R5)
	FMOVD F29, 8(R5)

	LSL  $4, R3, R5
	ADD  R8, R5, R5
	FMOVD F24, 0(R5)
	FMOVD F25, 8(R5)

	LSL  $4, R4, R5
	ADD  R8, R5, R5
	FMOVD F20, 0(R5)
	FMOVD F21, 8(R5)

	ADD  $1, R0, R0
	B    neon16r4f64_stage2_loop

neon16r4f64_done:
	// Copy back if we used scratch
	CMP  R8, R20
	BEQ  neon16r4f64_return_true

	MOVD $0, R0
neon16r4f64_copy_loop:
	CMP  $16, R0
	BGE  neon16r4f64_return_true
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon16r4f64_copy_loop

neon16r4f64_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon16r4f64_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Inverse transform, size 16, complex128, radix-4 variant
TEXT ·InverseNEONSize16Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $16, R13
	BNE  neon16r4f64_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $16, R0
	BLT  neon16r4f64_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $16, R0
	BLT  neon16r4f64_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $16, R0
	BLT  neon16r4f64_inv_return_false

	// Load static bit-reversal table
	MOVD $bitrev_size16_radix4<>(SB), R12

	MOVD R8, R20

	CMP  R8, R9
	BNE  neon16r4f64_inv_use_dst
	MOVD R11, R8

neon16r4f64_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon16r4f64_inv_bitrev_loop:
	CMP  $16, R0
	BGE  neon16r4f64_inv_stage1

	LSL  $3, R0, R1
	ADD  R12, R1, R1
	MOVD (R1), R2

	LSL  $4, R2, R3
	ADD  R9, R3, R3
	MOVD (R3), R4
	MOVD 8(R3), R5

	LSL  $4, R0, R3
	ADD  R8, R3, R3
	MOVD R4, (R3)
	MOVD R5, 8(R3)

	ADD  $1, R0, R0
	B    neon16r4f64_inv_bitrev_loop

neon16r4f64_inv_stage1:
	// Stage 1 (same as forward)
	MOVD $0, R0

neon16r4f64_inv_stage1_loop:
	CMP  $16, R0
	BGE  neon16r4f64_inv_stage2

	LSL  $4, R0, R1
	ADD  R8, R1, R1

	FMOVD 0(R1), F0
	FMOVD 8(R1), F1
	FMOVD 16(R1), F2
	FMOVD 24(R1), F3
	FMOVD 32(R1), F4
	FMOVD 40(R1), F5
	FMOVD 48(R1), F6
	FMOVD 56(R1), F7

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

	FNEGD F15, F20
	FMOVD F14, F21
	FADDD F20, F10, F22
	FADDD F21, F11, F23

	FMOVD F15, F24
	FNEGD F14, F25
	FADDD F24, F10, F26
	FADDD F25, F11, F27

	FMOVD F16, 0(R1)
	FMOVD F17, 8(R1)
	FMOVD F22, 16(R1)
	FMOVD F23, 24(R1)
	FMOVD F18, 32(R1)
	FMOVD F19, 40(R1)
	FMOVD F26, 48(R1)
	FMOVD F27, 56(R1)

	ADD  $4, R0, R0
	B    neon16r4f64_inv_stage1_loop

neon16r4f64_inv_stage2:
	// Stage 2 with conjugated twiddles
	MOVD $0, R0

neon16r4f64_inv_stage2_loop:
	CMP  $4, R0
	BGE  neon16r4f64_inv_done

	MOVD R0, R1
	ADD  $4, R1, R2
	ADD  $8, R1, R3
	ADD  $12, R1, R4

	LSL  $4, R1, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1
	FNEGD F1, F1

	LSL  $1, R1, R6
	ADD  R6, R1, R7
	LSL  $4, R6, R6
	ADD  R10, R6, R6
	FMOVD 0(R6), F2
	FMOVD 8(R6), F3
	FNEGD F3, F3

	LSL  $4, R7, R7
	ADD  R10, R7, R7
	FMOVD 0(R7), F4
	FMOVD 8(R7), F5
	FNEGD F5, F5

	LSL  $4, R1, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F6
	FMOVD 8(R5), F7

	LSL  $4, R2, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F8
	FMOVD 8(R5), F9

	LSL  $4, R3, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F10
	FMOVD 8(R5), F11

	LSL  $4, R4, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F12
	FMOVD 8(R5), F13

	// a1 = w1 * a1
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	// a2 = w2 * a2
	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	// a3 = w3 * a3
	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	FADDD F10, F6, F14
	FADDD F11, F7, F15
	FSUBD F10, F6, F16
	FSUBD F11, F7, F17

	FADDD F12, F8, F18
	FADDD F13, F9, F19
	FSUBD F12, F8, F20
	FSUBD F13, F9, F21

	FADDD F18, F14, F22
	FADDD F19, F15, F23
	FSUBD F18, F14, F24
	FSUBD F19, F15, F25

	FNEGD F21, F26
	FMOVD F20, F27
	FADDD F26, F16, F28
	FADDD F27, F17, F29

	FMOVD F21, F30
	FNEGD F20, F31
	FADDD F30, F16, F20
	FADDD F31, F17, F21

	LSL  $4, R1, R5
	ADD  R8, R5, R5
	FMOVD F22, 0(R5)
	FMOVD F23, 8(R5)

	LSL  $4, R2, R5
	ADD  R8, R5, R5
	FMOVD F28, 0(R5)
	FMOVD F29, 8(R5)

	LSL  $4, R3, R5
	ADD  R8, R5, R5
	FMOVD F24, 0(R5)
	FMOVD F25, 8(R5)

	LSL  $4, R4, R5
	ADD  R8, R5, R5
	FMOVD F20, 0(R5)
	FMOVD F21, 8(R5)

	ADD  $1, R0, R0
	B    neon16r4f64_inv_stage2_loop

neon16r4f64_inv_done:
	// Copy back if we used scratch
	CMP  R8, R20
	BEQ  neon16r4f64_inv_scale

	MOVD $0, R0
neon16r4f64_inv_copy_loop:
	CMP  $16, R0
	BGE  neon16r4f64_inv_scale
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon16r4f64_inv_copy_loop

neon16r4f64_inv_scale:
	// Apply 1/16 scaling
	MOVD $·neonInv16F64(SB), R1
	FMOVD (R1), F0
	MOVD $0, R0

neon16r4f64_inv_scale_loop:
	CMP  $16, R0
	BGE  neon16r4f64_inv_return_true
	LSL  $4, R0, R1
	ADD  R20, R1, R1
	FMOVD 0(R1), F2
	FMOVD 8(R1), F3
	FMULD F0, F2, F2
	FMULD F0, F3, F3
	FMOVD F2, 0(R1)
	FMOVD F3, 8(R1)
	ADD  $1, R0, R0
	B    neon16r4f64_inv_scale_loop

neon16r4f64_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon16r4f64_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Bit-reversal table for size 16 radix-4
GLOBL bitrev_size16_radix4<>(SB), RODATA, $128
DATA bitrev_size16_radix4<>+0(SB)/8, $0
DATA bitrev_size16_radix4<>+8(SB)/8, $4
DATA bitrev_size16_radix4<>+16(SB)/8, $8
DATA bitrev_size16_radix4<>+24(SB)/8, $12
DATA bitrev_size16_radix4<>+32(SB)/8, $1
DATA bitrev_size16_radix4<>+40(SB)/8, $5
DATA bitrev_size16_radix4<>+48(SB)/8, $9
DATA bitrev_size16_radix4<>+56(SB)/8, $13
DATA bitrev_size16_radix4<>+64(SB)/8, $2
DATA bitrev_size16_radix4<>+72(SB)/8, $6
DATA bitrev_size16_radix4<>+80(SB)/8, $10
DATA bitrev_size16_radix4<>+88(SB)/8, $14
DATA bitrev_size16_radix4<>+96(SB)/8, $3
DATA bitrev_size16_radix4<>+104(SB)/8, $7
DATA bitrev_size16_radix4<>+112(SB)/8, $11
DATA bitrev_size16_radix4<>+120(SB)/8, $15

//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-16 Radix-4 FFT Kernels for ARM64
// ===========================================================================
//
// Size 16 = 4^2, radix-4 algorithm uses 2 stages:
//   Stage 1: 4 radix-4 butterflies, stride=4
//   Stage 2: 1 group, 4 butterflies, twiddle step=1
//
// ===========================================================================

#include "textflag.h"

DATA ·neonInv16+0(SB)/4, $0x3d800000 // 1/16
GLOBL ·neonInv16(SB), RODATA, $4

// Forward transform, size 16, complex64, radix-4 variant
TEXT ·ForwardNEONSize16Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVD dst+0(FP), R8           // R8  = dst pointer
	MOVD src+24(FP), R9          // R9  = src pointer
	MOVD twiddle+48(FP), R10     // R10 = twiddle pointer
	MOVD scratch+72(FP), R11     // R11 = scratch pointer
	MOVD bitrev+96(FP), R12      // R12 = bitrev pointer
	MOVD src+32(FP), R13         // R13 = n (should be 16)

	// Verify n == 16
	CMP  $16, R13
	BNE  neon16r4_return_false

	// Validate all slice lengths >= 16
	MOVD dst+8(FP), R0
	CMP  $16, R0
	BLT  neon16r4_return_false

	MOVD twiddle+56(FP), R0
	CMP  $16, R0
	BLT  neon16r4_return_false

	MOVD scratch+80(FP), R0
	CMP  $16, R0
	BLT  neon16r4_return_false

	MOVD bitrev+104(FP), R0
	CMP  $16, R0
	BLT  neon16r4_return_false

	// Preserve dst pointer
	MOVD R8, R20

	// Select working buffer
	CMP  R8, R9
	BNE  neon16r4_use_dst
	MOVD R11, R8

neon16r4_use_dst:
	// =======================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =======================================================================
	MOVD $0, R0

neon16r4_bitrev_loop:
	CMP  $16, R0
	BGE  neon16r4_stage1

	LSL  $3, R0, R1
	ADD  R12, R1, R1
	MOVD (R1), R2

	LSL  $3, R2, R3
	ADD  R9, R3, R3
	MOVD (R3), R4

	LSL  $3, R0, R3
	ADD  R8, R3, R3
	MOVD R4, (R3)

	ADD  $1, R0, R0
	B    neon16r4_bitrev_loop

neon16r4_stage1:
	// =======================================================================
	// Stage 1: 4 radix-4 butterflies, stride=4
	// =======================================================================
	MOVD $0, R0

neon16r4_stage1_loop:
	CMP  $16, R0
	BGE  neon16r4_stage2

	// addr = &work[base]
	LSL  $3, R0, R1
	ADD  R8, R1, R1

	// Load a0..a3
	FMOVS 0(R1), F0
	FMOVS 4(R1), F1
	FMOVS 8(R1), F2
	FMOVS 12(R1), F3
	FMOVS 16(R1), F4
	FMOVS 20(R1), F5
	FMOVS 24(R1), F6
	FMOVS 28(R1), F7

	// t0 = a0 + a2, t1 = a0 - a2
	FADDS F4, F0, F8
	FADDS F5, F1, F9
	FSUBS F4, F0, F10
	FSUBS F5, F1, F11

	// t2 = a1 + a3, t3 = a1 - a3
	FADDS F6, F2, F12
	FADDS F7, F3, F13
	FSUBS F6, F2, F14
	FSUBS F7, F3, F15

	// b0 = t0 + t2, b2 = t0 - t2
	FADDS F12, F8, F16
	FADDS F13, F9, F17
	FSUBS F12, F8, F18
	FSUBS F13, F9, F19

	// (-i) * t3 => (t3.imag, -t3.real)
	FMOVS F15, F20
	FNEGS  F14, F21

	// b1 = t1 + (-i)*t3
	FADDS F20, F10, F22
	FADDS F21, F11, F23

	// i * t3 => (-t3.imag, t3.real)
	FNEGS  F15, F24
	FMOVS F14, F25

	// b3 = t1 + i*t3
	FADDS F24, F10, F26
	FADDS F25, F11, F27

	// Store results
	FMOVS F16, 0(R1)
	FMOVS F17, 4(R1)
	FMOVS F22, 8(R1)
	FMOVS F23, 12(R1)
	FMOVS F18, 16(R1)
	FMOVS F19, 20(R1)
	FMOVS F26, 24(R1)
	FMOVS F27, 28(R1)

	ADD  $4, R0, R0
	B    neon16r4_stage1_loop

neon16r4_stage2:
	// =======================================================================
	// Stage 2: 1 group, 4 butterflies, twiddle step=1
	// =======================================================================
	MOVD $0, R0

neon16r4_stage2_loop:
	CMP  $4, R0
	BGE  neon16r4_done

	// idx0=j, idx1=j+4, idx2=j+8, idx3=j+12
	MOVD R0, R1                // R1 = j
	ADD  $4, R1, R2            // R2 = j+4
	ADD  $8, R1, R3            // R3 = j+8
	ADD  $12, R1, R4           // R4 = j+12

	// Load twiddles: w1=tw[j], w2=tw[2j], w3=tw[3j]
	LSL  $3, R1, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1

	LSL  $1, R1, R6
	ADD  R6, R1, R7
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F2
	FMOVS 4(R6), F3

	LSL  $3, R7, R7
	ADD  R10, R7, R7            // R7 = &tw[3j]
	FMOVS 0(R7), F4
	FMOVS 4(R7), F5

	// Load a0
	LSL  $3, R1, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F6
	FMOVS 4(R5), F7

	// Load a1
	LSL  $3, R2, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F8
	FMOVS 4(R5), F9

	// Load a2
	LSL  $3, R3, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F10
	FMOVS 4(R5), F11

	// Load a3
	LSL  $3, R4, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F12
	FMOVS 4(R5), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	// a2 = w2 * a2
	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

	// a3 = w3 * a3
	FMULS F4, F12, F14
	FMULS F5, F13, F15
	FSUBS F15, F14, F14
	FMULS F4, F13, F15
	FMULS F5, F12, F16
	FADDS F16, F15, F15
	FMOVS F14, F12
	FMOVS F15, F13

	// t0 = a0 + a2, t1 = a0 - a2
	FADDS F10, F6, F14
	FADDS F11, F7, F15
	FSUBS F10, F6, F16
	FSUBS F11, F7, F17

	// t2 = a1 + a3, t3 = a1 - a3
	FADDS F12, F8, F18
	FADDS F13, F9, F19
	FSUBS F12, F8, F20
	FSUBS F13, F9, F21

	// b0 = t0 + t2, b2 = t0 - t2
	FADDS F18, F14, F22
	FADDS F19, F15, F23
	FSUBS F18, F14, F24
	FSUBS F19, F15, F25

	// (-i) * t3
	FMOVS F21, F26
	FNEGS  F20, F27

	// b1 = t1 + (-i)*t3
	FADDS F26, F16, F28
	FADDS F27, F17, F29

	// i * t3
	FNEGS  F21, F30
	FMOVS F20, F31

	// b3 = t1 + i*t3
	FADDS F30, F16, F20
	FADDS F31, F17, F21

	// Store results
	LSL  $3, R1, R5
	ADD  R8, R5, R5
	FMOVS F22, 0(R5)
	FMOVS F23, 4(R5)

	LSL  $3, R2, R5
	ADD  R8, R5, R5
	FMOVS F28, 0(R5)
	FMOVS F29, 4(R5)

	LSL  $3, R3, R5
	ADD  R8, R5, R5
	FMOVS F24, 0(R5)
	FMOVS F25, 4(R5)

	LSL  $3, R4, R5
	ADD  R8, R5, R5
	FMOVS F20, 0(R5)
	FMOVS F21, 4(R5)

	ADD  $1, R0, R0
	B    neon16r4_stage2_loop

neon16r4_done:
	// Copy back if we used scratch
	CMP  R8, R20
	BEQ  neon16r4_return_true

	MOVD $0, R0
neon16r4_copy_loop:
	CMP  $16, R0
	BGE  neon16r4_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon16r4_copy_loop

neon16r4_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon16r4_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// Inverse transform, size 16, complex64, radix-4 variant
TEXT ·InverseNEONSize16Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD bitrev+96(FP), R12
	MOVD src+32(FP), R13

	CMP  $16, R13
	BNE  neon16r4_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $16, R0
	BLT  neon16r4_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $16, R0
	BLT  neon16r4_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $16, R0
	BLT  neon16r4_inv_return_false

	MOVD bitrev+104(FP), R0
	CMP  $16, R0
	BLT  neon16r4_inv_return_false

	MOVD R8, R20

	CMP  R8, R9
	BNE  neon16r4_inv_use_dst
	MOVD R11, R8

neon16r4_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon16r4_inv_bitrev_loop:
	CMP  $16, R0
	BGE  neon16r4_inv_stage1

	LSL  $3, R0, R1
	ADD  R12, R1, R1
	MOVD (R1), R2

	LSL  $3, R2, R3
	ADD  R9, R3, R3
	MOVD (R3), R4

	LSL  $3, R0, R3
	ADD  R8, R3, R3
	MOVD R4, (R3)

	ADD  $1, R0, R0
	B    neon16r4_inv_bitrev_loop

neon16r4_inv_stage1:
	// Stage 1 (same as forward)
	MOVD $0, R0

neon16r4_inv_stage1_loop:
	CMP  $16, R0
	BGE  neon16r4_inv_stage2

	LSL  $3, R0, R1
	ADD  R8, R1, R1

	FMOVS 0(R1), F0
	FMOVS 4(R1), F1
	FMOVS 8(R1), F2
	FMOVS 12(R1), F3
	FMOVS 16(R1), F4
	FMOVS 20(R1), F5
	FMOVS 24(R1), F6
	FMOVS 28(R1), F7

	FADDS F4, F0, F8
	FADDS F5, F1, F9
	FSUBS F4, F0, F10
	FSUBS F5, F1, F11

	FADDS F6, F2, F12
	FADDS F7, F3, F13
	FSUBS F6, F2, F14
	FSUBS F7, F3, F15

	FADDS F12, F8, F16
	FADDS F13, F9, F17
	FSUBS F12, F8, F18
	FSUBS F13, F9, F19

	FNEGS F15, F20
	FMOVS F14, F21
	FADDS F20, F10, F22
	FADDS F21, F11, F23

	FMOVS F15, F24
	FNEGS F14, F25
	FADDS F24, F10, F26
	FADDS F25, F11, F27

	FMOVS F16, 0(R1)
	FMOVS F17, 4(R1)
	FMOVS F22, 8(R1)
	FMOVS F23, 12(R1)
	FMOVS F18, 16(R1)
	FMOVS F19, 20(R1)
	FMOVS F26, 24(R1)
	FMOVS F27, 28(R1)

	ADD  $4, R0, R0
	B    neon16r4_inv_stage1_loop

neon16r4_inv_stage2:
	// Stage 2 with conjugated twiddles
	MOVD $0, R0

neon16r4_inv_stage2_loop:
	CMP  $4, R0
	BGE  neon16r4_inv_done

	MOVD R0, R1
	ADD  $4, R1, R2
	ADD  $8, R1, R3
	ADD  $12, R1, R4

	LSL  $3, R1, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1
	FNEGS  F1, F1

	LSL  $1, R1, R6
	ADD  R6, R1, R7
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F2
	FMOVS 4(R6), F3
	FNEGS  F3, F3

	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F4
	FMOVS 4(R7), F5
	FNEGS  F5, F5

	LSL  $3, R1, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F6
	FMOVS 4(R5), F7

	LSL  $3, R2, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F8
	FMOVS 4(R5), F9

	LSL  $3, R3, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F10
	FMOVS 4(R5), F11

	LSL  $3, R4, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F12
	FMOVS 4(R5), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	// a2 = w2 * a2
	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

	// a3 = w3 * a3
	FMULS F4, F12, F14
	FMULS F5, F13, F15
	FSUBS F15, F14, F14
	FMULS F4, F13, F15
	FMULS F5, F12, F16
	FADDS F16, F15, F15
	FMOVS F14, F12
	FMOVS F15, F13

	FADDS F10, F6, F14
	FADDS F11, F7, F15
	FSUBS F10, F6, F16
	FSUBS F11, F7, F17

	FADDS F12, F8, F18
	FADDS F13, F9, F19
	FSUBS F12, F8, F20
	FSUBS F13, F9, F21

	FADDS F18, F14, F22
	FADDS F19, F15, F23
	FSUBS F18, F14, F24
	FSUBS F19, F15, F25

	FNEGS F21, F26
	FMOVS F20, F27
	FADDS F26, F16, F28
	FADDS F27, F17, F29

	FMOVS F21, F30
	FNEGS F20, F31
	FADDS F30, F16, F20
	FADDS F31, F17, F21

	LSL  $3, R1, R5
	ADD  R8, R5, R5
	FMOVS F22, 0(R5)
	FMOVS F23, 4(R5)

	LSL  $3, R2, R5
	ADD  R8, R5, R5
	FMOVS F28, 0(R5)
	FMOVS F29, 4(R5)

	LSL  $3, R3, R5
	ADD  R8, R5, R5
	FMOVS F24, 0(R5)
	FMOVS F25, 4(R5)

	LSL  $3, R4, R5
	ADD  R8, R5, R5
	FMOVS F20, 0(R5)
	FMOVS F21, 4(R5)

	ADD  $1, R0, R0
	B    neon16r4_inv_stage2_loop

neon16r4_inv_done:
	// Copy back if we used scratch
	CMP  R8, R20
	BEQ  neon16r4_inv_scale

	MOVD $0, R0
neon16r4_inv_copy_loop:
	CMP  $16, R0
	BGE  neon16r4_inv_scale
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon16r4_inv_copy_loop

neon16r4_inv_scale:
	// Apply 1/16 scaling
	MOVD $·neonInv16(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon16r4_inv_scale_loop:
	CMP  $16, R0
	BGE  neon16r4_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon16r4_inv_scale_loop

neon16r4_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon16r4_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

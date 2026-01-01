//go:build arm64 && fft_asm && !purego

// ===========================================================================
// NEON Size-8 Mixed-Radix (Radix-4 + Radix-2) FFT Kernels for ARM64
// ===========================================================================

#include "textflag.h"

// Forward transform, size 8, mixed radix
TEXT ·forwardNEONSize8Radix4Complex64Asm(SB), NOSPLIT, $0-121
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD bitrev+96(FP), R12
	MOVD src+32(FP), R13

	CMP  $8, R13
	BNE  neon8r4_return_false

	MOVD dst+8(FP), R0
	CMP  $8, R0
	BLT  neon8r4_return_false

	MOVD twiddle+56(FP), R0
	CMP  $8, R0
	BLT  neon8r4_return_false

	MOVD scratch+80(FP), R0
	CMP  $8, R0
	BLT  neon8r4_return_false

	MOVD bitrev+104(FP), R0
	CMP  $8, R0
	BLT  neon8r4_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon8r4_use_dst
	MOVD R11, R8

neon8r4_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon8r4_bitrev_loop:
	CMP  $8, R0
	BGE  neon8r4_stage1

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
	B    neon8r4_bitrev_loop

neon8r4_stage1:
	// Load x0,x2,x4,x6 and x1,x3,x5,x7 from work
	FMOVS 0(R8), F0
	FMOVS 4(R8), F1
	FMOVS 16(R8), F2
	FMOVS 20(R8), F3
	FMOVS 32(R8), F4
	FMOVS 36(R8), F5
	FMOVS 48(R8), F6
	FMOVS 52(R8), F7

	FMOVS 8(R8), F8
	FMOVS 12(R8), F9
	FMOVS 24(R8), F10
	FMOVS 28(R8), F11
	FMOVS 40(R8), F12
	FMOVS 44(R8), F13
	FMOVS 56(R8), F14
	FMOVS 60(R8), F15

	// Radix-4 butterfly 1: x0,x2,x4,x6 -> a0..a3
	FADDS F4, F0, F16
	FADDS F5, F1, F17
	FSUBS F4, F0, F18
	FSUBS F5, F1, F19

	FADDS F6, F2, F20
	FADDS F7, F3, F21
	FSUBS F6, F2, F22
	FSUBS F7, F3, F23

	FADDS F20, F16, F24       // a0
	FADDS F21, F17, F25
	FSUBS F20, F16, F26       // a2
	FSUBS F21, F17, F27

	FMOVS F23, F28            // (-i) * t3
	FNEGS  F22, F29
	FADDS F28, F18, F30       // a1
	FADDS F29, F19, F31

	FNEGS  F23, F6             // i * t3
	FMOVS F22, F7
	FADDS F6, F18, F4         // a3
	FADDS F7, F19, F5

	// Radix-4 butterfly 2: x1,x3,x5,x7 -> a4..a7
	FADDS F12, F8, F16
	FADDS F13, F9, F17
	FSUBS F12, F8, F18
	FSUBS F13, F9, F19

	FADDS F14, F10, F20
	FADDS F15, F11, F21
	FSUBS F14, F10, F22
	FSUBS F15, F11, F23

	FADDS F20, F16, F6        // a4
	FADDS F21, F17, F7
	FSUBS F20, F16, F8        // a6
	FSUBS F21, F17, F9

	FMOVS F23, F10
	FNEGS  F22, F11
	FADDS F10, F18, F12       // a5
	FADDS F11, F19, F13

	FNEGS  F23, F14
	FMOVS F22, F15
	FADDS F14, F18, F16       // a7
	FADDS F15, F19, F17

	// Store a0..a7 into work[0..7]
	FMOVS F24, 0(R8)
	FMOVS F25, 4(R8)
	FMOVS F30, 8(R8)
	FMOVS F31, 12(R8)
	FMOVS F26, 16(R8)
	FMOVS F27, 20(R8)
	FMOVS F4, 24(R8)
	FMOVS F5, 28(R8)
	FMOVS F6, 32(R8)
	FMOVS F7, 36(R8)
	FMOVS F12, 40(R8)
	FMOVS F13, 44(R8)
	FMOVS F8, 48(R8)
	FMOVS F9, 52(R8)
	FMOVS F16, 56(R8)
	FMOVS F17, 60(R8)

	// Stage 2: radix-2 butterflies with twiddles w1,w2,w3
	FMOVS 8(R10), F18
	FMOVS 12(R10), F19
	FMOVS 16(R10), F20
	FMOVS 20(R10), F21
	FMOVS 24(R10), F22
	FMOVS 28(R10), F23

	// a0,a4 -> work[0], work[4]
	FMOVS 0(R8), F0
	FMOVS 4(R8), F1
	FMOVS 32(R8), F2
	FMOVS 36(R8), F3
	FADDS F2, F0, F4
	FADDS F3, F1, F5
	FSUBS F2, F0, F6
	FSUBS F3, F1, F7

	// a1,a5 -> work[1], work[5]
	FMOVS 8(R8), F0
	FMOVS 12(R8), F1
	FMOVS 40(R8), F2
	FMOVS 44(R8), F3
	FMULS F18, F2, F8
	FMULS F19, F3, F9
	FSUBS F9, F8, F8
	FMULS F18, F3, F9
	FMULS F19, F2, F10
	FADDS F10, F9, F9
	FADDS F8, F0, F10
	FADDS F9, F1, F11
	FSUBS F8, F0, F12
	FSUBS F9, F1, F13

	// a2,a6 -> work[2], work[6]
	FMOVS 16(R8), F0
	FMOVS 20(R8), F1
	FMOVS 48(R8), F2
	FMOVS 52(R8), F3
	FMULS F20, F2, F8
	FMULS F21, F3, F9
	FSUBS F9, F8, F8
	FMULS F20, F3, F9
	FMULS F21, F2, F14
	FADDS F14, F9, F9
	FADDS F8, F0, F14
	FADDS F9, F1, F15
	FSUBS F8, F0, F16
	FSUBS F9, F1, F17

	// a3,a7 -> work[3], work[7]
	FMOVS 24(R8), F0
	FMOVS 28(R8), F1
	FMOVS 56(R8), F2
	FMOVS 60(R8), F3
	FMULS F22, F2, F8
	FMULS F23, F3, F9
	FSUBS F9, F8, F8
	FMULS F22, F3, F9
	FMULS F23, F2, F18
	FADDS F18, F9, F9
	FADDS F8, F0, F18
	FADDS F9, F1, F19
	FSUBS F8, F0, F20
	FSUBS F9, F1, F21

	// Store results
	FMOVS F4, 0(R8)
	FMOVS F5, 4(R8)
	FMOVS F10, 8(R8)
	FMOVS F11, 12(R8)
	FMOVS F14, 16(R8)
	FMOVS F15, 20(R8)
	FMOVS F18, 24(R8)
	FMOVS F19, 28(R8)
	FMOVS F6, 32(R8)
	FMOVS F7, 36(R8)
	FMOVS F12, 40(R8)
	FMOVS F13, 44(R8)
	FMOVS F16, 48(R8)
	FMOVS F17, 52(R8)
	FMOVS F20, 56(R8)
	FMOVS F21, 60(R8)

	CMP  R8, R20
	BEQ  neon8r4_return_true

	MOVD $0, R0
neon8r4_copy_loop:
	CMP  $8, R0
	BGE  neon8r4_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon8r4_copy_loop

neon8r4_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon8r4_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// Inverse transform, size 8, mixed radix
TEXT ·inverseNEONSize8Radix4Complex64Asm(SB), NOSPLIT, $0-121
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD bitrev+96(FP), R12
	MOVD src+32(FP), R13

	CMP  $8, R13
	BNE  neon8r4_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $8, R0
	BLT  neon8r4_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $8, R0
	BLT  neon8r4_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $8, R0
	BLT  neon8r4_inv_return_false

	MOVD bitrev+104(FP), R0
	CMP  $8, R0
	BLT  neon8r4_inv_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon8r4_inv_use_dst
	MOVD R11, R8

neon8r4_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon8r4_inv_bitrev_loop:
	CMP  $8, R0
	BGE  neon8r4_inv_stage1

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
	B    neon8r4_inv_bitrev_loop

neon8r4_inv_stage1:
	// Stage 1: same as forward to produce a0..a7 into work
	FMOVS 0(R8), F0
	FMOVS 4(R8), F1
	FMOVS 16(R8), F2
	FMOVS 20(R8), F3
	FMOVS 32(R8), F4
	FMOVS 36(R8), F5
	FMOVS 48(R8), F6
	FMOVS 52(R8), F7

	FMOVS 8(R8), F8
	FMOVS 12(R8), F9
	FMOVS 24(R8), F10
	FMOVS 28(R8), F11
	FMOVS 40(R8), F12
	FMOVS 44(R8), F13
	FMOVS 56(R8), F14
	FMOVS 60(R8), F15

	FADDS F4, F0, F16
	FADDS F5, F1, F17
	FSUBS F4, F0, F18
	FSUBS F5, F1, F19

	FADDS F6, F2, F20
	FADDS F7, F3, F21
	FSUBS F6, F2, F22
	FSUBS F7, F3, F23

	FADDS F20, F16, F24
	FADDS F21, F17, F25
	FSUBS F20, F16, F26
	FSUBS F21, F17, F27

	FMOVS F23, F28
	FNEGS  F22, F29
	FADDS F28, F18, F30
	FADDS F29, F19, F31

	FNEGS  F23, F6
	FMOVS F22, F7
	FADDS F6, F18, F4
	FADDS F7, F19, F5

	FADDS F12, F8, F16
	FADDS F13, F9, F17
	FSUBS F12, F8, F18
	FSUBS F13, F9, F19

	FADDS F14, F10, F20
	FADDS F15, F11, F21
	FSUBS F14, F10, F22
	FSUBS F15, F11, F23

	FADDS F20, F16, F6
	FADDS F21, F17, F7
	FSUBS F20, F16, F8
	FSUBS F21, F17, F9

	FMOVS F23, F10
	FNEGS  F22, F11
	FADDS F10, F18, F12
	FADDS F11, F19, F13

	FNEGS  F23, F14
	FMOVS F22, F15
	FADDS F14, F18, F16
	FADDS F15, F19, F17

	FMOVS F24, 0(R8)
	FMOVS F25, 4(R8)
	FMOVS F30, 8(R8)
	FMOVS F31, 12(R8)
	FMOVS F26, 16(R8)
	FMOVS F27, 20(R8)
	FMOVS F4, 24(R8)
	FMOVS F5, 28(R8)
	FMOVS F6, 32(R8)
	FMOVS F7, 36(R8)
	FMOVS F12, 40(R8)
	FMOVS F13, 44(R8)
	FMOVS F8, 48(R8)
	FMOVS F9, 52(R8)
	FMOVS F16, 56(R8)
	FMOVS F17, 60(R8)

	// Stage 2: radix-2 butterflies with conjugated twiddles
	FMOVS 8(R10), F18
	FMOVS 12(R10), F19
	FNEGS  F19, F19
	FMOVS 16(R10), F20
	FMOVS 20(R10), F21
	FNEGS  F21, F21
	FMOVS 24(R10), F22
	FMOVS 28(R10), F23
	FNEGS  F23, F23

	// a0,a4
	FMOVS 0(R8), F0
	FMOVS 4(R8), F1
	FMOVS 32(R8), F2
	FMOVS 36(R8), F3
	FADDS F2, F0, F4
	FADDS F3, F1, F5
	FSUBS F2, F0, F6
	FSUBS F3, F1, F7

	// a1,a5
	FMOVS 8(R8), F0
	FMOVS 12(R8), F1
	FMOVS 40(R8), F2
	FMOVS 44(R8), F3
	FMULS F18, F2, F8
	FMULS F19, F3, F9
	FSUBS F9, F8, F8
	FMULS F18, F3, F9
	FMULS F19, F2, F10
	FADDS F10, F9, F9
	FADDS F8, F0, F10
	FADDS F9, F1, F11
	FSUBS F8, F0, F12
	FSUBS F9, F1, F13

	// a2,a6
	FMOVS 16(R8), F0
	FMOVS 20(R8), F1
	FMOVS 48(R8), F2
	FMOVS 52(R8), F3
	FMULS F20, F2, F8
	FMULS F21, F3, F9
	FSUBS F9, F8, F8
	FMULS F20, F3, F9
	FMULS F21, F2, F14
	FADDS F14, F9, F9
	FADDS F8, F0, F14
	FADDS F9, F1, F15
	FSUBS F8, F0, F16
	FSUBS F9, F1, F17

	// a3,a7
	FMOVS 24(R8), F0
	FMOVS 28(R8), F1
	FMOVS 56(R8), F2
	FMOVS 60(R8), F3
	FMULS F22, F2, F8
	FMULS F23, F3, F9
	FSUBS F9, F8, F8
	FMULS F22, F3, F9
	FMULS F23, F2, F18
	FADDS F18, F9, F9
	FADDS F8, F0, F18
	FADDS F9, F1, F19
	FSUBS F8, F0, F20
	FSUBS F9, F1, F21

	// Store results
	FMOVS F4, 0(R8)
	FMOVS F5, 4(R8)
	FMOVS F10, 8(R8)
	FMOVS F11, 12(R8)
	FMOVS F14, 16(R8)
	FMOVS F15, 20(R8)
	FMOVS F18, 24(R8)
	FMOVS F19, 28(R8)
	FMOVS F6, 32(R8)
	FMOVS F7, 36(R8)
	FMOVS F12, 40(R8)
	FMOVS F13, 44(R8)
	FMOVS F16, 48(R8)
	FMOVS F17, 52(R8)
	FMOVS F20, 56(R8)
	FMOVS F21, 60(R8)

	CMP  R8, R20
	BEQ  neon8r4_inv_scale

	MOVD $0, R0
neon8r4_inv_copy_loop:
	CMP  $8, R0
	BGE  neon8r4_inv_scale
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon8r4_inv_copy_loop

neon8r4_inv_scale:
	MOVD $·neonInv8(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon8r4_inv_scale_loop:
	CMP  $8, R0
	BGE  neon8r4_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon8r4_inv_scale_loop

neon8r4_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon8r4_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

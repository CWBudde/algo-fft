//go:build arm64 && fft_asm && !purego

// ===========================================================================
// NEON Size-256 Radix-4 FFT Kernels for ARM64
// ===========================================================================
//
// Size 256 = 4^4, radix-4 algorithm uses 4 stages:
//   Stage 1: 64 butterflies, stride=4, no twiddle multiply (W^0 = 1)
//   Stage 2: 16 groups × 4 butterflies, twiddle step=16
//   Stage 3: 4 groups × 16 butterflies, twiddle step=4
//   Stage 4: 1 group × 64 butterflies, twiddle step=1
//
// ===========================================================================

#include "textflag.h"

DATA ·neonInv256Radix4+0(SB)/4, $0x3b800000 // 1/256
GLOBL ·neonInv256Radix4(SB), RODATA, $4

// Forward transform, size 256, complex64, radix-4 variant
TEXT ·forwardNEONSize256Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD bitrev+96(FP), R12
	MOVD src+32(FP), R13

	CMP  $256, R13
	BNE  neon256r4_return_false

	MOVD dst+8(FP), R0
	CMP  $256, R0
	BLT  neon256r4_return_false

	MOVD twiddle+56(FP), R0
	CMP  $256, R0
	BLT  neon256r4_return_false

	MOVD scratch+80(FP), R0
	CMP  $256, R0
	BLT  neon256r4_return_false

	MOVD bitrev+104(FP), R0
	CMP  $256, R0
	BLT  neon256r4_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon256r4_use_dst
	MOVD R11, R8

neon256r4_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon256r4_bitrev_loop:
	CMP  $256, R0
	BGE  neon256r4_stage1

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
	B    neon256r4_bitrev_loop

neon256r4_stage1:
	// Stage 1: 64 radix-4 butterflies
	MOVD $0, R0

neon256r4_stage1_loop:
	CMP  $256, R0
	BGE  neon256r4_stage2

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
	FMOVS F26, 8(R1)
	FMOVS F27, 12(R1)
	FMOVS F18, 16(R1)
	FMOVS F19, 20(R1)
	FMOVS F22, 24(R1)
	FMOVS F23, 28(R1)

	ADD  $4, R0, R0
	B    neon256r4_stage1_loop

neon256r4_stage2:
	// Stage 2: 16 groups × 4 butterflies, twiddle step=16
	MOVD $0, R0

neon256r4_stage2_outer:
	CMP  $256, R0
	BGE  neon256r4_stage3

	MOVD $0, R1

neon256r4_stage2_inner:
	CMP  $4, R1
	BGE  neon256r4_stage2_next

	ADD  R0, R1, R2
	ADD  $4, R2, R3
	ADD  $8, R2, R4
	ADD  $12, R2, R5

	// twiddle indices: j*16, j*32, j*48
	LSL  $4, R1, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F0
	FMOVS 4(R6), F1

	LSL  $5, R1, R7
	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F2
	FMOVS 4(R7), F3

	LSL  $4, R1, R6
	LSL  $5, R1, R7
	ADD  R6, R7, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F4
	FMOVS 4(R6), F5

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F6
	FMOVS 4(R6), F7

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F8
	FMOVS 4(R6), F9

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F10
	FMOVS 4(R6), F11

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F12
	FMOVS 4(R6), F13

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

	// Radix-4 butterfly
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

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS F22, 0(R6)
	FMOVS F23, 4(R6)

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS F20, 0(R6)
	FMOVS F21, 4(R6)

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS F24, 0(R6)
	FMOVS F25, 4(R6)

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS F28, 0(R6)
	FMOVS F29, 4(R6)

	ADD  $1, R1, R1
	B    neon256r4_stage2_inner

neon256r4_stage2_next:
	ADD  $16, R0, R0
	B    neon256r4_stage2_outer

neon256r4_stage3:
	// Stage 3: 4 groups × 16 butterflies, twiddle step=4
	MOVD $0, R0

neon256r4_stage3_outer:
	CMP  $256, R0
	BGE  neon256r4_stage4

	MOVD $0, R1

neon256r4_stage3_inner:
	CMP  $16, R1
	BGE  neon256r4_stage3_next

	ADD  R0, R1, R2
	ADD  $16, R2, R3
	ADD  $32, R2, R4
	ADD  $48, R2, R5

	// twiddle indices: j*4, j*8, j*12
	LSL  $2, R1, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F0
	FMOVS 4(R6), F1

	LSL  $3, R1, R7
	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F2
	FMOVS 4(R7), F3

	LSL  $2, R1, R6
	LSL  $3, R1, R7
	ADD  R6, R7, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F4
	FMOVS 4(R6), F5

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F6
	FMOVS 4(R6), F7

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F8
	FMOVS 4(R6), F9

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F10
	FMOVS 4(R6), F11

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F12
	FMOVS 4(R6), F13

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

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS F22, 0(R6)
	FMOVS F23, 4(R6)

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS F20, 0(R6)
	FMOVS F21, 4(R6)

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS F24, 0(R6)
	FMOVS F25, 4(R6)

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS F28, 0(R6)
	FMOVS F29, 4(R6)

	ADD  $1, R1, R1
	B    neon256r4_stage3_inner

neon256r4_stage3_next:
	ADD  $64, R0, R0
	B    neon256r4_stage3_outer

neon256r4_stage4:
	// Stage 4: 1 group × 64 butterflies, twiddle step=1
	MOVD $0, R0

neon256r4_stage4_loop:
	CMP  $64, R0
	BGE  neon256r4_done

	MOVD R0, R1
	ADD  $64, R1, R2
	ADD  $128, R1, R3
	ADD  $192, R1, R4

	// twiddle indices: j, 2j, 3j
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
	ADD  R10, R7, R7
	FMOVS 0(R7), F4
	FMOVS 4(R7), F5

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
	FMOVS F20, 0(R5)
	FMOVS F21, 4(R5)

	LSL  $3, R3, R5
	ADD  R8, R5, R5
	FMOVS F24, 0(R5)
	FMOVS F25, 4(R5)

	LSL  $3, R4, R5
	ADD  R8, R5, R5
	FMOVS F28, 0(R5)
	FMOVS F29, 4(R5)

	ADD  $1, R0, R0
	B    neon256r4_stage4_loop

neon256r4_done:
	CMP  R8, R20
	BEQ  neon256r4_return_true

	MOVD $0, R0
neon256r4_copy_loop:
	CMP  $256, R0
	BGE  neon256r4_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon256r4_copy_loop

neon256r4_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon256r4_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// Inverse transform, size 256, complex64, radix-4 variant
TEXT ·inverseNEONSize256Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD bitrev+96(FP), R12
	MOVD src+32(FP), R13

	CMP  $256, R13
	BNE  neon256r4_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $256, R0
	BLT  neon256r4_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $256, R0
	BLT  neon256r4_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $256, R0
	BLT  neon256r4_inv_return_false

	MOVD bitrev+104(FP), R0
	CMP  $256, R0
	BLT  neon256r4_inv_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon256r4_inv_use_dst
	MOVD R11, R8

neon256r4_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon256r4_inv_bitrev_loop:
	CMP  $256, R0
	BGE  neon256r4_inv_stage1

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
	B    neon256r4_inv_bitrev_loop

neon256r4_inv_stage1:
	// Stage 1 (inverse variant)
	MOVD $0, R0

neon256r4_inv_stage1_loop:
	CMP  $256, R0
	BGE  neon256r4_inv_stage2

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

	FMOVS F15, F20
	FNEGS  F14, F21
	FADDS F20, F10, F22
	FADDS F21, F11, F23

	FNEGS  F15, F24
	FMOVS F14, F25
	FADDS F24, F10, F26
	FADDS F25, F11, F27

	FMOVS F16, 0(R1)
	FMOVS F17, 4(R1)
	FMOVS F26, 8(R1)
	FMOVS F27, 12(R1)
	FMOVS F18, 16(R1)
	FMOVS F19, 20(R1)
	FMOVS F22, 24(R1)
	FMOVS F23, 28(R1)

	ADD  $4, R0, R0
	B    neon256r4_inv_stage1_loop

neon256r4_inv_stage2:
	// Stage 2 with conjugated twiddles
	MOVD $0, R0

neon256r4_inv_stage2_outer:
	CMP  $256, R0
	BGE  neon256r4_inv_stage3

	MOVD $0, R1

neon256r4_inv_stage2_inner:
	CMP  $4, R1
	BGE  neon256r4_inv_stage2_next

	ADD  R0, R1, R2
	ADD  $4, R2, R3
	ADD  $8, R2, R4
	ADD  $12, R2, R5

	// twiddle indices: j*16, j*32, j*48 (conjugated)
	LSL  $4, R1, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F0
	FMOVS 4(R6), F1
	FNEGS  F1, F1

	LSL  $5, R1, R7
	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F2
	FMOVS 4(R7), F3
	FNEGS  F3, F3

	LSL  $4, R1, R6
	LSL  $5, R1, R7
	ADD  R6, R7, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F4
	FMOVS 4(R6), F5
	FNEGS  F5, F5

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F6
	FMOVS 4(R6), F7

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F8
	FMOVS 4(R6), F9

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F10
	FMOVS 4(R6), F11

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F12
	FMOVS 4(R6), F13

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

	FMOVS F21, F26
	FNEGS  F20, F27
	FADDS F26, F16, F28
	FADDS F27, F17, F29

	FNEGS  F21, F30
	FMOVS F20, F31
	FADDS F30, F16, F20
	FADDS F31, F17, F21

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS F22, 0(R6)
	FMOVS F23, 4(R6)

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS F20, 0(R6)
	FMOVS F21, 4(R6)

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS F24, 0(R6)
	FMOVS F25, 4(R6)

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS F28, 0(R6)
	FMOVS F29, 4(R6)

	ADD  $1, R1, R1
	B    neon256r4_inv_stage2_inner

neon256r4_inv_stage2_next:
	ADD  $16, R0, R0
	B    neon256r4_inv_stage2_outer

neon256r4_inv_stage3:
	// Stage 3 with conjugated twiddles
	MOVD $0, R0

neon256r4_inv_stage3_outer:
	CMP  $256, R0
	BGE  neon256r4_inv_stage4

	MOVD $0, R1

neon256r4_inv_stage3_inner:
	CMP  $16, R1
	BGE  neon256r4_inv_stage3_next

	ADD  R0, R1, R2
	ADD  $16, R2, R3
	ADD  $32, R2, R4
	ADD  $48, R2, R5

	// twiddle indices: j*4, j*8, j*12 (conjugated)
	LSL  $2, R1, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F0
	FMOVS 4(R6), F1
	FNEGS  F1, F1

	LSL  $3, R1, R7
	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F2
	FMOVS 4(R7), F3
	FNEGS  F3, F3

	LSL  $2, R1, R6
	LSL  $3, R1, R7
	ADD  R6, R7, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F4
	FMOVS 4(R6), F5
	FNEGS  F5, F5

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F6
	FMOVS 4(R6), F7

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F8
	FMOVS 4(R6), F9

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F10
	FMOVS 4(R6), F11

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F12
	FMOVS 4(R6), F13

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

	FMOVS F21, F26
	FNEGS  F20, F27
	FADDS F26, F16, F28
	FADDS F27, F17, F29

	FNEGS  F21, F30
	FMOVS F20, F31
	FADDS F30, F16, F20
	FADDS F31, F17, F21

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS F22, 0(R6)
	FMOVS F23, 4(R6)

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS F20, 0(R6)
	FMOVS F21, 4(R6)

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS F24, 0(R6)
	FMOVS F25, 4(R6)

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS F28, 0(R6)
	FMOVS F29, 4(R6)

	ADD  $1, R1, R1
	B    neon256r4_inv_stage3_inner

neon256r4_inv_stage3_next:
	ADD  $64, R0, R0
	B    neon256r4_inv_stage3_outer

neon256r4_inv_stage4:
	// Stage 4 with conjugated twiddles
	MOVD $0, R0

neon256r4_inv_stage4_loop:
	CMP  $64, R0
	BGE  neon256r4_inv_done

	MOVD R0, R1
	ADD  $64, R1, R2
	ADD  $128, R1, R3
	ADD  $192, R1, R4

	// twiddle indices: j, 2j, 3j (conjugated)
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

	FMOVS F21, F26
	FNEGS  F20, F27
	FADDS F26, F16, F28
	FADDS F27, F17, F29

	FNEGS  F21, F30
	FMOVS F20, F31
	FADDS F30, F16, F20
	FADDS F31, F17, F21

	LSL  $3, R1, R5
	ADD  R8, R5, R5
	FMOVS F22, 0(R5)
	FMOVS F23, 4(R5)

	LSL  $3, R2, R5
	ADD  R8, R5, R5
	FMOVS F20, 0(R5)
	FMOVS F21, 4(R5)

	LSL  $3, R3, R5
	ADD  R8, R5, R5
	FMOVS F24, 0(R5)
	FMOVS F25, 4(R5)

	LSL  $3, R4, R5
	ADD  R8, R5, R5
	FMOVS F28, 0(R5)
	FMOVS F29, 4(R5)

	ADD  $1, R0, R0
	B    neon256r4_inv_stage4_loop

neon256r4_inv_done:
	CMP  R8, R20
	BEQ  neon256r4_inv_scale

	MOVD $0, R0
neon256r4_inv_copy_loop:
	CMP  $256, R0
	BGE  neon256r4_inv_scale
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon256r4_inv_copy_loop

neon256r4_inv_scale:
	MOVD $·neonInv256Radix4(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon256r4_inv_scale_loop:
	CMP  $256, R0
	BGE  neon256r4_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon256r4_inv_scale_loop

neon256r4_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon256r4_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

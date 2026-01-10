//go:build arm64 && asm && !purego

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
TEXT ·ForwardNEONSize256Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
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

	// Load static bit-reversal table
	MOVD $bitrev_size256_radix4<>(SB), R12

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
	MOVB R0, ret+96(FP)
	RET

neon256r4_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Inverse transform, size 256, complex64, radix-4 variant
TEXT ·InverseNEONSize256Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
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

	// Load static bit-reversal table
	MOVD $bitrev_size256_radix4<>(SB), R12

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
	MOVB R0, ret+96(FP)
	RET

neon256r4_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET
// Size-256 Radix-4
// Pattern: 0, 64, 128, 192, 16, 80, 144, 208, 32, 96, 160, 224, 48, 112, 176, 240, 4, 68, 132, 196, 20, 84, 148, 212, 36, 100, 164, 228, 52, 116, 180, 244, 8, 72, 136, 200, 24, 88, 152, 216, 40, 104, 168, 232, 56, 120, 184, 248, 12, 76, 140, 204, 28, 92, 156, 220, 44, 108, 172, 236, 60, 124, 188, 252, 1, 65, 129, 193, 17, 81, 145, 209, 33, 97, 161, 225, 49, 113, 177, 241, 5, 69, 133, 197, 21, 85, 149, 213, 37, 101, 165, 229, 53, 117, 181, 245, 9, 73, 137, 201, 25, 89, 153, 217, 41, 105, 169, 233, 57, 121, 185, 249, 13, 77, 141, 205, 29, 93, 157, 221, 45, 109, 173, 237, 61, 125, 189, 253, 2, 66, 130, 194, 18, 82, 146, 210, 34, 98, 162, 226, 50, 114, 178, 242, 6, 70, 134, 198, 22, 86, 150, 214, 38, 102, 166, 230, 54, 118, 182, 246, 10, 74, 138, 202, 26, 90, 154, 218, 42, 106, 170, 234, 58, 122, 186, 250, 14, 78, 142, 206, 30, 94, 158, 222, 46, 110, 174, 238, 62, 126, 190, 254, 3, 67, 131, 195, 19, 83, 147, 211, 35, 99, 163, 227, 51, 115, 179, 243, 7, 71, 135, 199, 23, 87, 151, 215, 39, 103, 167, 231, 55, 119, 183, 247, 11, 75, 139, 203, 27, 91, 155, 219, 43, 107, 171, 235, 59, 123, 187, 251, 15, 79, 143, 207, 31, 95, 159, 223, 47, 111, 175, 239, 63, 127, 191, 255
DATA bitrev_size256_radix4<>+0x000(SB)/8, $0   // bitrev[0] = 0
DATA bitrev_size256_radix4<>+0x008(SB)/8, $64  // bitrev[1] = 64
DATA bitrev_size256_radix4<>+0x010(SB)/8, $128 // bitrev[2] = 128
DATA bitrev_size256_radix4<>+0x018(SB)/8, $192 // bitrev[3] = 192
DATA bitrev_size256_radix4<>+0x020(SB)/8, $16  // bitrev[4] = 16
DATA bitrev_size256_radix4<>+0x028(SB)/8, $80  // bitrev[5] = 80
DATA bitrev_size256_radix4<>+0x030(SB)/8, $144 // bitrev[6] = 144
DATA bitrev_size256_radix4<>+0x038(SB)/8, $208 // bitrev[7] = 208
DATA bitrev_size256_radix4<>+0x040(SB)/8, $32  // bitrev[8] = 32
DATA bitrev_size256_radix4<>+0x048(SB)/8, $96  // bitrev[9] = 96
DATA bitrev_size256_radix4<>+0x050(SB)/8, $160 // bitrev[10] = 160
DATA bitrev_size256_radix4<>+0x058(SB)/8, $224 // bitrev[11] = 224
DATA bitrev_size256_radix4<>+0x060(SB)/8, $48  // bitrev[12] = 48
DATA bitrev_size256_radix4<>+0x068(SB)/8, $112 // bitrev[13] = 112
DATA bitrev_size256_radix4<>+0x070(SB)/8, $176 // bitrev[14] = 176
DATA bitrev_size256_radix4<>+0x078(SB)/8, $240 // bitrev[15] = 240
DATA bitrev_size256_radix4<>+0x080(SB)/8, $4   // bitrev[16] = 4
DATA bitrev_size256_radix4<>+0x088(SB)/8, $68  // bitrev[17] = 68
DATA bitrev_size256_radix4<>+0x090(SB)/8, $132 // bitrev[18] = 132
DATA bitrev_size256_radix4<>+0x098(SB)/8, $196 // bitrev[19] = 196
DATA bitrev_size256_radix4<>+0x0a0(SB)/8, $20  // bitrev[20] = 20
DATA bitrev_size256_radix4<>+0x0a8(SB)/8, $84  // bitrev[21] = 84
DATA bitrev_size256_radix4<>+0x0b0(SB)/8, $148 // bitrev[22] = 148
DATA bitrev_size256_radix4<>+0x0b8(SB)/8, $212 // bitrev[23] = 212
DATA bitrev_size256_radix4<>+0x0c0(SB)/8, $36  // bitrev[24] = 36
DATA bitrev_size256_radix4<>+0x0c8(SB)/8, $100 // bitrev[25] = 100
DATA bitrev_size256_radix4<>+0x0d0(SB)/8, $164 // bitrev[26] = 164
DATA bitrev_size256_radix4<>+0x0d8(SB)/8, $228 // bitrev[27] = 228
DATA bitrev_size256_radix4<>+0x0e0(SB)/8, $52  // bitrev[28] = 52
DATA bitrev_size256_radix4<>+0x0e8(SB)/8, $116 // bitrev[29] = 116
DATA bitrev_size256_radix4<>+0x0f0(SB)/8, $180 // bitrev[30] = 180
DATA bitrev_size256_radix4<>+0x0f8(SB)/8, $244 // bitrev[31] = 244
DATA bitrev_size256_radix4<>+0x100(SB)/8, $8   // bitrev[32] = 8
DATA bitrev_size256_radix4<>+0x108(SB)/8, $72  // bitrev[33] = 72
DATA bitrev_size256_radix4<>+0x110(SB)/8, $136 // bitrev[34] = 136
DATA bitrev_size256_radix4<>+0x118(SB)/8, $200 // bitrev[35] = 200
DATA bitrev_size256_radix4<>+0x120(SB)/8, $24  // bitrev[36] = 24
DATA bitrev_size256_radix4<>+0x128(SB)/8, $88  // bitrev[37] = 88
DATA bitrev_size256_radix4<>+0x130(SB)/8, $152 // bitrev[38] = 152
DATA bitrev_size256_radix4<>+0x138(SB)/8, $216 // bitrev[39] = 216
DATA bitrev_size256_radix4<>+0x140(SB)/8, $40  // bitrev[40] = 40
DATA bitrev_size256_radix4<>+0x148(SB)/8, $104 // bitrev[41] = 104
DATA bitrev_size256_radix4<>+0x150(SB)/8, $168 // bitrev[42] = 168
DATA bitrev_size256_radix4<>+0x158(SB)/8, $232 // bitrev[43] = 232
DATA bitrev_size256_radix4<>+0x160(SB)/8, $56  // bitrev[44] = 56
DATA bitrev_size256_radix4<>+0x168(SB)/8, $120 // bitrev[45] = 120
DATA bitrev_size256_radix4<>+0x170(SB)/8, $184 // bitrev[46] = 184
DATA bitrev_size256_radix4<>+0x178(SB)/8, $248 // bitrev[47] = 248
DATA bitrev_size256_radix4<>+0x180(SB)/8, $12  // bitrev[48] = 12
DATA bitrev_size256_radix4<>+0x188(SB)/8, $76  // bitrev[49] = 76
DATA bitrev_size256_radix4<>+0x190(SB)/8, $140 // bitrev[50] = 140
DATA bitrev_size256_radix4<>+0x198(SB)/8, $204 // bitrev[51] = 204
DATA bitrev_size256_radix4<>+0x1a0(SB)/8, $28  // bitrev[52] = 28
DATA bitrev_size256_radix4<>+0x1a8(SB)/8, $92  // bitrev[53] = 92
DATA bitrev_size256_radix4<>+0x1b0(SB)/8, $156 // bitrev[54] = 156
DATA bitrev_size256_radix4<>+0x1b8(SB)/8, $220 // bitrev[55] = 220
DATA bitrev_size256_radix4<>+0x1c0(SB)/8, $44  // bitrev[56] = 44
DATA bitrev_size256_radix4<>+0x1c8(SB)/8, $108 // bitrev[57] = 108
DATA bitrev_size256_radix4<>+0x1d0(SB)/8, $172 // bitrev[58] = 172
DATA bitrev_size256_radix4<>+0x1d8(SB)/8, $236 // bitrev[59] = 236
DATA bitrev_size256_radix4<>+0x1e0(SB)/8, $60  // bitrev[60] = 60
DATA bitrev_size256_radix4<>+0x1e8(SB)/8, $124 // bitrev[61] = 124
DATA bitrev_size256_radix4<>+0x1f0(SB)/8, $188 // bitrev[62] = 188
DATA bitrev_size256_radix4<>+0x1f8(SB)/8, $252 // bitrev[63] = 252
DATA bitrev_size256_radix4<>+0x200(SB)/8, $1   // bitrev[64] = 1
DATA bitrev_size256_radix4<>+0x208(SB)/8, $65  // bitrev[65] = 65
DATA bitrev_size256_radix4<>+0x210(SB)/8, $129 // bitrev[66] = 129
DATA bitrev_size256_radix4<>+0x218(SB)/8, $193 // bitrev[67] = 193
DATA bitrev_size256_radix4<>+0x220(SB)/8, $17  // bitrev[68] = 17
DATA bitrev_size256_radix4<>+0x228(SB)/8, $81  // bitrev[69] = 81
DATA bitrev_size256_radix4<>+0x230(SB)/8, $145 // bitrev[70] = 145
DATA bitrev_size256_radix4<>+0x238(SB)/8, $209 // bitrev[71] = 209
DATA bitrev_size256_radix4<>+0x240(SB)/8, $33  // bitrev[72] = 33
DATA bitrev_size256_radix4<>+0x248(SB)/8, $97  // bitrev[73] = 97
DATA bitrev_size256_radix4<>+0x250(SB)/8, $161 // bitrev[74] = 161
DATA bitrev_size256_radix4<>+0x258(SB)/8, $225 // bitrev[75] = 225
DATA bitrev_size256_radix4<>+0x260(SB)/8, $49  // bitrev[76] = 49
DATA bitrev_size256_radix4<>+0x268(SB)/8, $113 // bitrev[77] = 113
DATA bitrev_size256_radix4<>+0x270(SB)/8, $177 // bitrev[78] = 177
DATA bitrev_size256_radix4<>+0x278(SB)/8, $241 // bitrev[79] = 241
DATA bitrev_size256_radix4<>+0x280(SB)/8, $5   // bitrev[80] = 5
DATA bitrev_size256_radix4<>+0x288(SB)/8, $69  // bitrev[81] = 69
DATA bitrev_size256_radix4<>+0x290(SB)/8, $133 // bitrev[82] = 133
DATA bitrev_size256_radix4<>+0x298(SB)/8, $197 // bitrev[83] = 197
DATA bitrev_size256_radix4<>+0x2a0(SB)/8, $21  // bitrev[84] = 21
DATA bitrev_size256_radix4<>+0x2a8(SB)/8, $85  // bitrev[85] = 85
DATA bitrev_size256_radix4<>+0x2b0(SB)/8, $149 // bitrev[86] = 149
DATA bitrev_size256_radix4<>+0x2b8(SB)/8, $213 // bitrev[87] = 213
DATA bitrev_size256_radix4<>+0x2c0(SB)/8, $37  // bitrev[88] = 37
DATA bitrev_size256_radix4<>+0x2c8(SB)/8, $101 // bitrev[89] = 101
DATA bitrev_size256_radix4<>+0x2d0(SB)/8, $165 // bitrev[90] = 165
DATA bitrev_size256_radix4<>+0x2d8(SB)/8, $229 // bitrev[91] = 229
DATA bitrev_size256_radix4<>+0x2e0(SB)/8, $53  // bitrev[92] = 53
DATA bitrev_size256_radix4<>+0x2e8(SB)/8, $117 // bitrev[93] = 117
DATA bitrev_size256_radix4<>+0x2f0(SB)/8, $181 // bitrev[94] = 181
DATA bitrev_size256_radix4<>+0x2f8(SB)/8, $245 // bitrev[95] = 245
DATA bitrev_size256_radix4<>+0x300(SB)/8, $9   // bitrev[96] = 9
DATA bitrev_size256_radix4<>+0x308(SB)/8, $73  // bitrev[97] = 73
DATA bitrev_size256_radix4<>+0x310(SB)/8, $137 // bitrev[98] = 137
DATA bitrev_size256_radix4<>+0x318(SB)/8, $201 // bitrev[99] = 201
DATA bitrev_size256_radix4<>+0x320(SB)/8, $25  // bitrev[100] = 25
DATA bitrev_size256_radix4<>+0x328(SB)/8, $89  // bitrev[101] = 89
DATA bitrev_size256_radix4<>+0x330(SB)/8, $153 // bitrev[102] = 153
DATA bitrev_size256_radix4<>+0x338(SB)/8, $217 // bitrev[103] = 217
DATA bitrev_size256_radix4<>+0x340(SB)/8, $41  // bitrev[104] = 41
DATA bitrev_size256_radix4<>+0x348(SB)/8, $105 // bitrev[105] = 105
DATA bitrev_size256_radix4<>+0x350(SB)/8, $169 // bitrev[106] = 169
DATA bitrev_size256_radix4<>+0x358(SB)/8, $233 // bitrev[107] = 233
DATA bitrev_size256_radix4<>+0x360(SB)/8, $57  // bitrev[108] = 57
DATA bitrev_size256_radix4<>+0x368(SB)/8, $121 // bitrev[109] = 121
DATA bitrev_size256_radix4<>+0x370(SB)/8, $185 // bitrev[110] = 185
DATA bitrev_size256_radix4<>+0x378(SB)/8, $249 // bitrev[111] = 249
DATA bitrev_size256_radix4<>+0x380(SB)/8, $13  // bitrev[112] = 13
DATA bitrev_size256_radix4<>+0x388(SB)/8, $77  // bitrev[113] = 77
DATA bitrev_size256_radix4<>+0x390(SB)/8, $141 // bitrev[114] = 141
DATA bitrev_size256_radix4<>+0x398(SB)/8, $205 // bitrev[115] = 205
DATA bitrev_size256_radix4<>+0x3a0(SB)/8, $29  // bitrev[116] = 29
DATA bitrev_size256_radix4<>+0x3a8(SB)/8, $93  // bitrev[117] = 93
DATA bitrev_size256_radix4<>+0x3b0(SB)/8, $157 // bitrev[118] = 157
DATA bitrev_size256_radix4<>+0x3b8(SB)/8, $221 // bitrev[119] = 221
DATA bitrev_size256_radix4<>+0x3c0(SB)/8, $45  // bitrev[120] = 45
DATA bitrev_size256_radix4<>+0x3c8(SB)/8, $109 // bitrev[121] = 109
DATA bitrev_size256_radix4<>+0x3d0(SB)/8, $173 // bitrev[122] = 173
DATA bitrev_size256_radix4<>+0x3d8(SB)/8, $237 // bitrev[123] = 237
DATA bitrev_size256_radix4<>+0x3e0(SB)/8, $61  // bitrev[124] = 61
DATA bitrev_size256_radix4<>+0x3e8(SB)/8, $125 // bitrev[125] = 125
DATA bitrev_size256_radix4<>+0x3f0(SB)/8, $189 // bitrev[126] = 189
DATA bitrev_size256_radix4<>+0x3f8(SB)/8, $253 // bitrev[127] = 253
DATA bitrev_size256_radix4<>+0x400(SB)/8, $2   // bitrev[128] = 2
DATA bitrev_size256_radix4<>+0x408(SB)/8, $66  // bitrev[129] = 66
DATA bitrev_size256_radix4<>+0x410(SB)/8, $130 // bitrev[130] = 130
DATA bitrev_size256_radix4<>+0x418(SB)/8, $194 // bitrev[131] = 194
DATA bitrev_size256_radix4<>+0x420(SB)/8, $18  // bitrev[132] = 18
DATA bitrev_size256_radix4<>+0x428(SB)/8, $82  // bitrev[133] = 82
DATA bitrev_size256_radix4<>+0x430(SB)/8, $146 // bitrev[134] = 146
DATA bitrev_size256_radix4<>+0x438(SB)/8, $210 // bitrev[135] = 210
DATA bitrev_size256_radix4<>+0x440(SB)/8, $34  // bitrev[136] = 34
DATA bitrev_size256_radix4<>+0x448(SB)/8, $98  // bitrev[137] = 98
DATA bitrev_size256_radix4<>+0x450(SB)/8, $162 // bitrev[138] = 162
DATA bitrev_size256_radix4<>+0x458(SB)/8, $226 // bitrev[139] = 226
DATA bitrev_size256_radix4<>+0x460(SB)/8, $50  // bitrev[140] = 50
DATA bitrev_size256_radix4<>+0x468(SB)/8, $114 // bitrev[141] = 114
DATA bitrev_size256_radix4<>+0x470(SB)/8, $178 // bitrev[142] = 178
DATA bitrev_size256_radix4<>+0x478(SB)/8, $242 // bitrev[143] = 242
DATA bitrev_size256_radix4<>+0x480(SB)/8, $6   // bitrev[144] = 6
DATA bitrev_size256_radix4<>+0x488(SB)/8, $70  // bitrev[145] = 70
DATA bitrev_size256_radix4<>+0x490(SB)/8, $134 // bitrev[146] = 134
DATA bitrev_size256_radix4<>+0x498(SB)/8, $198 // bitrev[147] = 198
DATA bitrev_size256_radix4<>+0x4a0(SB)/8, $22  // bitrev[148] = 22
DATA bitrev_size256_radix4<>+0x4a8(SB)/8, $86  // bitrev[149] = 86
DATA bitrev_size256_radix4<>+0x4b0(SB)/8, $150 // bitrev[150] = 150
DATA bitrev_size256_radix4<>+0x4b8(SB)/8, $214 // bitrev[151] = 214
DATA bitrev_size256_radix4<>+0x4c0(SB)/8, $38  // bitrev[152] = 38
DATA bitrev_size256_radix4<>+0x4c8(SB)/8, $102 // bitrev[153] = 102
DATA bitrev_size256_radix4<>+0x4d0(SB)/8, $166 // bitrev[154] = 166
DATA bitrev_size256_radix4<>+0x4d8(SB)/8, $230 // bitrev[155] = 230
DATA bitrev_size256_radix4<>+0x4e0(SB)/8, $54  // bitrev[156] = 54
DATA bitrev_size256_radix4<>+0x4e8(SB)/8, $118 // bitrev[157] = 118
DATA bitrev_size256_radix4<>+0x4f0(SB)/8, $182 // bitrev[158] = 182
DATA bitrev_size256_radix4<>+0x4f8(SB)/8, $246 // bitrev[159] = 246
DATA bitrev_size256_radix4<>+0x500(SB)/8, $10  // bitrev[160] = 10
DATA bitrev_size256_radix4<>+0x508(SB)/8, $74  // bitrev[161] = 74
DATA bitrev_size256_radix4<>+0x510(SB)/8, $138 // bitrev[162] = 138
DATA bitrev_size256_radix4<>+0x518(SB)/8, $202 // bitrev[163] = 202
DATA bitrev_size256_radix4<>+0x520(SB)/8, $26  // bitrev[164] = 26
DATA bitrev_size256_radix4<>+0x528(SB)/8, $90  // bitrev[165] = 90
DATA bitrev_size256_radix4<>+0x530(SB)/8, $154 // bitrev[166] = 154
DATA bitrev_size256_radix4<>+0x538(SB)/8, $218 // bitrev[167] = 218
DATA bitrev_size256_radix4<>+0x540(SB)/8, $42  // bitrev[168] = 42
DATA bitrev_size256_radix4<>+0x548(SB)/8, $106 // bitrev[169] = 106
DATA bitrev_size256_radix4<>+0x550(SB)/8, $170 // bitrev[170] = 170
DATA bitrev_size256_radix4<>+0x558(SB)/8, $234 // bitrev[171] = 234
DATA bitrev_size256_radix4<>+0x560(SB)/8, $58  // bitrev[172] = 58
DATA bitrev_size256_radix4<>+0x568(SB)/8, $122 // bitrev[173] = 122
DATA bitrev_size256_radix4<>+0x570(SB)/8, $186 // bitrev[174] = 186
DATA bitrev_size256_radix4<>+0x578(SB)/8, $250 // bitrev[175] = 250
DATA bitrev_size256_radix4<>+0x580(SB)/8, $14  // bitrev[176] = 14
DATA bitrev_size256_radix4<>+0x588(SB)/8, $78  // bitrev[177] = 78
DATA bitrev_size256_radix4<>+0x590(SB)/8, $142 // bitrev[178] = 142
DATA bitrev_size256_radix4<>+0x598(SB)/8, $206 // bitrev[179] = 206
DATA bitrev_size256_radix4<>+0x5a0(SB)/8, $30  // bitrev[180] = 30
DATA bitrev_size256_radix4<>+0x5a8(SB)/8, $94  // bitrev[181] = 94
DATA bitrev_size256_radix4<>+0x5b0(SB)/8, $158 // bitrev[182] = 158
DATA bitrev_size256_radix4<>+0x5b8(SB)/8, $222 // bitrev[183] = 222
DATA bitrev_size256_radix4<>+0x5c0(SB)/8, $46  // bitrev[184] = 46
DATA bitrev_size256_radix4<>+0x5c8(SB)/8, $110 // bitrev[185] = 110
DATA bitrev_size256_radix4<>+0x5d0(SB)/8, $174 // bitrev[186] = 174
DATA bitrev_size256_radix4<>+0x5d8(SB)/8, $238 // bitrev[187] = 238
DATA bitrev_size256_radix4<>+0x5e0(SB)/8, $62  // bitrev[188] = 62
DATA bitrev_size256_radix4<>+0x5e8(SB)/8, $126 // bitrev[189] = 126
DATA bitrev_size256_radix4<>+0x5f0(SB)/8, $190 // bitrev[190] = 190
DATA bitrev_size256_radix4<>+0x5f8(SB)/8, $254 // bitrev[191] = 254
DATA bitrev_size256_radix4<>+0x600(SB)/8, $3   // bitrev[192] = 3
DATA bitrev_size256_radix4<>+0x608(SB)/8, $67  // bitrev[193] = 67
DATA bitrev_size256_radix4<>+0x610(SB)/8, $131 // bitrev[194] = 131
DATA bitrev_size256_radix4<>+0x618(SB)/8, $195 // bitrev[195] = 195
DATA bitrev_size256_radix4<>+0x620(SB)/8, $19  // bitrev[196] = 19
DATA bitrev_size256_radix4<>+0x628(SB)/8, $83  // bitrev[197] = 83
DATA bitrev_size256_radix4<>+0x630(SB)/8, $147 // bitrev[198] = 147
DATA bitrev_size256_radix4<>+0x638(SB)/8, $211 // bitrev[199] = 211
DATA bitrev_size256_radix4<>+0x640(SB)/8, $35  // bitrev[200] = 35
DATA bitrev_size256_radix4<>+0x648(SB)/8, $99  // bitrev[201] = 99
DATA bitrev_size256_radix4<>+0x650(SB)/8, $163 // bitrev[202] = 163
DATA bitrev_size256_radix4<>+0x658(SB)/8, $227 // bitrev[203] = 227
DATA bitrev_size256_radix4<>+0x660(SB)/8, $51  // bitrev[204] = 51
DATA bitrev_size256_radix4<>+0x668(SB)/8, $115 // bitrev[205] = 115
DATA bitrev_size256_radix4<>+0x670(SB)/8, $179 // bitrev[206] = 179
DATA bitrev_size256_radix4<>+0x678(SB)/8, $243 // bitrev[207] = 243
DATA bitrev_size256_radix4<>+0x680(SB)/8, $7   // bitrev[208] = 7
DATA bitrev_size256_radix4<>+0x688(SB)/8, $71  // bitrev[209] = 71
DATA bitrev_size256_radix4<>+0x690(SB)/8, $135 // bitrev[210] = 135
DATA bitrev_size256_radix4<>+0x698(SB)/8, $199 // bitrev[211] = 199
DATA bitrev_size256_radix4<>+0x6a0(SB)/8, $23  // bitrev[212] = 23
DATA bitrev_size256_radix4<>+0x6a8(SB)/8, $87  // bitrev[213] = 87
DATA bitrev_size256_radix4<>+0x6b0(SB)/8, $151 // bitrev[214] = 151
DATA bitrev_size256_radix4<>+0x6b8(SB)/8, $215 // bitrev[215] = 215
DATA bitrev_size256_radix4<>+0x6c0(SB)/8, $39  // bitrev[216] = 39
DATA bitrev_size256_radix4<>+0x6c8(SB)/8, $103 // bitrev[217] = 103
DATA bitrev_size256_radix4<>+0x6d0(SB)/8, $167 // bitrev[218] = 167
DATA bitrev_size256_radix4<>+0x6d8(SB)/8, $231 // bitrev[219] = 231
DATA bitrev_size256_radix4<>+0x6e0(SB)/8, $55  // bitrev[220] = 55
DATA bitrev_size256_radix4<>+0x6e8(SB)/8, $119 // bitrev[221] = 119
DATA bitrev_size256_radix4<>+0x6f0(SB)/8, $183 // bitrev[222] = 183
DATA bitrev_size256_radix4<>+0x6f8(SB)/8, $247 // bitrev[223] = 247
DATA bitrev_size256_radix4<>+0x700(SB)/8, $11  // bitrev[224] = 11
DATA bitrev_size256_radix4<>+0x708(SB)/8, $75  // bitrev[225] = 75
DATA bitrev_size256_radix4<>+0x710(SB)/8, $139 // bitrev[226] = 139
DATA bitrev_size256_radix4<>+0x718(SB)/8, $203 // bitrev[227] = 203
DATA bitrev_size256_radix4<>+0x720(SB)/8, $27  // bitrev[228] = 27
DATA bitrev_size256_radix4<>+0x728(SB)/8, $91  // bitrev[229] = 91
DATA bitrev_size256_radix4<>+0x730(SB)/8, $155 // bitrev[230] = 155
DATA bitrev_size256_radix4<>+0x738(SB)/8, $219 // bitrev[231] = 219
DATA bitrev_size256_radix4<>+0x740(SB)/8, $43  // bitrev[232] = 43
DATA bitrev_size256_radix4<>+0x748(SB)/8, $107 // bitrev[233] = 107
DATA bitrev_size256_radix4<>+0x750(SB)/8, $171 // bitrev[234] = 171
DATA bitrev_size256_radix4<>+0x758(SB)/8, $235 // bitrev[235] = 235
DATA bitrev_size256_radix4<>+0x760(SB)/8, $59  // bitrev[236] = 59
DATA bitrev_size256_radix4<>+0x768(SB)/8, $123 // bitrev[237] = 123
DATA bitrev_size256_radix4<>+0x770(SB)/8, $187 // bitrev[238] = 187
DATA bitrev_size256_radix4<>+0x778(SB)/8, $251 // bitrev[239] = 251
DATA bitrev_size256_radix4<>+0x780(SB)/8, $15  // bitrev[240] = 15
DATA bitrev_size256_radix4<>+0x788(SB)/8, $79  // bitrev[241] = 79
DATA bitrev_size256_radix4<>+0x790(SB)/8, $143 // bitrev[242] = 143
DATA bitrev_size256_radix4<>+0x798(SB)/8, $207 // bitrev[243] = 207
DATA bitrev_size256_radix4<>+0x7a0(SB)/8, $31  // bitrev[244] = 31
DATA bitrev_size256_radix4<>+0x7a8(SB)/8, $95  // bitrev[245] = 95
DATA bitrev_size256_radix4<>+0x7b0(SB)/8, $159 // bitrev[246] = 159
DATA bitrev_size256_radix4<>+0x7b8(SB)/8, $223 // bitrev[247] = 223
DATA bitrev_size256_radix4<>+0x7c0(SB)/8, $47  // bitrev[248] = 47
DATA bitrev_size256_radix4<>+0x7c8(SB)/8, $111 // bitrev[249] = 111
DATA bitrev_size256_radix4<>+0x7d0(SB)/8, $175 // bitrev[250] = 175
DATA bitrev_size256_radix4<>+0x7d8(SB)/8, $239 // bitrev[251] = 239
DATA bitrev_size256_radix4<>+0x7e0(SB)/8, $63  // bitrev[252] = 63
DATA bitrev_size256_radix4<>+0x7e8(SB)/8, $127 // bitrev[253] = 127
DATA bitrev_size256_radix4<>+0x7f0(SB)/8, $191 // bitrev[254] = 191
DATA bitrev_size256_radix4<>+0x7f8(SB)/8, $255 // bitrev[255] = 255

GLOBL bitrev_size256_radix4<>(SB), RODATA, $2048

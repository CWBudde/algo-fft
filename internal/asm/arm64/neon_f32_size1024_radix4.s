//go:build arm64 && !purego

// ===========================================================================
// NEON Size-1024 Radix-4 FFT Kernels for ARM64
// ===========================================================================
//
// Size 1024 = 4^5, radix-4 algorithm uses 5 stages:
//   Stage 1: 256 groups × 1 butterfly, stride=4,   no twiddle multiply (W^0 = 1)
//   Stage 2: 64 groups × 4 butterflies, twiddle step=64
//   Stage 3: 16 groups × 16 butterflies, twiddle step=16
//   Stage 4: 4 groups × 64 butterflies, twiddle step=4
//   Stage 5: 1 group × 256 butterflies, twiddle step=1
//
// ===========================================================================

#include "textflag.h"

DATA ·neonInv1024Radix4+0(SB)/4, $0x3a800000 // 1/1024
GLOBL ·neonInv1024Radix4(SB), RODATA, $4

// Forward transform, size 1024, complex64, radix-4 variant
TEXT ·ForwardNEONSize1024Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src_len+32(FP), R13

	CMP  $1024, R13
	BNE  neon1024r4_return_false

	MOVD dst_len+8(FP), R0
	CMP  $1024, R0
	BLT  neon1024r4_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $1024, R0
	BLT  neon1024r4_return_false

	MOVD scratch_len+80(FP), R0
	CMP  $1024, R0
	BLT  neon1024r4_return_false

	// Load static bit-reversal table
	MOVD $bitrev_size1024_radix4<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon1024r4_use_dst
	MOVD R11, R8

neon1024r4_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon1024r4_bitrev_loop:
	CMP  $1024, R0
	BGE  neon1024r4_stage1

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
	B    neon1024r4_bitrev_loop

neon1024r4_stage1:
	// Stage 1: 256 radix-4 butterflies
	MOVD $0, R0

neon1024r4_stage1_loop:
	CMP  $1024, R0
	BGE  neon1024r4_stage2

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
	B    neon1024r4_stage1_loop

neon1024r4_stage2:
	// Stage 2: 64 groups × 4 butterflies, twiddle step=64
	MOVD $0, R0

neon1024r4_stage2_outer:
	CMP  $1024, R0
	BGE  neon1024r4_stage3

	MOVD $0, R1

neon1024r4_stage2_inner:
	CMP  $4, R1
	BGE  neon1024r4_stage2_next

	ADD  R0, R1, R2
	ADD  $4, R2, R3
	ADD  $8, R2, R4
	ADD  $12, R2, R5

	// twiddle indices: j*64, j*128, j*192
	LSL  $6, R1, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F0
	FMOVS 4(R6), F1

	LSL  $7, R1, R7
	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F2
	FMOVS 4(R7), F3

	LSL  $6, R1, R6
	LSL  $7, R1, R7
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
	B    neon1024r4_stage2_inner

neon1024r4_stage2_next:
	ADD  $16, R0, R0
	B    neon1024r4_stage2_outer

neon1024r4_stage3:
	// Stage 3: 16 groups × 16 butterflies, twiddle step=16
	MOVD $0, R0

neon1024r4_stage3_outer:
	CMP  $1024, R0
	BGE  neon1024r4_stage4

	MOVD $0, R1

neon1024r4_stage3_inner:
	CMP  $16, R1
	BGE  neon1024r4_stage3_next

	ADD  R0, R1, R2
	ADD  $16, R2, R3
	ADD  $32, R2, R4
	ADD  $48, R2, R5

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
	B    neon1024r4_stage3_inner

neon1024r4_stage3_next:
	ADD  $64, R0, R0
	B    neon1024r4_stage3_outer

neon1024r4_stage4:
	// Stage 4: 4 groups × 64 butterflies, twiddle step=4
	MOVD $0, R0

neon1024r4_stage4_outer:
	CMP  $1024, R0
	BGE  neon1024r4_stage5

	MOVD $0, R1

neon1024r4_stage4_inner:
	CMP  $64, R1
	BGE  neon1024r4_stage4_next

	ADD  R0, R1, R2
	ADD  $64, R2, R3
	ADD  $128, R2, R4
	ADD  $192, R2, R5

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
	B    neon1024r4_stage4_inner

neon1024r4_stage4_next:
	ADD  $256, R0, R0
	B    neon1024r4_stage4_outer

neon1024r4_stage5:
	// Stage 5: 1 group × 256 butterflies, twiddle step=1
	MOVD $0, R0

neon1024r4_stage5_loop:
	CMP  $256, R0
	BGE  neon1024r4_done

	MOVD R0, R1
	ADD  $256, R1, R2
	ADD  $512, R1, R3
	ADD  $768, R1, R4

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
	B    neon1024r4_stage5_loop

neon1024r4_done:
	CMP  R8, R20
	BEQ  neon1024r4_return_true

	MOVD $0, R0
neon1024r4_copy_loop:
	CMP  $1024, R0
	BGE  neon1024r4_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon1024r4_copy_loop

neon1024r4_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon1024r4_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Inverse transform, size 1024, complex64, radix-4 variant
TEXT ·InverseNEONSize1024Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src_len+32(FP), R13

	CMP  $1024, R13
	BNE  neon1024r4_inv_return_false

	MOVD dst_len+8(FP), R0
	CMP  $1024, R0
	BLT  neon1024r4_inv_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $1024, R0
	BLT  neon1024r4_inv_return_false

	MOVD scratch_len+80(FP), R0
	CMP  $1024, R0
	BLT  neon1024r4_inv_return_false

	// Load static bit-reversal table
	MOVD $bitrev_size1024_radix4<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon1024r4_inv_use_dst
	MOVD R11, R8

neon1024r4_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon1024r4_inv_bitrev_loop:
	CMP  $1024, R0
	BGE  neon1024r4_inv_stage1

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
	B    neon1024r4_inv_bitrev_loop

neon1024r4_inv_stage1:
	// Stage 1 (inverse variant)
	MOVD $0, R0

neon1024r4_inv_stage1_loop:
	CMP  $1024, R0
	BGE  neon1024r4_inv_stage2

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
	B    neon1024r4_inv_stage1_loop

neon1024r4_inv_stage2:
	// Stage 2 with conjugated twiddles
	MOVD $0, R0

neon1024r4_inv_stage2_outer:
	CMP  $1024, R0
	BGE  neon1024r4_inv_stage3

	MOVD $0, R1

neon1024r4_inv_stage2_inner:
	CMP  $4, R1
	BGE  neon1024r4_inv_stage2_next

	ADD  R0, R1, R2
	ADD  $4, R2, R3
	ADD  $8, R2, R4
	ADD  $12, R2, R5

	// twiddle indices: j*64, j*128, j*192 (conjugated)
	LSL  $6, R1, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F0
	FMOVS 4(R6), F1
	FNEGS  F1, F1

	LSL  $7, R1, R7
	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F2
	FMOVS 4(R7), F3
	FNEGS  F3, F3

	LSL  $6, R1, R6
	LSL  $7, R1, R7
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
	B    neon1024r4_inv_stage2_inner

neon1024r4_inv_stage2_next:
	ADD  $16, R0, R0
	B    neon1024r4_inv_stage2_outer

neon1024r4_inv_stage3:
	// Stage 3 with conjugated twiddles
	MOVD $0, R0

neon1024r4_inv_stage3_outer:
	CMP  $1024, R0
	BGE  neon1024r4_inv_stage4

	MOVD $0, R1

neon1024r4_inv_stage3_inner:
	CMP  $16, R1
	BGE  neon1024r4_inv_stage3_next

	ADD  R0, R1, R2
	ADD  $16, R2, R3
	ADD  $32, R2, R4
	ADD  $48, R2, R5

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
	B    neon1024r4_inv_stage3_inner

neon1024r4_inv_stage3_next:
	ADD  $64, R0, R0
	B    neon1024r4_inv_stage3_outer

neon1024r4_inv_stage4:
	// Stage 4 with conjugated twiddles
	MOVD $0, R0

neon1024r4_inv_stage4_outer:
	CMP  $1024, R0
	BGE  neon1024r4_inv_stage5

	MOVD $0, R1

neon1024r4_inv_stage4_inner:
	CMP  $64, R1
	BGE  neon1024r4_inv_stage4_next

	ADD  R0, R1, R2
	ADD  $64, R2, R3
	ADD  $128, R2, R4
	ADD  $192, R2, R5

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
	B    neon1024r4_inv_stage4_inner

neon1024r4_inv_stage4_next:
	ADD  $256, R0, R0
	B    neon1024r4_inv_stage4_outer

neon1024r4_inv_stage5:
	// Stage 5 with conjugated twiddles
	MOVD $0, R0

neon1024r4_inv_stage5_loop:
	CMP  $256, R0
	BGE  neon1024r4_inv_done

	MOVD R0, R1
	ADD  $256, R1, R2
	ADD  $512, R1, R3
	ADD  $768, R1, R4

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
	B    neon1024r4_inv_stage5_loop

neon1024r4_inv_done:
	CMP  R8, R20
	BEQ  neon1024r4_inv_scale

	MOVD $0, R0
neon1024r4_inv_copy_loop:
	CMP  $1024, R0
	BGE  neon1024r4_inv_scale
	LSL  $3, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon1024r4_inv_copy_loop

neon1024r4_inv_scale:
	MOVD $·neonInv1024Radix4(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon1024r4_inv_scale_loop:
	CMP  $1024, R0
	BGE  neon1024r4_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon1024r4_inv_scale_loop

neon1024r4_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon1024r4_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET
// Size-1024 Radix-4 bit-reversal table (base-4 digit reversal)
DATA bitrev_size1024_radix4<>+0x000(SB)/8, $0    // bitrev[0] = 0
DATA bitrev_size1024_radix4<>+0x008(SB)/8, $256  // bitrev[1] = 256
DATA bitrev_size1024_radix4<>+0x010(SB)/8, $512  // bitrev[2] = 512
DATA bitrev_size1024_radix4<>+0x018(SB)/8, $768  // bitrev[3] = 768
DATA bitrev_size1024_radix4<>+0x020(SB)/8, $64   // bitrev[4] = 64
DATA bitrev_size1024_radix4<>+0x028(SB)/8, $320  // bitrev[5] = 320
DATA bitrev_size1024_radix4<>+0x030(SB)/8, $576  // bitrev[6] = 576
DATA bitrev_size1024_radix4<>+0x038(SB)/8, $832  // bitrev[7] = 832
DATA bitrev_size1024_radix4<>+0x040(SB)/8, $128  // bitrev[8] = 128
DATA bitrev_size1024_radix4<>+0x048(SB)/8, $384  // bitrev[9] = 384
DATA bitrev_size1024_radix4<>+0x050(SB)/8, $640  // bitrev[10] = 640
DATA bitrev_size1024_radix4<>+0x058(SB)/8, $896  // bitrev[11] = 896
DATA bitrev_size1024_radix4<>+0x060(SB)/8, $192  // bitrev[12] = 192
DATA bitrev_size1024_radix4<>+0x068(SB)/8, $448  // bitrev[13] = 448
DATA bitrev_size1024_radix4<>+0x070(SB)/8, $704  // bitrev[14] = 704
DATA bitrev_size1024_radix4<>+0x078(SB)/8, $960  // bitrev[15] = 960
DATA bitrev_size1024_radix4<>+0x080(SB)/8, $16   // bitrev[16] = 16
DATA bitrev_size1024_radix4<>+0x088(SB)/8, $272  // bitrev[17] = 272
DATA bitrev_size1024_radix4<>+0x090(SB)/8, $528  // bitrev[18] = 528
DATA bitrev_size1024_radix4<>+0x098(SB)/8, $784  // bitrev[19] = 784
DATA bitrev_size1024_radix4<>+0x0a0(SB)/8, $80   // bitrev[20] = 80
DATA bitrev_size1024_radix4<>+0x0a8(SB)/8, $336  // bitrev[21] = 336
DATA bitrev_size1024_radix4<>+0x0b0(SB)/8, $592  // bitrev[22] = 592
DATA bitrev_size1024_radix4<>+0x0b8(SB)/8, $848  // bitrev[23] = 848
DATA bitrev_size1024_radix4<>+0x0c0(SB)/8, $144  // bitrev[24] = 144
DATA bitrev_size1024_radix4<>+0x0c8(SB)/8, $400  // bitrev[25] = 400
DATA bitrev_size1024_radix4<>+0x0d0(SB)/8, $656  // bitrev[26] = 656
DATA bitrev_size1024_radix4<>+0x0d8(SB)/8, $912  // bitrev[27] = 912
DATA bitrev_size1024_radix4<>+0x0e0(SB)/8, $208  // bitrev[28] = 208
DATA bitrev_size1024_radix4<>+0x0e8(SB)/8, $464  // bitrev[29] = 464
DATA bitrev_size1024_radix4<>+0x0f0(SB)/8, $720  // bitrev[30] = 720
DATA bitrev_size1024_radix4<>+0x0f8(SB)/8, $976  // bitrev[31] = 976
DATA bitrev_size1024_radix4<>+0x100(SB)/8, $32   // bitrev[32] = 32
DATA bitrev_size1024_radix4<>+0x108(SB)/8, $288  // bitrev[33] = 288
DATA bitrev_size1024_radix4<>+0x110(SB)/8, $544  // bitrev[34] = 544
DATA bitrev_size1024_radix4<>+0x118(SB)/8, $800  // bitrev[35] = 800
DATA bitrev_size1024_radix4<>+0x120(SB)/8, $96   // bitrev[36] = 96
DATA bitrev_size1024_radix4<>+0x128(SB)/8, $352  // bitrev[37] = 352
DATA bitrev_size1024_radix4<>+0x130(SB)/8, $608  // bitrev[38] = 608
DATA bitrev_size1024_radix4<>+0x138(SB)/8, $864  // bitrev[39] = 864
DATA bitrev_size1024_radix4<>+0x140(SB)/8, $160  // bitrev[40] = 160
DATA bitrev_size1024_radix4<>+0x148(SB)/8, $416  // bitrev[41] = 416
DATA bitrev_size1024_radix4<>+0x150(SB)/8, $672  // bitrev[42] = 672
DATA bitrev_size1024_radix4<>+0x158(SB)/8, $928  // bitrev[43] = 928
DATA bitrev_size1024_radix4<>+0x160(SB)/8, $224  // bitrev[44] = 224
DATA bitrev_size1024_radix4<>+0x168(SB)/8, $480  // bitrev[45] = 480
DATA bitrev_size1024_radix4<>+0x170(SB)/8, $736  // bitrev[46] = 736
DATA bitrev_size1024_radix4<>+0x178(SB)/8, $992  // bitrev[47] = 992
DATA bitrev_size1024_radix4<>+0x180(SB)/8, $48   // bitrev[48] = 48
DATA bitrev_size1024_radix4<>+0x188(SB)/8, $304  // bitrev[49] = 304
DATA bitrev_size1024_radix4<>+0x190(SB)/8, $560  // bitrev[50] = 560
DATA bitrev_size1024_radix4<>+0x198(SB)/8, $816  // bitrev[51] = 816
DATA bitrev_size1024_radix4<>+0x1a0(SB)/8, $112  // bitrev[52] = 112
DATA bitrev_size1024_radix4<>+0x1a8(SB)/8, $368  // bitrev[53] = 368
DATA bitrev_size1024_radix4<>+0x1b0(SB)/8, $624  // bitrev[54] = 624
DATA bitrev_size1024_radix4<>+0x1b8(SB)/8, $880  // bitrev[55] = 880
DATA bitrev_size1024_radix4<>+0x1c0(SB)/8, $176  // bitrev[56] = 176
DATA bitrev_size1024_radix4<>+0x1c8(SB)/8, $432  // bitrev[57] = 432
DATA bitrev_size1024_radix4<>+0x1d0(SB)/8, $688  // bitrev[58] = 688
DATA bitrev_size1024_radix4<>+0x1d8(SB)/8, $944  // bitrev[59] = 944
DATA bitrev_size1024_radix4<>+0x1e0(SB)/8, $240  // bitrev[60] = 240
DATA bitrev_size1024_radix4<>+0x1e8(SB)/8, $496  // bitrev[61] = 496
DATA bitrev_size1024_radix4<>+0x1f0(SB)/8, $752  // bitrev[62] = 752
DATA bitrev_size1024_radix4<>+0x1f8(SB)/8, $1008 // bitrev[63] = 1008
DATA bitrev_size1024_radix4<>+0x200(SB)/8, $4    // bitrev[64] = 4
DATA bitrev_size1024_radix4<>+0x208(SB)/8, $260  // bitrev[65] = 260
DATA bitrev_size1024_radix4<>+0x210(SB)/8, $516  // bitrev[66] = 516
DATA bitrev_size1024_radix4<>+0x218(SB)/8, $772  // bitrev[67] = 772
DATA bitrev_size1024_radix4<>+0x220(SB)/8, $68   // bitrev[68] = 68
DATA bitrev_size1024_radix4<>+0x228(SB)/8, $324  // bitrev[69] = 324
DATA bitrev_size1024_radix4<>+0x230(SB)/8, $580  // bitrev[70] = 580
DATA bitrev_size1024_radix4<>+0x238(SB)/8, $836  // bitrev[71] = 836
DATA bitrev_size1024_radix4<>+0x240(SB)/8, $132  // bitrev[72] = 132
DATA bitrev_size1024_radix4<>+0x248(SB)/8, $388  // bitrev[73] = 388
DATA bitrev_size1024_radix4<>+0x250(SB)/8, $644  // bitrev[74] = 644
DATA bitrev_size1024_radix4<>+0x258(SB)/8, $900  // bitrev[75] = 900
DATA bitrev_size1024_radix4<>+0x260(SB)/8, $196  // bitrev[76] = 196
DATA bitrev_size1024_radix4<>+0x268(SB)/8, $452  // bitrev[77] = 452
DATA bitrev_size1024_radix4<>+0x270(SB)/8, $708  // bitrev[78] = 708
DATA bitrev_size1024_radix4<>+0x278(SB)/8, $964  // bitrev[79] = 964
DATA bitrev_size1024_radix4<>+0x280(SB)/8, $20   // bitrev[80] = 20
DATA bitrev_size1024_radix4<>+0x288(SB)/8, $276  // bitrev[81] = 276
DATA bitrev_size1024_radix4<>+0x290(SB)/8, $532  // bitrev[82] = 532
DATA bitrev_size1024_radix4<>+0x298(SB)/8, $788  // bitrev[83] = 788
DATA bitrev_size1024_radix4<>+0x2a0(SB)/8, $84   // bitrev[84] = 84
DATA bitrev_size1024_radix4<>+0x2a8(SB)/8, $340  // bitrev[85] = 340
DATA bitrev_size1024_radix4<>+0x2b0(SB)/8, $596  // bitrev[86] = 596
DATA bitrev_size1024_radix4<>+0x2b8(SB)/8, $852  // bitrev[87] = 852
DATA bitrev_size1024_radix4<>+0x2c0(SB)/8, $148  // bitrev[88] = 148
DATA bitrev_size1024_radix4<>+0x2c8(SB)/8, $404  // bitrev[89] = 404
DATA bitrev_size1024_radix4<>+0x2d0(SB)/8, $660  // bitrev[90] = 660
DATA bitrev_size1024_radix4<>+0x2d8(SB)/8, $916  // bitrev[91] = 916
DATA bitrev_size1024_radix4<>+0x2e0(SB)/8, $212  // bitrev[92] = 212
DATA bitrev_size1024_radix4<>+0x2e8(SB)/8, $468  // bitrev[93] = 468
DATA bitrev_size1024_radix4<>+0x2f0(SB)/8, $724  // bitrev[94] = 724
DATA bitrev_size1024_radix4<>+0x2f8(SB)/8, $980  // bitrev[95] = 980
DATA bitrev_size1024_radix4<>+0x300(SB)/8, $36   // bitrev[96] = 36
DATA bitrev_size1024_radix4<>+0x308(SB)/8, $292  // bitrev[97] = 292
DATA bitrev_size1024_radix4<>+0x310(SB)/8, $548  // bitrev[98] = 548
DATA bitrev_size1024_radix4<>+0x318(SB)/8, $804  // bitrev[99] = 804
DATA bitrev_size1024_radix4<>+0x320(SB)/8, $100  // bitrev[100] = 100
DATA bitrev_size1024_radix4<>+0x328(SB)/8, $356  // bitrev[101] = 356
DATA bitrev_size1024_radix4<>+0x330(SB)/8, $612  // bitrev[102] = 612
DATA bitrev_size1024_radix4<>+0x338(SB)/8, $868  // bitrev[103] = 868
DATA bitrev_size1024_radix4<>+0x340(SB)/8, $164  // bitrev[104] = 164
DATA bitrev_size1024_radix4<>+0x348(SB)/8, $420  // bitrev[105] = 420
DATA bitrev_size1024_radix4<>+0x350(SB)/8, $676  // bitrev[106] = 676
DATA bitrev_size1024_radix4<>+0x358(SB)/8, $932  // bitrev[107] = 932
DATA bitrev_size1024_radix4<>+0x360(SB)/8, $228  // bitrev[108] = 228
DATA bitrev_size1024_radix4<>+0x368(SB)/8, $484  // bitrev[109] = 484
DATA bitrev_size1024_radix4<>+0x370(SB)/8, $740  // bitrev[110] = 740
DATA bitrev_size1024_radix4<>+0x378(SB)/8, $996  // bitrev[111] = 996
DATA bitrev_size1024_radix4<>+0x380(SB)/8, $52   // bitrev[112] = 52
DATA bitrev_size1024_radix4<>+0x388(SB)/8, $308  // bitrev[113] = 308
DATA bitrev_size1024_radix4<>+0x390(SB)/8, $564  // bitrev[114] = 564
DATA bitrev_size1024_radix4<>+0x398(SB)/8, $820  // bitrev[115] = 820
DATA bitrev_size1024_radix4<>+0x3a0(SB)/8, $116  // bitrev[116] = 116
DATA bitrev_size1024_radix4<>+0x3a8(SB)/8, $372  // bitrev[117] = 372
DATA bitrev_size1024_radix4<>+0x3b0(SB)/8, $628  // bitrev[118] = 628
DATA bitrev_size1024_radix4<>+0x3b8(SB)/8, $884  // bitrev[119] = 884
DATA bitrev_size1024_radix4<>+0x3c0(SB)/8, $180  // bitrev[120] = 180
DATA bitrev_size1024_radix4<>+0x3c8(SB)/8, $436  // bitrev[121] = 436
DATA bitrev_size1024_radix4<>+0x3d0(SB)/8, $692  // bitrev[122] = 692
DATA bitrev_size1024_radix4<>+0x3d8(SB)/8, $948  // bitrev[123] = 948
DATA bitrev_size1024_radix4<>+0x3e0(SB)/8, $244  // bitrev[124] = 244
DATA bitrev_size1024_radix4<>+0x3e8(SB)/8, $500  // bitrev[125] = 500
DATA bitrev_size1024_radix4<>+0x3f0(SB)/8, $756  // bitrev[126] = 756
DATA bitrev_size1024_radix4<>+0x3f8(SB)/8, $1012 // bitrev[127] = 1012
DATA bitrev_size1024_radix4<>+0x400(SB)/8, $8    // bitrev[128] = 8
DATA bitrev_size1024_radix4<>+0x408(SB)/8, $264  // bitrev[129] = 264
DATA bitrev_size1024_radix4<>+0x410(SB)/8, $520  // bitrev[130] = 520
DATA bitrev_size1024_radix4<>+0x418(SB)/8, $776  // bitrev[131] = 776
DATA bitrev_size1024_radix4<>+0x420(SB)/8, $72   // bitrev[132] = 72
DATA bitrev_size1024_radix4<>+0x428(SB)/8, $328  // bitrev[133] = 328
DATA bitrev_size1024_radix4<>+0x430(SB)/8, $584  // bitrev[134] = 584
DATA bitrev_size1024_radix4<>+0x438(SB)/8, $840  // bitrev[135] = 840
DATA bitrev_size1024_radix4<>+0x440(SB)/8, $136  // bitrev[136] = 136
DATA bitrev_size1024_radix4<>+0x448(SB)/8, $392  // bitrev[137] = 392
DATA bitrev_size1024_radix4<>+0x450(SB)/8, $648  // bitrev[138] = 648
DATA bitrev_size1024_radix4<>+0x458(SB)/8, $904  // bitrev[139] = 904
DATA bitrev_size1024_radix4<>+0x460(SB)/8, $200  // bitrev[140] = 200
DATA bitrev_size1024_radix4<>+0x468(SB)/8, $456  // bitrev[141] = 456
DATA bitrev_size1024_radix4<>+0x470(SB)/8, $712  // bitrev[142] = 712
DATA bitrev_size1024_radix4<>+0x478(SB)/8, $968  // bitrev[143] = 968
DATA bitrev_size1024_radix4<>+0x480(SB)/8, $24   // bitrev[144] = 24
DATA bitrev_size1024_radix4<>+0x488(SB)/8, $280  // bitrev[145] = 280
DATA bitrev_size1024_radix4<>+0x490(SB)/8, $536  // bitrev[146] = 536
DATA bitrev_size1024_radix4<>+0x498(SB)/8, $792  // bitrev[147] = 792
DATA bitrev_size1024_radix4<>+0x4a0(SB)/8, $88   // bitrev[148] = 88
DATA bitrev_size1024_radix4<>+0x4a8(SB)/8, $344  // bitrev[149] = 344
DATA bitrev_size1024_radix4<>+0x4b0(SB)/8, $600  // bitrev[150] = 600
DATA bitrev_size1024_radix4<>+0x4b8(SB)/8, $856  // bitrev[151] = 856
DATA bitrev_size1024_radix4<>+0x4c0(SB)/8, $152  // bitrev[152] = 152
DATA bitrev_size1024_radix4<>+0x4c8(SB)/8, $408  // bitrev[153] = 408
DATA bitrev_size1024_radix4<>+0x4d0(SB)/8, $664  // bitrev[154] = 664
DATA bitrev_size1024_radix4<>+0x4d8(SB)/8, $920  // bitrev[155] = 920
DATA bitrev_size1024_radix4<>+0x4e0(SB)/8, $216  // bitrev[156] = 216
DATA bitrev_size1024_radix4<>+0x4e8(SB)/8, $472  // bitrev[157] = 472
DATA bitrev_size1024_radix4<>+0x4f0(SB)/8, $728  // bitrev[158] = 728
DATA bitrev_size1024_radix4<>+0x4f8(SB)/8, $984  // bitrev[159] = 984
DATA bitrev_size1024_radix4<>+0x500(SB)/8, $40   // bitrev[160] = 40
DATA bitrev_size1024_radix4<>+0x508(SB)/8, $296  // bitrev[161] = 296
DATA bitrev_size1024_radix4<>+0x510(SB)/8, $552  // bitrev[162] = 552
DATA bitrev_size1024_radix4<>+0x518(SB)/8, $808  // bitrev[163] = 808
DATA bitrev_size1024_radix4<>+0x520(SB)/8, $104  // bitrev[164] = 104
DATA bitrev_size1024_radix4<>+0x528(SB)/8, $360  // bitrev[165] = 360
DATA bitrev_size1024_radix4<>+0x530(SB)/8, $616  // bitrev[166] = 616
DATA bitrev_size1024_radix4<>+0x538(SB)/8, $872  // bitrev[167] = 872
DATA bitrev_size1024_radix4<>+0x540(SB)/8, $168  // bitrev[168] = 168
DATA bitrev_size1024_radix4<>+0x548(SB)/8, $424  // bitrev[169] = 424
DATA bitrev_size1024_radix4<>+0x550(SB)/8, $680  // bitrev[170] = 680
DATA bitrev_size1024_radix4<>+0x558(SB)/8, $936  // bitrev[171] = 936
DATA bitrev_size1024_radix4<>+0x560(SB)/8, $232  // bitrev[172] = 232
DATA bitrev_size1024_radix4<>+0x568(SB)/8, $488  // bitrev[173] = 488
DATA bitrev_size1024_radix4<>+0x570(SB)/8, $744  // bitrev[174] = 744
DATA bitrev_size1024_radix4<>+0x578(SB)/8, $1000 // bitrev[175] = 1000
DATA bitrev_size1024_radix4<>+0x580(SB)/8, $56   // bitrev[176] = 56
DATA bitrev_size1024_radix4<>+0x588(SB)/8, $312  // bitrev[177] = 312
DATA bitrev_size1024_radix4<>+0x590(SB)/8, $568  // bitrev[178] = 568
DATA bitrev_size1024_radix4<>+0x598(SB)/8, $824  // bitrev[179] = 824
DATA bitrev_size1024_radix4<>+0x5a0(SB)/8, $120  // bitrev[180] = 120
DATA bitrev_size1024_radix4<>+0x5a8(SB)/8, $376  // bitrev[181] = 376
DATA bitrev_size1024_radix4<>+0x5b0(SB)/8, $632  // bitrev[182] = 632
DATA bitrev_size1024_radix4<>+0x5b8(SB)/8, $888  // bitrev[183] = 888
DATA bitrev_size1024_radix4<>+0x5c0(SB)/8, $184  // bitrev[184] = 184
DATA bitrev_size1024_radix4<>+0x5c8(SB)/8, $440  // bitrev[185] = 440
DATA bitrev_size1024_radix4<>+0x5d0(SB)/8, $696  // bitrev[186] = 696
DATA bitrev_size1024_radix4<>+0x5d8(SB)/8, $952  // bitrev[187] = 952
DATA bitrev_size1024_radix4<>+0x5e0(SB)/8, $248  // bitrev[188] = 248
DATA bitrev_size1024_radix4<>+0x5e8(SB)/8, $504  // bitrev[189] = 504
DATA bitrev_size1024_radix4<>+0x5f0(SB)/8, $760  // bitrev[190] = 760
DATA bitrev_size1024_radix4<>+0x5f8(SB)/8, $1016 // bitrev[191] = 1016
DATA bitrev_size1024_radix4<>+0x600(SB)/8, $12   // bitrev[192] = 12
DATA bitrev_size1024_radix4<>+0x608(SB)/8, $268  // bitrev[193] = 268
DATA bitrev_size1024_radix4<>+0x610(SB)/8, $524  // bitrev[194] = 524
DATA bitrev_size1024_radix4<>+0x618(SB)/8, $780  // bitrev[195] = 780
DATA bitrev_size1024_radix4<>+0x620(SB)/8, $76   // bitrev[196] = 76
DATA bitrev_size1024_radix4<>+0x628(SB)/8, $332  // bitrev[197] = 332
DATA bitrev_size1024_radix4<>+0x630(SB)/8, $588  // bitrev[198] = 588
DATA bitrev_size1024_radix4<>+0x638(SB)/8, $844  // bitrev[199] = 844
DATA bitrev_size1024_radix4<>+0x640(SB)/8, $140  // bitrev[200] = 140
DATA bitrev_size1024_radix4<>+0x648(SB)/8, $396  // bitrev[201] = 396
DATA bitrev_size1024_radix4<>+0x650(SB)/8, $652  // bitrev[202] = 652
DATA bitrev_size1024_radix4<>+0x658(SB)/8, $908  // bitrev[203] = 908
DATA bitrev_size1024_radix4<>+0x660(SB)/8, $204  // bitrev[204] = 204
DATA bitrev_size1024_radix4<>+0x668(SB)/8, $460  // bitrev[205] = 460
DATA bitrev_size1024_radix4<>+0x670(SB)/8, $716  // bitrev[206] = 716
DATA bitrev_size1024_radix4<>+0x678(SB)/8, $972  // bitrev[207] = 972
DATA bitrev_size1024_radix4<>+0x680(SB)/8, $28   // bitrev[208] = 28
DATA bitrev_size1024_radix4<>+0x688(SB)/8, $284  // bitrev[209] = 284
DATA bitrev_size1024_radix4<>+0x690(SB)/8, $540  // bitrev[210] = 540
DATA bitrev_size1024_radix4<>+0x698(SB)/8, $796  // bitrev[211] = 796
DATA bitrev_size1024_radix4<>+0x6a0(SB)/8, $92   // bitrev[212] = 92
DATA bitrev_size1024_radix4<>+0x6a8(SB)/8, $348  // bitrev[213] = 348
DATA bitrev_size1024_radix4<>+0x6b0(SB)/8, $604  // bitrev[214] = 604
DATA bitrev_size1024_radix4<>+0x6b8(SB)/8, $860  // bitrev[215] = 860
DATA bitrev_size1024_radix4<>+0x6c0(SB)/8, $156  // bitrev[216] = 156
DATA bitrev_size1024_radix4<>+0x6c8(SB)/8, $412  // bitrev[217] = 412
DATA bitrev_size1024_radix4<>+0x6d0(SB)/8, $668  // bitrev[218] = 668
DATA bitrev_size1024_radix4<>+0x6d8(SB)/8, $924  // bitrev[219] = 924
DATA bitrev_size1024_radix4<>+0x6e0(SB)/8, $220  // bitrev[220] = 220
DATA bitrev_size1024_radix4<>+0x6e8(SB)/8, $476  // bitrev[221] = 476
DATA bitrev_size1024_radix4<>+0x6f0(SB)/8, $732  // bitrev[222] = 732
DATA bitrev_size1024_radix4<>+0x6f8(SB)/8, $988  // bitrev[223] = 988
DATA bitrev_size1024_radix4<>+0x700(SB)/8, $44   // bitrev[224] = 44
DATA bitrev_size1024_radix4<>+0x708(SB)/8, $300  // bitrev[225] = 300
DATA bitrev_size1024_radix4<>+0x710(SB)/8, $556  // bitrev[226] = 556
DATA bitrev_size1024_radix4<>+0x718(SB)/8, $812  // bitrev[227] = 812
DATA bitrev_size1024_radix4<>+0x720(SB)/8, $108  // bitrev[228] = 108
DATA bitrev_size1024_radix4<>+0x728(SB)/8, $364  // bitrev[229] = 364
DATA bitrev_size1024_radix4<>+0x730(SB)/8, $620  // bitrev[230] = 620
DATA bitrev_size1024_radix4<>+0x738(SB)/8, $876  // bitrev[231] = 876
DATA bitrev_size1024_radix4<>+0x740(SB)/8, $172  // bitrev[232] = 172
DATA bitrev_size1024_radix4<>+0x748(SB)/8, $428  // bitrev[233] = 428
DATA bitrev_size1024_radix4<>+0x750(SB)/8, $684  // bitrev[234] = 684
DATA bitrev_size1024_radix4<>+0x758(SB)/8, $940  // bitrev[235] = 940
DATA bitrev_size1024_radix4<>+0x760(SB)/8, $236  // bitrev[236] = 236
DATA bitrev_size1024_radix4<>+0x768(SB)/8, $492  // bitrev[237] = 492
DATA bitrev_size1024_radix4<>+0x770(SB)/8, $748  // bitrev[238] = 748
DATA bitrev_size1024_radix4<>+0x778(SB)/8, $1004 // bitrev[239] = 1004
DATA bitrev_size1024_radix4<>+0x780(SB)/8, $60   // bitrev[240] = 60
DATA bitrev_size1024_radix4<>+0x788(SB)/8, $316  // bitrev[241] = 316
DATA bitrev_size1024_radix4<>+0x790(SB)/8, $572  // bitrev[242] = 572
DATA bitrev_size1024_radix4<>+0x798(SB)/8, $828  // bitrev[243] = 828
DATA bitrev_size1024_radix4<>+0x7a0(SB)/8, $124  // bitrev[244] = 124
DATA bitrev_size1024_radix4<>+0x7a8(SB)/8, $380  // bitrev[245] = 380
DATA bitrev_size1024_radix4<>+0x7b0(SB)/8, $636  // bitrev[246] = 636
DATA bitrev_size1024_radix4<>+0x7b8(SB)/8, $892  // bitrev[247] = 892
DATA bitrev_size1024_radix4<>+0x7c0(SB)/8, $188  // bitrev[248] = 188
DATA bitrev_size1024_radix4<>+0x7c8(SB)/8, $444  // bitrev[249] = 444
DATA bitrev_size1024_radix4<>+0x7d0(SB)/8, $700  // bitrev[250] = 700
DATA bitrev_size1024_radix4<>+0x7d8(SB)/8, $956  // bitrev[251] = 956
DATA bitrev_size1024_radix4<>+0x7e0(SB)/8, $252  // bitrev[252] = 252
DATA bitrev_size1024_radix4<>+0x7e8(SB)/8, $508  // bitrev[253] = 508
DATA bitrev_size1024_radix4<>+0x7f0(SB)/8, $764  // bitrev[254] = 764
DATA bitrev_size1024_radix4<>+0x7f8(SB)/8, $1020 // bitrev[255] = 1020
DATA bitrev_size1024_radix4<>+0x800(SB)/8, $1    // bitrev[256] = 1
DATA bitrev_size1024_radix4<>+0x808(SB)/8, $257  // bitrev[257] = 257
DATA bitrev_size1024_radix4<>+0x810(SB)/8, $513  // bitrev[258] = 513
DATA bitrev_size1024_radix4<>+0x818(SB)/8, $769  // bitrev[259] = 769
DATA bitrev_size1024_radix4<>+0x820(SB)/8, $65   // bitrev[260] = 65
DATA bitrev_size1024_radix4<>+0x828(SB)/8, $321  // bitrev[261] = 321
DATA bitrev_size1024_radix4<>+0x830(SB)/8, $577  // bitrev[262] = 577
DATA bitrev_size1024_radix4<>+0x838(SB)/8, $833  // bitrev[263] = 833
DATA bitrev_size1024_radix4<>+0x840(SB)/8, $129  // bitrev[264] = 129
DATA bitrev_size1024_radix4<>+0x848(SB)/8, $385  // bitrev[265] = 385
DATA bitrev_size1024_radix4<>+0x850(SB)/8, $641  // bitrev[266] = 641
DATA bitrev_size1024_radix4<>+0x858(SB)/8, $897  // bitrev[267] = 897
DATA bitrev_size1024_radix4<>+0x860(SB)/8, $193  // bitrev[268] = 193
DATA bitrev_size1024_radix4<>+0x868(SB)/8, $449  // bitrev[269] = 449
DATA bitrev_size1024_radix4<>+0x870(SB)/8, $705  // bitrev[270] = 705
DATA bitrev_size1024_radix4<>+0x878(SB)/8, $961  // bitrev[271] = 961
DATA bitrev_size1024_radix4<>+0x880(SB)/8, $17   // bitrev[272] = 17
DATA bitrev_size1024_radix4<>+0x888(SB)/8, $273  // bitrev[273] = 273
DATA bitrev_size1024_radix4<>+0x890(SB)/8, $529  // bitrev[274] = 529
DATA bitrev_size1024_radix4<>+0x898(SB)/8, $785  // bitrev[275] = 785
DATA bitrev_size1024_radix4<>+0x8a0(SB)/8, $81   // bitrev[276] = 81
DATA bitrev_size1024_radix4<>+0x8a8(SB)/8, $337  // bitrev[277] = 337
DATA bitrev_size1024_radix4<>+0x8b0(SB)/8, $593  // bitrev[278] = 593
DATA bitrev_size1024_radix4<>+0x8b8(SB)/8, $849  // bitrev[279] = 849
DATA bitrev_size1024_radix4<>+0x8c0(SB)/8, $145  // bitrev[280] = 145
DATA bitrev_size1024_radix4<>+0x8c8(SB)/8, $401  // bitrev[281] = 401
DATA bitrev_size1024_radix4<>+0x8d0(SB)/8, $657  // bitrev[282] = 657
DATA bitrev_size1024_radix4<>+0x8d8(SB)/8, $913  // bitrev[283] = 913
DATA bitrev_size1024_radix4<>+0x8e0(SB)/8, $209  // bitrev[284] = 209
DATA bitrev_size1024_radix4<>+0x8e8(SB)/8, $465  // bitrev[285] = 465
DATA bitrev_size1024_radix4<>+0x8f0(SB)/8, $721  // bitrev[286] = 721
DATA bitrev_size1024_radix4<>+0x8f8(SB)/8, $977  // bitrev[287] = 977
DATA bitrev_size1024_radix4<>+0x900(SB)/8, $33   // bitrev[288] = 33
DATA bitrev_size1024_radix4<>+0x908(SB)/8, $289  // bitrev[289] = 289
DATA bitrev_size1024_radix4<>+0x910(SB)/8, $545  // bitrev[290] = 545
DATA bitrev_size1024_radix4<>+0x918(SB)/8, $801  // bitrev[291] = 801
DATA bitrev_size1024_radix4<>+0x920(SB)/8, $97   // bitrev[292] = 97
DATA bitrev_size1024_radix4<>+0x928(SB)/8, $353  // bitrev[293] = 353
DATA bitrev_size1024_radix4<>+0x930(SB)/8, $609  // bitrev[294] = 609
DATA bitrev_size1024_radix4<>+0x938(SB)/8, $865  // bitrev[295] = 865
DATA bitrev_size1024_radix4<>+0x940(SB)/8, $161  // bitrev[296] = 161
DATA bitrev_size1024_radix4<>+0x948(SB)/8, $417  // bitrev[297] = 417
DATA bitrev_size1024_radix4<>+0x950(SB)/8, $673  // bitrev[298] = 673
DATA bitrev_size1024_radix4<>+0x958(SB)/8, $929  // bitrev[299] = 929
DATA bitrev_size1024_radix4<>+0x960(SB)/8, $225  // bitrev[300] = 225
DATA bitrev_size1024_radix4<>+0x968(SB)/8, $481  // bitrev[301] = 481
DATA bitrev_size1024_radix4<>+0x970(SB)/8, $737  // bitrev[302] = 737
DATA bitrev_size1024_radix4<>+0x978(SB)/8, $993  // bitrev[303] = 993
DATA bitrev_size1024_radix4<>+0x980(SB)/8, $49   // bitrev[304] = 49
DATA bitrev_size1024_radix4<>+0x988(SB)/8, $305  // bitrev[305] = 305
DATA bitrev_size1024_radix4<>+0x990(SB)/8, $561  // bitrev[306] = 561
DATA bitrev_size1024_radix4<>+0x998(SB)/8, $817  // bitrev[307] = 817
DATA bitrev_size1024_radix4<>+0x9a0(SB)/8, $113  // bitrev[308] = 113
DATA bitrev_size1024_radix4<>+0x9a8(SB)/8, $369  // bitrev[309] = 369
DATA bitrev_size1024_radix4<>+0x9b0(SB)/8, $625  // bitrev[310] = 625
DATA bitrev_size1024_radix4<>+0x9b8(SB)/8, $881  // bitrev[311] = 881
DATA bitrev_size1024_radix4<>+0x9c0(SB)/8, $177  // bitrev[312] = 177
DATA bitrev_size1024_radix4<>+0x9c8(SB)/8, $433  // bitrev[313] = 433
DATA bitrev_size1024_radix4<>+0x9d0(SB)/8, $689  // bitrev[314] = 689
DATA bitrev_size1024_radix4<>+0x9d8(SB)/8, $945  // bitrev[315] = 945
DATA bitrev_size1024_radix4<>+0x9e0(SB)/8, $241  // bitrev[316] = 241
DATA bitrev_size1024_radix4<>+0x9e8(SB)/8, $497  // bitrev[317] = 497
DATA bitrev_size1024_radix4<>+0x9f0(SB)/8, $753  // bitrev[318] = 753
DATA bitrev_size1024_radix4<>+0x9f8(SB)/8, $1009 // bitrev[319] = 1009
DATA bitrev_size1024_radix4<>+0xa00(SB)/8, $5    // bitrev[320] = 5
DATA bitrev_size1024_radix4<>+0xa08(SB)/8, $261  // bitrev[321] = 261
DATA bitrev_size1024_radix4<>+0xa10(SB)/8, $517  // bitrev[322] = 517
DATA bitrev_size1024_radix4<>+0xa18(SB)/8, $773  // bitrev[323] = 773
DATA bitrev_size1024_radix4<>+0xa20(SB)/8, $69   // bitrev[324] = 69
DATA bitrev_size1024_radix4<>+0xa28(SB)/8, $325  // bitrev[325] = 325
DATA bitrev_size1024_radix4<>+0xa30(SB)/8, $581  // bitrev[326] = 581
DATA bitrev_size1024_radix4<>+0xa38(SB)/8, $837  // bitrev[327] = 837
DATA bitrev_size1024_radix4<>+0xa40(SB)/8, $133  // bitrev[328] = 133
DATA bitrev_size1024_radix4<>+0xa48(SB)/8, $389  // bitrev[329] = 389
DATA bitrev_size1024_radix4<>+0xa50(SB)/8, $645  // bitrev[330] = 645
DATA bitrev_size1024_radix4<>+0xa58(SB)/8, $901  // bitrev[331] = 901
DATA bitrev_size1024_radix4<>+0xa60(SB)/8, $197  // bitrev[332] = 197
DATA bitrev_size1024_radix4<>+0xa68(SB)/8, $453  // bitrev[333] = 453
DATA bitrev_size1024_radix4<>+0xa70(SB)/8, $709  // bitrev[334] = 709
DATA bitrev_size1024_radix4<>+0xa78(SB)/8, $965  // bitrev[335] = 965
DATA bitrev_size1024_radix4<>+0xa80(SB)/8, $21   // bitrev[336] = 21
DATA bitrev_size1024_radix4<>+0xa88(SB)/8, $277  // bitrev[337] = 277
DATA bitrev_size1024_radix4<>+0xa90(SB)/8, $533  // bitrev[338] = 533
DATA bitrev_size1024_radix4<>+0xa98(SB)/8, $789  // bitrev[339] = 789
DATA bitrev_size1024_radix4<>+0xaa0(SB)/8, $85   // bitrev[340] = 85
DATA bitrev_size1024_radix4<>+0xaa8(SB)/8, $341  // bitrev[341] = 341
DATA bitrev_size1024_radix4<>+0xab0(SB)/8, $597  // bitrev[342] = 597
DATA bitrev_size1024_radix4<>+0xab8(SB)/8, $853  // bitrev[343] = 853
DATA bitrev_size1024_radix4<>+0xac0(SB)/8, $149  // bitrev[344] = 149
DATA bitrev_size1024_radix4<>+0xac8(SB)/8, $405  // bitrev[345] = 405
DATA bitrev_size1024_radix4<>+0xad0(SB)/8, $661  // bitrev[346] = 661
DATA bitrev_size1024_radix4<>+0xad8(SB)/8, $917  // bitrev[347] = 917
DATA bitrev_size1024_radix4<>+0xae0(SB)/8, $213  // bitrev[348] = 213
DATA bitrev_size1024_radix4<>+0xae8(SB)/8, $469  // bitrev[349] = 469
DATA bitrev_size1024_radix4<>+0xaf0(SB)/8, $725  // bitrev[350] = 725
DATA bitrev_size1024_radix4<>+0xaf8(SB)/8, $981  // bitrev[351] = 981
DATA bitrev_size1024_radix4<>+0xb00(SB)/8, $37   // bitrev[352] = 37
DATA bitrev_size1024_radix4<>+0xb08(SB)/8, $293  // bitrev[353] = 293
DATA bitrev_size1024_radix4<>+0xb10(SB)/8, $549  // bitrev[354] = 549
DATA bitrev_size1024_radix4<>+0xb18(SB)/8, $805  // bitrev[355] = 805
DATA bitrev_size1024_radix4<>+0xb20(SB)/8, $101  // bitrev[356] = 101
DATA bitrev_size1024_radix4<>+0xb28(SB)/8, $357  // bitrev[357] = 357
DATA bitrev_size1024_radix4<>+0xb30(SB)/8, $613  // bitrev[358] = 613
DATA bitrev_size1024_radix4<>+0xb38(SB)/8, $869  // bitrev[359] = 869
DATA bitrev_size1024_radix4<>+0xb40(SB)/8, $165  // bitrev[360] = 165
DATA bitrev_size1024_radix4<>+0xb48(SB)/8, $421  // bitrev[361] = 421
DATA bitrev_size1024_radix4<>+0xb50(SB)/8, $677  // bitrev[362] = 677
DATA bitrev_size1024_radix4<>+0xb58(SB)/8, $933  // bitrev[363] = 933
DATA bitrev_size1024_radix4<>+0xb60(SB)/8, $229  // bitrev[364] = 229
DATA bitrev_size1024_radix4<>+0xb68(SB)/8, $485  // bitrev[365] = 485
DATA bitrev_size1024_radix4<>+0xb70(SB)/8, $741  // bitrev[366] = 741
DATA bitrev_size1024_radix4<>+0xb78(SB)/8, $997  // bitrev[367] = 997
DATA bitrev_size1024_radix4<>+0xb80(SB)/8, $53   // bitrev[368] = 53
DATA bitrev_size1024_radix4<>+0xb88(SB)/8, $309  // bitrev[369] = 309
DATA bitrev_size1024_radix4<>+0xb90(SB)/8, $565  // bitrev[370] = 565
DATA bitrev_size1024_radix4<>+0xb98(SB)/8, $821  // bitrev[371] = 821
DATA bitrev_size1024_radix4<>+0xba0(SB)/8, $117  // bitrev[372] = 117
DATA bitrev_size1024_radix4<>+0xba8(SB)/8, $373  // bitrev[373] = 373
DATA bitrev_size1024_radix4<>+0xbb0(SB)/8, $629  // bitrev[374] = 629
DATA bitrev_size1024_radix4<>+0xbb8(SB)/8, $885  // bitrev[375] = 885
DATA bitrev_size1024_radix4<>+0xbc0(SB)/8, $181  // bitrev[376] = 181
DATA bitrev_size1024_radix4<>+0xbc8(SB)/8, $437  // bitrev[377] = 437
DATA bitrev_size1024_radix4<>+0xbd0(SB)/8, $693  // bitrev[378] = 693
DATA bitrev_size1024_radix4<>+0xbd8(SB)/8, $949  // bitrev[379] = 949
DATA bitrev_size1024_radix4<>+0xbe0(SB)/8, $245  // bitrev[380] = 245
DATA bitrev_size1024_radix4<>+0xbe8(SB)/8, $501  // bitrev[381] = 501
DATA bitrev_size1024_radix4<>+0xbf0(SB)/8, $757  // bitrev[382] = 757
DATA bitrev_size1024_radix4<>+0xbf8(SB)/8, $1013 // bitrev[383] = 1013
DATA bitrev_size1024_radix4<>+0xc00(SB)/8, $9    // bitrev[384] = 9
DATA bitrev_size1024_radix4<>+0xc08(SB)/8, $265  // bitrev[385] = 265
DATA bitrev_size1024_radix4<>+0xc10(SB)/8, $521  // bitrev[386] = 521
DATA bitrev_size1024_radix4<>+0xc18(SB)/8, $777  // bitrev[387] = 777
DATA bitrev_size1024_radix4<>+0xc20(SB)/8, $73   // bitrev[388] = 73
DATA bitrev_size1024_radix4<>+0xc28(SB)/8, $329  // bitrev[389] = 329
DATA bitrev_size1024_radix4<>+0xc30(SB)/8, $585  // bitrev[390] = 585
DATA bitrev_size1024_radix4<>+0xc38(SB)/8, $841  // bitrev[391] = 841
DATA bitrev_size1024_radix4<>+0xc40(SB)/8, $137  // bitrev[392] = 137
DATA bitrev_size1024_radix4<>+0xc48(SB)/8, $393  // bitrev[393] = 393
DATA bitrev_size1024_radix4<>+0xc50(SB)/8, $649  // bitrev[394] = 649
DATA bitrev_size1024_radix4<>+0xc58(SB)/8, $905  // bitrev[395] = 905
DATA bitrev_size1024_radix4<>+0xc60(SB)/8, $201  // bitrev[396] = 201
DATA bitrev_size1024_radix4<>+0xc68(SB)/8, $457  // bitrev[397] = 457
DATA bitrev_size1024_radix4<>+0xc70(SB)/8, $713  // bitrev[398] = 713
DATA bitrev_size1024_radix4<>+0xc78(SB)/8, $969  // bitrev[399] = 969
DATA bitrev_size1024_radix4<>+0xc80(SB)/8, $25   // bitrev[400] = 25
DATA bitrev_size1024_radix4<>+0xc88(SB)/8, $281  // bitrev[401] = 281
DATA bitrev_size1024_radix4<>+0xc90(SB)/8, $537  // bitrev[402] = 537
DATA bitrev_size1024_radix4<>+0xc98(SB)/8, $793  // bitrev[403] = 793
DATA bitrev_size1024_radix4<>+0xca0(SB)/8, $89   // bitrev[404] = 89
DATA bitrev_size1024_radix4<>+0xca8(SB)/8, $345  // bitrev[405] = 345
DATA bitrev_size1024_radix4<>+0xcb0(SB)/8, $601  // bitrev[406] = 601
DATA bitrev_size1024_radix4<>+0xcb8(SB)/8, $857  // bitrev[407] = 857
DATA bitrev_size1024_radix4<>+0xcc0(SB)/8, $153  // bitrev[408] = 153
DATA bitrev_size1024_radix4<>+0xcc8(SB)/8, $409  // bitrev[409] = 409
DATA bitrev_size1024_radix4<>+0xcd0(SB)/8, $665  // bitrev[410] = 665
DATA bitrev_size1024_radix4<>+0xcd8(SB)/8, $921  // bitrev[411] = 921
DATA bitrev_size1024_radix4<>+0xce0(SB)/8, $217  // bitrev[412] = 217
DATA bitrev_size1024_radix4<>+0xce8(SB)/8, $473  // bitrev[413] = 473
DATA bitrev_size1024_radix4<>+0xcf0(SB)/8, $729  // bitrev[414] = 729
DATA bitrev_size1024_radix4<>+0xcf8(SB)/8, $985  // bitrev[415] = 985
DATA bitrev_size1024_radix4<>+0xd00(SB)/8, $41   // bitrev[416] = 41
DATA bitrev_size1024_radix4<>+0xd08(SB)/8, $297  // bitrev[417] = 297
DATA bitrev_size1024_radix4<>+0xd10(SB)/8, $553  // bitrev[418] = 553
DATA bitrev_size1024_radix4<>+0xd18(SB)/8, $809  // bitrev[419] = 809
DATA bitrev_size1024_radix4<>+0xd20(SB)/8, $105  // bitrev[420] = 105
DATA bitrev_size1024_radix4<>+0xd28(SB)/8, $361  // bitrev[421] = 361
DATA bitrev_size1024_radix4<>+0xd30(SB)/8, $617  // bitrev[422] = 617
DATA bitrev_size1024_radix4<>+0xd38(SB)/8, $873  // bitrev[423] = 873
DATA bitrev_size1024_radix4<>+0xd40(SB)/8, $169  // bitrev[424] = 169
DATA bitrev_size1024_radix4<>+0xd48(SB)/8, $425  // bitrev[425] = 425
DATA bitrev_size1024_radix4<>+0xd50(SB)/8, $681  // bitrev[426] = 681
DATA bitrev_size1024_radix4<>+0xd58(SB)/8, $937  // bitrev[427] = 937
DATA bitrev_size1024_radix4<>+0xd60(SB)/8, $233  // bitrev[428] = 233
DATA bitrev_size1024_radix4<>+0xd68(SB)/8, $489  // bitrev[429] = 489
DATA bitrev_size1024_radix4<>+0xd70(SB)/8, $745  // bitrev[430] = 745
DATA bitrev_size1024_radix4<>+0xd78(SB)/8, $1001 // bitrev[431] = 1001
DATA bitrev_size1024_radix4<>+0xd80(SB)/8, $57   // bitrev[432] = 57
DATA bitrev_size1024_radix4<>+0xd88(SB)/8, $313  // bitrev[433] = 313
DATA bitrev_size1024_radix4<>+0xd90(SB)/8, $569  // bitrev[434] = 569
DATA bitrev_size1024_radix4<>+0xd98(SB)/8, $825  // bitrev[435] = 825
DATA bitrev_size1024_radix4<>+0xda0(SB)/8, $121  // bitrev[436] = 121
DATA bitrev_size1024_radix4<>+0xda8(SB)/8, $377  // bitrev[437] = 377
DATA bitrev_size1024_radix4<>+0xdb0(SB)/8, $633  // bitrev[438] = 633
DATA bitrev_size1024_radix4<>+0xdb8(SB)/8, $889  // bitrev[439] = 889
DATA bitrev_size1024_radix4<>+0xdc0(SB)/8, $185  // bitrev[440] = 185
DATA bitrev_size1024_radix4<>+0xdc8(SB)/8, $441  // bitrev[441] = 441
DATA bitrev_size1024_radix4<>+0xdd0(SB)/8, $697  // bitrev[442] = 697
DATA bitrev_size1024_radix4<>+0xdd8(SB)/8, $953  // bitrev[443] = 953
DATA bitrev_size1024_radix4<>+0xde0(SB)/8, $249  // bitrev[444] = 249
DATA bitrev_size1024_radix4<>+0xde8(SB)/8, $505  // bitrev[445] = 505
DATA bitrev_size1024_radix4<>+0xdf0(SB)/8, $761  // bitrev[446] = 761
DATA bitrev_size1024_radix4<>+0xdf8(SB)/8, $1017 // bitrev[447] = 1017
DATA bitrev_size1024_radix4<>+0xe00(SB)/8, $13   // bitrev[448] = 13
DATA bitrev_size1024_radix4<>+0xe08(SB)/8, $269  // bitrev[449] = 269
DATA bitrev_size1024_radix4<>+0xe10(SB)/8, $525  // bitrev[450] = 525
DATA bitrev_size1024_radix4<>+0xe18(SB)/8, $781  // bitrev[451] = 781
DATA bitrev_size1024_radix4<>+0xe20(SB)/8, $77   // bitrev[452] = 77
DATA bitrev_size1024_radix4<>+0xe28(SB)/8, $333  // bitrev[453] = 333
DATA bitrev_size1024_radix4<>+0xe30(SB)/8, $589  // bitrev[454] = 589
DATA bitrev_size1024_radix4<>+0xe38(SB)/8, $845  // bitrev[455] = 845
DATA bitrev_size1024_radix4<>+0xe40(SB)/8, $141  // bitrev[456] = 141
DATA bitrev_size1024_radix4<>+0xe48(SB)/8, $397  // bitrev[457] = 397
DATA bitrev_size1024_radix4<>+0xe50(SB)/8, $653  // bitrev[458] = 653
DATA bitrev_size1024_radix4<>+0xe58(SB)/8, $909  // bitrev[459] = 909
DATA bitrev_size1024_radix4<>+0xe60(SB)/8, $205  // bitrev[460] = 205
DATA bitrev_size1024_radix4<>+0xe68(SB)/8, $461  // bitrev[461] = 461
DATA bitrev_size1024_radix4<>+0xe70(SB)/8, $717  // bitrev[462] = 717
DATA bitrev_size1024_radix4<>+0xe78(SB)/8, $973  // bitrev[463] = 973
DATA bitrev_size1024_radix4<>+0xe80(SB)/8, $29   // bitrev[464] = 29
DATA bitrev_size1024_radix4<>+0xe88(SB)/8, $285  // bitrev[465] = 285
DATA bitrev_size1024_radix4<>+0xe90(SB)/8, $541  // bitrev[466] = 541
DATA bitrev_size1024_radix4<>+0xe98(SB)/8, $797  // bitrev[467] = 797
DATA bitrev_size1024_radix4<>+0xea0(SB)/8, $93   // bitrev[468] = 93
DATA bitrev_size1024_radix4<>+0xea8(SB)/8, $349  // bitrev[469] = 349
DATA bitrev_size1024_radix4<>+0xeb0(SB)/8, $605  // bitrev[470] = 605
DATA bitrev_size1024_radix4<>+0xeb8(SB)/8, $861  // bitrev[471] = 861
DATA bitrev_size1024_radix4<>+0xec0(SB)/8, $157  // bitrev[472] = 157
DATA bitrev_size1024_radix4<>+0xec8(SB)/8, $413  // bitrev[473] = 413
DATA bitrev_size1024_radix4<>+0xed0(SB)/8, $669  // bitrev[474] = 669
DATA bitrev_size1024_radix4<>+0xed8(SB)/8, $925  // bitrev[475] = 925
DATA bitrev_size1024_radix4<>+0xee0(SB)/8, $221  // bitrev[476] = 221
DATA bitrev_size1024_radix4<>+0xee8(SB)/8, $477  // bitrev[477] = 477
DATA bitrev_size1024_radix4<>+0xef0(SB)/8, $733  // bitrev[478] = 733
DATA bitrev_size1024_radix4<>+0xef8(SB)/8, $989  // bitrev[479] = 989
DATA bitrev_size1024_radix4<>+0xf00(SB)/8, $45   // bitrev[480] = 45
DATA bitrev_size1024_radix4<>+0xf08(SB)/8, $301  // bitrev[481] = 301
DATA bitrev_size1024_radix4<>+0xf10(SB)/8, $557  // bitrev[482] = 557
DATA bitrev_size1024_radix4<>+0xf18(SB)/8, $813  // bitrev[483] = 813
DATA bitrev_size1024_radix4<>+0xf20(SB)/8, $109  // bitrev[484] = 109
DATA bitrev_size1024_radix4<>+0xf28(SB)/8, $365  // bitrev[485] = 365
DATA bitrev_size1024_radix4<>+0xf30(SB)/8, $621  // bitrev[486] = 621
DATA bitrev_size1024_radix4<>+0xf38(SB)/8, $877  // bitrev[487] = 877
DATA bitrev_size1024_radix4<>+0xf40(SB)/8, $173  // bitrev[488] = 173
DATA bitrev_size1024_radix4<>+0xf48(SB)/8, $429  // bitrev[489] = 429
DATA bitrev_size1024_radix4<>+0xf50(SB)/8, $685  // bitrev[490] = 685
DATA bitrev_size1024_radix4<>+0xf58(SB)/8, $941  // bitrev[491] = 941
DATA bitrev_size1024_radix4<>+0xf60(SB)/8, $237  // bitrev[492] = 237
DATA bitrev_size1024_radix4<>+0xf68(SB)/8, $493  // bitrev[493] = 493
DATA bitrev_size1024_radix4<>+0xf70(SB)/8, $749  // bitrev[494] = 749
DATA bitrev_size1024_radix4<>+0xf78(SB)/8, $1005 // bitrev[495] = 1005
DATA bitrev_size1024_radix4<>+0xf80(SB)/8, $61   // bitrev[496] = 61
DATA bitrev_size1024_radix4<>+0xf88(SB)/8, $317  // bitrev[497] = 317
DATA bitrev_size1024_radix4<>+0xf90(SB)/8, $573  // bitrev[498] = 573
DATA bitrev_size1024_radix4<>+0xf98(SB)/8, $829  // bitrev[499] = 829
DATA bitrev_size1024_radix4<>+0xfa0(SB)/8, $125  // bitrev[500] = 125
DATA bitrev_size1024_radix4<>+0xfa8(SB)/8, $381  // bitrev[501] = 381
DATA bitrev_size1024_radix4<>+0xfb0(SB)/8, $637  // bitrev[502] = 637
DATA bitrev_size1024_radix4<>+0xfb8(SB)/8, $893  // bitrev[503] = 893
DATA bitrev_size1024_radix4<>+0xfc0(SB)/8, $189  // bitrev[504] = 189
DATA bitrev_size1024_radix4<>+0xfc8(SB)/8, $445  // bitrev[505] = 445
DATA bitrev_size1024_radix4<>+0xfd0(SB)/8, $701  // bitrev[506] = 701
DATA bitrev_size1024_radix4<>+0xfd8(SB)/8, $957  // bitrev[507] = 957
DATA bitrev_size1024_radix4<>+0xfe0(SB)/8, $253  // bitrev[508] = 253
DATA bitrev_size1024_radix4<>+0xfe8(SB)/8, $509  // bitrev[509] = 509
DATA bitrev_size1024_radix4<>+0xff0(SB)/8, $765  // bitrev[510] = 765
DATA bitrev_size1024_radix4<>+0xff8(SB)/8, $1021 // bitrev[511] = 1021
DATA bitrev_size1024_radix4<>+0x1000(SB)/8, $2    // bitrev[512] = 2
DATA bitrev_size1024_radix4<>+0x1008(SB)/8, $258  // bitrev[513] = 258
DATA bitrev_size1024_radix4<>+0x1010(SB)/8, $514  // bitrev[514] = 514
DATA bitrev_size1024_radix4<>+0x1018(SB)/8, $770  // bitrev[515] = 770
DATA bitrev_size1024_radix4<>+0x1020(SB)/8, $66   // bitrev[516] = 66
DATA bitrev_size1024_radix4<>+0x1028(SB)/8, $322  // bitrev[517] = 322
DATA bitrev_size1024_radix4<>+0x1030(SB)/8, $578  // bitrev[518] = 578
DATA bitrev_size1024_radix4<>+0x1038(SB)/8, $834  // bitrev[519] = 834
DATA bitrev_size1024_radix4<>+0x1040(SB)/8, $130  // bitrev[520] = 130
DATA bitrev_size1024_radix4<>+0x1048(SB)/8, $386  // bitrev[521] = 386
DATA bitrev_size1024_radix4<>+0x1050(SB)/8, $642  // bitrev[522] = 642
DATA bitrev_size1024_radix4<>+0x1058(SB)/8, $898  // bitrev[523] = 898
DATA bitrev_size1024_radix4<>+0x1060(SB)/8, $194  // bitrev[524] = 194
DATA bitrev_size1024_radix4<>+0x1068(SB)/8, $450  // bitrev[525] = 450
DATA bitrev_size1024_radix4<>+0x1070(SB)/8, $706  // bitrev[526] = 706
DATA bitrev_size1024_radix4<>+0x1078(SB)/8, $962  // bitrev[527] = 962
DATA bitrev_size1024_radix4<>+0x1080(SB)/8, $18   // bitrev[528] = 18
DATA bitrev_size1024_radix4<>+0x1088(SB)/8, $274  // bitrev[529] = 274
DATA bitrev_size1024_radix4<>+0x1090(SB)/8, $530  // bitrev[530] = 530
DATA bitrev_size1024_radix4<>+0x1098(SB)/8, $786  // bitrev[531] = 786
DATA bitrev_size1024_radix4<>+0x10a0(SB)/8, $82   // bitrev[532] = 82
DATA bitrev_size1024_radix4<>+0x10a8(SB)/8, $338  // bitrev[533] = 338
DATA bitrev_size1024_radix4<>+0x10b0(SB)/8, $594  // bitrev[534] = 594
DATA bitrev_size1024_radix4<>+0x10b8(SB)/8, $850  // bitrev[535] = 850
DATA bitrev_size1024_radix4<>+0x10c0(SB)/8, $146  // bitrev[536] = 146
DATA bitrev_size1024_radix4<>+0x10c8(SB)/8, $402  // bitrev[537] = 402
DATA bitrev_size1024_radix4<>+0x10d0(SB)/8, $658  // bitrev[538] = 658
DATA bitrev_size1024_radix4<>+0x10d8(SB)/8, $914  // bitrev[539] = 914
DATA bitrev_size1024_radix4<>+0x10e0(SB)/8, $210  // bitrev[540] = 210
DATA bitrev_size1024_radix4<>+0x10e8(SB)/8, $466  // bitrev[541] = 466
DATA bitrev_size1024_radix4<>+0x10f0(SB)/8, $722  // bitrev[542] = 722
DATA bitrev_size1024_radix4<>+0x10f8(SB)/8, $978  // bitrev[543] = 978
DATA bitrev_size1024_radix4<>+0x1100(SB)/8, $34   // bitrev[544] = 34
DATA bitrev_size1024_radix4<>+0x1108(SB)/8, $290  // bitrev[545] = 290
DATA bitrev_size1024_radix4<>+0x1110(SB)/8, $546  // bitrev[546] = 546
DATA bitrev_size1024_radix4<>+0x1118(SB)/8, $802  // bitrev[547] = 802
DATA bitrev_size1024_radix4<>+0x1120(SB)/8, $98   // bitrev[548] = 98
DATA bitrev_size1024_radix4<>+0x1128(SB)/8, $354  // bitrev[549] = 354
DATA bitrev_size1024_radix4<>+0x1130(SB)/8, $610  // bitrev[550] = 610
DATA bitrev_size1024_radix4<>+0x1138(SB)/8, $866  // bitrev[551] = 866
DATA bitrev_size1024_radix4<>+0x1140(SB)/8, $162  // bitrev[552] = 162
DATA bitrev_size1024_radix4<>+0x1148(SB)/8, $418  // bitrev[553] = 418
DATA bitrev_size1024_radix4<>+0x1150(SB)/8, $674  // bitrev[554] = 674
DATA bitrev_size1024_radix4<>+0x1158(SB)/8, $930  // bitrev[555] = 930
DATA bitrev_size1024_radix4<>+0x1160(SB)/8, $226  // bitrev[556] = 226
DATA bitrev_size1024_radix4<>+0x1168(SB)/8, $482  // bitrev[557] = 482
DATA bitrev_size1024_radix4<>+0x1170(SB)/8, $738  // bitrev[558] = 738
DATA bitrev_size1024_radix4<>+0x1178(SB)/8, $994  // bitrev[559] = 994
DATA bitrev_size1024_radix4<>+0x1180(SB)/8, $50   // bitrev[560] = 50
DATA bitrev_size1024_radix4<>+0x1188(SB)/8, $306  // bitrev[561] = 306
DATA bitrev_size1024_radix4<>+0x1190(SB)/8, $562  // bitrev[562] = 562
DATA bitrev_size1024_radix4<>+0x1198(SB)/8, $818  // bitrev[563] = 818
DATA bitrev_size1024_radix4<>+0x11a0(SB)/8, $114  // bitrev[564] = 114
DATA bitrev_size1024_radix4<>+0x11a8(SB)/8, $370  // bitrev[565] = 370
DATA bitrev_size1024_radix4<>+0x11b0(SB)/8, $626  // bitrev[566] = 626
DATA bitrev_size1024_radix4<>+0x11b8(SB)/8, $882  // bitrev[567] = 882
DATA bitrev_size1024_radix4<>+0x11c0(SB)/8, $178  // bitrev[568] = 178
DATA bitrev_size1024_radix4<>+0x11c8(SB)/8, $434  // bitrev[569] = 434
DATA bitrev_size1024_radix4<>+0x11d0(SB)/8, $690  // bitrev[570] = 690
DATA bitrev_size1024_radix4<>+0x11d8(SB)/8, $946  // bitrev[571] = 946
DATA bitrev_size1024_radix4<>+0x11e0(SB)/8, $242  // bitrev[572] = 242
DATA bitrev_size1024_radix4<>+0x11e8(SB)/8, $498  // bitrev[573] = 498
DATA bitrev_size1024_radix4<>+0x11f0(SB)/8, $754  // bitrev[574] = 754
DATA bitrev_size1024_radix4<>+0x11f8(SB)/8, $1010 // bitrev[575] = 1010
DATA bitrev_size1024_radix4<>+0x1200(SB)/8, $6    // bitrev[576] = 6
DATA bitrev_size1024_radix4<>+0x1208(SB)/8, $262  // bitrev[577] = 262
DATA bitrev_size1024_radix4<>+0x1210(SB)/8, $518  // bitrev[578] = 518
DATA bitrev_size1024_radix4<>+0x1218(SB)/8, $774  // bitrev[579] = 774
DATA bitrev_size1024_radix4<>+0x1220(SB)/8, $70   // bitrev[580] = 70
DATA bitrev_size1024_radix4<>+0x1228(SB)/8, $326  // bitrev[581] = 326
DATA bitrev_size1024_radix4<>+0x1230(SB)/8, $582  // bitrev[582] = 582
DATA bitrev_size1024_radix4<>+0x1238(SB)/8, $838  // bitrev[583] = 838
DATA bitrev_size1024_radix4<>+0x1240(SB)/8, $134  // bitrev[584] = 134
DATA bitrev_size1024_radix4<>+0x1248(SB)/8, $390  // bitrev[585] = 390
DATA bitrev_size1024_radix4<>+0x1250(SB)/8, $646  // bitrev[586] = 646
DATA bitrev_size1024_radix4<>+0x1258(SB)/8, $902  // bitrev[587] = 902
DATA bitrev_size1024_radix4<>+0x1260(SB)/8, $198  // bitrev[588] = 198
DATA bitrev_size1024_radix4<>+0x1268(SB)/8, $454  // bitrev[589] = 454
DATA bitrev_size1024_radix4<>+0x1270(SB)/8, $710  // bitrev[590] = 710
DATA bitrev_size1024_radix4<>+0x1278(SB)/8, $966  // bitrev[591] = 966
DATA bitrev_size1024_radix4<>+0x1280(SB)/8, $22   // bitrev[592] = 22
DATA bitrev_size1024_radix4<>+0x1288(SB)/8, $278  // bitrev[593] = 278
DATA bitrev_size1024_radix4<>+0x1290(SB)/8, $534  // bitrev[594] = 534
DATA bitrev_size1024_radix4<>+0x1298(SB)/8, $790  // bitrev[595] = 790
DATA bitrev_size1024_radix4<>+0x12a0(SB)/8, $86   // bitrev[596] = 86
DATA bitrev_size1024_radix4<>+0x12a8(SB)/8, $342  // bitrev[597] = 342
DATA bitrev_size1024_radix4<>+0x12b0(SB)/8, $598  // bitrev[598] = 598
DATA bitrev_size1024_radix4<>+0x12b8(SB)/8, $854  // bitrev[599] = 854
DATA bitrev_size1024_radix4<>+0x12c0(SB)/8, $150  // bitrev[600] = 150
DATA bitrev_size1024_radix4<>+0x12c8(SB)/8, $406  // bitrev[601] = 406
DATA bitrev_size1024_radix4<>+0x12d0(SB)/8, $662  // bitrev[602] = 662
DATA bitrev_size1024_radix4<>+0x12d8(SB)/8, $918  // bitrev[603] = 918
DATA bitrev_size1024_radix4<>+0x12e0(SB)/8, $214  // bitrev[604] = 214
DATA bitrev_size1024_radix4<>+0x12e8(SB)/8, $470  // bitrev[605] = 470
DATA bitrev_size1024_radix4<>+0x12f0(SB)/8, $726  // bitrev[606] = 726
DATA bitrev_size1024_radix4<>+0x12f8(SB)/8, $982  // bitrev[607] = 982
DATA bitrev_size1024_radix4<>+0x1300(SB)/8, $38   // bitrev[608] = 38
DATA bitrev_size1024_radix4<>+0x1308(SB)/8, $294  // bitrev[609] = 294
DATA bitrev_size1024_radix4<>+0x1310(SB)/8, $550  // bitrev[610] = 550
DATA bitrev_size1024_radix4<>+0x1318(SB)/8, $806  // bitrev[611] = 806
DATA bitrev_size1024_radix4<>+0x1320(SB)/8, $102  // bitrev[612] = 102
DATA bitrev_size1024_radix4<>+0x1328(SB)/8, $358  // bitrev[613] = 358
DATA bitrev_size1024_radix4<>+0x1330(SB)/8, $614  // bitrev[614] = 614
DATA bitrev_size1024_radix4<>+0x1338(SB)/8, $870  // bitrev[615] = 870
DATA bitrev_size1024_radix4<>+0x1340(SB)/8, $166  // bitrev[616] = 166
DATA bitrev_size1024_radix4<>+0x1348(SB)/8, $422  // bitrev[617] = 422
DATA bitrev_size1024_radix4<>+0x1350(SB)/8, $678  // bitrev[618] = 678
DATA bitrev_size1024_radix4<>+0x1358(SB)/8, $934  // bitrev[619] = 934
DATA bitrev_size1024_radix4<>+0x1360(SB)/8, $230  // bitrev[620] = 230
DATA bitrev_size1024_radix4<>+0x1368(SB)/8, $486  // bitrev[621] = 486
DATA bitrev_size1024_radix4<>+0x1370(SB)/8, $742  // bitrev[622] = 742
DATA bitrev_size1024_radix4<>+0x1378(SB)/8, $998  // bitrev[623] = 998
DATA bitrev_size1024_radix4<>+0x1380(SB)/8, $54   // bitrev[624] = 54
DATA bitrev_size1024_radix4<>+0x1388(SB)/8, $310  // bitrev[625] = 310
DATA bitrev_size1024_radix4<>+0x1390(SB)/8, $566  // bitrev[626] = 566
DATA bitrev_size1024_radix4<>+0x1398(SB)/8, $822  // bitrev[627] = 822
DATA bitrev_size1024_radix4<>+0x13a0(SB)/8, $118  // bitrev[628] = 118
DATA bitrev_size1024_radix4<>+0x13a8(SB)/8, $374  // bitrev[629] = 374
DATA bitrev_size1024_radix4<>+0x13b0(SB)/8, $630  // bitrev[630] = 630
DATA bitrev_size1024_radix4<>+0x13b8(SB)/8, $886  // bitrev[631] = 886
DATA bitrev_size1024_radix4<>+0x13c0(SB)/8, $182  // bitrev[632] = 182
DATA bitrev_size1024_radix4<>+0x13c8(SB)/8, $438  // bitrev[633] = 438
DATA bitrev_size1024_radix4<>+0x13d0(SB)/8, $694  // bitrev[634] = 694
DATA bitrev_size1024_radix4<>+0x13d8(SB)/8, $950  // bitrev[635] = 950
DATA bitrev_size1024_radix4<>+0x13e0(SB)/8, $246  // bitrev[636] = 246
DATA bitrev_size1024_radix4<>+0x13e8(SB)/8, $502  // bitrev[637] = 502
DATA bitrev_size1024_radix4<>+0x13f0(SB)/8, $758  // bitrev[638] = 758
DATA bitrev_size1024_radix4<>+0x13f8(SB)/8, $1014 // bitrev[639] = 1014
DATA bitrev_size1024_radix4<>+0x1400(SB)/8, $10   // bitrev[640] = 10
DATA bitrev_size1024_radix4<>+0x1408(SB)/8, $266  // bitrev[641] = 266
DATA bitrev_size1024_radix4<>+0x1410(SB)/8, $522  // bitrev[642] = 522
DATA bitrev_size1024_radix4<>+0x1418(SB)/8, $778  // bitrev[643] = 778
DATA bitrev_size1024_radix4<>+0x1420(SB)/8, $74   // bitrev[644] = 74
DATA bitrev_size1024_radix4<>+0x1428(SB)/8, $330  // bitrev[645] = 330
DATA bitrev_size1024_radix4<>+0x1430(SB)/8, $586  // bitrev[646] = 586
DATA bitrev_size1024_radix4<>+0x1438(SB)/8, $842  // bitrev[647] = 842
DATA bitrev_size1024_radix4<>+0x1440(SB)/8, $138  // bitrev[648] = 138
DATA bitrev_size1024_radix4<>+0x1448(SB)/8, $394  // bitrev[649] = 394
DATA bitrev_size1024_radix4<>+0x1450(SB)/8, $650  // bitrev[650] = 650
DATA bitrev_size1024_radix4<>+0x1458(SB)/8, $906  // bitrev[651] = 906
DATA bitrev_size1024_radix4<>+0x1460(SB)/8, $202  // bitrev[652] = 202
DATA bitrev_size1024_radix4<>+0x1468(SB)/8, $458  // bitrev[653] = 458
DATA bitrev_size1024_radix4<>+0x1470(SB)/8, $714  // bitrev[654] = 714
DATA bitrev_size1024_radix4<>+0x1478(SB)/8, $970  // bitrev[655] = 970
DATA bitrev_size1024_radix4<>+0x1480(SB)/8, $26   // bitrev[656] = 26
DATA bitrev_size1024_radix4<>+0x1488(SB)/8, $282  // bitrev[657] = 282
DATA bitrev_size1024_radix4<>+0x1490(SB)/8, $538  // bitrev[658] = 538
DATA bitrev_size1024_radix4<>+0x1498(SB)/8, $794  // bitrev[659] = 794
DATA bitrev_size1024_radix4<>+0x14a0(SB)/8, $90   // bitrev[660] = 90
DATA bitrev_size1024_radix4<>+0x14a8(SB)/8, $346  // bitrev[661] = 346
DATA bitrev_size1024_radix4<>+0x14b0(SB)/8, $602  // bitrev[662] = 602
DATA bitrev_size1024_radix4<>+0x14b8(SB)/8, $858  // bitrev[663] = 858
DATA bitrev_size1024_radix4<>+0x14c0(SB)/8, $154  // bitrev[664] = 154
DATA bitrev_size1024_radix4<>+0x14c8(SB)/8, $410  // bitrev[665] = 410
DATA bitrev_size1024_radix4<>+0x14d0(SB)/8, $666  // bitrev[666] = 666
DATA bitrev_size1024_radix4<>+0x14d8(SB)/8, $922  // bitrev[667] = 922
DATA bitrev_size1024_radix4<>+0x14e0(SB)/8, $218  // bitrev[668] = 218
DATA bitrev_size1024_radix4<>+0x14e8(SB)/8, $474  // bitrev[669] = 474
DATA bitrev_size1024_radix4<>+0x14f0(SB)/8, $730  // bitrev[670] = 730
DATA bitrev_size1024_radix4<>+0x14f8(SB)/8, $986  // bitrev[671] = 986
DATA bitrev_size1024_radix4<>+0x1500(SB)/8, $42   // bitrev[672] = 42
DATA bitrev_size1024_radix4<>+0x1508(SB)/8, $298  // bitrev[673] = 298
DATA bitrev_size1024_radix4<>+0x1510(SB)/8, $554  // bitrev[674] = 554
DATA bitrev_size1024_radix4<>+0x1518(SB)/8, $810  // bitrev[675] = 810
DATA bitrev_size1024_radix4<>+0x1520(SB)/8, $106  // bitrev[676] = 106
DATA bitrev_size1024_radix4<>+0x1528(SB)/8, $362  // bitrev[677] = 362
DATA bitrev_size1024_radix4<>+0x1530(SB)/8, $618  // bitrev[678] = 618
DATA bitrev_size1024_radix4<>+0x1538(SB)/8, $874  // bitrev[679] = 874
DATA bitrev_size1024_radix4<>+0x1540(SB)/8, $170  // bitrev[680] = 170
DATA bitrev_size1024_radix4<>+0x1548(SB)/8, $426  // bitrev[681] = 426
DATA bitrev_size1024_radix4<>+0x1550(SB)/8, $682  // bitrev[682] = 682
DATA bitrev_size1024_radix4<>+0x1558(SB)/8, $938  // bitrev[683] = 938
DATA bitrev_size1024_radix4<>+0x1560(SB)/8, $234  // bitrev[684] = 234
DATA bitrev_size1024_radix4<>+0x1568(SB)/8, $490  // bitrev[685] = 490
DATA bitrev_size1024_radix4<>+0x1570(SB)/8, $746  // bitrev[686] = 746
DATA bitrev_size1024_radix4<>+0x1578(SB)/8, $1002 // bitrev[687] = 1002
DATA bitrev_size1024_radix4<>+0x1580(SB)/8, $58   // bitrev[688] = 58
DATA bitrev_size1024_radix4<>+0x1588(SB)/8, $314  // bitrev[689] = 314
DATA bitrev_size1024_radix4<>+0x1590(SB)/8, $570  // bitrev[690] = 570
DATA bitrev_size1024_radix4<>+0x1598(SB)/8, $826  // bitrev[691] = 826
DATA bitrev_size1024_radix4<>+0x15a0(SB)/8, $122  // bitrev[692] = 122
DATA bitrev_size1024_radix4<>+0x15a8(SB)/8, $378  // bitrev[693] = 378
DATA bitrev_size1024_radix4<>+0x15b0(SB)/8, $634  // bitrev[694] = 634
DATA bitrev_size1024_radix4<>+0x15b8(SB)/8, $890  // bitrev[695] = 890
DATA bitrev_size1024_radix4<>+0x15c0(SB)/8, $186  // bitrev[696] = 186
DATA bitrev_size1024_radix4<>+0x15c8(SB)/8, $442  // bitrev[697] = 442
DATA bitrev_size1024_radix4<>+0x15d0(SB)/8, $698  // bitrev[698] = 698
DATA bitrev_size1024_radix4<>+0x15d8(SB)/8, $954  // bitrev[699] = 954
DATA bitrev_size1024_radix4<>+0x15e0(SB)/8, $250  // bitrev[700] = 250
DATA bitrev_size1024_radix4<>+0x15e8(SB)/8, $506  // bitrev[701] = 506
DATA bitrev_size1024_radix4<>+0x15f0(SB)/8, $762  // bitrev[702] = 762
DATA bitrev_size1024_radix4<>+0x15f8(SB)/8, $1018 // bitrev[703] = 1018
DATA bitrev_size1024_radix4<>+0x1600(SB)/8, $14   // bitrev[704] = 14
DATA bitrev_size1024_radix4<>+0x1608(SB)/8, $270  // bitrev[705] = 270
DATA bitrev_size1024_radix4<>+0x1610(SB)/8, $526  // bitrev[706] = 526
DATA bitrev_size1024_radix4<>+0x1618(SB)/8, $782  // bitrev[707] = 782
DATA bitrev_size1024_radix4<>+0x1620(SB)/8, $78   // bitrev[708] = 78
DATA bitrev_size1024_radix4<>+0x1628(SB)/8, $334  // bitrev[709] = 334
DATA bitrev_size1024_radix4<>+0x1630(SB)/8, $590  // bitrev[710] = 590
DATA bitrev_size1024_radix4<>+0x1638(SB)/8, $846  // bitrev[711] = 846
DATA bitrev_size1024_radix4<>+0x1640(SB)/8, $142  // bitrev[712] = 142
DATA bitrev_size1024_radix4<>+0x1648(SB)/8, $398  // bitrev[713] = 398
DATA bitrev_size1024_radix4<>+0x1650(SB)/8, $654  // bitrev[714] = 654
DATA bitrev_size1024_radix4<>+0x1658(SB)/8, $910  // bitrev[715] = 910
DATA bitrev_size1024_radix4<>+0x1660(SB)/8, $206  // bitrev[716] = 206
DATA bitrev_size1024_radix4<>+0x1668(SB)/8, $462  // bitrev[717] = 462
DATA bitrev_size1024_radix4<>+0x1670(SB)/8, $718  // bitrev[718] = 718
DATA bitrev_size1024_radix4<>+0x1678(SB)/8, $974  // bitrev[719] = 974
DATA bitrev_size1024_radix4<>+0x1680(SB)/8, $30   // bitrev[720] = 30
DATA bitrev_size1024_radix4<>+0x1688(SB)/8, $286  // bitrev[721] = 286
DATA bitrev_size1024_radix4<>+0x1690(SB)/8, $542  // bitrev[722] = 542
DATA bitrev_size1024_radix4<>+0x1698(SB)/8, $798  // bitrev[723] = 798
DATA bitrev_size1024_radix4<>+0x16a0(SB)/8, $94   // bitrev[724] = 94
DATA bitrev_size1024_radix4<>+0x16a8(SB)/8, $350  // bitrev[725] = 350
DATA bitrev_size1024_radix4<>+0x16b0(SB)/8, $606  // bitrev[726] = 606
DATA bitrev_size1024_radix4<>+0x16b8(SB)/8, $862  // bitrev[727] = 862
DATA bitrev_size1024_radix4<>+0x16c0(SB)/8, $158  // bitrev[728] = 158
DATA bitrev_size1024_radix4<>+0x16c8(SB)/8, $414  // bitrev[729] = 414
DATA bitrev_size1024_radix4<>+0x16d0(SB)/8, $670  // bitrev[730] = 670
DATA bitrev_size1024_radix4<>+0x16d8(SB)/8, $926  // bitrev[731] = 926
DATA bitrev_size1024_radix4<>+0x16e0(SB)/8, $222  // bitrev[732] = 222
DATA bitrev_size1024_radix4<>+0x16e8(SB)/8, $478  // bitrev[733] = 478
DATA bitrev_size1024_radix4<>+0x16f0(SB)/8, $734  // bitrev[734] = 734
DATA bitrev_size1024_radix4<>+0x16f8(SB)/8, $990  // bitrev[735] = 990
DATA bitrev_size1024_radix4<>+0x1700(SB)/8, $46   // bitrev[736] = 46
DATA bitrev_size1024_radix4<>+0x1708(SB)/8, $302  // bitrev[737] = 302
DATA bitrev_size1024_radix4<>+0x1710(SB)/8, $558  // bitrev[738] = 558
DATA bitrev_size1024_radix4<>+0x1718(SB)/8, $814  // bitrev[739] = 814
DATA bitrev_size1024_radix4<>+0x1720(SB)/8, $110  // bitrev[740] = 110
DATA bitrev_size1024_radix4<>+0x1728(SB)/8, $366  // bitrev[741] = 366
DATA bitrev_size1024_radix4<>+0x1730(SB)/8, $622  // bitrev[742] = 622
DATA bitrev_size1024_radix4<>+0x1738(SB)/8, $878  // bitrev[743] = 878
DATA bitrev_size1024_radix4<>+0x1740(SB)/8, $174  // bitrev[744] = 174
DATA bitrev_size1024_radix4<>+0x1748(SB)/8, $430  // bitrev[745] = 430
DATA bitrev_size1024_radix4<>+0x1750(SB)/8, $686  // bitrev[746] = 686
DATA bitrev_size1024_radix4<>+0x1758(SB)/8, $942  // bitrev[747] = 942
DATA bitrev_size1024_radix4<>+0x1760(SB)/8, $238  // bitrev[748] = 238
DATA bitrev_size1024_radix4<>+0x1768(SB)/8, $494  // bitrev[749] = 494
DATA bitrev_size1024_radix4<>+0x1770(SB)/8, $750  // bitrev[750] = 750
DATA bitrev_size1024_radix4<>+0x1778(SB)/8, $1006 // bitrev[751] = 1006
DATA bitrev_size1024_radix4<>+0x1780(SB)/8, $62   // bitrev[752] = 62
DATA bitrev_size1024_radix4<>+0x1788(SB)/8, $318  // bitrev[753] = 318
DATA bitrev_size1024_radix4<>+0x1790(SB)/8, $574  // bitrev[754] = 574
DATA bitrev_size1024_radix4<>+0x1798(SB)/8, $830  // bitrev[755] = 830
DATA bitrev_size1024_radix4<>+0x17a0(SB)/8, $126  // bitrev[756] = 126
DATA bitrev_size1024_radix4<>+0x17a8(SB)/8, $382  // bitrev[757] = 382
DATA bitrev_size1024_radix4<>+0x17b0(SB)/8, $638  // bitrev[758] = 638
DATA bitrev_size1024_radix4<>+0x17b8(SB)/8, $894  // bitrev[759] = 894
DATA bitrev_size1024_radix4<>+0x17c0(SB)/8, $190  // bitrev[760] = 190
DATA bitrev_size1024_radix4<>+0x17c8(SB)/8, $446  // bitrev[761] = 446
DATA bitrev_size1024_radix4<>+0x17d0(SB)/8, $702  // bitrev[762] = 702
DATA bitrev_size1024_radix4<>+0x17d8(SB)/8, $958  // bitrev[763] = 958
DATA bitrev_size1024_radix4<>+0x17e0(SB)/8, $254  // bitrev[764] = 254
DATA bitrev_size1024_radix4<>+0x17e8(SB)/8, $510  // bitrev[765] = 510
DATA bitrev_size1024_radix4<>+0x17f0(SB)/8, $766  // bitrev[766] = 766
DATA bitrev_size1024_radix4<>+0x17f8(SB)/8, $1022 // bitrev[767] = 1022
DATA bitrev_size1024_radix4<>+0x1800(SB)/8, $3    // bitrev[768] = 3
DATA bitrev_size1024_radix4<>+0x1808(SB)/8, $259  // bitrev[769] = 259
DATA bitrev_size1024_radix4<>+0x1810(SB)/8, $515  // bitrev[770] = 515
DATA bitrev_size1024_radix4<>+0x1818(SB)/8, $771  // bitrev[771] = 771
DATA bitrev_size1024_radix4<>+0x1820(SB)/8, $67   // bitrev[772] = 67
DATA bitrev_size1024_radix4<>+0x1828(SB)/8, $323  // bitrev[773] = 323
DATA bitrev_size1024_radix4<>+0x1830(SB)/8, $579  // bitrev[774] = 579
DATA bitrev_size1024_radix4<>+0x1838(SB)/8, $835  // bitrev[775] = 835
DATA bitrev_size1024_radix4<>+0x1840(SB)/8, $131  // bitrev[776] = 131
DATA bitrev_size1024_radix4<>+0x1848(SB)/8, $387  // bitrev[777] = 387
DATA bitrev_size1024_radix4<>+0x1850(SB)/8, $643  // bitrev[778] = 643
DATA bitrev_size1024_radix4<>+0x1858(SB)/8, $899  // bitrev[779] = 899
DATA bitrev_size1024_radix4<>+0x1860(SB)/8, $195  // bitrev[780] = 195
DATA bitrev_size1024_radix4<>+0x1868(SB)/8, $451  // bitrev[781] = 451
DATA bitrev_size1024_radix4<>+0x1870(SB)/8, $707  // bitrev[782] = 707
DATA bitrev_size1024_radix4<>+0x1878(SB)/8, $963  // bitrev[783] = 963
DATA bitrev_size1024_radix4<>+0x1880(SB)/8, $19   // bitrev[784] = 19
DATA bitrev_size1024_radix4<>+0x1888(SB)/8, $275  // bitrev[785] = 275
DATA bitrev_size1024_radix4<>+0x1890(SB)/8, $531  // bitrev[786] = 531
DATA bitrev_size1024_radix4<>+0x1898(SB)/8, $787  // bitrev[787] = 787
DATA bitrev_size1024_radix4<>+0x18a0(SB)/8, $83   // bitrev[788] = 83
DATA bitrev_size1024_radix4<>+0x18a8(SB)/8, $339  // bitrev[789] = 339
DATA bitrev_size1024_radix4<>+0x18b0(SB)/8, $595  // bitrev[790] = 595
DATA bitrev_size1024_radix4<>+0x18b8(SB)/8, $851  // bitrev[791] = 851
DATA bitrev_size1024_radix4<>+0x18c0(SB)/8, $147  // bitrev[792] = 147
DATA bitrev_size1024_radix4<>+0x18c8(SB)/8, $403  // bitrev[793] = 403
DATA bitrev_size1024_radix4<>+0x18d0(SB)/8, $659  // bitrev[794] = 659
DATA bitrev_size1024_radix4<>+0x18d8(SB)/8, $915  // bitrev[795] = 915
DATA bitrev_size1024_radix4<>+0x18e0(SB)/8, $211  // bitrev[796] = 211
DATA bitrev_size1024_radix4<>+0x18e8(SB)/8, $467  // bitrev[797] = 467
DATA bitrev_size1024_radix4<>+0x18f0(SB)/8, $723  // bitrev[798] = 723
DATA bitrev_size1024_radix4<>+0x18f8(SB)/8, $979  // bitrev[799] = 979
DATA bitrev_size1024_radix4<>+0x1900(SB)/8, $35   // bitrev[800] = 35
DATA bitrev_size1024_radix4<>+0x1908(SB)/8, $291  // bitrev[801] = 291
DATA bitrev_size1024_radix4<>+0x1910(SB)/8, $547  // bitrev[802] = 547
DATA bitrev_size1024_radix4<>+0x1918(SB)/8, $803  // bitrev[803] = 803
DATA bitrev_size1024_radix4<>+0x1920(SB)/8, $99   // bitrev[804] = 99
DATA bitrev_size1024_radix4<>+0x1928(SB)/8, $355  // bitrev[805] = 355
DATA bitrev_size1024_radix4<>+0x1930(SB)/8, $611  // bitrev[806] = 611
DATA bitrev_size1024_radix4<>+0x1938(SB)/8, $867  // bitrev[807] = 867
DATA bitrev_size1024_radix4<>+0x1940(SB)/8, $163  // bitrev[808] = 163
DATA bitrev_size1024_radix4<>+0x1948(SB)/8, $419  // bitrev[809] = 419
DATA bitrev_size1024_radix4<>+0x1950(SB)/8, $675  // bitrev[810] = 675
DATA bitrev_size1024_radix4<>+0x1958(SB)/8, $931  // bitrev[811] = 931
DATA bitrev_size1024_radix4<>+0x1960(SB)/8, $227  // bitrev[812] = 227
DATA bitrev_size1024_radix4<>+0x1968(SB)/8, $483  // bitrev[813] = 483
DATA bitrev_size1024_radix4<>+0x1970(SB)/8, $739  // bitrev[814] = 739
DATA bitrev_size1024_radix4<>+0x1978(SB)/8, $995  // bitrev[815] = 995
DATA bitrev_size1024_radix4<>+0x1980(SB)/8, $51   // bitrev[816] = 51
DATA bitrev_size1024_radix4<>+0x1988(SB)/8, $307  // bitrev[817] = 307
DATA bitrev_size1024_radix4<>+0x1990(SB)/8, $563  // bitrev[818] = 563
DATA bitrev_size1024_radix4<>+0x1998(SB)/8, $819  // bitrev[819] = 819
DATA bitrev_size1024_radix4<>+0x19a0(SB)/8, $115  // bitrev[820] = 115
DATA bitrev_size1024_radix4<>+0x19a8(SB)/8, $371  // bitrev[821] = 371
DATA bitrev_size1024_radix4<>+0x19b0(SB)/8, $627  // bitrev[822] = 627
DATA bitrev_size1024_radix4<>+0x19b8(SB)/8, $883  // bitrev[823] = 883
DATA bitrev_size1024_radix4<>+0x19c0(SB)/8, $179  // bitrev[824] = 179
DATA bitrev_size1024_radix4<>+0x19c8(SB)/8, $435  // bitrev[825] = 435
DATA bitrev_size1024_radix4<>+0x19d0(SB)/8, $691  // bitrev[826] = 691
DATA bitrev_size1024_radix4<>+0x19d8(SB)/8, $947  // bitrev[827] = 947
DATA bitrev_size1024_radix4<>+0x19e0(SB)/8, $243  // bitrev[828] = 243
DATA bitrev_size1024_radix4<>+0x19e8(SB)/8, $499  // bitrev[829] = 499
DATA bitrev_size1024_radix4<>+0x19f0(SB)/8, $755  // bitrev[830] = 755
DATA bitrev_size1024_radix4<>+0x19f8(SB)/8, $1011 // bitrev[831] = 1011
DATA bitrev_size1024_radix4<>+0x1a00(SB)/8, $7    // bitrev[832] = 7
DATA bitrev_size1024_radix4<>+0x1a08(SB)/8, $263  // bitrev[833] = 263
DATA bitrev_size1024_radix4<>+0x1a10(SB)/8, $519  // bitrev[834] = 519
DATA bitrev_size1024_radix4<>+0x1a18(SB)/8, $775  // bitrev[835] = 775
DATA bitrev_size1024_radix4<>+0x1a20(SB)/8, $71   // bitrev[836] = 71
DATA bitrev_size1024_radix4<>+0x1a28(SB)/8, $327  // bitrev[837] = 327
DATA bitrev_size1024_radix4<>+0x1a30(SB)/8, $583  // bitrev[838] = 583
DATA bitrev_size1024_radix4<>+0x1a38(SB)/8, $839  // bitrev[839] = 839
DATA bitrev_size1024_radix4<>+0x1a40(SB)/8, $135  // bitrev[840] = 135
DATA bitrev_size1024_radix4<>+0x1a48(SB)/8, $391  // bitrev[841] = 391
DATA bitrev_size1024_radix4<>+0x1a50(SB)/8, $647  // bitrev[842] = 647
DATA bitrev_size1024_radix4<>+0x1a58(SB)/8, $903  // bitrev[843] = 903
DATA bitrev_size1024_radix4<>+0x1a60(SB)/8, $199  // bitrev[844] = 199
DATA bitrev_size1024_radix4<>+0x1a68(SB)/8, $455  // bitrev[845] = 455
DATA bitrev_size1024_radix4<>+0x1a70(SB)/8, $711  // bitrev[846] = 711
DATA bitrev_size1024_radix4<>+0x1a78(SB)/8, $967  // bitrev[847] = 967
DATA bitrev_size1024_radix4<>+0x1a80(SB)/8, $23   // bitrev[848] = 23
DATA bitrev_size1024_radix4<>+0x1a88(SB)/8, $279  // bitrev[849] = 279
DATA bitrev_size1024_radix4<>+0x1a90(SB)/8, $535  // bitrev[850] = 535
DATA bitrev_size1024_radix4<>+0x1a98(SB)/8, $791  // bitrev[851] = 791
DATA bitrev_size1024_radix4<>+0x1aa0(SB)/8, $87   // bitrev[852] = 87
DATA bitrev_size1024_radix4<>+0x1aa8(SB)/8, $343  // bitrev[853] = 343
DATA bitrev_size1024_radix4<>+0x1ab0(SB)/8, $599  // bitrev[854] = 599
DATA bitrev_size1024_radix4<>+0x1ab8(SB)/8, $855  // bitrev[855] = 855
DATA bitrev_size1024_radix4<>+0x1ac0(SB)/8, $151  // bitrev[856] = 151
DATA bitrev_size1024_radix4<>+0x1ac8(SB)/8, $407  // bitrev[857] = 407
DATA bitrev_size1024_radix4<>+0x1ad0(SB)/8, $663  // bitrev[858] = 663
DATA bitrev_size1024_radix4<>+0x1ad8(SB)/8, $919  // bitrev[859] = 919
DATA bitrev_size1024_radix4<>+0x1ae0(SB)/8, $215  // bitrev[860] = 215
DATA bitrev_size1024_radix4<>+0x1ae8(SB)/8, $471  // bitrev[861] = 471
DATA bitrev_size1024_radix4<>+0x1af0(SB)/8, $727  // bitrev[862] = 727
DATA bitrev_size1024_radix4<>+0x1af8(SB)/8, $983  // bitrev[863] = 983
DATA bitrev_size1024_radix4<>+0x1b00(SB)/8, $39   // bitrev[864] = 39
DATA bitrev_size1024_radix4<>+0x1b08(SB)/8, $295  // bitrev[865] = 295
DATA bitrev_size1024_radix4<>+0x1b10(SB)/8, $551  // bitrev[866] = 551
DATA bitrev_size1024_radix4<>+0x1b18(SB)/8, $807  // bitrev[867] = 807
DATA bitrev_size1024_radix4<>+0x1b20(SB)/8, $103  // bitrev[868] = 103
DATA bitrev_size1024_radix4<>+0x1b28(SB)/8, $359  // bitrev[869] = 359
DATA bitrev_size1024_radix4<>+0x1b30(SB)/8, $615  // bitrev[870] = 615
DATA bitrev_size1024_radix4<>+0x1b38(SB)/8, $871  // bitrev[871] = 871
DATA bitrev_size1024_radix4<>+0x1b40(SB)/8, $167  // bitrev[872] = 167
DATA bitrev_size1024_radix4<>+0x1b48(SB)/8, $423  // bitrev[873] = 423
DATA bitrev_size1024_radix4<>+0x1b50(SB)/8, $679  // bitrev[874] = 679
DATA bitrev_size1024_radix4<>+0x1b58(SB)/8, $935  // bitrev[875] = 935
DATA bitrev_size1024_radix4<>+0x1b60(SB)/8, $231  // bitrev[876] = 231
DATA bitrev_size1024_radix4<>+0x1b68(SB)/8, $487  // bitrev[877] = 487
DATA bitrev_size1024_radix4<>+0x1b70(SB)/8, $743  // bitrev[878] = 743
DATA bitrev_size1024_radix4<>+0x1b78(SB)/8, $999  // bitrev[879] = 999
DATA bitrev_size1024_radix4<>+0x1b80(SB)/8, $55   // bitrev[880] = 55
DATA bitrev_size1024_radix4<>+0x1b88(SB)/8, $311  // bitrev[881] = 311
DATA bitrev_size1024_radix4<>+0x1b90(SB)/8, $567  // bitrev[882] = 567
DATA bitrev_size1024_radix4<>+0x1b98(SB)/8, $823  // bitrev[883] = 823
DATA bitrev_size1024_radix4<>+0x1ba0(SB)/8, $119  // bitrev[884] = 119
DATA bitrev_size1024_radix4<>+0x1ba8(SB)/8, $375  // bitrev[885] = 375
DATA bitrev_size1024_radix4<>+0x1bb0(SB)/8, $631  // bitrev[886] = 631
DATA bitrev_size1024_radix4<>+0x1bb8(SB)/8, $887  // bitrev[887] = 887
DATA bitrev_size1024_radix4<>+0x1bc0(SB)/8, $183  // bitrev[888] = 183
DATA bitrev_size1024_radix4<>+0x1bc8(SB)/8, $439  // bitrev[889] = 439
DATA bitrev_size1024_radix4<>+0x1bd0(SB)/8, $695  // bitrev[890] = 695
DATA bitrev_size1024_radix4<>+0x1bd8(SB)/8, $951  // bitrev[891] = 951
DATA bitrev_size1024_radix4<>+0x1be0(SB)/8, $247  // bitrev[892] = 247
DATA bitrev_size1024_radix4<>+0x1be8(SB)/8, $503  // bitrev[893] = 503
DATA bitrev_size1024_radix4<>+0x1bf0(SB)/8, $759  // bitrev[894] = 759
DATA bitrev_size1024_radix4<>+0x1bf8(SB)/8, $1015 // bitrev[895] = 1015
DATA bitrev_size1024_radix4<>+0x1c00(SB)/8, $11   // bitrev[896] = 11
DATA bitrev_size1024_radix4<>+0x1c08(SB)/8, $267  // bitrev[897] = 267
DATA bitrev_size1024_radix4<>+0x1c10(SB)/8, $523  // bitrev[898] = 523
DATA bitrev_size1024_radix4<>+0x1c18(SB)/8, $779  // bitrev[899] = 779
DATA bitrev_size1024_radix4<>+0x1c20(SB)/8, $75   // bitrev[900] = 75
DATA bitrev_size1024_radix4<>+0x1c28(SB)/8, $331  // bitrev[901] = 331
DATA bitrev_size1024_radix4<>+0x1c30(SB)/8, $587  // bitrev[902] = 587
DATA bitrev_size1024_radix4<>+0x1c38(SB)/8, $843  // bitrev[903] = 843
DATA bitrev_size1024_radix4<>+0x1c40(SB)/8, $139  // bitrev[904] = 139
DATA bitrev_size1024_radix4<>+0x1c48(SB)/8, $395  // bitrev[905] = 395
DATA bitrev_size1024_radix4<>+0x1c50(SB)/8, $651  // bitrev[906] = 651
DATA bitrev_size1024_radix4<>+0x1c58(SB)/8, $907  // bitrev[907] = 907
DATA bitrev_size1024_radix4<>+0x1c60(SB)/8, $203  // bitrev[908] = 203
DATA bitrev_size1024_radix4<>+0x1c68(SB)/8, $459  // bitrev[909] = 459
DATA bitrev_size1024_radix4<>+0x1c70(SB)/8, $715  // bitrev[910] = 715
DATA bitrev_size1024_radix4<>+0x1c78(SB)/8, $971  // bitrev[911] = 971
DATA bitrev_size1024_radix4<>+0x1c80(SB)/8, $27   // bitrev[912] = 27
DATA bitrev_size1024_radix4<>+0x1c88(SB)/8, $283  // bitrev[913] = 283
DATA bitrev_size1024_radix4<>+0x1c90(SB)/8, $539  // bitrev[914] = 539
DATA bitrev_size1024_radix4<>+0x1c98(SB)/8, $795  // bitrev[915] = 795
DATA bitrev_size1024_radix4<>+0x1ca0(SB)/8, $91   // bitrev[916] = 91
DATA bitrev_size1024_radix4<>+0x1ca8(SB)/8, $347  // bitrev[917] = 347
DATA bitrev_size1024_radix4<>+0x1cb0(SB)/8, $603  // bitrev[918] = 603
DATA bitrev_size1024_radix4<>+0x1cb8(SB)/8, $859  // bitrev[919] = 859
DATA bitrev_size1024_radix4<>+0x1cc0(SB)/8, $155  // bitrev[920] = 155
DATA bitrev_size1024_radix4<>+0x1cc8(SB)/8, $411  // bitrev[921] = 411
DATA bitrev_size1024_radix4<>+0x1cd0(SB)/8, $667  // bitrev[922] = 667
DATA bitrev_size1024_radix4<>+0x1cd8(SB)/8, $923  // bitrev[923] = 923
DATA bitrev_size1024_radix4<>+0x1ce0(SB)/8, $219  // bitrev[924] = 219
DATA bitrev_size1024_radix4<>+0x1ce8(SB)/8, $475  // bitrev[925] = 475
DATA bitrev_size1024_radix4<>+0x1cf0(SB)/8, $731  // bitrev[926] = 731
DATA bitrev_size1024_radix4<>+0x1cf8(SB)/8, $987  // bitrev[927] = 987
DATA bitrev_size1024_radix4<>+0x1d00(SB)/8, $43   // bitrev[928] = 43
DATA bitrev_size1024_radix4<>+0x1d08(SB)/8, $299  // bitrev[929] = 299
DATA bitrev_size1024_radix4<>+0x1d10(SB)/8, $555  // bitrev[930] = 555
DATA bitrev_size1024_radix4<>+0x1d18(SB)/8, $811  // bitrev[931] = 811
DATA bitrev_size1024_radix4<>+0x1d20(SB)/8, $107  // bitrev[932] = 107
DATA bitrev_size1024_radix4<>+0x1d28(SB)/8, $363  // bitrev[933] = 363
DATA bitrev_size1024_radix4<>+0x1d30(SB)/8, $619  // bitrev[934] = 619
DATA bitrev_size1024_radix4<>+0x1d38(SB)/8, $875  // bitrev[935] = 875
DATA bitrev_size1024_radix4<>+0x1d40(SB)/8, $171  // bitrev[936] = 171
DATA bitrev_size1024_radix4<>+0x1d48(SB)/8, $427  // bitrev[937] = 427
DATA bitrev_size1024_radix4<>+0x1d50(SB)/8, $683  // bitrev[938] = 683
DATA bitrev_size1024_radix4<>+0x1d58(SB)/8, $939  // bitrev[939] = 939
DATA bitrev_size1024_radix4<>+0x1d60(SB)/8, $235  // bitrev[940] = 235
DATA bitrev_size1024_radix4<>+0x1d68(SB)/8, $491  // bitrev[941] = 491
DATA bitrev_size1024_radix4<>+0x1d70(SB)/8, $747  // bitrev[942] = 747
DATA bitrev_size1024_radix4<>+0x1d78(SB)/8, $1003 // bitrev[943] = 1003
DATA bitrev_size1024_radix4<>+0x1d80(SB)/8, $59   // bitrev[944] = 59
DATA bitrev_size1024_radix4<>+0x1d88(SB)/8, $315  // bitrev[945] = 315
DATA bitrev_size1024_radix4<>+0x1d90(SB)/8, $571  // bitrev[946] = 571
DATA bitrev_size1024_radix4<>+0x1d98(SB)/8, $827  // bitrev[947] = 827
DATA bitrev_size1024_radix4<>+0x1da0(SB)/8, $123  // bitrev[948] = 123
DATA bitrev_size1024_radix4<>+0x1da8(SB)/8, $379  // bitrev[949] = 379
DATA bitrev_size1024_radix4<>+0x1db0(SB)/8, $635  // bitrev[950] = 635
DATA bitrev_size1024_radix4<>+0x1db8(SB)/8, $891  // bitrev[951] = 891
DATA bitrev_size1024_radix4<>+0x1dc0(SB)/8, $187  // bitrev[952] = 187
DATA bitrev_size1024_radix4<>+0x1dc8(SB)/8, $443  // bitrev[953] = 443
DATA bitrev_size1024_radix4<>+0x1dd0(SB)/8, $699  // bitrev[954] = 699
DATA bitrev_size1024_radix4<>+0x1dd8(SB)/8, $955  // bitrev[955] = 955
DATA bitrev_size1024_radix4<>+0x1de0(SB)/8, $251  // bitrev[956] = 251
DATA bitrev_size1024_radix4<>+0x1de8(SB)/8, $507  // bitrev[957] = 507
DATA bitrev_size1024_radix4<>+0x1df0(SB)/8, $763  // bitrev[958] = 763
DATA bitrev_size1024_radix4<>+0x1df8(SB)/8, $1019 // bitrev[959] = 1019
DATA bitrev_size1024_radix4<>+0x1e00(SB)/8, $15   // bitrev[960] = 15
DATA bitrev_size1024_radix4<>+0x1e08(SB)/8, $271  // bitrev[961] = 271
DATA bitrev_size1024_radix4<>+0x1e10(SB)/8, $527  // bitrev[962] = 527
DATA bitrev_size1024_radix4<>+0x1e18(SB)/8, $783  // bitrev[963] = 783
DATA bitrev_size1024_radix4<>+0x1e20(SB)/8, $79   // bitrev[964] = 79
DATA bitrev_size1024_radix4<>+0x1e28(SB)/8, $335  // bitrev[965] = 335
DATA bitrev_size1024_radix4<>+0x1e30(SB)/8, $591  // bitrev[966] = 591
DATA bitrev_size1024_radix4<>+0x1e38(SB)/8, $847  // bitrev[967] = 847
DATA bitrev_size1024_radix4<>+0x1e40(SB)/8, $143  // bitrev[968] = 143
DATA bitrev_size1024_radix4<>+0x1e48(SB)/8, $399  // bitrev[969] = 399
DATA bitrev_size1024_radix4<>+0x1e50(SB)/8, $655  // bitrev[970] = 655
DATA bitrev_size1024_radix4<>+0x1e58(SB)/8, $911  // bitrev[971] = 911
DATA bitrev_size1024_radix4<>+0x1e60(SB)/8, $207  // bitrev[972] = 207
DATA bitrev_size1024_radix4<>+0x1e68(SB)/8, $463  // bitrev[973] = 463
DATA bitrev_size1024_radix4<>+0x1e70(SB)/8, $719  // bitrev[974] = 719
DATA bitrev_size1024_radix4<>+0x1e78(SB)/8, $975  // bitrev[975] = 975
DATA bitrev_size1024_radix4<>+0x1e80(SB)/8, $31   // bitrev[976] = 31
DATA bitrev_size1024_radix4<>+0x1e88(SB)/8, $287  // bitrev[977] = 287
DATA bitrev_size1024_radix4<>+0x1e90(SB)/8, $543  // bitrev[978] = 543
DATA bitrev_size1024_radix4<>+0x1e98(SB)/8, $799  // bitrev[979] = 799
DATA bitrev_size1024_radix4<>+0x1ea0(SB)/8, $95   // bitrev[980] = 95
DATA bitrev_size1024_radix4<>+0x1ea8(SB)/8, $351  // bitrev[981] = 351
DATA bitrev_size1024_radix4<>+0x1eb0(SB)/8, $607  // bitrev[982] = 607
DATA bitrev_size1024_radix4<>+0x1eb8(SB)/8, $863  // bitrev[983] = 863
DATA bitrev_size1024_radix4<>+0x1ec0(SB)/8, $159  // bitrev[984] = 159
DATA bitrev_size1024_radix4<>+0x1ec8(SB)/8, $415  // bitrev[985] = 415
DATA bitrev_size1024_radix4<>+0x1ed0(SB)/8, $671  // bitrev[986] = 671
DATA bitrev_size1024_radix4<>+0x1ed8(SB)/8, $927  // bitrev[987] = 927
DATA bitrev_size1024_radix4<>+0x1ee0(SB)/8, $223  // bitrev[988] = 223
DATA bitrev_size1024_radix4<>+0x1ee8(SB)/8, $479  // bitrev[989] = 479
DATA bitrev_size1024_radix4<>+0x1ef0(SB)/8, $735  // bitrev[990] = 735
DATA bitrev_size1024_radix4<>+0x1ef8(SB)/8, $991  // bitrev[991] = 991
DATA bitrev_size1024_radix4<>+0x1f00(SB)/8, $47   // bitrev[992] = 47
DATA bitrev_size1024_radix4<>+0x1f08(SB)/8, $303  // bitrev[993] = 303
DATA bitrev_size1024_radix4<>+0x1f10(SB)/8, $559  // bitrev[994] = 559
DATA bitrev_size1024_radix4<>+0x1f18(SB)/8, $815  // bitrev[995] = 815
DATA bitrev_size1024_radix4<>+0x1f20(SB)/8, $111  // bitrev[996] = 111
DATA bitrev_size1024_radix4<>+0x1f28(SB)/8, $367  // bitrev[997] = 367
DATA bitrev_size1024_radix4<>+0x1f30(SB)/8, $623  // bitrev[998] = 623
DATA bitrev_size1024_radix4<>+0x1f38(SB)/8, $879  // bitrev[999] = 879
DATA bitrev_size1024_radix4<>+0x1f40(SB)/8, $175  // bitrev[1000] = 175
DATA bitrev_size1024_radix4<>+0x1f48(SB)/8, $431  // bitrev[1001] = 431
DATA bitrev_size1024_radix4<>+0x1f50(SB)/8, $687  // bitrev[1002] = 687
DATA bitrev_size1024_radix4<>+0x1f58(SB)/8, $943  // bitrev[1003] = 943
DATA bitrev_size1024_radix4<>+0x1f60(SB)/8, $239  // bitrev[1004] = 239
DATA bitrev_size1024_radix4<>+0x1f68(SB)/8, $495  // bitrev[1005] = 495
DATA bitrev_size1024_radix4<>+0x1f70(SB)/8, $751  // bitrev[1006] = 751
DATA bitrev_size1024_radix4<>+0x1f78(SB)/8, $1007 // bitrev[1007] = 1007
DATA bitrev_size1024_radix4<>+0x1f80(SB)/8, $63   // bitrev[1008] = 63
DATA bitrev_size1024_radix4<>+0x1f88(SB)/8, $319  // bitrev[1009] = 319
DATA bitrev_size1024_radix4<>+0x1f90(SB)/8, $575  // bitrev[1010] = 575
DATA bitrev_size1024_radix4<>+0x1f98(SB)/8, $831  // bitrev[1011] = 831
DATA bitrev_size1024_radix4<>+0x1fa0(SB)/8, $127  // bitrev[1012] = 127
DATA bitrev_size1024_radix4<>+0x1fa8(SB)/8, $383  // bitrev[1013] = 383
DATA bitrev_size1024_radix4<>+0x1fb0(SB)/8, $639  // bitrev[1014] = 639
DATA bitrev_size1024_radix4<>+0x1fb8(SB)/8, $895  // bitrev[1015] = 895
DATA bitrev_size1024_radix4<>+0x1fc0(SB)/8, $191  // bitrev[1016] = 191
DATA bitrev_size1024_radix4<>+0x1fc8(SB)/8, $447  // bitrev[1017] = 447
DATA bitrev_size1024_radix4<>+0x1fd0(SB)/8, $703  // bitrev[1018] = 703
DATA bitrev_size1024_radix4<>+0x1fd8(SB)/8, $959  // bitrev[1019] = 959
DATA bitrev_size1024_radix4<>+0x1fe0(SB)/8, $255  // bitrev[1020] = 255
DATA bitrev_size1024_radix4<>+0x1fe8(SB)/8, $511  // bitrev[1021] = 511
DATA bitrev_size1024_radix4<>+0x1ff0(SB)/8, $767  // bitrev[1022] = 767
DATA bitrev_size1024_radix4<>+0x1ff8(SB)/8, $1023 // bitrev[1023] = 1023
GLOBL bitrev_size1024_radix4<>(SB), RODATA, $8192

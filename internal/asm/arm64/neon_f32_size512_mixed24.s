//go:build arm64 && !purego

// ===========================================================================
// NEON Size-512 Mixed-Radix (Radix-4 + Radix-2) FFT Kernels for ARM64
// ===========================================================================
//
// Size 512 = 4 * 4 * 4 * 4 * 2, mixed-radix algorithm:
//   Stage 1: 128 radix-4 butterflies (no twiddles), stride=4
//   Stage 2: radix-4 with twiddles, 32 groups, step=32
//   Stage 3: radix-4 with twiddles, 8 groups, step=8
//   Stage 4: radix-4 with twiddles, 2 groups, step=2
//   Stage 5: radix-2 with twiddles, size=512, step=1
//
// Each complex64 element is 8 bytes (real f32 + imag f32).
//
// ===========================================================================

#include "textflag.h"

// Forward transform, size 512, complex64, mixed radix
// func ForwardNEONSize512Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool
TEXT ·ForwardNEONSize512Radix4Then2Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src_len+32(FP), R13

	CMP  $512, R13
	BNE  neon512m24_return_false

	MOVD dst_len+8(FP), R0
	CMP  $512, R0
	BLT  neon512m24_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $512, R0
	BLT  neon512m24_return_false

	MOVD scratch_len+80(FP), R0
	CMP  $512, R0
	BLT  neon512m24_return_false

	MOVD $bitrev_size512_mixed24<>(SB), R12
	MOVD R8, R20

	CMP  R8, R9
	BNE  neon512m24_use_dst
	MOVD R11, R8

neon512m24_use_dst:
	// =========================================================================
	// Bit-reversal permutation
	// =========================================================================
	MOVD $0, R0

neon512m24_bitrev_loop:
	CMP  $512, R0
	BGE  neon512m24_stage1

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
	B    neon512m24_bitrev_loop

neon512m24_stage1:
	// =========================================================================
	// Stage 1: 128 radix-4 butterflies (no twiddles)
	// =========================================================================
	MOVD $0, R14

neon512m24_stage1_loop:
	CMP  $512, R14
	BGE  neon512m24_stage2

	LSL  $3, R14, R1
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
	FNEGS F14, F21
	FADDS F20, F10, F22
	FADDS F21, F11, F23

	FNEGS F15, F24
	FMOVS F14, F25
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

	ADD  $4, R14, R14
	B    neon512m24_stage1_loop

neon512m24_stage2:
	// =========================================================================
	// Stage 2: radix-4 with twiddles, 32 groups
	// Twiddle indices: j*32, j*64, j*96
	// =========================================================================
	MOVD $0, R14

neon512m24_stage2_base:
	CMP  $512, R14
	BGE  neon512m24_stage3

	MOVD $0, R15

neon512m24_stage2_j:
	CMP  $4, R15
	BGE  neon512m24_stage2_next

	ADD  R14, R15, R0
	ADD  $4, R0, R1
	ADD  $8, R0, R2
	ADD  $12, R0, R3

	// Twiddles: w1=tw[j*32], w2=tw[j*64], w3=tw[j*96]
	LSL  $5, R15, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1

	LSL  $6, R15, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F2
	FMOVS 4(R5), F3

	MOVD $96, R6
	MUL  R15, R6, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F4
	FMOVS 4(R5), F5

	// Load values
	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F6
	FMOVS 4(R7), F7

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F8
	FMOVS 4(R7), F9

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F10
	FMOVS 4(R7), F11

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F12
	FMOVS 4(R7), F13

	// Apply twiddles (complex multiply)
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

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

	FADDS F21, F16, F26
	FSUBS F20, F17, F27

	FSUBS F21, F16, F28
	FADDS F20, F17, F29

	// Store
	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS F22, 0(R7)
	FMOVS F23, 4(R7)

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS F26, 0(R7)
	FMOVS F27, 4(R7)

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS F24, 0(R7)
	FMOVS F25, 4(R7)

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS F28, 0(R7)
	FMOVS F29, 4(R7)

	ADD  $1, R15, R15
	B    neon512m24_stage2_j

neon512m24_stage2_next:
	ADD  $16, R14, R14
	B    neon512m24_stage2_base

neon512m24_stage3:
	// =========================================================================
	// Stage 3: radix-4 with twiddles, 8 groups
	// Twiddle indices: j*8, j*16, j*24
	// =========================================================================
	MOVD $0, R14

neon512m24_stage3_base:
	CMP  $512, R14
	BGE  neon512m24_stage4

	MOVD $0, R15

neon512m24_stage3_j:
	CMP  $16, R15
	BGE  neon512m24_stage3_next

	ADD  R14, R15, R0
	ADD  $16, R0, R1
	ADD  $32, R0, R2
	ADD  $48, R0, R3

	// Twiddles: w1=tw[j*8], w2=tw[j*16], w3=tw[j*24]
	LSL  $3, R15, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1

	LSL  $4, R15, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F2
	FMOVS 4(R5), F3

	MOVD $24, R6
	MUL  R15, R6, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F4
	FMOVS 4(R5), F5

	// Load values
	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F6
	FMOVS 4(R7), F7

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F8
	FMOVS 4(R7), F9

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F10
	FMOVS 4(R7), F11

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F12
	FMOVS 4(R7), F13

	// Apply twiddles
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

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

	FADDS F21, F16, F26
	FSUBS F20, F17, F27

	FSUBS F21, F16, F28
	FADDS F20, F17, F29

	// Store
	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS F22, 0(R7)
	FMOVS F23, 4(R7)

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS F26, 0(R7)
	FMOVS F27, 4(R7)

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS F24, 0(R7)
	FMOVS F25, 4(R7)

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS F28, 0(R7)
	FMOVS F29, 4(R7)

	ADD  $1, R15, R15
	B    neon512m24_stage3_j

neon512m24_stage3_next:
	ADD  $64, R14, R14
	B    neon512m24_stage3_base

neon512m24_stage4:
	// =========================================================================
	// Stage 4: radix-4 with twiddles, 2 groups
	// Twiddle indices: j*2, j*4, j*6
	// =========================================================================
	MOVD $0, R14

neon512m24_stage4_base:
	CMP  $512, R14
	BGE  neon512m24_stage5

	MOVD $0, R15

neon512m24_stage4_j:
	CMP  $64, R15
	BGE  neon512m24_stage4_next

	ADD  R14, R15, R0
	ADD  $64, R0, R1
	ADD  $128, R0, R2
	ADD  $192, R0, R3

	// Twiddles: w1=tw[j*2], w2=tw[j*4], w3=tw[j*6]
	LSL  $1, R15, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1

	LSL  $2, R15, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F2
	FMOVS 4(R5), F3

	MOVD $6, R6
	MUL  R15, R6, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F4
	FMOVS 4(R5), F5

	// Load values
	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F6
	FMOVS 4(R7), F7

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F8
	FMOVS 4(R7), F9

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F10
	FMOVS 4(R7), F11

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F12
	FMOVS 4(R7), F13

	// Apply twiddles
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

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

	FADDS F21, F16, F26
	FSUBS F20, F17, F27

	FSUBS F21, F16, F28
	FADDS F20, F17, F29

	// Store
	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS F22, 0(R7)
	FMOVS F23, 4(R7)

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS F26, 0(R7)
	FMOVS F27, 4(R7)

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS F24, 0(R7)
	FMOVS F25, 4(R7)

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS F28, 0(R7)
	FMOVS F29, 4(R7)

	ADD  $1, R15, R15
	B    neon512m24_stage4_j

neon512m24_stage4_next:
	ADD  $256, R14, R14
	B    neon512m24_stage4_base

neon512m24_stage5:
	// =========================================================================
	// Stage 5: radix-2 with twiddles, size=512, step=1
	// =========================================================================
	MOVD $0, R0

neon512m24_stage5_loop:
	CMP  $256, R0
	BGE  neon512m24_done

	ADD  $256, R0, R1

	// Load twiddle[j]
	LSL  $3, R0, R2
	ADD  R10, R2, R2
	FMOVS 0(R2), F0
	FMOVS 4(R2), F1

	// Load a, b
	LSL  $3, R0, R2
	ADD  R8, R2, R2
	FMOVS 0(R2), F2
	FMOVS 4(R2), F3

	LSL  $3, R1, R2
	ADD  R8, R2, R2
	FMOVS 0(R2), F4
	FMOVS 4(R2), F5

	// wb = w * b
	FMULS F0, F4, F6
	FMULS F1, F5, F7
	FSUBS F7, F6, F6
	FMULS F0, F5, F7
	FMULS F1, F4, F8
	FADDS F8, F7, F7

	// Butterfly
	FADDS F6, F2, F8
	FADDS F7, F3, F9
	FSUBS F6, F2, F10
	FSUBS F7, F3, F11

	// Store
	LSL  $3, R0, R2
	ADD  R8, R2, R2
	FMOVS F8, 0(R2)
	FMOVS F9, 4(R2)

	LSL  $3, R1, R2
	ADD  R8, R2, R2
	FMOVS F10, 0(R2)
	FMOVS F11, 4(R2)

	ADD  $1, R0, R0
	B    neon512m24_stage5_loop

neon512m24_done:
	CMP  R8, R20
	BEQ  neon512m24_return_true

	MOVD $0, R0
neon512m24_copy_loop:
	CMP  $512, R0
	BGE  neon512m24_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon512m24_copy_loop

neon512m24_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon512m24_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform
// ===========================================================================
TEXT ·InverseNEONSize512Radix4Then2Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src_len+32(FP), R13

	CMP  $512, R13
	BNE  neon512m24_inv_return_false

	MOVD dst_len+8(FP), R0
	CMP  $512, R0
	BLT  neon512m24_inv_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $512, R0
	BLT  neon512m24_inv_return_false

	MOVD scratch_len+80(FP), R0
	CMP  $512, R0
	BLT  neon512m24_inv_return_false

	MOVD $bitrev_size512_mixed24<>(SB), R12
	MOVD R8, R20

	CMP  R8, R9
	BNE  neon512m24_inv_use_dst
	MOVD R11, R8

neon512m24_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon512m24_inv_bitrev_loop:
	CMP  $512, R0
	BGE  neon512m24_inv_stage1

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
	B    neon512m24_inv_bitrev_loop

neon512m24_inv_stage1:
	// Stage 1: 128 radix-4 butterflies (inverse)
	MOVD $0, R14

neon512m24_inv_stage1_loop:
	CMP  $512, R14
	BGE  neon512m24_inv_stage2

	LSL  $3, R14, R1
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

	// For inverse: i * t3
	FNEGS F15, F20
	FMOVS F14, F21

	FADDS F20, F10, F22
	FADDS F21, F11, F23

	// For inverse: (-i) * t3
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

	ADD  $4, R14, R14
	B    neon512m24_inv_stage1_loop

neon512m24_inv_stage2:
	// Stage 2 with conjugated twiddles
	MOVD $0, R14

neon512m24_inv_stage2_base:
	CMP  $512, R14
	BGE  neon512m24_inv_stage3

	MOVD $0, R15

neon512m24_inv_stage2_j:
	CMP  $4, R15
	BGE  neon512m24_inv_stage2_next

	ADD  R14, R15, R0
	ADD  $4, R0, R1
	ADD  $8, R0, R2
	ADD  $12, R0, R3

	LSL  $5, R15, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1
	FNEGS F1, F1

	LSL  $6, R15, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F2
	FMOVS 4(R5), F3
	FNEGS F3, F3

	MOVD $96, R6
	MUL  R15, R6, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F4
	FMOVS 4(R5), F5
	FNEGS F5, F5

	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F6
	FMOVS 4(R7), F7

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F8
	FMOVS 4(R7), F9

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F10
	FMOVS 4(R7), F11

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F12
	FMOVS 4(R7), F13

	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

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

	FSUBS F21, F16, F26
	FADDS F20, F17, F27

	FADDS F21, F16, F28
	FSUBS F20, F17, F29

	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS F22, 0(R7)
	FMOVS F23, 4(R7)

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS F26, 0(R7)
	FMOVS F27, 4(R7)

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS F24, 0(R7)
	FMOVS F25, 4(R7)

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS F28, 0(R7)
	FMOVS F29, 4(R7)

	ADD  $1, R15, R15
	B    neon512m24_inv_stage2_j

neon512m24_inv_stage2_next:
	ADD  $16, R14, R14
	B    neon512m24_inv_stage2_base

neon512m24_inv_stage3:
	// Stage 3 with conjugated twiddles
	MOVD $0, R14

neon512m24_inv_stage3_base:
	CMP  $512, R14
	BGE  neon512m24_inv_stage4

	MOVD $0, R15

neon512m24_inv_stage3_j:
	CMP  $16, R15
	BGE  neon512m24_inv_stage3_next

	ADD  R14, R15, R0
	ADD  $16, R0, R1
	ADD  $32, R0, R2
	ADD  $48, R0, R3

	LSL  $3, R15, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1
	FNEGS F1, F1

	LSL  $4, R15, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F2
	FMOVS 4(R5), F3
	FNEGS F3, F3

	MOVD $24, R6
	MUL  R15, R6, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F4
	FMOVS 4(R5), F5
	FNEGS F5, F5

	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F6
	FMOVS 4(R7), F7

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F8
	FMOVS 4(R7), F9

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F10
	FMOVS 4(R7), F11

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F12
	FMOVS 4(R7), F13

	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

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

	FSUBS F21, F16, F26
	FADDS F20, F17, F27

	FADDS F21, F16, F28
	FSUBS F20, F17, F29

	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS F22, 0(R7)
	FMOVS F23, 4(R7)

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS F26, 0(R7)
	FMOVS F27, 4(R7)

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS F24, 0(R7)
	FMOVS F25, 4(R7)

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS F28, 0(R7)
	FMOVS F29, 4(R7)

	ADD  $1, R15, R15
	B    neon512m24_inv_stage3_j

neon512m24_inv_stage3_next:
	ADD  $64, R14, R14
	B    neon512m24_inv_stage3_base

neon512m24_inv_stage4:
	// Stage 4 with conjugated twiddles
	MOVD $0, R14

neon512m24_inv_stage4_base:
	CMP  $512, R14
	BGE  neon512m24_inv_stage5

	MOVD $0, R15

neon512m24_inv_stage4_j:
	CMP  $64, R15
	BGE  neon512m24_inv_stage4_next

	ADD  R14, R15, R0
	ADD  $64, R0, R1
	ADD  $128, R0, R2
	ADD  $192, R0, R3

	LSL  $1, R15, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1
	FNEGS F1, F1

	LSL  $2, R15, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F2
	FMOVS 4(R5), F3
	FNEGS F3, F3

	MOVD $6, R6
	MUL  R15, R6, R4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F4
	FMOVS 4(R5), F5
	FNEGS F5, F5

	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F6
	FMOVS 4(R7), F7

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F8
	FMOVS 4(R7), F9

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F10
	FMOVS 4(R7), F11

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F12
	FMOVS 4(R7), F13

	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

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

	FSUBS F21, F16, F26
	FADDS F20, F17, F27

	FADDS F21, F16, F28
	FSUBS F20, F17, F29

	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS F22, 0(R7)
	FMOVS F23, 4(R7)

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS F26, 0(R7)
	FMOVS F27, 4(R7)

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS F24, 0(R7)
	FMOVS F25, 4(R7)

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS F28, 0(R7)
	FMOVS F29, 4(R7)

	ADD  $1, R15, R15
	B    neon512m24_inv_stage4_j

neon512m24_inv_stage4_next:
	ADD  $256, R14, R14
	B    neon512m24_inv_stage4_base

neon512m24_inv_stage5:
	// Stage 5 with conjugated twiddles
	MOVD $0, R0

neon512m24_inv_stage5_loop:
	CMP  $256, R0
	BGE  neon512m24_inv_copy

	ADD  $256, R0, R1

	LSL  $3, R0, R2
	ADD  R10, R2, R2
	FMOVS 0(R2), F0
	FMOVS 4(R2), F1
	FNEGS F1, F1

	LSL  $3, R0, R2
	ADD  R8, R2, R2
	FMOVS 0(R2), F2
	FMOVS 4(R2), F3

	LSL  $3, R1, R2
	ADD  R8, R2, R2
	FMOVS 0(R2), F4
	FMOVS 4(R2), F5

	FMULS F0, F4, F6
	FMULS F1, F5, F7
	FSUBS F7, F6, F6
	FMULS F0, F5, F7
	FMULS F1, F4, F8
	FADDS F8, F7, F7

	FADDS F6, F2, F8
	FADDS F7, F3, F9
	FSUBS F6, F2, F10
	FSUBS F7, F3, F11

	LSL  $3, R0, R2
	ADD  R8, R2, R2
	FMOVS F8, 0(R2)
	FMOVS F9, 4(R2)

	LSL  $3, R1, R2
	ADD  R8, R2, R2
	FMOVS F10, 0(R2)
	FMOVS F11, 4(R2)

	ADD  $1, R0, R0
	B    neon512m24_inv_stage5_loop

neon512m24_inv_copy:
	CMP  R8, R20
	BEQ  neon512m24_inv_scale

	MOVD $0, R0
neon512m24_inv_copy_loop:
	CMP  $512, R0
	BGE  neon512m24_inv_scale
	LSL  $3, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon512m24_inv_copy_loop

neon512m24_inv_scale:
	MOVD $·neonInv512(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon512m24_inv_scale_loop:
	CMP  $512, R0
	BGE  neon512m24_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon512m24_inv_scale_loop

neon512m24_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon512m24_inv_return_false:
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Bit-reversal table for size 512 mixed-radix 4,4,4,4,2
// ===========================================================================
GLOBL bitrev_size512_mixed24<>(SB), RODATA, $4096
DATA bitrev_size512_mixed24<>+0x000(SB)/8, $0
DATA bitrev_size512_mixed24<>+0x008(SB)/8, $128
DATA bitrev_size512_mixed24<>+0x010(SB)/8, $256
DATA bitrev_size512_mixed24<>+0x018(SB)/8, $384
DATA bitrev_size512_mixed24<>+0x020(SB)/8, $32
DATA bitrev_size512_mixed24<>+0x028(SB)/8, $160
DATA bitrev_size512_mixed24<>+0x030(SB)/8, $288
DATA bitrev_size512_mixed24<>+0x038(SB)/8, $416
DATA bitrev_size512_mixed24<>+0x040(SB)/8, $64
DATA bitrev_size512_mixed24<>+0x048(SB)/8, $192
DATA bitrev_size512_mixed24<>+0x050(SB)/8, $320
DATA bitrev_size512_mixed24<>+0x058(SB)/8, $448
DATA bitrev_size512_mixed24<>+0x060(SB)/8, $96
DATA bitrev_size512_mixed24<>+0x068(SB)/8, $224
DATA bitrev_size512_mixed24<>+0x070(SB)/8, $352
DATA bitrev_size512_mixed24<>+0x078(SB)/8, $480
DATA bitrev_size512_mixed24<>+0x080(SB)/8, $8
DATA bitrev_size512_mixed24<>+0x088(SB)/8, $136
DATA bitrev_size512_mixed24<>+0x090(SB)/8, $264
DATA bitrev_size512_mixed24<>+0x098(SB)/8, $392
DATA bitrev_size512_mixed24<>+0x0A0(SB)/8, $40
DATA bitrev_size512_mixed24<>+0x0A8(SB)/8, $168
DATA bitrev_size512_mixed24<>+0x0B0(SB)/8, $296
DATA bitrev_size512_mixed24<>+0x0B8(SB)/8, $424
DATA bitrev_size512_mixed24<>+0x0C0(SB)/8, $72
DATA bitrev_size512_mixed24<>+0x0C8(SB)/8, $200
DATA bitrev_size512_mixed24<>+0x0D0(SB)/8, $328
DATA bitrev_size512_mixed24<>+0x0D8(SB)/8, $456
DATA bitrev_size512_mixed24<>+0x0E0(SB)/8, $104
DATA bitrev_size512_mixed24<>+0x0E8(SB)/8, $232
DATA bitrev_size512_mixed24<>+0x0F0(SB)/8, $360
DATA bitrev_size512_mixed24<>+0x0F8(SB)/8, $488
DATA bitrev_size512_mixed24<>+0x100(SB)/8, $16
DATA bitrev_size512_mixed24<>+0x108(SB)/8, $144
DATA bitrev_size512_mixed24<>+0x110(SB)/8, $272
DATA bitrev_size512_mixed24<>+0x118(SB)/8, $400
DATA bitrev_size512_mixed24<>+0x120(SB)/8, $48
DATA bitrev_size512_mixed24<>+0x128(SB)/8, $176
DATA bitrev_size512_mixed24<>+0x130(SB)/8, $304
DATA bitrev_size512_mixed24<>+0x138(SB)/8, $432
DATA bitrev_size512_mixed24<>+0x140(SB)/8, $80
DATA bitrev_size512_mixed24<>+0x148(SB)/8, $208
DATA bitrev_size512_mixed24<>+0x150(SB)/8, $336
DATA bitrev_size512_mixed24<>+0x158(SB)/8, $464
DATA bitrev_size512_mixed24<>+0x160(SB)/8, $112
DATA bitrev_size512_mixed24<>+0x168(SB)/8, $240
DATA bitrev_size512_mixed24<>+0x170(SB)/8, $368
DATA bitrev_size512_mixed24<>+0x178(SB)/8, $496
DATA bitrev_size512_mixed24<>+0x180(SB)/8, $24
DATA bitrev_size512_mixed24<>+0x188(SB)/8, $152
DATA bitrev_size512_mixed24<>+0x190(SB)/8, $280
DATA bitrev_size512_mixed24<>+0x198(SB)/8, $408
DATA bitrev_size512_mixed24<>+0x1A0(SB)/8, $56
DATA bitrev_size512_mixed24<>+0x1A8(SB)/8, $184
DATA bitrev_size512_mixed24<>+0x1B0(SB)/8, $312
DATA bitrev_size512_mixed24<>+0x1B8(SB)/8, $440
DATA bitrev_size512_mixed24<>+0x1C0(SB)/8, $88
DATA bitrev_size512_mixed24<>+0x1C8(SB)/8, $216
DATA bitrev_size512_mixed24<>+0x1D0(SB)/8, $344
DATA bitrev_size512_mixed24<>+0x1D8(SB)/8, $472
DATA bitrev_size512_mixed24<>+0x1E0(SB)/8, $120
DATA bitrev_size512_mixed24<>+0x1E8(SB)/8, $248
DATA bitrev_size512_mixed24<>+0x1F0(SB)/8, $376
DATA bitrev_size512_mixed24<>+0x1F8(SB)/8, $504
DATA bitrev_size512_mixed24<>+0x200(SB)/8, $2
DATA bitrev_size512_mixed24<>+0x208(SB)/8, $130
DATA bitrev_size512_mixed24<>+0x210(SB)/8, $258
DATA bitrev_size512_mixed24<>+0x218(SB)/8, $386
DATA bitrev_size512_mixed24<>+0x220(SB)/8, $34
DATA bitrev_size512_mixed24<>+0x228(SB)/8, $162
DATA bitrev_size512_mixed24<>+0x230(SB)/8, $290
DATA bitrev_size512_mixed24<>+0x238(SB)/8, $418
DATA bitrev_size512_mixed24<>+0x240(SB)/8, $66
DATA bitrev_size512_mixed24<>+0x248(SB)/8, $194
DATA bitrev_size512_mixed24<>+0x250(SB)/8, $322
DATA bitrev_size512_mixed24<>+0x258(SB)/8, $450
DATA bitrev_size512_mixed24<>+0x260(SB)/8, $98
DATA bitrev_size512_mixed24<>+0x268(SB)/8, $226
DATA bitrev_size512_mixed24<>+0x270(SB)/8, $354
DATA bitrev_size512_mixed24<>+0x278(SB)/8, $482
DATA bitrev_size512_mixed24<>+0x280(SB)/8, $10
DATA bitrev_size512_mixed24<>+0x288(SB)/8, $138
DATA bitrev_size512_mixed24<>+0x290(SB)/8, $266
DATA bitrev_size512_mixed24<>+0x298(SB)/8, $394
DATA bitrev_size512_mixed24<>+0x2A0(SB)/8, $42
DATA bitrev_size512_mixed24<>+0x2A8(SB)/8, $170
DATA bitrev_size512_mixed24<>+0x2B0(SB)/8, $298
DATA bitrev_size512_mixed24<>+0x2B8(SB)/8, $426
DATA bitrev_size512_mixed24<>+0x2C0(SB)/8, $74
DATA bitrev_size512_mixed24<>+0x2C8(SB)/8, $202
DATA bitrev_size512_mixed24<>+0x2D0(SB)/8, $330
DATA bitrev_size512_mixed24<>+0x2D8(SB)/8, $458
DATA bitrev_size512_mixed24<>+0x2E0(SB)/8, $106
DATA bitrev_size512_mixed24<>+0x2E8(SB)/8, $234
DATA bitrev_size512_mixed24<>+0x2F0(SB)/8, $362
DATA bitrev_size512_mixed24<>+0x2F8(SB)/8, $490
DATA bitrev_size512_mixed24<>+0x300(SB)/8, $18
DATA bitrev_size512_mixed24<>+0x308(SB)/8, $146
DATA bitrev_size512_mixed24<>+0x310(SB)/8, $274
DATA bitrev_size512_mixed24<>+0x318(SB)/8, $402
DATA bitrev_size512_mixed24<>+0x320(SB)/8, $50
DATA bitrev_size512_mixed24<>+0x328(SB)/8, $178
DATA bitrev_size512_mixed24<>+0x330(SB)/8, $306
DATA bitrev_size512_mixed24<>+0x338(SB)/8, $434
DATA bitrev_size512_mixed24<>+0x340(SB)/8, $82
DATA bitrev_size512_mixed24<>+0x348(SB)/8, $210
DATA bitrev_size512_mixed24<>+0x350(SB)/8, $338
DATA bitrev_size512_mixed24<>+0x358(SB)/8, $466
DATA bitrev_size512_mixed24<>+0x360(SB)/8, $114
DATA bitrev_size512_mixed24<>+0x368(SB)/8, $242
DATA bitrev_size512_mixed24<>+0x370(SB)/8, $370
DATA bitrev_size512_mixed24<>+0x378(SB)/8, $498
DATA bitrev_size512_mixed24<>+0x380(SB)/8, $26
DATA bitrev_size512_mixed24<>+0x388(SB)/8, $154
DATA bitrev_size512_mixed24<>+0x390(SB)/8, $282
DATA bitrev_size512_mixed24<>+0x398(SB)/8, $410
DATA bitrev_size512_mixed24<>+0x3A0(SB)/8, $58
DATA bitrev_size512_mixed24<>+0x3A8(SB)/8, $186
DATA bitrev_size512_mixed24<>+0x3B0(SB)/8, $314
DATA bitrev_size512_mixed24<>+0x3B8(SB)/8, $442
DATA bitrev_size512_mixed24<>+0x3C0(SB)/8, $90
DATA bitrev_size512_mixed24<>+0x3C8(SB)/8, $218
DATA bitrev_size512_mixed24<>+0x3D0(SB)/8, $346
DATA bitrev_size512_mixed24<>+0x3D8(SB)/8, $474
DATA bitrev_size512_mixed24<>+0x3E0(SB)/8, $122
DATA bitrev_size512_mixed24<>+0x3E8(SB)/8, $250
DATA bitrev_size512_mixed24<>+0x3F0(SB)/8, $378
DATA bitrev_size512_mixed24<>+0x3F8(SB)/8, $506
DATA bitrev_size512_mixed24<>+0x400(SB)/8, $4
DATA bitrev_size512_mixed24<>+0x408(SB)/8, $132
DATA bitrev_size512_mixed24<>+0x410(SB)/8, $260
DATA bitrev_size512_mixed24<>+0x418(SB)/8, $388
DATA bitrev_size512_mixed24<>+0x420(SB)/8, $36
DATA bitrev_size512_mixed24<>+0x428(SB)/8, $164
DATA bitrev_size512_mixed24<>+0x430(SB)/8, $292
DATA bitrev_size512_mixed24<>+0x438(SB)/8, $420
DATA bitrev_size512_mixed24<>+0x440(SB)/8, $68
DATA bitrev_size512_mixed24<>+0x448(SB)/8, $196
DATA bitrev_size512_mixed24<>+0x450(SB)/8, $324
DATA bitrev_size512_mixed24<>+0x458(SB)/8, $452
DATA bitrev_size512_mixed24<>+0x460(SB)/8, $100
DATA bitrev_size512_mixed24<>+0x468(SB)/8, $228
DATA bitrev_size512_mixed24<>+0x470(SB)/8, $356
DATA bitrev_size512_mixed24<>+0x478(SB)/8, $484
DATA bitrev_size512_mixed24<>+0x480(SB)/8, $12
DATA bitrev_size512_mixed24<>+0x488(SB)/8, $140
DATA bitrev_size512_mixed24<>+0x490(SB)/8, $268
DATA bitrev_size512_mixed24<>+0x498(SB)/8, $396
DATA bitrev_size512_mixed24<>+0x4A0(SB)/8, $44
DATA bitrev_size512_mixed24<>+0x4A8(SB)/8, $172
DATA bitrev_size512_mixed24<>+0x4B0(SB)/8, $300
DATA bitrev_size512_mixed24<>+0x4B8(SB)/8, $428
DATA bitrev_size512_mixed24<>+0x4C0(SB)/8, $76
DATA bitrev_size512_mixed24<>+0x4C8(SB)/8, $204
DATA bitrev_size512_mixed24<>+0x4D0(SB)/8, $332
DATA bitrev_size512_mixed24<>+0x4D8(SB)/8, $460
DATA bitrev_size512_mixed24<>+0x4E0(SB)/8, $108
DATA bitrev_size512_mixed24<>+0x4E8(SB)/8, $236
DATA bitrev_size512_mixed24<>+0x4F0(SB)/8, $364
DATA bitrev_size512_mixed24<>+0x4F8(SB)/8, $492
DATA bitrev_size512_mixed24<>+0x500(SB)/8, $20
DATA bitrev_size512_mixed24<>+0x508(SB)/8, $148
DATA bitrev_size512_mixed24<>+0x510(SB)/8, $276
DATA bitrev_size512_mixed24<>+0x518(SB)/8, $404
DATA bitrev_size512_mixed24<>+0x520(SB)/8, $52
DATA bitrev_size512_mixed24<>+0x528(SB)/8, $180
DATA bitrev_size512_mixed24<>+0x530(SB)/8, $308
DATA bitrev_size512_mixed24<>+0x538(SB)/8, $436
DATA bitrev_size512_mixed24<>+0x540(SB)/8, $84
DATA bitrev_size512_mixed24<>+0x548(SB)/8, $212
DATA bitrev_size512_mixed24<>+0x550(SB)/8, $340
DATA bitrev_size512_mixed24<>+0x558(SB)/8, $468
DATA bitrev_size512_mixed24<>+0x560(SB)/8, $116
DATA bitrev_size512_mixed24<>+0x568(SB)/8, $244
DATA bitrev_size512_mixed24<>+0x570(SB)/8, $372
DATA bitrev_size512_mixed24<>+0x578(SB)/8, $500
DATA bitrev_size512_mixed24<>+0x580(SB)/8, $28
DATA bitrev_size512_mixed24<>+0x588(SB)/8, $156
DATA bitrev_size512_mixed24<>+0x590(SB)/8, $284
DATA bitrev_size512_mixed24<>+0x598(SB)/8, $412
DATA bitrev_size512_mixed24<>+0x5A0(SB)/8, $60
DATA bitrev_size512_mixed24<>+0x5A8(SB)/8, $188
DATA bitrev_size512_mixed24<>+0x5B0(SB)/8, $316
DATA bitrev_size512_mixed24<>+0x5B8(SB)/8, $444
DATA bitrev_size512_mixed24<>+0x5C0(SB)/8, $92
DATA bitrev_size512_mixed24<>+0x5C8(SB)/8, $220
DATA bitrev_size512_mixed24<>+0x5D0(SB)/8, $348
DATA bitrev_size512_mixed24<>+0x5D8(SB)/8, $476
DATA bitrev_size512_mixed24<>+0x5E0(SB)/8, $124
DATA bitrev_size512_mixed24<>+0x5E8(SB)/8, $252
DATA bitrev_size512_mixed24<>+0x5F0(SB)/8, $380
DATA bitrev_size512_mixed24<>+0x5F8(SB)/8, $508
DATA bitrev_size512_mixed24<>+0x600(SB)/8, $6
DATA bitrev_size512_mixed24<>+0x608(SB)/8, $134
DATA bitrev_size512_mixed24<>+0x610(SB)/8, $262
DATA bitrev_size512_mixed24<>+0x618(SB)/8, $390
DATA bitrev_size512_mixed24<>+0x620(SB)/8, $38
DATA bitrev_size512_mixed24<>+0x628(SB)/8, $166
DATA bitrev_size512_mixed24<>+0x630(SB)/8, $294
DATA bitrev_size512_mixed24<>+0x638(SB)/8, $422
DATA bitrev_size512_mixed24<>+0x640(SB)/8, $70
DATA bitrev_size512_mixed24<>+0x648(SB)/8, $198
DATA bitrev_size512_mixed24<>+0x650(SB)/8, $326
DATA bitrev_size512_mixed24<>+0x658(SB)/8, $454
DATA bitrev_size512_mixed24<>+0x660(SB)/8, $102
DATA bitrev_size512_mixed24<>+0x668(SB)/8, $230
DATA bitrev_size512_mixed24<>+0x670(SB)/8, $358
DATA bitrev_size512_mixed24<>+0x678(SB)/8, $486
DATA bitrev_size512_mixed24<>+0x680(SB)/8, $14
DATA bitrev_size512_mixed24<>+0x688(SB)/8, $142
DATA bitrev_size512_mixed24<>+0x690(SB)/8, $270
DATA bitrev_size512_mixed24<>+0x698(SB)/8, $398
DATA bitrev_size512_mixed24<>+0x6A0(SB)/8, $46
DATA bitrev_size512_mixed24<>+0x6A8(SB)/8, $174
DATA bitrev_size512_mixed24<>+0x6B0(SB)/8, $302
DATA bitrev_size512_mixed24<>+0x6B8(SB)/8, $430
DATA bitrev_size512_mixed24<>+0x6C0(SB)/8, $78
DATA bitrev_size512_mixed24<>+0x6C8(SB)/8, $206
DATA bitrev_size512_mixed24<>+0x6D0(SB)/8, $334
DATA bitrev_size512_mixed24<>+0x6D8(SB)/8, $462
DATA bitrev_size512_mixed24<>+0x6E0(SB)/8, $110
DATA bitrev_size512_mixed24<>+0x6E8(SB)/8, $238
DATA bitrev_size512_mixed24<>+0x6F0(SB)/8, $366
DATA bitrev_size512_mixed24<>+0x6F8(SB)/8, $494
DATA bitrev_size512_mixed24<>+0x700(SB)/8, $22
DATA bitrev_size512_mixed24<>+0x708(SB)/8, $150
DATA bitrev_size512_mixed24<>+0x710(SB)/8, $278
DATA bitrev_size512_mixed24<>+0x718(SB)/8, $406
DATA bitrev_size512_mixed24<>+0x720(SB)/8, $54
DATA bitrev_size512_mixed24<>+0x728(SB)/8, $182
DATA bitrev_size512_mixed24<>+0x730(SB)/8, $310
DATA bitrev_size512_mixed24<>+0x738(SB)/8, $438
DATA bitrev_size512_mixed24<>+0x740(SB)/8, $86
DATA bitrev_size512_mixed24<>+0x748(SB)/8, $214
DATA bitrev_size512_mixed24<>+0x750(SB)/8, $342
DATA bitrev_size512_mixed24<>+0x758(SB)/8, $470
DATA bitrev_size512_mixed24<>+0x760(SB)/8, $118
DATA bitrev_size512_mixed24<>+0x768(SB)/8, $246
DATA bitrev_size512_mixed24<>+0x770(SB)/8, $374
DATA bitrev_size512_mixed24<>+0x778(SB)/8, $502
DATA bitrev_size512_mixed24<>+0x780(SB)/8, $30
DATA bitrev_size512_mixed24<>+0x788(SB)/8, $158
DATA bitrev_size512_mixed24<>+0x790(SB)/8, $286
DATA bitrev_size512_mixed24<>+0x798(SB)/8, $414
DATA bitrev_size512_mixed24<>+0x7A0(SB)/8, $62
DATA bitrev_size512_mixed24<>+0x7A8(SB)/8, $190
DATA bitrev_size512_mixed24<>+0x7B0(SB)/8, $318
DATA bitrev_size512_mixed24<>+0x7B8(SB)/8, $446
DATA bitrev_size512_mixed24<>+0x7C0(SB)/8, $94
DATA bitrev_size512_mixed24<>+0x7C8(SB)/8, $222
DATA bitrev_size512_mixed24<>+0x7D0(SB)/8, $350
DATA bitrev_size512_mixed24<>+0x7D8(SB)/8, $478
DATA bitrev_size512_mixed24<>+0x7E0(SB)/8, $126
DATA bitrev_size512_mixed24<>+0x7E8(SB)/8, $254
DATA bitrev_size512_mixed24<>+0x7F0(SB)/8, $382
DATA bitrev_size512_mixed24<>+0x7F8(SB)/8, $510
DATA bitrev_size512_mixed24<>+0x800(SB)/8, $1
DATA bitrev_size512_mixed24<>+0x808(SB)/8, $129
DATA bitrev_size512_mixed24<>+0x810(SB)/8, $257
DATA bitrev_size512_mixed24<>+0x818(SB)/8, $385
DATA bitrev_size512_mixed24<>+0x820(SB)/8, $33
DATA bitrev_size512_mixed24<>+0x828(SB)/8, $161
DATA bitrev_size512_mixed24<>+0x830(SB)/8, $289
DATA bitrev_size512_mixed24<>+0x838(SB)/8, $417
DATA bitrev_size512_mixed24<>+0x840(SB)/8, $65
DATA bitrev_size512_mixed24<>+0x848(SB)/8, $193
DATA bitrev_size512_mixed24<>+0x850(SB)/8, $321
DATA bitrev_size512_mixed24<>+0x858(SB)/8, $449
DATA bitrev_size512_mixed24<>+0x860(SB)/8, $97
DATA bitrev_size512_mixed24<>+0x868(SB)/8, $225
DATA bitrev_size512_mixed24<>+0x870(SB)/8, $353
DATA bitrev_size512_mixed24<>+0x878(SB)/8, $481
DATA bitrev_size512_mixed24<>+0x880(SB)/8, $9
DATA bitrev_size512_mixed24<>+0x888(SB)/8, $137
DATA bitrev_size512_mixed24<>+0x890(SB)/8, $265
DATA bitrev_size512_mixed24<>+0x898(SB)/8, $393
DATA bitrev_size512_mixed24<>+0x8A0(SB)/8, $41
DATA bitrev_size512_mixed24<>+0x8A8(SB)/8, $169
DATA bitrev_size512_mixed24<>+0x8B0(SB)/8, $297
DATA bitrev_size512_mixed24<>+0x8B8(SB)/8, $425
DATA bitrev_size512_mixed24<>+0x8C0(SB)/8, $73
DATA bitrev_size512_mixed24<>+0x8C8(SB)/8, $201
DATA bitrev_size512_mixed24<>+0x8D0(SB)/8, $329
DATA bitrev_size512_mixed24<>+0x8D8(SB)/8, $457
DATA bitrev_size512_mixed24<>+0x8E0(SB)/8, $105
DATA bitrev_size512_mixed24<>+0x8E8(SB)/8, $233
DATA bitrev_size512_mixed24<>+0x8F0(SB)/8, $361
DATA bitrev_size512_mixed24<>+0x8F8(SB)/8, $489
DATA bitrev_size512_mixed24<>+0x900(SB)/8, $17
DATA bitrev_size512_mixed24<>+0x908(SB)/8, $145
DATA bitrev_size512_mixed24<>+0x910(SB)/8, $273
DATA bitrev_size512_mixed24<>+0x918(SB)/8, $401
DATA bitrev_size512_mixed24<>+0x920(SB)/8, $49
DATA bitrev_size512_mixed24<>+0x928(SB)/8, $177
DATA bitrev_size512_mixed24<>+0x930(SB)/8, $305
DATA bitrev_size512_mixed24<>+0x938(SB)/8, $433
DATA bitrev_size512_mixed24<>+0x940(SB)/8, $81
DATA bitrev_size512_mixed24<>+0x948(SB)/8, $209
DATA bitrev_size512_mixed24<>+0x950(SB)/8, $337
DATA bitrev_size512_mixed24<>+0x958(SB)/8, $465
DATA bitrev_size512_mixed24<>+0x960(SB)/8, $113
DATA bitrev_size512_mixed24<>+0x968(SB)/8, $241
DATA bitrev_size512_mixed24<>+0x970(SB)/8, $369
DATA bitrev_size512_mixed24<>+0x978(SB)/8, $497
DATA bitrev_size512_mixed24<>+0x980(SB)/8, $25
DATA bitrev_size512_mixed24<>+0x988(SB)/8, $153
DATA bitrev_size512_mixed24<>+0x990(SB)/8, $281
DATA bitrev_size512_mixed24<>+0x998(SB)/8, $409
DATA bitrev_size512_mixed24<>+0x9A0(SB)/8, $57
DATA bitrev_size512_mixed24<>+0x9A8(SB)/8, $185
DATA bitrev_size512_mixed24<>+0x9B0(SB)/8, $313
DATA bitrev_size512_mixed24<>+0x9B8(SB)/8, $441
DATA bitrev_size512_mixed24<>+0x9C0(SB)/8, $89
DATA bitrev_size512_mixed24<>+0x9C8(SB)/8, $217
DATA bitrev_size512_mixed24<>+0x9D0(SB)/8, $345
DATA bitrev_size512_mixed24<>+0x9D8(SB)/8, $473
DATA bitrev_size512_mixed24<>+0x9E0(SB)/8, $121
DATA bitrev_size512_mixed24<>+0x9E8(SB)/8, $249
DATA bitrev_size512_mixed24<>+0x9F0(SB)/8, $377
DATA bitrev_size512_mixed24<>+0x9F8(SB)/8, $505
DATA bitrev_size512_mixed24<>+0xA00(SB)/8, $3
DATA bitrev_size512_mixed24<>+0xA08(SB)/8, $131
DATA bitrev_size512_mixed24<>+0xA10(SB)/8, $259
DATA bitrev_size512_mixed24<>+0xA18(SB)/8, $387
DATA bitrev_size512_mixed24<>+0xA20(SB)/8, $35
DATA bitrev_size512_mixed24<>+0xA28(SB)/8, $163
DATA bitrev_size512_mixed24<>+0xA30(SB)/8, $291
DATA bitrev_size512_mixed24<>+0xA38(SB)/8, $419
DATA bitrev_size512_mixed24<>+0xA40(SB)/8, $67
DATA bitrev_size512_mixed24<>+0xA48(SB)/8, $195
DATA bitrev_size512_mixed24<>+0xA50(SB)/8, $323
DATA bitrev_size512_mixed24<>+0xA58(SB)/8, $451
DATA bitrev_size512_mixed24<>+0xA60(SB)/8, $99
DATA bitrev_size512_mixed24<>+0xA68(SB)/8, $227
DATA bitrev_size512_mixed24<>+0xA70(SB)/8, $355
DATA bitrev_size512_mixed24<>+0xA78(SB)/8, $483
DATA bitrev_size512_mixed24<>+0xA80(SB)/8, $11
DATA bitrev_size512_mixed24<>+0xA88(SB)/8, $139
DATA bitrev_size512_mixed24<>+0xA90(SB)/8, $267
DATA bitrev_size512_mixed24<>+0xA98(SB)/8, $395
DATA bitrev_size512_mixed24<>+0xAA0(SB)/8, $43
DATA bitrev_size512_mixed24<>+0xAA8(SB)/8, $171
DATA bitrev_size512_mixed24<>+0xAB0(SB)/8, $299
DATA bitrev_size512_mixed24<>+0xAB8(SB)/8, $427
DATA bitrev_size512_mixed24<>+0xAC0(SB)/8, $75
DATA bitrev_size512_mixed24<>+0xAC8(SB)/8, $203
DATA bitrev_size512_mixed24<>+0xAD0(SB)/8, $331
DATA bitrev_size512_mixed24<>+0xAD8(SB)/8, $459
DATA bitrev_size512_mixed24<>+0xAE0(SB)/8, $107
DATA bitrev_size512_mixed24<>+0xAE8(SB)/8, $235
DATA bitrev_size512_mixed24<>+0xAF0(SB)/8, $363
DATA bitrev_size512_mixed24<>+0xAF8(SB)/8, $491
DATA bitrev_size512_mixed24<>+0xB00(SB)/8, $19
DATA bitrev_size512_mixed24<>+0xB08(SB)/8, $147
DATA bitrev_size512_mixed24<>+0xB10(SB)/8, $275
DATA bitrev_size512_mixed24<>+0xB18(SB)/8, $403
DATA bitrev_size512_mixed24<>+0xB20(SB)/8, $51
DATA bitrev_size512_mixed24<>+0xB28(SB)/8, $179
DATA bitrev_size512_mixed24<>+0xB30(SB)/8, $307
DATA bitrev_size512_mixed24<>+0xB38(SB)/8, $435
DATA bitrev_size512_mixed24<>+0xB40(SB)/8, $83
DATA bitrev_size512_mixed24<>+0xB48(SB)/8, $211
DATA bitrev_size512_mixed24<>+0xB50(SB)/8, $339
DATA bitrev_size512_mixed24<>+0xB58(SB)/8, $467
DATA bitrev_size512_mixed24<>+0xB60(SB)/8, $115
DATA bitrev_size512_mixed24<>+0xB68(SB)/8, $243
DATA bitrev_size512_mixed24<>+0xB70(SB)/8, $371
DATA bitrev_size512_mixed24<>+0xB78(SB)/8, $499
DATA bitrev_size512_mixed24<>+0xB80(SB)/8, $27
DATA bitrev_size512_mixed24<>+0xB88(SB)/8, $155
DATA bitrev_size512_mixed24<>+0xB90(SB)/8, $283
DATA bitrev_size512_mixed24<>+0xB98(SB)/8, $411
DATA bitrev_size512_mixed24<>+0xBA0(SB)/8, $59
DATA bitrev_size512_mixed24<>+0xBA8(SB)/8, $187
DATA bitrev_size512_mixed24<>+0xBB0(SB)/8, $315
DATA bitrev_size512_mixed24<>+0xBB8(SB)/8, $443
DATA bitrev_size512_mixed24<>+0xBC0(SB)/8, $91
DATA bitrev_size512_mixed24<>+0xBC8(SB)/8, $219
DATA bitrev_size512_mixed24<>+0xBD0(SB)/8, $347
DATA bitrev_size512_mixed24<>+0xBD8(SB)/8, $475
DATA bitrev_size512_mixed24<>+0xBE0(SB)/8, $123
DATA bitrev_size512_mixed24<>+0xBE8(SB)/8, $251
DATA bitrev_size512_mixed24<>+0xBF0(SB)/8, $379
DATA bitrev_size512_mixed24<>+0xBF8(SB)/8, $507
DATA bitrev_size512_mixed24<>+0xC00(SB)/8, $5
DATA bitrev_size512_mixed24<>+0xC08(SB)/8, $133
DATA bitrev_size512_mixed24<>+0xC10(SB)/8, $261
DATA bitrev_size512_mixed24<>+0xC18(SB)/8, $389
DATA bitrev_size512_mixed24<>+0xC20(SB)/8, $37
DATA bitrev_size512_mixed24<>+0xC28(SB)/8, $165
DATA bitrev_size512_mixed24<>+0xC30(SB)/8, $293
DATA bitrev_size512_mixed24<>+0xC38(SB)/8, $421
DATA bitrev_size512_mixed24<>+0xC40(SB)/8, $69
DATA bitrev_size512_mixed24<>+0xC48(SB)/8, $197
DATA bitrev_size512_mixed24<>+0xC50(SB)/8, $325
DATA bitrev_size512_mixed24<>+0xC58(SB)/8, $453
DATA bitrev_size512_mixed24<>+0xC60(SB)/8, $101
DATA bitrev_size512_mixed24<>+0xC68(SB)/8, $229
DATA bitrev_size512_mixed24<>+0xC70(SB)/8, $357
DATA bitrev_size512_mixed24<>+0xC78(SB)/8, $485
DATA bitrev_size512_mixed24<>+0xC80(SB)/8, $13
DATA bitrev_size512_mixed24<>+0xC88(SB)/8, $141
DATA bitrev_size512_mixed24<>+0xC90(SB)/8, $269
DATA bitrev_size512_mixed24<>+0xC98(SB)/8, $397
DATA bitrev_size512_mixed24<>+0xCA0(SB)/8, $45
DATA bitrev_size512_mixed24<>+0xCA8(SB)/8, $173
DATA bitrev_size512_mixed24<>+0xCB0(SB)/8, $301
DATA bitrev_size512_mixed24<>+0xCB8(SB)/8, $429
DATA bitrev_size512_mixed24<>+0xCC0(SB)/8, $77
DATA bitrev_size512_mixed24<>+0xCC8(SB)/8, $205
DATA bitrev_size512_mixed24<>+0xCD0(SB)/8, $333
DATA bitrev_size512_mixed24<>+0xCD8(SB)/8, $461
DATA bitrev_size512_mixed24<>+0xCE0(SB)/8, $109
DATA bitrev_size512_mixed24<>+0xCE8(SB)/8, $237
DATA bitrev_size512_mixed24<>+0xCF0(SB)/8, $365
DATA bitrev_size512_mixed24<>+0xCF8(SB)/8, $493
DATA bitrev_size512_mixed24<>+0xD00(SB)/8, $21
DATA bitrev_size512_mixed24<>+0xD08(SB)/8, $149
DATA bitrev_size512_mixed24<>+0xD10(SB)/8, $277
DATA bitrev_size512_mixed24<>+0xD18(SB)/8, $405
DATA bitrev_size512_mixed24<>+0xD20(SB)/8, $53
DATA bitrev_size512_mixed24<>+0xD28(SB)/8, $181
DATA bitrev_size512_mixed24<>+0xD30(SB)/8, $309
DATA bitrev_size512_mixed24<>+0xD38(SB)/8, $437
DATA bitrev_size512_mixed24<>+0xD40(SB)/8, $85
DATA bitrev_size512_mixed24<>+0xD48(SB)/8, $213
DATA bitrev_size512_mixed24<>+0xD50(SB)/8, $341
DATA bitrev_size512_mixed24<>+0xD58(SB)/8, $469
DATA bitrev_size512_mixed24<>+0xD60(SB)/8, $117
DATA bitrev_size512_mixed24<>+0xD68(SB)/8, $245
DATA bitrev_size512_mixed24<>+0xD70(SB)/8, $373
DATA bitrev_size512_mixed24<>+0xD78(SB)/8, $501
DATA bitrev_size512_mixed24<>+0xD80(SB)/8, $29
DATA bitrev_size512_mixed24<>+0xD88(SB)/8, $157
DATA bitrev_size512_mixed24<>+0xD90(SB)/8, $285
DATA bitrev_size512_mixed24<>+0xD98(SB)/8, $413
DATA bitrev_size512_mixed24<>+0xDA0(SB)/8, $61
DATA bitrev_size512_mixed24<>+0xDA8(SB)/8, $189
DATA bitrev_size512_mixed24<>+0xDB0(SB)/8, $317
DATA bitrev_size512_mixed24<>+0xDB8(SB)/8, $445
DATA bitrev_size512_mixed24<>+0xDC0(SB)/8, $93
DATA bitrev_size512_mixed24<>+0xDC8(SB)/8, $221
DATA bitrev_size512_mixed24<>+0xDD0(SB)/8, $349
DATA bitrev_size512_mixed24<>+0xDD8(SB)/8, $477
DATA bitrev_size512_mixed24<>+0xDE0(SB)/8, $125
DATA bitrev_size512_mixed24<>+0xDE8(SB)/8, $253
DATA bitrev_size512_mixed24<>+0xDF0(SB)/8, $381
DATA bitrev_size512_mixed24<>+0xDF8(SB)/8, $509
DATA bitrev_size512_mixed24<>+0xE00(SB)/8, $7
DATA bitrev_size512_mixed24<>+0xE08(SB)/8, $135
DATA bitrev_size512_mixed24<>+0xE10(SB)/8, $263
DATA bitrev_size512_mixed24<>+0xE18(SB)/8, $391
DATA bitrev_size512_mixed24<>+0xE20(SB)/8, $39
DATA bitrev_size512_mixed24<>+0xE28(SB)/8, $167
DATA bitrev_size512_mixed24<>+0xE30(SB)/8, $295
DATA bitrev_size512_mixed24<>+0xE38(SB)/8, $423
DATA bitrev_size512_mixed24<>+0xE40(SB)/8, $71
DATA bitrev_size512_mixed24<>+0xE48(SB)/8, $199
DATA bitrev_size512_mixed24<>+0xE50(SB)/8, $327
DATA bitrev_size512_mixed24<>+0xE58(SB)/8, $455
DATA bitrev_size512_mixed24<>+0xE60(SB)/8, $103
DATA bitrev_size512_mixed24<>+0xE68(SB)/8, $231
DATA bitrev_size512_mixed24<>+0xE70(SB)/8, $359
DATA bitrev_size512_mixed24<>+0xE78(SB)/8, $487
DATA bitrev_size512_mixed24<>+0xE80(SB)/8, $15
DATA bitrev_size512_mixed24<>+0xE88(SB)/8, $143
DATA bitrev_size512_mixed24<>+0xE90(SB)/8, $271
DATA bitrev_size512_mixed24<>+0xE98(SB)/8, $399
DATA bitrev_size512_mixed24<>+0xEA0(SB)/8, $47
DATA bitrev_size512_mixed24<>+0xEA8(SB)/8, $175
DATA bitrev_size512_mixed24<>+0xEB0(SB)/8, $303
DATA bitrev_size512_mixed24<>+0xEB8(SB)/8, $431
DATA bitrev_size512_mixed24<>+0xEC0(SB)/8, $79
DATA bitrev_size512_mixed24<>+0xEC8(SB)/8, $207
DATA bitrev_size512_mixed24<>+0xED0(SB)/8, $335
DATA bitrev_size512_mixed24<>+0xED8(SB)/8, $463
DATA bitrev_size512_mixed24<>+0xEE0(SB)/8, $111
DATA bitrev_size512_mixed24<>+0xEE8(SB)/8, $239
DATA bitrev_size512_mixed24<>+0xEF0(SB)/8, $367
DATA bitrev_size512_mixed24<>+0xEF8(SB)/8, $495
DATA bitrev_size512_mixed24<>+0xF00(SB)/8, $23
DATA bitrev_size512_mixed24<>+0xF08(SB)/8, $151
DATA bitrev_size512_mixed24<>+0xF10(SB)/8, $279
DATA bitrev_size512_mixed24<>+0xF18(SB)/8, $407
DATA bitrev_size512_mixed24<>+0xF20(SB)/8, $55
DATA bitrev_size512_mixed24<>+0xF28(SB)/8, $183
DATA bitrev_size512_mixed24<>+0xF30(SB)/8, $311
DATA bitrev_size512_mixed24<>+0xF38(SB)/8, $439
DATA bitrev_size512_mixed24<>+0xF40(SB)/8, $87
DATA bitrev_size512_mixed24<>+0xF48(SB)/8, $215
DATA bitrev_size512_mixed24<>+0xF50(SB)/8, $343
DATA bitrev_size512_mixed24<>+0xF58(SB)/8, $471
DATA bitrev_size512_mixed24<>+0xF60(SB)/8, $119
DATA bitrev_size512_mixed24<>+0xF68(SB)/8, $247
DATA bitrev_size512_mixed24<>+0xF70(SB)/8, $375
DATA bitrev_size512_mixed24<>+0xF78(SB)/8, $503
DATA bitrev_size512_mixed24<>+0xF80(SB)/8, $31
DATA bitrev_size512_mixed24<>+0xF88(SB)/8, $159
DATA bitrev_size512_mixed24<>+0xF90(SB)/8, $287
DATA bitrev_size512_mixed24<>+0xF98(SB)/8, $415
DATA bitrev_size512_mixed24<>+0xFA0(SB)/8, $63
DATA bitrev_size512_mixed24<>+0xFA8(SB)/8, $191
DATA bitrev_size512_mixed24<>+0xFB0(SB)/8, $319
DATA bitrev_size512_mixed24<>+0xFB8(SB)/8, $447
DATA bitrev_size512_mixed24<>+0xFC0(SB)/8, $95
DATA bitrev_size512_mixed24<>+0xFC8(SB)/8, $223
DATA bitrev_size512_mixed24<>+0xFD0(SB)/8, $351
DATA bitrev_size512_mixed24<>+0xFD8(SB)/8, $479
DATA bitrev_size512_mixed24<>+0xFE0(SB)/8, $127
DATA bitrev_size512_mixed24<>+0xFE8(SB)/8, $255
DATA bitrev_size512_mixed24<>+0xFF0(SB)/8, $383
DATA bitrev_size512_mixed24<>+0xFF8(SB)/8, $511

// Inverse scale constant: 1/512 in float32 = 0x3B000000
GLOBL ·neonInv512(SB), RODATA, $4
DATA ·neonInv512(SB)/4, $0x3b000000

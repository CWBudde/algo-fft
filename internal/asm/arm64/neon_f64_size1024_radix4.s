//go:build arm64 && !purego

// ===========================================================================
// NEON Size-1024 Radix-4 FFT Kernels for ARM64 (complex128)
// ===========================================================================
//
// Size 1024 = 4^5, radix-4 algorithm uses 5 stages:
//   Stage 1: 256 groups x 1 butterfly (no twiddles), stride=4
//   Stage 2: 64 groups x 4 butterflies, span=16,  twiddle step=64
//   Stage 3: 16 groups x 16 butterflies, span=64,  twiddle step=16
//   Stage 4: 4 groups x 64 butterflies, span=256, twiddle step=4
//   Stage 5: 1 group x 256 butterflies, span=1024, twiddle step=1
//
// Each complex128 element is 16 bytes (real f64 + imag f64).
//
// ===========================================================================

#include "textflag.h"

DATA ·neonInv1024R4F64+0(SB)/8, $0x3f50000000000000 // 1/1024 = 0.0009765625
GLOBL ·neonInv1024R4F64(SB), RODATA, $8

// Forward transform, size 1024, complex128, radix-4 variant
// func ForwardNEONSize1024Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool
TEXT ·ForwardNEONSize1024Radix4Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src_len+32(FP), R13

	CMP  $1024, R13
	BNE  neon1024r4f64_return_false

	MOVD dst_len+8(FP), R0
	CMP  $1024, R0
	BLT  neon1024r4f64_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $1024, R0
	BLT  neon1024r4f64_return_false

	MOVD scratch_len+80(FP), R0
	CMP  $1024, R0
	BLT  neon1024r4f64_return_false

	MOVD $bitrev_size1024_radix4_f64<>(SB), R12
	MOVD R8, R20

	CMP  R8, R9
	BNE  neon1024r4f64_use_dst
	MOVD R11, R8

neon1024r4f64_use_dst:
	// =========================================================================
	// Bit-reversal permutation
	// =========================================================================
	MOVD $0, R0

neon1024r4f64_bitrev_loop:
	CMP  $1024, R0
	BGE  neon1024r4f64_stage1

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
	B    neon1024r4f64_bitrev_loop

neon1024r4f64_stage1:
	// =========================================================================
	// Stage 1: 256 radix-4 butterflies (no twiddles)
	// =========================================================================
	MOVD $0, R14

neon1024r4f64_stage1_loop:
	CMP  $1024, R14
	BGE  neon1024r4f64_stage2

	LSL  $4, R14, R1
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

	FMOVD F15, F20
	FNEGD F14, F21

	FADDD F20, F10, F22
	FADDD F21, F11, F23

	FNEGD F15, F24
	FMOVD F14, F25

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

	ADD  $4, R14, R14
	B    neon1024r4f64_stage1_loop

neon1024r4f64_stage2:
	// =========================================================================
	// Stage 2: radix-4 with twiddles, span=16, twiddle step=64
	// 64 groups of 4 butterflies
	// =========================================================================
	MOVD $0, R14

neon1024r4f64_stage2_base:
	CMP  $1024, R14
	BGE  neon1024r4f64_stage3

	MOVD $0, R15

neon1024r4f64_stage2_j:
	CMP  $4, R15
	BGE  neon1024r4f64_stage2_next

	ADD  R14, R15, R0
	ADD  $4, R0, R1
	ADD  $8, R0, R2
	ADD  $12, R0, R3

	// Twiddles: w1=tw[j*64], w2=tw[j*128], w3=tw[j*192]
	LSL  $6, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1

	LSL  $7, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3

	MOVD $192, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5

	// Load values
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// Apply twiddles
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly
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

	FADDD F21, F16, F26
	FSUBD F20, F17, F27

	FSUBD F21, F16, F28
	FADDD F20, F17, F29

	// Store
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon1024r4f64_stage2_j

neon1024r4f64_stage2_next:
	ADD  $16, R14, R14
	B    neon1024r4f64_stage2_base

neon1024r4f64_stage3:
	// =========================================================================
	// Stage 3: radix-4 with twiddles, span=64, twiddle step=16
	// 16 groups of 16 butterflies
	// =========================================================================
	MOVD $0, R14

neon1024r4f64_stage3_base:
	CMP  $1024, R14
	BGE  neon1024r4f64_stage4

	MOVD $0, R15

neon1024r4f64_stage3_j:
	CMP  $16, R15
	BGE  neon1024r4f64_stage3_next

	ADD  R14, R15, R0
	ADD  $16, R0, R1
	ADD  $32, R0, R2
	ADD  $48, R0, R3

	// Twiddles: w1=tw[j*16], w2=tw[j*32], w3=tw[j*48]
	LSL  $4, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1

	LSL  $5, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3

	MOVD $48, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5

	// Load values
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// Apply twiddles
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly
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

	FADDD F21, F16, F26
	FSUBD F20, F17, F27

	FSUBD F21, F16, F28
	FADDD F20, F17, F29

	// Store
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon1024r4f64_stage3_j

neon1024r4f64_stage3_next:
	ADD  $64, R14, R14
	B    neon1024r4f64_stage3_base

neon1024r4f64_stage4:
	// =========================================================================
	// Stage 4: radix-4 with twiddles, span=256, twiddle step=4
	// 4 groups of 64 butterflies
	// =========================================================================
	MOVD $0, R14

neon1024r4f64_stage4_base:
	CMP  $1024, R14
	BGE  neon1024r4f64_stage5

	MOVD $0, R15

neon1024r4f64_stage4_j:
	CMP  $64, R15
	BGE  neon1024r4f64_stage4_next

	ADD  R14, R15, R0
	ADD  $64, R0, R1
	ADD  $128, R0, R2
	ADD  $192, R0, R3

	// Twiddles: w1=tw[j*4], w2=tw[j*8], w3=tw[j*12]
	LSL  $2, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1

	LSL  $3, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3

	MOVD $12, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5

	// Load values
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// Apply twiddles
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly
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

	FADDD F21, F16, F26
	FSUBD F20, F17, F27

	FSUBD F21, F16, F28
	FADDD F20, F17, F29

	// Store
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon1024r4f64_stage4_j

neon1024r4f64_stage4_next:
	ADD  $256, R14, R14
	B    neon1024r4f64_stage4_base

neon1024r4f64_stage5:
	// =========================================================================
	// Stage 5: radix-4 with twiddles, span=1024, twiddle step=1
	// 1 group of 256 butterflies
	// =========================================================================
	MOVD $0, R15

neon1024r4f64_stage5_loop:
	CMP  $256, R15
	BGE  neon1024r4f64_done

	MOVD R15, R0
	ADD  $256, R15, R1
	ADD  $512, R15, R2
	ADD  $768, R15, R3

	// Twiddles: w1=tw[j], w2=tw[2j], w3=tw[3j]
	LSL  $4, R15, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1

	LSL  $1, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3

	ADD  R4, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5

	// Load values
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// Apply twiddles
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly
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

	FADDD F21, F16, F26
	FSUBD F20, F17, F27

	FSUBD F21, F16, F28
	FADDD F20, F17, F29

	// Store
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon1024r4f64_stage5_loop

neon1024r4f64_done:
	CMP  R8, R20
	BEQ  neon1024r4f64_return_true

	MOVD $0, R0
neon1024r4f64_copy_loop:
	CMP  $1024, R0
	BGE  neon1024r4f64_return_true
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon1024r4f64_copy_loop

neon1024r4f64_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon1024r4f64_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform
// ===========================================================================
TEXT ·InverseNEONSize1024Radix4Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src_len+32(FP), R13

	CMP  $1024, R13
	BNE  neon1024r4f64_inv_return_false

	MOVD dst_len+8(FP), R0
	CMP  $1024, R0
	BLT  neon1024r4f64_inv_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $1024, R0
	BLT  neon1024r4f64_inv_return_false

	MOVD scratch_len+80(FP), R0
	CMP  $1024, R0
	BLT  neon1024r4f64_inv_return_false

	MOVD $bitrev_size1024_radix4_f64<>(SB), R12
	MOVD R8, R20

	CMP  R8, R9
	BNE  neon1024r4f64_inv_use_dst
	MOVD R11, R8

neon1024r4f64_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon1024r4f64_inv_bitrev_loop:
	CMP  $1024, R0
	BGE  neon1024r4f64_inv_stage1

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
	B    neon1024r4f64_inv_bitrev_loop

neon1024r4f64_inv_stage1:
	// Stage 1: 256 radix-4 butterflies (inverse)
	MOVD $0, R14

neon1024r4f64_inv_stage1_loop:
	CMP  $1024, R14
	BGE  neon1024r4f64_inv_stage2

	LSL  $4, R14, R1
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

	FMOVD F16, 0(R1)
	FMOVD F17, 8(R1)
	FMOVD F22, 16(R1)
	FMOVD F23, 24(R1)
	FMOVD F18, 32(R1)
	FMOVD F19, 40(R1)
	FMOVD F26, 48(R1)
	FMOVD F27, 56(R1)

	ADD  $4, R14, R14
	B    neon1024r4f64_inv_stage1_loop

neon1024r4f64_inv_stage2:
	// Stage 2: radix-4 with conjugated twiddles, span=16, twiddle step=64
	MOVD $0, R14

neon1024r4f64_inv_stage2_base:
	CMP  $1024, R14
	BGE  neon1024r4f64_inv_stage3

	MOVD $0, R15

neon1024r4f64_inv_stage2_j:
	CMP  $4, R15
	BGE  neon1024r4f64_inv_stage2_next

	ADD  R14, R15, R0
	ADD  $4, R0, R1
	ADD  $8, R0, R2
	ADD  $12, R0, R3

	LSL  $6, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1
	FNEGD F1, F1

	LSL  $7, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3
	FNEGD F3, F3

	MOVD $192, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5
	FNEGD F5, F5

	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

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

	FSUBD F21, F16, F26
	FADDD F20, F17, F27

	FADDD F21, F16, F28
	FSUBD F20, F17, F29

	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon1024r4f64_inv_stage2_j

neon1024r4f64_inv_stage2_next:
	ADD  $16, R14, R14
	B    neon1024r4f64_inv_stage2_base

neon1024r4f64_inv_stage3:
	// Stage 3: radix-4 with conjugated twiddles, span=64, twiddle step=16
	MOVD $0, R14

neon1024r4f64_inv_stage3_base:
	CMP  $1024, R14
	BGE  neon1024r4f64_inv_stage4

	MOVD $0, R15

neon1024r4f64_inv_stage3_j:
	CMP  $16, R15
	BGE  neon1024r4f64_inv_stage3_next

	ADD  R14, R15, R0
	ADD  $16, R0, R1
	ADD  $32, R0, R2
	ADD  $48, R0, R3

	LSL  $4, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1
	FNEGD F1, F1

	LSL  $5, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3
	FNEGD F3, F3

	MOVD $48, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5
	FNEGD F5, F5

	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

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

	FSUBD F21, F16, F26
	FADDD F20, F17, F27

	FADDD F21, F16, F28
	FSUBD F20, F17, F29

	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon1024r4f64_inv_stage3_j

neon1024r4f64_inv_stage3_next:
	ADD  $64, R14, R14
	B    neon1024r4f64_inv_stage3_base

neon1024r4f64_inv_stage4:
	// Stage 4: radix-4 with conjugated twiddles, span=256, twiddle step=4
	MOVD $0, R14

neon1024r4f64_inv_stage4_base:
	CMP  $1024, R14
	BGE  neon1024r4f64_inv_stage5

	MOVD $0, R15

neon1024r4f64_inv_stage4_j:
	CMP  $64, R15
	BGE  neon1024r4f64_inv_stage4_next

	ADD  R14, R15, R0
	ADD  $64, R0, R1
	ADD  $128, R0, R2
	ADD  $192, R0, R3

	LSL  $2, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1
	FNEGD F1, F1

	LSL  $3, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3
	FNEGD F3, F3

	MOVD $12, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5
	FNEGD F5, F5

	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

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

	FSUBD F21, F16, F26
	FADDD F20, F17, F27

	FADDD F21, F16, F28
	FSUBD F20, F17, F29

	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon1024r4f64_inv_stage4_j

neon1024r4f64_inv_stage4_next:
	ADD  $256, R14, R14
	B    neon1024r4f64_inv_stage4_base

neon1024r4f64_inv_stage5:
	// Stage 5: radix-4 with conjugated twiddles, span=1024, twiddle step=1
	MOVD $0, R15

neon1024r4f64_inv_stage5_loop:
	CMP  $256, R15
	BGE  neon1024r4f64_inv_copy

	MOVD R15, R0
	ADD  $256, R15, R1
	ADD  $512, R15, R2
	ADD  $768, R15, R3

	LSL  $4, R15, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1
	FNEGD F1, F1

	LSL  $1, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3
	FNEGD F3, F3

	ADD  R4, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5
	FNEGD F5, F5

	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

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

	FSUBD F21, F16, F26
	FADDD F20, F17, F27

	FADDD F21, F16, F28
	FSUBD F20, F17, F29

	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon1024r4f64_inv_stage5_loop

neon1024r4f64_inv_copy:
	CMP  R8, R20
	BEQ  neon1024r4f64_inv_scale

	MOVD $0, R0
neon1024r4f64_inv_copy_loop:
	CMP  $1024, R0
	BGE  neon1024r4f64_inv_scale
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon1024r4f64_inv_copy_loop

neon1024r4f64_inv_scale:
	MOVD $·neonInv1024R4F64(SB), R1
	FMOVD (R1), F0
	MOVD $0, R0

neon1024r4f64_inv_scale_loop:
	CMP  $1024, R0
	BGE  neon1024r4f64_inv_return_true
	LSL  $4, R0, R1
	ADD  R20, R1, R1
	FMOVD 0(R1), F2
	FMOVD 8(R1), F3
	FMULD F0, F2, F2
	FMULD F0, F3, F3
	FMOVD F2, 0(R1)
	FMOVD F3, 8(R1)
	ADD  $1, R0, R0
	B    neon1024r4f64_inv_scale_loop

neon1024r4f64_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon1024r4f64_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Bit-reversal table for size 1024 radix-4
// Reverses 5 base-4 digits
// ===========================================================================
DATA bitrev_size1024_radix4_f64<>+0x000(SB)/8, $0
DATA bitrev_size1024_radix4_f64<>+0x008(SB)/8, $256
DATA bitrev_size1024_radix4_f64<>+0x010(SB)/8, $512
DATA bitrev_size1024_radix4_f64<>+0x018(SB)/8, $768
DATA bitrev_size1024_radix4_f64<>+0x020(SB)/8, $64
DATA bitrev_size1024_radix4_f64<>+0x028(SB)/8, $320
DATA bitrev_size1024_radix4_f64<>+0x030(SB)/8, $576
DATA bitrev_size1024_radix4_f64<>+0x038(SB)/8, $832
DATA bitrev_size1024_radix4_f64<>+0x040(SB)/8, $128
DATA bitrev_size1024_radix4_f64<>+0x048(SB)/8, $384
DATA bitrev_size1024_radix4_f64<>+0x050(SB)/8, $640
DATA bitrev_size1024_radix4_f64<>+0x058(SB)/8, $896
DATA bitrev_size1024_radix4_f64<>+0x060(SB)/8, $192
DATA bitrev_size1024_radix4_f64<>+0x068(SB)/8, $448
DATA bitrev_size1024_radix4_f64<>+0x070(SB)/8, $704
DATA bitrev_size1024_radix4_f64<>+0x078(SB)/8, $960
DATA bitrev_size1024_radix4_f64<>+0x080(SB)/8, $16
DATA bitrev_size1024_radix4_f64<>+0x088(SB)/8, $272
DATA bitrev_size1024_radix4_f64<>+0x090(SB)/8, $528
DATA bitrev_size1024_radix4_f64<>+0x098(SB)/8, $784
DATA bitrev_size1024_radix4_f64<>+0x0A0(SB)/8, $80
DATA bitrev_size1024_radix4_f64<>+0x0A8(SB)/8, $336
DATA bitrev_size1024_radix4_f64<>+0x0B0(SB)/8, $592
DATA bitrev_size1024_radix4_f64<>+0x0B8(SB)/8, $848
DATA bitrev_size1024_radix4_f64<>+0x0C0(SB)/8, $144
DATA bitrev_size1024_radix4_f64<>+0x0C8(SB)/8, $400
DATA bitrev_size1024_radix4_f64<>+0x0D0(SB)/8, $656
DATA bitrev_size1024_radix4_f64<>+0x0D8(SB)/8, $912
DATA bitrev_size1024_radix4_f64<>+0x0E0(SB)/8, $208
DATA bitrev_size1024_radix4_f64<>+0x0E8(SB)/8, $464
DATA bitrev_size1024_radix4_f64<>+0x0F0(SB)/8, $720
DATA bitrev_size1024_radix4_f64<>+0x0F8(SB)/8, $976
DATA bitrev_size1024_radix4_f64<>+0x100(SB)/8, $32
DATA bitrev_size1024_radix4_f64<>+0x108(SB)/8, $288
DATA bitrev_size1024_radix4_f64<>+0x110(SB)/8, $544
DATA bitrev_size1024_radix4_f64<>+0x118(SB)/8, $800
DATA bitrev_size1024_radix4_f64<>+0x120(SB)/8, $96
DATA bitrev_size1024_radix4_f64<>+0x128(SB)/8, $352
DATA bitrev_size1024_radix4_f64<>+0x130(SB)/8, $608
DATA bitrev_size1024_radix4_f64<>+0x138(SB)/8, $864
DATA bitrev_size1024_radix4_f64<>+0x140(SB)/8, $160
DATA bitrev_size1024_radix4_f64<>+0x148(SB)/8, $416
DATA bitrev_size1024_radix4_f64<>+0x150(SB)/8, $672
DATA bitrev_size1024_radix4_f64<>+0x158(SB)/8, $928
DATA bitrev_size1024_radix4_f64<>+0x160(SB)/8, $224
DATA bitrev_size1024_radix4_f64<>+0x168(SB)/8, $480
DATA bitrev_size1024_radix4_f64<>+0x170(SB)/8, $736
DATA bitrev_size1024_radix4_f64<>+0x178(SB)/8, $992
DATA bitrev_size1024_radix4_f64<>+0x180(SB)/8, $48
DATA bitrev_size1024_radix4_f64<>+0x188(SB)/8, $304
DATA bitrev_size1024_radix4_f64<>+0x190(SB)/8, $560
DATA bitrev_size1024_radix4_f64<>+0x198(SB)/8, $816
DATA bitrev_size1024_radix4_f64<>+0x1A0(SB)/8, $112
DATA bitrev_size1024_radix4_f64<>+0x1A8(SB)/8, $368
DATA bitrev_size1024_radix4_f64<>+0x1B0(SB)/8, $624
DATA bitrev_size1024_radix4_f64<>+0x1B8(SB)/8, $880
DATA bitrev_size1024_radix4_f64<>+0x1C0(SB)/8, $176
DATA bitrev_size1024_radix4_f64<>+0x1C8(SB)/8, $432
DATA bitrev_size1024_radix4_f64<>+0x1D0(SB)/8, $688
DATA bitrev_size1024_radix4_f64<>+0x1D8(SB)/8, $944
DATA bitrev_size1024_radix4_f64<>+0x1E0(SB)/8, $240
DATA bitrev_size1024_radix4_f64<>+0x1E8(SB)/8, $496
DATA bitrev_size1024_radix4_f64<>+0x1F0(SB)/8, $752
DATA bitrev_size1024_radix4_f64<>+0x1F8(SB)/8, $1008
DATA bitrev_size1024_radix4_f64<>+0x200(SB)/8, $4
DATA bitrev_size1024_radix4_f64<>+0x208(SB)/8, $260
DATA bitrev_size1024_radix4_f64<>+0x210(SB)/8, $516
DATA bitrev_size1024_radix4_f64<>+0x218(SB)/8, $772
DATA bitrev_size1024_radix4_f64<>+0x220(SB)/8, $68
DATA bitrev_size1024_radix4_f64<>+0x228(SB)/8, $324
DATA bitrev_size1024_radix4_f64<>+0x230(SB)/8, $580
DATA bitrev_size1024_radix4_f64<>+0x238(SB)/8, $836
DATA bitrev_size1024_radix4_f64<>+0x240(SB)/8, $132
DATA bitrev_size1024_radix4_f64<>+0x248(SB)/8, $388
DATA bitrev_size1024_radix4_f64<>+0x250(SB)/8, $644
DATA bitrev_size1024_radix4_f64<>+0x258(SB)/8, $900
DATA bitrev_size1024_radix4_f64<>+0x260(SB)/8, $196
DATA bitrev_size1024_radix4_f64<>+0x268(SB)/8, $452
DATA bitrev_size1024_radix4_f64<>+0x270(SB)/8, $708
DATA bitrev_size1024_radix4_f64<>+0x278(SB)/8, $964
DATA bitrev_size1024_radix4_f64<>+0x280(SB)/8, $20
DATA bitrev_size1024_radix4_f64<>+0x288(SB)/8, $276
DATA bitrev_size1024_radix4_f64<>+0x290(SB)/8, $532
DATA bitrev_size1024_radix4_f64<>+0x298(SB)/8, $788
DATA bitrev_size1024_radix4_f64<>+0x2A0(SB)/8, $84
DATA bitrev_size1024_radix4_f64<>+0x2A8(SB)/8, $340
DATA bitrev_size1024_radix4_f64<>+0x2B0(SB)/8, $596
DATA bitrev_size1024_radix4_f64<>+0x2B8(SB)/8, $852
DATA bitrev_size1024_radix4_f64<>+0x2C0(SB)/8, $148
DATA bitrev_size1024_radix4_f64<>+0x2C8(SB)/8, $404
DATA bitrev_size1024_radix4_f64<>+0x2D0(SB)/8, $660
DATA bitrev_size1024_radix4_f64<>+0x2D8(SB)/8, $916
DATA bitrev_size1024_radix4_f64<>+0x2E0(SB)/8, $212
DATA bitrev_size1024_radix4_f64<>+0x2E8(SB)/8, $468
DATA bitrev_size1024_radix4_f64<>+0x2F0(SB)/8, $724
DATA bitrev_size1024_radix4_f64<>+0x2F8(SB)/8, $980
DATA bitrev_size1024_radix4_f64<>+0x300(SB)/8, $36
DATA bitrev_size1024_radix4_f64<>+0x308(SB)/8, $292
DATA bitrev_size1024_radix4_f64<>+0x310(SB)/8, $548
DATA bitrev_size1024_radix4_f64<>+0x318(SB)/8, $804
DATA bitrev_size1024_radix4_f64<>+0x320(SB)/8, $100
DATA bitrev_size1024_radix4_f64<>+0x328(SB)/8, $356
DATA bitrev_size1024_radix4_f64<>+0x330(SB)/8, $612
DATA bitrev_size1024_radix4_f64<>+0x338(SB)/8, $868
DATA bitrev_size1024_radix4_f64<>+0x340(SB)/8, $164
DATA bitrev_size1024_radix4_f64<>+0x348(SB)/8, $420
DATA bitrev_size1024_radix4_f64<>+0x350(SB)/8, $676
DATA bitrev_size1024_radix4_f64<>+0x358(SB)/8, $932
DATA bitrev_size1024_radix4_f64<>+0x360(SB)/8, $228
DATA bitrev_size1024_radix4_f64<>+0x368(SB)/8, $484
DATA bitrev_size1024_radix4_f64<>+0x370(SB)/8, $740
DATA bitrev_size1024_radix4_f64<>+0x378(SB)/8, $996
DATA bitrev_size1024_radix4_f64<>+0x380(SB)/8, $52
DATA bitrev_size1024_radix4_f64<>+0x388(SB)/8, $308
DATA bitrev_size1024_radix4_f64<>+0x390(SB)/8, $564
DATA bitrev_size1024_radix4_f64<>+0x398(SB)/8, $820
DATA bitrev_size1024_radix4_f64<>+0x3A0(SB)/8, $116
DATA bitrev_size1024_radix4_f64<>+0x3A8(SB)/8, $372
DATA bitrev_size1024_radix4_f64<>+0x3B0(SB)/8, $628
DATA bitrev_size1024_radix4_f64<>+0x3B8(SB)/8, $884
DATA bitrev_size1024_radix4_f64<>+0x3C0(SB)/8, $180
DATA bitrev_size1024_radix4_f64<>+0x3C8(SB)/8, $436
DATA bitrev_size1024_radix4_f64<>+0x3D0(SB)/8, $692
DATA bitrev_size1024_radix4_f64<>+0x3D8(SB)/8, $948
DATA bitrev_size1024_radix4_f64<>+0x3E0(SB)/8, $244
DATA bitrev_size1024_radix4_f64<>+0x3E8(SB)/8, $500
DATA bitrev_size1024_radix4_f64<>+0x3F0(SB)/8, $756
DATA bitrev_size1024_radix4_f64<>+0x3F8(SB)/8, $1012
DATA bitrev_size1024_radix4_f64<>+0x400(SB)/8, $8
DATA bitrev_size1024_radix4_f64<>+0x408(SB)/8, $264
DATA bitrev_size1024_radix4_f64<>+0x410(SB)/8, $520
DATA bitrev_size1024_radix4_f64<>+0x418(SB)/8, $776
DATA bitrev_size1024_radix4_f64<>+0x420(SB)/8, $72
DATA bitrev_size1024_radix4_f64<>+0x428(SB)/8, $328
DATA bitrev_size1024_radix4_f64<>+0x430(SB)/8, $584
DATA bitrev_size1024_radix4_f64<>+0x438(SB)/8, $840
DATA bitrev_size1024_radix4_f64<>+0x440(SB)/8, $136
DATA bitrev_size1024_radix4_f64<>+0x448(SB)/8, $392
DATA bitrev_size1024_radix4_f64<>+0x450(SB)/8, $648
DATA bitrev_size1024_radix4_f64<>+0x458(SB)/8, $904
DATA bitrev_size1024_radix4_f64<>+0x460(SB)/8, $200
DATA bitrev_size1024_radix4_f64<>+0x468(SB)/8, $456
DATA bitrev_size1024_radix4_f64<>+0x470(SB)/8, $712
DATA bitrev_size1024_radix4_f64<>+0x478(SB)/8, $968
DATA bitrev_size1024_radix4_f64<>+0x480(SB)/8, $24
DATA bitrev_size1024_radix4_f64<>+0x488(SB)/8, $280
DATA bitrev_size1024_radix4_f64<>+0x490(SB)/8, $536
DATA bitrev_size1024_radix4_f64<>+0x498(SB)/8, $792
DATA bitrev_size1024_radix4_f64<>+0x4A0(SB)/8, $88
DATA bitrev_size1024_radix4_f64<>+0x4A8(SB)/8, $344
DATA bitrev_size1024_radix4_f64<>+0x4B0(SB)/8, $600
DATA bitrev_size1024_radix4_f64<>+0x4B8(SB)/8, $856
DATA bitrev_size1024_radix4_f64<>+0x4C0(SB)/8, $152
DATA bitrev_size1024_radix4_f64<>+0x4C8(SB)/8, $408
DATA bitrev_size1024_radix4_f64<>+0x4D0(SB)/8, $664
DATA bitrev_size1024_radix4_f64<>+0x4D8(SB)/8, $920
DATA bitrev_size1024_radix4_f64<>+0x4E0(SB)/8, $216
DATA bitrev_size1024_radix4_f64<>+0x4E8(SB)/8, $472
DATA bitrev_size1024_radix4_f64<>+0x4F0(SB)/8, $728
DATA bitrev_size1024_radix4_f64<>+0x4F8(SB)/8, $984
DATA bitrev_size1024_radix4_f64<>+0x500(SB)/8, $40
DATA bitrev_size1024_radix4_f64<>+0x508(SB)/8, $296
DATA bitrev_size1024_radix4_f64<>+0x510(SB)/8, $552
DATA bitrev_size1024_radix4_f64<>+0x518(SB)/8, $808
DATA bitrev_size1024_radix4_f64<>+0x520(SB)/8, $104
DATA bitrev_size1024_radix4_f64<>+0x528(SB)/8, $360
DATA bitrev_size1024_radix4_f64<>+0x530(SB)/8, $616
DATA bitrev_size1024_radix4_f64<>+0x538(SB)/8, $872
DATA bitrev_size1024_radix4_f64<>+0x540(SB)/8, $168
DATA bitrev_size1024_radix4_f64<>+0x548(SB)/8, $424
DATA bitrev_size1024_radix4_f64<>+0x550(SB)/8, $680
DATA bitrev_size1024_radix4_f64<>+0x558(SB)/8, $936
DATA bitrev_size1024_radix4_f64<>+0x560(SB)/8, $232
DATA bitrev_size1024_radix4_f64<>+0x568(SB)/8, $488
DATA bitrev_size1024_radix4_f64<>+0x570(SB)/8, $744
DATA bitrev_size1024_radix4_f64<>+0x578(SB)/8, $1000
DATA bitrev_size1024_radix4_f64<>+0x580(SB)/8, $56
DATA bitrev_size1024_radix4_f64<>+0x588(SB)/8, $312
DATA bitrev_size1024_radix4_f64<>+0x590(SB)/8, $568
DATA bitrev_size1024_radix4_f64<>+0x598(SB)/8, $824
DATA bitrev_size1024_radix4_f64<>+0x5A0(SB)/8, $120
DATA bitrev_size1024_radix4_f64<>+0x5A8(SB)/8, $376
DATA bitrev_size1024_radix4_f64<>+0x5B0(SB)/8, $632
DATA bitrev_size1024_radix4_f64<>+0x5B8(SB)/8, $888
DATA bitrev_size1024_radix4_f64<>+0x5C0(SB)/8, $184
DATA bitrev_size1024_radix4_f64<>+0x5C8(SB)/8, $440
DATA bitrev_size1024_radix4_f64<>+0x5D0(SB)/8, $696
DATA bitrev_size1024_radix4_f64<>+0x5D8(SB)/8, $952
DATA bitrev_size1024_radix4_f64<>+0x5E0(SB)/8, $248
DATA bitrev_size1024_radix4_f64<>+0x5E8(SB)/8, $504
DATA bitrev_size1024_radix4_f64<>+0x5F0(SB)/8, $760
DATA bitrev_size1024_radix4_f64<>+0x5F8(SB)/8, $1016
DATA bitrev_size1024_radix4_f64<>+0x600(SB)/8, $12
DATA bitrev_size1024_radix4_f64<>+0x608(SB)/8, $268
DATA bitrev_size1024_radix4_f64<>+0x610(SB)/8, $524
DATA bitrev_size1024_radix4_f64<>+0x618(SB)/8, $780
DATA bitrev_size1024_radix4_f64<>+0x620(SB)/8, $76
DATA bitrev_size1024_radix4_f64<>+0x628(SB)/8, $332
DATA bitrev_size1024_radix4_f64<>+0x630(SB)/8, $588
DATA bitrev_size1024_radix4_f64<>+0x638(SB)/8, $844
DATA bitrev_size1024_radix4_f64<>+0x640(SB)/8, $140
DATA bitrev_size1024_radix4_f64<>+0x648(SB)/8, $396
DATA bitrev_size1024_radix4_f64<>+0x650(SB)/8, $652
DATA bitrev_size1024_radix4_f64<>+0x658(SB)/8, $908
DATA bitrev_size1024_radix4_f64<>+0x660(SB)/8, $204
DATA bitrev_size1024_radix4_f64<>+0x668(SB)/8, $460
DATA bitrev_size1024_radix4_f64<>+0x670(SB)/8, $716
DATA bitrev_size1024_radix4_f64<>+0x678(SB)/8, $972
DATA bitrev_size1024_radix4_f64<>+0x680(SB)/8, $28
DATA bitrev_size1024_radix4_f64<>+0x688(SB)/8, $284
DATA bitrev_size1024_radix4_f64<>+0x690(SB)/8, $540
DATA bitrev_size1024_radix4_f64<>+0x698(SB)/8, $796
DATA bitrev_size1024_radix4_f64<>+0x6A0(SB)/8, $92
DATA bitrev_size1024_radix4_f64<>+0x6A8(SB)/8, $348
DATA bitrev_size1024_radix4_f64<>+0x6B0(SB)/8, $604
DATA bitrev_size1024_radix4_f64<>+0x6B8(SB)/8, $860
DATA bitrev_size1024_radix4_f64<>+0x6C0(SB)/8, $156
DATA bitrev_size1024_radix4_f64<>+0x6C8(SB)/8, $412
DATA bitrev_size1024_radix4_f64<>+0x6D0(SB)/8, $668
DATA bitrev_size1024_radix4_f64<>+0x6D8(SB)/8, $924
DATA bitrev_size1024_radix4_f64<>+0x6E0(SB)/8, $220
DATA bitrev_size1024_radix4_f64<>+0x6E8(SB)/8, $476
DATA bitrev_size1024_radix4_f64<>+0x6F0(SB)/8, $732
DATA bitrev_size1024_radix4_f64<>+0x6F8(SB)/8, $988
DATA bitrev_size1024_radix4_f64<>+0x700(SB)/8, $44
DATA bitrev_size1024_radix4_f64<>+0x708(SB)/8, $300
DATA bitrev_size1024_radix4_f64<>+0x710(SB)/8, $556
DATA bitrev_size1024_radix4_f64<>+0x718(SB)/8, $812
DATA bitrev_size1024_radix4_f64<>+0x720(SB)/8, $108
DATA bitrev_size1024_radix4_f64<>+0x728(SB)/8, $364
DATA bitrev_size1024_radix4_f64<>+0x730(SB)/8, $620
DATA bitrev_size1024_radix4_f64<>+0x738(SB)/8, $876
DATA bitrev_size1024_radix4_f64<>+0x740(SB)/8, $172
DATA bitrev_size1024_radix4_f64<>+0x748(SB)/8, $428
DATA bitrev_size1024_radix4_f64<>+0x750(SB)/8, $684
DATA bitrev_size1024_radix4_f64<>+0x758(SB)/8, $940
DATA bitrev_size1024_radix4_f64<>+0x760(SB)/8, $236
DATA bitrev_size1024_radix4_f64<>+0x768(SB)/8, $492
DATA bitrev_size1024_radix4_f64<>+0x770(SB)/8, $748
DATA bitrev_size1024_radix4_f64<>+0x778(SB)/8, $1004
DATA bitrev_size1024_radix4_f64<>+0x780(SB)/8, $60
DATA bitrev_size1024_radix4_f64<>+0x788(SB)/8, $316
DATA bitrev_size1024_radix4_f64<>+0x790(SB)/8, $572
DATA bitrev_size1024_radix4_f64<>+0x798(SB)/8, $828
DATA bitrev_size1024_radix4_f64<>+0x7A0(SB)/8, $124
DATA bitrev_size1024_radix4_f64<>+0x7A8(SB)/8, $380
DATA bitrev_size1024_radix4_f64<>+0x7B0(SB)/8, $636
DATA bitrev_size1024_radix4_f64<>+0x7B8(SB)/8, $892
DATA bitrev_size1024_radix4_f64<>+0x7C0(SB)/8, $188
DATA bitrev_size1024_radix4_f64<>+0x7C8(SB)/8, $444
DATA bitrev_size1024_radix4_f64<>+0x7D0(SB)/8, $700
DATA bitrev_size1024_radix4_f64<>+0x7D8(SB)/8, $956
DATA bitrev_size1024_radix4_f64<>+0x7E0(SB)/8, $252
DATA bitrev_size1024_radix4_f64<>+0x7E8(SB)/8, $508
DATA bitrev_size1024_radix4_f64<>+0x7F0(SB)/8, $764
DATA bitrev_size1024_radix4_f64<>+0x7F8(SB)/8, $1020
DATA bitrev_size1024_radix4_f64<>+0x800(SB)/8, $1
DATA bitrev_size1024_radix4_f64<>+0x808(SB)/8, $257
DATA bitrev_size1024_radix4_f64<>+0x810(SB)/8, $513
DATA bitrev_size1024_radix4_f64<>+0x818(SB)/8, $769
DATA bitrev_size1024_radix4_f64<>+0x820(SB)/8, $65
DATA bitrev_size1024_radix4_f64<>+0x828(SB)/8, $321
DATA bitrev_size1024_radix4_f64<>+0x830(SB)/8, $577
DATA bitrev_size1024_radix4_f64<>+0x838(SB)/8, $833
DATA bitrev_size1024_radix4_f64<>+0x840(SB)/8, $129
DATA bitrev_size1024_radix4_f64<>+0x848(SB)/8, $385
DATA bitrev_size1024_radix4_f64<>+0x850(SB)/8, $641
DATA bitrev_size1024_radix4_f64<>+0x858(SB)/8, $897
DATA bitrev_size1024_radix4_f64<>+0x860(SB)/8, $193
DATA bitrev_size1024_radix4_f64<>+0x868(SB)/8, $449
DATA bitrev_size1024_radix4_f64<>+0x870(SB)/8, $705
DATA bitrev_size1024_radix4_f64<>+0x878(SB)/8, $961
DATA bitrev_size1024_radix4_f64<>+0x880(SB)/8, $17
DATA bitrev_size1024_radix4_f64<>+0x888(SB)/8, $273
DATA bitrev_size1024_radix4_f64<>+0x890(SB)/8, $529
DATA bitrev_size1024_radix4_f64<>+0x898(SB)/8, $785
DATA bitrev_size1024_radix4_f64<>+0x8A0(SB)/8, $81
DATA bitrev_size1024_radix4_f64<>+0x8A8(SB)/8, $337
DATA bitrev_size1024_radix4_f64<>+0x8B0(SB)/8, $593
DATA bitrev_size1024_radix4_f64<>+0x8B8(SB)/8, $849
DATA bitrev_size1024_radix4_f64<>+0x8C0(SB)/8, $145
DATA bitrev_size1024_radix4_f64<>+0x8C8(SB)/8, $401
DATA bitrev_size1024_radix4_f64<>+0x8D0(SB)/8, $657
DATA bitrev_size1024_radix4_f64<>+0x8D8(SB)/8, $913
DATA bitrev_size1024_radix4_f64<>+0x8E0(SB)/8, $209
DATA bitrev_size1024_radix4_f64<>+0x8E8(SB)/8, $465
DATA bitrev_size1024_radix4_f64<>+0x8F0(SB)/8, $721
DATA bitrev_size1024_radix4_f64<>+0x8F8(SB)/8, $977
DATA bitrev_size1024_radix4_f64<>+0x900(SB)/8, $33
DATA bitrev_size1024_radix4_f64<>+0x908(SB)/8, $289
DATA bitrev_size1024_radix4_f64<>+0x910(SB)/8, $545
DATA bitrev_size1024_radix4_f64<>+0x918(SB)/8, $801
DATA bitrev_size1024_radix4_f64<>+0x920(SB)/8, $97
DATA bitrev_size1024_radix4_f64<>+0x928(SB)/8, $353
DATA bitrev_size1024_radix4_f64<>+0x930(SB)/8, $609
DATA bitrev_size1024_radix4_f64<>+0x938(SB)/8, $865
DATA bitrev_size1024_radix4_f64<>+0x940(SB)/8, $161
DATA bitrev_size1024_radix4_f64<>+0x948(SB)/8, $417
DATA bitrev_size1024_radix4_f64<>+0x950(SB)/8, $673
DATA bitrev_size1024_radix4_f64<>+0x958(SB)/8, $929
DATA bitrev_size1024_radix4_f64<>+0x960(SB)/8, $225
DATA bitrev_size1024_radix4_f64<>+0x968(SB)/8, $481
DATA bitrev_size1024_radix4_f64<>+0x970(SB)/8, $737
DATA bitrev_size1024_radix4_f64<>+0x978(SB)/8, $993
DATA bitrev_size1024_radix4_f64<>+0x980(SB)/8, $49
DATA bitrev_size1024_radix4_f64<>+0x988(SB)/8, $305
DATA bitrev_size1024_radix4_f64<>+0x990(SB)/8, $561
DATA bitrev_size1024_radix4_f64<>+0x998(SB)/8, $817
DATA bitrev_size1024_radix4_f64<>+0x9A0(SB)/8, $113
DATA bitrev_size1024_radix4_f64<>+0x9A8(SB)/8, $369
DATA bitrev_size1024_radix4_f64<>+0x9B0(SB)/8, $625
DATA bitrev_size1024_radix4_f64<>+0x9B8(SB)/8, $881
DATA bitrev_size1024_radix4_f64<>+0x9C0(SB)/8, $177
DATA bitrev_size1024_radix4_f64<>+0x9C8(SB)/8, $433
DATA bitrev_size1024_radix4_f64<>+0x9D0(SB)/8, $689
DATA bitrev_size1024_radix4_f64<>+0x9D8(SB)/8, $945
DATA bitrev_size1024_radix4_f64<>+0x9E0(SB)/8, $241
DATA bitrev_size1024_radix4_f64<>+0x9E8(SB)/8, $497
DATA bitrev_size1024_radix4_f64<>+0x9F0(SB)/8, $753
DATA bitrev_size1024_radix4_f64<>+0x9F8(SB)/8, $1009
DATA bitrev_size1024_radix4_f64<>+0xA00(SB)/8, $5
DATA bitrev_size1024_radix4_f64<>+0xA08(SB)/8, $261
DATA bitrev_size1024_radix4_f64<>+0xA10(SB)/8, $517
DATA bitrev_size1024_radix4_f64<>+0xA18(SB)/8, $773
DATA bitrev_size1024_radix4_f64<>+0xA20(SB)/8, $69
DATA bitrev_size1024_radix4_f64<>+0xA28(SB)/8, $325
DATA bitrev_size1024_radix4_f64<>+0xA30(SB)/8, $581
DATA bitrev_size1024_radix4_f64<>+0xA38(SB)/8, $837
DATA bitrev_size1024_radix4_f64<>+0xA40(SB)/8, $133
DATA bitrev_size1024_radix4_f64<>+0xA48(SB)/8, $389
DATA bitrev_size1024_radix4_f64<>+0xA50(SB)/8, $645
DATA bitrev_size1024_radix4_f64<>+0xA58(SB)/8, $901
DATA bitrev_size1024_radix4_f64<>+0xA60(SB)/8, $197
DATA bitrev_size1024_radix4_f64<>+0xA68(SB)/8, $453
DATA bitrev_size1024_radix4_f64<>+0xA70(SB)/8, $709
DATA bitrev_size1024_radix4_f64<>+0xA78(SB)/8, $965
DATA bitrev_size1024_radix4_f64<>+0xA80(SB)/8, $21
DATA bitrev_size1024_radix4_f64<>+0xA88(SB)/8, $277
DATA bitrev_size1024_radix4_f64<>+0xA90(SB)/8, $533
DATA bitrev_size1024_radix4_f64<>+0xA98(SB)/8, $789
DATA bitrev_size1024_radix4_f64<>+0xAA0(SB)/8, $85
DATA bitrev_size1024_radix4_f64<>+0xAA8(SB)/8, $341
DATA bitrev_size1024_radix4_f64<>+0xAB0(SB)/8, $597
DATA bitrev_size1024_radix4_f64<>+0xAB8(SB)/8, $853
DATA bitrev_size1024_radix4_f64<>+0xAC0(SB)/8, $149
DATA bitrev_size1024_radix4_f64<>+0xAC8(SB)/8, $405
DATA bitrev_size1024_radix4_f64<>+0xAD0(SB)/8, $661
DATA bitrev_size1024_radix4_f64<>+0xAD8(SB)/8, $917
DATA bitrev_size1024_radix4_f64<>+0xAE0(SB)/8, $213
DATA bitrev_size1024_radix4_f64<>+0xAE8(SB)/8, $469
DATA bitrev_size1024_radix4_f64<>+0xAF0(SB)/8, $725
DATA bitrev_size1024_radix4_f64<>+0xAF8(SB)/8, $981
DATA bitrev_size1024_radix4_f64<>+0xB00(SB)/8, $37
DATA bitrev_size1024_radix4_f64<>+0xB08(SB)/8, $293
DATA bitrev_size1024_radix4_f64<>+0xB10(SB)/8, $549
DATA bitrev_size1024_radix4_f64<>+0xB18(SB)/8, $805
DATA bitrev_size1024_radix4_f64<>+0xB20(SB)/8, $101
DATA bitrev_size1024_radix4_f64<>+0xB28(SB)/8, $357
DATA bitrev_size1024_radix4_f64<>+0xB30(SB)/8, $613
DATA bitrev_size1024_radix4_f64<>+0xB38(SB)/8, $869
DATA bitrev_size1024_radix4_f64<>+0xB40(SB)/8, $165
DATA bitrev_size1024_radix4_f64<>+0xB48(SB)/8, $421
DATA bitrev_size1024_radix4_f64<>+0xB50(SB)/8, $677
DATA bitrev_size1024_radix4_f64<>+0xB58(SB)/8, $933
DATA bitrev_size1024_radix4_f64<>+0xB60(SB)/8, $229
DATA bitrev_size1024_radix4_f64<>+0xB68(SB)/8, $485
DATA bitrev_size1024_radix4_f64<>+0xB70(SB)/8, $741
DATA bitrev_size1024_radix4_f64<>+0xB78(SB)/8, $997
DATA bitrev_size1024_radix4_f64<>+0xB80(SB)/8, $53
DATA bitrev_size1024_radix4_f64<>+0xB88(SB)/8, $309
DATA bitrev_size1024_radix4_f64<>+0xB90(SB)/8, $565
DATA bitrev_size1024_radix4_f64<>+0xB98(SB)/8, $821
DATA bitrev_size1024_radix4_f64<>+0xBA0(SB)/8, $117
DATA bitrev_size1024_radix4_f64<>+0xBA8(SB)/8, $373
DATA bitrev_size1024_radix4_f64<>+0xBB0(SB)/8, $629
DATA bitrev_size1024_radix4_f64<>+0xBB8(SB)/8, $885
DATA bitrev_size1024_radix4_f64<>+0xBC0(SB)/8, $181
DATA bitrev_size1024_radix4_f64<>+0xBC8(SB)/8, $437
DATA bitrev_size1024_radix4_f64<>+0xBD0(SB)/8, $693
DATA bitrev_size1024_radix4_f64<>+0xBD8(SB)/8, $949
DATA bitrev_size1024_radix4_f64<>+0xBE0(SB)/8, $245
DATA bitrev_size1024_radix4_f64<>+0xBE8(SB)/8, $501
DATA bitrev_size1024_radix4_f64<>+0xBF0(SB)/8, $757
DATA bitrev_size1024_radix4_f64<>+0xBF8(SB)/8, $1013
DATA bitrev_size1024_radix4_f64<>+0xC00(SB)/8, $9
DATA bitrev_size1024_radix4_f64<>+0xC08(SB)/8, $265
DATA bitrev_size1024_radix4_f64<>+0xC10(SB)/8, $521
DATA bitrev_size1024_radix4_f64<>+0xC18(SB)/8, $777
DATA bitrev_size1024_radix4_f64<>+0xC20(SB)/8, $73
DATA bitrev_size1024_radix4_f64<>+0xC28(SB)/8, $329
DATA bitrev_size1024_radix4_f64<>+0xC30(SB)/8, $585
DATA bitrev_size1024_radix4_f64<>+0xC38(SB)/8, $841
DATA bitrev_size1024_radix4_f64<>+0xC40(SB)/8, $137
DATA bitrev_size1024_radix4_f64<>+0xC48(SB)/8, $393
DATA bitrev_size1024_radix4_f64<>+0xC50(SB)/8, $649
DATA bitrev_size1024_radix4_f64<>+0xC58(SB)/8, $905
DATA bitrev_size1024_radix4_f64<>+0xC60(SB)/8, $201
DATA bitrev_size1024_radix4_f64<>+0xC68(SB)/8, $457
DATA bitrev_size1024_radix4_f64<>+0xC70(SB)/8, $713
DATA bitrev_size1024_radix4_f64<>+0xC78(SB)/8, $969
DATA bitrev_size1024_radix4_f64<>+0xC80(SB)/8, $25
DATA bitrev_size1024_radix4_f64<>+0xC88(SB)/8, $281
DATA bitrev_size1024_radix4_f64<>+0xC90(SB)/8, $537
DATA bitrev_size1024_radix4_f64<>+0xC98(SB)/8, $793
DATA bitrev_size1024_radix4_f64<>+0xCA0(SB)/8, $89
DATA bitrev_size1024_radix4_f64<>+0xCA8(SB)/8, $345
DATA bitrev_size1024_radix4_f64<>+0xCB0(SB)/8, $601
DATA bitrev_size1024_radix4_f64<>+0xCB8(SB)/8, $857
DATA bitrev_size1024_radix4_f64<>+0xCC0(SB)/8, $153
DATA bitrev_size1024_radix4_f64<>+0xCC8(SB)/8, $409
DATA bitrev_size1024_radix4_f64<>+0xCD0(SB)/8, $665
DATA bitrev_size1024_radix4_f64<>+0xCD8(SB)/8, $921
DATA bitrev_size1024_radix4_f64<>+0xCE0(SB)/8, $217
DATA bitrev_size1024_radix4_f64<>+0xCE8(SB)/8, $473
DATA bitrev_size1024_radix4_f64<>+0xCF0(SB)/8, $729
DATA bitrev_size1024_radix4_f64<>+0xCF8(SB)/8, $985
DATA bitrev_size1024_radix4_f64<>+0xD00(SB)/8, $41
DATA bitrev_size1024_radix4_f64<>+0xD08(SB)/8, $297
DATA bitrev_size1024_radix4_f64<>+0xD10(SB)/8, $553
DATA bitrev_size1024_radix4_f64<>+0xD18(SB)/8, $809
DATA bitrev_size1024_radix4_f64<>+0xD20(SB)/8, $105
DATA bitrev_size1024_radix4_f64<>+0xD28(SB)/8, $361
DATA bitrev_size1024_radix4_f64<>+0xD30(SB)/8, $617
DATA bitrev_size1024_radix4_f64<>+0xD38(SB)/8, $873
DATA bitrev_size1024_radix4_f64<>+0xD40(SB)/8, $169
DATA bitrev_size1024_radix4_f64<>+0xD48(SB)/8, $425
DATA bitrev_size1024_radix4_f64<>+0xD50(SB)/8, $681
DATA bitrev_size1024_radix4_f64<>+0xD58(SB)/8, $937
DATA bitrev_size1024_radix4_f64<>+0xD60(SB)/8, $233
DATA bitrev_size1024_radix4_f64<>+0xD68(SB)/8, $489
DATA bitrev_size1024_radix4_f64<>+0xD70(SB)/8, $745
DATA bitrev_size1024_radix4_f64<>+0xD78(SB)/8, $1001
DATA bitrev_size1024_radix4_f64<>+0xD80(SB)/8, $57
DATA bitrev_size1024_radix4_f64<>+0xD88(SB)/8, $313
DATA bitrev_size1024_radix4_f64<>+0xD90(SB)/8, $569
DATA bitrev_size1024_radix4_f64<>+0xD98(SB)/8, $825
DATA bitrev_size1024_radix4_f64<>+0xDA0(SB)/8, $121
DATA bitrev_size1024_radix4_f64<>+0xDA8(SB)/8, $377
DATA bitrev_size1024_radix4_f64<>+0xDB0(SB)/8, $633
DATA bitrev_size1024_radix4_f64<>+0xDB8(SB)/8, $889
DATA bitrev_size1024_radix4_f64<>+0xDC0(SB)/8, $185
DATA bitrev_size1024_radix4_f64<>+0xDC8(SB)/8, $441
DATA bitrev_size1024_radix4_f64<>+0xDD0(SB)/8, $697
DATA bitrev_size1024_radix4_f64<>+0xDD8(SB)/8, $953
DATA bitrev_size1024_radix4_f64<>+0xDE0(SB)/8, $249
DATA bitrev_size1024_radix4_f64<>+0xDE8(SB)/8, $505
DATA bitrev_size1024_radix4_f64<>+0xDF0(SB)/8, $761
DATA bitrev_size1024_radix4_f64<>+0xDF8(SB)/8, $1017
DATA bitrev_size1024_radix4_f64<>+0xE00(SB)/8, $13
DATA bitrev_size1024_radix4_f64<>+0xE08(SB)/8, $269
DATA bitrev_size1024_radix4_f64<>+0xE10(SB)/8, $525
DATA bitrev_size1024_radix4_f64<>+0xE18(SB)/8, $781
DATA bitrev_size1024_radix4_f64<>+0xE20(SB)/8, $77
DATA bitrev_size1024_radix4_f64<>+0xE28(SB)/8, $333
DATA bitrev_size1024_radix4_f64<>+0xE30(SB)/8, $589
DATA bitrev_size1024_radix4_f64<>+0xE38(SB)/8, $845
DATA bitrev_size1024_radix4_f64<>+0xE40(SB)/8, $141
DATA bitrev_size1024_radix4_f64<>+0xE48(SB)/8, $397
DATA bitrev_size1024_radix4_f64<>+0xE50(SB)/8, $653
DATA bitrev_size1024_radix4_f64<>+0xE58(SB)/8, $909
DATA bitrev_size1024_radix4_f64<>+0xE60(SB)/8, $205
DATA bitrev_size1024_radix4_f64<>+0xE68(SB)/8, $461
DATA bitrev_size1024_radix4_f64<>+0xE70(SB)/8, $717
DATA bitrev_size1024_radix4_f64<>+0xE78(SB)/8, $973
DATA bitrev_size1024_radix4_f64<>+0xE80(SB)/8, $29
DATA bitrev_size1024_radix4_f64<>+0xE88(SB)/8, $285
DATA bitrev_size1024_radix4_f64<>+0xE90(SB)/8, $541
DATA bitrev_size1024_radix4_f64<>+0xE98(SB)/8, $797
DATA bitrev_size1024_radix4_f64<>+0xEA0(SB)/8, $93
DATA bitrev_size1024_radix4_f64<>+0xEA8(SB)/8, $349
DATA bitrev_size1024_radix4_f64<>+0xEB0(SB)/8, $605
DATA bitrev_size1024_radix4_f64<>+0xEB8(SB)/8, $861
DATA bitrev_size1024_radix4_f64<>+0xEC0(SB)/8, $157
DATA bitrev_size1024_radix4_f64<>+0xEC8(SB)/8, $413
DATA bitrev_size1024_radix4_f64<>+0xED0(SB)/8, $669
DATA bitrev_size1024_radix4_f64<>+0xED8(SB)/8, $925
DATA bitrev_size1024_radix4_f64<>+0xEE0(SB)/8, $221
DATA bitrev_size1024_radix4_f64<>+0xEE8(SB)/8, $477
DATA bitrev_size1024_radix4_f64<>+0xEF0(SB)/8, $733
DATA bitrev_size1024_radix4_f64<>+0xEF8(SB)/8, $989
DATA bitrev_size1024_radix4_f64<>+0xF00(SB)/8, $45
DATA bitrev_size1024_radix4_f64<>+0xF08(SB)/8, $301
DATA bitrev_size1024_radix4_f64<>+0xF10(SB)/8, $557
DATA bitrev_size1024_radix4_f64<>+0xF18(SB)/8, $813
DATA bitrev_size1024_radix4_f64<>+0xF20(SB)/8, $109
DATA bitrev_size1024_radix4_f64<>+0xF28(SB)/8, $365
DATA bitrev_size1024_radix4_f64<>+0xF30(SB)/8, $621
DATA bitrev_size1024_radix4_f64<>+0xF38(SB)/8, $877
DATA bitrev_size1024_radix4_f64<>+0xF40(SB)/8, $173
DATA bitrev_size1024_radix4_f64<>+0xF48(SB)/8, $429
DATA bitrev_size1024_radix4_f64<>+0xF50(SB)/8, $685
DATA bitrev_size1024_radix4_f64<>+0xF58(SB)/8, $941
DATA bitrev_size1024_radix4_f64<>+0xF60(SB)/8, $237
DATA bitrev_size1024_radix4_f64<>+0xF68(SB)/8, $493
DATA bitrev_size1024_radix4_f64<>+0xF70(SB)/8, $749
DATA bitrev_size1024_radix4_f64<>+0xF78(SB)/8, $1005
DATA bitrev_size1024_radix4_f64<>+0xF80(SB)/8, $61
DATA bitrev_size1024_radix4_f64<>+0xF88(SB)/8, $317
DATA bitrev_size1024_radix4_f64<>+0xF90(SB)/8, $573
DATA bitrev_size1024_radix4_f64<>+0xF98(SB)/8, $829
DATA bitrev_size1024_radix4_f64<>+0xFA0(SB)/8, $125
DATA bitrev_size1024_radix4_f64<>+0xFA8(SB)/8, $381
DATA bitrev_size1024_radix4_f64<>+0xFB0(SB)/8, $637
DATA bitrev_size1024_radix4_f64<>+0xFB8(SB)/8, $893
DATA bitrev_size1024_radix4_f64<>+0xFC0(SB)/8, $189
DATA bitrev_size1024_radix4_f64<>+0xFC8(SB)/8, $445
DATA bitrev_size1024_radix4_f64<>+0xFD0(SB)/8, $701
DATA bitrev_size1024_radix4_f64<>+0xFD8(SB)/8, $957
DATA bitrev_size1024_radix4_f64<>+0xFE0(SB)/8, $253
DATA bitrev_size1024_radix4_f64<>+0xFE8(SB)/8, $509
DATA bitrev_size1024_radix4_f64<>+0xFF0(SB)/8, $765
DATA bitrev_size1024_radix4_f64<>+0xFF8(SB)/8, $1021
DATA bitrev_size1024_radix4_f64<>+0x1000(SB)/8, $2
DATA bitrev_size1024_radix4_f64<>+0x1008(SB)/8, $258
DATA bitrev_size1024_radix4_f64<>+0x1010(SB)/8, $514
DATA bitrev_size1024_radix4_f64<>+0x1018(SB)/8, $770
DATA bitrev_size1024_radix4_f64<>+0x1020(SB)/8, $66
DATA bitrev_size1024_radix4_f64<>+0x1028(SB)/8, $322
DATA bitrev_size1024_radix4_f64<>+0x1030(SB)/8, $578
DATA bitrev_size1024_radix4_f64<>+0x1038(SB)/8, $834
DATA bitrev_size1024_radix4_f64<>+0x1040(SB)/8, $130
DATA bitrev_size1024_radix4_f64<>+0x1048(SB)/8, $386
DATA bitrev_size1024_radix4_f64<>+0x1050(SB)/8, $642
DATA bitrev_size1024_radix4_f64<>+0x1058(SB)/8, $898
DATA bitrev_size1024_radix4_f64<>+0x1060(SB)/8, $194
DATA bitrev_size1024_radix4_f64<>+0x1068(SB)/8, $450
DATA bitrev_size1024_radix4_f64<>+0x1070(SB)/8, $706
DATA bitrev_size1024_radix4_f64<>+0x1078(SB)/8, $962
DATA bitrev_size1024_radix4_f64<>+0x1080(SB)/8, $18
DATA bitrev_size1024_radix4_f64<>+0x1088(SB)/8, $274
DATA bitrev_size1024_radix4_f64<>+0x1090(SB)/8, $530
DATA bitrev_size1024_radix4_f64<>+0x1098(SB)/8, $786
DATA bitrev_size1024_radix4_f64<>+0x10A0(SB)/8, $82
DATA bitrev_size1024_radix4_f64<>+0x10A8(SB)/8, $338
DATA bitrev_size1024_radix4_f64<>+0x10B0(SB)/8, $594
DATA bitrev_size1024_radix4_f64<>+0x10B8(SB)/8, $850
DATA bitrev_size1024_radix4_f64<>+0x10C0(SB)/8, $146
DATA bitrev_size1024_radix4_f64<>+0x10C8(SB)/8, $402
DATA bitrev_size1024_radix4_f64<>+0x10D0(SB)/8, $658
DATA bitrev_size1024_radix4_f64<>+0x10D8(SB)/8, $914
DATA bitrev_size1024_radix4_f64<>+0x10E0(SB)/8, $210
DATA bitrev_size1024_radix4_f64<>+0x10E8(SB)/8, $466
DATA bitrev_size1024_radix4_f64<>+0x10F0(SB)/8, $722
DATA bitrev_size1024_radix4_f64<>+0x10F8(SB)/8, $978
DATA bitrev_size1024_radix4_f64<>+0x1100(SB)/8, $34
DATA bitrev_size1024_radix4_f64<>+0x1108(SB)/8, $290
DATA bitrev_size1024_radix4_f64<>+0x1110(SB)/8, $546
DATA bitrev_size1024_radix4_f64<>+0x1118(SB)/8, $802
DATA bitrev_size1024_radix4_f64<>+0x1120(SB)/8, $98
DATA bitrev_size1024_radix4_f64<>+0x1128(SB)/8, $354
DATA bitrev_size1024_radix4_f64<>+0x1130(SB)/8, $610
DATA bitrev_size1024_radix4_f64<>+0x1138(SB)/8, $866
DATA bitrev_size1024_radix4_f64<>+0x1140(SB)/8, $162
DATA bitrev_size1024_radix4_f64<>+0x1148(SB)/8, $418
DATA bitrev_size1024_radix4_f64<>+0x1150(SB)/8, $674
DATA bitrev_size1024_radix4_f64<>+0x1158(SB)/8, $930
DATA bitrev_size1024_radix4_f64<>+0x1160(SB)/8, $226
DATA bitrev_size1024_radix4_f64<>+0x1168(SB)/8, $482
DATA bitrev_size1024_radix4_f64<>+0x1170(SB)/8, $738
DATA bitrev_size1024_radix4_f64<>+0x1178(SB)/8, $994
DATA bitrev_size1024_radix4_f64<>+0x1180(SB)/8, $50
DATA bitrev_size1024_radix4_f64<>+0x1188(SB)/8, $306
DATA bitrev_size1024_radix4_f64<>+0x1190(SB)/8, $562
DATA bitrev_size1024_radix4_f64<>+0x1198(SB)/8, $818
DATA bitrev_size1024_radix4_f64<>+0x11A0(SB)/8, $114
DATA bitrev_size1024_radix4_f64<>+0x11A8(SB)/8, $370
DATA bitrev_size1024_radix4_f64<>+0x11B0(SB)/8, $626
DATA bitrev_size1024_radix4_f64<>+0x11B8(SB)/8, $882
DATA bitrev_size1024_radix4_f64<>+0x11C0(SB)/8, $178
DATA bitrev_size1024_radix4_f64<>+0x11C8(SB)/8, $434
DATA bitrev_size1024_radix4_f64<>+0x11D0(SB)/8, $690
DATA bitrev_size1024_radix4_f64<>+0x11D8(SB)/8, $946
DATA bitrev_size1024_radix4_f64<>+0x11E0(SB)/8, $242
DATA bitrev_size1024_radix4_f64<>+0x11E8(SB)/8, $498
DATA bitrev_size1024_radix4_f64<>+0x11F0(SB)/8, $754
DATA bitrev_size1024_radix4_f64<>+0x11F8(SB)/8, $1010
DATA bitrev_size1024_radix4_f64<>+0x1200(SB)/8, $6
DATA bitrev_size1024_radix4_f64<>+0x1208(SB)/8, $262
DATA bitrev_size1024_radix4_f64<>+0x1210(SB)/8, $518
DATA bitrev_size1024_radix4_f64<>+0x1218(SB)/8, $774
DATA bitrev_size1024_radix4_f64<>+0x1220(SB)/8, $70
DATA bitrev_size1024_radix4_f64<>+0x1228(SB)/8, $326
DATA bitrev_size1024_radix4_f64<>+0x1230(SB)/8, $582
DATA bitrev_size1024_radix4_f64<>+0x1238(SB)/8, $838
DATA bitrev_size1024_radix4_f64<>+0x1240(SB)/8, $134
DATA bitrev_size1024_radix4_f64<>+0x1248(SB)/8, $390
DATA bitrev_size1024_radix4_f64<>+0x1250(SB)/8, $646
DATA bitrev_size1024_radix4_f64<>+0x1258(SB)/8, $902
DATA bitrev_size1024_radix4_f64<>+0x1260(SB)/8, $198
DATA bitrev_size1024_radix4_f64<>+0x1268(SB)/8, $454
DATA bitrev_size1024_radix4_f64<>+0x1270(SB)/8, $710
DATA bitrev_size1024_radix4_f64<>+0x1278(SB)/8, $966
DATA bitrev_size1024_radix4_f64<>+0x1280(SB)/8, $22
DATA bitrev_size1024_radix4_f64<>+0x1288(SB)/8, $278
DATA bitrev_size1024_radix4_f64<>+0x1290(SB)/8, $534
DATA bitrev_size1024_radix4_f64<>+0x1298(SB)/8, $790
DATA bitrev_size1024_radix4_f64<>+0x12A0(SB)/8, $86
DATA bitrev_size1024_radix4_f64<>+0x12A8(SB)/8, $342
DATA bitrev_size1024_radix4_f64<>+0x12B0(SB)/8, $598
DATA bitrev_size1024_radix4_f64<>+0x12B8(SB)/8, $854
DATA bitrev_size1024_radix4_f64<>+0x12C0(SB)/8, $150
DATA bitrev_size1024_radix4_f64<>+0x12C8(SB)/8, $406
DATA bitrev_size1024_radix4_f64<>+0x12D0(SB)/8, $662
DATA bitrev_size1024_radix4_f64<>+0x12D8(SB)/8, $918
DATA bitrev_size1024_radix4_f64<>+0x12E0(SB)/8, $214
DATA bitrev_size1024_radix4_f64<>+0x12E8(SB)/8, $470
DATA bitrev_size1024_radix4_f64<>+0x12F0(SB)/8, $726
DATA bitrev_size1024_radix4_f64<>+0x12F8(SB)/8, $982
DATA bitrev_size1024_radix4_f64<>+0x1300(SB)/8, $38
DATA bitrev_size1024_radix4_f64<>+0x1308(SB)/8, $294
DATA bitrev_size1024_radix4_f64<>+0x1310(SB)/8, $550
DATA bitrev_size1024_radix4_f64<>+0x1318(SB)/8, $806
DATA bitrev_size1024_radix4_f64<>+0x1320(SB)/8, $102
DATA bitrev_size1024_radix4_f64<>+0x1328(SB)/8, $358
DATA bitrev_size1024_radix4_f64<>+0x1330(SB)/8, $614
DATA bitrev_size1024_radix4_f64<>+0x1338(SB)/8, $870
DATA bitrev_size1024_radix4_f64<>+0x1340(SB)/8, $166
DATA bitrev_size1024_radix4_f64<>+0x1348(SB)/8, $422
DATA bitrev_size1024_radix4_f64<>+0x1350(SB)/8, $678
DATA bitrev_size1024_radix4_f64<>+0x1358(SB)/8, $934
DATA bitrev_size1024_radix4_f64<>+0x1360(SB)/8, $230
DATA bitrev_size1024_radix4_f64<>+0x1368(SB)/8, $486
DATA bitrev_size1024_radix4_f64<>+0x1370(SB)/8, $742
DATA bitrev_size1024_radix4_f64<>+0x1378(SB)/8, $998
DATA bitrev_size1024_radix4_f64<>+0x1380(SB)/8, $54
DATA bitrev_size1024_radix4_f64<>+0x1388(SB)/8, $310
DATA bitrev_size1024_radix4_f64<>+0x1390(SB)/8, $566
DATA bitrev_size1024_radix4_f64<>+0x1398(SB)/8, $822
DATA bitrev_size1024_radix4_f64<>+0x13A0(SB)/8, $118
DATA bitrev_size1024_radix4_f64<>+0x13A8(SB)/8, $374
DATA bitrev_size1024_radix4_f64<>+0x13B0(SB)/8, $630
DATA bitrev_size1024_radix4_f64<>+0x13B8(SB)/8, $886
DATA bitrev_size1024_radix4_f64<>+0x13C0(SB)/8, $182
DATA bitrev_size1024_radix4_f64<>+0x13C8(SB)/8, $438
DATA bitrev_size1024_radix4_f64<>+0x13D0(SB)/8, $694
DATA bitrev_size1024_radix4_f64<>+0x13D8(SB)/8, $950
DATA bitrev_size1024_radix4_f64<>+0x13E0(SB)/8, $246
DATA bitrev_size1024_radix4_f64<>+0x13E8(SB)/8, $502
DATA bitrev_size1024_radix4_f64<>+0x13F0(SB)/8, $758
DATA bitrev_size1024_radix4_f64<>+0x13F8(SB)/8, $1014
DATA bitrev_size1024_radix4_f64<>+0x1400(SB)/8, $10
DATA bitrev_size1024_radix4_f64<>+0x1408(SB)/8, $266
DATA bitrev_size1024_radix4_f64<>+0x1410(SB)/8, $522
DATA bitrev_size1024_radix4_f64<>+0x1418(SB)/8, $778
DATA bitrev_size1024_radix4_f64<>+0x1420(SB)/8, $74
DATA bitrev_size1024_radix4_f64<>+0x1428(SB)/8, $330
DATA bitrev_size1024_radix4_f64<>+0x1430(SB)/8, $586
DATA bitrev_size1024_radix4_f64<>+0x1438(SB)/8, $842
DATA bitrev_size1024_radix4_f64<>+0x1440(SB)/8, $138
DATA bitrev_size1024_radix4_f64<>+0x1448(SB)/8, $394
DATA bitrev_size1024_radix4_f64<>+0x1450(SB)/8, $650
DATA bitrev_size1024_radix4_f64<>+0x1458(SB)/8, $906
DATA bitrev_size1024_radix4_f64<>+0x1460(SB)/8, $202
DATA bitrev_size1024_radix4_f64<>+0x1468(SB)/8, $458
DATA bitrev_size1024_radix4_f64<>+0x1470(SB)/8, $714
DATA bitrev_size1024_radix4_f64<>+0x1478(SB)/8, $970
DATA bitrev_size1024_radix4_f64<>+0x1480(SB)/8, $26
DATA bitrev_size1024_radix4_f64<>+0x1488(SB)/8, $282
DATA bitrev_size1024_radix4_f64<>+0x1490(SB)/8, $538
DATA bitrev_size1024_radix4_f64<>+0x1498(SB)/8, $794
DATA bitrev_size1024_radix4_f64<>+0x14A0(SB)/8, $90
DATA bitrev_size1024_radix4_f64<>+0x14A8(SB)/8, $346
DATA bitrev_size1024_radix4_f64<>+0x14B0(SB)/8, $602
DATA bitrev_size1024_radix4_f64<>+0x14B8(SB)/8, $858
DATA bitrev_size1024_radix4_f64<>+0x14C0(SB)/8, $154
DATA bitrev_size1024_radix4_f64<>+0x14C8(SB)/8, $410
DATA bitrev_size1024_radix4_f64<>+0x14D0(SB)/8, $666
DATA bitrev_size1024_radix4_f64<>+0x14D8(SB)/8, $922
DATA bitrev_size1024_radix4_f64<>+0x14E0(SB)/8, $218
DATA bitrev_size1024_radix4_f64<>+0x14E8(SB)/8, $474
DATA bitrev_size1024_radix4_f64<>+0x14F0(SB)/8, $730
DATA bitrev_size1024_radix4_f64<>+0x14F8(SB)/8, $986
DATA bitrev_size1024_radix4_f64<>+0x1500(SB)/8, $42
DATA bitrev_size1024_radix4_f64<>+0x1508(SB)/8, $298
DATA bitrev_size1024_radix4_f64<>+0x1510(SB)/8, $554
DATA bitrev_size1024_radix4_f64<>+0x1518(SB)/8, $810
DATA bitrev_size1024_radix4_f64<>+0x1520(SB)/8, $106
DATA bitrev_size1024_radix4_f64<>+0x1528(SB)/8, $362
DATA bitrev_size1024_radix4_f64<>+0x1530(SB)/8, $618
DATA bitrev_size1024_radix4_f64<>+0x1538(SB)/8, $874
DATA bitrev_size1024_radix4_f64<>+0x1540(SB)/8, $170
DATA bitrev_size1024_radix4_f64<>+0x1548(SB)/8, $426
DATA bitrev_size1024_radix4_f64<>+0x1550(SB)/8, $682
DATA bitrev_size1024_radix4_f64<>+0x1558(SB)/8, $938
DATA bitrev_size1024_radix4_f64<>+0x1560(SB)/8, $234
DATA bitrev_size1024_radix4_f64<>+0x1568(SB)/8, $490
DATA bitrev_size1024_radix4_f64<>+0x1570(SB)/8, $746
DATA bitrev_size1024_radix4_f64<>+0x1578(SB)/8, $1002
DATA bitrev_size1024_radix4_f64<>+0x1580(SB)/8, $58
DATA bitrev_size1024_radix4_f64<>+0x1588(SB)/8, $314
DATA bitrev_size1024_radix4_f64<>+0x1590(SB)/8, $570
DATA bitrev_size1024_radix4_f64<>+0x1598(SB)/8, $826
DATA bitrev_size1024_radix4_f64<>+0x15A0(SB)/8, $122
DATA bitrev_size1024_radix4_f64<>+0x15A8(SB)/8, $378
DATA bitrev_size1024_radix4_f64<>+0x15B0(SB)/8, $634
DATA bitrev_size1024_radix4_f64<>+0x15B8(SB)/8, $890
DATA bitrev_size1024_radix4_f64<>+0x15C0(SB)/8, $186
DATA bitrev_size1024_radix4_f64<>+0x15C8(SB)/8, $442
DATA bitrev_size1024_radix4_f64<>+0x15D0(SB)/8, $698
DATA bitrev_size1024_radix4_f64<>+0x15D8(SB)/8, $954
DATA bitrev_size1024_radix4_f64<>+0x15E0(SB)/8, $250
DATA bitrev_size1024_radix4_f64<>+0x15E8(SB)/8, $506
DATA bitrev_size1024_radix4_f64<>+0x15F0(SB)/8, $762
DATA bitrev_size1024_radix4_f64<>+0x15F8(SB)/8, $1018
DATA bitrev_size1024_radix4_f64<>+0x1600(SB)/8, $14
DATA bitrev_size1024_radix4_f64<>+0x1608(SB)/8, $270
DATA bitrev_size1024_radix4_f64<>+0x1610(SB)/8, $526
DATA bitrev_size1024_radix4_f64<>+0x1618(SB)/8, $782
DATA bitrev_size1024_radix4_f64<>+0x1620(SB)/8, $78
DATA bitrev_size1024_radix4_f64<>+0x1628(SB)/8, $334
DATA bitrev_size1024_radix4_f64<>+0x1630(SB)/8, $590
DATA bitrev_size1024_radix4_f64<>+0x1638(SB)/8, $846
DATA bitrev_size1024_radix4_f64<>+0x1640(SB)/8, $142
DATA bitrev_size1024_radix4_f64<>+0x1648(SB)/8, $398
DATA bitrev_size1024_radix4_f64<>+0x1650(SB)/8, $654
DATA bitrev_size1024_radix4_f64<>+0x1658(SB)/8, $910
DATA bitrev_size1024_radix4_f64<>+0x1660(SB)/8, $206
DATA bitrev_size1024_radix4_f64<>+0x1668(SB)/8, $462
DATA bitrev_size1024_radix4_f64<>+0x1670(SB)/8, $718
DATA bitrev_size1024_radix4_f64<>+0x1678(SB)/8, $974
DATA bitrev_size1024_radix4_f64<>+0x1680(SB)/8, $30
DATA bitrev_size1024_radix4_f64<>+0x1688(SB)/8, $286
DATA bitrev_size1024_radix4_f64<>+0x1690(SB)/8, $542
DATA bitrev_size1024_radix4_f64<>+0x1698(SB)/8, $798
DATA bitrev_size1024_radix4_f64<>+0x16A0(SB)/8, $94
DATA bitrev_size1024_radix4_f64<>+0x16A8(SB)/8, $350
DATA bitrev_size1024_radix4_f64<>+0x16B0(SB)/8, $606
DATA bitrev_size1024_radix4_f64<>+0x16B8(SB)/8, $862
DATA bitrev_size1024_radix4_f64<>+0x16C0(SB)/8, $158
DATA bitrev_size1024_radix4_f64<>+0x16C8(SB)/8, $414
DATA bitrev_size1024_radix4_f64<>+0x16D0(SB)/8, $670
DATA bitrev_size1024_radix4_f64<>+0x16D8(SB)/8, $926
DATA bitrev_size1024_radix4_f64<>+0x16E0(SB)/8, $222
DATA bitrev_size1024_radix4_f64<>+0x16E8(SB)/8, $478
DATA bitrev_size1024_radix4_f64<>+0x16F0(SB)/8, $734
DATA bitrev_size1024_radix4_f64<>+0x16F8(SB)/8, $990
DATA bitrev_size1024_radix4_f64<>+0x1700(SB)/8, $46
DATA bitrev_size1024_radix4_f64<>+0x1708(SB)/8, $302
DATA bitrev_size1024_radix4_f64<>+0x1710(SB)/8, $558
DATA bitrev_size1024_radix4_f64<>+0x1718(SB)/8, $814
DATA bitrev_size1024_radix4_f64<>+0x1720(SB)/8, $110
DATA bitrev_size1024_radix4_f64<>+0x1728(SB)/8, $366
DATA bitrev_size1024_radix4_f64<>+0x1730(SB)/8, $622
DATA bitrev_size1024_radix4_f64<>+0x1738(SB)/8, $878
DATA bitrev_size1024_radix4_f64<>+0x1740(SB)/8, $174
DATA bitrev_size1024_radix4_f64<>+0x1748(SB)/8, $430
DATA bitrev_size1024_radix4_f64<>+0x1750(SB)/8, $686
DATA bitrev_size1024_radix4_f64<>+0x1758(SB)/8, $942
DATA bitrev_size1024_radix4_f64<>+0x1760(SB)/8, $238
DATA bitrev_size1024_radix4_f64<>+0x1768(SB)/8, $494
DATA bitrev_size1024_radix4_f64<>+0x1770(SB)/8, $750
DATA bitrev_size1024_radix4_f64<>+0x1778(SB)/8, $1006
DATA bitrev_size1024_radix4_f64<>+0x1780(SB)/8, $62
DATA bitrev_size1024_radix4_f64<>+0x1788(SB)/8, $318
DATA bitrev_size1024_radix4_f64<>+0x1790(SB)/8, $574
DATA bitrev_size1024_radix4_f64<>+0x1798(SB)/8, $830
DATA bitrev_size1024_radix4_f64<>+0x17A0(SB)/8, $126
DATA bitrev_size1024_radix4_f64<>+0x17A8(SB)/8, $382
DATA bitrev_size1024_radix4_f64<>+0x17B0(SB)/8, $638
DATA bitrev_size1024_radix4_f64<>+0x17B8(SB)/8, $894
DATA bitrev_size1024_radix4_f64<>+0x17C0(SB)/8, $190
DATA bitrev_size1024_radix4_f64<>+0x17C8(SB)/8, $446
DATA bitrev_size1024_radix4_f64<>+0x17D0(SB)/8, $702
DATA bitrev_size1024_radix4_f64<>+0x17D8(SB)/8, $958
DATA bitrev_size1024_radix4_f64<>+0x17E0(SB)/8, $254
DATA bitrev_size1024_radix4_f64<>+0x17E8(SB)/8, $510
DATA bitrev_size1024_radix4_f64<>+0x17F0(SB)/8, $766
DATA bitrev_size1024_radix4_f64<>+0x17F8(SB)/8, $1022
DATA bitrev_size1024_radix4_f64<>+0x1800(SB)/8, $3
DATA bitrev_size1024_radix4_f64<>+0x1808(SB)/8, $259
DATA bitrev_size1024_radix4_f64<>+0x1810(SB)/8, $515
DATA bitrev_size1024_radix4_f64<>+0x1818(SB)/8, $771
DATA bitrev_size1024_radix4_f64<>+0x1820(SB)/8, $67
DATA bitrev_size1024_radix4_f64<>+0x1828(SB)/8, $323
DATA bitrev_size1024_radix4_f64<>+0x1830(SB)/8, $579
DATA bitrev_size1024_radix4_f64<>+0x1838(SB)/8, $835
DATA bitrev_size1024_radix4_f64<>+0x1840(SB)/8, $131
DATA bitrev_size1024_radix4_f64<>+0x1848(SB)/8, $387
DATA bitrev_size1024_radix4_f64<>+0x1850(SB)/8, $643
DATA bitrev_size1024_radix4_f64<>+0x1858(SB)/8, $899
DATA bitrev_size1024_radix4_f64<>+0x1860(SB)/8, $195
DATA bitrev_size1024_radix4_f64<>+0x1868(SB)/8, $451
DATA bitrev_size1024_radix4_f64<>+0x1870(SB)/8, $707
DATA bitrev_size1024_radix4_f64<>+0x1878(SB)/8, $963
DATA bitrev_size1024_radix4_f64<>+0x1880(SB)/8, $19
DATA bitrev_size1024_radix4_f64<>+0x1888(SB)/8, $275
DATA bitrev_size1024_radix4_f64<>+0x1890(SB)/8, $531
DATA bitrev_size1024_radix4_f64<>+0x1898(SB)/8, $787
DATA bitrev_size1024_radix4_f64<>+0x18A0(SB)/8, $83
DATA bitrev_size1024_radix4_f64<>+0x18A8(SB)/8, $339
DATA bitrev_size1024_radix4_f64<>+0x18B0(SB)/8, $595
DATA bitrev_size1024_radix4_f64<>+0x18B8(SB)/8, $851
DATA bitrev_size1024_radix4_f64<>+0x18C0(SB)/8, $147
DATA bitrev_size1024_radix4_f64<>+0x18C8(SB)/8, $403
DATA bitrev_size1024_radix4_f64<>+0x18D0(SB)/8, $659
DATA bitrev_size1024_radix4_f64<>+0x18D8(SB)/8, $915
DATA bitrev_size1024_radix4_f64<>+0x18E0(SB)/8, $211
DATA bitrev_size1024_radix4_f64<>+0x18E8(SB)/8, $467
DATA bitrev_size1024_radix4_f64<>+0x18F0(SB)/8, $723
DATA bitrev_size1024_radix4_f64<>+0x18F8(SB)/8, $979
DATA bitrev_size1024_radix4_f64<>+0x1900(SB)/8, $35
DATA bitrev_size1024_radix4_f64<>+0x1908(SB)/8, $291
DATA bitrev_size1024_radix4_f64<>+0x1910(SB)/8, $547
DATA bitrev_size1024_radix4_f64<>+0x1918(SB)/8, $803
DATA bitrev_size1024_radix4_f64<>+0x1920(SB)/8, $99
DATA bitrev_size1024_radix4_f64<>+0x1928(SB)/8, $355
DATA bitrev_size1024_radix4_f64<>+0x1930(SB)/8, $611
DATA bitrev_size1024_radix4_f64<>+0x1938(SB)/8, $867
DATA bitrev_size1024_radix4_f64<>+0x1940(SB)/8, $163
DATA bitrev_size1024_radix4_f64<>+0x1948(SB)/8, $419
DATA bitrev_size1024_radix4_f64<>+0x1950(SB)/8, $675
DATA bitrev_size1024_radix4_f64<>+0x1958(SB)/8, $931
DATA bitrev_size1024_radix4_f64<>+0x1960(SB)/8, $227
DATA bitrev_size1024_radix4_f64<>+0x1968(SB)/8, $483
DATA bitrev_size1024_radix4_f64<>+0x1970(SB)/8, $739
DATA bitrev_size1024_radix4_f64<>+0x1978(SB)/8, $995
DATA bitrev_size1024_radix4_f64<>+0x1980(SB)/8, $51
DATA bitrev_size1024_radix4_f64<>+0x1988(SB)/8, $307
DATA bitrev_size1024_radix4_f64<>+0x1990(SB)/8, $563
DATA bitrev_size1024_radix4_f64<>+0x1998(SB)/8, $819
DATA bitrev_size1024_radix4_f64<>+0x19A0(SB)/8, $115
DATA bitrev_size1024_radix4_f64<>+0x19A8(SB)/8, $371
DATA bitrev_size1024_radix4_f64<>+0x19B0(SB)/8, $627
DATA bitrev_size1024_radix4_f64<>+0x19B8(SB)/8, $883
DATA bitrev_size1024_radix4_f64<>+0x19C0(SB)/8, $179
DATA bitrev_size1024_radix4_f64<>+0x19C8(SB)/8, $435
DATA bitrev_size1024_radix4_f64<>+0x19D0(SB)/8, $691
DATA bitrev_size1024_radix4_f64<>+0x19D8(SB)/8, $947
DATA bitrev_size1024_radix4_f64<>+0x19E0(SB)/8, $243
DATA bitrev_size1024_radix4_f64<>+0x19E8(SB)/8, $499
DATA bitrev_size1024_radix4_f64<>+0x19F0(SB)/8, $755
DATA bitrev_size1024_radix4_f64<>+0x19F8(SB)/8, $1011
DATA bitrev_size1024_radix4_f64<>+0x1A00(SB)/8, $7
DATA bitrev_size1024_radix4_f64<>+0x1A08(SB)/8, $263
DATA bitrev_size1024_radix4_f64<>+0x1A10(SB)/8, $519
DATA bitrev_size1024_radix4_f64<>+0x1A18(SB)/8, $775
DATA bitrev_size1024_radix4_f64<>+0x1A20(SB)/8, $71
DATA bitrev_size1024_radix4_f64<>+0x1A28(SB)/8, $327
DATA bitrev_size1024_radix4_f64<>+0x1A30(SB)/8, $583
DATA bitrev_size1024_radix4_f64<>+0x1A38(SB)/8, $839
DATA bitrev_size1024_radix4_f64<>+0x1A40(SB)/8, $135
DATA bitrev_size1024_radix4_f64<>+0x1A48(SB)/8, $391
DATA bitrev_size1024_radix4_f64<>+0x1A50(SB)/8, $647
DATA bitrev_size1024_radix4_f64<>+0x1A58(SB)/8, $903
DATA bitrev_size1024_radix4_f64<>+0x1A60(SB)/8, $199
DATA bitrev_size1024_radix4_f64<>+0x1A68(SB)/8, $455
DATA bitrev_size1024_radix4_f64<>+0x1A70(SB)/8, $711
DATA bitrev_size1024_radix4_f64<>+0x1A78(SB)/8, $967
DATA bitrev_size1024_radix4_f64<>+0x1A80(SB)/8, $23
DATA bitrev_size1024_radix4_f64<>+0x1A88(SB)/8, $279
DATA bitrev_size1024_radix4_f64<>+0x1A90(SB)/8, $535
DATA bitrev_size1024_radix4_f64<>+0x1A98(SB)/8, $791
DATA bitrev_size1024_radix4_f64<>+0x1AA0(SB)/8, $87
DATA bitrev_size1024_radix4_f64<>+0x1AA8(SB)/8, $343
DATA bitrev_size1024_radix4_f64<>+0x1AB0(SB)/8, $599
DATA bitrev_size1024_radix4_f64<>+0x1AB8(SB)/8, $855
DATA bitrev_size1024_radix4_f64<>+0x1AC0(SB)/8, $151
DATA bitrev_size1024_radix4_f64<>+0x1AC8(SB)/8, $407
DATA bitrev_size1024_radix4_f64<>+0x1AD0(SB)/8, $663
DATA bitrev_size1024_radix4_f64<>+0x1AD8(SB)/8, $919
DATA bitrev_size1024_radix4_f64<>+0x1AE0(SB)/8, $215
DATA bitrev_size1024_radix4_f64<>+0x1AE8(SB)/8, $471
DATA bitrev_size1024_radix4_f64<>+0x1AF0(SB)/8, $727
DATA bitrev_size1024_radix4_f64<>+0x1AF8(SB)/8, $983
DATA bitrev_size1024_radix4_f64<>+0x1B00(SB)/8, $39
DATA bitrev_size1024_radix4_f64<>+0x1B08(SB)/8, $295
DATA bitrev_size1024_radix4_f64<>+0x1B10(SB)/8, $551
DATA bitrev_size1024_radix4_f64<>+0x1B18(SB)/8, $807
DATA bitrev_size1024_radix4_f64<>+0x1B20(SB)/8, $103
DATA bitrev_size1024_radix4_f64<>+0x1B28(SB)/8, $359
DATA bitrev_size1024_radix4_f64<>+0x1B30(SB)/8, $615
DATA bitrev_size1024_radix4_f64<>+0x1B38(SB)/8, $871
DATA bitrev_size1024_radix4_f64<>+0x1B40(SB)/8, $167
DATA bitrev_size1024_radix4_f64<>+0x1B48(SB)/8, $423
DATA bitrev_size1024_radix4_f64<>+0x1B50(SB)/8, $679
DATA bitrev_size1024_radix4_f64<>+0x1B58(SB)/8, $935
DATA bitrev_size1024_radix4_f64<>+0x1B60(SB)/8, $231
DATA bitrev_size1024_radix4_f64<>+0x1B68(SB)/8, $487
DATA bitrev_size1024_radix4_f64<>+0x1B70(SB)/8, $743
DATA bitrev_size1024_radix4_f64<>+0x1B78(SB)/8, $999
DATA bitrev_size1024_radix4_f64<>+0x1B80(SB)/8, $55
DATA bitrev_size1024_radix4_f64<>+0x1B88(SB)/8, $311
DATA bitrev_size1024_radix4_f64<>+0x1B90(SB)/8, $567
DATA bitrev_size1024_radix4_f64<>+0x1B98(SB)/8, $823
DATA bitrev_size1024_radix4_f64<>+0x1BA0(SB)/8, $119
DATA bitrev_size1024_radix4_f64<>+0x1BA8(SB)/8, $375
DATA bitrev_size1024_radix4_f64<>+0x1BB0(SB)/8, $631
DATA bitrev_size1024_radix4_f64<>+0x1BB8(SB)/8, $887
DATA bitrev_size1024_radix4_f64<>+0x1BC0(SB)/8, $183
DATA bitrev_size1024_radix4_f64<>+0x1BC8(SB)/8, $439
DATA bitrev_size1024_radix4_f64<>+0x1BD0(SB)/8, $695
DATA bitrev_size1024_radix4_f64<>+0x1BD8(SB)/8, $951
DATA bitrev_size1024_radix4_f64<>+0x1BE0(SB)/8, $247
DATA bitrev_size1024_radix4_f64<>+0x1BE8(SB)/8, $503
DATA bitrev_size1024_radix4_f64<>+0x1BF0(SB)/8, $759
DATA bitrev_size1024_radix4_f64<>+0x1BF8(SB)/8, $1015
DATA bitrev_size1024_radix4_f64<>+0x1C00(SB)/8, $11
DATA bitrev_size1024_radix4_f64<>+0x1C08(SB)/8, $267
DATA bitrev_size1024_radix4_f64<>+0x1C10(SB)/8, $523
DATA bitrev_size1024_radix4_f64<>+0x1C18(SB)/8, $779
DATA bitrev_size1024_radix4_f64<>+0x1C20(SB)/8, $75
DATA bitrev_size1024_radix4_f64<>+0x1C28(SB)/8, $331
DATA bitrev_size1024_radix4_f64<>+0x1C30(SB)/8, $587
DATA bitrev_size1024_radix4_f64<>+0x1C38(SB)/8, $843
DATA bitrev_size1024_radix4_f64<>+0x1C40(SB)/8, $139
DATA bitrev_size1024_radix4_f64<>+0x1C48(SB)/8, $395
DATA bitrev_size1024_radix4_f64<>+0x1C50(SB)/8, $651
DATA bitrev_size1024_radix4_f64<>+0x1C58(SB)/8, $907
DATA bitrev_size1024_radix4_f64<>+0x1C60(SB)/8, $203
DATA bitrev_size1024_radix4_f64<>+0x1C68(SB)/8, $459
DATA bitrev_size1024_radix4_f64<>+0x1C70(SB)/8, $715
DATA bitrev_size1024_radix4_f64<>+0x1C78(SB)/8, $971
DATA bitrev_size1024_radix4_f64<>+0x1C80(SB)/8, $27
DATA bitrev_size1024_radix4_f64<>+0x1C88(SB)/8, $283
DATA bitrev_size1024_radix4_f64<>+0x1C90(SB)/8, $539
DATA bitrev_size1024_radix4_f64<>+0x1C98(SB)/8, $795
DATA bitrev_size1024_radix4_f64<>+0x1CA0(SB)/8, $91
DATA bitrev_size1024_radix4_f64<>+0x1CA8(SB)/8, $347
DATA bitrev_size1024_radix4_f64<>+0x1CB0(SB)/8, $603
DATA bitrev_size1024_radix4_f64<>+0x1CB8(SB)/8, $859
DATA bitrev_size1024_radix4_f64<>+0x1CC0(SB)/8, $155
DATA bitrev_size1024_radix4_f64<>+0x1CC8(SB)/8, $411
DATA bitrev_size1024_radix4_f64<>+0x1CD0(SB)/8, $667
DATA bitrev_size1024_radix4_f64<>+0x1CD8(SB)/8, $923
DATA bitrev_size1024_radix4_f64<>+0x1CE0(SB)/8, $219
DATA bitrev_size1024_radix4_f64<>+0x1CE8(SB)/8, $475
DATA bitrev_size1024_radix4_f64<>+0x1CF0(SB)/8, $731
DATA bitrev_size1024_radix4_f64<>+0x1CF8(SB)/8, $987
DATA bitrev_size1024_radix4_f64<>+0x1D00(SB)/8, $43
DATA bitrev_size1024_radix4_f64<>+0x1D08(SB)/8, $299
DATA bitrev_size1024_radix4_f64<>+0x1D10(SB)/8, $555
DATA bitrev_size1024_radix4_f64<>+0x1D18(SB)/8, $811
DATA bitrev_size1024_radix4_f64<>+0x1D20(SB)/8, $107
DATA bitrev_size1024_radix4_f64<>+0x1D28(SB)/8, $363
DATA bitrev_size1024_radix4_f64<>+0x1D30(SB)/8, $619
DATA bitrev_size1024_radix4_f64<>+0x1D38(SB)/8, $875
DATA bitrev_size1024_radix4_f64<>+0x1D40(SB)/8, $171
DATA bitrev_size1024_radix4_f64<>+0x1D48(SB)/8, $427
DATA bitrev_size1024_radix4_f64<>+0x1D50(SB)/8, $683
DATA bitrev_size1024_radix4_f64<>+0x1D58(SB)/8, $939
DATA bitrev_size1024_radix4_f64<>+0x1D60(SB)/8, $235
DATA bitrev_size1024_radix4_f64<>+0x1D68(SB)/8, $491
DATA bitrev_size1024_radix4_f64<>+0x1D70(SB)/8, $747
DATA bitrev_size1024_radix4_f64<>+0x1D78(SB)/8, $1003
DATA bitrev_size1024_radix4_f64<>+0x1D80(SB)/8, $59
DATA bitrev_size1024_radix4_f64<>+0x1D88(SB)/8, $315
DATA bitrev_size1024_radix4_f64<>+0x1D90(SB)/8, $571
DATA bitrev_size1024_radix4_f64<>+0x1D98(SB)/8, $827
DATA bitrev_size1024_radix4_f64<>+0x1DA0(SB)/8, $123
DATA bitrev_size1024_radix4_f64<>+0x1DA8(SB)/8, $379
DATA bitrev_size1024_radix4_f64<>+0x1DB0(SB)/8, $635
DATA bitrev_size1024_radix4_f64<>+0x1DB8(SB)/8, $891
DATA bitrev_size1024_radix4_f64<>+0x1DC0(SB)/8, $187
DATA bitrev_size1024_radix4_f64<>+0x1DC8(SB)/8, $443
DATA bitrev_size1024_radix4_f64<>+0x1DD0(SB)/8, $699
DATA bitrev_size1024_radix4_f64<>+0x1DD8(SB)/8, $955
DATA bitrev_size1024_radix4_f64<>+0x1DE0(SB)/8, $251
DATA bitrev_size1024_radix4_f64<>+0x1DE8(SB)/8, $507
DATA bitrev_size1024_radix4_f64<>+0x1DF0(SB)/8, $763
DATA bitrev_size1024_radix4_f64<>+0x1DF8(SB)/8, $1019
DATA bitrev_size1024_radix4_f64<>+0x1E00(SB)/8, $15
DATA bitrev_size1024_radix4_f64<>+0x1E08(SB)/8, $271
DATA bitrev_size1024_radix4_f64<>+0x1E10(SB)/8, $527
DATA bitrev_size1024_radix4_f64<>+0x1E18(SB)/8, $783
DATA bitrev_size1024_radix4_f64<>+0x1E20(SB)/8, $79
DATA bitrev_size1024_radix4_f64<>+0x1E28(SB)/8, $335
DATA bitrev_size1024_radix4_f64<>+0x1E30(SB)/8, $591
DATA bitrev_size1024_radix4_f64<>+0x1E38(SB)/8, $847
DATA bitrev_size1024_radix4_f64<>+0x1E40(SB)/8, $143
DATA bitrev_size1024_radix4_f64<>+0x1E48(SB)/8, $399
DATA bitrev_size1024_radix4_f64<>+0x1E50(SB)/8, $655
DATA bitrev_size1024_radix4_f64<>+0x1E58(SB)/8, $911
DATA bitrev_size1024_radix4_f64<>+0x1E60(SB)/8, $207
DATA bitrev_size1024_radix4_f64<>+0x1E68(SB)/8, $463
DATA bitrev_size1024_radix4_f64<>+0x1E70(SB)/8, $719
DATA bitrev_size1024_radix4_f64<>+0x1E78(SB)/8, $975
DATA bitrev_size1024_radix4_f64<>+0x1E80(SB)/8, $31
DATA bitrev_size1024_radix4_f64<>+0x1E88(SB)/8, $287
DATA bitrev_size1024_radix4_f64<>+0x1E90(SB)/8, $543
DATA bitrev_size1024_radix4_f64<>+0x1E98(SB)/8, $799
DATA bitrev_size1024_radix4_f64<>+0x1EA0(SB)/8, $95
DATA bitrev_size1024_radix4_f64<>+0x1EA8(SB)/8, $351
DATA bitrev_size1024_radix4_f64<>+0x1EB0(SB)/8, $607
DATA bitrev_size1024_radix4_f64<>+0x1EB8(SB)/8, $863
DATA bitrev_size1024_radix4_f64<>+0x1EC0(SB)/8, $159
DATA bitrev_size1024_radix4_f64<>+0x1EC8(SB)/8, $415
DATA bitrev_size1024_radix4_f64<>+0x1ED0(SB)/8, $671
DATA bitrev_size1024_radix4_f64<>+0x1ED8(SB)/8, $927
DATA bitrev_size1024_radix4_f64<>+0x1EE0(SB)/8, $223
DATA bitrev_size1024_radix4_f64<>+0x1EE8(SB)/8, $479
DATA bitrev_size1024_radix4_f64<>+0x1EF0(SB)/8, $735
DATA bitrev_size1024_radix4_f64<>+0x1EF8(SB)/8, $991
DATA bitrev_size1024_radix4_f64<>+0x1F00(SB)/8, $47
DATA bitrev_size1024_radix4_f64<>+0x1F08(SB)/8, $303
DATA bitrev_size1024_radix4_f64<>+0x1F10(SB)/8, $559
DATA bitrev_size1024_radix4_f64<>+0x1F18(SB)/8, $815
DATA bitrev_size1024_radix4_f64<>+0x1F20(SB)/8, $111
DATA bitrev_size1024_radix4_f64<>+0x1F28(SB)/8, $367
DATA bitrev_size1024_radix4_f64<>+0x1F30(SB)/8, $623
DATA bitrev_size1024_radix4_f64<>+0x1F38(SB)/8, $879
DATA bitrev_size1024_radix4_f64<>+0x1F40(SB)/8, $175
DATA bitrev_size1024_radix4_f64<>+0x1F48(SB)/8, $431
DATA bitrev_size1024_radix4_f64<>+0x1F50(SB)/8, $687
DATA bitrev_size1024_radix4_f64<>+0x1F58(SB)/8, $943
DATA bitrev_size1024_radix4_f64<>+0x1F60(SB)/8, $239
DATA bitrev_size1024_radix4_f64<>+0x1F68(SB)/8, $495
DATA bitrev_size1024_radix4_f64<>+0x1F70(SB)/8, $751
DATA bitrev_size1024_radix4_f64<>+0x1F78(SB)/8, $1007
DATA bitrev_size1024_radix4_f64<>+0x1F80(SB)/8, $63
DATA bitrev_size1024_radix4_f64<>+0x1F88(SB)/8, $319
DATA bitrev_size1024_radix4_f64<>+0x1F90(SB)/8, $575
DATA bitrev_size1024_radix4_f64<>+0x1F98(SB)/8, $831
DATA bitrev_size1024_radix4_f64<>+0x1FA0(SB)/8, $127
DATA bitrev_size1024_radix4_f64<>+0x1FA8(SB)/8, $383
DATA bitrev_size1024_radix4_f64<>+0x1FB0(SB)/8, $639
DATA bitrev_size1024_radix4_f64<>+0x1FB8(SB)/8, $895
DATA bitrev_size1024_radix4_f64<>+0x1FC0(SB)/8, $191
DATA bitrev_size1024_radix4_f64<>+0x1FC8(SB)/8, $447
DATA bitrev_size1024_radix4_f64<>+0x1FD0(SB)/8, $703
DATA bitrev_size1024_radix4_f64<>+0x1FD8(SB)/8, $959
DATA bitrev_size1024_radix4_f64<>+0x1FE0(SB)/8, $255
DATA bitrev_size1024_radix4_f64<>+0x1FE8(SB)/8, $511
DATA bitrev_size1024_radix4_f64<>+0x1FF0(SB)/8, $767
DATA bitrev_size1024_radix4_f64<>+0x1FF8(SB)/8, $1023
GLOBL bitrev_size1024_radix4_f64<>(SB), RODATA, $8192

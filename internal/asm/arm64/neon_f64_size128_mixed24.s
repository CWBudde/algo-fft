//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-128 Mixed-Radix (Radix-4 + Radix-2) FFT Kernels for ARM64 (complex128)
// ===========================================================================
//
// Size 128 = 4 * 4 * 4 * 2, mixed-radix algorithm:
//   Stage 1: 32 radix-4 butterflies (no twiddles), stride=4
//   Stage 2: radix-4 with twiddles, size=16, step=8
//   Stage 3: radix-4 with twiddles, size=64, step=2
//   Stage 4: radix-2 with twiddles, size=128, step=1
//
// Each complex128 element is 16 bytes (real f64 + imag f64).
//
// ===========================================================================

#include "textflag.h"

// Note: neonInv128F64 is defined in neon_f64_size128_radix2.s to avoid duplicate symbols

// Forward transform, size 128, complex128, mixed radix
// func ForwardNEONSize128MixedRadix24Complex128Asm(dst, src, twiddle, scratch []complex128) bool
TEXT ·ForwardNEONSize128MixedRadix24Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $128, R13
	BNE  neon128m24f64_return_false

	MOVD dst+8(FP), R0
	CMP  $128, R0
	BLT  neon128m24f64_return_false

	MOVD twiddle+56(FP), R0
	CMP  $128, R0
	BLT  neon128m24f64_return_false

	MOVD scratch+80(FP), R0
	CMP  $128, R0
	BLT  neon128m24f64_return_false

	MOVD $bitrev_size128_mixed24_f64<>(SB), R12
	MOVD R8, R20

	CMP  R8, R9
	BNE  neon128m24f64_use_dst
	MOVD R11, R8

neon128m24f64_use_dst:
	// =========================================================================
	// Bit-reversal permutation
	// =========================================================================
	MOVD $0, R0

neon128m24f64_bitrev_loop:
	CMP  $128, R0
	BGE  neon128m24f64_stage1

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
	B    neon128m24f64_bitrev_loop

neon128m24f64_stage1:
	// =========================================================================
	// Stage 1: 32 radix-4 butterflies (no twiddles)
	// =========================================================================
	MOVD $0, R14

neon128m24f64_stage1_loop:
	CMP  $128, R14
	BGE  neon128m24f64_stage2

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
	B    neon128m24f64_stage1_loop

neon128m24f64_stage2:
	// =========================================================================
	// Stage 2: radix-4 with twiddles, size=16, step=8
	// 8 groups of 4 butterflies
	// =========================================================================
	MOVD $0, R14

neon128m24f64_stage2_base:
	CMP  $128, R14
	BGE  neon128m24f64_stage3

	MOVD $0, R15

neon128m24f64_stage2_j:
	CMP  $4, R15
	BGE  neon128m24f64_stage2_next

	ADD  R14, R15, R0
	ADD  $4, R0, R1
	ADD  $8, R0, R2
	ADD  $12, R0, R3

	// Twiddles: w1=tw[j*8], w2=tw[j*16], w3=tw[j*24]
	LSL  $3, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1

	LSL  $4, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3

	MOVD $24, R6
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
	B    neon128m24f64_stage2_j

neon128m24f64_stage2_next:
	ADD  $16, R14, R14
	B    neon128m24f64_stage2_base

neon128m24f64_stage3:
	// =========================================================================
	// Stage 3: radix-4 with twiddles, size=64, step=2
	// 2 groups of 16 butterflies
	// =========================================================================
	MOVD $0, R14

neon128m24f64_stage3_base:
	CMP  $128, R14
	BGE  neon128m24f64_stage4

	MOVD $0, R15

neon128m24f64_stage3_j:
	CMP  $16, R15
	BGE  neon128m24f64_stage3_next

	ADD  R14, R15, R0
	ADD  $16, R0, R1
	ADD  $32, R0, R2
	ADD  $48, R0, R3

	// Twiddles: w1=tw[j*2], w2=tw[j*4], w3=tw[j*6]
	LSL  $1, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1

	LSL  $2, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3

	MOVD $6, R6
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
	B    neon128m24f64_stage3_j

neon128m24f64_stage3_next:
	ADD  $64, R14, R14
	B    neon128m24f64_stage3_base

neon128m24f64_stage4:
	// =========================================================================
	// Stage 4: radix-2 with twiddles, size=128, step=1
	// =========================================================================
	MOVD $0, R0

neon128m24f64_stage4_loop:
	CMP  $64, R0
	BGE  neon128m24f64_done

	ADD  $64, R0, R1

	// Load twiddle[j]
	LSL  $4, R0, R2
	ADD  R10, R2, R2
	FMOVD 0(R2), F0
	FMOVD 8(R2), F1

	// Load a, b
	LSL  $4, R0, R2
	ADD  R8, R2, R2
	FMOVD 0(R2), F2
	FMOVD 8(R2), F3

	LSL  $4, R1, R2
	ADD  R8, R2, R2
	FMOVD 0(R2), F4
	FMOVD 8(R2), F5

	// wb = w * b
	FMULD F0, F4, F6
	FMULD F1, F5, F7
	FSUBD F7, F6, F6
	FMULD F0, F5, F7
	FMULD F1, F4, F8
	FADDD F8, F7, F7

	// Butterfly
	FADDD F6, F2, F8
	FADDD F7, F3, F9
	FSUBD F6, F2, F10
	FSUBD F7, F3, F11

	// Store
	LSL  $4, R0, R2
	ADD  R8, R2, R2
	FMOVD F8, 0(R2)
	FMOVD F9, 8(R2)

	LSL  $4, R1, R2
	ADD  R8, R2, R2
	FMOVD F10, 0(R2)
	FMOVD F11, 8(R2)

	ADD  $1, R0, R0
	B    neon128m24f64_stage4_loop

neon128m24f64_done:
	CMP  R8, R20
	BEQ  neon128m24f64_return_true

	MOVD $0, R0
neon128m24f64_copy_loop:
	CMP  $128, R0
	BGE  neon128m24f64_return_true
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon128m24f64_copy_loop

neon128m24f64_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon128m24f64_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform
// ===========================================================================
TEXT ·InverseNEONSize128MixedRadix24Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $128, R13
	BNE  neon128m24f64_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $128, R0
	BLT  neon128m24f64_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $128, R0
	BLT  neon128m24f64_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $128, R0
	BLT  neon128m24f64_inv_return_false

	MOVD $bitrev_size128_mixed24_f64<>(SB), R12
	MOVD R8, R20

	CMP  R8, R9
	BNE  neon128m24f64_inv_use_dst
	MOVD R11, R8

neon128m24f64_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon128m24f64_inv_bitrev_loop:
	CMP  $128, R0
	BGE  neon128m24f64_inv_stage1

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
	B    neon128m24f64_inv_bitrev_loop

neon128m24f64_inv_stage1:
	// Stage 1: 32 radix-4 butterflies (inverse)
	MOVD $0, R14

neon128m24f64_inv_stage1_loop:
	CMP  $128, R14
	BGE  neon128m24f64_inv_stage2

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
	B    neon128m24f64_inv_stage1_loop

neon128m24f64_inv_stage2:
	// Stage 2 with conjugated twiddles
	MOVD $0, R14

neon128m24f64_inv_stage2_base:
	CMP  $128, R14
	BGE  neon128m24f64_inv_stage3

	MOVD $0, R15

neon128m24f64_inv_stage2_j:
	CMP  $4, R15
	BGE  neon128m24f64_inv_stage2_next

	ADD  R14, R15, R0
	ADD  $4, R0, R1
	ADD  $8, R0, R2
	ADD  $12, R0, R3

	LSL  $3, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1
	FNEGD F1, F1

	LSL  $4, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3
	FNEGD F3, F3

	MOVD $24, R6
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
	B    neon128m24f64_inv_stage2_j

neon128m24f64_inv_stage2_next:
	ADD  $16, R14, R14
	B    neon128m24f64_inv_stage2_base

neon128m24f64_inv_stage3:
	// Stage 3 with conjugated twiddles
	MOVD $0, R14

neon128m24f64_inv_stage3_base:
	CMP  $128, R14
	BGE  neon128m24f64_inv_stage4

	MOVD $0, R15

neon128m24f64_inv_stage3_j:
	CMP  $16, R15
	BGE  neon128m24f64_inv_stage3_next

	ADD  R14, R15, R0
	ADD  $16, R0, R1
	ADD  $32, R0, R2
	ADD  $48, R0, R3

	LSL  $1, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1
	FNEGD F1, F1

	LSL  $2, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3
	FNEGD F3, F3

	MOVD $6, R6
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
	B    neon128m24f64_inv_stage3_j

neon128m24f64_inv_stage3_next:
	ADD  $64, R14, R14
	B    neon128m24f64_inv_stage3_base

neon128m24f64_inv_stage4:
	// Stage 4 with conjugated twiddles
	MOVD $0, R0

neon128m24f64_inv_stage4_loop:
	CMP  $64, R0
	BGE  neon128m24f64_inv_copy

	ADD  $64, R0, R1

	LSL  $4, R0, R2
	ADD  R10, R2, R2
	FMOVD 0(R2), F0
	FMOVD 8(R2), F1
	FNEGD F1, F1

	LSL  $4, R0, R2
	ADD  R8, R2, R2
	FMOVD 0(R2), F2
	FMOVD 8(R2), F3

	LSL  $4, R1, R2
	ADD  R8, R2, R2
	FMOVD 0(R2), F4
	FMOVD 8(R2), F5

	FMULD F0, F4, F6
	FMULD F1, F5, F7
	FSUBD F7, F6, F6
	FMULD F0, F5, F7
	FMULD F1, F4, F8
	FADDD F8, F7, F7

	FADDD F6, F2, F8
	FADDD F7, F3, F9
	FSUBD F6, F2, F10
	FSUBD F7, F3, F11

	LSL  $4, R0, R2
	ADD  R8, R2, R2
	FMOVD F8, 0(R2)
	FMOVD F9, 8(R2)

	LSL  $4, R1, R2
	ADD  R8, R2, R2
	FMOVD F10, 0(R2)
	FMOVD F11, 8(R2)

	ADD  $1, R0, R0
	B    neon128m24f64_inv_stage4_loop

neon128m24f64_inv_copy:
	CMP  R8, R20
	BEQ  neon128m24f64_inv_scale

	MOVD $0, R0
neon128m24f64_inv_copy_loop:
	CMP  $128, R0
	BGE  neon128m24f64_inv_scale
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon128m24f64_inv_copy_loop

neon128m24f64_inv_scale:
	MOVD $·neonInv128F64(SB), R1
	FMOVD (R1), F0
	MOVD $0, R0

neon128m24f64_inv_scale_loop:
	CMP  $128, R0
	BGE  neon128m24f64_inv_return_true
	LSL  $4, R0, R1
	ADD  R20, R1, R1
	FMOVD 0(R1), F2
	FMOVD 8(R1), F3
	FMULD F0, F2, F2
	FMULD F0, F3, F3
	FMOVD F2, 0(R1)
	FMOVD F3, 8(R1)
	ADD  $1, R0, R0
	B    neon128m24f64_inv_scale_loop

neon128m24f64_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon128m24f64_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Bit-reversal table for size 128 mixed-radix 4,4,4,2
// ===========================================================================
DATA bitrev_size128_mixed24_f64<>+0x000(SB)/8, $0
DATA bitrev_size128_mixed24_f64<>+0x008(SB)/8, $32
DATA bitrev_size128_mixed24_f64<>+0x010(SB)/8, $64
DATA bitrev_size128_mixed24_f64<>+0x018(SB)/8, $96
DATA bitrev_size128_mixed24_f64<>+0x020(SB)/8, $8
DATA bitrev_size128_mixed24_f64<>+0x028(SB)/8, $40
DATA bitrev_size128_mixed24_f64<>+0x030(SB)/8, $72
DATA bitrev_size128_mixed24_f64<>+0x038(SB)/8, $104
DATA bitrev_size128_mixed24_f64<>+0x040(SB)/8, $16
DATA bitrev_size128_mixed24_f64<>+0x048(SB)/8, $48
DATA bitrev_size128_mixed24_f64<>+0x050(SB)/8, $80
DATA bitrev_size128_mixed24_f64<>+0x058(SB)/8, $112
DATA bitrev_size128_mixed24_f64<>+0x060(SB)/8, $24
DATA bitrev_size128_mixed24_f64<>+0x068(SB)/8, $56
DATA bitrev_size128_mixed24_f64<>+0x070(SB)/8, $88
DATA bitrev_size128_mixed24_f64<>+0x078(SB)/8, $120
DATA bitrev_size128_mixed24_f64<>+0x080(SB)/8, $2
DATA bitrev_size128_mixed24_f64<>+0x088(SB)/8, $34
DATA bitrev_size128_mixed24_f64<>+0x090(SB)/8, $66
DATA bitrev_size128_mixed24_f64<>+0x098(SB)/8, $98
DATA bitrev_size128_mixed24_f64<>+0x0A0(SB)/8, $10
DATA bitrev_size128_mixed24_f64<>+0x0A8(SB)/8, $42
DATA bitrev_size128_mixed24_f64<>+0x0B0(SB)/8, $74
DATA bitrev_size128_mixed24_f64<>+0x0B8(SB)/8, $106
DATA bitrev_size128_mixed24_f64<>+0x0C0(SB)/8, $18
DATA bitrev_size128_mixed24_f64<>+0x0C8(SB)/8, $50
DATA bitrev_size128_mixed24_f64<>+0x0D0(SB)/8, $82
DATA bitrev_size128_mixed24_f64<>+0x0D8(SB)/8, $114
DATA bitrev_size128_mixed24_f64<>+0x0E0(SB)/8, $26
DATA bitrev_size128_mixed24_f64<>+0x0E8(SB)/8, $58
DATA bitrev_size128_mixed24_f64<>+0x0F0(SB)/8, $90
DATA bitrev_size128_mixed24_f64<>+0x0F8(SB)/8, $122
DATA bitrev_size128_mixed24_f64<>+0x100(SB)/8, $4
DATA bitrev_size128_mixed24_f64<>+0x108(SB)/8, $36
DATA bitrev_size128_mixed24_f64<>+0x110(SB)/8, $68
DATA bitrev_size128_mixed24_f64<>+0x118(SB)/8, $100
DATA bitrev_size128_mixed24_f64<>+0x120(SB)/8, $12
DATA bitrev_size128_mixed24_f64<>+0x128(SB)/8, $44
DATA bitrev_size128_mixed24_f64<>+0x130(SB)/8, $76
DATA bitrev_size128_mixed24_f64<>+0x138(SB)/8, $108
DATA bitrev_size128_mixed24_f64<>+0x140(SB)/8, $20
DATA bitrev_size128_mixed24_f64<>+0x148(SB)/8, $52
DATA bitrev_size128_mixed24_f64<>+0x150(SB)/8, $84
DATA bitrev_size128_mixed24_f64<>+0x158(SB)/8, $116
DATA bitrev_size128_mixed24_f64<>+0x160(SB)/8, $28
DATA bitrev_size128_mixed24_f64<>+0x168(SB)/8, $60
DATA bitrev_size128_mixed24_f64<>+0x170(SB)/8, $92
DATA bitrev_size128_mixed24_f64<>+0x178(SB)/8, $124
DATA bitrev_size128_mixed24_f64<>+0x180(SB)/8, $6
DATA bitrev_size128_mixed24_f64<>+0x188(SB)/8, $38
DATA bitrev_size128_mixed24_f64<>+0x190(SB)/8, $70
DATA bitrev_size128_mixed24_f64<>+0x198(SB)/8, $102
DATA bitrev_size128_mixed24_f64<>+0x1A0(SB)/8, $14
DATA bitrev_size128_mixed24_f64<>+0x1A8(SB)/8, $46
DATA bitrev_size128_mixed24_f64<>+0x1B0(SB)/8, $78
DATA bitrev_size128_mixed24_f64<>+0x1B8(SB)/8, $110
DATA bitrev_size128_mixed24_f64<>+0x1C0(SB)/8, $22
DATA bitrev_size128_mixed24_f64<>+0x1C8(SB)/8, $54
DATA bitrev_size128_mixed24_f64<>+0x1D0(SB)/8, $86
DATA bitrev_size128_mixed24_f64<>+0x1D8(SB)/8, $118
DATA bitrev_size128_mixed24_f64<>+0x1E0(SB)/8, $30
DATA bitrev_size128_mixed24_f64<>+0x1E8(SB)/8, $62
DATA bitrev_size128_mixed24_f64<>+0x1F0(SB)/8, $94
DATA bitrev_size128_mixed24_f64<>+0x1F8(SB)/8, $126
DATA bitrev_size128_mixed24_f64<>+0x200(SB)/8, $1
DATA bitrev_size128_mixed24_f64<>+0x208(SB)/8, $33
DATA bitrev_size128_mixed24_f64<>+0x210(SB)/8, $65
DATA bitrev_size128_mixed24_f64<>+0x218(SB)/8, $97
DATA bitrev_size128_mixed24_f64<>+0x220(SB)/8, $9
DATA bitrev_size128_mixed24_f64<>+0x228(SB)/8, $41
DATA bitrev_size128_mixed24_f64<>+0x230(SB)/8, $73
DATA bitrev_size128_mixed24_f64<>+0x238(SB)/8, $105
DATA bitrev_size128_mixed24_f64<>+0x240(SB)/8, $17
DATA bitrev_size128_mixed24_f64<>+0x248(SB)/8, $49
DATA bitrev_size128_mixed24_f64<>+0x250(SB)/8, $81
DATA bitrev_size128_mixed24_f64<>+0x258(SB)/8, $113
DATA bitrev_size128_mixed24_f64<>+0x260(SB)/8, $25
DATA bitrev_size128_mixed24_f64<>+0x268(SB)/8, $57
DATA bitrev_size128_mixed24_f64<>+0x270(SB)/8, $89
DATA bitrev_size128_mixed24_f64<>+0x278(SB)/8, $121
DATA bitrev_size128_mixed24_f64<>+0x280(SB)/8, $3
DATA bitrev_size128_mixed24_f64<>+0x288(SB)/8, $35
DATA bitrev_size128_mixed24_f64<>+0x290(SB)/8, $67
DATA bitrev_size128_mixed24_f64<>+0x298(SB)/8, $99
DATA bitrev_size128_mixed24_f64<>+0x2A0(SB)/8, $11
DATA bitrev_size128_mixed24_f64<>+0x2A8(SB)/8, $43
DATA bitrev_size128_mixed24_f64<>+0x2B0(SB)/8, $75
DATA bitrev_size128_mixed24_f64<>+0x2B8(SB)/8, $107
DATA bitrev_size128_mixed24_f64<>+0x2C0(SB)/8, $19
DATA bitrev_size128_mixed24_f64<>+0x2C8(SB)/8, $51
DATA bitrev_size128_mixed24_f64<>+0x2D0(SB)/8, $83
DATA bitrev_size128_mixed24_f64<>+0x2D8(SB)/8, $115
DATA bitrev_size128_mixed24_f64<>+0x2E0(SB)/8, $27
DATA bitrev_size128_mixed24_f64<>+0x2E8(SB)/8, $59
DATA bitrev_size128_mixed24_f64<>+0x2F0(SB)/8, $91
DATA bitrev_size128_mixed24_f64<>+0x2F8(SB)/8, $123
DATA bitrev_size128_mixed24_f64<>+0x300(SB)/8, $5
DATA bitrev_size128_mixed24_f64<>+0x308(SB)/8, $37
DATA bitrev_size128_mixed24_f64<>+0x310(SB)/8, $69
DATA bitrev_size128_mixed24_f64<>+0x318(SB)/8, $101
DATA bitrev_size128_mixed24_f64<>+0x320(SB)/8, $13
DATA bitrev_size128_mixed24_f64<>+0x328(SB)/8, $45
DATA bitrev_size128_mixed24_f64<>+0x330(SB)/8, $77
DATA bitrev_size128_mixed24_f64<>+0x338(SB)/8, $109
DATA bitrev_size128_mixed24_f64<>+0x340(SB)/8, $21
DATA bitrev_size128_mixed24_f64<>+0x348(SB)/8, $53
DATA bitrev_size128_mixed24_f64<>+0x350(SB)/8, $85
DATA bitrev_size128_mixed24_f64<>+0x358(SB)/8, $117
DATA bitrev_size128_mixed24_f64<>+0x360(SB)/8, $29
DATA bitrev_size128_mixed24_f64<>+0x368(SB)/8, $61
DATA bitrev_size128_mixed24_f64<>+0x370(SB)/8, $93
DATA bitrev_size128_mixed24_f64<>+0x378(SB)/8, $125
DATA bitrev_size128_mixed24_f64<>+0x380(SB)/8, $7
DATA bitrev_size128_mixed24_f64<>+0x388(SB)/8, $39
DATA bitrev_size128_mixed24_f64<>+0x390(SB)/8, $71
DATA bitrev_size128_mixed24_f64<>+0x398(SB)/8, $103
DATA bitrev_size128_mixed24_f64<>+0x3A0(SB)/8, $15
DATA bitrev_size128_mixed24_f64<>+0x3A8(SB)/8, $47
DATA bitrev_size128_mixed24_f64<>+0x3B0(SB)/8, $79
DATA bitrev_size128_mixed24_f64<>+0x3B8(SB)/8, $111
DATA bitrev_size128_mixed24_f64<>+0x3C0(SB)/8, $23
DATA bitrev_size128_mixed24_f64<>+0x3C8(SB)/8, $55
DATA bitrev_size128_mixed24_f64<>+0x3D0(SB)/8, $87
DATA bitrev_size128_mixed24_f64<>+0x3D8(SB)/8, $119
DATA bitrev_size128_mixed24_f64<>+0x3E0(SB)/8, $31
DATA bitrev_size128_mixed24_f64<>+0x3E8(SB)/8, $63
DATA bitrev_size128_mixed24_f64<>+0x3F0(SB)/8, $95
DATA bitrev_size128_mixed24_f64<>+0x3F8(SB)/8, $127
GLOBL bitrev_size128_mixed24_f64<>(SB), RODATA, $1024

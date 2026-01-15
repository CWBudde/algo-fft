//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-4 Radix-4 FFT Kernels for ARM64 (complex128)
// ===========================================================================

#include "textflag.h"

DATA ·neonInv4F64+0(SB)/8, $0x3fd0000000000000 // 1/4
GLOBL ·neonInv4F64(SB), RODATA, $8

// Forward transform, size 4, radix-4 (no bit-reversal needed).
TEXT ·ForwardNEONSize4Radix4Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $4, R13
	BNE  neon4r4f64_return_false

	MOVD dst+8(FP), R0
	CMP  $4, R0
	BLT  neon4r4f64_return_false

	MOVD twiddle+56(FP), R0
	CMP  $4, R0
	BLT  neon4r4f64_return_false

	MOVD scratch+80(FP), R0
	CMP  $4, R0
	BLT  neon4r4f64_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon4r4f64_use_dst
	MOVD R11, R8

neon4r4f64_use_dst:
	FMOVD 0(R9), F0
	FMOVD 8(R9), F1
	FMOVD 16(R9), F2
	FMOVD 24(R9), F3
	FMOVD 32(R9), F4
	FMOVD 40(R9), F5
	FMOVD 48(R9), F6
	FMOVD 56(R9), F7

	FSUBD F4, F0, F8
	FSUBD F5, F1, F9
	FADDD F4, F0, F10
	FADDD F5, F1, F11

	FSUBD F6, F2, F12
	FSUBD F7, F3, F13
	FADDD F6, F2, F14
	FADDD F7, F3, F15

	FMOVD F13, F16
	FNEGD F12, F17

	FADDD F14, F10, F18
	FADDD F15, F11, F19
	FADDD F16, F8, F20
	FADDD F17, F9, F21
	FSUBD F14, F10, F22
	FSUBD F15, F11, F23
	FSUBD F16, F8, F24
	FSUBD F17, F9, F25

	FMOVD F18, 0(R8)
	FMOVD F19, 8(R8)
	FMOVD F20, 16(R8)
	FMOVD F21, 24(R8)
	FMOVD F22, 32(R8)
	FMOVD F23, 40(R8)
	FMOVD F24, 48(R8)
	FMOVD F25, 56(R8)

	CMP  R8, R20
	BEQ  neon4r4f64_return_true

	MOVD $0, R0
neon4r4f64_copy_loop:
	CMP  $4, R0
	BGE  neon4r4f64_return_true
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon4r4f64_copy_loop

neon4r4f64_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon4r4f64_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Inverse transform, size 4, radix-4 (no bit-reversal needed).
TEXT ·InverseNEONSize4Radix4Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $4, R13
	BNE  neon4r4f64_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $4, R0
	BLT  neon4r4f64_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $4, R0
	BLT  neon4r4f64_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $4, R0
	BLT  neon4r4f64_inv_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon4r4f64_inv_use_dst
	MOVD R11, R8

neon4r4f64_inv_use_dst:
	FMOVD 0(R9), F0
	FMOVD 8(R9), F1
	FMOVD 16(R9), F2
	FMOVD 24(R9), F3
	FMOVD 32(R9), F4
	FMOVD 40(R9), F5
	FMOVD 48(R9), F6
	FMOVD 56(R9), F7

	FSUBD F4, F0, F8
	FSUBD F5, F1, F9
	FADDD F4, F0, F10
	FADDD F5, F1, F11

	FSUBD F6, F2, F12
	FSUBD F7, F3, F13
	FADDD F6, F2, F14
	FADDD F7, F3, F15

	FNEGD F13, F16
	FMOVD F12, F17

	FADDD F14, F10, F18
	FADDD F15, F11, F19
	FADDD F16, F8, F20
	FADDD F17, F9, F21
	FSUBD F14, F10, F22
	FSUBD F15, F11, F23
	FSUBD F16, F8, F24
	FSUBD F17, F9, F25

	FMOVD F18, 0(R8)
	FMOVD F19, 8(R8)
	FMOVD F20, 16(R8)
	FMOVD F21, 24(R8)
	FMOVD F22, 32(R8)
	FMOVD F23, 40(R8)
	FMOVD F24, 48(R8)
	FMOVD F25, 56(R8)

	CMP  R8, R20
	BEQ  neon4r4f64_inv_scale

	MOVD $0, R0
neon4r4f64_inv_copy_loop:
	CMP  $4, R0
	BGE  neon4r4f64_inv_scale
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon4r4f64_inv_copy_loop

neon4r4f64_inv_scale:
	MOVD $·neonInv4F64(SB), R1
	FMOVD (R1), F0
	MOVD $0, R0

neon4r4f64_inv_scale_loop:
	CMP  $4, R0
	BGE  neon4r4f64_inv_return_true
	LSL  $4, R0, R1
	ADD  R20, R1, R1
	FMOVD 0(R1), F2
	FMOVD 8(R1), F3
	FMULD F0, F2, F2
	FMULD F0, F3, F3
	FMOVD F2, 0(R1)
	FMOVD F3, 8(R1)
	ADD  $1, R0, R0
	B    neon4r4f64_inv_scale_loop

neon4r4f64_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon4r4f64_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

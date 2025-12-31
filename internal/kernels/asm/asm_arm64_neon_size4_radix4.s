//go:build arm64 && fft_asm && !purego

// ===========================================================================
// NEON Size-4 Radix-4 FFT Kernels for ARM64
// ===========================================================================

#include "textflag.h"

// Forward transform, size 4, radix-4 (no bit-reversal needed).
TEXT ·forwardNEONSize4Radix4Complex64Asm(SB), NOSPLIT, $0-121
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD bitrev+96(FP), R12
	MOVD src+32(FP), R13

	CMP  $4, R13
	BNE  neon4r4_return_false

	MOVD dst+8(FP), R0
	CMP  $4, R0
	BLT  neon4r4_return_false

	MOVD twiddle+56(FP), R0
	CMP  $4, R0
	BLT  neon4r4_return_false

	MOVD scratch+80(FP), R0
	CMP  $4, R0
	BLT  neon4r4_return_false

	MOVD bitrev+104(FP), R0
	CBZ  R0, neon4r4_bitrev_ok
	CMP  $4, R0
	BLT  neon4r4_return_false

neon4r4_bitrev_ok:
	MOVD R8, R20
	CMP  R8, R9
	BNE  neon4r4_use_dst
	MOVD R11, R8

neon4r4_use_dst:
	FMOVS 0(R9), F0
	FMOVS 4(R9), F1
	FMOVS 8(R9), F2
	FMOVS 12(R9), F3
	FMOVS 16(R9), F4
	FMOVS 20(R9), F5
	FMOVS 24(R9), F6
	FMOVS 28(R9), F7

	FSUBS F4, F0, F8
	FSUBS F5, F1, F9
	FADDS F4, F0, F10
	FADDS F5, F1, F11

	FSUBS F6, F2, F12
	FSUBS F7, F3, F13
	FADDS F6, F2, F14
	FADDS F7, F3, F15

	FMOVS F13, F16
	FNEGS F12, F17

	FADDS F14, F10, F18
	FADDS F15, F11, F19
	FADDS F16, F8, F20
	FADDS F17, F9, F21
	FSUBS F14, F10, F22
	FSUBS F15, F11, F23
	FSUBS F16, F8, F24
	FSUBS F17, F9, F25

	FMOVS F18, 0(R8)
	FMOVS F19, 4(R8)
	FMOVS F20, 8(R8)
	FMOVS F21, 12(R8)
	FMOVS F22, 16(R8)
	FMOVS F23, 20(R8)
	FMOVS F24, 24(R8)
	FMOVS F25, 28(R8)

	CMP  R8, R20
	BEQ  neon4r4_return_true

	MOVD $0, R0
neon4r4_copy_loop:
	CMP  $4, R0
	BGE  neon4r4_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon4r4_copy_loop

neon4r4_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon4r4_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// Inverse transform, size 4, radix-4 (no bit-reversal needed).
TEXT ·inverseNEONSize4Radix4Complex64Asm(SB), NOSPLIT, $0-121
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD bitrev+96(FP), R12
	MOVD src+32(FP), R13

	CMP  $4, R13
	BNE  neon4r4_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $4, R0
	BLT  neon4r4_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $4, R0
	BLT  neon4r4_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $4, R0
	BLT  neon4r4_inv_return_false

	MOVD bitrev+104(FP), R0
	CBZ  R0, neon4r4_inv_bitrev_ok
	CMP  $4, R0
	BLT  neon4r4_inv_return_false

neon4r4_inv_bitrev_ok:
	MOVD R8, R20
	CMP  R8, R9
	BNE  neon4r4_inv_use_dst
	MOVD R11, R8

neon4r4_inv_use_dst:
	FMOVS 0(R9), F0
	FMOVS 4(R9), F1
	FMOVS 8(R9), F2
	FMOVS 12(R9), F3
	FMOVS 16(R9), F4
	FMOVS 20(R9), F5
	FMOVS 24(R9), F6
	FMOVS 28(R9), F7

	FSUBS F4, F0, F8
	FSUBS F5, F1, F9
	FADDS F4, F0, F10
	FADDS F5, F1, F11

	FSUBS F6, F2, F12
	FSUBS F7, F3, F13
	FADDS F6, F2, F14
	FADDS F7, F3, F15

	FNEGS F13, F16
	FMOVS F12, F17

	FADDS F14, F10, F18
	FADDS F15, F11, F19
	FADDS F16, F8, F20
	FADDS F17, F9, F21
	FSUBS F14, F10, F22
	FSUBS F15, F11, F23
	FSUBS F16, F8, F24
	FSUBS F17, F9, F25

	FMOVS F18, 0(R8)
	FMOVS F19, 4(R8)
	FMOVS F20, 8(R8)
	FMOVS F21, 12(R8)
	FMOVS F22, 16(R8)
	FMOVS F23, 20(R8)
	FMOVS F24, 24(R8)
	FMOVS F25, 28(R8)

	CMP  R8, R20
	BEQ  neon4r4_inv_scale

	MOVD $0, R0
neon4r4_inv_copy_loop:
	CMP  $4, R0
	BGE  neon4r4_inv_scale
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon4r4_inv_copy_loop

neon4r4_inv_scale:
	MOVD $·neonInv4(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon4r4_inv_scale_loop:
	CMP  $4, R0
	BGE  neon4r4_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon4r4_inv_scale_loop

neon4r4_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon4r4_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

//go:build arm64 && fft_asm && !purego

// ===========================================================================
// NEON Size-256 Radix-2 FFT Kernels for ARM64
// ===========================================================================

#include "textflag.h"

DATA ·neonInv256+0(SB)/4, $0x3b800000 // 1/256
GLOBL ·neonInv256(SB), RODATA, $4

// Forward transform, size 256, complex64, radix-2
TEXT ·forwardNEONSize256Radix2Complex64Asm(SB), NOSPLIT, $0-121
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD bitrev+96(FP), R12
	MOVD src+32(FP), R13

	CMP  $256, R13
	BNE  neon256r2_return_false

	MOVD dst+8(FP), R0
	CMP  $256, R0
	BLT  neon256r2_return_false

	MOVD twiddle+56(FP), R0
	CMP  $256, R0
	BLT  neon256r2_return_false

	MOVD scratch+80(FP), R0
	CMP  $256, R0
	BLT  neon256r2_return_false

	MOVD bitrev+104(FP), R0
	CMP  $256, R0
	BLT  neon256r2_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon256r2_use_dst
	MOVD R11, R8

neon256r2_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon256r2_bitrev_loop:
	CMP  $256, R0
	BGE  neon256r2_stage

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
	B    neon256r2_bitrev_loop

neon256r2_stage:
	MOVD $2, R14               // size

neon256r2_size_loop:
	CMP  $256, R14
	BGT  neon256r2_done

	LSR  $1, R14, R15          // half
	UDIV R14, R13, R16         // step = n/size

	MOVD $0, R17               // base

neon256r2_base_loop:
	CMP  R13, R17
	BGE  neon256r2_next_size

	MOVD $0, R0                // j

neon256r2_inner_loop:
	CMP  R15, R0
	BGE  neon256r2_next_base

	ADD  R17, R0, R1           // idx_a
	ADD  R1, R15, R2           // idx_b

	MUL  R0, R16, R3
	LSL  $3, R3, R3
	ADD  R10, R3, R3
	FMOVS 0(R3), F0
	FMOVS 4(R3), F1

	LSL  $3, R1, R4
	ADD  R8, R4, R4
	FMOVS 0(R4), F2
	FMOVS 4(R4), F3

	LSL  $3, R2, R4
	ADD  R8, R4, R4
	FMOVS 0(R4), F4
	FMOVS 4(R4), F5

	// wb = w * b
	FMULS F0, F4, F6
	FMULS F1, F5, F7
	FSUBS F7, F6, F6
	FMULS F0, F5, F7
	FMULS F1, F4, F8
	FADDS F8, F7, F7

	FADDS F6, F2, F9
	FADDS F7, F3, F10
	FSUBS F6, F2, F11
	FSUBS F7, F3, F12

	LSL  $3, R1, R4
	ADD  R8, R4, R4
	FMOVS F9, 0(R4)
	FMOVS F10, 4(R4)

	LSL  $3, R2, R4
	ADD  R8, R4, R4
	FMOVS F11, 0(R4)
	FMOVS F12, 4(R4)

	ADD  $1, R0, R0
	B    neon256r2_inner_loop

neon256r2_next_base:
	ADD  R14, R17, R17
	B    neon256r2_base_loop

neon256r2_next_size:
	LSL  $1, R14, R14
	B    neon256r2_size_loop

neon256r2_done:
	CMP  R8, R20
	BEQ  neon256r2_return_true

	MOVD $0, R0
neon256r2_copy_loop:
	CMP  $256, R0
	BGE  neon256r2_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon256r2_copy_loop

neon256r2_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon256r2_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// Inverse transform, size 256, complex64, radix-2
TEXT ·inverseNEONSize256Radix2Complex64Asm(SB), NOSPLIT, $0-121
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD bitrev+96(FP), R12
	MOVD src+32(FP), R13

	CMP  $256, R13
	BNE  neon256r2_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $256, R0
	BLT  neon256r2_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $256, R0
	BLT  neon256r2_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $256, R0
	BLT  neon256r2_inv_return_false

	MOVD bitrev+104(FP), R0
	CMP  $256, R0
	BLT  neon256r2_inv_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon256r2_inv_use_dst
	MOVD R11, R8

neon256r2_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon256r2_inv_bitrev_loop:
	CMP  $256, R0
	BGE  neon256r2_inv_stage

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
	B    neon256r2_inv_bitrev_loop

neon256r2_inv_stage:
	MOVD $2, R14               // size

neon256r2_inv_size_loop:
	CMP  $256, R14
	BGT  neon256r2_inv_done

	LSR  $1, R14, R15          // half
	UDIV R14, R13, R16         // step = n/size

	MOVD $0, R17               // base

neon256r2_inv_base_loop:
	CMP  R13, R17
	BGE  neon256r2_inv_next_size

	MOVD $0, R0                // j

neon256r2_inv_inner_loop:
	CMP  R15, R0
	BGE  neon256r2_inv_next_base

	ADD  R17, R0, R1           // idx_a
	ADD  R1, R15, R2           // idx_b

	MUL  R0, R16, R3
	LSL  $3, R3, R3
	ADD  R10, R3, R3
	FMOVS 0(R3), F0
	FMOVS 4(R3), F1
	FNEGS  F1, F1

	LSL  $3, R1, R4
	ADD  R8, R4, R4
	FMOVS 0(R4), F2
	FMOVS 4(R4), F3

	LSL  $3, R2, R4
	ADD  R8, R4, R4
	FMOVS 0(R4), F4
	FMOVS 4(R4), F5

	// wb = w * b
	FMULS F0, F4, F6
	FMULS F1, F5, F7
	FSUBS F7, F6, F6
	FMULS F0, F5, F7
	FMULS F1, F4, F8
	FADDS F8, F7, F7

	FADDS F6, F2, F9
	FADDS F7, F3, F10
	FSUBS F6, F2, F11
	FSUBS F7, F3, F12

	LSL  $3, R1, R4
	ADD  R8, R4, R4
	FMOVS F9, 0(R4)
	FMOVS F10, 4(R4)

	LSL  $3, R2, R4
	ADD  R8, R4, R4
	FMOVS F11, 0(R4)
	FMOVS F12, 4(R4)

	ADD  $1, R0, R0
	B    neon256r2_inv_inner_loop

neon256r2_inv_next_base:
	ADD  R14, R17, R17
	B    neon256r2_inv_base_loop

neon256r2_inv_next_size:
	LSL  $1, R14, R14
	B    neon256r2_inv_size_loop

neon256r2_inv_done:
	CMP  R8, R20
	BEQ  neon256r2_inv_scale

	MOVD $0, R0
neon256r2_inv_copy_loop:
	CMP  $256, R0
	BGE  neon256r2_inv_scale
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon256r2_inv_copy_loop

neon256r2_inv_scale:
	MOVD $·neonInv256(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon256r2_inv_scale_loop:
	CMP  $256, R0
	BGE  neon256r2_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon256r2_inv_scale_loop

neon256r2_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon256r2_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

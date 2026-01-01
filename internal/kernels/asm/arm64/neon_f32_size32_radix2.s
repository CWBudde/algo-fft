//go:build arm64 && fft_asm && !purego

// ===========================================================================
// NEON Size-32 Radix-2 FFT Kernels for ARM64
// ===========================================================================

#include "textflag.h"

DATA ·neonInv32+0(SB)/4, $0x3d000000 // 1/32
GLOBL ·neonInv32(SB), RODATA, $4

// Forward transform, size 32, complex64, radix-2
TEXT ·ForwardNEONSize32Complex64Asm(SB), NOSPLIT, $0-121
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD bitrev+96(FP), R12
	MOVD src+32(FP), R13

	CMP  $32, R13
	BNE  neon32r2_return_false

	MOVD dst+8(FP), R0
	CMP  $32, R0
	BLT  neon32r2_return_false

	MOVD twiddle+56(FP), R0
	CMP  $32, R0
	BLT  neon32r2_return_false

	MOVD scratch+80(FP), R0
	CMP  $32, R0
	BLT  neon32r2_return_false

	MOVD bitrev+104(FP), R0
	CMP  $32, R0
	BLT  neon32r2_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon32r2_use_dst
	MOVD R11, R8

neon32r2_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon32r2_bitrev_loop:
	CMP  $32, R0
	BGE  neon32r2_stage

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
	B    neon32r2_bitrev_loop

neon32r2_stage:
	MOVD $2, R14               // size

neon32r2_size_loop:
	CMP  $32, R14
	BGT  neon32r2_done

	LSR  $1, R14, R15          // half
	UDIV R14, R13, R16         // step = n/size

	MOVD $0, R17               // base

neon32r2_base_loop:
	CMP  R13, R17
	BGE  neon32r2_next_size

	MOVD $0, R0                // j

neon32r2_inner_loop:
	CMP  R15, R0
	BGE  neon32r2_next_base

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
	B    neon32r2_inner_loop

neon32r2_next_base:
	ADD  R14, R17, R17
	B    neon32r2_base_loop

neon32r2_next_size:
	LSL  $1, R14, R14
	B    neon32r2_size_loop

neon32r2_done:
	CMP  R8, R20
	BEQ  neon32r2_return_true

	MOVD $0, R0
neon32r2_copy_loop:
	CMP  $32, R0
	BGE  neon32r2_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon32r2_copy_loop

neon32r2_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon32r2_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// Inverse transform, size 32, complex64, radix-2
TEXT ·InverseNEONSize32Complex64Asm(SB), NOSPLIT, $0-121
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD bitrev+96(FP), R12
	MOVD src+32(FP), R13

	CMP  $32, R13
	BNE  neon32r2_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $32, R0
	BLT  neon32r2_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $32, R0
	BLT  neon32r2_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $32, R0
	BLT  neon32r2_inv_return_false

	MOVD bitrev+104(FP), R0
	CMP  $32, R0
	BLT  neon32r2_inv_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon32r2_inv_use_dst
	MOVD R11, R8

neon32r2_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon32r2_inv_bitrev_loop:
	CMP  $32, R0
	BGE  neon32r2_inv_stage

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
	B    neon32r2_inv_bitrev_loop

neon32r2_inv_stage:
	MOVD $2, R14

neon32r2_inv_size_loop:
	CMP  $32, R14
	BGT  neon32r2_inv_done

	LSR  $1, R14, R15
	UDIV R14, R13, R16

	MOVD $0, R17

neon32r2_inv_base_loop:
	CMP  R13, R17
	BGE  neon32r2_inv_next_size

	MOVD $0, R0

neon32r2_inv_inner_loop:
	CMP  R15, R0
	BGE  neon32r2_inv_next_base

	ADD  R17, R0, R1
	ADD  R1, R15, R2

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
	B    neon32r2_inv_inner_loop

neon32r2_inv_next_base:
	ADD  R14, R17, R17
	B    neon32r2_inv_base_loop

neon32r2_inv_next_size:
	LSL  $1, R14, R14
	B    neon32r2_inv_size_loop

neon32r2_inv_done:
	CMP  R8, R20
	BEQ  neon32r2_inv_scale

	MOVD $0, R0
neon32r2_inv_copy_loop:
	CMP  $32, R0
	BGE  neon32r2_inv_scale
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon32r2_inv_copy_loop

neon32r2_inv_scale:
	MOVD $·neonInv32(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon32r2_inv_scale_loop:
	CMP  $32, R0
	BGE  neon32r2_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon32r2_inv_scale_loop

neon32r2_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon32r2_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-16 Radix-2 FFT Kernels for ARM64 (complex128)
// ===========================================================================

#include "textflag.h"

// Forward transform, size 16, complex128, radix-2
TEXT ·ForwardNEONSize16Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $16, R13
	BNE  neon16r2f64_return_false

	MOVD dst+8(FP), R0
	CMP  $16, R0
	BLT  neon16r2f64_return_false

	MOVD twiddle+56(FP), R0
	CMP  $16, R0
	BLT  neon16r2f64_return_false

	MOVD scratch+80(FP), R0
	CMP  $16, R0
	BLT  neon16r2f64_return_false

	MOVD bitrev+104(FP), R0
	CMP  $16, R0
	BLT  neon16r2f64_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon16r2f64_use_dst
	MOVD R11, R8

neon16r2f64_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon16r2f64_bitrev_loop:
	CMP  $16, R0
	BGE  neon16r2f64_stage

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
	B    neon16r2f64_bitrev_loop

neon16r2f64_stage:
	MOVD $2, R14               // size

neon16r2f64_size_loop:
	CMP  $16, R14
	BGT  neon16r2f64_done

	LSR  $1, R14, R15          // half
	UDIV R14, R13, R16         // step = n/size

	MOVD $0, R17               // base

neon16r2f64_base_loop:
	CMP  R13, R17
	BGE  neon16r2f64_next_size

	MOVD $0, R0                // j

neon16r2f64_inner_loop:
	CMP  R15, R0
	BGE  neon16r2f64_next_base

	ADD  R17, R0, R1           // idx_a
	ADD  R1, R15, R2           // idx_b

	MUL  R0, R16, R3
	LSL  $4, R3, R3
	ADD  R10, R3, R3
	FMOVD 0(R3), F0
	FMOVD 8(R3), F1

	LSL  $4, R1, R4
	ADD  R8, R4, R4
	FMOVD 0(R4), F2
	FMOVD 8(R4), F3

	LSL  $4, R2, R4
	ADD  R8, R4, R4
	FMOVD 0(R4), F4
	FMOVD 8(R4), F5

	// wb = w * b
	FMULD F0, F4, F6
	FMULD F1, F5, F7
	FSUBD F7, F6, F6
	FMULD F0, F5, F7
	FMULD F1, F4, F8
	FADDD F8, F7, F7

	FADDD F6, F2, F9
	FADDD F7, F3, F10
	FSUBD F6, F2, F11
	FSUBD F7, F3, F12

	LSL  $4, R1, R4
	ADD  R8, R4, R4
	FMOVD F9, 0(R4)
	FMOVD F10, 8(R4)

	LSL  $4, R2, R4
	ADD  R8, R4, R4
	FMOVD F11, 0(R4)
	FMOVD F12, 8(R4)

	ADD  $1, R0, R0
	B    neon16r2f64_inner_loop

neon16r2f64_next_base:
	ADD  R14, R17, R17
	B    neon16r2f64_base_loop

neon16r2f64_next_size:
	LSL  $1, R14, R14
	B    neon16r2f64_size_loop

neon16r2f64_done:
	CMP  R8, R20
	BEQ  neon16r2f64_return_true

	MOVD $0, R0
neon16r2f64_copy_loop:
	CMP  $16, R0
	BGE  neon16r2f64_return_true
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon16r2f64_copy_loop

neon16r2f64_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon16r2f64_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// Inverse transform, size 16, complex128, radix-2
TEXT ·InverseNEONSize16Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $16, R13
	BNE  neon16r2f64_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $16, R0
	BLT  neon16r2f64_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $16, R0
	BLT  neon16r2f64_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $16, R0
	BLT  neon16r2f64_inv_return_false

	MOVD bitrev+104(FP), R0
	CMP  $16, R0
	BLT  neon16r2f64_inv_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon16r2f64_inv_use_dst
	MOVD R11, R8

neon16r2f64_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon16r2f64_inv_bitrev_loop:
	CMP  $16, R0
	BGE  neon16r2f64_inv_stage

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
	B    neon16r2f64_inv_bitrev_loop

neon16r2f64_inv_stage:
	MOVD $2, R14               // size

neon16r2f64_inv_size_loop:
	CMP  $16, R14
	BGT  neon16r2f64_inv_done

	LSR  $1, R14, R15          // half
	UDIV R14, R13, R16         // step = n/size

	MOVD $0, R17               // base

neon16r2f64_inv_base_loop:
	CMP  R13, R17
	BGE  neon16r2f64_inv_next_size

	MOVD $0, R0                // j

neon16r2f64_inv_inner_loop:
	CMP  R15, R0
	BGE  neon16r2f64_inv_next_base

	ADD  R17, R0, R1           // idx_a
	ADD  R1, R15, R2           // idx_b

	MUL  R0, R16, R3
	LSL  $4, R3, R3
	ADD  R10, R3, R3
	FMOVD 0(R3), F0
	FMOVD 8(R3), F1
	FNEGD F1, F1

	LSL  $4, R1, R4
	ADD  R8, R4, R4
	FMOVD 0(R4), F2
	FMOVD 8(R4), F3

	LSL  $4, R2, R4
	ADD  R8, R4, R4
	FMOVD 0(R4), F4
	FMOVD 8(R4), F5

	// wb = w * b
	FMULD F0, F4, F6
	FMULD F1, F5, F7
	FSUBD F7, F6, F6
	FMULD F0, F5, F7
	FMULD F1, F4, F8
	FADDD F8, F7, F7

	FADDD F6, F2, F9
	FADDD F7, F3, F10
	FSUBD F6, F2, F11
	FSUBD F7, F3, F12

	LSL  $4, R1, R4
	ADD  R8, R4, R4
	FMOVD F9, 0(R4)
	FMOVD F10, 8(R4)

	LSL  $4, R2, R4
	ADD  R8, R4, R4
	FMOVD F11, 0(R4)
	FMOVD F12, 8(R4)

	ADD  $1, R0, R0
	B    neon16r2f64_inv_inner_loop

neon16r2f64_inv_next_base:
	ADD  R14, R17, R17
	B    neon16r2f64_inv_base_loop

neon16r2f64_inv_next_size:
	LSL  $1, R14, R14
	B    neon16r2f64_inv_size_loop

neon16r2f64_inv_done:
	CMP  R8, R20
	BEQ  neon16r2f64_inv_scale

	MOVD $0, R0
neon16r2f64_inv_copy_loop:
	CMP  $16, R0
	BGE  neon16r2f64_inv_scale
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon16r2f64_inv_copy_loop

neon16r2f64_inv_scale:
	MOVD $·neonInv16F64(SB), R1
	FMOVD (R1), F0
	MOVD $0, R0

neon16r2f64_inv_scale_loop:
	CMP  $16, R0
	BGE  neon16r2f64_inv_return_true
	LSL  $4, R0, R1
	ADD  R20, R1, R1
	FMOVD 0(R1), F2
	FMOVD 8(R1), F3
	FMULD F0, F2, F2
	FMULD F0, F3, F3
	FMOVD F2, 0(R1)
	FMOVD F3, 8(R1)
	ADD  $1, R0, R0
	B    neon16r2f64_inv_scale_loop

neon16r2f64_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon16r2f64_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

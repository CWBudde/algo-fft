//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-8 Radix-2 FFT Kernels for ARM64 (complex128)
// ===========================================================================

#include "textflag.h"

DATA ·neonInv8F64+0(SB)/8, $0x3fc0000000000000 // 1/8 = 0.125
GLOBL ·neonInv8F64(SB), RODATA, $8

// Forward transform, size 8, complex128, radix-2
TEXT ·ForwardNEONSize8Radix2Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $8, R13
	BNE  neon8r2f64_return_false

	MOVD dst+8(FP), R0
	CMP  $8, R0
	BLT  neon8r2f64_return_false

	MOVD twiddle+56(FP), R0
	CMP  $8, R0
	BLT  neon8r2f64_return_false

	MOVD scratch+80(FP), R0
	CMP  $8, R0
	BLT  neon8r2f64_return_false

	MOVD $bitrev_size8_radix2<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon8r2f64_use_dst
	MOVD R11, R8

neon8r2f64_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon8r2f64_bitrev_loop:
	CMP  $8, R0
	BGE  neon8r2f64_stage

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
	B    neon8r2f64_bitrev_loop

neon8r2f64_stage:
	MOVD $2, R14               // size

neon8r2f64_size_loop:
	CMP  $8, R14
	BGT  neon8r2f64_done

	LSR  $1, R14, R15          // half
	UDIV R14, R13, R16         // step = n/size

	MOVD $0, R17               // base

neon8r2f64_base_loop:
	CMP  R13, R17
	BGE  neon8r2f64_next_size

	MOVD $0, R0                // j

neon8r2f64_inner_loop:
	CMP  R15, R0
	BGE  neon8r2f64_next_base

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
	B    neon8r2f64_inner_loop

neon8r2f64_next_base:
	ADD  R14, R17, R17
	B    neon8r2f64_base_loop

neon8r2f64_next_size:
	LSL  $1, R14, R14
	B    neon8r2f64_size_loop

neon8r2f64_done:
	CMP  R8, R20
	BEQ  neon8r2f64_return_true

	MOVD $0, R0
neon8r2f64_copy_loop:
	CMP  $8, R0
	BGE  neon8r2f64_return_true
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon8r2f64_copy_loop

neon8r2f64_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon8r2f64_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Inverse transform, size 8, complex128, radix-2
TEXT ·InverseNEONSize8Radix2Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $8, R13
	BNE  neon8r2f64_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $8, R0
	BLT  neon8r2f64_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $8, R0
	BLT  neon8r2f64_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $8, R0
	BLT  neon8r2f64_inv_return_false

	MOVD $bitrev_size8_radix2<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon8r2f64_inv_use_dst
	MOVD R11, R8

neon8r2f64_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon8r2f64_inv_bitrev_loop:
	CMP  $8, R0
	BGE  neon8r2f64_inv_stage

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
	B    neon8r2f64_inv_bitrev_loop

neon8r2f64_inv_stage:
	MOVD $2, R14               // size

neon8r2f64_inv_size_loop:
	CMP  $8, R14
	BGT  neon8r2f64_inv_done

	LSR  $1, R14, R15          // half
	UDIV R14, R13, R16         // step = n/size

	MOVD $0, R17               // base

neon8r2f64_inv_base_loop:
	CMP  R13, R17
	BGE  neon8r2f64_inv_next_size

	MOVD $0, R0                // j

neon8r2f64_inv_inner_loop:
	CMP  R15, R0
	BGE  neon8r2f64_inv_next_base

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
	B    neon8r2f64_inv_inner_loop

neon8r2f64_inv_next_base:
	ADD  R14, R17, R17
	B    neon8r2f64_inv_base_loop

neon8r2f64_inv_next_size:
	LSL  $1, R14, R14
	B    neon8r2f64_inv_size_loop

neon8r2f64_inv_done:
	CMP  R8, R20
	BEQ  neon8r2f64_inv_scale

	MOVD $0, R0
neon8r2f64_inv_copy_loop:
	CMP  $8, R0
	BGE  neon8r2f64_inv_scale
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon8r2f64_inv_copy_loop

neon8r2f64_inv_scale:
	MOVD $·neonInv8F64(SB), R1
	FMOVD (R1), F0
	MOVD $0, R0

neon8r2f64_inv_scale_loop:
	CMP  $8, R0
	BGE  neon8r2f64_inv_return_true

	LSL  $4, R0, R3
	ADD  R20, R3, R3
	FMOVD 0(R3), F3
	FMOVD 8(R3), F4
	FMULD F0, F3, F3
	FMULD F0, F4, F4
	FMOVD F3, 0(R3)
	FMOVD F4, 8(R3)

	ADD  $1, R0, R0
	B    neon8r2f64_inv_scale_loop

neon8r2f64_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon8r2f64_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Bit-reversal table for size 8 radix-2
GLOBL bitrev_size8_radix2<>(SB), RODATA, $64
DATA bitrev_size8_radix2<>+0(SB)/8, $0
DATA bitrev_size8_radix2<>+8(SB)/8, $4
DATA bitrev_size8_radix2<>+16(SB)/8, $2
DATA bitrev_size8_radix2<>+24(SB)/8, $6
DATA bitrev_size8_radix2<>+32(SB)/8, $1
DATA bitrev_size8_radix2<>+40(SB)/8, $5
DATA bitrev_size8_radix2<>+48(SB)/8, $3
DATA bitrev_size8_radix2<>+56(SB)/8, $7

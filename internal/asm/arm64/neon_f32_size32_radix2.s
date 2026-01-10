//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-32 Radix-2 FFT Kernels for ARM64
// ===========================================================================

#include "textflag.h"

DATA ·neonInv32+0(SB)/4, $0x3d000000 // 1/32
GLOBL ·neonInv32(SB), RODATA, $4

// Forward transform, size 32, complex64, radix-2
TEXT ·ForwardNEONSize32Radix2Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
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

	MOVD $bitrev_size32_radix2<>(SB), R12

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
	MOVB R0, ret+96(FP)
	RET

neon32r2_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Inverse transform, size 32, complex64, radix-2
TEXT ·InverseNEONSize32Radix2Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
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

	MOVD $bitrev_size32_radix2<>(SB), R12

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
	MOVB R0, ret+96(FP)
	RET

neon32r2_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Data Section
// ===========================================================================

// Bit-reversal permutation for size 32: [0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31]
DATA bitrev_size32_radix2<>+0x00(SB)/8, $0   // bitrev[0] = 0
DATA bitrev_size32_radix2<>+0x08(SB)/8, $16  // bitrev[1] = 16
DATA bitrev_size32_radix2<>+0x10(SB)/8, $8   // bitrev[2] = 8
DATA bitrev_size32_radix2<>+0x18(SB)/8, $24  // bitrev[3] = 24
DATA bitrev_size32_radix2<>+0x20(SB)/8, $4   // bitrev[4] = 4
DATA bitrev_size32_radix2<>+0x28(SB)/8, $20  // bitrev[5] = 20
DATA bitrev_size32_radix2<>+0x30(SB)/8, $12  // bitrev[6] = 12
DATA bitrev_size32_radix2<>+0x38(SB)/8, $28  // bitrev[7] = 28
DATA bitrev_size32_radix2<>+0x40(SB)/8, $2   // bitrev[8] = 2
DATA bitrev_size32_radix2<>+0x48(SB)/8, $18  // bitrev[9] = 18
DATA bitrev_size32_radix2<>+0x50(SB)/8, $10  // bitrev[10] = 10
DATA bitrev_size32_radix2<>+0x58(SB)/8, $26  // bitrev[11] = 26
DATA bitrev_size32_radix2<>+0x60(SB)/8, $6   // bitrev[12] = 6
DATA bitrev_size32_radix2<>+0x68(SB)/8, $22  // bitrev[13] = 22
DATA bitrev_size32_radix2<>+0x70(SB)/8, $14  // bitrev[14] = 14
DATA bitrev_size32_radix2<>+0x78(SB)/8, $30  // bitrev[15] = 30
DATA bitrev_size32_radix2<>+0x80(SB)/8, $1   // bitrev[16] = 1
DATA bitrev_size32_radix2<>+0x88(SB)/8, $17  // bitrev[17] = 17
DATA bitrev_size32_radix2<>+0x90(SB)/8, $9   // bitrev[18] = 9
DATA bitrev_size32_radix2<>+0x98(SB)/8, $25  // bitrev[19] = 25
DATA bitrev_size32_radix2<>+0xa0(SB)/8, $5   // bitrev[20] = 5
DATA bitrev_size32_radix2<>+0xa8(SB)/8, $21  // bitrev[21] = 21
DATA bitrev_size32_radix2<>+0xb0(SB)/8, $13  // bitrev[22] = 13
DATA bitrev_size32_radix2<>+0xb8(SB)/8, $29  // bitrev[23] = 29
DATA bitrev_size32_radix2<>+0xc0(SB)/8, $3   // bitrev[24] = 3
DATA bitrev_size32_radix2<>+0xc8(SB)/8, $19  // bitrev[25] = 19
DATA bitrev_size32_radix2<>+0xd0(SB)/8, $11  // bitrev[26] = 11
DATA bitrev_size32_radix2<>+0xd8(SB)/8, $27  // bitrev[27] = 27
DATA bitrev_size32_radix2<>+0xe0(SB)/8, $7   // bitrev[28] = 7
DATA bitrev_size32_radix2<>+0xe8(SB)/8, $23  // bitrev[29] = 23
DATA bitrev_size32_radix2<>+0xf0(SB)/8, $15  // bitrev[30] = 15
DATA bitrev_size32_radix2<>+0xf8(SB)/8, $31  // bitrev[31] = 31
GLOBL bitrev_size32_radix2<>(SB), RODATA, $256

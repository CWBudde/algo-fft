//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-64 Radix-2 FFT Kernels for ARM64
// ===========================================================================

#include "textflag.h"

// Forward transform, size 64, complex64, radix-2
TEXT ·ForwardNEONSize64Radix2Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $64, R13
	BNE  neon64r2_return_false

	MOVD dst+8(FP), R0
	CMP  $64, R0
	BLT  neon64r2_return_false

	MOVD twiddle+56(FP), R0
	CMP  $64, R0
	BLT  neon64r2_return_false

	MOVD scratch+80(FP), R0
	CMP  $64, R0
	BLT  neon64r2_return_false

	MOVD $bitrev_size64_radix2<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon64r2_use_dst
	MOVD R11, R8

neon64r2_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon64r2_bitrev_loop:
	CMP  $64, R0
	BGE  neon64r2_stage

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
	B    neon64r2_bitrev_loop

neon64r2_stage:
	MOVD $2, R14               // size

neon64r2_size_loop:
	CMP  $64, R14
	BGT  neon64r2_done

	LSR  $1, R14, R15          // half
	UDIV R14, R13, R16         // step = n/size

	MOVD $0, R17               // base

neon64r2_base_loop:
	CMP  R13, R17
	BGE  neon64r2_next_size

	MOVD $0, R0                // j

neon64r2_inner_loop:
	CMP  R15, R0
	BGE  neon64r2_next_base

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
	B    neon64r2_inner_loop

neon64r2_next_base:
	ADD  R14, R17, R17
	B    neon64r2_base_loop

neon64r2_next_size:
	LSL  $1, R14, R14
	B    neon64r2_size_loop

neon64r2_done:
	CMP  R8, R20
	BEQ  neon64r2_return_true

	MOVD $0, R0
neon64r2_copy_loop:
	CMP  $64, R0
	BGE  neon64r2_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon64r2_copy_loop

neon64r2_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon64r2_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Inverse transform, size 64, complex64, radix-2
TEXT ·InverseNEONSize64Radix2Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $64, R13
	BNE  neon64r2_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $64, R0
	BLT  neon64r2_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $64, R0
	BLT  neon64r2_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $64, R0
	BLT  neon64r2_inv_return_false

	MOVD $bitrev_size64_radix2<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon64r2_inv_use_dst
	MOVD R11, R8

neon64r2_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon64r2_inv_bitrev_loop:
	CMP  $64, R0
	BGE  neon64r2_inv_stage

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
	B    neon64r2_inv_bitrev_loop

neon64r2_inv_stage:
	MOVD $2, R14               // size

neon64r2_inv_size_loop:
	CMP  $64, R14
	BGT  neon64r2_inv_done

	LSR  $1, R14, R15          // half
	UDIV R14, R13, R16         // step = n/size

	MOVD $0, R17               // base

neon64r2_inv_base_loop:
	CMP  R13, R17
	BGE  neon64r2_inv_next_size

	MOVD $0, R0                // j

neon64r2_inv_inner_loop:
	CMP  R15, R0
	BGE  neon64r2_inv_next_base

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
	B    neon64r2_inv_inner_loop

neon64r2_inv_next_base:
	ADD  R14, R17, R17
	B    neon64r2_inv_base_loop

neon64r2_inv_next_size:
	LSL  $1, R14, R14
	B    neon64r2_inv_size_loop

neon64r2_inv_done:
	CMP  R8, R20
	BEQ  neon64r2_inv_scale

	MOVD $0, R0
neon64r2_inv_copy_loop:
	CMP  $64, R0
	BGE  neon64r2_inv_scale
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon64r2_inv_copy_loop

neon64r2_inv_scale:
	MOVD $·neonInv64(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon64r2_inv_scale_loop:
	CMP  $64, R0
	BGE  neon64r2_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon64r2_inv_scale_loop

neon64r2_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon64r2_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Data Section
// ===========================================================================

// Bit-reversal permutation for size 64: [0, 32, 16, 48, 8, 40, 24, 56, 4, 36, 20, 52, 12, 44, 28, 60, 2, 34, 18, 50, 10, 42, 26, 58, 6, 38, 22, 54, 14, 46, 30, 62, 1, 33, 17, 49, 9, 41, 25, 57, 5, 37, 21, 53, 13, 45, 29, 61, 3, 35, 19, 51, 11, 43, 27, 59, 7, 39, 23, 55, 15, 47, 31, 63]
DATA bitrev_size64_radix2<>+0x000(SB)/8, $0   // bitrev[0] = 0
DATA bitrev_size64_radix2<>+0x008(SB)/8, $32  // bitrev[1] = 32
DATA bitrev_size64_radix2<>+0x010(SB)/8, $16  // bitrev[2] = 16
DATA bitrev_size64_radix2<>+0x018(SB)/8, $48  // bitrev[3] = 48
DATA bitrev_size64_radix2<>+0x020(SB)/8, $8   // bitrev[4] = 8
DATA bitrev_size64_radix2<>+0x028(SB)/8, $40  // bitrev[5] = 40
DATA bitrev_size64_radix2<>+0x030(SB)/8, $24  // bitrev[6] = 24
DATA bitrev_size64_radix2<>+0x038(SB)/8, $56  // bitrev[7] = 56
DATA bitrev_size64_radix2<>+0x040(SB)/8, $4   // bitrev[8] = 4
DATA bitrev_size64_radix2<>+0x048(SB)/8, $36  // bitrev[9] = 36
DATA bitrev_size64_radix2<>+0x050(SB)/8, $20  // bitrev[10] = 20
DATA bitrev_size64_radix2<>+0x058(SB)/8, $52  // bitrev[11] = 52
DATA bitrev_size64_radix2<>+0x060(SB)/8, $12  // bitrev[12] = 12
DATA bitrev_size64_radix2<>+0x068(SB)/8, $44  // bitrev[13] = 44
DATA bitrev_size64_radix2<>+0x070(SB)/8, $28  // bitrev[14] = 28
DATA bitrev_size64_radix2<>+0x078(SB)/8, $60  // bitrev[15] = 60
DATA bitrev_size64_radix2<>+0x080(SB)/8, $2   // bitrev[16] = 2
DATA bitrev_size64_radix2<>+0x088(SB)/8, $34  // bitrev[17] = 34
DATA bitrev_size64_radix2<>+0x090(SB)/8, $18  // bitrev[18] = 18
DATA bitrev_size64_radix2<>+0x098(SB)/8, $50  // bitrev[19] = 50
DATA bitrev_size64_radix2<>+0x0A0(SB)/8, $10  // bitrev[20] = 10
DATA bitrev_size64_radix2<>+0x0A8(SB)/8, $42  // bitrev[21] = 42
DATA bitrev_size64_radix2<>+0x0B0(SB)/8, $26  // bitrev[22] = 26
DATA bitrev_size64_radix2<>+0x0B8(SB)/8, $58  // bitrev[23] = 58
DATA bitrev_size64_radix2<>+0x0C0(SB)/8, $6   // bitrev[24] = 6
DATA bitrev_size64_radix2<>+0x0C8(SB)/8, $38  // bitrev[25] = 38
DATA bitrev_size64_radix2<>+0x0D0(SB)/8, $22  // bitrev[26] = 22
DATA bitrev_size64_radix2<>+0x0D8(SB)/8, $54  // bitrev[27] = 54
DATA bitrev_size64_radix2<>+0x0E0(SB)/8, $14  // bitrev[28] = 14
DATA bitrev_size64_radix2<>+0x0E8(SB)/8, $46  // bitrev[29] = 46
DATA bitrev_size64_radix2<>+0x0F0(SB)/8, $30  // bitrev[30] = 30
DATA bitrev_size64_radix2<>+0x0F8(SB)/8, $62  // bitrev[31] = 62
DATA bitrev_size64_radix2<>+0x100(SB)/8, $1   // bitrev[32] = 1
DATA bitrev_size64_radix2<>+0x108(SB)/8, $33  // bitrev[33] = 33
DATA bitrev_size64_radix2<>+0x110(SB)/8, $17  // bitrev[34] = 17
DATA bitrev_size64_radix2<>+0x118(SB)/8, $49  // bitrev[35] = 49
DATA bitrev_size64_radix2<>+0x120(SB)/8, $9   // bitrev[36] = 9
DATA bitrev_size64_radix2<>+0x128(SB)/8, $41  // bitrev[37] = 41
DATA bitrev_size64_radix2<>+0x130(SB)/8, $25  // bitrev[38] = 25
DATA bitrev_size64_radix2<>+0x138(SB)/8, $57  // bitrev[39] = 57
DATA bitrev_size64_radix2<>+0x140(SB)/8, $5   // bitrev[40] = 5
DATA bitrev_size64_radix2<>+0x148(SB)/8, $37  // bitrev[41] = 37
DATA bitrev_size64_radix2<>+0x150(SB)/8, $21  // bitrev[42] = 21
DATA bitrev_size64_radix2<>+0x158(SB)/8, $53  // bitrev[43] = 53
DATA bitrev_size64_radix2<>+0x160(SB)/8, $13  // bitrev[44] = 13
DATA bitrev_size64_radix2<>+0x168(SB)/8, $45  // bitrev[45] = 45
DATA bitrev_size64_radix2<>+0x170(SB)/8, $29  // bitrev[46] = 29
DATA bitrev_size64_radix2<>+0x178(SB)/8, $61  // bitrev[47] = 61
DATA bitrev_size64_radix2<>+0x180(SB)/8, $3   // bitrev[48] = 3
DATA bitrev_size64_radix2<>+0x188(SB)/8, $35  // bitrev[49] = 35
DATA bitrev_size64_radix2<>+0x190(SB)/8, $19  // bitrev[50] = 19
DATA bitrev_size64_radix2<>+0x198(SB)/8, $51  // bitrev[51] = 51
DATA bitrev_size64_radix2<>+0x1A0(SB)/8, $11  // bitrev[52] = 11
DATA bitrev_size64_radix2<>+0x1A8(SB)/8, $43  // bitrev[53] = 43
DATA bitrev_size64_radix2<>+0x1B0(SB)/8, $27  // bitrev[54] = 27
DATA bitrev_size64_radix2<>+0x1B8(SB)/8, $59  // bitrev[55] = 59
DATA bitrev_size64_radix2<>+0x1C0(SB)/8, $7   // bitrev[56] = 7
DATA bitrev_size64_radix2<>+0x1C8(SB)/8, $39  // bitrev[57] = 39
DATA bitrev_size64_radix2<>+0x1D0(SB)/8, $23  // bitrev[58] = 23
DATA bitrev_size64_radix2<>+0x1D8(SB)/8, $55  // bitrev[59] = 55
DATA bitrev_size64_radix2<>+0x1E0(SB)/8, $15  // bitrev[60] = 15
DATA bitrev_size64_radix2<>+0x1E8(SB)/8, $47  // bitrev[61] = 47
DATA bitrev_size64_radix2<>+0x1F0(SB)/8, $31  // bitrev[62] = 31
DATA bitrev_size64_radix2<>+0x1F8(SB)/8, $63  // bitrev[63] = 63
GLOBL bitrev_size64_radix2<>(SB), RODATA, $512

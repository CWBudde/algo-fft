//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-128 Radix-2 FFT Kernels for ARM64
// ===========================================================================

#include "textflag.h"

DATA ·neonInv128+0(SB)/4, $0x3c000000 // 1/128
GLOBL ·neonInv128(SB), RODATA, $4

// Forward transform, size 128, complex64, radix-2
TEXT ·ForwardNEONSize128Radix2Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $128, R13
	BNE  neon128r2_return_false

	MOVD dst+8(FP), R0
	CMP  $128, R0
	BLT  neon128r2_return_false

	MOVD twiddle+56(FP), R0
	CMP  $128, R0
	BLT  neon128r2_return_false

	MOVD scratch+80(FP), R0
	CMP  $128, R0
	BLT  neon128r2_return_false

	MOVD $bitrev_size128_radix2<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon128r2_use_dst
	MOVD R11, R8

neon128r2_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon128r2_bitrev_loop:
	CMP  $128, R0
	BGE  neon128r2_stage

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
	B    neon128r2_bitrev_loop

neon128r2_stage:
	MOVD $2, R14

neon128r2_size_loop:
	CMP  $128, R14
	BGT  neon128r2_done

	LSR  $1, R14, R15
	UDIV R14, R13, R16

	MOVD $0, R17

neon128r2_base_loop:
	CMP  R13, R17
	BGE  neon128r2_next_size

	MOVD $0, R0

neon128r2_inner_loop:
	CMP  R15, R0
	BGE  neon128r2_next_base

	ADD  R17, R0, R1
	ADD  R1, R15, R2

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
	B    neon128r2_inner_loop

neon128r2_next_base:
	ADD  R14, R17, R17
	B    neon128r2_base_loop

neon128r2_next_size:
	LSL  $1, R14, R14
	B    neon128r2_size_loop

neon128r2_done:
	CMP  R8, R20
	BEQ  neon128r2_return_true

	MOVD $0, R0
neon128r2_copy_loop:
	CMP  $128, R0
	BGE  neon128r2_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon128r2_copy_loop

neon128r2_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon128r2_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Inverse transform, size 128, complex64, radix-2
TEXT ·InverseNEONSize128Radix2Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $128, R13
	BNE  neon128r2_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $128, R0
	BLT  neon128r2_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $128, R0
	BLT  neon128r2_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $128, R0
	BLT  neon128r2_inv_return_false

	MOVD $bitrev_size128_radix2<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon128r2_inv_use_dst
	MOVD R11, R8

neon128r2_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon128r2_inv_bitrev_loop:
	CMP  $128, R0
	BGE  neon128r2_inv_stage

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
	B    neon128r2_inv_bitrev_loop

neon128r2_inv_stage:
	MOVD $2, R14

neon128r2_inv_size_loop:
	CMP  $128, R14
	BGT  neon128r2_inv_done

	LSR  $1, R14, R15
	UDIV R14, R13, R16

	MOVD $0, R17

neon128r2_inv_base_loop:
	CMP  R13, R17
	BGE  neon128r2_inv_next_size

	MOVD $0, R0

neon128r2_inv_inner_loop:
	CMP  R15, R0
	BGE  neon128r2_inv_next_base

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
	B    neon128r2_inv_inner_loop

neon128r2_inv_next_base:
	ADD  R14, R17, R17
	B    neon128r2_inv_base_loop

neon128r2_inv_next_size:
	LSL  $1, R14, R14
	B    neon128r2_inv_size_loop

neon128r2_inv_done:
	CMP  R8, R20
	BEQ  neon128r2_inv_scale

	MOVD $0, R0
neon128r2_inv_copy_loop:
	CMP  $128, R0
	BGE  neon128r2_inv_scale
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon128r2_inv_copy_loop

neon128r2_inv_scale:
	MOVD $·neonInv128(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon128r2_inv_scale_loop:
	CMP  $128, R0
	BGE  neon128r2_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon128r2_inv_scale_loop

neon128r2_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon128r2_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Data Section
// ===========================================================================

// Bit-reversal permutation for size 128 (7-bit reversal)
DATA bitrev_size128_radix2<>+0x000(SB)/8, $0    // bitrev[0] = 0
DATA bitrev_size128_radix2<>+0x008(SB)/8, $64   // bitrev[1] = 64
DATA bitrev_size128_radix2<>+0x010(SB)/8, $32   // bitrev[2] = 32
DATA bitrev_size128_radix2<>+0x018(SB)/8, $96   // bitrev[3] = 96
DATA bitrev_size128_radix2<>+0x020(SB)/8, $16   // bitrev[4] = 16
DATA bitrev_size128_radix2<>+0x028(SB)/8, $80   // bitrev[5] = 80
DATA bitrev_size128_radix2<>+0x030(SB)/8, $48   // bitrev[6] = 48
DATA bitrev_size128_radix2<>+0x038(SB)/8, $112  // bitrev[7] = 112
DATA bitrev_size128_radix2<>+0x040(SB)/8, $8    // bitrev[8] = 8
DATA bitrev_size128_radix2<>+0x048(SB)/8, $72   // bitrev[9] = 72
DATA bitrev_size128_radix2<>+0x050(SB)/8, $40   // bitrev[10] = 40
DATA bitrev_size128_radix2<>+0x058(SB)/8, $104  // bitrev[11] = 104
DATA bitrev_size128_radix2<>+0x060(SB)/8, $24   // bitrev[12] = 24
DATA bitrev_size128_radix2<>+0x068(SB)/8, $88   // bitrev[13] = 88
DATA bitrev_size128_radix2<>+0x070(SB)/8, $56   // bitrev[14] = 56
DATA bitrev_size128_radix2<>+0x078(SB)/8, $120  // bitrev[15] = 120
DATA bitrev_size128_radix2<>+0x080(SB)/8, $4    // bitrev[16] = 4
DATA bitrev_size128_radix2<>+0x088(SB)/8, $68   // bitrev[17] = 68
DATA bitrev_size128_radix2<>+0x090(SB)/8, $36   // bitrev[18] = 36
DATA bitrev_size128_radix2<>+0x098(SB)/8, $100  // bitrev[19] = 100
DATA bitrev_size128_radix2<>+0x0A0(SB)/8, $20   // bitrev[20] = 20
DATA bitrev_size128_radix2<>+0x0A8(SB)/8, $84   // bitrev[21] = 84
DATA bitrev_size128_radix2<>+0x0B0(SB)/8, $52   // bitrev[22] = 52
DATA bitrev_size128_radix2<>+0x0B8(SB)/8, $116  // bitrev[23] = 116
DATA bitrev_size128_radix2<>+0x0C0(SB)/8, $12   // bitrev[24] = 12
DATA bitrev_size128_radix2<>+0x0C8(SB)/8, $76   // bitrev[25] = 76
DATA bitrev_size128_radix2<>+0x0D0(SB)/8, $44   // bitrev[26] = 44
DATA bitrev_size128_radix2<>+0x0D8(SB)/8, $108  // bitrev[27] = 108
DATA bitrev_size128_radix2<>+0x0E0(SB)/8, $28   // bitrev[28] = 28
DATA bitrev_size128_radix2<>+0x0E8(SB)/8, $92   // bitrev[29] = 92
DATA bitrev_size128_radix2<>+0x0F0(SB)/8, $60   // bitrev[30] = 60
DATA bitrev_size128_radix2<>+0x0F8(SB)/8, $124  // bitrev[31] = 124
DATA bitrev_size128_radix2<>+0x100(SB)/8, $2    // bitrev[32] = 2
DATA bitrev_size128_radix2<>+0x108(SB)/8, $66   // bitrev[33] = 66
DATA bitrev_size128_radix2<>+0x110(SB)/8, $34   // bitrev[34] = 34
DATA bitrev_size128_radix2<>+0x118(SB)/8, $98   // bitrev[35] = 98
DATA bitrev_size128_radix2<>+0x120(SB)/8, $18   // bitrev[36] = 18
DATA bitrev_size128_radix2<>+0x128(SB)/8, $82   // bitrev[37] = 82
DATA bitrev_size128_radix2<>+0x130(SB)/8, $50   // bitrev[38] = 50
DATA bitrev_size128_radix2<>+0x138(SB)/8, $114  // bitrev[39] = 114
DATA bitrev_size128_radix2<>+0x140(SB)/8, $10   // bitrev[40] = 10
DATA bitrev_size128_radix2<>+0x148(SB)/8, $74   // bitrev[41] = 74
DATA bitrev_size128_radix2<>+0x150(SB)/8, $42   // bitrev[42] = 42
DATA bitrev_size128_radix2<>+0x158(SB)/8, $106  // bitrev[43] = 106
DATA bitrev_size128_radix2<>+0x160(SB)/8, $26   // bitrev[44] = 26
DATA bitrev_size128_radix2<>+0x168(SB)/8, $90   // bitrev[45] = 90
DATA bitrev_size128_radix2<>+0x170(SB)/8, $58   // bitrev[46] = 58
DATA bitrev_size128_radix2<>+0x178(SB)/8, $122  // bitrev[47] = 122
DATA bitrev_size128_radix2<>+0x180(SB)/8, $6    // bitrev[48] = 6
DATA bitrev_size128_radix2<>+0x188(SB)/8, $70   // bitrev[49] = 70
DATA bitrev_size128_radix2<>+0x190(SB)/8, $38   // bitrev[50] = 38
DATA bitrev_size128_radix2<>+0x198(SB)/8, $102  // bitrev[51] = 102
DATA bitrev_size128_radix2<>+0x1A0(SB)/8, $22   // bitrev[52] = 22
DATA bitrev_size128_radix2<>+0x1A8(SB)/8, $86   // bitrev[53] = 86
DATA bitrev_size128_radix2<>+0x1B0(SB)/8, $54   // bitrev[54] = 54
DATA bitrev_size128_radix2<>+0x1B8(SB)/8, $118  // bitrev[55] = 118
DATA bitrev_size128_radix2<>+0x1C0(SB)/8, $14   // bitrev[56] = 14
DATA bitrev_size128_radix2<>+0x1C8(SB)/8, $78   // bitrev[57] = 78
DATA bitrev_size128_radix2<>+0x1D0(SB)/8, $46   // bitrev[58] = 46
DATA bitrev_size128_radix2<>+0x1D8(SB)/8, $110  // bitrev[59] = 110
DATA bitrev_size128_radix2<>+0x1E0(SB)/8, $30   // bitrev[60] = 30
DATA bitrev_size128_radix2<>+0x1E8(SB)/8, $94   // bitrev[61] = 94
DATA bitrev_size128_radix2<>+0x1F0(SB)/8, $62   // bitrev[62] = 62
DATA bitrev_size128_radix2<>+0x1F8(SB)/8, $126  // bitrev[63] = 126
DATA bitrev_size128_radix2<>+0x200(SB)/8, $1    // bitrev[64] = 1
DATA bitrev_size128_radix2<>+0x208(SB)/8, $65   // bitrev[65] = 65
DATA bitrev_size128_radix2<>+0x210(SB)/8, $33   // bitrev[66] = 33
DATA bitrev_size128_radix2<>+0x218(SB)/8, $97   // bitrev[67] = 97
DATA bitrev_size128_radix2<>+0x220(SB)/8, $17   // bitrev[68] = 17
DATA bitrev_size128_radix2<>+0x228(SB)/8, $81   // bitrev[69] = 81
DATA bitrev_size128_radix2<>+0x230(SB)/8, $49   // bitrev[70] = 49
DATA bitrev_size128_radix2<>+0x238(SB)/8, $113  // bitrev[71] = 113
DATA bitrev_size128_radix2<>+0x240(SB)/8, $9    // bitrev[72] = 9
DATA bitrev_size128_radix2<>+0x248(SB)/8, $73   // bitrev[73] = 73
DATA bitrev_size128_radix2<>+0x250(SB)/8, $41   // bitrev[74] = 41
DATA bitrev_size128_radix2<>+0x258(SB)/8, $105  // bitrev[75] = 105
DATA bitrev_size128_radix2<>+0x260(SB)/8, $25   // bitrev[76] = 25
DATA bitrev_size128_radix2<>+0x268(SB)/8, $89   // bitrev[77] = 89
DATA bitrev_size128_radix2<>+0x270(SB)/8, $57   // bitrev[78] = 57
DATA bitrev_size128_radix2<>+0x278(SB)/8, $121  // bitrev[79] = 121
DATA bitrev_size128_radix2<>+0x280(SB)/8, $5    // bitrev[80] = 5
DATA bitrev_size128_radix2<>+0x288(SB)/8, $69   // bitrev[81] = 69
DATA bitrev_size128_radix2<>+0x290(SB)/8, $37   // bitrev[82] = 37
DATA bitrev_size128_radix2<>+0x298(SB)/8, $101  // bitrev[83] = 101
DATA bitrev_size128_radix2<>+0x2A0(SB)/8, $21   // bitrev[84] = 21
DATA bitrev_size128_radix2<>+0x2A8(SB)/8, $85   // bitrev[85] = 85
DATA bitrev_size128_radix2<>+0x2B0(SB)/8, $53   // bitrev[86] = 53
DATA bitrev_size128_radix2<>+0x2B8(SB)/8, $117  // bitrev[87] = 117
DATA bitrev_size128_radix2<>+0x2C0(SB)/8, $13   // bitrev[88] = 13
DATA bitrev_size128_radix2<>+0x2C8(SB)/8, $77   // bitrev[89] = 77
DATA bitrev_size128_radix2<>+0x2D0(SB)/8, $45   // bitrev[90] = 45
DATA bitrev_size128_radix2<>+0x2D8(SB)/8, $109  // bitrev[91] = 109
DATA bitrev_size128_radix2<>+0x2E0(SB)/8, $29   // bitrev[92] = 29
DATA bitrev_size128_radix2<>+0x2E8(SB)/8, $93   // bitrev[93] = 93
DATA bitrev_size128_radix2<>+0x2F0(SB)/8, $61   // bitrev[94] = 61
DATA bitrev_size128_radix2<>+0x2F8(SB)/8, $125  // bitrev[95] = 125
DATA bitrev_size128_radix2<>+0x300(SB)/8, $3    // bitrev[96] = 3
DATA bitrev_size128_radix2<>+0x308(SB)/8, $67   // bitrev[97] = 67
DATA bitrev_size128_radix2<>+0x310(SB)/8, $35   // bitrev[98] = 35
DATA bitrev_size128_radix2<>+0x318(SB)/8, $99   // bitrev[99] = 99
DATA bitrev_size128_radix2<>+0x320(SB)/8, $19   // bitrev[100] = 19
DATA bitrev_size128_radix2<>+0x328(SB)/8, $83   // bitrev[101] = 83
DATA bitrev_size128_radix2<>+0x330(SB)/8, $51   // bitrev[102] = 51
DATA bitrev_size128_radix2<>+0x338(SB)/8, $115  // bitrev[103] = 115
DATA bitrev_size128_radix2<>+0x340(SB)/8, $11   // bitrev[104] = 11
DATA bitrev_size128_radix2<>+0x348(SB)/8, $75   // bitrev[105] = 75
DATA bitrev_size128_radix2<>+0x350(SB)/8, $43   // bitrev[106] = 43
DATA bitrev_size128_radix2<>+0x358(SB)/8, $107  // bitrev[107] = 107
DATA bitrev_size128_radix2<>+0x360(SB)/8, $27   // bitrev[108] = 27
DATA bitrev_size128_radix2<>+0x368(SB)/8, $91   // bitrev[109] = 91
DATA bitrev_size128_radix2<>+0x370(SB)/8, $59   // bitrev[110] = 59
DATA bitrev_size128_radix2<>+0x378(SB)/8, $123  // bitrev[111] = 123
DATA bitrev_size128_radix2<>+0x380(SB)/8, $7    // bitrev[112] = 7
DATA bitrev_size128_radix2<>+0x388(SB)/8, $71   // bitrev[113] = 71
DATA bitrev_size128_radix2<>+0x390(SB)/8, $39   // bitrev[114] = 39
DATA bitrev_size128_radix2<>+0x398(SB)/8, $103  // bitrev[115] = 103
DATA bitrev_size128_radix2<>+0x3A0(SB)/8, $23   // bitrev[116] = 23
DATA bitrev_size128_radix2<>+0x3A8(SB)/8, $87   // bitrev[117] = 87
DATA bitrev_size128_radix2<>+0x3B0(SB)/8, $55   // bitrev[118] = 55
DATA bitrev_size128_radix2<>+0x3B8(SB)/8, $119  // bitrev[119] = 119
DATA bitrev_size128_radix2<>+0x3C0(SB)/8, $15   // bitrev[120] = 15
DATA bitrev_size128_radix2<>+0x3C8(SB)/8, $79   // bitrev[121] = 79
DATA bitrev_size128_radix2<>+0x3D0(SB)/8, $47   // bitrev[122] = 47
DATA bitrev_size128_radix2<>+0x3D8(SB)/8, $111  // bitrev[123] = 111
DATA bitrev_size128_radix2<>+0x3E0(SB)/8, $31   // bitrev[124] = 31
DATA bitrev_size128_radix2<>+0x3E8(SB)/8, $95   // bitrev[125] = 95
DATA bitrev_size128_radix2<>+0x3F0(SB)/8, $63   // bitrev[126] = 63
DATA bitrev_size128_radix2<>+0x3F8(SB)/8, $127  // bitrev[127] = 127
GLOBL bitrev_size128_radix2<>(SB), RODATA, $1024

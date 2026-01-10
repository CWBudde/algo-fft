//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-256 Radix-2 FFT Kernels for ARM64
// ===========================================================================

#include "textflag.h"

DATA ·neonInv256+0(SB)/4, $0x3b800000 // 1/256
GLOBL ·neonInv256(SB), RODATA, $4

// Forward transform, size 256, complex64, radix-2
TEXT ·ForwardNEONSize256Radix2Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
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

	MOVD $bitrev_size256_radix2<>(SB), R12

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
	MOVB R0, ret+96(FP)
	RET

neon256r2_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Inverse transform, size 256, complex64, radix-2
TEXT ·InverseNEONSize256Radix2Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
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

	MOVD $bitrev_size256_radix2<>(SB), R12

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
	MOVB R0, ret+96(FP)
	RET

neon256r2_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Data Section
// ===========================================================================

// Bit-reversal permutation for size 256 (8-bit reversal)
DATA bitrev_size256_radix2<>+0x000(SB)/8, $0    // bitrev[0] = 0
DATA bitrev_size256_radix2<>+0x008(SB)/8, $128  // bitrev[1] = 128
DATA bitrev_size256_radix2<>+0x010(SB)/8, $64   // bitrev[2] = 64
DATA bitrev_size256_radix2<>+0x018(SB)/8, $192  // bitrev[3] = 192
DATA bitrev_size256_radix2<>+0x020(SB)/8, $32   // bitrev[4] = 32
DATA bitrev_size256_radix2<>+0x028(SB)/8, $160  // bitrev[5] = 160
DATA bitrev_size256_radix2<>+0x030(SB)/8, $96   // bitrev[6] = 96
DATA bitrev_size256_radix2<>+0x038(SB)/8, $224  // bitrev[7] = 224
DATA bitrev_size256_radix2<>+0x040(SB)/8, $16   // bitrev[8] = 16
DATA bitrev_size256_radix2<>+0x048(SB)/8, $144  // bitrev[9] = 144
DATA bitrev_size256_radix2<>+0x050(SB)/8, $80   // bitrev[10] = 80
DATA bitrev_size256_radix2<>+0x058(SB)/8, $208  // bitrev[11] = 208
DATA bitrev_size256_radix2<>+0x060(SB)/8, $48   // bitrev[12] = 48
DATA bitrev_size256_radix2<>+0x068(SB)/8, $176  // bitrev[13] = 176
DATA bitrev_size256_radix2<>+0x070(SB)/8, $112  // bitrev[14] = 112
DATA bitrev_size256_radix2<>+0x078(SB)/8, $240  // bitrev[15] = 240
DATA bitrev_size256_radix2<>+0x080(SB)/8, $8    // bitrev[16] = 8
DATA bitrev_size256_radix2<>+0x088(SB)/8, $136  // bitrev[17] = 136
DATA bitrev_size256_radix2<>+0x090(SB)/8, $72   // bitrev[18] = 72
DATA bitrev_size256_radix2<>+0x098(SB)/8, $200  // bitrev[19] = 200
DATA bitrev_size256_radix2<>+0x0A0(SB)/8, $40   // bitrev[20] = 40
DATA bitrev_size256_radix2<>+0x0A8(SB)/8, $168  // bitrev[21] = 168
DATA bitrev_size256_radix2<>+0x0B0(SB)/8, $104  // bitrev[22] = 104
DATA bitrev_size256_radix2<>+0x0B8(SB)/8, $232  // bitrev[23] = 232
DATA bitrev_size256_radix2<>+0x0C0(SB)/8, $24   // bitrev[24] = 24
DATA bitrev_size256_radix2<>+0x0C8(SB)/8, $152  // bitrev[25] = 152
DATA bitrev_size256_radix2<>+0x0D0(SB)/8, $88   // bitrev[26] = 88
DATA bitrev_size256_radix2<>+0x0D8(SB)/8, $216  // bitrev[27] = 216
DATA bitrev_size256_radix2<>+0x0E0(SB)/8, $56   // bitrev[28] = 56
DATA bitrev_size256_radix2<>+0x0E8(SB)/8, $184  // bitrev[29] = 184
DATA bitrev_size256_radix2<>+0x0F0(SB)/8, $120  // bitrev[30] = 120
DATA bitrev_size256_radix2<>+0x0F8(SB)/8, $248  // bitrev[31] = 248
DATA bitrev_size256_radix2<>+0x100(SB)/8, $4    // bitrev[32] = 4
DATA bitrev_size256_radix2<>+0x108(SB)/8, $132  // bitrev[33] = 132
DATA bitrev_size256_radix2<>+0x110(SB)/8, $68   // bitrev[34] = 68
DATA bitrev_size256_radix2<>+0x118(SB)/8, $196  // bitrev[35] = 196
DATA bitrev_size256_radix2<>+0x120(SB)/8, $36   // bitrev[36] = 36
DATA bitrev_size256_radix2<>+0x128(SB)/8, $164  // bitrev[37] = 164
DATA bitrev_size256_radix2<>+0x130(SB)/8, $100  // bitrev[38] = 100
DATA bitrev_size256_radix2<>+0x138(SB)/8, $228  // bitrev[39] = 228
DATA bitrev_size256_radix2<>+0x140(SB)/8, $20   // bitrev[40] = 20
DATA bitrev_size256_radix2<>+0x148(SB)/8, $148  // bitrev[41] = 148
DATA bitrev_size256_radix2<>+0x150(SB)/8, $84   // bitrev[42] = 84
DATA bitrev_size256_radix2<>+0x158(SB)/8, $212  // bitrev[43] = 212
DATA bitrev_size256_radix2<>+0x160(SB)/8, $52   // bitrev[44] = 52
DATA bitrev_size256_radix2<>+0x168(SB)/8, $180  // bitrev[45] = 180
DATA bitrev_size256_radix2<>+0x170(SB)/8, $116  // bitrev[46] = 116
DATA bitrev_size256_radix2<>+0x178(SB)/8, $244  // bitrev[47] = 244
DATA bitrev_size256_radix2<>+0x180(SB)/8, $12   // bitrev[48] = 12
DATA bitrev_size256_radix2<>+0x188(SB)/8, $140  // bitrev[49] = 140
DATA bitrev_size256_radix2<>+0x190(SB)/8, $76   // bitrev[50] = 76
DATA bitrev_size256_radix2<>+0x198(SB)/8, $204  // bitrev[51] = 204
DATA bitrev_size256_radix2<>+0x1A0(SB)/8, $44   // bitrev[52] = 44
DATA bitrev_size256_radix2<>+0x1A8(SB)/8, $172  // bitrev[53] = 172
DATA bitrev_size256_radix2<>+0x1B0(SB)/8, $108  // bitrev[54] = 108
DATA bitrev_size256_radix2<>+0x1B8(SB)/8, $236  // bitrev[55] = 236
DATA bitrev_size256_radix2<>+0x1C0(SB)/8, $28   // bitrev[56] = 28
DATA bitrev_size256_radix2<>+0x1C8(SB)/8, $156  // bitrev[57] = 156
DATA bitrev_size256_radix2<>+0x1D0(SB)/8, $92   // bitrev[58] = 92
DATA bitrev_size256_radix2<>+0x1D8(SB)/8, $220  // bitrev[59] = 220
DATA bitrev_size256_radix2<>+0x1E0(SB)/8, $60   // bitrev[60] = 60
DATA bitrev_size256_radix2<>+0x1E8(SB)/8, $188  // bitrev[61] = 188
DATA bitrev_size256_radix2<>+0x1F0(SB)/8, $124  // bitrev[62] = 124
DATA bitrev_size256_radix2<>+0x1F8(SB)/8, $252  // bitrev[63] = 252
DATA bitrev_size256_radix2<>+0x200(SB)/8, $2    // bitrev[64] = 2
DATA bitrev_size256_radix2<>+0x208(SB)/8, $130  // bitrev[65] = 130
DATA bitrev_size256_radix2<>+0x210(SB)/8, $66   // bitrev[66] = 66
DATA bitrev_size256_radix2<>+0x218(SB)/8, $194  // bitrev[67] = 194
DATA bitrev_size256_radix2<>+0x220(SB)/8, $34   // bitrev[68] = 34
DATA bitrev_size256_radix2<>+0x228(SB)/8, $162  // bitrev[69] = 162
DATA bitrev_size256_radix2<>+0x230(SB)/8, $98   // bitrev[70] = 98
DATA bitrev_size256_radix2<>+0x238(SB)/8, $226  // bitrev[71] = 226
DATA bitrev_size256_radix2<>+0x240(SB)/8, $18   // bitrev[72] = 18
DATA bitrev_size256_radix2<>+0x248(SB)/8, $146  // bitrev[73] = 146
DATA bitrev_size256_radix2<>+0x250(SB)/8, $82   // bitrev[74] = 82
DATA bitrev_size256_radix2<>+0x258(SB)/8, $210  // bitrev[75] = 210
DATA bitrev_size256_radix2<>+0x260(SB)/8, $50   // bitrev[76] = 50
DATA bitrev_size256_radix2<>+0x268(SB)/8, $178  // bitrev[77] = 178
DATA bitrev_size256_radix2<>+0x270(SB)/8, $114  // bitrev[78] = 114
DATA bitrev_size256_radix2<>+0x278(SB)/8, $242  // bitrev[79] = 242
DATA bitrev_size256_radix2<>+0x280(SB)/8, $10   // bitrev[80] = 10
DATA bitrev_size256_radix2<>+0x288(SB)/8, $138  // bitrev[81] = 138
DATA bitrev_size256_radix2<>+0x290(SB)/8, $74   // bitrev[82] = 74
DATA bitrev_size256_radix2<>+0x298(SB)/8, $202  // bitrev[83] = 202
DATA bitrev_size256_radix2<>+0x2A0(SB)/8, $42   // bitrev[84] = 42
DATA bitrev_size256_radix2<>+0x2A8(SB)/8, $170  // bitrev[85] = 170
DATA bitrev_size256_radix2<>+0x2B0(SB)/8, $106  // bitrev[86] = 106
DATA bitrev_size256_radix2<>+0x2B8(SB)/8, $234  // bitrev[87] = 234
DATA bitrev_size256_radix2<>+0x2C0(SB)/8, $26   // bitrev[88] = 26
DATA bitrev_size256_radix2<>+0x2C8(SB)/8, $154  // bitrev[89] = 154
DATA bitrev_size256_radix2<>+0x2D0(SB)/8, $90   // bitrev[90] = 90
DATA bitrev_size256_radix2<>+0x2D8(SB)/8, $218  // bitrev[91] = 218
DATA bitrev_size256_radix2<>+0x2E0(SB)/8, $58   // bitrev[92] = 58
DATA bitrev_size256_radix2<>+0x2E8(SB)/8, $186  // bitrev[93] = 186
DATA bitrev_size256_radix2<>+0x2F0(SB)/8, $122  // bitrev[94] = 122
DATA bitrev_size256_radix2<>+0x2F8(SB)/8, $250  // bitrev[95] = 250
DATA bitrev_size256_radix2<>+0x300(SB)/8, $6    // bitrev[96] = 6
DATA bitrev_size256_radix2<>+0x308(SB)/8, $134  // bitrev[97] = 134
DATA bitrev_size256_radix2<>+0x310(SB)/8, $70   // bitrev[98] = 70
DATA bitrev_size256_radix2<>+0x318(SB)/8, $198  // bitrev[99] = 198
DATA bitrev_size256_radix2<>+0x320(SB)/8, $38   // bitrev[100] = 38
DATA bitrev_size256_radix2<>+0x328(SB)/8, $166  // bitrev[101] = 166
DATA bitrev_size256_radix2<>+0x330(SB)/8, $102  // bitrev[102] = 102
DATA bitrev_size256_radix2<>+0x338(SB)/8, $230  // bitrev[103] = 230
DATA bitrev_size256_radix2<>+0x340(SB)/8, $22   // bitrev[104] = 22
DATA bitrev_size256_radix2<>+0x348(SB)/8, $150  // bitrev[105] = 150
DATA bitrev_size256_radix2<>+0x350(SB)/8, $86   // bitrev[106] = 86
DATA bitrev_size256_radix2<>+0x358(SB)/8, $214  // bitrev[107] = 214
DATA bitrev_size256_radix2<>+0x360(SB)/8, $54   // bitrev[108] = 54
DATA bitrev_size256_radix2<>+0x368(SB)/8, $182  // bitrev[109] = 182
DATA bitrev_size256_radix2<>+0x370(SB)/8, $118  // bitrev[110] = 118
DATA bitrev_size256_radix2<>+0x378(SB)/8, $246  // bitrev[111] = 246
DATA bitrev_size256_radix2<>+0x380(SB)/8, $14   // bitrev[112] = 14
DATA bitrev_size256_radix2<>+0x388(SB)/8, $142  // bitrev[113] = 142
DATA bitrev_size256_radix2<>+0x390(SB)/8, $78   // bitrev[114] = 78
DATA bitrev_size256_radix2<>+0x398(SB)/8, $206  // bitrev[115] = 206
DATA bitrev_size256_radix2<>+0x3A0(SB)/8, $46   // bitrev[116] = 46
DATA bitrev_size256_radix2<>+0x3A8(SB)/8, $174  // bitrev[117] = 174
DATA bitrev_size256_radix2<>+0x3B0(SB)/8, $110  // bitrev[118] = 110
DATA bitrev_size256_radix2<>+0x3B8(SB)/8, $238  // bitrev[119] = 238
DATA bitrev_size256_radix2<>+0x3C0(SB)/8, $30   // bitrev[120] = 30
DATA bitrev_size256_radix2<>+0x3C8(SB)/8, $158  // bitrev[121] = 158
DATA bitrev_size256_radix2<>+0x3D0(SB)/8, $94   // bitrev[122] = 94
DATA bitrev_size256_radix2<>+0x3D8(SB)/8, $222  // bitrev[123] = 222
DATA bitrev_size256_radix2<>+0x3E0(SB)/8, $62   // bitrev[124] = 62
DATA bitrev_size256_radix2<>+0x3E8(SB)/8, $190  // bitrev[125] = 190
DATA bitrev_size256_radix2<>+0x3F0(SB)/8, $126  // bitrev[126] = 126
DATA bitrev_size256_radix2<>+0x3F8(SB)/8, $254  // bitrev[127] = 254
DATA bitrev_size256_radix2<>+0x400(SB)/8, $1    // bitrev[128] = 1
DATA bitrev_size256_radix2<>+0x408(SB)/8, $129  // bitrev[129] = 129
DATA bitrev_size256_radix2<>+0x410(SB)/8, $65   // bitrev[130] = 65
DATA bitrev_size256_radix2<>+0x418(SB)/8, $193  // bitrev[131] = 193
DATA bitrev_size256_radix2<>+0x420(SB)/8, $33   // bitrev[132] = 33
DATA bitrev_size256_radix2<>+0x428(SB)/8, $161  // bitrev[133] = 161
DATA bitrev_size256_radix2<>+0x430(SB)/8, $97   // bitrev[134] = 97
DATA bitrev_size256_radix2<>+0x438(SB)/8, $225  // bitrev[135] = 225
DATA bitrev_size256_radix2<>+0x440(SB)/8, $17   // bitrev[136] = 17
DATA bitrev_size256_radix2<>+0x448(SB)/8, $145  // bitrev[137] = 145
DATA bitrev_size256_radix2<>+0x450(SB)/8, $81   // bitrev[138] = 81
DATA bitrev_size256_radix2<>+0x458(SB)/8, $209  // bitrev[139] = 209
DATA bitrev_size256_radix2<>+0x460(SB)/8, $49   // bitrev[140] = 49
DATA bitrev_size256_radix2<>+0x468(SB)/8, $177  // bitrev[141] = 177
DATA bitrev_size256_radix2<>+0x470(SB)/8, $113  // bitrev[142] = 113
DATA bitrev_size256_radix2<>+0x478(SB)/8, $241  // bitrev[143] = 241
DATA bitrev_size256_radix2<>+0x480(SB)/8, $9    // bitrev[144] = 9
DATA bitrev_size256_radix2<>+0x488(SB)/8, $137  // bitrev[145] = 137
DATA bitrev_size256_radix2<>+0x490(SB)/8, $73   // bitrev[146] = 73
DATA bitrev_size256_radix2<>+0x498(SB)/8, $201  // bitrev[147] = 201
DATA bitrev_size256_radix2<>+0x4A0(SB)/8, $41   // bitrev[148] = 41
DATA bitrev_size256_radix2<>+0x4A8(SB)/8, $169  // bitrev[149] = 169
DATA bitrev_size256_radix2<>+0x4B0(SB)/8, $105  // bitrev[150] = 105
DATA bitrev_size256_radix2<>+0x4B8(SB)/8, $233  // bitrev[151] = 233
DATA bitrev_size256_radix2<>+0x4C0(SB)/8, $25   // bitrev[152] = 25
DATA bitrev_size256_radix2<>+0x4C8(SB)/8, $153  // bitrev[153] = 153
DATA bitrev_size256_radix2<>+0x4D0(SB)/8, $89   // bitrev[154] = 89
DATA bitrev_size256_radix2<>+0x4D8(SB)/8, $217  // bitrev[155] = 217
DATA bitrev_size256_radix2<>+0x4E0(SB)/8, $57   // bitrev[156] = 57
DATA bitrev_size256_radix2<>+0x4E8(SB)/8, $185  // bitrev[157] = 185
DATA bitrev_size256_radix2<>+0x4F0(SB)/8, $121  // bitrev[158] = 121
DATA bitrev_size256_radix2<>+0x4F8(SB)/8, $249  // bitrev[159] = 249
DATA bitrev_size256_radix2<>+0x500(SB)/8, $5    // bitrev[160] = 5
DATA bitrev_size256_radix2<>+0x508(SB)/8, $133  // bitrev[161] = 133
DATA bitrev_size256_radix2<>+0x510(SB)/8, $69   // bitrev[162] = 69
DATA bitrev_size256_radix2<>+0x518(SB)/8, $197  // bitrev[163] = 197
DATA bitrev_size256_radix2<>+0x520(SB)/8, $37   // bitrev[164] = 37
DATA bitrev_size256_radix2<>+0x528(SB)/8, $165  // bitrev[165] = 165
DATA bitrev_size256_radix2<>+0x530(SB)/8, $101  // bitrev[166] = 101
DATA bitrev_size256_radix2<>+0x538(SB)/8, $229  // bitrev[167] = 229
DATA bitrev_size256_radix2<>+0x540(SB)/8, $21   // bitrev[168] = 21
DATA bitrev_size256_radix2<>+0x548(SB)/8, $149  // bitrev[169] = 149
DATA bitrev_size256_radix2<>+0x550(SB)/8, $85   // bitrev[170] = 85
DATA bitrev_size256_radix2<>+0x558(SB)/8, $213  // bitrev[171] = 213
DATA bitrev_size256_radix2<>+0x560(SB)/8, $53   // bitrev[172] = 53
DATA bitrev_size256_radix2<>+0x568(SB)/8, $181  // bitrev[173] = 181
DATA bitrev_size256_radix2<>+0x570(SB)/8, $117  // bitrev[174] = 117
DATA bitrev_size256_radix2<>+0x578(SB)/8, $245  // bitrev[175] = 245
DATA bitrev_size256_radix2<>+0x580(SB)/8, $13   // bitrev[176] = 13
DATA bitrev_size256_radix2<>+0x588(SB)/8, $141  // bitrev[177] = 141
DATA bitrev_size256_radix2<>+0x590(SB)/8, $77   // bitrev[178] = 77
DATA bitrev_size256_radix2<>+0x598(SB)/8, $205  // bitrev[179] = 205
DATA bitrev_size256_radix2<>+0x5A0(SB)/8, $45   // bitrev[180] = 45
DATA bitrev_size256_radix2<>+0x5A8(SB)/8, $173  // bitrev[181] = 173
DATA bitrev_size256_radix2<>+0x5B0(SB)/8, $109  // bitrev[182] = 109
DATA bitrev_size256_radix2<>+0x5B8(SB)/8, $237  // bitrev[183] = 237
DATA bitrev_size256_radix2<>+0x5C0(SB)/8, $29   // bitrev[184] = 29
DATA bitrev_size256_radix2<>+0x5C8(SB)/8, $157  // bitrev[185] = 157
DATA bitrev_size256_radix2<>+0x5D0(SB)/8, $93   // bitrev[186] = 93
DATA bitrev_size256_radix2<>+0x5D8(SB)/8, $221  // bitrev[187] = 221
DATA bitrev_size256_radix2<>+0x5E0(SB)/8, $61   // bitrev[188] = 61
DATA bitrev_size256_radix2<>+0x5E8(SB)/8, $189  // bitrev[189] = 189
DATA bitrev_size256_radix2<>+0x5F0(SB)/8, $125  // bitrev[190] = 125
DATA bitrev_size256_radix2<>+0x5F8(SB)/8, $253  // bitrev[191] = 253
DATA bitrev_size256_radix2<>+0x600(SB)/8, $3    // bitrev[192] = 3
DATA bitrev_size256_radix2<>+0x608(SB)/8, $131  // bitrev[193] = 131
DATA bitrev_size256_radix2<>+0x610(SB)/8, $67   // bitrev[194] = 67
DATA bitrev_size256_radix2<>+0x618(SB)/8, $195  // bitrev[195] = 195
DATA bitrev_size256_radix2<>+0x620(SB)/8, $35   // bitrev[196] = 35
DATA bitrev_size256_radix2<>+0x628(SB)/8, $163  // bitrev[197] = 163
DATA bitrev_size256_radix2<>+0x630(SB)/8, $99   // bitrev[198] = 99
DATA bitrev_size256_radix2<>+0x638(SB)/8, $227  // bitrev[199] = 227
DATA bitrev_size256_radix2<>+0x640(SB)/8, $19   // bitrev[200] = 19
DATA bitrev_size256_radix2<>+0x648(SB)/8, $147  // bitrev[201] = 147
DATA bitrev_size256_radix2<>+0x650(SB)/8, $83   // bitrev[202] = 83
DATA bitrev_size256_radix2<>+0x658(SB)/8, $211  // bitrev[203] = 211
DATA bitrev_size256_radix2<>+0x660(SB)/8, $51   // bitrev[204] = 51
DATA bitrev_size256_radix2<>+0x668(SB)/8, $179  // bitrev[205] = 179
DATA bitrev_size256_radix2<>+0x670(SB)/8, $115  // bitrev[206] = 115
DATA bitrev_size256_radix2<>+0x678(SB)/8, $243  // bitrev[207] = 243
DATA bitrev_size256_radix2<>+0x680(SB)/8, $11   // bitrev[208] = 11
DATA bitrev_size256_radix2<>+0x688(SB)/8, $139  // bitrev[209] = 139
DATA bitrev_size256_radix2<>+0x690(SB)/8, $75   // bitrev[210] = 75
DATA bitrev_size256_radix2<>+0x698(SB)/8, $203  // bitrev[211] = 203
DATA bitrev_size256_radix2<>+0x6A0(SB)/8, $43   // bitrev[212] = 43
DATA bitrev_size256_radix2<>+0x6A8(SB)/8, $171  // bitrev[213] = 171
DATA bitrev_size256_radix2<>+0x6B0(SB)/8, $107  // bitrev[214] = 107
DATA bitrev_size256_radix2<>+0x6B8(SB)/8, $235  // bitrev[215] = 235
DATA bitrev_size256_radix2<>+0x6C0(SB)/8, $27   // bitrev[216] = 27
DATA bitrev_size256_radix2<>+0x6C8(SB)/8, $155  // bitrev[217] = 155
DATA bitrev_size256_radix2<>+0x6D0(SB)/8, $91   // bitrev[218] = 91
DATA bitrev_size256_radix2<>+0x6D8(SB)/8, $219  // bitrev[219] = 219
DATA bitrev_size256_radix2<>+0x6E0(SB)/8, $59   // bitrev[220] = 59
DATA bitrev_size256_radix2<>+0x6E8(SB)/8, $187  // bitrev[221] = 187
DATA bitrev_size256_radix2<>+0x6F0(SB)/8, $123  // bitrev[222] = 123
DATA bitrev_size256_radix2<>+0x6F8(SB)/8, $251  // bitrev[223] = 251
DATA bitrev_size256_radix2<>+0x700(SB)/8, $7    // bitrev[224] = 7
DATA bitrev_size256_radix2<>+0x708(SB)/8, $135  // bitrev[225] = 135
DATA bitrev_size256_radix2<>+0x710(SB)/8, $71   // bitrev[226] = 71
DATA bitrev_size256_radix2<>+0x718(SB)/8, $199  // bitrev[227] = 199
DATA bitrev_size256_radix2<>+0x720(SB)/8, $39   // bitrev[228] = 39
DATA bitrev_size256_radix2<>+0x728(SB)/8, $167  // bitrev[229] = 167
DATA bitrev_size256_radix2<>+0x730(SB)/8, $103  // bitrev[230] = 103
DATA bitrev_size256_radix2<>+0x738(SB)/8, $231  // bitrev[231] = 231
DATA bitrev_size256_radix2<>+0x740(SB)/8, $23   // bitrev[232] = 23
DATA bitrev_size256_radix2<>+0x748(SB)/8, $151  // bitrev[233] = 151
DATA bitrev_size256_radix2<>+0x750(SB)/8, $87   // bitrev[234] = 87
DATA bitrev_size256_radix2<>+0x758(SB)/8, $215  // bitrev[235] = 215
DATA bitrev_size256_radix2<>+0x760(SB)/8, $55   // bitrev[236] = 55
DATA bitrev_size256_radix2<>+0x768(SB)/8, $183  // bitrev[237] = 183
DATA bitrev_size256_radix2<>+0x770(SB)/8, $119  // bitrev[238] = 119
DATA bitrev_size256_radix2<>+0x778(SB)/8, $247  // bitrev[239] = 247
DATA bitrev_size256_radix2<>+0x780(SB)/8, $15   // bitrev[240] = 15
DATA bitrev_size256_radix2<>+0x788(SB)/8, $143  // bitrev[241] = 143
DATA bitrev_size256_radix2<>+0x790(SB)/8, $79   // bitrev[242] = 79
DATA bitrev_size256_radix2<>+0x798(SB)/8, $207  // bitrev[243] = 207
DATA bitrev_size256_radix2<>+0x7A0(SB)/8, $47   // bitrev[244] = 47
DATA bitrev_size256_radix2<>+0x7A8(SB)/8, $175  // bitrev[245] = 175
DATA bitrev_size256_radix2<>+0x7B0(SB)/8, $111  // bitrev[246] = 111
DATA bitrev_size256_radix2<>+0x7B8(SB)/8, $239  // bitrev[247] = 239
DATA bitrev_size256_radix2<>+0x7C0(SB)/8, $31   // bitrev[248] = 31
DATA bitrev_size256_radix2<>+0x7C8(SB)/8, $159  // bitrev[249] = 159
DATA bitrev_size256_radix2<>+0x7D0(SB)/8, $95   // bitrev[250] = 95
DATA bitrev_size256_radix2<>+0x7D8(SB)/8, $223  // bitrev[251] = 223
DATA bitrev_size256_radix2<>+0x7E0(SB)/8, $63   // bitrev[252] = 63
DATA bitrev_size256_radix2<>+0x7E8(SB)/8, $191  // bitrev[253] = 191
DATA bitrev_size256_radix2<>+0x7F0(SB)/8, $127  // bitrev[254] = 127
DATA bitrev_size256_radix2<>+0x7F8(SB)/8, $255  // bitrev[255] = 255
GLOBL bitrev_size256_radix2<>(SB), RODATA, $2048

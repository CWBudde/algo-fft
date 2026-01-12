//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-128 FFT Kernels for AMD64 (complex128)
// ===========================================================================
//
// Size-specific entrypoints for n==128 that use XMM operations for
// correctness and a fixed-size DIT schedule.
//
// Radix-2: 7 stages (128 = 2^7)
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 128, complex128, radix-2
// ===========================================================================
TEXT ·ForwardAVX2Size128Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // dst pointer
	MOVQ src+24(FP), R9      // src pointer
	MOVQ twiddle+48(FP), R10 // twiddle pointer
	MOVQ scratch+72(FP), R11 // scratch pointer
	MOVQ src+32(FP), R13     // n (should be 128)

	CMPQ R13, $128
	JNE  size128_128_r2_return_false

	// Validate all slice lengths >= 128
	MOVQ dst+8(FP), AX
	CMPQ AX, $128
	JL   size128_128_r2_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $128
	JL   size128_128_r2_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $128
	JL   size128_128_r2_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size128_128_r2_use_dst
	MOVQ R11, R8             // In-place: use scratch as work

size128_128_r2_use_dst:
	// -----------------------------------------------------------------------
	// Bit-reversal permutation + Stage 1 (identity twiddles)
	// -----------------------------------------------------------------------
	// Bitrev pattern: [0,64,32,96, 16,80,48,112, 8,72,40,104, 24,88,56,120,
	//                  4,68,36,100, 20,84,52,116, 12,76,44,108, 28,92,60,124,
	//                  2,66,34,98, 18,82,50,114, 10,74,42,106, 26,90,58,122,
	//                  6,70,38,102, 22,86,54,118, 14,78,46,110, 30,94,62,126,
	//                  1,65,33,97, 17,81,49,113, 9,73,41,105, 25,89,57,121,
	//                  5,69,37,101, 21,85,53,117, 13,77,45,109, 29,93,61,125,
	//                  3,67,35,99, 19,83,51,115, 11,75,43,107, 27,91,59,123,
	//                  7,71,39,103, 23,87,55,119, 15,79,47,111, 31,95,63,127]
	// Stage 1 butterflies: a' = a + b, b' = a - b (twiddle[0] = 1).
	//
	// TODO: Check if a smaller loop odd/even is more efficient here.

	// (0,64) -> work[0], work[1]
	MOVUPD 0(R9), X0
	MOVUPD 1024(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 0(R8)
	MOVUPD X3, 16(R8)

	// (32,96) -> work[2], work[3]
	MOVUPD 512(R9), X0
	MOVUPD 1536(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 32(R8)
	MOVUPD X3, 48(R8)

	// (16,80) -> work[4], work[5]
	MOVUPD 256(R9), X0
	MOVUPD 1280(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 64(R8)
	MOVUPD X3, 80(R8)

	// (48,112) -> work[6], work[7]
	MOVUPD 768(R9), X0
	MOVUPD 1792(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 96(R8)
	MOVUPD X3, 112(R8)

	// (8,72) -> work[8], work[9]
	MOVUPD 128(R9), X0
	MOVUPD 1152(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 128(R8)
	MOVUPD X3, 144(R8)

	// (40,104) -> work[10], work[11]
	MOVUPD 640(R9), X0
	MOVUPD 1664(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 160(R8)
	MOVUPD X3, 176(R8)

	// (24,88) -> work[12], work[13]
	MOVUPD 384(R9), X0
	MOVUPD 1408(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 192(R8)
	MOVUPD X3, 208(R8)

	// (56,120) -> work[14], work[15]
	MOVUPD 896(R9), X0
	MOVUPD 1920(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 224(R8)
	MOVUPD X3, 240(R8)

	// (4,68) -> work[16], work[17]
	MOVUPD 64(R9), X0
	MOVUPD 1088(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 256(R8)
	MOVUPD X3, 272(R8)

	// (36,100) -> work[18], work[19]
	MOVUPD 576(R9), X0
	MOVUPD 1600(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 288(R8)
	MOVUPD X3, 304(R8)

	// (20,84) -> work[20], work[21]
	MOVUPD 320(R9), X0
	MOVUPD 1344(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 320(R8)
	MOVUPD X3, 336(R8)

	// (52,116) -> work[22], work[23]
	MOVUPD 832(R9), X0
	MOVUPD 1856(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 352(R8)
	MOVUPD X3, 368(R8)

	// (12,76) -> work[24], work[25]
	MOVUPD 192(R9), X0
	MOVUPD 1216(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 384(R8)
	MOVUPD X3, 400(R8)

	// (44,108) -> work[26], work[27]
	MOVUPD 704(R9), X0
	MOVUPD 1728(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 416(R8)
	MOVUPD X3, 432(R8)

	// (28,92) -> work[28], work[29]
	MOVUPD 448(R9), X0
	MOVUPD 1472(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 448(R8)
	MOVUPD X3, 464(R8)

	// (60,124) -> work[30], work[31]
	MOVUPD 960(R9), X0
	MOVUPD 1984(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 480(R8)
	MOVUPD X3, 496(R8)

	// (2,66) -> work[32], work[33]
	MOVUPD 32(R9), X0
	MOVUPD 1056(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 512(R8)
	MOVUPD X3, 528(R8)

	// (34,98) -> work[34], work[35]
	MOVUPD 544(R9), X0
	MOVUPD 1568(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 544(R8)
	MOVUPD X3, 560(R8)

	// (18,82) -> work[36], work[37]
	MOVUPD 288(R9), X0
	MOVUPD 1312(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 576(R8)
	MOVUPD X3, 592(R8)

	// (50,114) -> work[38], work[39]
	MOVUPD 800(R9), X0
	MOVUPD 1824(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 608(R8)
	MOVUPD X3, 624(R8)

	// (10,74) -> work[40], work[41]
	MOVUPD 160(R9), X0
	MOVUPD 1184(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 640(R8)
	MOVUPD X3, 656(R8)

	// (42,106) -> work[42], work[43]
	MOVUPD 672(R9), X0
	MOVUPD 1696(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 672(R8)
	MOVUPD X3, 688(R8)

	// (26,90) -> work[44], work[45]
	MOVUPD 416(R9), X0
	MOVUPD 1440(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 704(R8)
	MOVUPD X3, 720(R8)

	// (58,122) -> work[46], work[47]
	MOVUPD 928(R9), X0
	MOVUPD 1952(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 736(R8)
	MOVUPD X3, 752(R8)

	// (6,70) -> work[48], work[49]
	MOVUPD 96(R9), X0
	MOVUPD 1120(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 768(R8)
	MOVUPD X3, 784(R8)

	// (38,102) -> work[50], work[51]
	MOVUPD 608(R9), X0
	MOVUPD 1632(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 800(R8)
	MOVUPD X3, 816(R8)

	// (22,86) -> work[52], work[53]
	MOVUPD 352(R9), X0
	MOVUPD 1376(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 832(R8)
	MOVUPD X3, 848(R8)

	// (54,118) -> work[54], work[55]
	MOVUPD 864(R9), X0
	MOVUPD 1888(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 864(R8)
	MOVUPD X3, 880(R8)

	// (14,78) -> work[56], work[57]
	MOVUPD 224(R9), X0
	MOVUPD 1248(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 896(R8)
	MOVUPD X3, 912(R8)

	// (46,110) -> work[58], work[59]
	MOVUPD 736(R9), X0
	MOVUPD 1760(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 928(R8)
	MOVUPD X3, 944(R8)

	// (30,94) -> work[60], work[61]
	MOVUPD 480(R9), X0
	MOVUPD 1504(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 960(R8)
	MOVUPD X3, 976(R8)

	// (62,126) -> work[62], work[63]
	MOVUPD 992(R9), X0
	MOVUPD 2016(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 992(R8)
	MOVUPD X3, 1008(R8)

	// (1,65) -> work[64], work[65]
	MOVUPD 16(R9), X0
	MOVUPD 1040(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1024(R8)
	MOVUPD X3, 1040(R8)

	// (33,97) -> work[66], work[67]
	MOVUPD 528(R9), X0
	MOVUPD 1552(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1056(R8)
	MOVUPD X3, 1072(R8)

	// (17,81) -> work[68], work[69]
	MOVUPD 272(R9), X0
	MOVUPD 1296(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1088(R8)
	MOVUPD X3, 1104(R8)

	// (49,113) -> work[70], work[71]
	MOVUPD 784(R9), X0
	MOVUPD 1808(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1120(R8)
	MOVUPD X3, 1136(R8)

	// (9,73) -> work[72], work[73]
	MOVUPD 144(R9), X0
	MOVUPD 1168(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1152(R8)
	MOVUPD X3, 1168(R8)

	// (41,105) -> work[74], work[75]
	MOVUPD 656(R9), X0
	MOVUPD 1680(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1184(R8)
	MOVUPD X3, 1200(R8)

	// (25,89) -> work[76], work[77]
	MOVUPD 400(R9), X0
	MOVUPD 1424(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1216(R8)
	MOVUPD X3, 1232(R8)

	// (57,121) -> work[78], work[79]
	MOVUPD 912(R9), X0
	MOVUPD 1936(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1248(R8)
	MOVUPD X3, 1264(R8)

	// (5,69) -> work[80], work[81]
	MOVUPD 80(R9), X0
	MOVUPD 1104(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1280(R8)
	MOVUPD X3, 1296(R8)

	// (37,101) -> work[82], work[83]
	MOVUPD 592(R9), X0
	MOVUPD 1616(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1312(R8)
	MOVUPD X3, 1328(R8)

	// (21,85) -> work[84], work[85]
	MOVUPD 336(R9), X0
	MOVUPD 1360(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1344(R8)
	MOVUPD X3, 1360(R8)

	// (53,117) -> work[86], work[87]
	MOVUPD 848(R9), X0
	MOVUPD 1872(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1376(R8)
	MOVUPD X3, 1392(R8)

	// (13,77) -> work[88], work[89]
	MOVUPD 208(R9), X0
	MOVUPD 1232(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1408(R8)
	MOVUPD X3, 1424(R8)

	// (45,109) -> work[90], work[91]
	MOVUPD 720(R9), X0
	MOVUPD 1744(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1440(R8)
	MOVUPD X3, 1456(R8)

	// (29,93) -> work[92], work[93]
	MOVUPD 464(R9), X0
	MOVUPD 1488(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1472(R8)
	MOVUPD X3, 1488(R8)

	// (61,125) -> work[94], work[95]
	MOVUPD 976(R9), X0
	MOVUPD 2000(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1504(R8)
	MOVUPD X3, 1520(R8)

	// (3,67) -> work[96], work[97]
	MOVUPD 48(R9), X0
	MOVUPD 1072(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1536(R8)
	MOVUPD X3, 1552(R8)

	// (35,99) -> work[98], work[99]
	MOVUPD 560(R9), X0
	MOVUPD 1584(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1568(R8)
	MOVUPD X3, 1584(R8)

	// (19,83) -> work[100], work[101]
	MOVUPD 304(R9), X0
	MOVUPD 1328(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1600(R8)
	MOVUPD X3, 1616(R8)

	// (51,115) -> work[102], work[103]
	MOVUPD 816(R9), X0
	MOVUPD 1840(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1632(R8)
	MOVUPD X3, 1648(R8)

	// (11,75) -> work[104], work[105]
	MOVUPD 176(R9), X0
	MOVUPD 1200(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1664(R8)
	MOVUPD X3, 1680(R8)

	// (43,107) -> work[106], work[107]
	MOVUPD 688(R9), X0
	MOVUPD 1712(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1696(R8)
	MOVUPD X3, 1712(R8)

	// (27,91) -> work[108], work[109]
	MOVUPD 432(R9), X0
	MOVUPD 1456(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1728(R8)
	MOVUPD X3, 1744(R8)

	// (59,123) -> work[110], work[111]
	MOVUPD 944(R9), X0
	MOVUPD 1968(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1760(R8)
	MOVUPD X3, 1776(R8)

	// (7,71) -> work[112], work[113]
	MOVUPD 112(R9), X0
	MOVUPD 1136(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1792(R8)
	MOVUPD X3, 1808(R8)

	// (39,103) -> work[114], work[115]
	MOVUPD 624(R9), X0
	MOVUPD 1648(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1824(R8)
	MOVUPD X3, 1840(R8)

	// (23,87) -> work[116], work[117]
	MOVUPD 368(R9), X0
	MOVUPD 1392(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1856(R8)
	MOVUPD X3, 1872(R8)

	// (55,119) -> work[118], work[119]
	MOVUPD 880(R9), X0
	MOVUPD 1904(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1888(R8)
	MOVUPD X3, 1904(R8)

	// (15,79) -> work[120], work[121]
	MOVUPD 240(R9), X0
	MOVUPD 1264(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1920(R8)
	MOVUPD X3, 1936(R8)

	// (47,111) -> work[122], work[123]
	MOVUPD 752(R9), X0
	MOVUPD 1776(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1952(R8)
	MOVUPD X3, 1968(R8)

	// (31,95) -> work[124], work[125]
	MOVUPD 496(R9), X0
	MOVUPD 1520(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1984(R8)
	MOVUPD X3, 2000(R8)

	// (63,127) -> work[126], work[127]
	MOVUPD 1008(R9), X0
	MOVUPD 2032(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 2016(R8)
	MOVUPD X3, 2032(R8)

size128_128_r2_stage2:
	// -----------------------------------------------------------------------
	// Stage 2: size=4, half=2, step=32
	// -----------------------------------------------------------------------
	MOVQ $32, BX             // step
	XORQ CX, CX              // base

size128_128_r2_stage2_base:
	CMPQ CX, $128
	JGE  size128_128_r2_stage3

	XORQ DX, DX              // j

size128_128_r2_stage2_j:
	CMPQ DX, $2
	JGE  size128_128_r2_stage2_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $2, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMADDSUB231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size128_128_r2_stage2_j

size128_128_r2_stage2_next:
	ADDQ $4, CX
	JMP  size128_128_r2_stage2_base

size128_128_r2_stage3:
	// -----------------------------------------------------------------------
	// Stage 3: size=8, half=4, step=16
	// -----------------------------------------------------------------------
	MOVQ $16, BX
	XORQ CX, CX

size128_128_r2_stage3_base:
	CMPQ CX, $128
	JGE  size128_128_r2_stage4

	XORQ DX, DX

size128_128_r2_stage3_j:
	CMPQ DX, $4
	JGE  size128_128_r2_stage3_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $4, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMADDSUB231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size128_128_r2_stage3_j

size128_128_r2_stage3_next:
	ADDQ $8, CX
	JMP  size128_128_r2_stage3_base

size128_128_r2_stage4:
	// -----------------------------------------------------------------------
	// Stage 4: size=16, half=8, step=8
	// -----------------------------------------------------------------------
	MOVQ $8, BX
	XORQ CX, CX

size128_128_r2_stage4_base:
	CMPQ CX, $128
	JGE  size128_128_r2_stage5

	XORQ DX, DX

size128_128_r2_stage4_j:
	CMPQ DX, $8
	JGE  size128_128_r2_stage4_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $8, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMADDSUB231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size128_128_r2_stage4_j

size128_128_r2_stage4_next:
	ADDQ $16, CX
	JMP  size128_128_r2_stage4_base

size128_128_r2_stage5:
	// -----------------------------------------------------------------------
	// Stage 5: size=32, half=16, step=4
	// -----------------------------------------------------------------------
	MOVQ $4, BX
	XORQ CX, CX

size128_128_r2_stage5_base:
	CMPQ CX, $128
	JGE  size128_128_r2_stage6

	XORQ DX, DX

size128_128_r2_stage5_j:
	CMPQ DX, $16
	JGE  size128_128_r2_stage5_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $16, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMADDSUB231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size128_128_r2_stage5_j

size128_128_r2_stage5_next:
	ADDQ $32, CX
	JMP  size128_128_r2_stage5_base

size128_128_r2_stage6:
	// -----------------------------------------------------------------------
	// Stage 6: size=64, half=32, step=2
	// -----------------------------------------------------------------------
	MOVQ $2, BX
	XORQ CX, CX

size128_128_r2_stage6_base:
	CMPQ CX, $128
	JGE  size128_128_r2_stage7

	XORQ DX, DX

size128_128_r2_stage6_j:
	CMPQ DX, $32
	JGE  size128_128_r2_stage6_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $32, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMADDSUB231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size128_128_r2_stage6_j

size128_128_r2_stage6_next:
	ADDQ $64, CX
	JMP  size128_128_r2_stage6_base

size128_128_r2_stage7:
	// -----------------------------------------------------------------------
	// Stage 7: size=128, half=64, step=1
	// -----------------------------------------------------------------------
	MOVQ $1, BX
	XORQ CX, CX
	XORQ DX, DX

size128_128_r2_stage7_j:
	CMPQ DX, $64
	JGE  size128_128_r2_finalize

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $64, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMADDSUB231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size128_128_r2_stage7_j

size128_128_r2_finalize:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size128_128_r2_done

	XORQ CX, CX
size128_128_r2_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $2048           // 128 * 16 bytes
	JL   size128_128_r2_copy_loop

size128_128_r2_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size128_128_r2_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 128, complex128, radix-2
// ===========================================================================
TEXT ·InverseAVX2Size128Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	CMPQ R13, $128
	JNE  size128_inv_128_r2_return_false

	// Validate all slice lengths >= 128
	MOVQ dst+8(FP), AX
	CMPQ AX, $128
	JL   size128_inv_128_r2_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $128
	JL   size128_inv_128_r2_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $128
	JL   size128_inv_128_r2_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size128_inv_128_r2_use_dst
	MOVQ R11, R8

size128_inv_128_r2_use_dst:
	// -----------------------------------------------------------------------
	// Bit-reversal permutation + Stage 1 (identity twiddles)
	// -----------------------------------------------------------------------
	// Bitrev pattern: [0,64,32,96, 16,80,48,112, 8,72,40,104, 24,88,56,120,
	//                  4,68,36,100, 20,84,52,116, 12,76,44,108, 28,92,60,124,
	//                  2,66,34,98, 18,82,50,114, 10,74,42,106, 26,90,58,122,
	//                  6,70,38,102, 22,86,54,118, 14,78,46,110, 30,94,62,126,
	//                  1,65,33,97, 17,81,49,113, 9,73,41,105, 25,89,57,121,
	//                  5,69,37,101, 21,85,53,117, 13,77,45,109, 29,93,61,125,
	//                  3,67,35,99, 19,83,51,115, 11,75,43,107, 27,91,59,123,
	//                  7,71,39,103, 23,87,55,119, 15,79,47,111, 31,95,63,127]
	// Stage 1 butterflies: a' = a + b, b' = a - b (twiddle[0] = 1).

	// (0,64) -> work[0], work[1]
	MOVUPD 0(R9), X0
	MOVUPD 1024(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 0(R8)
	MOVUPD X3, 16(R8)

	// (32,96) -> work[2], work[3]
	MOVUPD 512(R9), X0
	MOVUPD 1536(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 32(R8)
	MOVUPD X3, 48(R8)

	// (16,80) -> work[4], work[5]
	MOVUPD 256(R9), X0
	MOVUPD 1280(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 64(R8)
	MOVUPD X3, 80(R8)

	// (48,112) -> work[6], work[7]
	MOVUPD 768(R9), X0
	MOVUPD 1792(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 96(R8)
	MOVUPD X3, 112(R8)

	// (8,72) -> work[8], work[9]
	MOVUPD 128(R9), X0
	MOVUPD 1152(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 128(R8)
	MOVUPD X3, 144(R8)

	// (40,104) -> work[10], work[11]
	MOVUPD 640(R9), X0
	MOVUPD 1664(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 160(R8)
	MOVUPD X3, 176(R8)

	// (24,88) -> work[12], work[13]
	MOVUPD 384(R9), X0
	MOVUPD 1408(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 192(R8)
	MOVUPD X3, 208(R8)

	// (56,120) -> work[14], work[15]
	MOVUPD 896(R9), X0
	MOVUPD 1920(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 224(R8)
	MOVUPD X3, 240(R8)

	// (4,68) -> work[16], work[17]
	MOVUPD 64(R9), X0
	MOVUPD 1088(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 256(R8)
	MOVUPD X3, 272(R8)

	// (36,100) -> work[18], work[19]
	MOVUPD 576(R9), X0
	MOVUPD 1600(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 288(R8)
	MOVUPD X3, 304(R8)

	// (20,84) -> work[20], work[21]
	MOVUPD 320(R9), X0
	MOVUPD 1344(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 320(R8)
	MOVUPD X3, 336(R8)

	// (52,116) -> work[22], work[23]
	MOVUPD 832(R9), X0
	MOVUPD 1856(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 352(R8)
	MOVUPD X3, 368(R8)

	// (12,76) -> work[24], work[25]
	MOVUPD 192(R9), X0
	MOVUPD 1216(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 384(R8)
	MOVUPD X3, 400(R8)

	// (44,108) -> work[26], work[27]
	MOVUPD 704(R9), X0
	MOVUPD 1728(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 416(R8)
	MOVUPD X3, 432(R8)

	// (28,92) -> work[28], work[29]
	MOVUPD 448(R9), X0
	MOVUPD 1472(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 448(R8)
	MOVUPD X3, 464(R8)

	// (60,124) -> work[30], work[31]
	MOVUPD 960(R9), X0
	MOVUPD 1984(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 480(R8)
	MOVUPD X3, 496(R8)

	// (2,66) -> work[32], work[33]
	MOVUPD 32(R9), X0
	MOVUPD 1056(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 512(R8)
	MOVUPD X3, 528(R8)

	// (34,98) -> work[34], work[35]
	MOVUPD 544(R9), X0
	MOVUPD 1568(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 544(R8)
	MOVUPD X3, 560(R8)

	// (18,82) -> work[36], work[37]
	MOVUPD 288(R9), X0
	MOVUPD 1312(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 576(R8)
	MOVUPD X3, 592(R8)

	// (50,114) -> work[38], work[39]
	MOVUPD 800(R9), X0
	MOVUPD 1824(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 608(R8)
	MOVUPD X3, 624(R8)

	// (10,74) -> work[40], work[41]
	MOVUPD 160(R9), X0
	MOVUPD 1184(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 640(R8)
	MOVUPD X3, 656(R8)

	// (42,106) -> work[42], work[43]
	MOVUPD 672(R9), X0
	MOVUPD 1696(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 672(R8)
	MOVUPD X3, 688(R8)

	// (26,90) -> work[44], work[45]
	MOVUPD 416(R9), X0
	MOVUPD 1440(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 704(R8)
	MOVUPD X3, 720(R8)

	// (58,122) -> work[46], work[47]
	MOVUPD 928(R9), X0
	MOVUPD 1952(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 736(R8)
	MOVUPD X3, 752(R8)

	// (6,70) -> work[48], work[49]
	MOVUPD 96(R9), X0
	MOVUPD 1120(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 768(R8)
	MOVUPD X3, 784(R8)

	// (38,102) -> work[50], work[51]
	MOVUPD 608(R9), X0
	MOVUPD 1632(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 800(R8)
	MOVUPD X3, 816(R8)

	// (22,86) -> work[52], work[53]
	MOVUPD 352(R9), X0
	MOVUPD 1376(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 832(R8)
	MOVUPD X3, 848(R8)

	// (54,118) -> work[54], work[55]
	MOVUPD 864(R9), X0
	MOVUPD 1888(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 864(R8)
	MOVUPD X3, 880(R8)

	// (14,78) -> work[56], work[57]
	MOVUPD 224(R9), X0
	MOVUPD 1248(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 896(R8)
	MOVUPD X3, 912(R8)

	// (46,110) -> work[58], work[59]
	MOVUPD 736(R9), X0
	MOVUPD 1760(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 928(R8)
	MOVUPD X3, 944(R8)

	// (30,94) -> work[60], work[61]
	MOVUPD 480(R9), X0
	MOVUPD 1504(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 960(R8)
	MOVUPD X3, 976(R8)

	// (62,126) -> work[62], work[63]
	MOVUPD 992(R9), X0
	MOVUPD 2016(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 992(R8)
	MOVUPD X3, 1008(R8)

	// (1,65) -> work[64], work[65]
	MOVUPD 16(R9), X0
	MOVUPD 1040(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1024(R8)
	MOVUPD X3, 1040(R8)

	// (33,97) -> work[66], work[67]
	MOVUPD 528(R9), X0
	MOVUPD 1552(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1056(R8)
	MOVUPD X3, 1072(R8)

	// (17,81) -> work[68], work[69]
	MOVUPD 272(R9), X0
	MOVUPD 1296(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1088(R8)
	MOVUPD X3, 1104(R8)

	// (49,113) -> work[70], work[71]
	MOVUPD 784(R9), X0
	MOVUPD 1808(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1120(R8)
	MOVUPD X3, 1136(R8)

	// (9,73) -> work[72], work[73]
	MOVUPD 144(R9), X0
	MOVUPD 1168(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1152(R8)
	MOVUPD X3, 1168(R8)

	// (41,105) -> work[74], work[75]
	MOVUPD 656(R9), X0
	MOVUPD 1680(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1184(R8)
	MOVUPD X3, 1200(R8)

	// (25,89) -> work[76], work[77]
	MOVUPD 400(R9), X0
	MOVUPD 1424(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1216(R8)
	MOVUPD X3, 1232(R8)

	// (57,121) -> work[78], work[79]
	MOVUPD 912(R9), X0
	MOVUPD 1936(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1248(R8)
	MOVUPD X3, 1264(R8)

	// (5,69) -> work[80], work[81]
	MOVUPD 80(R9), X0
	MOVUPD 1104(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1280(R8)
	MOVUPD X3, 1296(R8)

	// (37,101) -> work[82], work[83]
	MOVUPD 592(R9), X0
	MOVUPD 1616(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1312(R8)
	MOVUPD X3, 1328(R8)

	// (21,85) -> work[84], work[85]
	MOVUPD 336(R9), X0
	MOVUPD 1360(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1344(R8)
	MOVUPD X3, 1360(R8)

	// (53,117) -> work[86], work[87]
	MOVUPD 848(R9), X0
	MOVUPD 1872(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1376(R8)
	MOVUPD X3, 1392(R8)

	// (13,77) -> work[88], work[89]
	MOVUPD 208(R9), X0
	MOVUPD 1232(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1408(R8)
	MOVUPD X3, 1424(R8)

	// (45,109) -> work[90], work[91]
	MOVUPD 720(R9), X0
	MOVUPD 1744(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1440(R8)
	MOVUPD X3, 1456(R8)

	// (29,93) -> work[92], work[93]
	MOVUPD 464(R9), X0
	MOVUPD 1488(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1472(R8)
	MOVUPD X3, 1488(R8)

	// (61,125) -> work[94], work[95]
	MOVUPD 976(R9), X0
	MOVUPD 2000(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1504(R8)
	MOVUPD X3, 1520(R8)

	// (3,67) -> work[96], work[97]
	MOVUPD 48(R9), X0
	MOVUPD 1072(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1536(R8)
	MOVUPD X3, 1552(R8)

	// (35,99) -> work[98], work[99]
	MOVUPD 560(R9), X0
	MOVUPD 1584(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1568(R8)
	MOVUPD X3, 1584(R8)

	// (19,83) -> work[100], work[101]
	MOVUPD 304(R9), X0
	MOVUPD 1328(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1600(R8)
	MOVUPD X3, 1616(R8)

	// (51,115) -> work[102], work[103]
	MOVUPD 816(R9), X0
	MOVUPD 1840(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1632(R8)
	MOVUPD X3, 1648(R8)

	// (11,75) -> work[104], work[105]
	MOVUPD 176(R9), X0
	MOVUPD 1200(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1664(R8)
	MOVUPD X3, 1680(R8)

	// (43,107) -> work[106], work[107]
	MOVUPD 688(R9), X0
	MOVUPD 1712(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1696(R8)
	MOVUPD X3, 1712(R8)

	// (27,91) -> work[108], work[109]
	MOVUPD 432(R9), X0
	MOVUPD 1456(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1728(R8)
	MOVUPD X3, 1744(R8)

	// (59,123) -> work[110], work[111]
	MOVUPD 944(R9), X0
	MOVUPD 1968(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1760(R8)
	MOVUPD X3, 1776(R8)

	// (7,71) -> work[112], work[113]
	MOVUPD 112(R9), X0
	MOVUPD 1136(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1792(R8)
	MOVUPD X3, 1808(R8)

	// (39,103) -> work[114], work[115]
	MOVUPD 624(R9), X0
	MOVUPD 1648(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1824(R8)
	MOVUPD X3, 1840(R8)

	// (23,87) -> work[116], work[117]
	MOVUPD 368(R9), X0
	MOVUPD 1392(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1856(R8)
	MOVUPD X3, 1872(R8)

	// (55,119) -> work[118], work[119]
	MOVUPD 880(R9), X0
	MOVUPD 1904(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1888(R8)
	MOVUPD X3, 1904(R8)

	// (15,79) -> work[120], work[121]
	MOVUPD 240(R9), X0
	MOVUPD 1264(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1920(R8)
	MOVUPD X3, 1936(R8)

	// (47,111) -> work[122], work[123]
	MOVUPD 752(R9), X0
	MOVUPD 1776(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1952(R8)
	MOVUPD X3, 1968(R8)

	// (31,95) -> work[124], work[125]
	MOVUPD 496(R9), X0
	MOVUPD 1520(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 1984(R8)
	MOVUPD X3, 2000(R8)

	// (63,127) -> work[126], work[127]
	MOVUPD 1008(R9), X0
	MOVUPD 2032(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, 2016(R8)
	MOVUPD X3, 2032(R8)

size128_inv_128_r2_stage2:
	MOVQ $32, BX
	XORQ CX, CX

size128_inv_128_r2_stage2_base:
	CMPQ CX, $128
	JGE  size128_inv_128_r2_stage3
	XORQ DX, DX

size128_inv_128_r2_stage2_j:
	CMPQ DX, $2
	JGE  size128_inv_128_r2_stage2_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $2, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size128_inv_128_r2_stage2_j

size128_inv_128_r2_stage2_next:
	ADDQ $4, CX
	JMP  size128_inv_128_r2_stage2_base

size128_inv_128_r2_stage3:
	MOVQ $16, BX
	XORQ CX, CX

size128_inv_128_r2_stage3_base:
	CMPQ CX, $128
	JGE  size128_inv_128_r2_stage4
	XORQ DX, DX

size128_inv_128_r2_stage3_j:
	CMPQ DX, $4
	JGE  size128_inv_128_r2_stage3_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $4, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size128_inv_128_r2_stage3_j

size128_inv_128_r2_stage3_next:
	ADDQ $8, CX
	JMP  size128_inv_128_r2_stage3_base

size128_inv_128_r2_stage4:
	MOVQ $8, BX
	XORQ CX, CX

size128_inv_128_r2_stage4_base:
	CMPQ CX, $128
	JGE  size128_inv_128_r2_stage5
	XORQ DX, DX

size128_inv_128_r2_stage4_j:
	CMPQ DX, $8
	JGE  size128_inv_128_r2_stage4_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $8, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size128_inv_128_r2_stage4_j

size128_inv_128_r2_stage4_next:
	ADDQ $16, CX
	JMP  size128_inv_128_r2_stage4_base

size128_inv_128_r2_stage5:
	MOVQ $4, BX
	XORQ CX, CX

size128_inv_128_r2_stage5_base:
	CMPQ CX, $128
	JGE  size128_inv_128_r2_stage6
	XORQ DX, DX

size128_inv_128_r2_stage5_j:
	CMPQ DX, $16
	JGE  size128_inv_128_r2_stage5_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $16, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size128_inv_128_r2_stage5_j

size128_inv_128_r2_stage5_next:
	ADDQ $32, CX
	JMP  size128_inv_128_r2_stage5_base

size128_inv_128_r2_stage6:
	MOVQ $2, BX
	XORQ CX, CX

size128_inv_128_r2_stage6_base:
	CMPQ CX, $128
	JGE  size128_inv_128_r2_stage7
	XORQ DX, DX

size128_inv_128_r2_stage6_j:
	CMPQ DX, $32
	JGE  size128_inv_128_r2_stage6_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $32, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size128_inv_128_r2_stage6_j

size128_inv_128_r2_stage6_next:
	ADDQ $64, CX
	JMP  size128_inv_128_r2_stage6_base

size128_inv_128_r2_stage7:
	MOVQ $1, BX
	XORQ CX, CX
	XORQ DX, DX

size128_inv_128_r2_stage7_j:
	CMPQ DX, $64
	JGE  size128_inv_128_r2_scale

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $64, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size128_inv_128_r2_stage7_j

size128_inv_128_r2_scale:
	// Apply 1/n scaling (1/128 = 0.0078125)
	MOVQ ·oneTwentyEighth64(SB), AX  // 1/128 in float64
	VMOVQ AX, X9
	VMOVDDUP X9, X9

	XORQ CX, CX
size128_inv_128_r2_scale_loop:
	CMPQ CX, $128
	JGE  size128_inv_128_r2_finalize
	MOVQ CX, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X0
	VMULPD X9, X0, X0
	MOVUPD X0, (R8)(SI*1)
	INCQ CX
	JMP  size128_inv_128_r2_scale_loop

size128_inv_128_r2_finalize:
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size128_inv_128_r2_done

	XORQ CX, CX
size128_inv_128_r2_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $2048
	JL   size128_inv_128_r2_copy_loop

size128_inv_128_r2_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size128_inv_128_r2_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

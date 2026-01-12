//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-256 Radix-2 FFT Kernels for AMD64 (complex128)
// ===========================================================================

#include "textflag.h"

// Forward transform, size 256, complex128, radix-2
TEXT ·ForwardSSE2Size256Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	// Check size n == 256
	CMPQ R13, $256
	JNE  fwd_err

	// Check if dst == src (in-place)
	CMPQ R8, R9
	JNE  fwd_use_dst
	MOVQ R11, R8       // Use scratch as temp dst if in-place

fwd_use_dst:
	// -----------------------------------------------------------------------
	// FUSED: Bit-reversal permutation + Stage 1 (identity twiddles)
	// Unrolled even half, executed twice (second pass uses +16 src, +2048 dst).
	// -----------------------------------------------------------------------
	MOVQ R8, R14 // save work base
	MOVQ R9, R15 // save src base
	XORQ BX, BX  // pass counter (even/odd)

fwd_stage1_pass:
	// (0,128) -> work[0], work[1]
	MOVUPD 0(R9), X0         // src[0]
	MOVUPD 2048(R9), X1      // src[128]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 0(R8)         // work[0]
	MOVUPD X3, 16(R8)        // work[1]

	// (64,192) -> work[2], work[3]
	MOVUPD 1024(R9), X0      // src[64]
	MOVUPD 3072(R9), X1      // src[192]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 32(R8)        // work[2]
	MOVUPD X3, 48(R8)        // work[3]

	// (32,160) -> work[4], work[5]
	MOVUPD 512(R9), X0       // src[32]
	MOVUPD 2560(R9), X1      // src[160]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 64(R8)        // work[4]
	MOVUPD X3, 80(R8)        // work[5]

	// (96,224) -> work[6], work[7]
	MOVUPD 1536(R9), X0      // src[96]
	MOVUPD 3584(R9), X1      // src[224]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 96(R8)        // work[6]
	MOVUPD X3, 112(R8)       // work[7]

	// (16,144) -> work[8], work[9]
	MOVUPD 256(R9), X0       // src[16]
	MOVUPD 2304(R9), X1      // src[144]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 128(R8)       // work[8]
	MOVUPD X3, 144(R8)       // work[9]

	// (80,208) -> work[10], work[11]
	MOVUPD 1280(R9), X0      // src[80]
	MOVUPD 3328(R9), X1      // src[208]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 160(R8)       // work[10]
	MOVUPD X3, 176(R8)       // work[11]

	// (48,176) -> work[12], work[13]
	MOVUPD 768(R9), X0       // src[48]
	MOVUPD 2816(R9), X1      // src[176]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 192(R8)       // work[12]
	MOVUPD X3, 208(R8)       // work[13]

	// (112,240) -> work[14], work[15]
	MOVUPD 1792(R9), X0      // src[112]
	MOVUPD 3840(R9), X1      // src[240]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 224(R8)       // work[14]
	MOVUPD X3, 240(R8)       // work[15]

	// (8,136) -> work[16], work[17]
	MOVUPD 128(R9), X0       // src[8]
	MOVUPD 2176(R9), X1      // src[136]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 256(R8)       // work[16]
	MOVUPD X3, 272(R8)       // work[17]

	// (72,200) -> work[18], work[19]
	MOVUPD 1152(R9), X0      // src[72]
	MOVUPD 3200(R9), X1      // src[200]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 288(R8)       // work[18]
	MOVUPD X3, 304(R8)       // work[19]

	// (40,168) -> work[20], work[21]
	MOVUPD 640(R9), X0       // src[40]
	MOVUPD 2688(R9), X1      // src[168]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 320(R8)       // work[20]
	MOVUPD X3, 336(R8)       // work[21]

	// (104,232) -> work[22], work[23]
	MOVUPD 1664(R9), X0      // src[104]
	MOVUPD 3712(R9), X1      // src[232]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 352(R8)       // work[22]
	MOVUPD X3, 368(R8)       // work[23]

	// (24,152) -> work[24], work[25]
	MOVUPD 384(R9), X0       // src[24]
	MOVUPD 2432(R9), X1      // src[152]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 384(R8)       // work[24]
	MOVUPD X3, 400(R8)       // work[25]

	// (88,216) -> work[26], work[27]
	MOVUPD 1408(R9), X0      // src[88]
	MOVUPD 3456(R9), X1      // src[216]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 416(R8)       // work[26]
	MOVUPD X3, 432(R8)       // work[27]

	// (56,184) -> work[28], work[29]
	MOVUPD 896(R9), X0       // src[56]
	MOVUPD 2944(R9), X1      // src[184]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 448(R8)       // work[28]
	MOVUPD X3, 464(R8)       // work[29]

	// (120,248) -> work[30], work[31]
	MOVUPD 1920(R9), X0      // src[120]
	MOVUPD 3968(R9), X1      // src[248]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 480(R8)       // work[30]
	MOVUPD X3, 496(R8)       // work[31]

	// (4,132) -> work[32], work[33]
	MOVUPD 64(R9), X0        // src[4]
	MOVUPD 2112(R9), X1      // src[132]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 512(R8)       // work[32]
	MOVUPD X3, 528(R8)       // work[33]

	// (68,196) -> work[34], work[35]
	MOVUPD 1088(R9), X0      // src[68]
	MOVUPD 3136(R9), X1      // src[196]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 544(R8)       // work[34]
	MOVUPD X3, 560(R8)       // work[35]

	// (36,164) -> work[36], work[37]
	MOVUPD 576(R9), X0       // src[36]
	MOVUPD 2624(R9), X1      // src[164]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 576(R8)       // work[36]
	MOVUPD X3, 592(R8)       // work[37]

	// (100,228) -> work[38], work[39]
	MOVUPD 1600(R9), X0      // src[100]
	MOVUPD 3648(R9), X1      // src[228]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 608(R8)       // work[38]
	MOVUPD X3, 624(R8)       // work[39]

	// (20,148) -> work[40], work[41]
	MOVUPD 320(R9), X0       // src[20]
	MOVUPD 2368(R9), X1      // src[148]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 640(R8)       // work[40]
	MOVUPD X3, 656(R8)       // work[41]

	// (84,212) -> work[42], work[43]
	MOVUPD 1344(R9), X0      // src[84]
	MOVUPD 3392(R9), X1      // src[212]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 672(R8)       // work[42]
	MOVUPD X3, 688(R8)       // work[43]

	// (52,180) -> work[44], work[45]
	MOVUPD 832(R9), X0       // src[52]
	MOVUPD 2880(R9), X1      // src[180]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 704(R8)       // work[44]
	MOVUPD X3, 720(R8)       // work[45]

	// (116,244) -> work[46], work[47]
	MOVUPD 1856(R9), X0      // src[116]
	MOVUPD 3904(R9), X1      // src[244]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 736(R8)       // work[46]
	MOVUPD X3, 752(R8)       // work[47]

	// (12,140) -> work[48], work[49]
	MOVUPD 192(R9), X0       // src[12]
	MOVUPD 2240(R9), X1      // src[140]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 768(R8)       // work[48]
	MOVUPD X3, 784(R8)       // work[49]

	// (76,204) -> work[50], work[51]
	MOVUPD 1216(R9), X0      // src[76]
	MOVUPD 3264(R9), X1      // src[204]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 800(R8)       // work[50]
	MOVUPD X3, 816(R8)       // work[51]

	// (44,172) -> work[52], work[53]
	MOVUPD 704(R9), X0       // src[44]
	MOVUPD 2752(R9), X1      // src[172]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 832(R8)       // work[52]
	MOVUPD X3, 848(R8)       // work[53]

	// (108,236) -> work[54], work[55]
	MOVUPD 1728(R9), X0      // src[108]
	MOVUPD 3776(R9), X1      // src[236]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 864(R8)       // work[54]
	MOVUPD X3, 880(R8)       // work[55]

	// (28,156) -> work[56], work[57]
	MOVUPD 448(R9), X0       // src[28]
	MOVUPD 2496(R9), X1      // src[156]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 896(R8)       // work[56]
	MOVUPD X3, 912(R8)       // work[57]

	// (92,220) -> work[58], work[59]
	MOVUPD 1472(R9), X0      // src[92]
	MOVUPD 3520(R9), X1      // src[220]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 928(R8)       // work[58]
	MOVUPD X3, 944(R8)       // work[59]

	// (60,188) -> work[60], work[61]
	MOVUPD 960(R9), X0       // src[60]
	MOVUPD 3008(R9), X1      // src[188]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 960(R8)       // work[60]
	MOVUPD X3, 976(R8)       // work[61]

	// (124,252) -> work[62], work[63]
	MOVUPD 1984(R9), X0      // src[124]
	MOVUPD 4032(R9), X1      // src[252]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 992(R8)       // work[62]
	MOVUPD X3, 1008(R8)      // work[63]

	// (2,130) -> work[64], work[65]
	MOVUPD 32(R9), X0        // src[2]
	MOVUPD 2080(R9), X1      // src[130]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1024(R8)      // work[64]
	MOVUPD X3, 1040(R8)      // work[65]

	// (66,194) -> work[66], work[67]
	MOVUPD 1056(R9), X0      // src[66]
	MOVUPD 3104(R9), X1      // src[194]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1056(R8)      // work[66]
	MOVUPD X3, 1072(R8)      // work[67]

	// (34,162) -> work[68], work[69]
	MOVUPD 544(R9), X0       // src[34]
	MOVUPD 2592(R9), X1      // src[162]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1088(R8)      // work[68]
	MOVUPD X3, 1104(R8)      // work[69]

	// (98,226) -> work[70], work[71]
	MOVUPD 1568(R9), X0      // src[98]
	MOVUPD 3616(R9), X1      // src[226]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1120(R8)      // work[70]
	MOVUPD X3, 1136(R8)      // work[71]

	// (18,146) -> work[72], work[73]
	MOVUPD 288(R9), X0       // src[18]
	MOVUPD 2336(R9), X1      // src[146]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1152(R8)      // work[72]
	MOVUPD X3, 1168(R8)      // work[73]

	// (82,210) -> work[74], work[75]
	MOVUPD 1312(R9), X0      // src[82]
	MOVUPD 3360(R9), X1      // src[210]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1184(R8)      // work[74]
	MOVUPD X3, 1200(R8)      // work[75]

	// (50,178) -> work[76], work[77]
	MOVUPD 800(R9), X0       // src[50]
	MOVUPD 2848(R9), X1      // src[178]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1216(R8)      // work[76]
	MOVUPD X3, 1232(R8)      // work[77]

	// (114,242) -> work[78], work[79]
	MOVUPD 1824(R9), X0      // src[114]
	MOVUPD 3872(R9), X1      // src[242]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1248(R8)      // work[78]
	MOVUPD X3, 1264(R8)      // work[79]

	// (10,138) -> work[80], work[81]
	MOVUPD 160(R9), X0       // src[10]
	MOVUPD 2208(R9), X1      // src[138]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1280(R8)      // work[80]
	MOVUPD X3, 1296(R8)      // work[81]

	// (74,202) -> work[82], work[83]
	MOVUPD 1184(R9), X0      // src[74]
	MOVUPD 3232(R9), X1      // src[202]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1312(R8)      // work[82]
	MOVUPD X3, 1328(R8)      // work[83]

	// (42,170) -> work[84], work[85]
	MOVUPD 672(R9), X0       // src[42]
	MOVUPD 2720(R9), X1      // src[170]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1344(R8)      // work[84]
	MOVUPD X3, 1360(R8)      // work[85]

	// (106,234) -> work[86], work[87]
	MOVUPD 1696(R9), X0      // src[106]
	MOVUPD 3744(R9), X1      // src[234]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1376(R8)      // work[86]
	MOVUPD X3, 1392(R8)      // work[87]

	// (26,154) -> work[88], work[89]
	MOVUPD 416(R9), X0       // src[26]
	MOVUPD 2464(R9), X1      // src[154]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1408(R8)      // work[88]
	MOVUPD X3, 1424(R8)      // work[89]

	// (90,218) -> work[90], work[91]
	MOVUPD 1440(R9), X0      // src[90]
	MOVUPD 3488(R9), X1      // src[218]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1440(R8)      // work[90]
	MOVUPD X3, 1456(R8)      // work[91]

	// (58,186) -> work[92], work[93]
	MOVUPD 928(R9), X0       // src[58]
	MOVUPD 2976(R9), X1      // src[186]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1472(R8)      // work[92]
	MOVUPD X3, 1488(R8)      // work[93]

	// (122,250) -> work[94], work[95]
	MOVUPD 1952(R9), X0      // src[122]
	MOVUPD 4000(R9), X1      // src[250]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1504(R8)      // work[94]
	MOVUPD X3, 1520(R8)      // work[95]

	// (6,134) -> work[96], work[97]
	MOVUPD 96(R9), X0        // src[6]
	MOVUPD 2144(R9), X1      // src[134]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1536(R8)      // work[96]
	MOVUPD X3, 1552(R8)      // work[97]

	// (70,198) -> work[98], work[99]
	MOVUPD 1120(R9), X0      // src[70]
	MOVUPD 3168(R9), X1      // src[198]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1568(R8)      // work[98]
	MOVUPD X3, 1584(R8)      // work[99]

	// (38,166) -> work[100], work[101]
	MOVUPD 608(R9), X0       // src[38]
	MOVUPD 2656(R9), X1      // src[166]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1600(R8)      // work[100]
	MOVUPD X3, 1616(R8)      // work[101]

	// (102,230) -> work[102], work[103]
	MOVUPD 1632(R9), X0      // src[102]
	MOVUPD 3680(R9), X1      // src[230]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1632(R8)      // work[102]
	MOVUPD X3, 1648(R8)      // work[103]

	// (22,150) -> work[104], work[105]
	MOVUPD 352(R9), X0       // src[22]
	MOVUPD 2400(R9), X1      // src[150]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1664(R8)      // work[104]
	MOVUPD X3, 1680(R8)      // work[105]

	// (86,214) -> work[106], work[107]
	MOVUPD 1376(R9), X0      // src[86]
	MOVUPD 3424(R9), X1      // src[214]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1696(R8)      // work[106]
	MOVUPD X3, 1712(R8)      // work[107]

	// (54,182) -> work[108], work[109]
	MOVUPD 864(R9), X0       // src[54]
	MOVUPD 2912(R9), X1      // src[182]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1728(R8)      // work[108]
	MOVUPD X3, 1744(R8)      // work[109]

	// (118,246) -> work[110], work[111]
	MOVUPD 1888(R9), X0      // src[118]
	MOVUPD 3936(R9), X1      // src[246]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1760(R8)      // work[110]
	MOVUPD X3, 1776(R8)      // work[111]

	// (14,142) -> work[112], work[113]
	MOVUPD 224(R9), X0       // src[14]
	MOVUPD 2272(R9), X1      // src[142]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1792(R8)      // work[112]
	MOVUPD X3, 1808(R8)      // work[113]

	// (78,206) -> work[114], work[115]
	MOVUPD 1248(R9), X0      // src[78]
	MOVUPD 3296(R9), X1      // src[206]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1824(R8)      // work[114]
	MOVUPD X3, 1840(R8)      // work[115]

	// (46,174) -> work[116], work[117]
	MOVUPD 736(R9), X0       // src[46]
	MOVUPD 2784(R9), X1      // src[174]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1856(R8)      // work[116]
	MOVUPD X3, 1872(R8)      // work[117]

	// (110,238) -> work[118], work[119]
	MOVUPD 1760(R9), X0      // src[110]
	MOVUPD 3808(R9), X1      // src[238]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1888(R8)      // work[118]
	MOVUPD X3, 1904(R8)      // work[119]

	// (30,158) -> work[120], work[121]
	MOVUPD 480(R9), X0       // src[30]
	MOVUPD 2528(R9), X1      // src[158]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1920(R8)      // work[120]
	MOVUPD X3, 1936(R8)      // work[121]

	// (94,222) -> work[122], work[123]
	MOVUPD 1504(R9), X0      // src[94]
	MOVUPD 3552(R9), X1      // src[222]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1952(R8)      // work[122]
	MOVUPD X3, 1968(R8)      // work[123]

	// (62,190) -> work[124], work[125]
	MOVUPD 992(R9), X0       // src[62]
	MOVUPD 3040(R9), X1      // src[190]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1984(R8)      // work[124]
	MOVUPD X3, 2000(R8)      // work[125]

	// (126,254) -> work[126], work[127]
	MOVUPD 2016(R9), X0      // src[126]
	MOVUPD 4064(R9), X1      // src[254]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 2016(R8)      // work[126]
	MOVUPD X3, 2032(R8)      // work[127]

	INCQ BX                  // next pass
	CMPQ BX, $2              // done after odd pass
	JGE  fwd_stage1_done
	LEAQ 2048(R14), R8       // work offset for odd half
	LEAQ 16(R15), R9         // src offset for odd half
	JMP  fwd_stage1_pass

fwd_stage1_done:
	MOVQ R14, R8             // restore work base

	// Stage 2: dist 2 - 64 blocks of 4
	MOVQ R8, SI              // work base
	MOVQ $64, CX             // blocks
	MOVUPS ·maskNegLoPD(SB), X14
fwd_stage2_loop:
	MOVQ $2, DX              // half=2
fwd_stage2_inner:
	MOVUPD (SI), X0          // a
  MOVUPD 32(SI), X1        // b
	MOVQ $2, AX              // k = 2 - DX
  SUBQ DX, AX              // k (0..1)
  SHLQ $6, AX              // k * 64
  SHLQ $4, AX              // k * 64 * 16
  MOVUPD (R10)(AX*1), X10  // twiddle[k*64]
	MOVAPD X1, X2            // b
  UNPCKLPD X2, X2          // b.re
  MULPD X10, X2            // b.re * w
	MOVAPD X1, X3            // b
  UNPCKHPD X3, X3          // b.im
  MOVAPD X10, X4           // w
  SHUFPD $1, X4, X4        // swap
  MULPD X3, X4             // b.im * w
	XORPD X14, X4            // multiply by i
  ADDPD X4, X2             // t = w * b
	MOVAPD X0, X3            // a
  ADDPD X2, X0             // a + t
  SUBPD X2, X3             // a - t
	MOVUPD X0, (SI)          // out a
  MOVUPD X3, 32(SI)        // out b
	ADDQ $16, SI             // next j
  DECQ DX                  // next j
  JNZ fwd_stage2_inner
	ADDQ $32, SI             // next block
  DECQ CX                  // next block
  JNZ fwd_stage2_loop

	// Stage 3: dist 4
	MOVQ R8, SI             // Reset pointer
	MOVQ $32, CX            // Loop counter (256 / 8 = 32)

fwd_s3_loop:
	MOVQ $4, DX             // Inner loop counter

fwd_s3_inner:
	MOVUPD (SI), X0         // Load A
	MOVUPD 64(SI), X1       // Load B (at dist 4*16 = 64 bytes)

	// Twiddle factor calculation
	MOVQ $4, AX             // Base
	SUBQ DX, AX             // k = 4 - DX
	SHLQ $5, AX             // k * 32 (stride for stage 3)
	SHLQ $4, AX             // Convert to bytes (*16)
	MOVUPD (R10)(AX*1), X10 // Load Twiddle W

	// Complex Multiply B * W
	MOVAPD X1, X2
	UNPCKLPD X2, X2         // (Re(B), Re(B))
	MULPD X10, X2           // Re(B)*Re(W), Re(B)*Im(W)

	MOVAPD X1, X3
	UNPCKHPD X3, X3         // (Im(B), Im(B))
	MOVAPD X10, X4
	SHUFPD $1, X4, X4       // (Im(W), Re(W))
	MULPD X3, X4            // Im(B)*Im(W), Im(B)*Re(W)
	XORPD X14, X4           // -Im(B)*Im(W), Im(B)*Re(W)
	ADDPD X4, X2            // X2 = B * W

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0            // A + B*W
	SUBPD X2, X3            // A - B*W

	MOVUPD X0, (SI)
	MOVUPD X3, 64(SI)

	ADDQ $16, SI            // Next pair
	DECQ DX
	JNZ fwd_s3_inner

	ADDQ $64, SI            // Skip next block
	DECQ CX
	JNZ fwd_s3_loop

	// Stage 4: dist 8
	MOVQ R8, SI
	MOVQ $16, CX            // 256 / 16 = 16

fwd_s4_loop:
	MOVQ $8, DX

fwd_s4_inner:
	MOVUPD (SI), X0
	MOVUPD 128(SI), X1      // dist 8*16 = 128

	MOVQ $8, AX
	SUBQ DX, AX
	SHLQ $4, AX             // k * 16 (stride)
	SHLQ $4, AX             // bytes
	MOVUPD (R10)(AX*1), X10

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2
	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X14, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 128(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ fwd_s4_inner

	ADDQ $128, SI
	DECQ CX
	JNZ fwd_s4_loop

	// Stage 5: dist 16
	MOVQ R8, SI
	MOVQ $8, CX             // 256 / 32 = 8

fwd_s5_loop:
	MOVQ $16, DX

fwd_s5_inner:
	MOVUPD (SI), X0
	MOVUPD 256(SI), X1      // dist 16*16 = 256

	MOVQ $16, AX
	SUBQ DX, AX
	SHLQ $3, AX             // k * 8 (stride)
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2
	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X14, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 256(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ fwd_s5_inner

	ADDQ $256, SI
	DECQ CX
	JNZ fwd_s5_loop

	// Stage 6: dist 32
	MOVQ R8, SI
	MOVQ $4, CX             // 256 / 64 = 4

fwd_s6_loop:
	MOVQ $32, DX

fwd_s6_inner:
	MOVUPD (SI), X0
	MOVUPD 512(SI), X1      // dist 32*16 = 512

	MOVQ $32, AX
	SUBQ DX, AX
	SHLQ $2, AX             // k * 4 (stride)
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2
	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X14, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 512(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ fwd_s6_inner

	ADDQ $512, SI
	DECQ CX
	JNZ fwd_s6_loop

	// Stage 7: dist 64
	MOVQ R8, SI
	MOVQ $2, CX             // 256 / 128 = 2

fwd_s7_loop:
	MOVQ $64, DX

fwd_s7_inner:
	MOVUPD (SI), X0
	MOVUPD 1024(SI), X1     // dist 64*16 = 1024

	MOVQ $64, AX
	SUBQ DX, AX
	SHLQ $1, AX             // k * 2 (stride)
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2
	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X14, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 1024(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ fwd_s7_inner

	ADDQ $1024, SI
	DECQ CX
	JNZ fwd_s7_loop

	// Stage 8: dist 128
	MOVQ R8, SI
	MOVQ $128, DX           // Single loop

fwd_s8_inner:
	MOVUPD (SI), X0
	MOVUPD 2048(SI), X1     // dist 128*16 = 2048

	MOVQ $128, AX
	SUBQ DX, AX
	                        // k * 1 (stride) -> no shift needed
	SHLQ $4, AX             // bytes
	MOVUPD (R10)(AX*1), X10

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2
	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X14, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 2048(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ fwd_s8_inner

	// Copy to dst if needed
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   fwd_done

	MOVQ $128, CX           // 256 / 2 = 128 iterations of 2xComplex128 (32 bytes)
	MOVQ R8, SI
	MOVQ R14, DI

fwd_copy_loop:
	MOVUPD (SI), X0
	MOVUPD 16(SI), X1
	MOVUPD X0, (DI)
	MOVUPD X1, 16(DI)
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ fwd_copy_loop

fwd_done:
	MOVB $1, ret+96(FP)
	RET

fwd_err:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 256, complex128, radix-2
TEXT ·InverseSSE2Size256Radix2Complex128Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	CMPQ R13, $256
	JNE  inv_err

	CMPQ R8, R9
	JNE  inv_use_dst
	MOVQ R11, R8

inv_use_dst:
	// -----------------------------------------------------------------------
	// FUSED: Bit-reversal permutation + Stage 1 (identity twiddles)
	// Unrolled even half, executed twice (second pass uses +16 src, +2048 dst).
	// -----------------------------------------------------------------------
	MOVQ R8, R14 // save work base
	MOVQ R9, R15 // save src base
	XORQ BX, BX  // pass counter (even/odd)

inv_stage1_pass:
	// (0,128) -> work[0], work[1]
	MOVUPD 0(R9), X0         // src[0]
	MOVUPD 2048(R9), X1      // src[128]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 0(R8)         // work[0]
	MOVUPD X3, 16(R8)        // work[1]

	// (64,192) -> work[2], work[3]
	MOVUPD 1024(R9), X0      // src[64]
	MOVUPD 3072(R9), X1      // src[192]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 32(R8)        // work[2]
	MOVUPD X3, 48(R8)        // work[3]

	// (32,160) -> work[4], work[5]
	MOVUPD 512(R9), X0       // src[32]
	MOVUPD 2560(R9), X1      // src[160]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 64(R8)        // work[4]
	MOVUPD X3, 80(R8)        // work[5]

	// (96,224) -> work[6], work[7]
	MOVUPD 1536(R9), X0      // src[96]
	MOVUPD 3584(R9), X1      // src[224]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 96(R8)        // work[6]
	MOVUPD X3, 112(R8)       // work[7]

	// (16,144) -> work[8], work[9]
	MOVUPD 256(R9), X0       // src[16]
	MOVUPD 2304(R9), X1      // src[144]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 128(R8)       // work[8]
	MOVUPD X3, 144(R8)       // work[9]

	// (80,208) -> work[10], work[11]
	MOVUPD 1280(R9), X0      // src[80]
	MOVUPD 3328(R9), X1      // src[208]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 160(R8)       // work[10]
	MOVUPD X3, 176(R8)       // work[11]

	// (48,176) -> work[12], work[13]
	MOVUPD 768(R9), X0       // src[48]
	MOVUPD 2816(R9), X1      // src[176]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 192(R8)       // work[12]
	MOVUPD X3, 208(R8)       // work[13]

	// (112,240) -> work[14], work[15]
	MOVUPD 1792(R9), X0      // src[112]
	MOVUPD 3840(R9), X1      // src[240]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 224(R8)       // work[14]
	MOVUPD X3, 240(R8)       // work[15]

	// (8,136) -> work[16], work[17]
	MOVUPD 128(R9), X0       // src[8]
	MOVUPD 2176(R9), X1      // src[136]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 256(R8)       // work[16]
	MOVUPD X3, 272(R8)       // work[17]

	// (72,200) -> work[18], work[19]
	MOVUPD 1152(R9), X0      // src[72]
	MOVUPD 3200(R9), X1      // src[200]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 288(R8)       // work[18]
	MOVUPD X3, 304(R8)       // work[19]

	// (40,168) -> work[20], work[21]
	MOVUPD 640(R9), X0       // src[40]
	MOVUPD 2688(R9), X1      // src[168]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 320(R8)       // work[20]
	MOVUPD X3, 336(R8)       // work[21]

	// (104,232) -> work[22], work[23]
	MOVUPD 1664(R9), X0      // src[104]
	MOVUPD 3712(R9), X1      // src[232]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 352(R8)       // work[22]
	MOVUPD X3, 368(R8)       // work[23]

	// (24,152) -> work[24], work[25]
	MOVUPD 384(R9), X0       // src[24]
	MOVUPD 2432(R9), X1      // src[152]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 384(R8)       // work[24]
	MOVUPD X3, 400(R8)       // work[25]

	// (88,216) -> work[26], work[27]
	MOVUPD 1408(R9), X0      // src[88]
	MOVUPD 3456(R9), X1      // src[216]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 416(R8)       // work[26]
	MOVUPD X3, 432(R8)       // work[27]

	// (56,184) -> work[28], work[29]
	MOVUPD 896(R9), X0       // src[56]
	MOVUPD 2944(R9), X1      // src[184]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 448(R8)       // work[28]
	MOVUPD X3, 464(R8)       // work[29]

	// (120,248) -> work[30], work[31]
	MOVUPD 1920(R9), X0      // src[120]
	MOVUPD 3968(R9), X1      // src[248]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 480(R8)       // work[30]
	MOVUPD X3, 496(R8)       // work[31]

	// (4,132) -> work[32], work[33]
	MOVUPD 64(R9), X0        // src[4]
	MOVUPD 2112(R9), X1      // src[132]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 512(R8)       // work[32]
	MOVUPD X3, 528(R8)       // work[33]

	// (68,196) -> work[34], work[35]
	MOVUPD 1088(R9), X0      // src[68]
	MOVUPD 3136(R9), X1      // src[196]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 544(R8)       // work[34]
	MOVUPD X3, 560(R8)       // work[35]

	// (36,164) -> work[36], work[37]
	MOVUPD 576(R9), X0       // src[36]
	MOVUPD 2624(R9), X1      // src[164]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 576(R8)       // work[36]
	MOVUPD X3, 592(R8)       // work[37]

	// (100,228) -> work[38], work[39]
	MOVUPD 1600(R9), X0      // src[100]
	MOVUPD 3648(R9), X1      // src[228]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 608(R8)       // work[38]
	MOVUPD X3, 624(R8)       // work[39]

	// (20,148) -> work[40], work[41]
	MOVUPD 320(R9), X0       // src[20]
	MOVUPD 2368(R9), X1      // src[148]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 640(R8)       // work[40]
	MOVUPD X3, 656(R8)       // work[41]

	// (84,212) -> work[42], work[43]
	MOVUPD 1344(R9), X0      // src[84]
	MOVUPD 3392(R9), X1      // src[212]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 672(R8)       // work[42]
	MOVUPD X3, 688(R8)       // work[43]

	// (52,180) -> work[44], work[45]
	MOVUPD 832(R9), X0       // src[52]
	MOVUPD 2880(R9), X1      // src[180]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 704(R8)       // work[44]
	MOVUPD X3, 720(R8)       // work[45]

	// (116,244) -> work[46], work[47]
	MOVUPD 1856(R9), X0      // src[116]
	MOVUPD 3904(R9), X1      // src[244]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 736(R8)       // work[46]
	MOVUPD X3, 752(R8)       // work[47]

	// (12,140) -> work[48], work[49]
	MOVUPD 192(R9), X0       // src[12]
	MOVUPD 2240(R9), X1      // src[140]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 768(R8)       // work[48]
	MOVUPD X3, 784(R8)       // work[49]

	// (76,204) -> work[50], work[51]
	MOVUPD 1216(R9), X0      // src[76]
	MOVUPD 3264(R9), X1      // src[204]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 800(R8)       // work[50]
	MOVUPD X3, 816(R8)       // work[51]

	// (44,172) -> work[52], work[53]
	MOVUPD 704(R9), X0       // src[44]
	MOVUPD 2752(R9), X1      // src[172]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 832(R8)       // work[52]
	MOVUPD X3, 848(R8)       // work[53]

	// (108,236) -> work[54], work[55]
	MOVUPD 1728(R9), X0      // src[108]
	MOVUPD 3776(R9), X1      // src[236]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 864(R8)       // work[54]
	MOVUPD X3, 880(R8)       // work[55]

	// (28,156) -> work[56], work[57]
	MOVUPD 448(R9), X0       // src[28]
	MOVUPD 2496(R9), X1      // src[156]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 896(R8)       // work[56]
	MOVUPD X3, 912(R8)       // work[57]

	// (92,220) -> work[58], work[59]
	MOVUPD 1472(R9), X0      // src[92]
	MOVUPD 3520(R9), X1      // src[220]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 928(R8)       // work[58]
	MOVUPD X3, 944(R8)       // work[59]

	// (60,188) -> work[60], work[61]
	MOVUPD 960(R9), X0       // src[60]
	MOVUPD 3008(R9), X1      // src[188]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 960(R8)       // work[60]
	MOVUPD X3, 976(R8)       // work[61]

	// (124,252) -> work[62], work[63]
	MOVUPD 1984(R9), X0      // src[124]
	MOVUPD 4032(R9), X1      // src[252]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 992(R8)       // work[62]
	MOVUPD X3, 1008(R8)      // work[63]

	// (2,130) -> work[64], work[65]
	MOVUPD 32(R9), X0        // src[2]
	MOVUPD 2080(R9), X1      // src[130]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1024(R8)      // work[64]
	MOVUPD X3, 1040(R8)      // work[65]

	// (66,194) -> work[66], work[67]
	MOVUPD 1056(R9), X0      // src[66]
	MOVUPD 3104(R9), X1      // src[194]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1056(R8)      // work[66]
	MOVUPD X3, 1072(R8)      // work[67]

	// (34,162) -> work[68], work[69]
	MOVUPD 544(R9), X0       // src[34]
	MOVUPD 2592(R9), X1      // src[162]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1088(R8)      // work[68]
	MOVUPD X3, 1104(R8)      // work[69]

	// (98,226) -> work[70], work[71]
	MOVUPD 1568(R9), X0      // src[98]
	MOVUPD 3616(R9), X1      // src[226]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1120(R8)      // work[70]
	MOVUPD X3, 1136(R8)      // work[71]

	// (18,146) -> work[72], work[73]
	MOVUPD 288(R9), X0       // src[18]
	MOVUPD 2336(R9), X1      // src[146]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1152(R8)      // work[72]
	MOVUPD X3, 1168(R8)      // work[73]

	// (82,210) -> work[74], work[75]
	MOVUPD 1312(R9), X0      // src[82]
	MOVUPD 3360(R9), X1      // src[210]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1184(R8)      // work[74]
	MOVUPD X3, 1200(R8)      // work[75]

	// (50,178) -> work[76], work[77]
	MOVUPD 800(R9), X0       // src[50]
	MOVUPD 2848(R9), X1      // src[178]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1216(R8)      // work[76]
	MOVUPD X3, 1232(R8)      // work[77]

	// (114,242) -> work[78], work[79]
	MOVUPD 1824(R9), X0      // src[114]
	MOVUPD 3872(R9), X1      // src[242]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1248(R8)      // work[78]
	MOVUPD X3, 1264(R8)      // work[79]

	// (10,138) -> work[80], work[81]
	MOVUPD 160(R9), X0       // src[10]
	MOVUPD 2208(R9), X1      // src[138]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1280(R8)      // work[80]
	MOVUPD X3, 1296(R8)      // work[81]

	// (74,202) -> work[82], work[83]
	MOVUPD 1184(R9), X0      // src[74]
	MOVUPD 3232(R9), X1      // src[202]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1312(R8)      // work[82]
	MOVUPD X3, 1328(R8)      // work[83]

	// (42,170) -> work[84], work[85]
	MOVUPD 672(R9), X0       // src[42]
	MOVUPD 2720(R9), X1      // src[170]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1344(R8)      // work[84]
	MOVUPD X3, 1360(R8)      // work[85]

	// (106,234) -> work[86], work[87]
	MOVUPD 1696(R9), X0      // src[106]
	MOVUPD 3744(R9), X1      // src[234]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1376(R8)      // work[86]
	MOVUPD X3, 1392(R8)      // work[87]

	// (26,154) -> work[88], work[89]
	MOVUPD 416(R9), X0       // src[26]
	MOVUPD 2464(R9), X1      // src[154]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1408(R8)      // work[88]
	MOVUPD X3, 1424(R8)      // work[89]

	// (90,218) -> work[90], work[91]
	MOVUPD 1440(R9), X0      // src[90]
	MOVUPD 3488(R9), X1      // src[218]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1440(R8)      // work[90]
	MOVUPD X3, 1456(R8)      // work[91]

	// (58,186) -> work[92], work[93]
	MOVUPD 928(R9), X0       // src[58]
	MOVUPD 2976(R9), X1      // src[186]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1472(R8)      // work[92]
	MOVUPD X3, 1488(R8)      // work[93]

	// (122,250) -> work[94], work[95]
	MOVUPD 1952(R9), X0      // src[122]
	MOVUPD 4000(R9), X1      // src[250]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1504(R8)      // work[94]
	MOVUPD X3, 1520(R8)      // work[95]

	// (6,134) -> work[96], work[97]
	MOVUPD 96(R9), X0        // src[6]
	MOVUPD 2144(R9), X1      // src[134]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1536(R8)      // work[96]
	MOVUPD X3, 1552(R8)      // work[97]

	// (70,198) -> work[98], work[99]
	MOVUPD 1120(R9), X0      // src[70]
	MOVUPD 3168(R9), X1      // src[198]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1568(R8)      // work[98]
	MOVUPD X3, 1584(R8)      // work[99]

	// (38,166) -> work[100], work[101]
	MOVUPD 608(R9), X0       // src[38]
	MOVUPD 2656(R9), X1      // src[166]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1600(R8)      // work[100]
	MOVUPD X3, 1616(R8)      // work[101]

	// (102,230) -> work[102], work[103]
	MOVUPD 1632(R9), X0      // src[102]
	MOVUPD 3680(R9), X1      // src[230]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1632(R8)      // work[102]
	MOVUPD X3, 1648(R8)      // work[103]

	// (22,150) -> work[104], work[105]
	MOVUPD 352(R9), X0       // src[22]
	MOVUPD 2400(R9), X1      // src[150]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1664(R8)      // work[104]
	MOVUPD X3, 1680(R8)      // work[105]

	// (86,214) -> work[106], work[107]
	MOVUPD 1376(R9), X0      // src[86]
	MOVUPD 3424(R9), X1      // src[214]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1696(R8)      // work[106]
	MOVUPD X3, 1712(R8)      // work[107]

	// (54,182) -> work[108], work[109]
	MOVUPD 864(R9), X0       // src[54]
	MOVUPD 2912(R9), X1      // src[182]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1728(R8)      // work[108]
	MOVUPD X3, 1744(R8)      // work[109]

	// (118,246) -> work[110], work[111]
	MOVUPD 1888(R9), X0      // src[118]
	MOVUPD 3936(R9), X1      // src[246]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1760(R8)      // work[110]
	MOVUPD X3, 1776(R8)      // work[111]

	// (14,142) -> work[112], work[113]
	MOVUPD 224(R9), X0       // src[14]
	MOVUPD 2272(R9), X1      // src[142]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1792(R8)      // work[112]
	MOVUPD X3, 1808(R8)      // work[113]

	// (78,206) -> work[114], work[115]
	MOVUPD 1248(R9), X0      // src[78]
	MOVUPD 3296(R9), X1      // src[206]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1824(R8)      // work[114]
	MOVUPD X3, 1840(R8)      // work[115]

	// (46,174) -> work[116], work[117]
	MOVUPD 736(R9), X0       // src[46]
	MOVUPD 2784(R9), X1      // src[174]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1856(R8)      // work[116]
	MOVUPD X3, 1872(R8)      // work[117]

	// (110,238) -> work[118], work[119]
	MOVUPD 1760(R9), X0      // src[110]
	MOVUPD 3808(R9), X1      // src[238]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1888(R8)      // work[118]
	MOVUPD X3, 1904(R8)      // work[119]

	// (30,158) -> work[120], work[121]
	MOVUPD 480(R9), X0       // src[30]
	MOVUPD 2528(R9), X1      // src[158]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1920(R8)      // work[120]
	MOVUPD X3, 1936(R8)      // work[121]

	// (94,222) -> work[122], work[123]
	MOVUPD 1504(R9), X0      // src[94]
	MOVUPD 3552(R9), X1      // src[222]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1952(R8)      // work[122]
	MOVUPD X3, 1968(R8)      // work[123]

	// (62,190) -> work[124], work[125]
	MOVUPD 992(R9), X0       // src[62]
	MOVUPD 3040(R9), X1      // src[190]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 1984(R8)      // work[124]
	MOVUPD X3, 2000(R8)      // work[125]

	// (126,254) -> work[126], work[127]
	MOVUPD 2016(R9), X0      // src[126]
	MOVUPD 4064(R9), X1      // src[254]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 2016(R8)      // work[126]
	MOVUPD X3, 2032(R8)      // work[127]

	INCQ BX                  // next pass
	CMPQ BX, $2              // done after odd pass
	JGE  inv_stage1_done
	LEAQ 2048(R14), R8       // work offset for odd half
	LEAQ 16(R15), R9         // src offset for odd half
	JMP  inv_stage1_pass

inv_stage1_done:
	MOVQ R14, R8             // restore work base

	// Prepare masks for next stages
	MOVUPS ·maskNegHiPD(SB), X14 // Conjugate W: (wr, wi) -> (wr, -wi)
	MOVUPS ·maskNegLoPD(SB), X13 // Negate term in complex mul

	// Stage 2: dist 2 - 64 blocks of 4
	MOVQ R8, SI              // work base
	MOVQ $64, CX             // blocks
inv_stage2_loop:
	MOVQ $2, DX              // half=2
inv_stage2_inner:
	MOVUPD (SI), X0          // a
  MOVUPD 32(SI), X1        // b
	MOVQ $2, AX              // k = 2 - DX
  SUBQ DX, AX              // k (0..1)
  SHLQ $6, AX              // k * 64
  SHLQ $4, AX              // k * 64 * 16
  MOVUPD (R10)(AX*1), X10  // twiddle[k*64]
  XORPD X14, X10           // conj(w)
	MOVAPD X1, X2            // b
  UNPCKLPD X2, X2          // b.re
  MULPD X10, X2            // b.re * w
	MOVAPD X1, X3            // b
  UNPCKHPD X3, X3          // b.im
  MOVAPD X10, X4           // w
  SHUFPD $1, X4, X4        // swap
  MULPD X3, X4             // b.im * w
	XORPD X13, X4            // multiply by i
  ADDPD X4, X2             // t = w * b
	MOVAPD X0, X3            // a
  ADDPD X2, X0             // a + t
  SUBPD X2, X3             // a - t
	MOVUPD X0, (SI)          // out a
  MOVUPD X3, 32(SI)        // out b
	ADDQ $16, SI             // next j
  DECQ DX                  // next j
  JNZ inv_stage2_inner
	ADDQ $32, SI             // next block
  DECQ CX                  // next block
  JNZ inv_stage2_loop

	// Stage 3
	MOVQ R8, SI
	MOVQ $32, CX

inv_s3_loop:
	MOVQ $4, DX

inv_s3_inner:
	MOVUPD (SI), X0
	MOVUPD 64(SI), X1

	MOVQ $4, AX
	SUBQ DX, AX
	SHLQ $5, AX             // k * 32
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X14, X10          // Conj(W)

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2

	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X13, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 64(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ inv_s3_inner

	ADDQ $64, SI
	DECQ CX
	JNZ inv_s3_loop

	// Stage 4
	MOVQ R8, SI
	MOVQ $16, CX

inv_s4_loop:
	MOVQ $8, DX

inv_s4_inner:
	MOVUPD (SI), X0
	MOVUPD 128(SI), X1

	MOVQ $8, AX
	SUBQ DX, AX
	SHLQ $4, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X14, X10          // Conj(W)

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2

	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X13, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 128(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ inv_s4_inner

	ADDQ $128, SI
	DECQ CX
	JNZ inv_s4_loop

	// Stage 5
	MOVQ R8, SI
	MOVQ $8, CX

inv_s5_loop:
	MOVQ $16, DX

inv_s5_inner:
	MOVUPD (SI), X0
	MOVUPD 256(SI), X1

	MOVQ $16, AX
	SUBQ DX, AX
	SHLQ $3, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X14, X10          // Conj(W)

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2

	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X13, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 256(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ inv_s5_inner

	ADDQ $256, SI
	DECQ CX
	JNZ inv_s5_loop

	// Stage 6
	MOVQ R8, SI
	MOVQ $4, CX

inv_s6_loop:
	MOVQ $32, DX

inv_s6_inner:
	MOVUPD (SI), X0
	MOVUPD 512(SI), X1

	MOVQ $32, AX
	SUBQ DX, AX
	SHLQ $2, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X14, X10          // Conj(W)

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2

	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X13, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 512(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ inv_s6_inner

	ADDQ $512, SI
	DECQ CX
	JNZ inv_s6_loop

	// Stage 7
	MOVQ R8, SI
	MOVQ $2, CX

inv_s7_loop:
	MOVQ $64, DX

inv_s7_inner:
	MOVUPD (SI), X0
	MOVUPD 1024(SI), X1

	MOVQ $64, AX
	SUBQ DX, AX
	SHLQ $1, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X14, X10          // Conj(W)

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2

	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X13, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 1024(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ inv_s7_inner

	ADDQ $1024, SI
	DECQ CX
	JNZ inv_s7_loop

	// Stage 8
	MOVQ R8, SI
	MOVQ $128, DX

inv_s8_inner:
	MOVUPD (SI), X0
	MOVUPD 2048(SI), X1

	MOVQ $128, AX
	SUBQ DX, AX
	                        // k * 1
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X14, X10          // Conj(W)

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2

	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X13, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 2048(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ inv_s8_inner

	// Scale by 1/256
	MOVSD ·twoFiftySixth64(SB), X15
	SHUFPD $0, X15, X15     // Broadcast scaler
	MOVQ $128, CX
	MOVQ R8, SI

inv_scale:
	MOVUPD (SI), X0
	MOVUPD 16(SI), X1
	MULPD X15, X0
	MULPD X15, X1
	MOVUPD X0, (SI)
	MOVUPD X1, 16(SI)
	ADDQ $32, SI
	DECQ CX
	JNZ inv_scale

	// Copy to dst
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   inv_done

	MOVQ $128, CX
	MOVQ R8, SI
	MOVQ R14, DI

inv_copy:
	MOVUPD (SI), X0
	MOVUPD 16(SI), X1
	MOVUPD X0, (DI)
	MOVUPD X1, 16(DI)
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ inv_copy

inv_done:
	MOVB $1, ret+96(FP)
	RET

inv_err:
	MOVB $0, ret+96(FP)
	RET

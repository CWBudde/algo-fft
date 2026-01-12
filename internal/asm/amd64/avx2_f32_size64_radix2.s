//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-64 Radix-2 FFT Kernels for AMD64
// ===========================================================================

#include "textflag.h"

// Forward transform, size 64, complex64
// Fully unrolled 6-stage FFT with AVX2 vectorization
//
// This kernel implements a complete radix-2 DIT FFT for exactly 64 complex64 values.
// All 6 stages are fully unrolled with hardcoded twiddle factor indices:
//   Stage 1 (size=2):  32 butterflies, step=32, twiddle index 0 for all
//   Stage 2 (size=4):  32 butterflies in 8 groups, step=16, twiddle indices [0,16]
//   Stage 3 (size=8):  32 butterflies in 4 groups, step=8, twiddle indices [0,8,16,24]
//   Stage 4 (size=16): 32 butterflies in 2 groups, step=4, twiddle indices [0,4,8,12,16,20,24,28]
//   Stage 5 (size=32): 32 butterflies in 1 group, step=2, twiddle indices [0,2,...,30]
//   Stage 6 (size=64): 32 butterflies, step=1, twiddle indices [0,1,2,...,31]
//
// Register allocation:
//   R8:  work buffer (dst or scratch for in-place)
//   R9:  src pointer
//   R10: twiddle pointer
//   R11: scratch pointer
//   R12: bitrev pointer
//   Data stored in memory (R8), processed in groups of 4 YMM registers
//
TEXT ·ForwardAVX2Size64Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 64)

	// Verify n == 64
	CMPQ R13, $64
	JNE  size64_return_false

	// Validate all slice lengths >= 64
	MOVQ dst+8(FP), AX
	CMPQ AX, $64
	JL   size64_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $64
	JL   size64_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $64
	JL   size64_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size64_fwd_start

	// In-place: use scratch
	MOVQ R11, R8

size64_fwd_start:
	// =======================================================================
	// FUSED: Bit-reversal permutation + STAGE 1 (identity twiddles)
	// =======================================================================
	// Load bit-reversed data directly into YMM registers and compute Stage 1.
	// Bitrev pattern: [0,32,16,48, 8,40,24,56, 4,36,20,52, 12,44,28,60, ...]
	// Each YMM holds 4 complex64 values (32 bytes total).
	// Stage 1 butterfly: pairs (d0,d1) and (d2,d3) → (d0+d1, d0-d1, d2+d3, d2-d3)

	// -----------------------------------------------------------------------
	// Group 0: work[0..3] = src[0,32,16,48] → Stage 1 butterfly → store at 0
	// -----------------------------------------------------------------------
	VMOVSD (R9), X0              // X0 = c0 (src[0])
	VMOVSD 256(R9), X4           // X4 = c32 (src[32], 32*8=256)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c0, c32]
	VMOVSD 128(R9), X5           // X5 = c16 (src[16], 16*8=128)
	VMOVSD 384(R9), X6           // X6 = c48 (src[48], 48*8=384)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c16, c48]
	VINSERTF128 $1, X5, Y0, Y0   // Y0 = [c0, c32, c16, c48]
	VPERMILPD $0x05, Y0, Y4      // Y4 = [c32, c0, c48, c16]
	VADDPS Y4, Y0, Y5            // Y5 = [c0+c32, c32+c0, c16+c48, c48+c16]
	VSUBPS Y0, Y4, Y6            // Y6 = [c32-c0, c0-c32, c48-c16, c16-c48]
	VBLENDPD $0x0A, Y6, Y5, Y0   // Y0 = [c0+c32, c0-c32, c16+c48, c16-c48]
	VMOVUPS Y0, (R8)

	// -----------------------------------------------------------------------
	// Group 1: work[4..7] = src[8,40,24,56] → Stage 1 butterfly → store at 32
	// -----------------------------------------------------------------------
	VMOVSD 64(R9), X0            // X0 = c8 (src[8], 8*8=64)
	VMOVSD 320(R9), X4           // X4 = c40 (src[40], 40*8=320)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c8, c40]
	VMOVSD 192(R9), X5           // X5 = c24 (src[24], 24*8=192)
	VMOVSD 448(R9), X6           // X6 = c56 (src[56], 56*8=448)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c24, c56]
	VINSERTF128 $1, X5, Y0, Y1   // Y1 = [c8, c40, c24, c56]
	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1
	VMOVUPS Y1, 32(R8)

	// -----------------------------------------------------------------------
	// Group 2: work[8..11] = src[4,36,20,52] → Stage 1 butterfly → store at 64
	// -----------------------------------------------------------------------
	VMOVSD 32(R9), X0            // X0 = c4 (src[4], 4*8=32)
	VMOVSD 288(R9), X4           // X4 = c36 (src[36], 36*8=288)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c4, c36]
	VMOVSD 160(R9), X5           // X5 = c20 (src[20], 20*8=160)
	VMOVSD 416(R9), X6           // X6 = c52 (src[52], 52*8=416)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c20, c52]
	VINSERTF128 $1, X5, Y0, Y2   // Y2 = [c4, c36, c20, c52]
	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y2, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y2
	VMOVUPS Y2, 64(R8)

	// -----------------------------------------------------------------------
	// Group 3: work[12..15] = src[12,44,28,60] → Stage 1 butterfly → store at 96
	// -----------------------------------------------------------------------
	VMOVSD 96(R9), X0            // X0 = c12 (src[12], 12*8=96)
	VMOVSD 352(R9), X4           // X4 = c44 (src[44], 44*8=352)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c12, c44]
	VMOVSD 224(R9), X5           // X5 = c28 (src[28], 28*8=224)
	VMOVSD 480(R9), X6           // X6 = c60 (src[60], 60*8=480)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c28, c60]
	VINSERTF128 $1, X5, Y0, Y3   // Y3 = [c12, c44, c28, c60]
	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y3, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y3
	VMOVUPS Y3, 96(R8)

	// -----------------------------------------------------------------------
	// Group 4: work[16..19] = src[2,34,18,50] → Stage 1 butterfly → store at 128
	// -----------------------------------------------------------------------
	VMOVSD 16(R9), X0            // X0 = c2 (src[2], 2*8=16)
	VMOVSD 272(R9), X4           // X4 = c34 (src[34], 34*8=272)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c2, c34]
	VMOVSD 144(R9), X5           // X5 = c18 (src[18], 18*8=144)
	VMOVSD 400(R9), X6           // X6 = c50 (src[50], 50*8=400)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c18, c50]
	VINSERTF128 $1, X5, Y0, Y0   // Y0 = [c2, c34, c18, c50]
	VPERMILPD $0x05, Y0, Y4
	VADDPS Y4, Y0, Y5
	VSUBPS Y0, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y0
	VMOVUPS Y0, 128(R8)

	// -----------------------------------------------------------------------
	// Group 5: work[20..23] = src[10,42,26,58] → Stage 1 butterfly → store at 160
	// -----------------------------------------------------------------------
	VMOVSD 80(R9), X0            // X0 = c10 (src[10], 10*8=80)
	VMOVSD 336(R9), X4           // X4 = c42 (src[42], 42*8=336)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c10, c42]
	VMOVSD 208(R9), X5           // X5 = c26 (src[26], 26*8=208)
	VMOVSD 464(R9), X6           // X6 = c58 (src[58], 58*8=464)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c26, c58]
	VINSERTF128 $1, X5, Y0, Y1   // Y1 = [c10, c42, c26, c58]
	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1
	VMOVUPS Y1, 160(R8)

	// -----------------------------------------------------------------------
	// Group 6: work[24..27] = src[6,38,22,54] → Stage 1 butterfly → store at 192
	// -----------------------------------------------------------------------
	VMOVSD 48(R9), X0            // X0 = c6 (src[6], 6*8=48)
	VMOVSD 304(R9), X4           // X4 = c38 (src[38], 38*8=304)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c6, c38]
	VMOVSD 176(R9), X5           // X5 = c22 (src[22], 22*8=176)
	VMOVSD 432(R9), X6           // X6 = c54 (src[54], 54*8=432)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c22, c54]
	VINSERTF128 $1, X5, Y0, Y2   // Y2 = [c6, c38, c22, c54]
	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y2, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y2
	VMOVUPS Y2, 192(R8)

	// -----------------------------------------------------------------------
	// Group 7: work[28..31] = src[14,46,30,62] → Stage 1 butterfly → store at 224
	// -----------------------------------------------------------------------
	VMOVSD 112(R9), X0           // X0 = c14 (src[14], 14*8=112)
	VMOVSD 368(R9), X4           // X4 = c46 (src[46], 46*8=368)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c14, c46]
	VMOVSD 240(R9), X5           // X5 = c30 (src[30], 30*8=240)
	VMOVSD 496(R9), X6           // X6 = c62 (src[62], 62*8=496)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c30, c62]
	VINSERTF128 $1, X5, Y0, Y3   // Y3 = [c14, c46, c30, c62]
	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y3, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y3
	VMOVUPS Y3, 224(R8)

	// -----------------------------------------------------------------------
	// Group 8: work[32..35] = src[1,33,17,49] → Stage 1 butterfly → store at 256
	// -----------------------------------------------------------------------
	VMOVSD 8(R9), X0             // X0 = c1 (src[1], 1*8=8)
	VMOVSD 264(R9), X4           // X4 = c33 (src[33], 33*8=264)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c1, c33]
	VMOVSD 136(R9), X5           // X5 = c17 (src[17], 17*8=136)
	VMOVSD 392(R9), X6           // X6 = c49 (src[49], 49*8=392)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c17, c49]
	VINSERTF128 $1, X5, Y0, Y0   // Y0 = [c1, c33, c17, c49]
	VPERMILPD $0x05, Y0, Y4
	VADDPS Y4, Y0, Y5
	VSUBPS Y0, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y0
	VMOVUPS Y0, 256(R8)

	// -----------------------------------------------------------------------
	// Group 9: work[36..39] = src[9,41,25,57] → Stage 1 butterfly → store at 288
	// -----------------------------------------------------------------------
	VMOVSD 72(R9), X0            // X0 = c9 (src[9], 9*8=72)
	VMOVSD 328(R9), X4           // X4 = c41 (src[41], 41*8=328)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c9, c41]
	VMOVSD 200(R9), X5           // X5 = c25 (src[25], 25*8=200)
	VMOVSD 456(R9), X6           // X6 = c57 (src[57], 57*8=456)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c25, c57]
	VINSERTF128 $1, X5, Y0, Y1   // Y1 = [c9, c41, c25, c57]
	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1
	VMOVUPS Y1, 288(R8)

	// -----------------------------------------------------------------------
	// Group 10: work[40..43] = src[5,37,21,53] → Stage 1 butterfly → store at 320
	// -----------------------------------------------------------------------
	VMOVSD 40(R9), X0            // X0 = c5 (src[5], 5*8=40)
	VMOVSD 296(R9), X4           // X4 = c37 (src[37], 37*8=296)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c5, c37]
	VMOVSD 168(R9), X5           // X5 = c21 (src[21], 21*8=168)
	VMOVSD 424(R9), X6           // X6 = c53 (src[53], 53*8=424)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c21, c53]
	VINSERTF128 $1, X5, Y0, Y2   // Y2 = [c5, c37, c21, c53]
	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y2, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y2
	VMOVUPS Y2, 320(R8)

	// -----------------------------------------------------------------------
	// Group 11: work[44..47] = src[13,45,29,61] → Stage 1 butterfly → store at 352
	// -----------------------------------------------------------------------
	VMOVSD 104(R9), X0           // X0 = c13 (src[13], 13*8=104)
	VMOVSD 360(R9), X4           // X4 = c45 (src[45], 45*8=360)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c13, c45]
	VMOVSD 232(R9), X5           // X5 = c29 (src[29], 29*8=232)
	VMOVSD 488(R9), X6           // X6 = c61 (src[61], 61*8=488)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c29, c61]
	VINSERTF128 $1, X5, Y0, Y3   // Y3 = [c13, c45, c29, c61]
	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y3, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y3
	VMOVUPS Y3, 352(R8)

	// -----------------------------------------------------------------------
	// Group 12: work[48..51] = src[3,35,19,51] → Stage 1 butterfly → store at 384
	// -----------------------------------------------------------------------
	VMOVSD 24(R9), X0            // X0 = c3 (src[3], 3*8=24)
	VMOVSD 280(R9), X4           // X4 = c35 (src[35], 35*8=280)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c3, c35]
	VMOVSD 152(R9), X5           // X5 = c19 (src[19], 19*8=152)
	VMOVSD 408(R9), X6           // X6 = c51 (src[51], 51*8=408)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c19, c51]
	VINSERTF128 $1, X5, Y0, Y0   // Y0 = [c3, c35, c19, c51]
	VPERMILPD $0x05, Y0, Y4
	VADDPS Y4, Y0, Y5
	VSUBPS Y0, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y0
	VMOVUPS Y0, 384(R8)

	// -----------------------------------------------------------------------
	// Group 13: work[52..55] = src[11,43,27,59] → Stage 1 butterfly → store at 416
	// -----------------------------------------------------------------------
	VMOVSD 88(R9), X0            // X0 = c11 (src[11], 11*8=88)
	VMOVSD 344(R9), X4           // X4 = c43 (src[43], 43*8=344)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c11, c43]
	VMOVSD 216(R9), X5           // X5 = c27 (src[27], 27*8=216)
	VMOVSD 472(R9), X6           // X6 = c59 (src[59], 59*8=472)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c27, c59]
	VINSERTF128 $1, X5, Y0, Y1   // Y1 = [c11, c43, c27, c59]
	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1
	VMOVUPS Y1, 416(R8)

	// -----------------------------------------------------------------------
	// Group 14: work[56..59] = src[7,39,23,55] → Stage 1 butterfly → store at 448
	// -----------------------------------------------------------------------
	VMOVSD 56(R9), X0            // X0 = c7 (src[7], 7*8=56)
	VMOVSD 312(R9), X4           // X4 = c39 (src[39], 39*8=312)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c7, c39]
	VMOVSD 184(R9), X5           // X5 = c23 (src[23], 23*8=184)
	VMOVSD 440(R9), X6           // X6 = c55 (src[55], 55*8=440)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c23, c55]
	VINSERTF128 $1, X5, Y0, Y2   // Y2 = [c7, c39, c23, c55]
	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y2, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y2
	VMOVUPS Y2, 448(R8)

	// -----------------------------------------------------------------------
	// Group 15: work[60..63] = src[15,47,31,63] → Stage 1 butterfly → store at 480
	// -----------------------------------------------------------------------
	VMOVSD 120(R9), X0           // X0 = c15 (src[15], 15*8=120)
	VMOVSD 376(R9), X4           // X4 = c47 (src[47], 47*8=376)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c15, c47]
	VMOVSD 248(R9), X5           // X5 = c31 (src[31], 31*8=248)
	VMOVSD 504(R9), X6           // X6 = c63 (src[63], 63*8=504)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c31, c63]
	VINSERTF128 $1, X5, Y0, Y3   // Y3 = [c15, c47, c31, c63]
	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y3, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y3
	VMOVUPS Y3, 480(R8)

	// =======================================================================
	// STAGE 2: size=4, half=2, step=16
	// =======================================================================
	// Twiddle factors: twiddle[0], twiddle[16]
	// twiddle[0] = (1, 0), twiddle[16] = (0, -1) for n=64

	VMOVSD (R10), X8           // twiddle[0]
	VMOVSD 128(R10), X9        // twiddle[16] (16 * 8 bytes)
	VPUNPCKLQDQ X9, X8, X8     // X8 = [tw0, tw16]
	VINSERTF128 $1, X8, Y8, Y8 // Y8 = [tw0, tw16, tw0, tw16]

	// Process all YMM registers with stage 2 butterflies
	// Each YMM has 4 complex values [d0, d1, d2, d3]
	// Butterfly pairs: (d0, d2) and (d1, d3)

	// Helper macro pattern for stage 2 (process one YMM at offset)
	// Load, extract halves, multiply, butterfly, store

	// Indices 0-3
	VMOVUPS (R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1    // Y1 = [d0, d1, d0, d1]
	VPERM2F128 $0x11, Y0, Y0, Y2    // Y2 = [d2, d3, d2, d3]
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, (R8)

	// Indices 4-7
	VMOVUPS 32(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 32(R8)

	// Indices 8-11
	VMOVUPS 64(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 64(R8)

	// Indices 12-15
	VMOVUPS 96(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 96(R8)

	// Indices 16-19
	VMOVUPS 128(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 128(R8)

	// Indices 20-23
	VMOVUPS 160(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 160(R8)

	// Indices 24-27
	VMOVUPS 192(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 192(R8)

	// Indices 28-31
	VMOVUPS 224(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 224(R8)

	// Indices 32-35
	VMOVUPS 256(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 256(R8)

	// Indices 36-39
	VMOVUPS 288(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 288(R8)

	// Indices 40-43
	VMOVUPS 320(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 320(R8)

	// Indices 44-47
	VMOVUPS 352(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 352(R8)

	// Indices 48-51
	VMOVUPS 384(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 384(R8)

	// Indices 52-55
	VMOVUPS 416(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 416(R8)

	// Indices 56-59
	VMOVUPS 448(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 448(R8)

	// Indices 60-63
	VMOVUPS 480(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 480(R8)

	// =======================================================================
	// STAGE 3: size=8, half=4, step=8
	// =======================================================================
	// Pairs: indices 0-3 with 4-7, 8-11 with 12-15, etc.
	// Twiddle factors: twiddle[0], twiddle[8], twiddle[16], twiddle[24]

	VMOVSD (R10), X8           // twiddle[0]
	VMOVSD 64(R10), X9         // twiddle[8]
	VPUNPCKLQDQ X9, X8, X8     // X8 = [tw0, tw8]
	VMOVSD 128(R10), X9        // twiddle[16]
	VMOVSD 192(R10), X10       // twiddle[24]
	VPUNPCKLQDQ X10, X9, X9    // X9 = [tw16, tw24]
	VINSERTF128 $1, X9, Y8, Y8 // Y8 = [tw0, tw8, tw16, tw24]

	// Extract twiddle components once
	VMOVSLDUP Y8, Y14          // Y14 = [w.r, w.r, ...]
	VMOVSHDUP Y8, Y15          // Y15 = [w.i, w.i, ...]

	// Group: indices 0-3 with 4-7
	VMOVUPS (R8), Y0           // a = indices 0-3
	VMOVUPS 32(R8), Y1         // b = indices 4-7
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2 // Y2 = t = w * b
	VADDPS Y2, Y0, Y3          // Y3 = a + t
	VSUBPS Y2, Y0, Y4          // Y4 = a - t
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 32(R8)

	// Group: indices 8-11 with 12-15
	VMOVUPS 64(R8), Y0
	VMOVUPS 96(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 64(R8)
	VMOVUPS Y4, 96(R8)

	// Group: indices 16-19 with 20-23
	VMOVUPS 128(R8), Y0
	VMOVUPS 160(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 128(R8)
	VMOVUPS Y4, 160(R8)

	// Group: indices 24-27 with 28-31
	VMOVUPS 192(R8), Y0
	VMOVUPS 224(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 192(R8)
	VMOVUPS Y4, 224(R8)

	// Group: indices 32-35 with 36-39
	VMOVUPS 256(R8), Y0
	VMOVUPS 288(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y4, 288(R8)

	// Group: indices 40-43 with 44-47
	VMOVUPS 320(R8), Y0
	VMOVUPS 352(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 320(R8)
	VMOVUPS Y4, 352(R8)

	// Group: indices 48-51 with 52-55
	VMOVUPS 384(R8), Y0
	VMOVUPS 416(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 384(R8)
	VMOVUPS Y4, 416(R8)

	// Group: indices 56-59 with 60-63
	VMOVUPS 448(R8), Y0
	VMOVUPS 480(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 448(R8)
	VMOVUPS Y4, 480(R8)

	// =======================================================================
	// STAGE 4: size=16, half=8, step=4
	// =======================================================================
	// Pairs: indices 0-7 with 8-15, 16-23 with 24-31, 32-39 with 40-47, 48-55 with 56-63
	// Twiddle factors: j=0..3 use [0,4,8,12], j=4..7 use [16,20,24,28]

	VMOVSD (R10), X8           // twiddle[0]
	VMOVSD 32(R10), X9         // twiddle[4]
	VPUNPCKLQDQ X9, X8, X8     // X8 = [tw0, tw4]
	VMOVSD 64(R10), X9         // twiddle[8]
	VMOVSD 96(R10), X10        // twiddle[12]
	VPUNPCKLQDQ X10, X9, X9    // X9 = [tw8, tw12]
	VINSERTF128 $1, X9, Y8, Y8 // Y8 = [tw0, tw4, tw8, tw12]

	VMOVSD 128(R10), X9        // twiddle[16]
	VMOVSD 160(R10), X10       // twiddle[20]
	VPUNPCKLQDQ X10, X9, X9    // X9 = [tw16, tw20]
	VMOVSD 192(R10), X10       // twiddle[24]
	VMOVSD 224(R10), X11       // twiddle[28]
	VPUNPCKLQDQ X11, X10, X10  // X10 = [tw24, tw28]
	VINSERTF128 $1, X10, Y9, Y9 // Y9 = [tw16, tw20, tw24, tw28]

	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15

	// Group 1: indices 0-3 with 8-11 (first half of 16-point group)
	VMOVUPS (R8), Y0
	VMOVUPS 64(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 64(R8)

	// Group 1: indices 4-7 with 12-15
	VMOVUPS 32(R8), Y0
	VMOVUPS 96(R8), Y1
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 32(R8)
	VMOVUPS Y4, 96(R8)

	// Group 2: indices 16-19 with 24-27
	VMOVUPS 128(R8), Y0
	VMOVUPS 192(R8), Y1
	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 128(R8)
	VMOVUPS Y4, 192(R8)

	// Group 2: indices 20-23 with 28-31
	VMOVUPS 160(R8), Y0
	VMOVUPS 224(R8), Y1
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 160(R8)
	VMOVUPS Y4, 224(R8)

	// Group 3: indices 32-35 with 40-43
	VMOVUPS 256(R8), Y0
	VMOVUPS 320(R8), Y1
	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y4, 320(R8)

	// Group 3: indices 36-39 with 44-47
	VMOVUPS 288(R8), Y0
	VMOVUPS 352(R8), Y1
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 288(R8)
	VMOVUPS Y4, 352(R8)

	// Group 4: indices 48-51 with 56-59
	VMOVUPS 384(R8), Y0
	VMOVUPS 448(R8), Y1
	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 384(R8)
	VMOVUPS Y4, 448(R8)

	// Group 4: indices 52-55 with 60-63
	VMOVUPS 416(R8), Y0
	VMOVUPS 480(R8), Y1
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 416(R8)
	VMOVUPS Y4, 480(R8)

	// =======================================================================
	// STAGE 5: size=32, half=16, step=2
	// =======================================================================
	// Pairs: indices 0-15 with 16-31, 32-47 with 48-63
	// Twiddle factors: j=0..3 -> [0,2,4,6], j=4..7 -> [8,10,12,14],
	//                   j=8..11 -> [16,18,20,22], j=12..15 -> [24,26,28,30]

	// Load twiddles for indices 0-3: tw[0,2,4,6]
	VMOVSD (R10), X8           // twiddle[0]
	VMOVSD 16(R10), X9         // twiddle[2]
	VPUNPCKLQDQ X9, X8, X8
	VMOVSD 32(R10), X9         // twiddle[4]
	VMOVSD 48(R10), X10        // twiddle[6]
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8 // Y8 = [tw0, tw2, tw4, tw6]

	// Load twiddles for indices 4-7: tw[8,10,12,14]
	VMOVSD 64(R10), X9         // twiddle[8]
	VMOVSD 80(R10), X10        // twiddle[10]
	VPUNPCKLQDQ X10, X9, X9
	VMOVSD 96(R10), X10        // twiddle[12]
	VMOVSD 112(R10), X11       // twiddle[14]
	VPUNPCKLQDQ X11, X10, X10
	VINSERTF128 $1, X10, Y9, Y9 // Y9 = [tw8, tw10, tw12, tw14]

	// Load twiddles for indices 8-11: tw[16,18,20,22]
	VMOVSD 128(R10), X10       // twiddle[16]
	VMOVSD 144(R10), X11       // twiddle[18]
	VPUNPCKLQDQ X11, X10, X10
	VMOVSD 160(R10), X11       // twiddle[20]
	VMOVSD 176(R10), X12       // twiddle[22]
	VPUNPCKLQDQ X12, X11, X11
	VINSERTF128 $1, X11, Y10, Y10 // Y10 = [tw16, tw18, tw20, tw22]

	// Load twiddles for indices 12-15: tw[24,26,28,30]
	VMOVSD 192(R10), X11       // twiddle[24]
	VMOVSD 208(R10), X12       // twiddle[26]
	VPUNPCKLQDQ X12, X11, X11
	VMOVSD 224(R10), X12       // twiddle[28]
	VMOVSD 240(R10), X13       // twiddle[30]
	VPUNPCKLQDQ X13, X12, X12
	VINSERTF128 $1, X12, Y11, Y11 // Y11 = [tw24, tw26, tw28, tw30]

	// Group 1: indices 0-3 with 16-19
	VMOVUPS (R8), Y0
	VMOVUPS 128(R8), Y1
	VMOVSLDUP Y8, Y12
	VMOVSHDUP Y8, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 128(R8)

	// Group 1: indices 4-7 with 20-23
	VMOVUPS 32(R8), Y0
	VMOVUPS 160(R8), Y1
	VMOVSLDUP Y9, Y12
	VMOVSHDUP Y9, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 32(R8)
	VMOVUPS Y4, 160(R8)

	// Group 1: indices 8-11 with 24-27
	VMOVUPS 64(R8), Y0
	VMOVUPS 192(R8), Y1
	VMOVSLDUP Y10, Y12
	VMOVSHDUP Y10, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 64(R8)
	VMOVUPS Y4, 192(R8)

	// Group 1: indices 12-15 with 28-31
	VMOVUPS 96(R8), Y0
	VMOVUPS 224(R8), Y1
	VMOVSLDUP Y11, Y12
	VMOVSHDUP Y11, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 96(R8)
	VMOVUPS Y4, 224(R8)

	// Group 2: indices 32-35 with 48-51
	VMOVUPS 256(R8), Y0
	VMOVUPS 384(R8), Y1
	VMOVSLDUP Y8, Y12
	VMOVSHDUP Y8, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y4, 384(R8)

	// Group 2: indices 36-39 with 52-55
	VMOVUPS 288(R8), Y0
	VMOVUPS 416(R8), Y1
	VMOVSLDUP Y9, Y12
	VMOVSHDUP Y9, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 288(R8)
	VMOVUPS Y4, 416(R8)

	// Group 2: indices 40-43 with 56-59
	VMOVUPS 320(R8), Y0
	VMOVUPS 448(R8), Y1
	VMOVSLDUP Y10, Y12
	VMOVSHDUP Y10, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 320(R8)
	VMOVUPS Y4, 448(R8)

	// Group 2: indices 44-47 with 60-63
	VMOVUPS 352(R8), Y0
	VMOVUPS 480(R8), Y1
	VMOVSLDUP Y11, Y12
	VMOVSHDUP Y11, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 352(R8)
	VMOVUPS Y4, 480(R8)

	// =======================================================================
	// STAGE 6: size=64, half=32, step=1
	// =======================================================================
	// Pairs: indices 0-31 with 32-63
	// Twiddle factors: twiddle[0,1,2,...,31]

	// Load twiddles for indices 0-3
	VMOVUPS (R10), Y8          // Y8 = [tw0, tw1, tw2, tw3]
	VMOVUPS 32(R10), Y9        // Y9 = [tw4, tw5, tw6, tw7]

	// Group: indices 0-3 with 32-35
	VMOVUPS (R8), Y0
	VMOVUPS 256(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 256(R8)

	// Group: indices 4-7 with 36-39
	VMOVUPS 32(R8), Y0
	VMOVUPS 288(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 32(R8)
	VMOVUPS Y4, 288(R8)

	// Load twiddles for indices 8-15
	VMOVUPS 64(R10), Y8        // Y8 = [tw8, tw9, tw10, tw11]
	VMOVUPS 96(R10), Y9        // Y9 = [tw12, tw13, tw14, tw15]

	// Group: indices 8-11 with 40-43
	VMOVUPS 64(R8), Y0
	VMOVUPS 320(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 64(R8)
	VMOVUPS Y4, 320(R8)

	// Group: indices 12-15 with 44-47
	VMOVUPS 96(R8), Y0
	VMOVUPS 352(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 96(R8)
	VMOVUPS Y4, 352(R8)

	// Load twiddles for indices 16-23
	VMOVUPS 128(R10), Y8       // Y8 = [tw16, tw17, tw18, tw19]
	VMOVUPS 160(R10), Y9       // Y9 = [tw20, tw21, tw22, tw23]

	// Group: indices 16-19 with 48-51
	VMOVUPS 128(R8), Y0
	VMOVUPS 384(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 128(R8)
	VMOVUPS Y4, 384(R8)

	// Group: indices 20-23 with 52-55
	VMOVUPS 160(R8), Y0
	VMOVUPS 416(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 160(R8)
	VMOVUPS Y4, 416(R8)

	// Load twiddles for indices 24-31
	VMOVUPS 192(R10), Y8       // Y8 = [tw24, tw25, tw26, tw27]
	VMOVUPS 224(R10), Y9       // Y9 = [tw28, tw29, tw30, tw31]

	// Group: indices 24-27 with 56-59
	VMOVUPS 192(R8), Y0
	VMOVUPS 448(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 192(R8)
	VMOVUPS Y4, 448(R8)

	// Group: indices 28-31 with 60-63
	VMOVUPS 224(R8), Y0
	VMOVUPS 480(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 224(R8)
	VMOVUPS Y4, 480(R8)

	// =======================================================================
	// Copy results to dst if we used scratch buffer
	// =======================================================================
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size64_done

	// Copy from scratch to dst (512 bytes = 16 YMM registers)
	VMOVUPS (R8), Y0
	VMOVUPS 32(R8), Y1
	VMOVUPS 64(R8), Y2
	VMOVUPS 96(R8), Y3
	VMOVUPS 128(R8), Y4
	VMOVUPS 160(R8), Y5
	VMOVUPS 192(R8), Y6
	VMOVUPS 224(R8), Y7
	VMOVUPS Y0, (R9)
	VMOVUPS Y1, 32(R9)
	VMOVUPS Y2, 64(R9)
	VMOVUPS Y3, 96(R9)
	VMOVUPS Y4, 128(R9)
	VMOVUPS Y5, 160(R9)
	VMOVUPS Y6, 192(R9)
	VMOVUPS Y7, 224(R9)

	VMOVUPS 256(R8), Y0
	VMOVUPS 288(R8), Y1
	VMOVUPS 320(R8), Y2
	VMOVUPS 352(R8), Y3
	VMOVUPS 384(R8), Y4
	VMOVUPS 416(R8), Y5
	VMOVUPS 448(R8), Y6
	VMOVUPS 480(R8), Y7
	VMOVUPS Y0, 256(R9)
	VMOVUPS Y1, 288(R9)
	VMOVUPS Y2, 320(R9)
	VMOVUPS Y3, 352(R9)
	VMOVUPS Y4, 384(R9)
	VMOVUPS Y5, 416(R9)
	VMOVUPS Y6, 448(R9)
	VMOVUPS Y7, 480(R9)

size64_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size64_return_false:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 64, complex64
// Fully unrolled 6-stage IFFT with AVX2/FMA vectorization (DIT).
//
// This kernel mirrors the forward DIT schedule and applies conjugated
// twiddle factors during each butterfly. Inputs are bit-reversed at the start,
// and the output is scaled by 1/64.
//
TEXT ·InverseAVX2Size64Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 64)

	// Verify n == 64
	CMPQ R13, $64
	JNE  inv_size64_return_false

	// Validate all slice lengths >= 64
	MOVQ dst+8(FP), AX
	CMPQ AX, $64
	JL   inv_size64_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $64
	JL   inv_size64_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $64
	JL   inv_size64_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  inv_size64_use_dst

	// In-place: use scratch
	MOVQ R11, R8
	JMP  inv_size64_bitrev

inv_size64_use_dst:
	// Out-of-place: use dst

inv_size64_bitrev:
	// =======================================================================
	// FUSED: Bit-reversal permutation + STAGE 1 (identity twiddles)
	// =======================================================================
	// Load bit-reversed data directly into YMM registers and compute Stage 1.
	// Bitrev pattern: [0,32,16,48, 8,40,24,56, 4,36,20,52, 12,44,28,60, ...]
	// Each YMM holds 4 complex64 values (32 bytes total).
	// Stage 1 butterfly: pairs (d0,d1) and (d2,d3) → (d0+d1, d0-d1, d2+d3, d2-d3)

	// -----------------------------------------------------------------------
	// Group 0: work[0..3] = src[0,32,16,48] → Stage 1 butterfly → store at 0
	// -----------------------------------------------------------------------
	VMOVSD (R9), X0              // X0 = c0 (src[0])
	VMOVSD 256(R9), X4           // X4 = c32 (src[32], 32*8=256)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c0, c32]
	VMOVSD 128(R9), X5           // X5 = c16 (src[16], 16*8=128)
	VMOVSD 384(R9), X6           // X6 = c48 (src[48], 48*8=384)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c16, c48]
	VINSERTF128 $1, X5, Y0, Y0   // Y0 = [c0, c32, c16, c48]
	VPERMILPD $0x05, Y0, Y4      // Y4 = [c32, c0, c48, c16]
	VADDPS Y4, Y0, Y5            // Y5 = [c0+c32, c32+c0, c16+c48, c48+c16]
	VSUBPS Y0, Y4, Y6            // Y6 = [c32-c0, c0-c32, c48-c16, c16-c48]
	VBLENDPD $0x0A, Y6, Y5, Y0   // Y0 = [c0+c32, c0-c32, c16+c48, c16-c48]
	VMOVUPS Y0, (R8)

	// -----------------------------------------------------------------------
	// Group 1: work[4..7] = src[8,40,24,56] → Stage 1 butterfly → store at 32
	// -----------------------------------------------------------------------
	VMOVSD 64(R9), X0            // X0 = c8 (src[8], 8*8=64)
	VMOVSD 320(R9), X4           // X4 = c40 (src[40], 40*8=320)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c8, c40]
	VMOVSD 192(R9), X5           // X5 = c24 (src[24], 24*8=192)
	VMOVSD 448(R9), X6           // X6 = c56 (src[56], 56*8=448)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c24, c56]
	VINSERTF128 $1, X5, Y0, Y1   // Y1 = [c8, c40, c24, c56]
	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1
	VMOVUPS Y1, 32(R8)

	// -----------------------------------------------------------------------
	// Group 2: work[8..11] = src[4,36,20,52] → Stage 1 butterfly → store at 64
	// -----------------------------------------------------------------------
	VMOVSD 32(R9), X0            // X0 = c4 (src[4], 4*8=32)
	VMOVSD 288(R9), X4           // X4 = c36 (src[36], 36*8=288)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c4, c36]
	VMOVSD 160(R9), X5           // X5 = c20 (src[20], 20*8=160)
	VMOVSD 416(R9), X6           // X6 = c52 (src[52], 52*8=416)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c20, c52]
	VINSERTF128 $1, X5, Y0, Y2   // Y2 = [c4, c36, c20, c52]
	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y2, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y2
	VMOVUPS Y2, 64(R8)

	// -----------------------------------------------------------------------
	// Group 3: work[12..15] = src[12,44,28,60] → Stage 1 butterfly → store at 96
	// -----------------------------------------------------------------------
	VMOVSD 96(R9), X0            // X0 = c12 (src[12], 12*8=96)
	VMOVSD 352(R9), X4           // X4 = c44 (src[44], 44*8=352)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c12, c44]
	VMOVSD 224(R9), X5           // X5 = c28 (src[28], 28*8=224)
	VMOVSD 480(R9), X6           // X6 = c60 (src[60], 60*8=480)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c28, c60]
	VINSERTF128 $1, X5, Y0, Y3   // Y3 = [c12, c44, c28, c60]
	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y3, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y3
	VMOVUPS Y3, 96(R8)

	// -----------------------------------------------------------------------
	// Group 4: work[16..19] = src[2,34,18,50] → Stage 1 butterfly → store at 128
	// -----------------------------------------------------------------------
	VMOVSD 16(R9), X0            // X0 = c2 (src[2], 2*8=16)
	VMOVSD 272(R9), X4           // X4 = c34 (src[34], 34*8=272)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c2, c34]
	VMOVSD 144(R9), X5           // X5 = c18 (src[18], 18*8=144)
	VMOVSD 400(R9), X6           // X6 = c50 (src[50], 50*8=400)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c18, c50]
	VINSERTF128 $1, X5, Y0, Y0   // Y0 = [c2, c34, c18, c50]
	VPERMILPD $0x05, Y0, Y4
	VADDPS Y4, Y0, Y5
	VSUBPS Y0, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y0
	VMOVUPS Y0, 128(R8)

	// -----------------------------------------------------------------------
	// Group 5: work[20..23] = src[10,42,26,58] → Stage 1 butterfly → store at 160
	// -----------------------------------------------------------------------
	VMOVSD 80(R9), X0            // X0 = c10 (src[10], 10*8=80)
	VMOVSD 336(R9), X4           // X4 = c42 (src[42], 42*8=336)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c10, c42]
	VMOVSD 208(R9), X5           // X5 = c26 (src[26], 26*8=208)
	VMOVSD 464(R9), X6           // X6 = c58 (src[58], 58*8=464)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c26, c58]
	VINSERTF128 $1, X5, Y0, Y1   // Y1 = [c10, c42, c26, c58]
	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1
	VMOVUPS Y1, 160(R8)

	// -----------------------------------------------------------------------
	// Group 6: work[24..27] = src[6,38,22,54] → Stage 1 butterfly → store at 192
	// -----------------------------------------------------------------------
	VMOVSD 48(R9), X0            // X0 = c6 (src[6], 6*8=48)
	VMOVSD 304(R9), X4           // X4 = c38 (src[38], 38*8=304)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c6, c38]
	VMOVSD 176(R9), X5           // X5 = c22 (src[22], 22*8=176)
	VMOVSD 432(R9), X6           // X6 = c54 (src[54], 54*8=432)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c22, c54]
	VINSERTF128 $1, X5, Y0, Y2   // Y2 = [c6, c38, c22, c54]
	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y2, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y2
	VMOVUPS Y2, 192(R8)

	// -----------------------------------------------------------------------
	// Group 7: work[28..31] = src[14,46,30,62] → Stage 1 butterfly → store at 224
	// -----------------------------------------------------------------------
	VMOVSD 112(R9), X0           // X0 = c14 (src[14], 14*8=112)
	VMOVSD 368(R9), X4           // X4 = c46 (src[46], 46*8=368)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c14, c46]
	VMOVSD 240(R9), X5           // X5 = c30 (src[30], 30*8=240)
	VMOVSD 496(R9), X6           // X6 = c62 (src[62], 62*8=496)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c30, c62]
	VINSERTF128 $1, X5, Y0, Y3   // Y3 = [c14, c46, c30, c62]
	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y3, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y3
	VMOVUPS Y3, 224(R8)

	// -----------------------------------------------------------------------
	// Group 8: work[32..35] = src[1,33,17,49] → Stage 1 butterfly → store at 256
	// -----------------------------------------------------------------------
	VMOVSD 8(R9), X0             // X0 = c1 (src[1], 1*8=8)
	VMOVSD 264(R9), X4           // X4 = c33 (src[33], 33*8=264)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c1, c33]
	VMOVSD 136(R9), X5           // X5 = c17 (src[17], 17*8=136)
	VMOVSD 392(R9), X6           // X6 = c49 (src[49], 49*8=392)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c17, c49]
	VINSERTF128 $1, X5, Y0, Y0   // Y0 = [c1, c33, c17, c49]
	VPERMILPD $0x05, Y0, Y4
	VADDPS Y4, Y0, Y5
	VSUBPS Y0, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y0
	VMOVUPS Y0, 256(R8)

	// -----------------------------------------------------------------------
	// Group 9: work[36..39] = src[9,41,25,57] → Stage 1 butterfly → store at 288
	// -----------------------------------------------------------------------
	VMOVSD 72(R9), X0            // X0 = c9 (src[9], 9*8=72)
	VMOVSD 328(R9), X4           // X4 = c41 (src[41], 41*8=328)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c9, c41]
	VMOVSD 200(R9), X5           // X5 = c25 (src[25], 25*8=200)
	VMOVSD 456(R9), X6           // X6 = c57 (src[57], 57*8=456)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c25, c57]
	VINSERTF128 $1, X5, Y0, Y1   // Y1 = [c9, c41, c25, c57]
	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1
	VMOVUPS Y1, 288(R8)

	// -----------------------------------------------------------------------
	// Group 10: work[40..43] = src[5,37,21,53] → Stage 1 butterfly → store at 320
	// -----------------------------------------------------------------------
	VMOVSD 40(R9), X0            // X0 = c5 (src[5], 5*8=40)
	VMOVSD 296(R9), X4           // X4 = c37 (src[37], 37*8=296)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c5, c37]
	VMOVSD 168(R9), X5           // X5 = c21 (src[21], 21*8=168)
	VMOVSD 424(R9), X6           // X6 = c53 (src[53], 53*8=424)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c21, c53]
	VINSERTF128 $1, X5, Y0, Y2   // Y2 = [c5, c37, c21, c53]
	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y2, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y2
	VMOVUPS Y2, 320(R8)

	// -----------------------------------------------------------------------
	// Group 11: work[44..47] = src[13,45,29,61] → Stage 1 butterfly → store at 352
	// -----------------------------------------------------------------------
	VMOVSD 104(R9), X0           // X0 = c13 (src[13], 13*8=104)
	VMOVSD 360(R9), X4           // X4 = c45 (src[45], 45*8=360)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c13, c45]
	VMOVSD 232(R9), X5           // X5 = c29 (src[29], 29*8=232)
	VMOVSD 488(R9), X6           // X6 = c61 (src[61], 61*8=488)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c29, c61]
	VINSERTF128 $1, X5, Y0, Y3   // Y3 = [c13, c45, c29, c61]
	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y3, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y3
	VMOVUPS Y3, 352(R8)

	// -----------------------------------------------------------------------
	// Group 12: work[48..51] = src[3,35,19,51] → Stage 1 butterfly → store at 384
	// -----------------------------------------------------------------------
	VMOVSD 24(R9), X0            // X0 = c3 (src[3], 3*8=24)
	VMOVSD 280(R9), X4           // X4 = c35 (src[35], 35*8=280)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c3, c35]
	VMOVSD 152(R9), X5           // X5 = c19 (src[19], 19*8=152)
	VMOVSD 408(R9), X6           // X6 = c51 (src[51], 51*8=408)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c19, c51]
	VINSERTF128 $1, X5, Y0, Y0   // Y0 = [c3, c35, c19, c51]
	VPERMILPD $0x05, Y0, Y4
	VADDPS Y4, Y0, Y5
	VSUBPS Y0, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y0
	VMOVUPS Y0, 384(R8)

	// -----------------------------------------------------------------------
	// Group 13: work[52..55] = src[11,43,27,59] → Stage 1 butterfly → store at 416
	// -----------------------------------------------------------------------
	VMOVSD 88(R9), X0            // X0 = c11 (src[11], 11*8=88)
	VMOVSD 344(R9), X4           // X4 = c43 (src[43], 43*8=344)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c11, c43]
	VMOVSD 216(R9), X5           // X5 = c27 (src[27], 27*8=216)
	VMOVSD 472(R9), X6           // X6 = c59 (src[59], 59*8=472)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c27, c59]
	VINSERTF128 $1, X5, Y0, Y1   // Y1 = [c11, c43, c27, c59]
	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1
	VMOVUPS Y1, 416(R8)

	// -----------------------------------------------------------------------
	// Group 14: work[56..59] = src[7,39,23,55] → Stage 1 butterfly → store at 448
	// -----------------------------------------------------------------------
	VMOVSD 56(R9), X0            // X0 = c7 (src[7], 7*8=56)
	VMOVSD 312(R9), X4           // X4 = c39 (src[39], 39*8=312)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c7, c39]
	VMOVSD 184(R9), X5           // X5 = c23 (src[23], 23*8=184)
	VMOVSD 440(R9), X6           // X6 = c55 (src[55], 55*8=440)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c23, c55]
	VINSERTF128 $1, X5, Y0, Y2   // Y2 = [c7, c39, c23, c55]
	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y2, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y2
	VMOVUPS Y2, 448(R8)

	// -----------------------------------------------------------------------
	// Group 15: work[60..63] = src[15,47,31,63] → Stage 1 butterfly → store at 480
	// -----------------------------------------------------------------------
	VMOVSD 120(R9), X0           // X0 = c15 (src[15], 15*8=120)
	VMOVSD 376(R9), X4           // X4 = c47 (src[47], 47*8=376)
	VPUNPCKLQDQ X4, X0, X0       // X0 = [c15, c47]
	VMOVSD 248(R9), X5           // X5 = c31 (src[31], 31*8=248)
	VMOVSD 504(R9), X6           // X6 = c63 (src[63], 63*8=504)
	VPUNPCKLQDQ X6, X5, X5       // X5 = [c31, c63]
	VINSERTF128 $1, X5, Y0, Y3   // Y3 = [c15, c47, c31, c63]
	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y3, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y3
	VMOVUPS Y3, 480(R8)

	// =======================================================================
	// STAGE 2: size=4, half=2, step=16
	// =======================================================================
	// Twiddle factors: twiddle[0], twiddle[16]
	// twiddle[0] = (1, 0), twiddle[16] = (0, -1) for n=64

	VMOVSD (R10), X8           // twiddle[0]
	VMOVSD 128(R10), X9        // twiddle[16] (16 * 8 bytes)
	VPUNPCKLQDQ X9, X8, X8     // X8 = [tw0, tw16]
	VINSERTF128 $1, X8, Y8, Y8 // Y8 = [tw0, tw16, tw0, tw16]

	// Process all YMM registers with stage 2 butterflies
	// Each YMM has 4 complex values [d0, d1, d2, d3]
	// Butterfly pairs: (d0, d2) and (d1, d3)

	// Helper macro pattern for stage 2 (process one YMM at offset)
	// Load, extract halves, multiply, butterfly, store

	// Indices 0-3
	VMOVUPS (R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1    // Y1 = [d0, d1, d0, d1]
	VPERM2F128 $0x11, Y0, Y0, Y2    // Y2 = [d2, d3, d2, d3]
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, (R8)

	// Indices 4-7
	VMOVUPS 32(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 32(R8)

	// Indices 8-11
	VMOVUPS 64(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 64(R8)

	// Indices 12-15
	VMOVUPS 96(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 96(R8)

	// Indices 16-19
	VMOVUPS 128(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 128(R8)

	// Indices 20-23
	VMOVUPS 160(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 160(R8)

	// Indices 24-27
	VMOVUPS 192(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 192(R8)

	// Indices 28-31
	VMOVUPS 224(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 224(R8)

	// Indices 32-35
	VMOVUPS 256(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 256(R8)

	// Indices 36-39
	VMOVUPS 288(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 288(R8)

	// Indices 40-43
	VMOVUPS 320(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 320(R8)

	// Indices 44-47
	VMOVUPS 352(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 352(R8)

	// Indices 48-51
	VMOVUPS 384(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 384(R8)

	// Indices 52-55
	VMOVUPS 416(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 416(R8)

	// Indices 56-59
	VMOVUPS 448(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 448(R8)

	// Indices 60-63
	VMOVUPS 480(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 480(R8)

	// =======================================================================
	// STAGE 3: size=8, half=4, step=8
	// =======================================================================
	// Pairs: indices 0-3 with 4-7, 8-11 with 12-15, etc.
	// Twiddle factors: twiddle[0], twiddle[8], twiddle[16], twiddle[24]

	VMOVSD (R10), X8           // twiddle[0]
	VMOVSD 64(R10), X9         // twiddle[8]
	VPUNPCKLQDQ X9, X8, X8     // X8 = [tw0, tw8]
	VMOVSD 128(R10), X9        // twiddle[16]
	VMOVSD 192(R10), X10       // twiddle[24]
	VPUNPCKLQDQ X10, X9, X9    // X9 = [tw16, tw24]
	VINSERTF128 $1, X9, Y8, Y8 // Y8 = [tw0, tw8, tw16, tw24]

	// Extract twiddle components once
	VMOVSLDUP Y8, Y14          // Y14 = [w.r, w.r, ...]
	VMOVSHDUP Y8, Y15          // Y15 = [w.i, w.i, ...]

	// Group: indices 0-3 with 4-7
	VMOVUPS (R8), Y0           // a = indices 0-3
	VMOVUPS 32(R8), Y1         // b = indices 4-7
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2 // Y2 = t = w * b
	VADDPS Y2, Y0, Y3          // Y3 = a + t
	VSUBPS Y2, Y0, Y4          // Y4 = a - t
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 32(R8)

	// Group: indices 8-11 with 12-15
	VMOVUPS 64(R8), Y0
	VMOVUPS 96(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 64(R8)
	VMOVUPS Y4, 96(R8)

	// Group: indices 16-19 with 20-23
	VMOVUPS 128(R8), Y0
	VMOVUPS 160(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 128(R8)
	VMOVUPS Y4, 160(R8)

	// Group: indices 24-27 with 28-31
	VMOVUPS 192(R8), Y0
	VMOVUPS 224(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 192(R8)
	VMOVUPS Y4, 224(R8)

	// Group: indices 32-35 with 36-39
	VMOVUPS 256(R8), Y0
	VMOVUPS 288(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y4, 288(R8)

	// Group: indices 40-43 with 44-47
	VMOVUPS 320(R8), Y0
	VMOVUPS 352(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 320(R8)
	VMOVUPS Y4, 352(R8)

	// Group: indices 48-51 with 52-55
	VMOVUPS 384(R8), Y0
	VMOVUPS 416(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 384(R8)
	VMOVUPS Y4, 416(R8)

	// Group: indices 56-59 with 60-63
	VMOVUPS 448(R8), Y0
	VMOVUPS 480(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 448(R8)
	VMOVUPS Y4, 480(R8)

	// =======================================================================
	// STAGE 4: size=16, half=8, step=4
	// =======================================================================
	// Pairs: indices 0-7 with 8-15, 16-23 with 24-31, 32-39 with 40-47, 48-55 with 56-63
	// Twiddle factors: j=0..3 use [0,4,8,12], j=4..7 use [16,20,24,28]

	VMOVSD (R10), X8           // twiddle[0]
	VMOVSD 32(R10), X9         // twiddle[4]
	VPUNPCKLQDQ X9, X8, X8     // X8 = [tw0, tw4]
	VMOVSD 64(R10), X9         // twiddle[8]
	VMOVSD 96(R10), X10        // twiddle[12]
	VPUNPCKLQDQ X10, X9, X9    // X9 = [tw8, tw12]
	VINSERTF128 $1, X9, Y8, Y8 // Y8 = [tw0, tw4, tw8, tw12]

	VMOVSD 128(R10), X9        // twiddle[16]
	VMOVSD 160(R10), X10       // twiddle[20]
	VPUNPCKLQDQ X10, X9, X9    // X9 = [tw16, tw20]
	VMOVSD 192(R10), X10       // twiddle[24]
	VMOVSD 224(R10), X11       // twiddle[28]
	VPUNPCKLQDQ X11, X10, X10  // X10 = [tw24, tw28]
	VINSERTF128 $1, X10, Y9, Y9 // Y9 = [tw16, tw20, tw24, tw28]

	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15

	// Group 1: indices 0-3 with 8-11 (first half of 16-point group)
	VMOVUPS (R8), Y0
	VMOVUPS 64(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 64(R8)

	// Group 1: indices 4-7 with 12-15
	VMOVUPS 32(R8), Y0
	VMOVUPS 96(R8), Y1
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 32(R8)
	VMOVUPS Y4, 96(R8)

	// Group 2: indices 16-19 with 24-27
	VMOVUPS 128(R8), Y0
	VMOVUPS 192(R8), Y1
	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 128(R8)
	VMOVUPS Y4, 192(R8)

	// Group 2: indices 20-23 with 28-31
	VMOVUPS 160(R8), Y0
	VMOVUPS 224(R8), Y1
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 160(R8)
	VMOVUPS Y4, 224(R8)

	// Group 3: indices 32-35 with 40-43
	VMOVUPS 256(R8), Y0
	VMOVUPS 320(R8), Y1
	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y4, 320(R8)

	// Group 3: indices 36-39 with 44-47
	VMOVUPS 288(R8), Y0
	VMOVUPS 352(R8), Y1
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 288(R8)
	VMOVUPS Y4, 352(R8)

	// Group 4: indices 48-51 with 56-59
	VMOVUPS 384(R8), Y0
	VMOVUPS 448(R8), Y1
	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 384(R8)
	VMOVUPS Y4, 448(R8)

	// Group 4: indices 52-55 with 60-63
	VMOVUPS 416(R8), Y0
	VMOVUPS 480(R8), Y1
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 416(R8)
	VMOVUPS Y4, 480(R8)

	// =======================================================================
	// STAGE 5: size=32, half=16, step=2
	// =======================================================================
	// Pairs: indices 0-15 with 16-31, 32-47 with 48-63
	// Twiddle factors: j=0..3 -> [0,2,4,6], j=4..7 -> [8,10,12,14],
	//                   j=8..11 -> [16,18,20,22], j=12..15 -> [24,26,28,30]

	// Load twiddles for indices 0-3: tw[0,2,4,6]
	VMOVSD (R10), X8           // twiddle[0]
	VMOVSD 16(R10), X9         // twiddle[2]
	VPUNPCKLQDQ X9, X8, X8
	VMOVSD 32(R10), X9         // twiddle[4]
	VMOVSD 48(R10), X10        // twiddle[6]
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8 // Y8 = [tw0, tw2, tw4, tw6]

	// Load twiddles for indices 4-7: tw[8,10,12,14]
	VMOVSD 64(R10), X9         // twiddle[8]
	VMOVSD 80(R10), X10        // twiddle[10]
	VPUNPCKLQDQ X10, X9, X9
	VMOVSD 96(R10), X10        // twiddle[12]
	VMOVSD 112(R10), X11       // twiddle[14]
	VPUNPCKLQDQ X11, X10, X10
	VINSERTF128 $1, X10, Y9, Y9 // Y9 = [tw8, tw10, tw12, tw14]

	// Load twiddles for indices 8-11: tw[16,18,20,22]
	VMOVSD 128(R10), X10       // twiddle[16]
	VMOVSD 144(R10), X11       // twiddle[18]
	VPUNPCKLQDQ X11, X10, X10
	VMOVSD 160(R10), X11       // twiddle[20]
	VMOVSD 176(R10), X12       // twiddle[22]
	VPUNPCKLQDQ X12, X11, X11
	VINSERTF128 $1, X11, Y10, Y10 // Y10 = [tw16, tw18, tw20, tw22]

	// Load twiddles for indices 12-15: tw[24,26,28,30]
	VMOVSD 192(R10), X11       // twiddle[24]
	VMOVSD 208(R10), X12       // twiddle[26]
	VPUNPCKLQDQ X12, X11, X11
	VMOVSD 224(R10), X12       // twiddle[28]
	VMOVSD 240(R10), X13       // twiddle[30]
	VPUNPCKLQDQ X13, X12, X12
	VINSERTF128 $1, X12, Y11, Y11 // Y11 = [tw24, tw26, tw28, tw30]

	// Group 1: indices 0-3 with 16-19
	VMOVUPS (R8), Y0
	VMOVUPS 128(R8), Y1
	VMOVSLDUP Y8, Y12
	VMOVSHDUP Y8, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 128(R8)

	// Group 1: indices 4-7 with 20-23
	VMOVUPS 32(R8), Y0
	VMOVUPS 160(R8), Y1
	VMOVSLDUP Y9, Y12
	VMOVSHDUP Y9, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 32(R8)
	VMOVUPS Y4, 160(R8)

	// Group 1: indices 8-11 with 24-27
	VMOVUPS 64(R8), Y0
	VMOVUPS 192(R8), Y1
	VMOVSLDUP Y10, Y12
	VMOVSHDUP Y10, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 64(R8)
	VMOVUPS Y4, 192(R8)

	// Group 1: indices 12-15 with 28-31
	VMOVUPS 96(R8), Y0
	VMOVUPS 224(R8), Y1
	VMOVSLDUP Y11, Y12
	VMOVSHDUP Y11, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 96(R8)
	VMOVUPS Y4, 224(R8)

	// Group 2: indices 32-35 with 48-51
	VMOVUPS 256(R8), Y0
	VMOVUPS 384(R8), Y1
	VMOVSLDUP Y8, Y12
	VMOVSHDUP Y8, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y4, 384(R8)

	// Group 2: indices 36-39 with 52-55
	VMOVUPS 288(R8), Y0
	VMOVUPS 416(R8), Y1
	VMOVSLDUP Y9, Y12
	VMOVSHDUP Y9, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 288(R8)
	VMOVUPS Y4, 416(R8)

	// Group 2: indices 40-43 with 56-59
	VMOVUPS 320(R8), Y0
	VMOVUPS 448(R8), Y1
	VMOVSLDUP Y10, Y12
	VMOVSHDUP Y10, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 320(R8)
	VMOVUPS Y4, 448(R8)

	// Group 2: indices 44-47 with 60-63
	VMOVUPS 352(R8), Y0
	VMOVUPS 480(R8), Y1
	VMOVSLDUP Y11, Y12
	VMOVSHDUP Y11, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 352(R8)
	VMOVUPS Y4, 480(R8)

	// =======================================================================
	// STAGE 6: size=64, half=32, step=1
	// =======================================================================
	// Pairs: indices 0-31 with 32-63
	// Twiddle factors: twiddle[0,1,2,...,31]

	// Load twiddles for indices 0-3
	VMOVUPS (R10), Y8          // Y8 = [tw0, tw1, tw2, tw3]
	VMOVUPS 32(R10), Y9        // Y9 = [tw4, tw5, tw6, tw7]

	// Group: indices 0-3 with 32-35
	VMOVUPS (R8), Y0
	VMOVUPS 256(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 256(R8)

	// Group: indices 4-7 with 36-39
	VMOVUPS 32(R8), Y0
	VMOVUPS 288(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 32(R8)
	VMOVUPS Y4, 288(R8)

	// Load twiddles for indices 8-15
	VMOVUPS 64(R10), Y8        // Y8 = [tw8, tw9, tw10, tw11]
	VMOVUPS 96(R10), Y9        // Y9 = [tw12, tw13, tw14, tw15]

	// Group: indices 8-11 with 40-43
	VMOVUPS 64(R8), Y0
	VMOVUPS 320(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 64(R8)
	VMOVUPS Y4, 320(R8)

	// Group: indices 12-15 with 44-47
	VMOVUPS 96(R8), Y0
	VMOVUPS 352(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 96(R8)
	VMOVUPS Y4, 352(R8)

	// Load twiddles for indices 16-23
	VMOVUPS 128(R10), Y8       // Y8 = [tw16, tw17, tw18, tw19]
	VMOVUPS 160(R10), Y9       // Y9 = [tw20, tw21, tw22, tw23]

	// Group: indices 16-19 with 48-51
	VMOVUPS 128(R8), Y0
	VMOVUPS 384(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 128(R8)
	VMOVUPS Y4, 384(R8)

	// Group: indices 20-23 with 52-55
	VMOVUPS 160(R8), Y0
	VMOVUPS 416(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 160(R8)
	VMOVUPS Y4, 416(R8)

	// Load twiddles for indices 24-31
	VMOVUPS 192(R10), Y8       // Y8 = [tw24, tw25, tw26, tw27]
	VMOVUPS 224(R10), Y9       // Y9 = [tw28, tw29, tw30, tw31]

	// Group: indices 24-27 with 56-59
	VMOVUPS 192(R8), Y0
	VMOVUPS 448(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 192(R8)
	VMOVUPS Y4, 448(R8)

	// Group: indices 28-31 with 60-63
	VMOVUPS 224(R8), Y0
	VMOVUPS 480(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 224(R8)
	VMOVUPS Y4, 480(R8)

	// =======================================================================
	// Apply 1/N scaling for inverse transform (1/64)
	// =======================================================================
	MOVL ·sixtyFourth32(SB), AX
	MOVD AX, X8
	VBROADCASTSS X8, Y8

	XORQ CX, CX
inv_size64_scale_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMULPS Y8, Y0, Y0
	VMOVUPS Y0, (R8)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $512
	JL   inv_size64_scale_loop

	// =======================================================================
	// Copy results to dst if we used scratch buffer
	// =======================================================================
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   inv_size64_done

	// Copy from scratch to dst (512 bytes = 16 YMM registers)
	VMOVUPS (R8), Y0
	VMOVUPS 32(R8), Y1
	VMOVUPS 64(R8), Y2
	VMOVUPS 96(R8), Y3
	VMOVUPS 128(R8), Y4
	VMOVUPS 160(R8), Y5
	VMOVUPS 192(R8), Y6
	VMOVUPS 224(R8), Y7
	VMOVUPS Y0, (R9)
	VMOVUPS Y1, 32(R9)
	VMOVUPS Y2, 64(R9)
	VMOVUPS Y3, 96(R9)
	VMOVUPS Y4, 128(R9)
	VMOVUPS Y5, 160(R9)
	VMOVUPS Y6, 192(R9)
	VMOVUPS Y7, 224(R9)

	VMOVUPS 256(R8), Y0
	VMOVUPS 288(R8), Y1
	VMOVUPS 320(R8), Y2
	VMOVUPS 352(R8), Y3
	VMOVUPS 384(R8), Y4
	VMOVUPS 416(R8), Y5
	VMOVUPS 448(R8), Y6
	VMOVUPS 480(R8), Y7
	VMOVUPS Y0, 256(R9)
	VMOVUPS Y1, 288(R9)
	VMOVUPS Y2, 320(R9)
	VMOVUPS Y3, 352(R9)
	VMOVUPS Y4, 384(R9)
	VMOVUPS Y5, 416(R9)
	VMOVUPS Y6, 448(R9)
	VMOVUPS Y7, 480(R9)

inv_size64_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

inv_size64_return_false:
	MOVB $0, ret+96(FP)
	RET

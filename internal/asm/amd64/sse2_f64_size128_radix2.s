//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-128 Radix-2 FFT Kernels for AMD64 (complex128)
// ===========================================================================

#include "textflag.h"

// Forward transform, size 128, complex128, radix-2
TEXT ·ForwardSSE2Size128Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	CMPQ R13, $128
	JNE  fwd_err

	CMPQ R8, R9
	JNE  fwd_use_dst
	MOVQ R11, R8

fwd_use_dst:
	// -----------------------------------------------------------------------
	// FUSED: Bit-reversal permutation + Stage 1 (identity twiddles)
	// Unrolled even half, executed twice (second pass uses +16 src, +1024 dst).
	// -----------------------------------------------------------------------
	MOVQ R8, R14 // save work base
	MOVQ R9, R15 // save src base
	XORQ BX, BX  // pass counter (even/odd)

fwd_stage1_pass:
	// (0,64) -> work[0], work[1]
	MOVUPD 0(R9), X0         // src[0]
	MOVUPD 1024(R9), X1      // src[64]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 0(R8)         // work[0]
	MOVUPD X3, 16(R8)        // work[1]

	// (32,96) -> work[2], work[3]
	MOVUPD 512(R9), X0       // src[32]
	MOVUPD 1536(R9), X1      // src[96]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 32(R8)        // work[2]
	MOVUPD X3, 48(R8)        // work[3]

	// (16,80) -> work[4], work[5]
	MOVUPD 256(R9), X0       // src[16]
	MOVUPD 1280(R9), X1      // src[80]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 64(R8)        // work[4]
	MOVUPD X3, 80(R8)        // work[5]

	// (48,112) -> work[6], work[7]
	MOVUPD 768(R9), X0       // src[48]
	MOVUPD 1792(R9), X1      // src[112]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 96(R8)        // work[6]
	MOVUPD X3, 112(R8)       // work[7]

	// (8,72) -> work[8], work[9]
	MOVUPD 128(R9), X0       // src[8]
	MOVUPD 1152(R9), X1      // src[72]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 128(R8)       // work[8]
	MOVUPD X3, 144(R8)       // work[9]

	// (40,104) -> work[10], work[11]
	MOVUPD 640(R9), X0       // src[40]
	MOVUPD 1664(R9), X1      // src[104]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 160(R8)       // work[10]
	MOVUPD X3, 176(R8)       // work[11]

	// (24,88) -> work[12], work[13]
	MOVUPD 384(R9), X0       // src[24]
	MOVUPD 1408(R9), X1      // src[88]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 192(R8)       // work[12]
	MOVUPD X3, 208(R8)       // work[13]

	// (56,120) -> work[14], work[15]
	MOVUPD 896(R9), X0       // src[56]
	MOVUPD 1920(R9), X1      // src[120]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 224(R8)       // work[14]
	MOVUPD X3, 240(R8)       // work[15]

	// (4,68) -> work[16], work[17]
	MOVUPD 64(R9), X0        // src[4]
	MOVUPD 1088(R9), X1      // src[68]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 256(R8)       // work[16]
	MOVUPD X3, 272(R8)       // work[17]

	// (36,100) -> work[18], work[19]
	MOVUPD 576(R9), X0       // src[36]
	MOVUPD 1600(R9), X1      // src[100]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 288(R8)       // work[18]
	MOVUPD X3, 304(R8)       // work[19]

	// (20,84) -> work[20], work[21]
	MOVUPD 320(R9), X0       // src[20]
	MOVUPD 1344(R9), X1      // src[84]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 320(R8)       // work[20]
	MOVUPD X3, 336(R8)       // work[21]

	// (52,116) -> work[22], work[23]
	MOVUPD 832(R9), X0       // src[52]
	MOVUPD 1856(R9), X1      // src[116]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 352(R8)       // work[22]
	MOVUPD X3, 368(R8)       // work[23]

	// (12,76) -> work[24], work[25]
	MOVUPD 192(R9), X0       // src[12]
	MOVUPD 1216(R9), X1      // src[76]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 384(R8)       // work[24]
	MOVUPD X3, 400(R8)       // work[25]

	// (44,108) -> work[26], work[27]
	MOVUPD 704(R9), X0       // src[44]
	MOVUPD 1728(R9), X1      // src[108]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 416(R8)       // work[26]
	MOVUPD X3, 432(R8)       // work[27]

	// (28,92) -> work[28], work[29]
	MOVUPD 448(R9), X0       // src[28]
	MOVUPD 1472(R9), X1      // src[92]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 448(R8)       // work[28]
	MOVUPD X3, 464(R8)       // work[29]

	// (60,124) -> work[30], work[31]
	MOVUPD 960(R9), X0       // src[60]
	MOVUPD 1984(R9), X1      // src[124]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 480(R8)       // work[30]
	MOVUPD X3, 496(R8)       // work[31]

	// (2,66) -> work[32], work[33]
	MOVUPD 32(R9), X0        // src[2]
	MOVUPD 1056(R9), X1      // src[66]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 512(R8)       // work[32]
	MOVUPD X3, 528(R8)       // work[33]

	// (34,98) -> work[34], work[35]
	MOVUPD 544(R9), X0       // src[34]
	MOVUPD 1568(R9), X1      // src[98]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 544(R8)       // work[34]
	MOVUPD X3, 560(R8)       // work[35]

	// (18,82) -> work[36], work[37]
	MOVUPD 288(R9), X0       // src[18]
	MOVUPD 1312(R9), X1      // src[82]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 576(R8)       // work[36]
	MOVUPD X3, 592(R8)       // work[37]

	// (50,114) -> work[38], work[39]
	MOVUPD 800(R9), X0       // src[50]
	MOVUPD 1824(R9), X1      // src[114]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 608(R8)       // work[38]
	MOVUPD X3, 624(R8)       // work[39]

	// (10,74) -> work[40], work[41]
	MOVUPD 160(R9), X0       // src[10]
	MOVUPD 1184(R9), X1      // src[74]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 640(R8)       // work[40]
	MOVUPD X3, 656(R8)       // work[41]

	// (42,106) -> work[42], work[43]
	MOVUPD 672(R9), X0       // src[42]
	MOVUPD 1696(R9), X1      // src[106]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 672(R8)       // work[42]
	MOVUPD X3, 688(R8)       // work[43]

	// (26,90) -> work[44], work[45]
	MOVUPD 416(R9), X0       // src[26]
	MOVUPD 1440(R9), X1      // src[90]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 704(R8)       // work[44]
	MOVUPD X3, 720(R8)       // work[45]

	// (58,122) -> work[46], work[47]
	MOVUPD 928(R9), X0       // src[58]
	MOVUPD 1952(R9), X1      // src[122]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 736(R8)       // work[46]
	MOVUPD X3, 752(R8)       // work[47]

	// (6,70) -> work[48], work[49]
	MOVUPD 96(R9), X0        // src[6]
	MOVUPD 1120(R9), X1      // src[70]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 768(R8)       // work[48]
	MOVUPD X3, 784(R8)       // work[49]

	// (38,102) -> work[50], work[51]
	MOVUPD 608(R9), X0       // src[38]
	MOVUPD 1632(R9), X1      // src[102]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 800(R8)       // work[50]
	MOVUPD X3, 816(R8)       // work[51]

	// (22,86) -> work[52], work[53]
	MOVUPD 352(R9), X0       // src[22]
	MOVUPD 1376(R9), X1      // src[86]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 832(R8)       // work[52]
	MOVUPD X3, 848(R8)       // work[53]

	// (54,118) -> work[54], work[55]
	MOVUPD 864(R9), X0       // src[54]
	MOVUPD 1888(R9), X1      // src[118]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 864(R8)       // work[54]
	MOVUPD X3, 880(R8)       // work[55]

	// (14,78) -> work[56], work[57]
	MOVUPD 224(R9), X0       // src[14]
	MOVUPD 1248(R9), X1      // src[78]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 896(R8)       // work[56]
	MOVUPD X3, 912(R8)       // work[57]

	// (46,110) -> work[58], work[59]
	MOVUPD 736(R9), X0       // src[46]
	MOVUPD 1760(R9), X1      // src[110]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 928(R8)       // work[58]
	MOVUPD X3, 944(R8)       // work[59]

	// (30,94) -> work[60], work[61]
	MOVUPD 480(R9), X0       // src[30]
	MOVUPD 1504(R9), X1      // src[94]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 960(R8)       // work[60]
	MOVUPD X3, 976(R8)       // work[61]

	// (62,126) -> work[62], work[63]
	MOVUPD 992(R9), X0       // src[62]
	MOVUPD 2016(R9), X1      // src[126]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 992(R8)       // work[62]
	MOVUPD X3, 1008(R8)      // work[63]

	INCQ BX                  // next pass
	CMPQ BX, $2              // done after odd pass
	JGE  fwd_stage1_done
	LEAQ 1024(R14), R8       // work offset for odd half
	LEAQ 16(R15), R9         // src offset for odd half
	JMP  fwd_stage1_pass

fwd_stage1_done:
	MOVQ R14, R8             // restore work base

	// Stage 2: dist 2 - 32 blocks of 4
	MOVQ R8, SI              // work base
	MOVQ $32, CX             // blocks
	MOVUPS ·maskNegLoPD(SB), X14
fwd_stage2_loop:
	MOVQ $2, DX              // half=2
fwd_stage2_inner:
	MOVUPD (SI), X0          // a
  MOVUPD 32(SI), X1        // b
	MOVQ $2, AX              // k = 2 - DX
  SUBQ DX, AX              // k (0..1)
  SHLQ $5, AX              // k * 32
  SHLQ $4, AX              // k * 32 * 16
  MOVUPD (R10)(AX*1), X10  // twiddle[k*32]
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
	MOVQ R8, SI
  MOVQ $16, CX
fwd_s3_loop:
	MOVQ $4, DX
fwd_s3_inner:
	MOVUPD (SI), X0
  MOVUPD 64(SI), X1
	MOVQ $4, AX
  SUBQ DX, AX
  SHLQ $4, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10 // k * 128/8 * 16 = k * 16 * 16
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
	MOVAPD X0, X3
  ADDPD X2, X0
  SUBPD X2, X3
	MOVUPD X0, (SI)
  MOVUPD X3, 64(SI)
	ADDQ $16, SI
  DECQ DX
  JNZ fwd_s3_inner
	ADDQ $64, SI
  DECQ CX
  JNZ fwd_s3_loop

	// Stage 4: dist 8
	MOVQ R8, SI
  MOVQ $8, CX
fwd_s4_loop:
	MOVQ $8, DX
fwd_s4_inner:
	MOVUPD (SI), X0
  MOVUPD 128(SI), X1
	MOVQ $8, AX
  SUBQ DX, AX
  SHLQ $3, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10 // k * 128/16 * 16 = k * 8 * 16
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
  MOVQ $4, CX
fwd_s5_loop:
	MOVQ $16, DX
fwd_s5_inner:
	MOVUPD (SI), X0
  MOVUPD 256(SI), X1
	MOVQ $16, AX
  SUBQ DX, AX
  SHLQ $2, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10 // k * 128/32 * 16 = k * 4 * 16
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
  MOVQ $2, CX
fwd_s6_loop:
	MOVQ $32, DX
fwd_s6_inner:
	MOVUPD (SI), X0
  MOVUPD 512(SI), X1
	MOVQ $32, AX
  SUBQ DX, AX
  SHLQ $1, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10 // k * 128/64 * 16 = k * 2 * 16
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
  MOVQ $64, DX
fwd_s7_inner:
	MOVUPD (SI), X0
  MOVUPD 1024(SI), X1
	MOVQ $64, AX
  SUBQ DX, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10 // k * 128/128 * 16 = k * 1 * 16
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
	MOVAPD X0, X3
  ADDPD X2, X0
  SUBPD X2, X3
	MOVUPD X0, (SI)
  MOVUPD X3, 1024(SI)
	ADDQ $16, SI
  DECQ DX
  JNZ fwd_s7_inner

	// Copy to dst
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   fwd_done
	MOVQ $64, CX
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

// Inverse transform, size 128, complex128, radix-2
TEXT ·InverseSSE2Size128Radix2Complex128Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	CMPQ R13, $128
	JNE  inv_err

	CMPQ R8, R9
	JNE  inv_use_dst
	MOVQ R11, R8

inv_use_dst:
	// -----------------------------------------------------------------------
	// FUSED: Bit-reversal permutation + Stage 1 (identity twiddles)
	// Unrolled even half, executed twice (second pass uses +16 src, +1024 dst).
	// -----------------------------------------------------------------------
	MOVQ R8, R14 // save work base
	MOVQ R9, R15 // save src base
	XORQ BX, BX  // pass counter (even/odd)

inv_stage1_pass:
	// (0,64) -> work[0], work[1]
	MOVUPD 0(R9), X0         // src[0]
	MOVUPD 1024(R9), X1      // src[64]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 0(R8)         // work[0]
	MOVUPD X3, 16(R8)        // work[1]

	// (32,96) -> work[2], work[3]
	MOVUPD 512(R9), X0       // src[32]
	MOVUPD 1536(R9), X1      // src[96]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 32(R8)        // work[2]
	MOVUPD X3, 48(R8)        // work[3]

	// (16,80) -> work[4], work[5]
	MOVUPD 256(R9), X0       // src[16]
	MOVUPD 1280(R9), X1      // src[80]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 64(R8)        // work[4]
	MOVUPD X3, 80(R8)        // work[5]

	// (48,112) -> work[6], work[7]
	MOVUPD 768(R9), X0       // src[48]
	MOVUPD 1792(R9), X1      // src[112]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 96(R8)        // work[6]
	MOVUPD X3, 112(R8)       // work[7]

	// (8,72) -> work[8], work[9]
	MOVUPD 128(R9), X0       // src[8]
	MOVUPD 1152(R9), X1      // src[72]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 128(R8)       // work[8]
	MOVUPD X3, 144(R8)       // work[9]

	// (40,104) -> work[10], work[11]
	MOVUPD 640(R9), X0       // src[40]
	MOVUPD 1664(R9), X1      // src[104]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 160(R8)       // work[10]
	MOVUPD X3, 176(R8)       // work[11]

	// (24,88) -> work[12], work[13]
	MOVUPD 384(R9), X0       // src[24]
	MOVUPD 1408(R9), X1      // src[88]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 192(R8)       // work[12]
	MOVUPD X3, 208(R8)       // work[13]

	// (56,120) -> work[14], work[15]
	MOVUPD 896(R9), X0       // src[56]
	MOVUPD 1920(R9), X1      // src[120]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 224(R8)       // work[14]
	MOVUPD X3, 240(R8)       // work[15]

	// (4,68) -> work[16], work[17]
	MOVUPD 64(R9), X0        // src[4]
	MOVUPD 1088(R9), X1      // src[68]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 256(R8)       // work[16]
	MOVUPD X3, 272(R8)       // work[17]

	// (36,100) -> work[18], work[19]
	MOVUPD 576(R9), X0       // src[36]
	MOVUPD 1600(R9), X1      // src[100]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 288(R8)       // work[18]
	MOVUPD X3, 304(R8)       // work[19]

	// (20,84) -> work[20], work[21]
	MOVUPD 320(R9), X0       // src[20]
	MOVUPD 1344(R9), X1      // src[84]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 320(R8)       // work[20]
	MOVUPD X3, 336(R8)       // work[21]

	// (52,116) -> work[22], work[23]
	MOVUPD 832(R9), X0       // src[52]
	MOVUPD 1856(R9), X1      // src[116]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 352(R8)       // work[22]
	MOVUPD X3, 368(R8)       // work[23]

	// (12,76) -> work[24], work[25]
	MOVUPD 192(R9), X0       // src[12]
	MOVUPD 1216(R9), X1      // src[76]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 384(R8)       // work[24]
	MOVUPD X3, 400(R8)       // work[25]

	// (44,108) -> work[26], work[27]
	MOVUPD 704(R9), X0       // src[44]
	MOVUPD 1728(R9), X1      // src[108]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 416(R8)       // work[26]
	MOVUPD X3, 432(R8)       // work[27]

	// (28,92) -> work[28], work[29]
	MOVUPD 448(R9), X0       // src[28]
	MOVUPD 1472(R9), X1      // src[92]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 448(R8)       // work[28]
	MOVUPD X3, 464(R8)       // work[29]

	// (60,124) -> work[30], work[31]
	MOVUPD 960(R9), X0       // src[60]
	MOVUPD 1984(R9), X1      // src[124]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 480(R8)       // work[30]
	MOVUPD X3, 496(R8)       // work[31]

	// (2,66) -> work[32], work[33]
	MOVUPD 32(R9), X0        // src[2]
	MOVUPD 1056(R9), X1      // src[66]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 512(R8)       // work[32]
	MOVUPD X3, 528(R8)       // work[33]

	// (34,98) -> work[34], work[35]
	MOVUPD 544(R9), X0       // src[34]
	MOVUPD 1568(R9), X1      // src[98]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 544(R8)       // work[34]
	MOVUPD X3, 560(R8)       // work[35]

	// (18,82) -> work[36], work[37]
	MOVUPD 288(R9), X0       // src[18]
	MOVUPD 1312(R9), X1      // src[82]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 576(R8)       // work[36]
	MOVUPD X3, 592(R8)       // work[37]

	// (50,114) -> work[38], work[39]
	MOVUPD 800(R9), X0       // src[50]
	MOVUPD 1824(R9), X1      // src[114]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 608(R8)       // work[38]
	MOVUPD X3, 624(R8)       // work[39]

	// (10,74) -> work[40], work[41]
	MOVUPD 160(R9), X0       // src[10]
	MOVUPD 1184(R9), X1      // src[74]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 640(R8)       // work[40]
	MOVUPD X3, 656(R8)       // work[41]

	// (42,106) -> work[42], work[43]
	MOVUPD 672(R9), X0       // src[42]
	MOVUPD 1696(R9), X1      // src[106]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 672(R8)       // work[42]
	MOVUPD X3, 688(R8)       // work[43]

	// (26,90) -> work[44], work[45]
	MOVUPD 416(R9), X0       // src[26]
	MOVUPD 1440(R9), X1      // src[90]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 704(R8)       // work[44]
	MOVUPD X3, 720(R8)       // work[45]

	// (58,122) -> work[46], work[47]
	MOVUPD 928(R9), X0       // src[58]
	MOVUPD 1952(R9), X1      // src[122]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 736(R8)       // work[46]
	MOVUPD X3, 752(R8)       // work[47]

	// (6,70) -> work[48], work[49]
	MOVUPD 96(R9), X0        // src[6]
	MOVUPD 1120(R9), X1      // src[70]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 768(R8)       // work[48]
	MOVUPD X3, 784(R8)       // work[49]

	// (38,102) -> work[50], work[51]
	MOVUPD 608(R9), X0       // src[38]
	MOVUPD 1632(R9), X1      // src[102]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 800(R8)       // work[50]
	MOVUPD X3, 816(R8)       // work[51]

	// (22,86) -> work[52], work[53]
	MOVUPD 352(R9), X0       // src[22]
	MOVUPD 1376(R9), X1      // src[86]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 832(R8)       // work[52]
	MOVUPD X3, 848(R8)       // work[53]

	// (54,118) -> work[54], work[55]
	MOVUPD 864(R9), X0       // src[54]
	MOVUPD 1888(R9), X1      // src[118]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 864(R8)       // work[54]
	MOVUPD X3, 880(R8)       // work[55]

	// (14,78) -> work[56], work[57]
	MOVUPD 224(R9), X0       // src[14]
	MOVUPD 1248(R9), X1      // src[78]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 896(R8)       // work[56]
	MOVUPD X3, 912(R8)       // work[57]

	// (46,110) -> work[58], work[59]
	MOVUPD 736(R9), X0       // src[46]
	MOVUPD 1760(R9), X1      // src[110]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 928(R8)       // work[58]
	MOVUPD X3, 944(R8)       // work[59]

	// (30,94) -> work[60], work[61]
	MOVUPD 480(R9), X0       // src[30]
	MOVUPD 1504(R9), X1      // src[94]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 960(R8)       // work[60]
	MOVUPD X3, 976(R8)       // work[61]

	// (62,126) -> work[62], work[63]
	MOVUPD 992(R9), X0       // src[62]
	MOVUPD 2016(R9), X1      // src[126]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 992(R8)       // work[62]
	MOVUPD X3, 1008(R8)      // work[63]

	INCQ BX                  // next pass
	CMPQ BX, $2              // done after odd pass
	JGE  inv_stage1_done
	LEAQ 1024(R14), R8       // work offset for odd half
	LEAQ 16(R15), R9         // src offset for odd half
	JMP  inv_stage1_pass

inv_stage1_done:
	MOVQ R14, R8             // restore work base

	MOVUPS ·maskNegHiPD(SB), X14 // for conj
	MOVUPS ·maskNegLoPD(SB), X13 // for i in complex mul

	// Stage 2: dist 2 - 32 blocks of 4
	MOVQ R8, SI              // work base
	MOVQ $32, CX             // blocks
inv_stage2_loop:
	MOVQ $2, DX              // half=2
inv_stage2_inner:
	MOVUPD (SI), X0          // a
  MOVUPD 32(SI), X1        // b
	MOVQ $2, AX              // k = 2 - DX
  SUBQ DX, AX              // k (0..1)
  SHLQ $5, AX              // k * 32
  SHLQ $4, AX              // k * 32 * 16
  MOVUPD (R10)(AX*1), X10  // twiddle[k*32]
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
  MOVQ $16, CX
inv_s3_loop:
	MOVQ $4, DX
inv_s3_inner:
	MOVUPD (SI), X0
  MOVUPD 64(SI), X1
	MOVQ $4, AX
  SUBQ DX, AX
  SHLQ $4, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10
  XORPD X14, X10
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
  MOVQ $8, CX
inv_s4_loop:
	MOVQ $8, DX
inv_s4_inner:
	MOVUPD (SI), X0
  MOVUPD 128(SI), X1
	MOVQ $8, AX
  SUBQ DX, AX
  SHLQ $3, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10
  XORPD X14, X10
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
  MOVQ $4, CX
inv_s5_loop:
	MOVQ $16, DX
inv_s5_inner:
	MOVUPD (SI), X0
  MOVUPD 256(SI), X1
	MOVQ $16, AX
  SUBQ DX, AX
  SHLQ $2, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10
  XORPD X14, X10
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
  MOVQ $2, CX
inv_s6_loop:
	MOVQ $32, DX
inv_s6_inner:
	MOVUPD (SI), X0
  MOVUPD 512(SI), X1
	MOVQ $32, AX
  SUBQ DX, AX
  SHLQ $1, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10
  XORPD X14, X10
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
  MOVQ $64, DX
inv_s7_inner:
	MOVUPD (SI), X0
  MOVUPD 1024(SI), X1
	MOVQ $64, AX
  SUBQ DX, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10
  XORPD X14, X10
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
	MOVAPD X0, X3
  ADDPD X2, X0
  SUBPD X2, X3
	MOVUPD X0, (SI)
  MOVUPD X3, 1024(SI)
	ADDQ $16, SI
  DECQ DX
  JNZ inv_s7_inner

	// Scale by 1/128
	MOVSD ·oneTwentyEighth64(SB), X15
  SHUFPD $0, X15, X15
	MOVQ $64, CX
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
	MOVQ $64, CX
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

//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-64 Radix-2 FFT Kernels for AMD64 (complex128)
// ===========================================================================

#include "textflag.h"

// Forward transform, size 64, complex128, radix-2
TEXT ·ForwardSSE2Size64Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	CMPQ R13, $64
	JNE  size64_sse2_128_fwd_err

	CMPQ R8, R9
	JNE  size64_sse2_128_fwd_use_dst
	MOVQ R11, R8

size64_sse2_128_fwd_use_dst:
	// -----------------------------------------------------------------------
	// FUSED: Bit-reversal permutation + Stage 1 (identity twiddles)
	// Unrolled even half, executed twice (second pass uses +16 src, +512 dst).
	// -----------------------------------------------------------------------
	MOVQ R8, R14 // save work base
	MOVQ R9, R15 // save src base
	XORQ BX, BX  // pass counter (even/odd)

size64_sse2_128_fwd_stage1_pass:
	// (0,32) -> work[0], work[1]
	MOVUPD 0(R9), X0         // src[0]
	MOVUPD 512(R9), X1       // src[32]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 0(R8)         // work[0]
	MOVUPD X3, 16(R8)        // work[1]

	// (16,48) -> work[2], work[3]
	MOVUPD 256(R9), X0       // src[16]
	MOVUPD 768(R9), X1       // src[48]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 32(R8)        // work[2]
	MOVUPD X3, 48(R8)        // work[3]

	// (8,40) -> work[4], work[5]
	MOVUPD 128(R9), X0       // src[8]
	MOVUPD 640(R9), X1       // src[40]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 64(R8)        // work[4]
	MOVUPD X3, 80(R8)        // work[5]

	// (24,56) -> work[6], work[7]
	MOVUPD 384(R9), X0       // src[24]
	MOVUPD 896(R9), X1       // src[56]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 96(R8)        // work[6]
	MOVUPD X3, 112(R8)       // work[7]

	// (4,36) -> work[8], work[9]
	MOVUPD 64(R9), X0        // src[4]
	MOVUPD 576(R9), X1       // src[36]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 128(R8)       // work[8]
	MOVUPD X3, 144(R8)       // work[9]

	// (20,52) -> work[10], work[11]
	MOVUPD 320(R9), X0       // src[20]
	MOVUPD 832(R9), X1       // src[52]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 160(R8)       // work[10]
	MOVUPD X3, 176(R8)       // work[11]

	// (12,44) -> work[12], work[13]
	MOVUPD 192(R9), X0       // src[12]
	MOVUPD 704(R9), X1       // src[44]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 192(R8)       // work[12]
	MOVUPD X3, 208(R8)       // work[13]

	// (28,60) -> work[14], work[15]
	MOVUPD 448(R9), X0       // src[28]
	MOVUPD 960(R9), X1       // src[60]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 224(R8)       // work[14]
	MOVUPD X3, 240(R8)       // work[15]

	// (2,34) -> work[16], work[17]
	MOVUPD 32(R9), X0        // src[2]
	MOVUPD 544(R9), X1       // src[34]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 256(R8)       // work[16]
	MOVUPD X3, 272(R8)       // work[17]

	// (18,50) -> work[18], work[19]
	MOVUPD 288(R9), X0       // src[18]
	MOVUPD 800(R9), X1       // src[50]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 288(R8)       // work[18]
	MOVUPD X3, 304(R8)       // work[19]

	// (10,42) -> work[20], work[21]
	MOVUPD 160(R9), X0       // src[10]
	MOVUPD 672(R9), X1       // src[42]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 320(R8)       // work[20]
	MOVUPD X3, 336(R8)       // work[21]

	// (26,58) -> work[22], work[23]
	MOVUPD 416(R9), X0       // src[26]
	MOVUPD 928(R9), X1       // src[58]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 352(R8)       // work[22]
	MOVUPD X3, 368(R8)       // work[23]

	// (6,38) -> work[24], work[25]
	MOVUPD 96(R9), X0        // src[6]
	MOVUPD 608(R9), X1       // src[38]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 384(R8)       // work[24]
	MOVUPD X3, 400(R8)       // work[25]

	// (22,54) -> work[26], work[27]
	MOVUPD 352(R9), X0       // src[22]
	MOVUPD 864(R9), X1       // src[54]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 416(R8)       // work[26]
	MOVUPD X3, 432(R8)       // work[27]

	// (14,46) -> work[28], work[29]
	MOVUPD 224(R9), X0       // src[14]
	MOVUPD 736(R9), X1       // src[46]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 448(R8)       // work[28]
	MOVUPD X3, 464(R8)       // work[29]

	// (30,62) -> work[30], work[31]
	MOVUPD 480(R9), X0       // src[30]
	MOVUPD 992(R9), X1       // src[62]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 480(R8)       // work[30]
	MOVUPD X3, 496(R8)       // work[31]

	INCQ BX                  // next pass
	CMPQ BX, $2              // done after odd pass
	JGE  size64_sse2_128_fwd_stage1_done
	LEAQ 512(R14), R8        // work offset for odd half
	LEAQ 16(R15), R9         // src offset for odd half
	JMP  size64_sse2_128_fwd_stage1_pass

size64_sse2_128_fwd_stage1_done:
	MOVQ R14, R8             // restore work base

	// Stage 2: dist 2 - 16 blocks of 4
	MOVQ R8, SI              // work base
	MOVQ $16, CX             // blocks
	MOVUPS ·maskNegLoPD(SB), X14
size64_sse2_128_fwd_stage2_loop:
	MOVQ $2, DX              // half=2
size64_sse2_128_fwd_stage2_inner:
	MOVUPD (SI), X0          // a
  MOVUPD 32(SI), X1        // b
	MOVQ $2, AX              // k = 2 - DX
  SUBQ DX, AX              // k (0..1)
  SHLQ $4, AX              // k * 16
  SHLQ $4, AX              // k * 16 * 16
  MOVUPD (R10)(AX*1), X10  // twiddle[k*16]
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
  JNZ size64_sse2_128_fwd_stage2_inner
	ADDQ $32, SI             // next block
  DECQ CX                  // next block
  JNZ size64_sse2_128_fwd_stage2_loop

	// Stage 3: dist 4 - 8 blocks of 8
	MOVQ R8, SI
	MOVQ $8, CX
size64_sse2_128_fwd_stage3_loop:
	MOVQ $4, DX
size64_sse2_128_fwd_stage3_inner:
	MOVUPD (SI), X0
  MOVUPD 64(SI), X1
	MOVQ $4, AX
  SUBQ DX, AX
  SHLQ $3, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10 // k * 64/8 * 16 = k * 8 * 16
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
  JNZ size64_sse2_128_fwd_stage3_inner
	ADDQ $64, SI
  DECQ CX
  JNZ size64_sse2_128_fwd_stage3_loop

	// Stage 4: dist 8 - 4 blocks of 16
	MOVQ R8, SI
	MOVQ $4, CX
size64_sse2_128_fwd_stage4_loop:
	MOVQ $8, DX
size64_sse2_128_fwd_stage4_inner:
	MOVUPD (SI), X0
  MOVUPD 128(SI), X1
	MOVQ $8, AX
  SUBQ DX, AX
  SHLQ $2, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10 // k * 64/16 * 16 = k * 4 * 16
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
  JNZ size64_sse2_128_fwd_stage4_inner
	ADDQ $128, SI
  DECQ CX
  JNZ size64_sse2_128_fwd_stage4_loop

	// Stage 5: dist 16 - 2 blocks of 32
	MOVQ R8, SI
	MOVQ $2, CX
size64_sse2_128_fwd_stage5_loop:
	MOVQ $16, DX
size64_sse2_128_fwd_stage5_inner:
	MOVUPD (SI), X0
  MOVUPD 256(SI), X1
	MOVQ $16, AX
  SUBQ DX, AX
  SHLQ $1, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10 // k * 64/32 * 16 = k * 2 * 16
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
  JNZ size64_sse2_128_fwd_stage5_inner
	ADDQ $256, SI
  DECQ CX
  JNZ size64_sse2_128_fwd_stage5_loop

	// Stage 6: dist 32 - 1 block of 64
	MOVQ R8, SI
	MOVQ $32, DX
size64_sse2_128_fwd_stage6_inner:
	MOVUPD (SI), X0
  MOVUPD 512(SI), X1
	MOVQ $32, AX
  SUBQ DX, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10 // k * 64/64 * 16 = k * 1 * 16
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
  JNZ size64_sse2_128_fwd_stage6_inner

	// Copy to dst
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   size64_sse2_128_fwd_done
	MOVQ $32, CX
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

size64_sse2_128_fwd_done:
	MOVB $1, ret+96(FP)
	RET
size64_sse2_128_fwd_err:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 64, complex128, radix-2
TEXT ·InverseSSE2Size64Radix2Complex128Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	CMPQ R13, $64
	JNE  size64_sse2_128_inv_err

	CMPQ R8, R9
	JNE  size64_sse2_128_inv_use_dst
	MOVQ R11, R8

size64_sse2_128_inv_use_dst:
	// -----------------------------------------------------------------------
	// FUSED: Bit-reversal permutation + Stage 1 (identity twiddles)
	// Unrolled even half, executed twice (second pass uses +16 src, +512 dst).
	// -----------------------------------------------------------------------
	MOVQ R8, R14 // save work base
	MOVQ R9, R15 // save src base
	XORQ BX, BX  // pass counter (even/odd)

size64_sse2_128_inv_stage1_pass:
	// (0,32) -> work[0], work[1]
	MOVUPD 0(R9), X0         // src[0]
	MOVUPD 512(R9), X1       // src[32]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 0(R8)         // work[0]
	MOVUPD X3, 16(R8)        // work[1]

	// (16,48) -> work[2], work[3]
	MOVUPD 256(R9), X0       // src[16]
	MOVUPD 768(R9), X1       // src[48]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 32(R8)        // work[2]
	MOVUPD X3, 48(R8)        // work[3]

	// (8,40) -> work[4], work[5]
	MOVUPD 128(R9), X0       // src[8]
	MOVUPD 640(R9), X1       // src[40]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 64(R8)        // work[4]
	MOVUPD X3, 80(R8)        // work[5]

	// (24,56) -> work[6], work[7]
	MOVUPD 384(R9), X0       // src[24]
	MOVUPD 896(R9), X1       // src[56]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 96(R8)        // work[6]
	MOVUPD X3, 112(R8)       // work[7]

	// (4,36) -> work[8], work[9]
	MOVUPD 64(R9), X0        // src[4]
	MOVUPD 576(R9), X1       // src[36]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 128(R8)       // work[8]
	MOVUPD X3, 144(R8)       // work[9]

	// (20,52) -> work[10], work[11]
	MOVUPD 320(R9), X0       // src[20]
	MOVUPD 832(R9), X1       // src[52]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 160(R8)       // work[10]
	MOVUPD X3, 176(R8)       // work[11]

	// (12,44) -> work[12], work[13]
	MOVUPD 192(R9), X0       // src[12]
	MOVUPD 704(R9), X1       // src[44]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 192(R8)       // work[12]
	MOVUPD X3, 208(R8)       // work[13]

	// (28,60) -> work[14], work[15]
	MOVUPD 448(R9), X0       // src[28]
	MOVUPD 960(R9), X1       // src[60]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 224(R8)       // work[14]
	MOVUPD X3, 240(R8)       // work[15]

	// (2,34) -> work[16], work[17]
	MOVUPD 32(R9), X0        // src[2]
	MOVUPD 544(R9), X1       // src[34]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 256(R8)       // work[16]
	MOVUPD X3, 272(R8)       // work[17]

	// (18,50) -> work[18], work[19]
	MOVUPD 288(R9), X0       // src[18]
	MOVUPD 800(R9), X1       // src[50]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 288(R8)       // work[18]
	MOVUPD X3, 304(R8)       // work[19]

	// (10,42) -> work[20], work[21]
	MOVUPD 160(R9), X0       // src[10]
	MOVUPD 672(R9), X1       // src[42]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 320(R8)       // work[20]
	MOVUPD X3, 336(R8)       // work[21]

	// (26,58) -> work[22], work[23]
	MOVUPD 416(R9), X0       // src[26]
	MOVUPD 928(R9), X1       // src[58]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 352(R8)       // work[22]
	MOVUPD X3, 368(R8)       // work[23]

	// (6,38) -> work[24], work[25]
	MOVUPD 96(R9), X0        // src[6]
	MOVUPD 608(R9), X1       // src[38]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 384(R8)       // work[24]
	MOVUPD X3, 400(R8)       // work[25]

	// (22,54) -> work[26], work[27]
	MOVUPD 352(R9), X0       // src[22]
	MOVUPD 864(R9), X1       // src[54]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 416(R8)       // work[26]
	MOVUPD X3, 432(R8)       // work[27]

	// (14,46) -> work[28], work[29]
	MOVUPD 224(R9), X0       // src[14]
	MOVUPD 736(R9), X1       // src[46]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 448(R8)       // work[28]
	MOVUPD X3, 464(R8)       // work[29]

	// (30,62) -> work[30], work[31]
	MOVUPD 480(R9), X0       // src[30]
	MOVUPD 992(R9), X1       // src[62]
	MOVAPD X0, X2            // a
	ADDPD X1, X2             // a + b
	MOVAPD X0, X3            // a
	SUBPD X1, X3             // a - b
	MOVUPD X2, 480(R8)       // work[30]
	MOVUPD X3, 496(R8)       // work[31]

	INCQ BX                  // next pass
	CMPQ BX, $2              // done after odd pass
	JGE  size64_sse2_128_inv_stage1_done
	LEAQ 512(R14), R8        // work offset for odd half
	LEAQ 16(R15), R9         // src offset for odd half
	JMP  size64_sse2_128_inv_stage1_pass

size64_sse2_128_inv_stage1_done:
	MOVQ R14, R8             // restore work base

	MOVUPS ·maskNegHiPD(SB), X14 // for conj
	MOVUPS ·maskNegLoPD(SB), X13 // for i in complex mul

	// Stage 2: dist 2 - 16 blocks of 4
	MOVQ R8, SI              // work base
	MOVQ $16, CX             // blocks
size64_sse2_128_inv_stage2_loop:
	MOVQ $2, DX              // half=2
size64_sse2_128_inv_stage2_inner:
	MOVUPD (SI), X0          // a
  MOVUPD 32(SI), X1        // b
	MOVQ $2, AX              // k = 2 - DX
  SUBQ DX, AX              // k (0..1)
  SHLQ $4, AX              // k * 16
  SHLQ $4, AX              // k * 16 * 16
  MOVUPD (R10)(AX*1), X10  // twiddle[k*16]
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
  JNZ size64_sse2_128_inv_stage2_inner
	ADDQ $32, SI             // next block
  DECQ CX                  // next block
  JNZ size64_sse2_128_inv_stage2_loop

	// Stage 3
	MOVQ R8, SI
  MOVQ $8, CX
size64_sse2_128_inv_stage3_loop:
	MOVQ $4, DX
size64_sse2_128_inv_stage3_inner:
	MOVUPD (SI), X0
  MOVUPD 64(SI), X1
	MOVQ $4, AX
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
  MOVUPD X3, 64(SI)
	ADDQ $16, SI
  DECQ DX
  JNZ size64_sse2_128_inv_stage3_inner
	ADDQ $64, SI
  DECQ CX
  JNZ size64_sse2_128_inv_stage3_loop

	// Stage 4
	MOVQ R8, SI
  MOVQ $4, CX
size64_sse2_128_inv_stage4_loop:
	MOVQ $8, DX
size64_sse2_128_inv_stage4_inner:
	MOVUPD (SI), X0
  MOVUPD 128(SI), X1
	MOVQ $8, AX
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
  MOVUPD X3, 128(SI)
	ADDQ $16, SI
  DECQ DX
  JNZ size64_sse2_128_inv_stage4_inner
	ADDQ $128, SI
  DECQ CX
  JNZ size64_sse2_128_inv_stage4_loop

	// Stage 5
	MOVQ R8, SI
  MOVQ $2, CX
size64_sse2_128_inv_stage5_loop:
	MOVQ $16, DX
size64_sse2_128_inv_stage5_inner:
	MOVUPD (SI), X0
  MOVUPD 256(SI), X1
	MOVQ $16, AX
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
  MOVUPD X3, 256(SI)
	ADDQ $16, SI
  DECQ DX
  JNZ size64_sse2_128_inv_stage5_inner
	ADDQ $256, SI
  DECQ CX
  JNZ size64_sse2_128_inv_stage5_loop

	// Stage 6
	MOVQ R8, SI
  MOVQ $32, DX
size64_sse2_128_inv_stage6_inner:
	MOVUPD (SI), X0
  MOVUPD 512(SI), X1
	MOVQ $32, AX
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
  MOVUPD X3, 512(SI)
	ADDQ $16, SI
  DECQ DX
  JNZ size64_sse2_128_inv_stage6_inner

	// Scale by 1/64
	MOVSD ·sixtyFourth64(SB), X15
  SHUFPD $0, X15, X15
	MOVQ $32, CX
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
	JE   size64_sse2_128_inv_done
	MOVQ $32, CX
  MOVQ R8, SI
  MOVQ R14, DI
inv_copy_loop:
	MOVUPD (SI), X0
  MOVUPD 16(SI), X1
  MOVUPD X0, (DI)
  MOVUPD X1, 16(DI)
  ADDQ $32, SI
  ADDQ $32, DI
  DECQ CX
  JNZ inv_copy_loop

size64_sse2_128_inv_done:
	MOVB $1, ret+96(FP)
	RET
size64_sse2_128_inv_err:
	MOVB $0, ret+96(FP)
	RET

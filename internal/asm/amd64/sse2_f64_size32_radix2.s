//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-32 Radix-2 FFT Kernels for AMD64 (complex128)
// ===========================================================================

#include "textflag.h"

// Forward transform, size 32, complex128, radix-2
TEXT ·ForwardSSE2Size32Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	CMPQ R13, $32
	JNE  size32_sse2_128_fwd_err

	CMPQ R8, R9
	JNE  size32_sse2_128_fwd_use_dst
	MOVQ R11, R8

size32_sse2_128_fwd_use_dst:
	// -----------------------------------------------------------------------
	// FUSED: Bit-reversal permutation + Stage 1 (identity twiddles)
	// Unrolled even half, executed twice (second pass uses +16 src, +256 dst).
	// -----------------------------------------------------------------------
	MOVQ R8, R14 // save work base
	MOVQ R9, R15 // save src base
	XORQ BX, BX  // pass counter (even/odd)

size32_sse2_128_fwd_stage1_pass:
	// (0,16) -> work[0], work[1]
	MOVUPD 0(R9), X0
	MOVUPD 256(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 0(R8)
	MOVUPD X3, 16(R8)

	// (8,24) -> work[2], work[3]
	MOVUPD 128(R9), X0
	MOVUPD 384(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 32(R8)
	MOVUPD X3, 48(R8)

	// (4,20) -> work[4], work[5]
	MOVUPD 64(R9), X0
	MOVUPD 320(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 64(R8)
	MOVUPD X3, 80(R8)

	// (12,28) -> work[6], work[7]
	MOVUPD 192(R9), X0
	MOVUPD 448(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 96(R8)
	MOVUPD X3, 112(R8)

	// (2,18) -> work[8], work[9]
	MOVUPD 32(R9), X0
	MOVUPD 288(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 128(R8)
	MOVUPD X3, 144(R8)

	// (10,26) -> work[10], work[11]
	MOVUPD 160(R9), X0
	MOVUPD 416(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 160(R8)
	MOVUPD X3, 176(R8)

	// (6,22) -> work[12], work[13]
	MOVUPD 96(R9), X0
	MOVUPD 352(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 192(R8)
	MOVUPD X3, 208(R8)

	// (14,30) -> work[14], work[15]
	MOVUPD 224(R9), X0
	MOVUPD 480(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 224(R8)
	MOVUPD X3, 240(R8)

	INCQ BX
	CMPQ BX, $2
	JGE  size32_sse2_128_fwd_stage1_done
	LEAQ 256(R14), R8 // work offset for odd half
	LEAQ 16(R15), R9  // src offset for odd half
	JMP  size32_sse2_128_fwd_stage1_pass

size32_sse2_128_fwd_stage1_done:
	MOVQ R14, R8 // restore work base

	// Stage 2: dist 2 - 8 blocks of 4
	MOVQ R8, SI
	MOVQ $8, CX
	MOVUPS ·maskNegLoPD(SB), X14
size32_sse2_128_fwd_stage2_loop:
	MOVQ $2, DX
size32_sse2_128_fwd_stage2_inner:
	MOVUPD (SI), X0
  MOVUPD 32(SI), X1
	MOVQ $2, AX
  SUBQ DX, AX
  SHLQ $3, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10 // k * 32/4 * 16 = k * 8 * 16
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
  MOVUPD X3, 32(SI)
	ADDQ $16, SI
  DECQ DX
  JNZ size32_sse2_128_fwd_stage2_inner
	ADDQ $32, SI
  DECQ CX
  JNZ size32_sse2_128_fwd_stage2_loop

	// Stage 3: dist 4 - 4 blocks of 8
	MOVQ R8, SI
	MOVQ $4, CX
	MOVUPS ·maskNegLoPD(SB), X14
size32_sse2_128_fwd_stage3_loop:
	MOVQ $4, DX
size32_sse2_128_fwd_stage3_inner:
	MOVUPD (SI), X0
  MOVUPD 64(SI), X1
	MOVQ $4, AX
  SUBQ DX, AX
  SHLQ $2, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10 // k * 32/8 * 16 = k * 4 * 16
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
  JNZ size32_sse2_128_fwd_stage3_inner
	ADDQ $64, SI
  DECQ CX
  JNZ size32_sse2_128_fwd_stage3_loop

	// Stage 4: dist 8 - 2 blocks of 16
	MOVQ R8, SI
	MOVQ $2, CX
size32_sse2_128_fwd_stage4_loop:
	MOVQ $8, DX
size32_sse2_128_fwd_stage4_inner:
	MOVUPD (SI), X0
  MOVUPD 128(SI), X1
	MOVQ $8, AX
  SUBQ DX, AX
  SHLQ $1, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10 // k * 32/16 * 16 = k * 2 * 16
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
  JNZ size32_sse2_128_fwd_stage4_inner
	ADDQ $128, SI
  DECQ CX
  JNZ size32_sse2_128_fwd_stage4_loop

	// Stage 5: dist 16 - 1 block of 32
	MOVQ R8, SI
	MOVQ $16, DX
size32_sse2_128_fwd_stage5_inner:
	MOVUPD (SI), X0
  MOVUPD 256(SI), X1
	MOVQ $16, AX
  SUBQ DX, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10 // k * 32/32 * 16 = k * 1 * 16
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
  JNZ size32_sse2_128_fwd_stage5_inner

	// Copy to dst
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   size32_sse2_128_fwd_done
	MOVQ $32, CX
  MOVQ R8, SI
  MOVQ R14, DI
size32_sse2_128_fwd_copy:
	MOVUPD (SI), X0
  MOVUPD X0, (DI)
  ADDQ $16, SI
  ADDQ $16, DI
  DECQ CX
  JNZ size32_sse2_128_fwd_copy

size32_sse2_128_fwd_done:
	MOVB $1, ret+96(FP)
	RET
size32_sse2_128_fwd_err:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 32, complex128, radix-2
TEXT ·InverseSSE2Size32Radix2Complex128Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	CMPQ R13, $32
	JNE  size32_sse2_128_inv_err

	CMPQ R8, R9
	JNE  size32_sse2_128_inv_use_dst
	MOVQ R11, R8

size32_sse2_128_inv_use_dst:
	// -----------------------------------------------------------------------
	// FUSED: Bit-reversal permutation + Stage 1 (identity twiddles)
	// Unrolled even half, executed twice (second pass uses +16 src, +256 dst).
	// -----------------------------------------------------------------------
	MOVQ R8, R14 // save work base
	MOVQ R9, R15 // save src base
	XORQ BX, BX  // pass counter (even/odd)

size32_sse2_128_inv_stage1_pass:
	// (0,16) -> work[0], work[1]
	MOVUPD 0(R9), X0
	MOVUPD 256(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 0(R8)
	MOVUPD X3, 16(R8)

	// (8,24) -> work[2], work[3]
	MOVUPD 128(R9), X0
	MOVUPD 384(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 32(R8)
	MOVUPD X3, 48(R8)

	// (4,20) -> work[4], work[5]
	MOVUPD 64(R9), X0
	MOVUPD 320(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 64(R8)
	MOVUPD X3, 80(R8)

	// (12,28) -> work[6], work[7]
	MOVUPD 192(R9), X0
	MOVUPD 448(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 96(R8)
	MOVUPD X3, 112(R8)

	// (2,18) -> work[8], work[9]
	MOVUPD 32(R9), X0
	MOVUPD 288(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 128(R8)
	MOVUPD X3, 144(R8)

	// (10,26) -> work[10], work[11]
	MOVUPD 160(R9), X0
	MOVUPD 416(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 160(R8)
	MOVUPD X3, 176(R8)

	// (6,22) -> work[12], work[13]
	MOVUPD 96(R9), X0
	MOVUPD 352(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 192(R8)
	MOVUPD X3, 208(R8)

	// (14,30) -> work[14], work[15]
	MOVUPD 224(R9), X0
	MOVUPD 480(R9), X1
	MOVAPD X0, X2
	ADDPD X1, X2
	MOVAPD X0, X3
	SUBPD X1, X3
	MOVUPD X2, 224(R8)
	MOVUPD X3, 240(R8)

	INCQ BX
	CMPQ BX, $2
	JGE  size32_sse2_128_inv_stage1_done
	LEAQ 256(R14), R8 // work offset for odd half
	LEAQ 16(R15), R9  // src offset for odd half
	JMP  size32_sse2_128_inv_stage1_pass

size32_sse2_128_inv_stage1_done:
	MOVQ R14, R8 // restore work base

	MOVUPS ·maskNegHiPD(SB), X14 // for conj
	MOVUPS ·maskNegLoPD(SB), X13 // for i in complex mul

	// Stage 2: dist 2 - 8 blocks of 4
	MOVQ R8, SI
	MOVQ $8, CX
size32_sse2_128_inv_stage2_loop:
	MOVQ $2, DX
size32_sse2_128_inv_stage2_inner:
	MOVUPD (SI), X0
  MOVUPD 32(SI), X1
	MOVQ $2, AX
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
  MOVUPD X3, 32(SI)
	ADDQ $16, SI
  DECQ DX
  JNZ size32_sse2_128_inv_stage2_inner
	ADDQ $32, SI
  DECQ CX
  JNZ size32_sse2_128_inv_stage2_loop

	// Stage 3: dist 4
	MOVQ R8, SI
	MOVQ $4, CX
size32_sse2_128_inv_stage3_loop:
	MOVQ $4, DX
size32_sse2_128_inv_stage3_inner:
	MOVUPD (SI), X0
  MOVUPD 64(SI), X1
	MOVQ $4, AX
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
  MOVUPD X3, 64(SI)
	ADDQ $16, SI
  DECQ DX
  JNZ size32_sse2_128_inv_stage3_inner
	ADDQ $64, SI
  DECQ CX
  JNZ size32_sse2_128_inv_stage3_loop

	// Stage 4: dist 8
	MOVQ R8, SI
	MOVQ $2, CX
size32_sse2_128_inv_stage4_loop:
	MOVQ $8, DX
size32_sse2_128_inv_stage4_inner:
	MOVUPD (SI), X0
  MOVUPD 128(SI), X1
	MOVQ $8, AX
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
  MOVUPD X3, 128(SI)
	ADDQ $16, SI
  DECQ DX
  JNZ size32_sse2_128_inv_stage4_inner
	ADDQ $128, SI
  DECQ CX
  JNZ size32_sse2_128_inv_stage4_loop

	// Stage 5: dist 16
	MOVQ R8, SI
	MOVQ $16, DX
size32_sse2_128_inv_stage5_inner:
	MOVUPD (SI), X0
  MOVUPD 256(SI), X1
	MOVQ $16, AX
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
  MOVUPD X3, 256(SI)
	ADDQ $16, SI
  DECQ DX
  JNZ size32_sse2_128_inv_stage5_inner

	// Scale by 1/32
	MOVSD ·thirtySecond64(SB), X15
  SHUFPD $0, X15, X15
	MOVQ $32, CX
  MOVQ R8, SI
size32_sse2_128_inv_scale:
	MOVUPD (SI), X0
  MULPD X15, X0
  MOVUPD X0, (SI)
  ADDQ $16, SI
  DECQ CX
  JNZ size32_sse2_128_inv_scale

	// Copy to dst
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   size32_sse2_128_inv_done
	MOVQ $32, CX
  MOVQ R8, SI
  MOVQ R14, DI
size32_sse2_128_inv_copy:
	MOVUPD (SI), X0
  MOVUPD X0, (DI)
  ADDQ $16, SI
  ADDQ $16, DI
  DECQ CX
  JNZ size32_sse2_128_inv_copy

size32_sse2_128_inv_done:
	MOVB $1, ret+96(FP)
	RET
size32_sse2_128_inv_err:
	MOVB $0, ret+96(FP)
	RET

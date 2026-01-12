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
	LEAQ ·bitrevSSE2Size32Radix2(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $32
	JNE  size32_sse2_128_fwd_err

	CMPQ R8, R9
	JNE  size32_sse2_128_fwd_use_dst
	MOVQ R11, R8

size32_sse2_128_fwd_use_dst:
	// Bit-reversal permutation + Stage 1 (load bitrev directly in Stage 1/2 loop)

	// Stage 1 & 2 (Combined) - 8 blocks of 4
	MOVQ R8, SI
	MOVQ $8, CX
	MOVUPS ·maskNegHiPD(SB), X15

size32_sse2_128_fwd_stage12_loop:
	MOVUPD (SI), X0
  MOVUPD 16(SI), X1
  MOVUPD 32(SI), X2
  MOVUPD 48(SI), X3
	MOVAPD X0, X8
  ADDPD X1, X0
  SUBPD X1, X8
	MOVAPD X2, X9
  ADDPD X3, X2
  SUBPD X3, X9
	MOVAPD X0, X10
  ADDPD X2, X0
  SUBPD X2, X10
	MOVAPD X9, X11
  SHUFPD $1, X11, X11
  XORPD X15, X11 // t = W3 * -i
	MOVAPD X8, X12
  ADDPD X11, X8
  SUBPD X11, X12
	MOVUPD X0, (SI)
  MOVUPD X8, 16(SI)
  MOVUPD X10, 32(SI)
  MOVUPD X12, 48(SI)
	ADDQ $64, SI
	DECQ CX
	JNZ  size32_sse2_128_fwd_stage12_loop

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
	LEAQ ·bitrevSSE2Size32Radix2(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $32
	JNE  size32_sse2_128_inv_err

	CMPQ R8, R9
	JNE  size32_sse2_128_inv_use_dst
	MOVQ R11, R8

size32_sse2_128_inv_use_dst:
	// Bit-reversal permutation + Stage 1 (load bitrev directly in Stage 1/2 loop)

	// Stage 1 & 2
	MOVQ R8, SI
	MOVQ $8, CX
	MOVUPS ·maskNegLoPD(SB), X15 // for i

size32_sse2_128_inv_stage12_loop:
	MOVUPD (SI), X0
  MOVUPD 16(SI), X1
  MOVUPD 32(SI), X2
  MOVUPD 48(SI), X3
	MOVAPD X0, X8
  ADDPD X1, X0
  SUBPD X1, X8
	MOVAPD X2, X9
  ADDPD X3, X2
  SUBPD X3, X9
	MOVAPD X0, X10
  ADDPD X2, X0
  SUBPD X2, X10
	MOVAPD X9, X11
  SHUFPD $1, X11, X11
  XORPD X15, X11 // t = W3 * i
	MOVAPD X8, X12
  ADDPD X11, X8
  SUBPD X11, X12
	MOVUPD X0, (SI)
  MOVUPD X8, 16(SI)
  MOVUPD X10, 32(SI)
  MOVUPD X12, 48(SI)
	ADDQ $64, SI
	DECQ CX
	JNZ  size32_sse2_128_inv_stage12_loop

	MOVUPS ·maskNegHiPD(SB), X14 // for conj
	MOVUPS ·maskNegLoPD(SB), X13 // for i in complex mul

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

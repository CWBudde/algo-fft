//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-512 Radix-2 FFT Kernels for AMD64 (complex128)
// ===========================================================================

#include "textflag.h"

// Forward transform, size 512, complex128, radix-2
TEXT ·ForwardSSE2Size512Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size512Radix2(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $512
	JNE  size512_r2_f64_return_false

	// Validate all slice lengths >= 512
	MOVQ dst+8(FP), AX
	CMPQ AX, $512
	JL   size512_r2_f64_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $512
	JL   size512_r2_f64_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $512
	JL   size512_r2_f64_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size512_r2_f64_use_dst
	MOVQ R11, R8

size512_r2_f64_use_dst:
	// Bit-reversal permutation (radix-2)
	XORQ CX, CX

size512_r2_f64_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X0
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	INCQ CX
	CMPQ CX, $512
	JL   size512_r2_f64_bitrev_loop

	// Stage 1: size=2, half=1, step=256
	MOVQ R8, SI
	MOVQ $256, CX
size512_r2_f64_stage1_loop:
	MOVUPD (SI), X0
	MOVUPD 16(SI), X1
	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 16(SI)
	ADDQ $32, SI
	DECQ CX
	JNZ  size512_r2_f64_stage1_loop

	// Stage 2: size=4, half=2, step=128
	MOVQ R8, SI
	MOVQ $128, CX
size512_r2_f64_stage2_loop:
	MOVQ $2, DX
size512_r2_f64_stage2_inner:
	MOVUPD (SI), X0
	MOVUPD 32(SI), X1
	MOVQ $2, AX
	SUBQ DX, AX
	SHLQ $7, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	// Complex multiply a1*w
	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 32(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_stage2_inner
	ADDQ $32, SI
	DECQ CX
	JNZ size512_r2_f64_stage2_loop

	// Stage 3: size=8, half=4, step=64
	MOVQ R8, SI
	MOVQ $64, CX
size512_r2_f64_stage3_loop:
	MOVQ $4, DX
size512_r2_f64_stage3_inner:
	MOVUPD (SI), X0
	MOVUPD 64(SI), X1
	MOVQ $4, AX
	SUBQ DX, AX
	SHLQ $6, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 64(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_stage3_inner
	ADDQ $64, SI
	DECQ CX
	JNZ size512_r2_f64_stage3_loop

	// Stage 4: size=16, half=8, step=32
	MOVQ R8, SI
	MOVQ $32, CX
size512_r2_f64_stage4_loop:
	MOVQ $8, DX
size512_r2_f64_stage4_inner:
	MOVUPD (SI), X0
	MOVUPD 128(SI), X1
	MOVQ $8, AX
	SUBQ DX, AX
	SHLQ $5, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 128(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_stage4_inner
	ADDQ $128, SI
	DECQ CX
	JNZ size512_r2_f64_stage4_loop

	// Stage 5: size=32, half=16, step=16
	MOVQ R8, SI
	MOVQ $16, CX
size512_r2_f64_stage5_loop:
	MOVQ $16, DX
size512_r2_f64_stage5_inner:
	MOVUPD (SI), X0
	MOVUPD 256(SI), X1
	MOVQ $16, AX
	SUBQ DX, AX
	SHLQ $4, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 256(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_stage5_inner
	ADDQ $256, SI
	DECQ CX
	JNZ size512_r2_f64_stage5_loop

	// Stage 6: size=64, half=32, step=8
	MOVQ R8, SI
	MOVQ $8, CX
size512_r2_f64_stage6_loop:
	MOVQ $32, DX
size512_r2_f64_stage6_inner:
	MOVUPD (SI), X0
	MOVUPD 512(SI), X1
	MOVQ $32, AX
	SUBQ DX, AX
	SHLQ $3, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 512(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_stage6_inner
	ADDQ $512, SI
	DECQ CX
	JNZ size512_r2_f64_stage6_loop

	// Stage 7: size=128, half=64, step=4
	MOVQ R8, SI
	MOVQ $4, CX
size512_r2_f64_stage7_loop:
	MOVQ $64, DX
size512_r2_f64_stage7_inner:
	MOVUPD (SI), X0
	MOVUPD 1024(SI), X1
	MOVQ $64, AX
	SUBQ DX, AX
	SHLQ $2, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 1024(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_stage7_inner
	ADDQ $1024, SI
	DECQ CX
	JNZ size512_r2_f64_stage7_loop

	// Stage 8: size=256, half=128, step=2
	MOVQ R8, SI
	MOVQ $2, CX
size512_r2_f64_stage8_loop:
	MOVQ $128, DX
size512_r2_f64_stage8_inner:
	MOVUPD (SI), X0
	MOVUPD 2048(SI), X1
	MOVQ $128, AX
	SUBQ DX, AX
	SHLQ $1, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 2048(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_stage8_inner
	ADDQ $2048, SI
	DECQ CX
	JNZ size512_r2_f64_stage8_loop

	// Stage 9: size=512, half=256, step=1
	MOVQ R8, SI
	MOVQ $256, DX
size512_r2_f64_stage9_inner:
	MOVUPD (SI), X0
	MOVUPD 4096(SI), X1
	MOVQ $256, AX
	SUBQ DX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 4096(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_stage9_inner

	// Copy to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size512_r2_f64_done

	MOVQ $512, CX
	MOVQ R8, SI
size512_r2_f64_copy_loop:
	MOVUPD (SI), X0
	MOVUPD X0, (R9)
	ADDQ $16, SI
	ADDQ $16, R9
	DECQ CX
	JNZ size512_r2_f64_copy_loop

size512_r2_f64_done:
	MOVB $1, ret+96(FP)
	RET

size512_r2_f64_return_false:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 512, complex128, radix-2
TEXT ·InverseSSE2Size512Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size512Radix2(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $512
	JNE  size512_r2_f64_inv_return_false

	// Validate all slice lengths >= 512
	MOVQ dst+8(FP), AX
	CMPQ AX, $512
	JL   size512_r2_f64_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $512
	JL   size512_r2_f64_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $512
	JL   size512_r2_f64_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size512_r2_f64_inv_use_dst
	MOVQ R11, R8

size512_r2_f64_inv_use_dst:
	// Bit-reversal permutation (radix-2)
	XORQ CX, CX

size512_r2_f64_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X0
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	INCQ CX
	CMPQ CX, $512
	JL   size512_r2_f64_inv_bitrev_loop

	MOVUPS ·maskNegHiPD(SB), X15

	// Stage 1: size=2, half=1, step=256
	MOVQ R8, SI
	MOVQ $256, CX
size512_r2_f64_inv_stage1_loop:
	MOVUPD (SI), X0
	MOVUPD 16(SI), X1
	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 16(SI)
	ADDQ $32, SI
	DECQ CX
	JNZ  size512_r2_f64_inv_stage1_loop

	// Stage 2: size=4, half=2, step=128
	MOVQ R8, SI
	MOVQ $128, CX
size512_r2_f64_inv_stage2_loop:
	MOVQ $2, DX
size512_r2_f64_inv_stage2_inner:
	MOVUPD (SI), X0
	MOVUPD 32(SI), X1
	MOVQ $2, AX
	SUBQ DX, AX
	SHLQ $7, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8
	XORPD X15, X8

	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 32(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_inv_stage2_inner
	ADDQ $32, SI
	DECQ CX
	JNZ size512_r2_f64_inv_stage2_loop

	// Stage 3: size=8, half=4, step=64
	MOVQ R8, SI
	MOVQ $64, CX
size512_r2_f64_inv_stage3_loop:
	MOVQ $4, DX
size512_r2_f64_inv_stage3_inner:
	MOVUPD (SI), X0
	MOVUPD 64(SI), X1
	MOVQ $4, AX
	SUBQ DX, AX
	SHLQ $6, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8
	XORPD X15, X8

	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 64(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_inv_stage3_inner
	ADDQ $64, SI
	DECQ CX
	JNZ size512_r2_f64_inv_stage3_loop

	// Stage 4: size=16, half=8, step=32
	MOVQ R8, SI
	MOVQ $32, CX
size512_r2_f64_inv_stage4_loop:
	MOVQ $8, DX
size512_r2_f64_inv_stage4_inner:
	MOVUPD (SI), X0
	MOVUPD 128(SI), X1
	MOVQ $8, AX
	SUBQ DX, AX
	SHLQ $5, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8
	XORPD X15, X8

	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 128(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_inv_stage4_inner
	ADDQ $128, SI
	DECQ CX
	JNZ size512_r2_f64_inv_stage4_loop

	// Stage 5: size=32, half=16, step=16
	MOVQ R8, SI
	MOVQ $16, CX
size512_r2_f64_inv_stage5_loop:
	MOVQ $16, DX
size512_r2_f64_inv_stage5_inner:
	MOVUPD (SI), X0
	MOVUPD 256(SI), X1
	MOVQ $16, AX
	SUBQ DX, AX
	SHLQ $4, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8
	XORPD X15, X8

	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 256(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_inv_stage5_inner
	ADDQ $256, SI
	DECQ CX
	JNZ size512_r2_f64_inv_stage5_loop

	// Stage 6: size=64, half=32, step=8
	MOVQ R8, SI
	MOVQ $8, CX
size512_r2_f64_inv_stage6_loop:
	MOVQ $32, DX
size512_r2_f64_inv_stage6_inner:
	MOVUPD (SI), X0
	MOVUPD 512(SI), X1
	MOVQ $32, AX
	SUBQ DX, AX
	SHLQ $3, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8
	XORPD X15, X8

	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 512(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_inv_stage6_inner
	ADDQ $512, SI
	DECQ CX
	JNZ size512_r2_f64_inv_stage6_loop

	// Stage 7: size=128, half=64, step=4
	MOVQ R8, SI
	MOVQ $4, CX
size512_r2_f64_inv_stage7_loop:
	MOVQ $64, DX
size512_r2_f64_inv_stage7_inner:
	MOVUPD (SI), X0
	MOVUPD 1024(SI), X1
	MOVQ $64, AX
	SUBQ DX, AX
	SHLQ $2, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8
	XORPD X15, X8

	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 1024(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_inv_stage7_inner
	ADDQ $1024, SI
	DECQ CX
	JNZ size512_r2_f64_inv_stage7_loop

	// Stage 8: size=256, half=128, step=2
	MOVQ R8, SI
	MOVQ $2, CX
size512_r2_f64_inv_stage8_loop:
	MOVQ $128, DX
size512_r2_f64_inv_stage8_inner:
	MOVUPD (SI), X0
	MOVUPD 2048(SI), X1
	MOVQ $128, AX
	SUBQ DX, AX
	SHLQ $1, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8
	XORPD X15, X8

	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 2048(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_inv_stage8_inner
	ADDQ $2048, SI
	DECQ CX
	JNZ size512_r2_f64_inv_stage8_loop

	// Stage 9: size=512, half=256, step=1
	MOVQ R8, SI
	MOVQ $256, DX
size512_r2_f64_inv_stage9_inner:
	MOVUPD (SI), X0
	MOVUPD 4096(SI), X1
	MOVQ $256, AX
	SUBQ DX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8
	XORPD X15, X8

	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD ·maskNegLoPD(SB), X6
	ADDPD X6, X4
	MOVAPD X4, X1

	MOVAPD X0, X2
	ADDPD X1, X0
	SUBPD X1, X2
	MOVUPD X0, (SI)
	MOVUPD X2, 4096(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ size512_r2_f64_inv_stage9_inner

	// Scale by 1/512
	MOVSD ·fiveHundredTwelfth64(SB), X15
	SHUFPD $0x00, X15, X15
	MOVQ $512, CX
	MOVQ R8, SI
size512_r2_f64_inv_scale_loop:
	MOVUPD (SI), X0
	MULPD X15, X0
	MOVUPD X0, (SI)
	ADDQ $16, SI
	DECQ CX
	JNZ size512_r2_f64_inv_scale_loop

	// Copy to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size512_r2_f64_inv_done

	MOVQ $512, CX
	MOVQ R8, SI
size512_r2_f64_inv_copy_loop:
	MOVUPD (SI), X0
	MOVUPD X0, (R9)
	ADDQ $16, SI
	ADDQ $16, R9
	DECQ CX
	JNZ size512_r2_f64_inv_copy_loop

size512_r2_f64_inv_done:
	MOVB $1, ret+96(FP)
	RET

size512_r2_f64_inv_return_false:
	MOVB $0, ret+96(FP)
	RET

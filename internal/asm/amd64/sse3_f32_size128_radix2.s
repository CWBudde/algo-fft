//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-128 Radix-2 FFT Kernels for AMD64
// ===========================================================================

#include "textflag.h"

// Forward transform, size 128, complex64, radix-2
TEXT ·ForwardSSE3Size128Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size128Radix2(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $128
	JNE  fwd_ret_false

	CMPQ R8, R9
	JNE  fwd_use_dst
	MOVQ R11, R8

fwd_use_dst:
	// Bit-reversal permutation + Stage 1 (load bitrev directly in Stage 1/2 loop)

	// Stage 1 & 2 (Combined) - 32 blocks of 4
	MOVQ R8, SI
	MOVQ $32, CX
	MOVUPS ·maskNegHiPS(SB), X15

fwd_stage12_loop:
	MOVQ (R12), DX
	MOVSD (R9)(DX*8), X0
	MOVQ 8(R12), DX
	MOVSD (R9)(DX*8), X1
	MOVQ 16(R12), DX
	MOVSD (R9)(DX*8), X2
	MOVQ 24(R12), DX
	MOVSD (R9)(DX*8), X3
	ADDQ $32, R12
	// Stage 1
	MOVAPS X0, X8
	ADDPS  X1, X8
	MOVAPS X0, X9
	SUBPS  X1, X9
	MOVAPS X8, X0
	MOVAPS X9, X1
	MOVAPS X2, X8
	ADDPS  X3, X8
	MOVAPS X2, X9
	SUBPS  X3, X9
	MOVAPS X8, X2
	MOVAPS X9, X3
	// Stage 2
	MOVAPS X0, X8
	ADDPS  X2, X8
	MOVAPS X0, X9
	SUBPS  X2, X9
	MOVAPS X8, X0
	MOVAPS X9, X2
	MOVAPS X3, X10
	SHUFPS $0xB1, X10, X10
	XORPS  X15, X10          // t = X3 * -i
	MOVAPS X1, X8
	ADDPS  X10, X8
	MOVAPS X1, X9
	SUBPS  X10, X9
	MOVAPS X8, X1
	MOVAPS X9, X3
	MOVSD X0, (SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)
	ADDQ $32, SI
	DECQ CX
	JNZ  fwd_stage12_loop

	// Stage 3 (dist 4) - 16 blocks of 8
	MOVQ R8, SI
	MOVQ $16, CX
fwd_s3_loop:
	MOVQ $4, DX
fwd_s3_inner:
	MOVSD (SI), X0
	MOVSD 32(SI), X1
	MOVQ $4, AX
	SUBQ DX, AX              // k = 0..3
	SHLQ $4, AX              // k * 128/8 = k * 16
	MOVSD (R10)(AX*8), X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X11, X1
	MULPS X12, X13
	ADDSUBPS X13, X1
	MOVAPS X0, X2
	ADDPS X1, X0
	SUBPS X1, X2
	MOVSD X0, (SI)
	MOVSD X2, 32(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ fwd_s3_inner
	ADDQ $32, SI
	DECQ CX
	JNZ fwd_s3_loop

	// Stage 4 (dist 8) - 8 blocks of 16
	MOVQ R8, SI
	MOVQ $8, CX
fwd_s4_loop:
	MOVQ $8, DX
fwd_s4_inner:
	MOVSD (SI), X0
	MOVSD 64(SI), X1
	MOVQ $8, AX
	SUBQ DX, AX              // k = 0..7
	SHLQ $3, AX              // k * 128/16 = k * 8
	MOVSD (R10)(AX*8), X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X11, X1
	MULPS X12, X13
	ADDSUBPS X13, X1
	MOVAPS X0, X2
	ADDPS X1, X0
	SUBPS X1, X2
	MOVSD X0, (SI)
	MOVSD X2, 64(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ fwd_s4_inner
	ADDQ $64, SI
	DECQ CX
	JNZ fwd_s4_loop

	// Stage 5 (dist 16) - 4 blocks of 32
	MOVQ R8, SI
	MOVQ $4, CX
fwd_s5_loop:
	MOVQ $16, DX
fwd_s5_inner:
	MOVSD (SI), X0
	MOVSD 128(SI), X1
	MOVQ $16, AX
	SUBQ DX, AX              // k = 0..15
	SHLQ $2, AX              // k * 128/32 = k * 4
	MOVSD (R10)(AX*8), X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X11, X1
	MULPS X12, X13
	ADDSUBPS X13, X1
	MOVAPS X0, X2
	ADDPS X1, X0
	SUBPS X1, X2
	MOVSD X0, (SI)
	MOVSD X2, 128(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ fwd_s5_inner
	ADDQ $128, SI
	DECQ CX
	JNZ fwd_s5_loop

	// Stage 6 (dist 32) - 2 blocks of 64
	MOVQ R8, SI
	MOVQ $2, CX
fwd_s6_loop:
	MOVQ $32, DX
fwd_s6_inner:
	MOVSD (SI), X0
	MOVSD 256(SI), X1
	MOVQ $32, AX
	SUBQ DX, AX              // k = 0..31
	SHLQ $1, AX              // k * 128/64 = k * 2
	MOVSD (R10)(AX*8), X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X11, X1
	MULPS X12, X13
	ADDSUBPS X13, X1
	MOVAPS X0, X2
	ADDPS X1, X0
	SUBPS X1, X2
	MOVSD X0, (SI)
	MOVSD X2, 256(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ fwd_s6_inner
	ADDQ $256, SI
	DECQ CX
	JNZ fwd_s6_loop

	// Stage 7 (dist 64) - 1 block of 128
	MOVQ R8, SI
	MOVQ $64, DX
fwd_s7_inner:
	MOVSD (SI), X0
	MOVSD 512(SI), X1
	MOVQ $64, AX
	SUBQ DX, AX              // k = 0..63
	MOVSD (R10)(AX*8), X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X11, X1
	MULPS X12, X13
	ADDSUBPS X13, X1
	MOVAPS X0, X2
	ADDPS X1, X0
	SUBPS X1, X2
	MOVSD X0, (SI)
	MOVSD X2, 512(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ fwd_s7_inner

	// Copy to dst
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   fwd_done
	MOVQ $64, CX
	MOVQ R8, SI
fwd_copy_loop:
	MOVUPS (SI), X0
	MOVUPS X0, (R14)
	ADDQ $16, SI
	ADDQ $16, R14
	DECQ CX
	JNZ fwd_copy_loop

fwd_done:
	MOVB $1, ret+96(FP)
	RET

fwd_ret_false:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 128, complex64, radix-2
TEXT ·InverseSSE3Size128Radix2Complex64Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size128Radix2(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $128
	JNE  inv_ret_false

	CMPQ R8, R9
	JNE  inv_use_dst
	MOVQ R11, R8

inv_use_dst:
	// Bit-reversal permutation + Stage 1 (load bitrev directly in Stage 1/2 loop)

	// Stage 1 & 2
	MOVQ R8, SI
	MOVQ $32, CX
	MOVUPS ·maskNegLoPS(SB), X15 // for i
	MOVUPS ·maskNegHiPS(SB), X14 // for conjugation

inv_stage12_loop:
	MOVQ (R12), DX
	MOVSD (R9)(DX*8), X0
	MOVQ 8(R12), DX
	MOVSD (R9)(DX*8), X1
	MOVQ 16(R12), DX
	MOVSD (R9)(DX*8), X2
	MOVQ 24(R12), DX
	MOVSD (R9)(DX*8), X3
	ADDQ $32, R12
	// Stage 1
	MOVAPS X0, X8
	ADDPS  X1, X8
	MOVAPS X0, X9
	SUBPS  X1, X9
	MOVAPS X8, X0
	MOVAPS X9, X1
	MOVAPS X2, X8
	ADDPS  X3, X8
	MOVAPS X2, X9
	SUBPS  X3, X9
	MOVAPS X8, X2
	MOVAPS X9, X3
	// Stage 2 (w=[1, i])
	MOVAPS X0, X8
	ADDPS  X2, X8
	MOVAPS X0, X9
	SUBPS  X2, X9
	MOVAPS X8, X0
	MOVAPS X9, X2
	MOVAPS X3, X10
	SHUFPS $0xB1, X10, X10
	XORPS  X15, X10          // t = X3 * i
	MOVAPS X1, X8
	ADDPS  X10, X8
	MOVAPS X1, X9
	SUBPS  X10, X9
	MOVAPS X8, X1
	MOVAPS X9, X3
	MOVSD X0, (SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)
	ADDQ $32, SI
	DECQ CX
	JNZ  inv_stage12_loop

	// Stage 3 (dist 4)
	MOVQ R8, SI
	MOVQ $16, CX
inv_s3_loop:
	MOVQ $4, DX
inv_s3_inner:
	MOVSD (SI), X0
	MOVSD 32(SI), X1
	MOVQ $4, AX
	SUBQ DX, AX
	SHLQ $4, AX
	MOVSD (R10)(AX*8), X10
	XORPS X14, X10           // Conjugate
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X11, X1
	MULPS X12, X13
	ADDSUBPS X13, X1
	MOVAPS X0, X2
	ADDPS X1, X0
	SUBPS X1, X2
	MOVSD X0, (SI)
	MOVSD X2, 32(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ inv_s3_inner
	ADDQ $32, SI
	DECQ CX
	JNZ inv_s3_loop

	// Stage 4 (dist 8)
	MOVQ R8, SI
	MOVQ $8, CX
inv_s4_loop:
	MOVQ $8, DX
inv_s4_inner:
	MOVSD (SI), X0
	MOVSD 64(SI), X1
	MOVQ $8, AX
	SUBQ DX, AX
	SHLQ $3, AX
	MOVSD (R10)(AX*8), X10
	XORPS X14, X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X11, X1
	MULPS X12, X13
	ADDSUBPS X13, X1
	MOVAPS X0, X2
	ADDPS X1, X0
	SUBPS X1, X2
	MOVSD X0, (SI)
	MOVSD X2, 64(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ inv_s4_inner
	ADDQ $64, SI
	DECQ CX
	JNZ inv_s4_loop

	// Stage 5 (dist 16)
	MOVQ R8, SI
	MOVQ $4, CX
inv_s5_loop:
	MOVQ $16, DX
inv_s5_inner:
	MOVSD (SI), X0
	MOVSD 128(SI), X1
	MOVQ $16, AX
	SUBQ DX, AX
	SHLQ $2, AX
	MOVSD (R10)(AX*8), X10
	XORPS X14, X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X11, X1
	MULPS X12, X13
	ADDSUBPS X13, X1
	MOVAPS X0, X2
	ADDPS X1, X0
	SUBPS X1, X2
	MOVSD X0, (SI)
	MOVSD X2, 128(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ inv_s5_inner
	ADDQ $128, SI
	DECQ CX
	JNZ inv_s5_loop

	// Stage 6 (dist 32)
	MOVQ R8, SI
	MOVQ $2, CX
inv_s6_loop:
	MOVQ $32, DX
inv_s6_inner:
	MOVSD (SI), X0
	MOVSD 256(SI), X1
	MOVQ $32, AX
	SUBQ DX, AX
	SHLQ $1, AX
	MOVSD (R10)(AX*8), X10
	XORPS X14, X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X11, X1
	MULPS X12, X13
	ADDSUBPS X13, X1
	MOVAPS X0, X2
	ADDPS X1, X0
	SUBPS X1, X2
	MOVSD X0, (SI)
	MOVSD X2, 256(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ inv_s6_inner
	ADDQ $256, SI
	DECQ CX
	JNZ inv_s6_loop

	// Stage 7 (dist 64)
	MOVQ R8, SI
	MOVQ $64, DX
inv_s7_inner:
	MOVSD (SI), X0
	MOVSD 512(SI), X1
	MOVQ $64, AX
	SUBQ DX, AX
	MOVSD (R10)(AX*8), X10
	XORPS X14, X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X11, X1
	MULPS X12, X13
	ADDSUBPS X13, X1
	MOVAPS X0, X2
	ADDPS X1, X0
	SUBPS X1, X2
	MOVSD X0, (SI)
	MOVSD X2, 512(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ inv_s7_inner

	// Scale by 1/128
	MOVSS ·oneTwentyEighth32(SB), X15
	SHUFPS $0x00, X15, X15
	MOVQ $64, CX
	MOVQ R8, SI
inv_scale_loop:
	MOVUPS (SI), X0
	MULPS X15, X0
	MOVUPS X0, (SI)
	ADDQ $16, SI
	DECQ CX
	JNZ inv_scale_loop

	// Copy to dst
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   inv_done
	MOVQ $64, CX
	MOVQ R8, SI
inv_copy_loop:
	MOVUPS (SI), X0
	MOVUPS X0, (R14)
	ADDQ $16, SI
	ADDQ $16, R14
	DECQ CX
	JNZ inv_copy_loop

inv_done:
	MOVB $1, ret+96(FP)
	RET

inv_ret_false:
	MOVB $0, ret+96(FP)
	RET

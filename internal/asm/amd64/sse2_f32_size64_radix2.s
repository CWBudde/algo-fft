//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-64 Radix-2 FFT Kernels for AMD64
// ===========================================================================

#include "textflag.h"

// Forward transform, size 64, complex64, radix-2
TEXT ·ForwardSSE2Size64Radix2Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $64
	JNE  fwd_ret_false

	CMPQ R8, R9
	JNE  fwd_use_dst
	MOVQ R11, R8

fwd_use_dst:
	// Bit-reversal
	XORQ CX, CX
fwd_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVSD (R9)(DX*8), X0
	MOVSD X0, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $64
	JL   fwd_bitrev_loop

	// Stage 1 & 2 (Combined)
	MOVQ R8, SI
	MOVQ $16, CX
	MOVUPS ·maskNegHiPS(SB), X15

fwd_stage12_loop:
	MOVSD (SI), X0
	MOVSD 8(SI), X1
	MOVSD 16(SI), X2
	MOVSD 24(SI), X3
	// Stage 1 (stride 1, w=1)
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
	// Stage 2 (stride 2, w=[1, -i])
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

	// Stages 3..6 (Generic Radix-2 Loop)
	MOVQ $4, R14             // size = 4
	MOVQ $16, R15            // half = 2 -> but we use distance!
	// Actually, let's use the standard loop from generic SSE2
	// but adapted for n=64.
	
	// Stage 3: size=8, dist=4, groups=8
	// Stage 4: size=16, dist=8, groups=4
	// Stage 5: size=32, dist=16, groups=2
	// Stage 6: size=64, dist=32, groups=1

	// I'll unroll them individually for performance.
	
	// Stage 3 (dist 4)
	MOVQ R8, SI
	MOVQ $8, CX
fwd_s3_loop:
	MOVQ $4, DX
fwd_s3_inner:
	MOVSD (SI), X0
	MOVSD 32(SI), X1         // 4 * 8 = 32
	// twiddle[k * 64/8] = tw[k * 8]
	MOVQ $4, AX
	SUBQ DX, AX              // AX = 0, 1, 2, 3
	SHLQ $3, AX              // AX = 0, 8, 16, 24
	MOVSD (R10)(AX*8), X10
	// Complex mul
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
	ADDQ $32, SI             // Skip the 'dist' part
	DECQ CX
	JNZ fwd_s3_loop

	// Stage 4 (dist 8)
	MOVQ R8, SI
	MOVQ $4, CX
fwd_s4_loop:
	MOVQ $8, DX
fwd_s4_inner:
	MOVSD (SI), X0
	MOVSD 64(SI), X1
	// twiddle[k * 64/16] = tw[k * 4]
	MOVQ $8, AX
	SUBQ DX, AX
	SHLQ $2, AX              // AX = 0, 4, 8... 28
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

	// Stage 5 (dist 16)
	MOVQ R8, SI
	MOVQ $2, CX
fwd_s5_loop:
	MOVQ $16, DX
fwd_s5_inner:
	MOVSD (SI), X0
	MOVSD 128(SI), X1
	// twiddle[k * 64/32] = tw[k * 2]
	MOVQ $16, AX
	SUBQ DX, AX
	SHLQ $1, AX              // AX = 0, 2, 4... 30
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

	// Stage 6 (dist 32)
	MOVQ R8, SI
	MOVQ $32, DX
fwd_s6_inner:
	MOVSD (SI), X0
	MOVSD 256(SI), X1
	// twiddle[k * 64/64] = tw[k]
	MOVQ $32, AX
	SUBQ DX, AX
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

	// Copy to dst
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   fwd_done
	MOVQ $32, CX
	MOVQ R8, SI
fwd_copy_loop:
	MOVUPS (SI), X0
	MOVUPS X0, (R14)
	ADDQ $16, SI
	ADDQ $16, R14
	DECQ CX
	JNZ fwd_copy_loop

fwd_done:
	MOVB $1, ret+120(FP)
	RET

fwd_ret_false:
	MOVB $0, ret+120(FP)
	RET

// Inverse transform, size 64, complex64, radix-2
TEXT ·InverseSSE2Size64Radix2Complex64Asm(SB), NOSPLIT, $0-121
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $64
	JNE  inv_ret_false

	CMPQ R8, R9
	JNE  inv_use_dst
	MOVQ R11, R8

inv_use_dst:
	XORQ CX, CX
inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVSD (R9)(DX*8), X0
	MOVSD X0, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $64
	JL   inv_bitrev_loop

	// Stage 1 & 2
	MOVQ R8, SI
	MOVQ $16, CX
	MOVUPS ·maskNegLoPS(SB), X15 // for i
	MOVUPS ·maskNegHiPS(SB), X14 // for conjugation

inv_stage12_loop:
	MOVSD (SI), X0
	MOVSD 8(SI), X1
	MOVSD 16(SI), X2
	MOVSD 24(SI), X3
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
	MOVQ $8, CX
inv_s3_loop:
	MOVQ $4, DX
inv_s3_inner:
	MOVSD (SI), X0
	MOVSD 32(SI), X1
	MOVQ $4, AX
	SUBQ DX, AX
	SHLQ $3, AX
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
	MOVQ $4, CX
inv_s4_loop:
	MOVQ $8, DX
inv_s4_inner:
	MOVSD (SI), X0
	MOVSD 64(SI), X1
	MOVQ $8, AX
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
	MOVSD X2, 64(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ inv_s4_inner
	ADDQ $64, SI
	DECQ CX
	JNZ inv_s4_loop

	// Stage 5 (dist 16)
	MOVQ R8, SI
	MOVQ $2, CX
inv_s5_loop:
	MOVQ $16, DX
inv_s5_inner:
	MOVSD (SI), X0
	MOVSD 128(SI), X1
	MOVQ $16, AX
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
	MOVSD X2, 128(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ inv_s5_inner
	ADDQ $128, SI
	DECQ CX
	JNZ inv_s5_loop

	// Stage 6 (dist 32)
	MOVQ R8, SI
	MOVQ $32, DX
inv_s6_inner:
	MOVSD (SI), X0
	MOVSD 256(SI), X1
	MOVQ $32, AX
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
	MOVSD X2, 256(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ inv_s6_inner

	// Scale by 1/64
	MOVSS ·sixtyFourth32(SB), X15
	SHUFPS $0x00, X15, X15
	MOVQ $32, CX
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
	MOVQ $32, CX
	MOVQ R8, SI
inv_copy_loop:
	MOVUPS (SI), X0
	MOVUPS X0, (R14)
	ADDQ $16, SI
	ADDQ $16, R14
	DECQ CX
	JNZ inv_copy_loop

inv_done:
	MOVB $1, ret+120(FP)
	RET

inv_ret_false:
	MOVB $0, ret+120(FP)
	RET

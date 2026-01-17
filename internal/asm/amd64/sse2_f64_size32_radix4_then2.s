//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-32 Radix-4-then-2 FFT Kernels for AMD64 (complex128)
// ===========================================================================

#include "textflag.h"

// Forward transform, size 32, complex128, radix-4-then-2
TEXT ·ForwardSSE2Size32Radix4Then2Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size32Radix4Then2(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $32
	JNE  m24_32_f64_fwd_err

	CMPQ R8, R9
	JNE  m24_32_f64_fwd_use_dst
	MOVQ R11, R8

m24_32_f64_fwd_use_dst:
	// Bit-reversal
	XORQ CX, CX
m24_32_f64_fwd_bitrev_loop:
	MOVQ (R12)(CX*8), DX; SHLQ $4, DX; MOVUPD (R9)(DX*1), X0
	MOVQ CX, AX; SHLQ $4, AX; MOVUPD X0, (R8)(AX*1)
	INCQ CX
	CMPQ CX, $32
	JL   m24_32_f64_fwd_bitrev_loop

	// Stage 1: Radix-4 butterflies (stride 4) - 8 butterflies
	MOVQ R8, SI
	MOVQ $8, CX
	MOVUPS ·maskNegHiPD(SB), X15

m24_32_f64_fwd_stage1_loop:
	MOVUPD (SI), X0; MOVUPD 16(SI), X1; MOVUPD 32(SI), X2; MOVUPD 48(SI), X3
	MOVAPD X0, X8; ADDPD X2, X8; MOVAPD X0, X9; SUBPD X2, X9 // t0, t1
	MOVAPD X1, X10; ADDPD X3, X10; MOVAPD X1, X11; SUBPD X3, X11 // t2, t3
	MOVAPD X11, X12; SHUFPD $1, X12, X12; XORPD X15, X12 // t3 * -i
	MOVAPD X8, X0; ADDPD X10, X0
	MOVAPD X9, X1; ADDPD X12, X1
	MOVAPD X8, X2; SUBPD X10, X2
	MOVAPD X9, X3; SUBPD X12, X3
	MOVUPD X0, (SI); MOVUPD X1, 16(SI); MOVUPD X2, 32(SI); MOVUPD X3, 48(SI)
	ADDQ $64, SI
	DECQ CX
	JNZ  m24_32_f64_fwd_stage1_loop

	// Stage 2: Radix-4 butterflies (stride 1) - 8 butterflies
	// 2 groups of 4 butterflies
	MOVQ R8, SI
	MOVQ $2, CX
	MOVUPS ·maskNegLoPD(SB), X14

m24_32_f64_fwd_stage2_loop:
	XORQ DX, DX
m24_32_f64_fwd_stage2_inner:
	// Twiddles for n=16 (within the group): j, 2j, 3j but scaled by 2?
	// Wait, Stage 2 in size 32 radix-4-then-2 is a radix-4 stage.
	// distance is 4. twiddle step is 2.
	MOVQ DX, AX; SHLQ $1, AX; SHLQ $4, AX; MOVUPD (R10)(AX*1), X8 // w[2j]
	MOVQ DX, AX; SHLQ $2, AX; SHLQ $4, AX; MOVUPD (R10)(AX*1), X9 // w[4j]
	MOVQ DX, AX; SHLQ $1, AX; ADDQ DX, AX; SHLQ $1, AX; SHLQ $4, AX; MOVUPD (R10)(AX*1), X10 // w[6j]

	MOVQ DX, AX; SHLQ $4, AX; MOVUPD (SI)(AX*1), X0
	ADDQ $64, AX; MOVUPD (SI)(AX*1), X1
	ADDQ $64, AX; MOVUPD (SI)(AX*1), X2
	ADDQ $64, AX; MOVUPD (SI)(AX*1), X3

	// Complex mul
	MOVAPD X1, X4; UNPCKLPD X4, X4; MULPD X8, X4; MOVAPD X1, X5; UNPCKHPD X5, X5; MOVAPD X8, X6; SHUFPD $1, X6, X6; MULPD X5, X6; XORPD X14, X6; ADDPD X6, X4; MOVAPD X4, X1
	MOVAPD X2, X4; UNPCKLPD X4, X4; MULPD X9, X4; MOVAPD X2, X5; UNPCKHPD X5, X5; MOVAPD X9, X6; SHUFPD $1, X6, X6; MULPD X5, X6; XORPD X14, X6; ADDPD X6, X4; MOVAPD X4, X2
	MOVAPD X3, X4; UNPCKLPD X4, X4; MULPD X10, X4; MOVAPD X3, X5; UNPCKHPD X5, X5; MOVAPD X10, X6; SHUFPD $1, X6, X6; MULPD X5, X6; XORPD X14, X6; ADDPD X6, X4; MOVAPD X4, X3

	// Radix-4
	MOVAPD X0, X8; ADDPD X2, X8; MOVAPD X0, X9; SUBPD X2, X9
	MOVAPD X1, X10; ADDPD X3, X10; MOVAPD X1, X11; SUBPD X3, X11
	MOVAPD X11, X12; SHUFPD $1, X12, X12; XORPD X15, X12
	MOVAPD X8, X0; ADDPD X10, X0
	MOVAPD X9, X1; ADDPD X12, X1
	MOVAPD X8, X2; SUBPD X10, X2
	MOVAPD X9, X3; SUBPD X12, X3

	MOVQ DX, AX; SHLQ $4, AX; MOVUPD X0, (SI)(AX*1)
	ADDQ $64, AX; MOVUPD X1, (SI)(AX*1)
	ADDQ $64, AX; MOVUPD X2, (SI)(AX*1)
	ADDQ $64, AX; MOVUPD X3, (SI)(AX*1)

	INCQ DX
	CMPQ DX, $4
	JL m24_32_f64_fwd_stage2_inner
	ADDQ $256, SI
	DECQ CX
	JNZ m24_32_f64_fwd_stage2_loop

	// Stage 3: Radix-2 (stride 16) - 1 block of 32
	MOVQ R8, SI
	MOVQ $16, DX
m24_32_f64_fwd_stage3_loop:
	MOVUPD (SI), X0; MOVUPD 256(SI), X1
	MOVQ $16, AX; SUBQ DX, AX; SHLQ $4, AX; MOVUPD (R10)(AX*1), X10 // w[k]
	MOVAPD X1, X2; UNPCKLPD X2, X2; MULPD X10, X2; MOVAPD X1, X3; UNPCKHPD X3, X3; MOVAPD X10, X4; SHUFPD $1, X4, X4; MULPD X3, X4; XORPD X14, X4; ADDPD X4, X2; MOVAPD X2, X1
	MOVAPD X0, X2; ADDPD X1, X0; SUBPD X1, X2
	MOVUPD X0, (SI); MOVUPD X2, 256(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ m24_32_f64_fwd_stage3_loop

	// Copy to dst
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   m24_32_f64_fwd_done
	MOVQ $32, CX; MOVQ R8, SI; MOVQ R14, DI
m24_32_f64_fwd_copy:
	MOVUPD (SI), X0; MOVUPD X0, (DI); ADDQ $16, SI; ADDQ $16, DI; DECQ CX; JNZ m24_32_f64_fwd_copy

m24_32_f64_fwd_done:
	MOVB $1, ret+96(FP)
	RET
m24_32_f64_fwd_err:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 32, complex128, radix-4-then-2
TEXT ·InverseSSE2Size32Radix4Then2Complex128Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size32Radix4Then2(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $32
	JNE  m24_32_f64_inv_err

	CMPQ R8, R9
	JNE  m24_32_f64_inv_use_dst
	MOVQ R11, R8

m24_32_f64_inv_use_dst:
	// Bit-reversal
	XORQ CX, CX
m24_32_f64_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX; SHLQ $4, DX; MOVUPD (R9)(DX*1), X0
	MOVQ CX, AX; SHLQ $4, AX; MOVUPD X0, (R8)(AX*1)
	INCQ CX
	CMPQ CX, $32
	JL   m24_32_f64_inv_bitrev_loop

	// Stage 1
	MOVQ R8, SI
	MOVQ $8, CX
	MOVUPS ·maskNegLoPD(SB), X14

m24_32_f64_inv_stage1_loop:
	MOVUPD (SI), X0; MOVUPD 16(SI), X1; MOVUPD 32(SI), X2; MOVUPD 48(SI), X3
	MOVAPD X0, X8; ADDPD X2, X8; MOVAPD X0, X9; SUBPD X2, X9 // t0, t1
	MOVAPD X1, X10; ADDPD X3, X10; MOVAPD X1, X11; SUBPD X3, X11 // t2, t3
	MOVAPD X11, X12; SHUFPD $1, X12, X12; XORPD X14, X12 // t3 * i
	MOVAPD X8, X0; ADDPD X10, X0
	MOVAPD X9, X1; ADDPD X12, X1
	MOVAPD X8, X2; SUBPD X10, X2
	MOVAPD X9, X3; SUBPD X12, X3
	MOVUPD X0, (SI); MOVUPD X1, 16(SI); MOVUPD X2, 32(SI); MOVUPD X3, 48(SI)
	ADDQ $64, SI
	DECQ CX
	JNZ  m24_32_f64_inv_stage1_loop

	// Stage 2
	MOVQ R8, SI
	MOVQ $2, CX
	MOVUPS ·maskNegHiPD(SB), X15

m24_32_f64_inv_stage2_loop:
	XORQ DX, DX
m24_32_f64_inv_stage2_inner:
	MOVQ DX, AX; SHLQ $1, AX; SHLQ $4, AX; MOVUPD (R10)(AX*1), X8; XORPD X15, X8 // conj(w[2j])
	MOVQ DX, AX; SHLQ $1, AX; SHLQ $1, AX; SHLQ $4, AX; MOVUPD (R10)(AX*1), X9; XORPD X15, X9 // conj(w[4j])
	MOVQ DX, AX; SHLQ $1, AX; ADDQ DX, AX; SHLQ $1, AX; SHLQ $4, AX; MOVUPD (R10)(AX*1), X10; XORPD X15, X10 // conj(w[6j])

	MOVQ DX, AX; SHLQ $4, AX; MOVUPD (SI)(AX*1), X0
	ADDQ $64, AX; MOVUPD (SI)(AX*1), X1
	ADDQ $64, AX; MOVUPD (SI)(AX*1), X2
	ADDQ $64, AX; MOVUPD (SI)(AX*1), X3

	// Complex mul
	MOVAPD X1, X4; UNPCKLPD X4, X4; MULPD X8, X4; MOVAPD X1, X5; UNPCKHPD X5, X5; MOVAPD X8, X6; SHUFPD $1, X6, X6; MULPD X5, X6; XORPD ·maskNegLoPD(SB), X6; ADDPD X6, X4; MOVAPD X4, X1
	MOVAPD X2, X4; UNPCKLPD X4, X4; MULPD X9, X4; MOVAPD X2, X5; UNPCKHPD X5, X5; MOVAPD X9, X6; SHUFPD $1, X6, X6; MULPD X5, X6; XORPD ·maskNegLoPD(SB), X6; ADDPD X6, X4; MOVAPD X4, X2
	MOVAPD X3, X4; UNPCKLPD X4, X4; MULPD X10, X4; MOVAPD X3, X5; UNPCKHPD X5, X5; MOVAPD X10, X6; SHUFPD $1, X6, X6; MULPD X5, X6; XORPD ·maskNegLoPD(SB), X6; ADDPD X6, X4; MOVAPD X4, X3

	// Radix-4 butterfly (+i)
	MOVAPD X0, X8; ADDPD X2, X8; MOVAPD X0, X9; SUBPD X2, X9 // t0, t1
	MOVAPD X1, X10; ADDPD X3, X10; MOVAPD X1, X11; SUBPD X3, X11 // t2, t3
	MOVAPD X11, X12; SHUFPD $1, X12, X12; XORPD ·maskNegLoPD(SB), X12 // t3 * i
	MOVAPD X8, X0; ADDPD X10, X0
	MOVAPD X9, X1; ADDPD X12, X1
	MOVAPD X8, X2; SUBPD X10, X2
	MOVAPD X9, X3; SUBPD X12, X3

	MOVQ DX, AX; SHLQ $4, AX; MOVUPD X0, (SI)(AX*1)
	ADDQ $64, AX; MOVUPD X1, (SI)(AX*1)
	ADDQ $64, AX; MOVUPD X2, (SI)(AX*1)
	ADDQ $64, AX; MOVUPD X3, (SI)(AX*1)

	INCQ DX
	CMPQ DX, $4
	JL m24_32_f64_inv_stage2_inner
	ADDQ $256, SI
	DECQ CX
	JNZ m24_32_f64_inv_stage2_loop

	// Stage 3: Radix-2 combine
	MOVQ R8, SI
	MOVQ $16, DX
m24_32_f64_inv_stage3_loop:
	MOVUPD (SI), X0; MOVUPD 256(SI), X1
	MOVQ $16, AX; SUBQ DX, AX; SHLQ $4, AX; MOVUPD (R10)(AX*1), X10; XORPD X15, X10 // conj(w[k])
	MOVAPD X1, X2; UNPCKLPD X2, X2; MULPD X10, X2; MOVAPD X1, X3; UNPCKHPD X3, X3; MOVAPD X10, X4; SHUFPD $1, X4, X4; MULPD X3, X4; XORPD ·maskNegLoPD(SB), X4; ADDPD X4, X2; MOVAPD X2, X1
	MOVAPD X0, X2; ADDPD X1, X0; SUBPD X1, X2
	MOVUPD X0, (SI); MOVUPD X2, 256(SI)
	ADDQ $16, SI
	DECQ DX
	JNZ m24_32_f64_inv_stage3_loop

	// Scale by 1/32
	MOVSD ·thirtySecond64(SB), X15; SHUFPD $0, X15, X15
	MOVQ $32, CX; MOVQ R8, SI
m24_32_f64_inv_scale:
	MOVUPD (SI), X0; MULPD X15, X0; MOVUPD X0, (SI); ADDQ $16, SI; DECQ CX; JNZ m24_32_f64_inv_scale

	// Copy to dst
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   m24_32_f64_inv_done
	MOVQ $32, CX; MOVQ R8, SI; MOVQ R14, DI
m24_32_f64_inv_copy:
	MOVUPD (SI), X0; MOVUPD X0, (DI); ADDQ $16, SI; ADDQ $16, DI; DECQ CX; JNZ m24_32_f64_inv_copy

m24_32_f64_inv_done:
	MOVB $1, ret+96(FP)
	RET
m24_32_f64_inv_err:
	MOVB $0, ret+96(FP)
	RET

//go:build amd64

// ===========================================================================
// SSE2 Complex Scale Helpers for AMD64
// ===========================================================================

#include "textflag.h"

// func ScaleComplex64SSE2Asm(dst []complex64, scale float32)
TEXT ·ScaleComplex64SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ dst+8(FP), CX
	TESTQ CX, CX
	JZ    sse2_scale64_done

	MOVSS scale+24(FP), X0
	SHUFPS $0x00, X0, X0

	XORQ AX, AX

sse2_scale64_loop:
	MOVQ CX, R8
	SUBQ AX, R8
	CMPQ R8, $2
	JL   sse2_scale64_tail

	MOVQ AX, R9
	SHLQ $3, R9

	MOVUPS (DI)(R9*1), X1
	MULPS X0, X1
	MOVUPS X1, (DI)(R9*1)

	ADDQ $2, AX
	JMP  sse2_scale64_loop

sse2_scale64_tail:
	CMPQ AX, CX
	JGE  sse2_scale64_done

sse2_scale64_scalar:
	MOVQ AX, R9
	SHLQ $3, R9

	MOVSS (DI)(R9*1), X1
	MULSS X0, X1
	MOVSS X1, (DI)(R9*1)

	MOVSS 4(DI)(R9*1), X2
	MULSS X0, X2
	MOVSS X2, 4(DI)(R9*1)

	INCQ AX
	CMPQ AX, CX
	JL   sse2_scale64_scalar

sse2_scale64_done:
	RET

// func ScaleComplex128SSE2Asm(dst []complex128, scale float64)
TEXT ·ScaleComplex128SSE2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ dst+8(FP), CX
	TESTQ CX, CX
	JZ    sse2_scale128_done

	MOVSD scale+24(FP), X0
	SHUFPD $0x00, X0, X0

	XORQ AX, AX

sse2_scale128_loop:
	CMPQ AX, CX
	JGE  sse2_scale128_done

	MOVQ AX, R9
	SHLQ $4, R9

	MOVUPD (DI)(R9*1), X1
	MULPD X0, X1
	MOVUPD X1, (DI)(R9*1)

	INCQ AX
	JMP  sse2_scale128_loop

sse2_scale128_done:
	RET

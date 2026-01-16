//go:build amd64

// ===========================================================================
// AVX2 Complex Scale Helpers for AMD64
// ===========================================================================

#include "textflag.h"

// func ScaleComplex64AVX2Asm(dst []complex64, scale float32)
TEXT ·ScaleComplex64AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ dst+8(FP), CX
	TESTQ CX, CX
	JZ    scale64_done

	MOVSS scale+24(FP), X0
	VBROADCASTSS X0, Y0

	XORQ AX, AX

scale64_loop:
	MOVQ CX, R8
	SUBQ AX, R8
	CMPQ R8, $4
	JL   scale64_scalar

	MOVQ AX, R9
	SHLQ $3, R9

	VMOVUPS (DI)(R9*1), Y1
	VMULPS Y0, Y1, Y1
	VMOVUPS Y1, (DI)(R9*1)

	ADDQ $4, AX
	JMP  scale64_loop

scale64_scalar:
	CMPQ AX, CX
	JGE  scale64_done

scale64_scalar_loop:
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
	JL   scale64_scalar_loop

scale64_done:
	VZEROUPPER
	RET

// func ScaleComplex128AVX2Asm(dst []complex128, scale float64)
TEXT ·ScaleComplex128AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ dst+8(FP), CX
	TESTQ CX, CX
	JZ    scale128_done

	MOVSD scale+24(FP), X0
	VBROADCASTSD X0, Y0

	XORQ AX, AX

scale128_loop:
	MOVQ CX, R8
	SUBQ AX, R8
	CMPQ R8, $2
	JL   scale128_scalar

	MOVQ AX, R9
	SHLQ $4, R9

	VMOVUPD (DI)(R9*1), Y1
	VMULPD Y0, Y1, Y1
	VMOVUPD Y1, (DI)(R9*1)

	ADDQ $2, AX
	JMP  scale128_loop

scale128_scalar:
	CMPQ AX, CX
	JGE  scale128_done

scale128_scalar_loop:
	MOVQ AX, R9
	SHLQ $4, R9

	MOVSD (DI)(R9*1), X1
	MULSD X0, X1
	MOVSD X1, (DI)(R9*1)

	MOVSD 8(DI)(R9*1), X2
	MULSD X0, X2
	MOVSD X2, 8(DI)(R9*1)

	INCQ AX
	CMPQ AX, CX
	JL   scale128_scalar_loop

scale128_done:
	VZEROUPPER
	RET

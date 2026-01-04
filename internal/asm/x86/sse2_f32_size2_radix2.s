//go:build 386 && asm && !purego

#include "textflag.h"

// func ForwardSSE2Size2Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
TEXT ·ForwardSSE2Size2Radix2Complex64Asm(SB), NOSPLIT, $0-60
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	CMPL DX, $2
	JNE  err

	MOVSD (CX), X0
	MOVSD 8(CX), X1

	MOVAPS X0, X2
	ADDPS  X1, X0
	SUBPS  X1, X2

	MOVSD X0, (AX)
	MOVSD X2, 8(AX)

	MOVB $1, ret+60(FP)
	RET
err:
	MOVB $0, ret+60(FP)
	RET

// func InverseSSE2Size2Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
TEXT ·InverseSSE2Size2Radix2Complex64Asm(SB), NOSPLIT, $0-60
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	CMPL DX, $2
	JNE  err

	MOVSD (CX), X0
	MOVSD 8(CX), X1

	MOVAPS X0, X2
	ADDPS  X1, X0
	SUBPS  X1, X2

	// Scale by 0.5
	MOVSS ·half32(SB), X3
	SHUFPS $0, X3, X3
	MULPS X3, X0
	MULPS X3, X2

	MOVSD X0, (AX)
	MOVSD X2, 8(AX)

	MOVB $1, ret+60(FP)
	RET
err:
	MOVB $0, ret+60(FP)
	RET

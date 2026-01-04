//go:build 386 && asm && !purego

#include "textflag.h"

// func ForwardSSE2Size2Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
TEXT ·ForwardSSE2Size2Radix2Complex128Asm(SB), NOSPLIT, $0-60
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	CMPL DX, $2
	JNE  err

	MOVUPD (CX), X0
	MOVUPD 16(CX), X1

	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2

	MOVUPD X0, (AX)
	MOVUPD X2, 16(AX)

	MOVB $1, ret+60(FP)
	RET
err:
	MOVB $0, ret+60(FP)
	RET

// func InverseSSE2Size2Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
TEXT ·InverseSSE2Size2Radix2Complex128Asm(SB), NOSPLIT, $0-60
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	CMPL DX, $2
	JNE  err

	MOVUPD (CX), X0
	MOVUPD 16(CX), X1

	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2

	// Scale by 0.5
	MOVSD ·half64(SB), X3
	SHUFPD $0, X3, X3
	MULPD X3, X0
	MULPD X3, X2

	MOVUPD X0, (AX)
	MOVUPD X2, 16(AX)

	MOVB $1, ret+60(FP)
	RET
err:
	MOVB $0, ret+60(FP)
	RET

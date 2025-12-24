//go:build arm64 && fft_asm

#include "textflag.h"

TEXT ·forwardNEONComplex64Asm(SB), NOSPLIT|NOFRAME|ABIInternal, $0-120
	MOVD $0, R0
	RET

TEXT ·inverseNEONComplex64Asm(SB), NOSPLIT|NOFRAME|ABIInternal, $0-120
	MOVD $0, R0
	RET

TEXT ·forwardNEONComplex128Asm(SB), NOSPLIT|NOFRAME|ABIInternal, $0-120
	MOVD $0, R0
	RET

TEXT ·inverseNEONComplex128Asm(SB), NOSPLIT|NOFRAME|ABIInternal, $0-120
	MOVD $0, R0
	RET

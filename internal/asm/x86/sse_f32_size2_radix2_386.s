//go:build 386 && asm && !purego

// ===========================================================================
// SSE (SSE1) Size-2 FFT Kernels for 386 (complex64)
// ===========================================================================
//
// Radix-2 FFT kernel for size 2 using only SSE instructions.
//
// y0 = x0 + x1
// y1 = x0 - x1
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 2, complex64, radix-2
// func ForwardSSESize2Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
// ===========================================================================
TEXT ·ForwardSSESize2Radix2Complex64Asm(SB), NOSPLIT, $0-60
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	// Verify n == 2
	CMPL DX, $2
	JNE  fwd_err

	// Load x0, x1
	MOVUPS (CX), X0      // X0 = [x0, x1] (low=x0, high=x1)

	// Split x0 and x1
	MOVAPS X0, X1
	MOVHLPS X0, X1       // X1 = [x1, x1] (high of X0 to low of X1)
	
	// Butterfly
	// y0 = x0 + x1
	// y1 = x0 - x1
	MOVAPS X0, X2
	ADDPS  X1, X0        // X0 low = x0 + x1
	SUBPS  X1, X2        // X2 low = x0 - x1

	// Store
	MOVLPS X0, (AX)      // Store y0
	MOVLPS X2, 8(AX)     // Store y1

	MOVB $1, ret+60(FP)
	RET
fwd_err:
	MOVB $0, ret+60(FP)
	RET

// ===========================================================================
// Inverse transform, size 2, complex64, radix-2
// func InverseSSESize2Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
// ===========================================================================
TEXT ·InverseSSESize2Radix2Complex64Asm(SB), NOSPLIT, $0-60
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	CMPL DX, $2
	JNE  inv_err

	// Load x0, x1
	MOVUPS (CX), X0      // X0 = [x0, x1]

	// Split
	MOVAPS X0, X1
	MOVHLPS X0, X1       // X1 = [x1, x1]

	// Butterfly
	MOVAPS X0, X2
	ADDPS  X1, X0        // X0 low = x0 + x1
	SUBPS  X1, X2        // X2 low = x0 - x1

	// Scale by 0.5
	MOVSS ·half32(SB), X3
	SHUFPS $0x00, X3, X3 // Broadcast 0.5
	
	MULPS X3, X0
	MULPS X3, X2

	// Store
	MOVLPS X0, (AX)
	MOVLPS X2, 8(AX)

	MOVB $1, ret+60(FP)
	RET
inv_err:
	MOVB $0, ret+60(FP)
	RET

//go:build 386 && asm && !purego

// ===========================================================================
// SSE (SSE1) Size-4 FFT Kernels for 386 (complex64)
// ===========================================================================
//
// Fully-unrolled radix-4 FFT kernel for size 4 using only SSE instructions.
// (Compatible with Pentium III and newer, or older CPUs with SSE support).
//
// Radix-4 Butterfly:
//   t0 = x0 + x2
//   t1 = x0 - x2
//   t2 = x1 + x3
//   t3 = x1 - x3
//
//   y0 = t0 + t2
//   y1 = t1 + t3*(-i)
//   y2 = t0 - t2
//   y3 = t1 - t3*(-i)
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 4, complex64, radix-4
// func ForwardSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
// ===========================================================================
TEXT ·ForwardSSESize4Radix4Complex64Asm(SB), NOSPLIT, $0-64
	// Load parameters
	MOVL dst+0(FP), AX       // AX = dst pointer
	MOVL src+12(FP), CX      // CX = src pointer
	MOVL src+16(FP), DX      // DX = n (should be 4)

	// Verify n == 4
	CMPL DX, $4
	JNE  size4_sse_32_fwd_return_false

	// Validate all slice lengths >= 4
	MOVL dst+4(FP), DX
	CMPL DX, $4
	JL   size4_sse_32_fwd_return_false

	// Load x0, x1 into X0
	MOVUPS (CX), X0      // X0 = [x0, x1] (low=x0, high=x1)
	// Load x2, x3 into X1
	MOVUPS 16(CX), X1    // X1 = [x2, x3] (low=x2, high=x3)

	// Butterfly 1
	// t0 = x0 + x2, t1 = x0 - x2
	// t2 = x1 + x3, t3 = x1 - x3
	MOVAPS X0, X2        // X2 = [x0, x1]
	ADDPS  X1, X0        // X0 = [t0, t2] (packed)
	SUBPS  X1, X2        // X2 = [x0-x2, x1-x3] = [t1, t3] (packed)

	// Process X0 [t0, t2]
	// y0 = t0 + t2
	// y2 = t0 - t2
	MOVAPS X0, X3        // X3 = [t0, t2]
	MOVHLPS X0, X3       // X3 = [t2, t2] (high of X0 to low of X3)
	
	MOVAPS X0, X4        // X4 = [t0, ...]
	ADDPS  X3, X4        // X4 low = t0+t2 = y0
	
	MOVAPS X0, X5        // X5 = [t0, ...]
	SUBPS  X3, X5        // X5 low = t0-t2 = y2

	// Process X2 [t1, t3]
	// y1 = t1 + t3*(-i)
	// y3 = t1 - t3*(-i)
	MOVAPS X2, X6        // X6 = [t1, t3]
	MOVHLPS X2, X6       // X6 = [t3, t3] (high of X2 to low of X6)
	
	// Calculate t3 * (-i) = (b, -a)
	MOVAPS X6, X7
	SHUFPS $0xB1, X7, X7 // Swap real/imag: (a, b) -> (b, a)
	MOVUPS ·maskNegHiPS(SB), X1
	XORPS  X1, X7        // X7 = (b, -a) = t3*(-i)

	MOVAPS X2, X1        // X1 = [t1, ...]
	ADDPS  X7, X1        // X1 low = t1 + t3*(-i) = y1

	MOVAPS X2, X3        // X3 = [t1, ...] (Reuse X3)
	SUBPS  X7, X3        // X3 low = t1 - t3*(-i) = y3

	// Store results
	// X4 low = y0
	// X1 low = y1
	// X5 low = y2
	// X3 low = y3
	MOVLPS X4, (AX)
	MOVLPS X1, 8(AX)
	MOVLPS X5, 16(AX)
	MOVLPS X3, 24(AX)

	MOVB $1, ret+60(FP)
	RET

size4_sse_32_fwd_return_false:
	MOVB $0, ret+60(FP)
	RET

// ===========================================================================
// Inverse transform, size 4, complex64, radix-4
// func InverseSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
// ===========================================================================
TEXT ·InverseSSESize4Radix4Complex64Asm(SB), NOSPLIT, $0-64
	// Load parameters
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	// Verify n == 4
	CMPL DX, $4
	JNE  size4_sse_32_inv_return_false

	// Validate lengths
	MOVL dst+4(FP), DX
	CMPL DX, $4
	JL   size4_sse_32_inv_return_false

	// Load x0, x1
	MOVUPS (CX), X0      // X0 = [x0, x1]
	// Load x2, x3
	MOVUPS 16(CX), X1    // X1 = [x2, x3]

	// Butterfly 1
	// t0 = x0 + x2, t1 = x0 - x2
	// t2 = x1 + x3, t3 = x1 - x3
	MOVAPS X0, X2        // X2 = [x0, x1]
	ADDPS  X1, X0        // X0 = [t0, t2]
	SUBPS  X1, X2        // X2 = [t1, t3]

	// Process X0 [t0, t2]
	// y0 = t0 + t2
	// y2 = t0 - t2
	MOVAPS X0, X3
	MOVHLPS X0, X3       // X3 = [t2, ...]
	
	MOVAPS X0, X4        // X4 = [t0, ...]
	ADDPS  X3, X4        // X4 low = y0
	
	MOVAPS X0, X5        // X5 = [t0, ...]
	SUBPS  X3, X5        // X5 low = y2

	// Process X2 [t1, t3]
	// y1 = t1 + t3*(i)
	// y3 = t1 - t3*(i)
	// t3 * i = (a, b)*i = (-b, a)
	// Swap: (b, a). Negate low: (-b, a).
	MOVAPS X2, X6
	MOVHLPS X2, X6       // X6 = [t3, ...]
	
	MOVAPS X6, X7
	SHUFPS $0xB1, X7, X7 // Swap real/imag
	MOVUPS ·maskNegLoPS(SB), X1
	XORPS  X1, X7        // X7 = (-b, a) = t3*i

	MOVAPS X2, X1        // X1 = [t1, ...]
	ADDPS  X7, X1        // X1 low = y1
	
	MOVAPS X2, X3        // X3 = [t1, ...]
	SUBPS  X7, X3        // X3 low = y3

	// Scale by 1/4
	MOVSS ·quarter32(SB), X7
	SHUFPS $0x00, X7, X7 // Broadcast
	
	MULPS X7, X4  // y0
	MULPS X7, X1  // y1
	MULPS X7, X5  // y2
	MULPS X7, X3  // y3

	// Store
	MOVLPS X4, (AX)
	MOVLPS X1, 8(AX)
	MOVLPS X5, 16(AX)
	MOVLPS X3, 24(AX)

	MOVB $1, ret+60(FP)
	RET

size4_sse_32_inv_return_false:
	MOVB $0, ret+60(FP)
	RET

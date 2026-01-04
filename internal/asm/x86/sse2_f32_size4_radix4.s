//go:build 386 && asm && !purego

// ===========================================================================
// SSE2 Size-4 FFT Kernels for 386 (complex64)
// ===========================================================================
//
// Fully-unrolled radix-4 FFT kernel for size 4.
// Adapted for 386 (8 XMM registers).
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
// func ForwardSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
// ===========================================================================
TEXT ·ForwardSSE2Size4Radix4Complex64Asm(SB), NOSPLIT, $0-64
	// Load parameters
	MOVL dst+0(FP), AX       // AX = dst pointer
	MOVL src+12(FP), CX      // CX = src pointer
	MOVL src+16(FP), DX      // DX = n (should be 4)

	// Verify n == 4
	CMPL DX, $4
	JNE  size4_sse2_32_fwd_return_false

	// Validate all slice lengths >= 4
	MOVL dst+4(FP), DX
	CMPL DX, $4
	JL   size4_sse2_32_fwd_return_false

	MOVL twiddle+28(FP), DX
	CMPL DX, $4
	JL   size4_sse2_32_fwd_return_false

	MOVL scratch+40(FP), DX
	CMPL DX, $4
	JL   size4_sse2_32_fwd_return_false

	// Load x0..x3
	// Using X0..X3
	MOVSD (CX), X0
	MOVSD 8(CX), X1
	MOVSD 16(CX), X2
	MOVSD 24(CX), X3

	// t0 = x0 + x2
	// t1 = x0 - x2
	// We need to preserve x0 for t1, so copy x0 to X4
	MOVAPS X0, X4
	ADDPS  X2, X0    // X0 = t0
	SUBPS  X2, X4    // X4 = t1 (x0 - x2)
	// X2 is now free

	// t2 = x1 + x3
	// t3 = x1 - x3
	MOVAPS X1, X5
	ADDPS  X3, X1    // X1 = t2
	SUBPS  X3, X5    // X5 = t3
	// X3 is now free

	// Current: X0=t0, X1=t2, X4=t1, X5=t3

	// t3 * (-i) = swap(t3) then negate high lane
	MOVAPS X5, X6
	SHUFPS $0xB1, X6, X6
	MOVUPS ·maskNegHiPS(SB), X7
	XORPS  X7, X6    // X6 = t3*(-i)

	// y0 = t0 + t2
	// y2 = t0 - t2
	MOVAPS X0, X2
	ADDPS  X1, X2    // X2 = y0
	MOVAPS X0, X3
	SUBPS  X1, X3    // X3 = y2
	
	// y1 = t1 + t3*(-i)
	// y3 = t1 - t3*(-i)
	MOVAPS X4, X0
	ADDPS  X6, X0    // X0 = y1
	MOVAPS X4, X1
	SUBPS  X6, X1    // X1 = y3

	// Store results
	// dst layout: y0, y1, y2, y3
	// Regs: X2=y0, X0=y1, X3=y2, X1=y3
	MOVSD X2, (AX)
	MOVSD X0, 8(AX)
	MOVSD X3, 16(AX)
	MOVSD X1, 24(AX)

	MOVB $1, ret+60(FP)
	RET

size4_sse2_32_fwd_return_false:
	MOVB $0, ret+60(FP)
	RET

// ===========================================================================
// Inverse transform, size 4, complex64, radix-4
// ===========================================================================
TEXT ·InverseSSE2Size4Radix4Complex64Asm(SB), NOSPLIT, $0-64
	// Load parameters
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	// Verify n == 4
	CMPL DX, $4
	JNE  size4_sse2_32_inv_return_false

	// Validate lengths
	MOVL dst+4(FP), DX
	CMPL DX, $4
	JL   size4_sse2_32_inv_return_false

	MOVL twiddle+28(FP), DX
	CMPL DX, $4
	JL   size4_sse2_32_inv_return_false

	MOVL scratch+40(FP), DX
	CMPL DX, $4
	JL   size4_sse2_32_inv_return_false

	// Load x0..x3
	MOVSD (CX), X0
	MOVSD 8(CX), X1
	MOVSD 16(CX), X2
	MOVSD 24(CX), X3

	// t0 = x0 + x2
	// t1 = x0 - x2
	MOVAPS X0, X4
	ADDPS  X2, X0    // t0
	SUBPS  X2, X4    // t1

	// t2 = x1 + x3
	// t3 = x1 - x3
	MOVAPS X1, X5
	ADDPS  X3, X1    // t2
	SUBPS  X3, X5    // t3

	// t3 * (+i) = swap(t3) then negate low lane
	MOVAPS X5, X6
	SHUFPS $0xB1, X6, X6
	MOVUPS ·maskNegLoPS(SB), X7
	XORPS  X7, X6    // t3*i

	// y0, y2
	MOVAPS X0, X2
	ADDPS  X1, X2    // y0
	MOVAPS X0, X3
	SUBPS  X1, X3    // y2

	// y1, y3
	MOVAPS X4, X0
	ADDPS  X6, X0    // y1
	MOVAPS X4, X1
	SUBPS  X6, X1    // y3

	// Scale by 1/4
	// X2=y0, X0=y1, X3=y2, X1=y3
	MOVSS ·quarter32(SB), X7
	SHUFPS $0x00, X7, X7  // broadcast
	
	MULPS X7, X2
	MULPS X7, X0
	MULPS X7, X3
	MULPS X7, X1

	// Store
	MOVSD X2, (AX)
	MOVSD X0, 8(AX)
	MOVSD X3, 16(AX)
	MOVSD X1, 24(AX)

	MOVB $1, ret+60(FP)
	RET

size4_sse2_32_inv_return_false:
	MOVB $0, ret+60(FP)
	RET

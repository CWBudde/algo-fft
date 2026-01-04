//go:build 386 && asm && !purego

// ===========================================================================
// SSE2 Size-4 FFT Kernels for 386 (complex128)
// ===========================================================================
//
// Fully-unrolled radix-4 FFT kernel for size 4.
// Adapted for 386 (8 XMM registers).
//
// ===========================================================================

#include "textflag.h"

// func ForwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
TEXT ·ForwardSSE2Size4Radix4Complex128Asm(SB), NOSPLIT, $0-64
	// Load parameters
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	CMPL DX, $4
	JNE  size4_sse2_128_32_fwd_false

	// Load x0..x3 (complex128 = 16 bytes each)
	MOVUPD (CX), X0
	MOVUPD 16(CX), X1
	MOVUPD 32(CX), X2
	MOVUPD 48(CX), X3

	// t0 = x0 + x2, t1 = x0 - x2
	MOVAPD X0, X4
	ADDPD  X2, X0    // X0 = t0
	SUBPD  X2, X4    // X4 = t1
	
	// t2 = x1 + x3, t3 = x1 - x3
	MOVAPD X1, X5
	ADDPD  X3, X1    // X1 = t2
	SUBPD  X3, X5    // X5 = t3

	// t3 * (-i) = swap(t3) then negate high lane
	MOVAPD X5, X6
	SHUFPD $1, X6, X6
	XORPD  ·maskNegHiPD(SB), X6 // X6 = t3*(-i)

	// y0, y2
	MOVAPD X0, X2
	ADDPD  X1, X2    // X2 = y0
	SUBPD  X1, X0    // X0 = y2
	
	// y1, y3
	MOVAPD X4, X3
	ADDPD  X6, X3    // X3 = y1
	SUBPD  X6, X4    // X4 = y3

	// Store results: y0, y1, y2, y3
	// Regs: X2=y0, X3=y1, X0=y2, X4=y3
	MOVUPD X2, (AX)
	MOVUPD X3, 16(AX)
	MOVUPD X0, 32(AX)
	MOVUPD X4, 48(AX)

	MOVB $1, ret+60(FP)
	RET

size4_sse2_128_32_fwd_false:
	MOVB $0, ret+60(FP)
	RET

// func InverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
TEXT ·InverseSSE2Size4Radix4Complex128Asm(SB), NOSPLIT, $0-64
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	CMPL DX, $4
	JNE  size4_sse2_128_32_inv_false

	MOVUPD (CX), X0
	MOVUPD 16(CX), X1
	MOVUPD 32(CX), X2
	MOVUPD 48(CX), X3

	// t0, t1
	MOVAPD X0, X4
	ADDPD  X2, X0    // t0
	SUBPD  X2, X4    // t1
	
	// t2, t3
	MOVAPD X1, X5
	ADDPD  X3, X1    // t2
	SUBPD  X3, X5    // t3

	// t3 * (+i) = swap(t3) then negate low lane
	MOVAPD X5, X6
	SHUFPD $1, X6, X6
	XORPD  ·maskNegLoPD(SB), X6 // t3*i

	// y0, y2
	MOVAPD X0, X2
	ADDPD  X1, X2    // y0
	SUBPD  X1, X0    // y2
	
	// y1, y3
	MOVAPD X4, X3
	ADDPD  X6, X3    // y1
	SUBPD  X6, X4    // y3

	// Scale by 1/4
	MOVSD  ·quarter64(SB), X7
	SHUFPD $0, X7, X7
	
	MULPD X7, X2
	MULPD X7, X3
	MULPD X7, X0
	MULPD X7, X4

	MOVUPD X2, (AX)
	MOVUPD X3, 16(AX)
	MOVUPD X0, 32(AX)
	MOVUPD X4, 48(AX)

	MOVB $1, ret+60(FP)
	RET

size4_sse2_128_32_inv_false:
	MOVB $0, ret+60(FP)
	RET

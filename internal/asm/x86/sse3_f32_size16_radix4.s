//go:build 386 && asm && !purego

// ===========================================================================
// SSE2 Size-16 Radix-4 FFT Kernels for 386 (complex64)
// ===========================================================================
//
// Radix-4 DIT FFT kernels for size 16.
//
// ===========================================================================

#include "textflag.h"

// func ForwardSSE3Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
TEXT ·ForwardSSE3Size16Radix4Complex64Asm(SB), NOSPLIT, $64-64
	// Load parameters
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	// Verify n == 16
	CMPL DX, $16
	JNE  fwd_ret_false

	// Select working buffer
	CMPL AX, CX
	JNE  fwd_use_dst
	MOVL scratch+36(FP), AX // Use scratch if src == dst
	
fwd_use_dst:
	// Store working buffer ptr
	MOVL AX, 0(SP)

	// Bit reversal
	MOVL bitrev+48(FP), DX
	XORL SI, SI // index

bitrev_loop:
	MOVL (DX)(SI*4), BX
	MOVSD (CX)(BX*8), X0
	MOVL 0(SP), DI
	MOVSD X0, (DI)(SI*8)
	INCL SI
	CMPL SI, $16
	JL   bitrev_loop

	// ==================================================================
	// Stage 1: 4 butterflies, stride 1 (contiguous in bit-reversed)
	// ==================================================================
	XORL SI, SI

stage1_loop:
	MOVL 0(SP), DI
	LEAL (DI)(SI*8), BX // base pointer

	MOVSD 0(BX), X0
	MOVSD 8(BX), X1
	MOVSD 16(BX), X2
	MOVSD 24(BX), X3

	// t0 = x0 + x2, t1 = x0 - x2
	MOVAPS X0, X4
	ADDPS  X2, X0    // X0 = t0
	SUBPS  X2, X4    // X4 = t1
	
	// t2 = x1 + x3, t3 = x1 - x3
	MOVAPS X1, X5
	ADDPS  X3, X1    // X1 = t2
	SUBPS  X3, X5    // X5 = t3

	// (-i)*t3 = (im, -re)
	MOVAPS X5, X6
	SHUFPS $0xB1, X6, X6
	MOVUPS ·maskNegHiPS(SB), X7
	XORPS  X7, X6    // X6 = (-i)*t3

	// i*t3 = (-im, re)
	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	MOVUPS ·maskNegLoPS(SB), X2
	XORPS  X2, X7    // X7 = i*t3

	// y0 = t0 + t2
	MOVAPS X0, X2
	ADDPS  X1, X2
	
	// y1 = t1 + (-i)*t3
	MOVAPS X4, X3
	ADDPS  X6, X3

	// y2 = t0 - t2
	SUBPS  X1, X0 
	
	// y3 = t1 + i*t3
	ADDPS  X7, X4

	// Store
	MOVSD X2, 0(BX)
	MOVSD X3, 8(BX)
	MOVSD X0, 16(BX)
	MOVSD X4, 24(BX)

	ADDL $4, SI
	CMPL SI, $16
	JL   stage1_loop

	// ==================================================================
	// Stage 2: 4 butterflies, distance 4
	// ==================================================================
	XORL SI, SI

stage2_loop:
	MOVL 0(SP), DI
	LEAL (DI)(SI*8), BX // ptr to x0

	// x0 stays in X0
	MOVSD 0(BX), X0

	// x1 * w1 -> X1
	MOVSD 32(BX), X1
	MOVL twiddle+24(FP), CX
	MOVSD (CX)(SI*8), X4    // w1
	MOVAPS X4, X5
	SHUFPS $0x00, X5, X5    // w1.re
	MOVAPS X4, X6
	SHUFPS $0x55, X6, X6    // w1.im
	MOVAPS X1, X7
	SHUFPS $0xB1, X7, X7    // x1.swap
	MULPS  X5, X1           // x1 * w1.re
	MULPS  X6, X7           // x1_swp * w1.im
	ADDSUBPS X7, X1         // x1 = x1 * w1

	// x2 * w2 -> X2
	MOVSD 64(BX), X2
	MOVL SI, AX
	SHLL $1, AX
	MOVSD (CX)(AX*8), X4    // w2
	MOVAPS X4, X5
	SHUFPS $0x00, X5, X5
	MOVAPS X4, X6
	SHUFPS $0x55, X6, X6
	MOVAPS X2, X7
	SHUFPS $0xB1, X7, X7
	MULPS  X5, X2
	MULPS  X6, X7
	ADDSUBPS X7, X2

	// x3 * w3 -> X3
	MOVSD 96(BX), X3
	MOVL SI, AX
	IMULL $3, AX
	MOVSD (CX)(AX*8), X4    // w3
	MOVAPS X4, X5
	SHUFPS $0x00, X5, X5
	MOVAPS X4, X6
	SHUFPS $0x55, X6, X6
	MOVAPS X3, X7
	SHUFPS $0xB1, X7, X7
	MULPS  X5, X3
	MULPS  X6, X7
	ADDSUBPS X7, X3

	// Butterfly on X0, X1, X2, X3
	MOVAPS X0, X4
	ADDPS  X2, X0    // t0
	SUBPS  X2, X4    // t1
	MOVAPS X1, X5
	ADDPS  X3, X1    // t2
	SUBPS  X3, X5    // t3

	// (-i)*t3
	MOVAPS X5, X6
	SHUFPS $0xB1, X6, X6
	MOVUPS ·maskNegHiPS(SB), X7
	XORPS  X7, X6
	
	// i*t3
	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	MOVUPS ·maskNegLoPS(SB), X2
	XORPS  X2, X7

	// y0, y1, y2, y3
	MOVAPS X0, X2
	ADDPS  X1, X2    // y0
	MOVAPS X4, X3
	ADDPS  X6, X3    // y1
	SUBPS  X1, X0    // y2
	ADDPS  X7, X4    // y3

	// Store
	MOVSD X2, 0(BX)
	MOVSD X3, 32(BX)
	MOVSD X0, 64(BX)
	MOVSD X4, 96(BX)
	
	INCL SI
	CMPL SI, $4
	JL   stage2_loop

	// Copy results to dst if needed
	MOVL dst+0(FP), AX
	MOVL 0(SP), CX
	CMPL AX, CX
	JE   fwd_done
	
	XORL SI, SI
fwd_copy_loop:
	MOVUPS (CX)(SI*1), X0
	MOVUPS X0, (AX)(SI*1)
	ADDL $16, SI
	CMPL SI, $128
	JL   fwd_copy_loop

fwd_done:
	MOVB $1, ret+60(FP)
	RET

fwd_ret_false:
	MOVB $0, ret+60(FP)
	RET

// func InverseSSE3Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
TEXT ·InverseSSE3Size16Radix4Complex64Asm(SB), NOSPLIT, $64-64
	// Load parameters
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	// Verify n == 16
	CMPL DX, $16
	JNE  inv_ret_false

	// Select working buffer
	CMPL AX, CX
	JNE  inv_use_dst
	MOVL scratch+36(FP), AX
	
inv_use_dst:
	MOVL AX, 0(SP)

	// Bit reversal
	MOVL bitrev+48(FP), DX
	XORL SI, SI

inv_bitrev_loop:
	MOVL (DX)(SI*4), BX
	MOVSD (CX)(BX*8), X0
	MOVL 0(SP), DI
	MOVSD X0, (DI)(SI*8)
	INCL SI
	CMPL SI, $16
	JL   inv_bitrev_loop

	// Stage 1 (inv)
	XORL SI, SI

inv_stage1_loop:
	MOVL 0(SP), DI
	LEAL (DI)(SI*8), BX

	MOVSD 0(BX), X0
	MOVSD 8(BX), X1
	MOVSD 16(BX), X2
	MOVSD 24(BX), X3

	// Butterfly
	MOVAPS X0, X4
	ADDPS  X2, X0    // t0
	SUBPS  X2, X4    // t1
	MOVAPS X1, X5
	ADDPS  X3, X1    // t2
	SUBPS  X3, X5    // t3

	// i*t3
	MOVAPS X5, X6
	SHUFPS $0xB1, X6, X6
	MOVUPS ·maskNegLoPS(SB), X7
	XORPS  X7, X6

	// (-i)*t3
	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	MOVUPS ·maskNegHiPS(SB), X2
	XORPS  X2, X7

	// y0, y1, y2, y3
	MOVAPS X0, X2
	ADDPS  X1, X2
	MOVAPS X4, X3
	ADDPS  X6, X3
	SUBPS  X1, X0
	ADDPS  X7, X4

	MOVSD X2, 0(BX)
	MOVSD X3, 8(BX)
	MOVSD X0, 16(BX)
	MOVSD X4, 24(BX)

	ADDL $4, SI
	CMPL SI, $16
	JL   inv_stage1_loop

	// Stage 2 (inv)
	XORL SI, SI
	MOVUPS ·maskNegHiPS(SB), X0 
	MOVUPS X0, 16(SP) // conjugation mask

inv_stage2_loop:
	MOVL 0(SP), DI
	LEAL (DI)(SI*8), BX

	// x0
	MOVSD 0(BX), X0

	// x1 * conj(w1)
	MOVSD 32(BX), X1
	MOVL twiddle+24(FP), CX
	MOVSD (CX)(SI*8), X4
	XORPS 16(SP), X4        // conjugate
	MOVAPS X4, X5
	SHUFPS $0x00, X5, X5
	MOVAPS X4, X6
	SHUFPS $0x55, X6, X6
	MOVAPS X1, X7
	SHUFPS $0xB1, X7, X7
	MULPS  X5, X1
	MULPS  X6, X7
	ADDSUBPS X7, X1

	// x2 * conj(w2)
	MOVSD 64(BX), X2
	MOVL SI, AX
	SHLL $1, AX
	MOVSD (CX)(AX*8), X4
	XORPS 16(SP), X4
	MOVAPS X4, X5
	SHUFPS $0x00, X5, X5
	MOVAPS X4, X6
	SHUFPS $0x55, X6, X6
	MOVAPS X2, X7
	SHUFPS $0xB1, X7, X7
	MULPS  X5, X2
	MULPS  X6, X7
	ADDSUBPS X7, X2

	// x3 * conj(w3)
	MOVSD 96(BX), X3
	MOVL SI, AX
	IMULL $3, AX
	MOVSD (CX)(AX*8), X4
	XORPS 16(SP), X4
	MOVAPS X4, X5
	SHUFPS $0x00, X5, X5
	MOVAPS X4, X6
	SHUFPS $0x55, X6, X6
	MOVAPS X3, X7
	SHUFPS $0xB1, X7, X7
	MULPS  X5, X3
	MULPS  X6, X7
	ADDSUBPS X7, X3

	// Butterfly
	MOVAPS X0, X4
	ADDPS  X2, X0
	SUBPS  X2, X4
	MOVAPS X1, X5
	ADDPS  X3, X1
	SUBPS  X3, X5

	// i*t3
	MOVAPS X5, X6
	SHUFPS $0xB1, X6, X6
	MOVUPS ·maskNegLoPS(SB), X7
	XORPS  X7, X6

	// (-i)*t3
	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	MOVUPS ·maskNegHiPS(SB), X2
	XORPS  X2, X7

	// y0, y1, y2, y3
	MOVAPS X0, X2
	ADDPS  X1, X2
	MOVAPS X4, X3
	ADDPS  X6, X3
	SUBPS  X1, X0
	ADDPS  X7, X4

	MOVSD X2, 0(BX)
	MOVSD X3, 32(BX)
	MOVSD X0, 64(BX)
	MOVSD X4, 96(BX)
	
	INCL SI
	CMPL SI, $4
	JL   inv_stage2_loop

	// Scale and copy
	MOVL dst+0(FP), AX
	MOVL 0(SP), CX
	MOVSS ·sixteenth32(SB), X7
	SHUFPS $0x00, X7, X7
	
	XORL SI, SI
inv_scale_loop:
	MOVUPS (CX)(SI*1), X0
	MULPS  X7, X0
	MOVUPS X0, (AX)(SI*1)
	ADDL $16, SI
	CMPL SI, $128
	JL   inv_scale_loop

	MOVB $1, ret+60(FP)
	RET

inv_ret_false:
	MOVB $0, ret+60(FP)
	RET

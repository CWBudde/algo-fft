//go:build 386 && asm && !purego

// ===========================================================================
// SSE (SSE1) Size-16 Radix-4 FFT Kernels for 386 (complex64)
// ===========================================================================
//
// Fully-unrolled radix-4 FFT kernel for size 16 using only SSE instructions.
// (Compatible with Pentium III and newer, or older CPUs with SSE support).
// Note: Does NOT use ADDSUBPS (which is SSE3).
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// SSE1 Complex multiply macro:
// Computes result = val * twiddle where both are complex64
// Without ADDSUBPS, we need more operations:
//   (a+bi)(c+di) = (ac-bd) + (ad+bc)i
//
// Input: val in X1, twiddle in X4
// Output: result in X1
// Clobbers: X4, X5, X6, X7
// ===========================================================================

// func ForwardSSESize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
TEXT ·ForwardSSESize16Radix4Complex64Asm(SB), NOSPLIT, $128-64
	// Stack layout:
	//   SP+0..3:   working buffer ptr
	//   SP+4..7:   saved SI
	//   SP+8..15:  temp
	//   SP+16..31: maskNegHiPS
	//   SP+32..47: maskNegLoPS
	//   SP+48..127: temp storage for stage values

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
	MOVL scratch+36(FP), AX

fwd_use_dst:
	MOVL AX, 0(SP)

	// Cache masks on stack
	MOVUPS ·maskNegHiPS(SB), X0
	MOVUPS X0, 16(SP)
	MOVUPS ·maskNegLoPS(SB), X0
	MOVUPS X0, 32(SP)

	// Bit reversal
	MOVL bitrev+48(FP), DX
	XORL SI, SI

bitrev_loop:
	MOVL (DX)(SI*4), BX
	MOVSD (CX)(BX*8), X0
	MOVL 0(SP), DI
	MOVSD X0, (DI)(SI*8)
	INCL SI
	CMPL SI, $16
	JL   bitrev_loop

	// ==================================================================
	// Stage 1: 4 butterflies, stride 1
	// ==================================================================
	XORL SI, SI

stage1_loop:
	MOVL 0(SP), DI
	LEAL (DI)(SI*8), BX

	MOVSD 0(BX), X0      // x0
	MOVSD 8(BX), X1      // x1
	MOVSD 16(BX), X2     // x2
	MOVSD 24(BX), X3     // x3

	// t0 = x0 + x2, t1 = x0 - x2
	MOVAPS X0, X4
	ADDPS  X2, X0        // X0 = t0
	SUBPS  X2, X4        // X4 = t1

	// t2 = x1 + x3, t3 = x1 - x3
	MOVAPS X1, X5
	ADDPS  X3, X1        // X1 = t2
	SUBPS  X3, X5        // X5 = t3

	// (-i)*t3 = (im, -re)
	MOVAPS X5, X6
	SHUFPS $0xB1, X6, X6
	XORPS  16(SP), X6    // X6 = (-i)*t3

	// i*t3 = (-im, re)
	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	XORPS  32(SP), X7    // X7 = i*t3

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
	LEAL (DI)(SI*8), BX

	// x0 stays in X0
	MOVSD 0(BX), X0

	// x1 * w1 -> X1 (SSE1 complex multiply)
	MOVSD 32(BX), X1     // x1 = (a, b)
	MOVL twiddle+24(FP), CX
	MOVSD (CX)(SI*8), X4 // w1 = (c, d)

	// Broadcast c and d
	MOVAPS X4, X5
	SHUFPS $0x00, X5, X5 // X5 = (c, c)
	MOVAPS X4, X6
	SHUFPS $0x55, X6, X6 // X6 = (d, d)

	// (a,b) * (c,c) = (ac, bc)
	MOVAPS X1, X7
	MULPS  X5, X7        // X7 = (ac, bc)

	// (b, a) * (d, d) = (bd, ad)
	SHUFPS $0xB1, X1, X1 // X1 = (b, a)
	MULPS  X6, X1        // X1 = (bd, ad)

	// Result: (ac-bd, bc+ad)
	// Extract real: ac from X7[0], bd from X1[0]
	// Extract imag: bc from X7[1], ad from X1[1]
	MOVAPS X7, X4
	SUBPS  X1, X4        // X4 = (ac-bd, bc-ad) - need to fix imag
	ADDPS  X1, X7        // X7 = (ac+bd, bc+ad) - need to fix real

	// Combine: take low from X4 (real), high from X7 (imag)
	SHUFPS $0x05, X4, X7 // X7 = (X4[1], X4[0], X7[1], X7[0])
	SHUFPS $0xD8, X7, X7 // X7 = (ac-bd, bc+ad, ..., ...)
	MOVAPS X7, X1

	// x2 * w2 -> X2
	MOVSD 64(BX), X2
	MOVL SI, AX
	SHLL $1, AX
	MOVSD (CX)(AX*8), X4

	MOVAPS X4, X5
	SHUFPS $0x00, X5, X5
	MOVAPS X4, X6
	SHUFPS $0x55, X6, X6
	MOVAPS X2, X7
	MULPS  X5, X7
	SHUFPS $0xB1, X2, X2
	MULPS  X6, X2

	MOVAPS X7, X4
	SUBPS  X2, X4
	ADDPS  X2, X7
	SHUFPS $0x05, X4, X7
	SHUFPS $0xD8, X7, X7
	MOVAPS X7, X2

	// x3 * w3 -> X3
	MOVSD 96(BX), X3
	MOVL SI, AX
	IMULL $3, AX
	MOVSD (CX)(AX*8), X4

	MOVAPS X4, X5
	SHUFPS $0x00, X5, X5
	MOVAPS X4, X6
	SHUFPS $0x55, X6, X6
	MOVAPS X3, X7
	MULPS  X5, X7
	SHUFPS $0xB1, X3, X3
	MULPS  X6, X3

	MOVAPS X7, X4
	SUBPS  X3, X4
	ADDPS  X3, X7
	SHUFPS $0x05, X4, X7
	SHUFPS $0xD8, X7, X7
	MOVAPS X7, X3

	// Butterfly on X0, X1, X2, X3
	MOVAPS X0, X4
	ADDPS  X2, X0        // t0
	SUBPS  X2, X4        // t1
	MOVAPS X1, X5
	ADDPS  X3, X1        // t2
	SUBPS  X3, X5        // t3

	// (-i)*t3
	MOVAPS X5, X6
	SHUFPS $0xB1, X6, X6
	XORPS  16(SP), X6

	// i*t3
	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	XORPS  32(SP), X7

	// y0, y1, y2, y3
	MOVAPS X0, X2
	ADDPS  X1, X2        // y0
	MOVAPS X4, X3
	ADDPS  X6, X3        // y1
	SUBPS  X1, X0        // y2
	ADDPS  X7, X4        // y3

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

// func InverseSSESize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
TEXT ·InverseSSESize16Radix4Complex64Asm(SB), NOSPLIT, $128-64
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

	// Cache masks on stack
	MOVUPS ·maskNegHiPS(SB), X0
	MOVUPS X0, 16(SP)
	MOVUPS ·maskNegLoPS(SB), X0
	MOVUPS X0, 32(SP)

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

	// ==================================================================
	// Stage 1 (inv): 4 butterflies, stride 1
	// For inverse FFT, use +i instead of -i
	// ==================================================================
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
	ADDPS  X2, X0        // t0
	SUBPS  X2, X4        // t1
	MOVAPS X1, X5
	ADDPS  X3, X1        // t2
	SUBPS  X3, X5        // t3

	// i*t3 (for inverse)
	MOVAPS X5, X6
	SHUFPS $0xB1, X6, X6
	XORPS  32(SP), X6

	// (-i)*t3
	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	XORPS  16(SP), X7

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

	// ==================================================================
	// Stage 2 (inv): with conjugate twiddles
	// ==================================================================
	XORL SI, SI

inv_stage2_loop:
	MOVL 0(SP), DI
	LEAL (DI)(SI*8), BX

	// x0
	MOVSD 0(BX), X0

	// x1 * conj(w1) -> X1
	MOVSD 32(BX), X1
	MOVL twiddle+24(FP), CX
	MOVSD (CX)(SI*8), X4
	XORPS 16(SP), X4     // conjugate (negate imag)

	MOVAPS X4, X5
	SHUFPS $0x00, X5, X5
	MOVAPS X4, X6
	SHUFPS $0x55, X6, X6
	MOVAPS X1, X7
	MULPS  X5, X7
	SHUFPS $0xB1, X1, X1
	MULPS  X6, X1

	MOVAPS X7, X4
	SUBPS  X1, X4
	ADDPS  X1, X7
	SHUFPS $0x05, X4, X7
	SHUFPS $0xD8, X7, X7
	MOVAPS X7, X1

	// x2 * conj(w2) -> X2
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
	MULPS  X5, X7
	SHUFPS $0xB1, X2, X2
	MULPS  X6, X2

	MOVAPS X7, X4
	SUBPS  X2, X4
	ADDPS  X2, X7
	SHUFPS $0x05, X4, X7
	SHUFPS $0xD8, X7, X7
	MOVAPS X7, X2

	// x3 * conj(w3) -> X3
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
	MULPS  X5, X7
	SHUFPS $0xB1, X3, X3
	MULPS  X6, X3

	MOVAPS X7, X4
	SUBPS  X3, X4
	ADDPS  X3, X7
	SHUFPS $0x05, X4, X7
	SHUFPS $0xD8, X7, X7
	MOVAPS X7, X3

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
	XORPS  32(SP), X6

	// (-i)*t3
	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	XORPS  16(SP), X7

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

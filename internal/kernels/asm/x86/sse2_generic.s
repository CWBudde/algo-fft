//go:build 386 && fft_asm && !purego

// =====================================================================
// SSE2-optimized FFT Assembly for 386
// =====================================================================
//
// This file provides SSE2-based scalar butterflies for complex64 DIT FFTs.
// It targets 32-bit x86 with SSE2 support (GOARCH=386).
//
// =====================================================================

#include "textflag.h"

// func forwardSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
TEXT ·ForwardSSE2Complex64Asm(SB), NOSPLIT, $36-61
	// ----------------------------------------------------------------
	// Load parameters and validate inputs
	// ----------------------------------------------------------------
	MOVL dst+0(FP), DI
	MOVL src+12(FP), SI
	MOVL twiddle+24(FP), BX
	MOVL scratch+36(FP), DX
	MOVL bitrev+48(FP), BP
	MOVL src+16(FP), AX // n

	MOVL DI, 0(SP)
	MOVL SI, 4(SP)
	MOVL BX, 8(SP)
	MOVL DX, 12(SP)
	MOVL BP, 16(SP)
	MOVL AX, 20(SP)

	TESTL AX, AX
	JZ   sse2_return_true

	// Validate slice lengths >= n
	MOVL dst+4(FP), CX
	CMPL CX, AX
	JL   sse2_return_false
	MOVL twiddle+28(FP), CX
	CMPL CX, AX
	JL   sse2_return_false
	MOVL scratch+40(FP), CX
	CMPL CX, AX
	JL   sse2_return_false
	MOVL bitrev+52(FP), CX
	CMPL CX, AX
	JL   sse2_return_false

	// Trivial n=1 case
	CMPL AX, $1
	JNE  sse2_check_power
	MOVL (SI), CX
	MOVL 4(SI), DX
	MOVL CX, (DI)
	MOVL DX, 4(DI)
	JMP  sse2_return_true

sse2_check_power:
	// Verify power of two: (n & (n-1)) == 0
	MOVL AX, CX
	LEAL -1(CX), DX
	ANDL DX, CX
	JNZ  sse2_return_false

	// Select working buffer
	CMPL DI, SI
	JNE  sse2_use_dst
	MOVL 12(SP), DI // scratch

sse2_use_dst:
	MOVL DI, 24(SP) // work pointer

	// ----------------------------------------------------------------
	// Bit-reversal permutation
	// ----------------------------------------------------------------
	XORL CX, CX

sse2_bitrev_loop:
	CMPL CX, 20(SP)
	JGE  sse2_bitrev_done
	MOVL 16(SP), BP
	MOVL (BP)(CX*4), DX
	MOVL 4(SP), SI
	MOVL (SI)(DX*8), AX
	MOVL 4(SI)(DX*8), BX
	MOVL 24(SP), DI
	MOVL AX, (DI)(CX*8)
	MOVL BX, 4(DI)(CX*8)
	INCL CX
	JMP  sse2_bitrev_loop

sse2_bitrev_done:
	// ----------------------------------------------------------------
	// DIT butterfly stages (scalar SSE)
	// ----------------------------------------------------------------
	MOVL $2, SI // size

sse2_size_loop:
	MOVL 20(SP), AX
	CMPL SI, AX
	JG   sse2_transform_done

	MOVL SI, DI
	SHRL $1, DI
	MOVL DI, 28(SP) // half

	MOVL 20(SP), AX
	XORL DX, DX
	DIVL SI
	MOVL AX, 32(SP) // step

	XORL CX, CX // base

sse2_base_loop:
	CMPL CX, 20(SP)
	JGE  sse2_next_size

	// Use vector path when twiddles are contiguous (step == 1) and half >= 2
	MOVL 32(SP), AX
	CMPL AX, $1
	JNE  sse2_scalar_butterflies
	MOVL 28(SP), AX
	CMPL AX, $2
	JL   sse2_scalar_butterflies

	XORL BP, BP // j

sse2_vec_loop:
	MOVL 28(SP), AX
	SUBL BP, AX
	CMPL AX, $2
	JL   sse2_scalar_remainder

	// offset1 = (base + j) * 8
	MOVL CX, DI
	ADDL BP, DI
	SHLL $3, DI

	// offset2 = offset1 + half*8
	MOVL 28(SP), BX
	SHLL $3, BX
	MOVL DI, DX
	ADDL BX, DX

	// Load a and b (2 complex64 each)
	MOVL 24(SP), AX
	MOVUPS (AX)(DI*1), X0
	MOVUPS (AX)(DX*1), X1

	// Load twiddles (2 complex64, contiguous)
	MOVL BP, AX
	SHLL $3, AX
	MOVL 8(SP), BX
	MOVUPS (BX)(AX*1), X2

	// t = w * b (SSE2, 2 complex64)
	MOVAPS X1, X3
	MULPS  X2, X3            // prod1 = b * w

	MOVAPS X2, X4
	SHUFPS $0xB1, X4, X4     // w_shuf

	MOVAPS X1, X5
	MULPS  X4, X5            // prod2 = b * w_shuf

	MOVAPS X3, X6
	SHUFPS $0xB1, X6, X6
	SUBPS  X6, X3            // real = prod1 - shuf(prod1)

	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	ADDPS  X7, X5            // imag = prod2 + shuf(prod2)

	SHUFPS $0x88, X3, X3     // real lanes
	SHUFPS $0x88, X5, X5     // imag lanes
	UNPCKLPS X5, X3          // t = [r0,i0,r1,i1]

	// Butterfly
	MOVAPS X0, X6
	ADDPS  X3, X6
	MOVAPS X0, X7
	SUBPS  X3, X7

	// Store results
	MOVL 24(SP), AX
	MOVUPS X6, (AX)(DI*1)
	MOVUPS X7, (AX)(DX*1)

	ADDL $2, BP
	JMP  sse2_vec_loop

sse2_scalar_remainder:
	CMPL BP, 28(SP)
	JGE  sse2_next_base

sse2_scalar_loop:
	CMPL BP, 28(SP)
	JGE  sse2_next_base

	// offset1 = (base + j) * 8
	MOVL CX, DI
	ADDL BP, DI
	SHLL $3, DI

	// offset2 = offset1 + half*8
	MOVL 28(SP), BX
	SHLL $3, BX
	MOVL DI, DX
	ADDL BX, DX

	// Load a and b
	MOVL 24(SP), AX
	MOVSS (AX)(DI*1), X0
	MOVSS 4(AX)(DI*1), X1
	MOVSS (AX)(DX*1), X2
	MOVSS 4(AX)(DX*1), X3

	// Load twiddle w = twiddle[j*step]
	MOVL BP, AX
	IMULL 32(SP), AX
	SHLL $3, AX
	MOVL 8(SP), BX
	MOVSS (BX)(AX*1), X4
	MOVSS 4(BX)(AX*1), X5

	// t.real = br*wr - bi*wi
	MOVAPS X2, X6
	MULSS  X4, X6
	MOVAPS X3, X7
	MULSS  X5, X7
	SUBSS  X7, X6

	// t.imag = br*wi + bi*wr
	MOVAPS X2, X7
	MULSS  X5, X7
	MOVAPS X3, X2
	MULSS  X4, X2
	ADDSS  X2, X7

	// a' = a + t, b' = a - t
	MOVAPS X0, X2
	ADDSS  X6, X2
	MOVAPS X1, X3
	ADDSS  X7, X3
	MOVAPS X0, X4
	SUBSS  X6, X4
	MOVAPS X1, X5
	SUBSS  X7, X5

	// Store results
	MOVL 24(SP), AX
	MOVSS X2, (AX)(DI*1)
	MOVSS X3, 4(AX)(DI*1)
	MOVSS X4, (AX)(DX*1)
	MOVSS X5, 4(AX)(DX*1)

	INCL BP
	JMP  sse2_scalar_loop

sse2_scalar_butterflies:
	XORL BP, BP
	JMP  sse2_scalar_loop

sse2_next_base:
	ADDL SI, CX
	JMP  sse2_base_loop

sse2_next_size:
	SHLL $1, SI
	JMP  sse2_size_loop

sse2_transform_done:
	// Copy back if work != dst
	MOVL 0(SP), AX
	MOVL 24(SP), DI
	CMPL AX, DI
	JE   sse2_return_true

	XORL CX, CX

sse2_copy_loop:
	CMPL CX, 20(SP)
	JGE  sse2_return_true
	MOVL (DI)(CX*8), BX
	MOVL 4(DI)(CX*8), DX
	MOVL BX, (AX)(CX*8)
	MOVL DX, 4(AX)(CX*8)
	INCL CX
	JMP  sse2_copy_loop

sse2_return_true:
	MOVB $1, ret+60(FP)
	RET

sse2_return_false:
	MOVB $0, ret+60(FP)
	RET

// func inverseSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
TEXT ·InverseSSE2Complex64Asm(SB), NOSPLIT, $36-61
	// ----------------------------------------------------------------
	// Load parameters and validate inputs
	// ----------------------------------------------------------------
	MOVL dst+0(FP), DI
	MOVL src+12(FP), SI
	MOVL twiddle+24(FP), BX
	MOVL scratch+36(FP), DX
	MOVL bitrev+48(FP), BP
	MOVL src+16(FP), AX // n

	MOVL DI, 0(SP)
	MOVL SI, 4(SP)
	MOVL BX, 8(SP)
	MOVL DX, 12(SP)
	MOVL BP, 16(SP)
	MOVL AX, 20(SP)

	TESTL AX, AX
	JZ   inv_sse2_return_true

	// Validate slice lengths >= n
	MOVL dst+4(FP), CX
	CMPL CX, AX
	JL   inv_sse2_return_false
	MOVL twiddle+28(FP), CX
	CMPL CX, AX
	JL   inv_sse2_return_false
	MOVL scratch+40(FP), CX
	CMPL CX, AX
	JL   inv_sse2_return_false
	MOVL bitrev+52(FP), CX
	CMPL CX, AX
	JL   inv_sse2_return_false

	// Trivial n=1 case
	CMPL AX, $1
	JNE  inv_sse2_check_power
	MOVL (SI), CX
	MOVL 4(SI), DX
	MOVL CX, (DI)
	MOVL DX, 4(DI)
	JMP  inv_sse2_return_true

inv_sse2_check_power:
	// Verify power of two
	MOVL AX, CX
	LEAL -1(CX), DX
	ANDL DX, CX
	JNZ  inv_sse2_return_false

	// Select working buffer
	CMPL DI, SI
	JNE  inv_sse2_use_dst
	MOVL 12(SP), DI // scratch

inv_sse2_use_dst:
	MOVL DI, 24(SP) // work pointer

	// ----------------------------------------------------------------
	// Bit-reversal permutation
	// ----------------------------------------------------------------
	XORL CX, CX

inv_sse2_bitrev_loop:
	CMPL CX, 20(SP)
	JGE  inv_sse2_bitrev_done
	MOVL 16(SP), BP
	MOVL (BP)(CX*4), DX
	MOVL 4(SP), SI
	MOVL (SI)(DX*8), AX
	MOVL 4(SI)(DX*8), BX
	MOVL 24(SP), DI
	MOVL AX, (DI)(CX*8)
	MOVL BX, 4(DI)(CX*8)
	INCL CX
	JMP  inv_sse2_bitrev_loop

inv_sse2_bitrev_done:
	// ----------------------------------------------------------------
	// DIT butterfly stages with conjugate twiddles
	// ----------------------------------------------------------------
	MOVL $2, SI // size

inv_sse2_size_loop:
	MOVL 20(SP), AX
	CMPL SI, AX
	JG   inv_sse2_transform_done

	MOVL SI, DI
	SHRL $1, DI
	MOVL DI, 28(SP) // half

	MOVL 20(SP), AX
	XORL DX, DX
	DIVL SI
	MOVL AX, 32(SP) // step

	XORL CX, CX // base

inv_sse2_base_loop:
	CMPL CX, 20(SP)
	JGE  inv_sse2_next_size

	// Use vector path when twiddles are contiguous (step == 1) and half >= 2
	MOVL 32(SP), AX
	CMPL AX, $1
	JNE  inv_sse2_scalar_butterflies
	MOVL 28(SP), AX
	CMPL AX, $2
	JL   inv_sse2_scalar_butterflies

	XORL BP, BP // j

inv_sse2_vec_loop:
	MOVL 28(SP), AX
	SUBL BP, AX
	CMPL AX, $2
	JL   inv_sse2_scalar_remainder

	// offset1 = (base + j) * 8
	MOVL CX, DI
	ADDL BP, DI
	SHLL $3, DI

	// offset2 = offset1 + half*8
	MOVL 28(SP), BX
	SHLL $3, BX
	MOVL DI, DX
	ADDL BX, DX

	// Load a and b (2 complex64 each)
	MOVL 24(SP), AX
	MOVUPS (AX)(DI*1), X0
	MOVUPS (AX)(DX*1), X1

	// Load twiddles (2 complex64, contiguous)
	MOVL BP, AX
	SHLL $3, AX
	MOVL 8(SP), BX
	MOVUPS (BX)(AX*1), X2

	// t = conj(w) * b (SSE2, 2 complex64)
	MOVAPS X1, X3
	MULPS  X2, X3            // prod1 = b * w

	MOVAPS X2, X4
	SHUFPS $0xB1, X4, X4     // w_shuf

	MOVAPS X1, X5
	MULPS  X4, X5            // prod2 = b * w_shuf

	MOVAPS X3, X6
	SHUFPS $0xB1, X6, X6
	ADDPS  X6, X3            // real = prod1 + shuf(prod1)

	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	SUBPS  X5, X7            // imag = shuf(prod2) - prod2

	SHUFPS $0x88, X3, X3     // real lanes
	SHUFPS $0x88, X7, X7     // imag lanes
	UNPCKLPS X7, X3          // t = [r0,i0,r1,i1]

	// Butterfly
	MOVAPS X0, X6
	ADDPS  X3, X6
	MOVAPS X0, X7
	SUBPS  X3, X7

	// Store results
	MOVL 24(SP), AX
	MOVUPS X6, (AX)(DI*1)
	MOVUPS X7, (AX)(DX*1)

	ADDL $2, BP
	JMP  inv_sse2_vec_loop

inv_sse2_scalar_remainder:
	CMPL BP, 28(SP)
	JGE  inv_sse2_next_base

inv_sse2_scalar_loop:
	CMPL BP, 28(SP)
	JGE  inv_sse2_next_base

	// offset1 = (base + j) * 8
	MOVL CX, DI
	ADDL BP, DI
	SHLL $3, DI

	// offset2 = offset1 + half*8
	MOVL 28(SP), BX
	SHLL $3, BX
	MOVL DI, DX
	ADDL BX, DX

	// Load a and b
	MOVL 24(SP), AX
	MOVSS (AX)(DI*1), X0
	MOVSS 4(AX)(DI*1), X1
	MOVSS (AX)(DX*1), X2
	MOVSS 4(AX)(DX*1), X3

	// Load twiddle w = twiddle[j*step]
	MOVL BP, AX
	IMULL 32(SP), AX
	SHLL $3, AX
	MOVL 8(SP), BX
	MOVSS (BX)(AX*1), X4
	MOVSS 4(BX)(AX*1), X5

	// t.real = br*wr + bi*wi
	MOVAPS X2, X6
	MULSS  X4, X6
	MOVAPS X3, X7
	MULSS  X5, X7
	ADDSS  X7, X6

	// t.imag = bi*wr - br*wi
	MOVAPS X3, X7
	MULSS  X4, X7
	MOVAPS X2, X2
	MULSS  X5, X2
	SUBSS  X2, X7

	// a' = a + t, b' = a - t
	MOVAPS X0, X2
	ADDSS  X6, X2
	MOVAPS X1, X3
	ADDSS  X7, X3
	MOVAPS X0, X4
	SUBSS  X6, X4
	MOVAPS X1, X5
	SUBSS  X7, X5

	// Store results
	MOVL 24(SP), AX
	MOVSS X2, (AX)(DI*1)
	MOVSS X3, 4(AX)(DI*1)
	MOVSS X4, (AX)(DX*1)
	MOVSS X5, 4(AX)(DX*1)

	INCL BP
	JMP  inv_sse2_scalar_loop

inv_sse2_scalar_butterflies:
	XORL BP, BP
	JMP  inv_sse2_scalar_loop

inv_sse2_next_base:
	ADDL SI, CX
	JMP  inv_sse2_base_loop

inv_sse2_next_size:
	SHLL $1, SI
	JMP  inv_sse2_size_loop

inv_sse2_transform_done:
	// Copy back if work != dst
	MOVL 0(SP), AX
	MOVL 24(SP), DI
	CMPL AX, DI
	JE   inv_sse2_scale

	XORL CX, CX

inv_sse2_copy_loop:
	CMPL CX, 20(SP)
	JGE  inv_sse2_scale
	MOVL (DI)(CX*8), BX
	MOVL 4(DI)(CX*8), DX
	MOVL BX, (AX)(CX*8)
	MOVL DX, 4(AX)(CX*8)
	INCL CX
	JMP  inv_sse2_copy_loop

inv_sse2_scale:
	// scale = 1.0 / n
	MOVL 20(SP), AX
	CVTSL2SS AX, X0
	MOVSS    ·one32(SB), X1
	DIVSS    X0, X1

	XORL CX, CX

inv_sse2_scale_loop:
	CMPL CX, 20(SP)
	JGE  inv_sse2_return_true
	MOVL 0(SP), AX
	// real
	MOVSS (AX)(CX*8), X0
	MULSS X1, X0
	MOVSS X0, (AX)(CX*8)
	// imag
	MOVSS 4(AX)(CX*8), X0
	MULSS X1, X0
	MOVSS X0, 4(AX)(CX*8)
	INCL CX
	JMP  inv_sse2_scale_loop

inv_sse2_return_true:
	MOVB $1, ret+60(FP)
	RET

inv_sse2_return_false:
	MOVB $0, ret+60(FP)
	RET

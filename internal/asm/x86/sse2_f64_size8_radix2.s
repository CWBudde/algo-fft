//go:build 386 && asm && !purego

// ===========================================================================
// SSE2 Size-8 Radix-2 FFT Kernels for 386 (complex128)
// ===========================================================================
//
// Radix-2 DIT FFT kernels for size 8.
// Adapted for 386 (8 XMM registers).
//
// ===========================================================================

#include "textflag.h"

// func ForwardSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
TEXT ·ForwardSSE2Size8Radix2Complex128Asm(SB), NOSPLIT, $64-64
	// Load parameters
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	CMPL DX, $8
	JNE  size8_sse2_128_32_fwd_false

	// Select working buffer
	CMPL AX, CX
	JNE  fwd_8_128_use_dst
	MOVL scratch+36(FP), AX
	
fwd_8_128_use_dst:
	MOVL AX, 0(SP) // working buffer ptr

	// Bit reversal
	MOVL bitrev+48(FP), DX
	XORL SI, SI
bitrev_8_128_loop:
	MOVL (DX)(SI*4), BX
	SHLL $4, BX
	MOVUPD (CX)(BX*1), X0
	MOVL 0(SP), DI
	MOVL SI, BX
	SHLL $4, BX
	MOVUPD X0, (DI)(BX*1)
	INCL SI
	CMPL SI, $8
	JL   bitrev_8_128_loop

	// ==================================================================
	// Stage 1: 4 butterflies, stride 1, w=1
	// (0,1), (2,3), (4,5), (6,7)
	// ==================================================================
	MOVL 0(SP), DI
	XORL SI, SI
stage1_8_128_loop:
	MOVUPD (DI)(SI*1), X0
	MOVUPD 16(DI)(SI*1), X1
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, (DI)(SI*1)
	MOVUPD X2, 16(DI)(SI*1)
	ADDL $32, SI
	CMPL SI, $128
	JL   stage1_8_128_loop

	// ==================================================================
	// Stage 2: 4 butterflies, stride 2, w={1, -i}
	// (0,2), (1,3), (4,6), (5,7)
	// ==================================================================
	// BF (0,2) w=1
	MOVUPD 0(DI), X0
	MOVUPD 32(DI), X1
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 0(DI)
	MOVUPD X2, 32(DI)

	// BF (4,6) w=1
	MOVUPD 64(DI), X0
	MOVUPD 96(DI), X1
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 64(DI)
	MOVUPD X2, 96(DI)

	// BF (1,3) w=-i
	MOVUPS ·maskNegHiPD(SB), X7
	MOVUPD 16(DI), X0
	MOVUPD 48(DI), X1
	SHUFPD $1, X1, X1
	XORPD  X7, X1    // X1 * -i
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 16(DI)
	MOVUPD X2, 48(DI)

	// BF (5,7) w=-i
	MOVUPD 80(DI), X0
	MOVUPD 112(DI), X1
	SHUFPD $1, X1, X1
	XORPD  X7, X1
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 80(DI)
	MOVUPD X2, 112(DI)

	// ==================================================================
	// Stage 3: 4 butterflies, stride 4, w={1, w1, -i, w3}
	// (0,4), (1,5), (2,6), (3,7)
	// ==================================================================
	MOVL twiddle+24(FP), CX

	// BF (0,4) w=1
	MOVUPD 0(DI), X0
	MOVUPD 64(DI), X1
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 0(DI)
	MOVUPD X2, 64(DI)

	// BF (1,5) w=w1
	MOVUPD 16(DI), X0
	MOVUPD 80(DI), X1
	MOVUPD 16(CX), X2 // w1
	MOVAPD X2, X3
	SHUFPD $0, X3, X3 // w1.re
	MOVAPD X2, X4
	SHUFPD $3, X4, X4 // w1.im
	MOVAPD X1, X5
	MULPD  X3, X1
	SHUFPD $1, X5, X5
	MULPD  X4, X5
	XORPD  ·maskNegLoPD(SB), X5
	ADDPD  X5, X1    // t = X1 * w1
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 16(DI)
	MOVUPD X2, 80(DI)

	// BF (2,6) w=-i
	MOVUPD 32(DI), X0
	MOVUPD 96(DI), X1
	SHUFPD $1, X1, X1
	XORPD  X7, X1    // X7 still has maskNegHiPD
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 32(DI)
	MOVUPD X2, 96(DI)

	// BF (3,7) w=w3
	MOVUPD 48(DI), X0
	MOVUPD 112(DI), X1
	MOVUPD 48(CX), X2 // w3
	MOVAPD X2, X3
	SHUFPD $0, X3, X3
	MOVAPD X2, X4
	SHUFPD $3, X4, X4
	MOVAPD X1, X5
	MULPD  X3, X1
	SHUFPD $1, X5, X5
	MULPD  X4, X5
	XORPD  ·maskNegLoPD(SB), X5
	ADDPD  X5, X1
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 48(DI)
	MOVUPD X2, 112(DI)

	// Copy results to dst if needed
	MOVL dst+0(FP), AX
	MOVL 0(SP), CX
	CMPL AX, CX
	JE   fwd_8_128_done
	
	XORL SI, SI
fwd_8_128_copy:
	MOVUPD (CX)(SI*1), X0
	MOVUPD X0, (AX)(SI*1)
	ADDL $16, SI
	CMPL SI, $128
	JL   fwd_8_128_copy

fwd_8_128_done:
	MOVB $1, ret+60(FP)
	RET

size8_sse2_128_32_fwd_false:
	MOVB $0, ret+60(FP)
	RET

// func InverseSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
TEXT ·InverseSSE2Size8Radix2Complex128Asm(SB), NOSPLIT, $64-64
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	CMPL DX, $8
	JNE  inv_8_128_err

	CMPL AX, CX
	JNE  inv_8_128_use_dst
	MOVL scratch+36(FP), AX
inv_8_128_use_dst:
	MOVL AX, 0(SP)

	// Bit reversal
	MOVL bitrev+48(FP), DX
	XORL SI, SI
inv_bitrev_8_128_loop:
	MOVL (DX)(SI*4), BX
	SHLL $4, BX
	MOVUPD (CX)(BX*1), X0
	MOVL 0(SP), DI
	MOVL SI, BX
	SHLL $4, BX
	MOVUPD X0, (DI)(BX*1)
	INCL SI
	CMPL SI, $8
	JL   inv_bitrev_8_128_loop

	// Stage 1
	MOVL 0(SP), DI
	XORL SI, SI
inv_stage1_8_128_loop:
	MOVUPD (DI)(SI*1), X0
	MOVUPD 16(DI)(SI*1), X1
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, (DI)(SI*1)
	MOVUPD X2, 16(DI)(SI*1)
	ADDL $32, SI
	CMPL SI, $128
	JL   inv_stage1_8_128_loop

	// Stage 2: w={1, i}
	// BF (0,2) w=1
	MOVUPD 0(DI), X0
	MOVUPD 32(DI), X1
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 0(DI)
	MOVUPD X2, 32(DI)

	// BF (4,6) w=1
	MOVUPD 64(DI), X0
	MOVUPD 96(DI), X1
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 64(DI)
	MOVUPD X2, 96(DI)

	// BF (1,3) w=i
	MOVUPS ·maskNegLoPD(SB), X7
	MOVUPD 16(DI), X0
	MOVUPD 48(DI), X1
	SHUFPD $1, X1, X1
	XORPD  X7, X1    // X1 * i = (-im, re)
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 16(DI)
	MOVUPD X2, 48(DI)

	// BF (5,7) w=i
	MOVUPD 80(DI), X0
	MOVUPD 112(DI), X1
	SHUFPD $1, X1, X1
	XORPD  X7, X1
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 80(DI)
	MOVUPD X2, 112(DI)

	// Stage 3: w={1, conj(w1), i, conj(w3)}
	MOVL twiddle+24(FP), CX
	MOVUPS ·maskNegHiPD(SB), X6 // for conj

	// BF (0,4)
	MOVUPD 0(DI), X0
	MOVUPD 64(DI), X1
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 0(DI)
	MOVUPD X2, 64(DI)

	// BF (1,5)
	MOVUPD 16(DI), X0
	MOVUPD 80(DI), X1
	MOVUPD 16(CX), X2
	XORPD  X6, X2 // conj(w1)
	MOVAPD X2, X3
	SHUFPD $0, X3, X3
	MOVAPD X2, X4
	SHUFPD $3, X4, X4
	MOVAPD X1, X5
	MULPD  X3, X1
	SHUFPD $1, X5, X5
	MULPD  X4, X5
	XORPD  ·maskNegLoPD(SB), X5
	ADDPD  X5, X1
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 16(DI)
	MOVUPD X2, 80(DI)

	// BF (2,6)
	MOVUPD 32(DI), X0
	MOVUPD 96(DI), X1
	SHUFPD $1, X1, X1
	XORPD  X7, X1    // w=i
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 32(DI)
	MOVUPD X2, 96(DI)

	// BF (3,7)
	MOVUPD 48(DI), X0
	MOVUPD 112(DI), X1
	MOVUPD 48(CX), X2
	XORPD  X6, X2 // conj(w3)
	MOVAPD X2, X3
	SHUFPD $0, X3, X3
	MOVAPD X2, X4
	SHUFPD $3, X4, X4
	MOVAPD X1, X5
	MULPD  X3, X1
	SHUFPD $1, X5, X5
	MULPD  X4, X5
	XORPD  ·maskNegLoPD(SB), X5
	ADDPD  X5, X1
	MOVAPD X0, X2
	ADDPD  X1, X0
	SUBPD  X1, X2
	MOVUPD X0, 48(DI)
	MOVUPD X2, 112(DI)

	// Scale 1/8 and Copy
	MOVL dst+0(FP), AX
	MOVL 0(SP), CX
	MOVSD ·eighth64(SB), X7
	SHUFPD $0, X7, X7
	
	XORL SI, SI
inv_8_128_scale:
	MOVUPD (CX)(SI*1), X0
	MULPD  X7, X0
	MOVUPD X0, (AX)(SI*1)
	ADDL $16, SI
	CMPL SI, $128
	JL   inv_8_128_scale

	MOVB $1, ret+60(FP)
	RET

inv_8_128_err:
	MOVB $0, ret+60(FP)
	RET

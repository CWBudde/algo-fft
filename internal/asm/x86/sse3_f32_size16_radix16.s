//go:build 386 && asm && !purego

// ===========================================================================
// SSE2 Size-16 Radix-16 FFT Kernels for 386 (complex64)
// ===========================================================================
//
// Radix-16 (4x4) FFT kernels for size 16.
// Adapted for 386 (8 XMM registers) from AMD64 implementation.
// NOTE: This 386 port is not yet correct for size-16 radix-16. It produces
// mismatched output (e.g., index 8 in asm_386_test). The kernel is currently
// excluded from 386 dispatch until the row/column stages are revalidated.
// TODO(386): Compare per-stage output against amd64/avx2 reference and fix.
//
// ===========================================================================

#include "textflag.h"

// func ForwardSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
TEXT ·ForwardSSE3Size16Radix16Complex64Asm(SB), NOSPLIT, $16-64
	// Load parameters
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	// Verify n == 16
	CMPL DX, $16
	JNE  fwd_ret_false

	// Select working buffer (Store on stack at 0(SP))
	CMPL AX, CX
	JNE  fwd_use_dst
	MOVL scratch+36(FP), AX
	
fwd_use_dst:
	MOVL AX, 0(SP)
	MOVL AX, DI // DI = current work buffer

	// Bit reversal
	MOVL bitrev+48(FP), DX
	XORL SI, SI 

fwd_bitrev_loop:
	MOVL (DX)(SI*4), BX
  MOVSD (CX)(BX*8), X0
  MOVSD X0, (DI)(SI*8)
  INCL SI
  CMPL SI, $16
  JL fwd_bitrev_loop

	// ==================================================================
	// Step 1: Column FFTs
	// ==================================================================
	
	// --- Group 1: Columns 0 & 1 ---
	MOVUPS 0(DI), X0
  MOVUPS 32(DI), X2
  MOVUPS 64(DI), X4
  MOVUPS 96(DI), X6
	MOVAPS X0, X1
  ADDPS X4, X1
  SUBPS X4, X0
  MOVAPS X2, X3
  ADDPS X6, X3
  SUBPS X6, X2
	MOVAPS X1, X4
  ADDPS X3, X4
  MOVUPS X4, 0(DI)
  SUBPS X3, X1
  MOVUPS X1, 64(DI)
	MOVAPS X2, X5
  SHUFPS $0xB1, X5, X5
  XORPS ·maskNegLoPS(SB), X5
  MOVAPS X0, X4
  SUBPS X5, X4
  MOVUPS X4, 32(DI)
  ADDPS X5, X0
  MOVUPS X0, 96(DI)

	// --- Group 2: Columns 2 & 3 ---
	MOVUPS 16(DI), X0
  MOVUPS 48(DI), X2
  MOVUPS 80(DI), X4
  MOVUPS 112(DI), X6
	MOVAPS X0, X1
  ADDPS X4, X1
  SUBPS X4, X0
  MOVAPS X2, X3
  ADDPS X6, X3
  SUBPS X6, X2
	MOVAPS X1, X4
  ADDPS X3, X4
  MOVUPS X4, 16(DI)
  SUBPS X3, X1
  MOVUPS X1, 80(DI)
	MOVAPS X2, X5
  SHUFPS $0xB1, X5, X5
  XORPS ·maskNegLoPS(SB), X5
  MOVAPS X0, X4
  SUBPS X5, X4
  MOVUPS X4, 48(DI)
  ADDPS X5, X0
  MOVUPS X0, 112(DI)

	// ==================================================================
	// Step 2: Twiddle Factors
	// ==================================================================
	MOVL twiddle+24(FP), BX

	MOVUPS 0(BX), X7
  MOVAPS X7, X4
  SHUFPS $0xA0, X4, X4
  MOVAPS X7, X5
  SHUFPS $0xF5, X5, X5
	MOVUPS 32(DI), X0
  MOVAPS X0, X1
  MULPS X4, X1
	MOVAPS X0, X2
  SHUFPS $0xB1, X2, X2
  MULPS X5, X2
  ADDSUBPS X2, X1
  MOVUPS X1, 32(DI)

	MOVUPS 16(BX), X7
  MOVAPS X7, X4
  SHUFPS $0xA0, X4, X4
  MOVAPS X7, X5
  SHUFPS $0xF5, X5, X5
	MOVUPS 48(DI), X0
  MOVAPS X0, X1
  MULPS X4, X1
	MOVAPS X0, X2
  SHUFPS $0xB1, X2, X2
  MULPS X5, X2
  ADDSUBPS X2, X1
  MOVUPS X1, 48(DI)

	MOVSD 0(BX), X7
  MOVHPS 16(BX), X7
  MOVAPS X7, X4
  SHUFPS $0xA0, X4, X4
  MOVAPS X7, X5
  SHUFPS $0xF5, X5, X5
	MOVUPS 64(DI), X0
  MOVAPS X0, X1
  MULPS X4, X1
	MOVAPS X0, X2
  SHUFPS $0xB1, X2, X2
  MULPS X5, X2
  ADDSUBPS X2, X1
  MOVUPS X1, 64(DI)

	MOVSD 32(BX), X7
  MOVHPS 48(BX), X7
  MOVAPS X7, X4
  SHUFPS $0xA0, X4, X4
  MOVAPS X7, X5
  SHUFPS $0xF5, X5, X5
	MOVUPS 80(DI), X0
  MOVAPS X0, X1
  MULPS X4, X1
	MOVAPS X0, X2
  SHUFPS $0xB1, X2, X2
  MULPS X5, X2
  ADDSUBPS X2, X1
  MOVUPS X1, 80(DI)

	MOVSD 0(BX), X7
  MOVHPS 24(BX), X7
  MOVAPS X7, X4
  SHUFPS $0xA0, X4, X4
  MOVAPS X7, X5
  SHUFPS $0xF5, X5, X5
	MOVUPS 96(DI), X0
  MOVAPS X0, X1
  MULPS X4, X1
	MOVAPS X0, X2
  SHUFPS $0xB1, X2, X2
  MULPS X5, X2
  ADDSUBPS X2, X1
  MOVUPS X1, 96(DI)

	MOVSD 48(BX), X7
  MOVSD 8(BX), X6
  XORPS X4, X4
  SUBPS X6, X4
  MOVLHPS X4, X7
	MOVAPS X7, X4
  SHUFPS $0xA0, X4, X4
  MOVAPS X7, X5
  SHUFPS $0xF5, X5, X5
	MOVUPS 112(DI), X0
  MOVAPS X0, X1
  MULPS X4, X1
	MOVAPS X0, X2
  SHUFPS $0xB1, X2, X2
  MULPS X5, X2
  ADDSUBPS X2, X1
  MOVUPS X1, 112(DI)

	// ==================================================================
	// Step 3: Row FFTs
	// ==================================================================
	XORL SI, SI
row_loop:
	MOVUPS (DI)(SI*1), X0
  MOVUPS 16(DI)(SI*1), X1
  MOVAPS X0, X2
  ADDPS X1, X2
  SUBPS X1, X0
	MOVAPS X2, X3
  MOVHLPS X2, X3
  MOVAPS X2, X4
  ADDPS X3, X4
  SUBPS X3, X2
	MOVAPS X0, X3
  MOVHLPS X0, X3
  SHUFPS $0xB1, X3, X3
  XORPS ·maskNegLoPS(SB), X3
	MOVAPS X0, X5
  SUBPS X3, X5
  ADDPS X3, X0
	UNPCKLPD X5, X4
  UNPCKLPD X0, X2
  MOVUPS X4, (DI)(SI*1)
  MOVUPS X2, 16(DI)(SI*1)
	ADDL $32, SI
  CMPL SI, $128
  JL row_loop

	// ==================================================================
	// Step 4: Transpose and Store
	// ==================================================================
	MOVL dst+0(FP), AX
	MOVUPS 0(DI), X0
  MOVUPS 32(DI), X2
  MOVAPS X0, X7
  UNPCKLPD X2, X7
  MOVUPS X7, 0(AX)
  MOVAPS X0, X7
  UNPCKHPD X2, X7
  MOVUPS X7, 32(AX)
	MOVUPS 64(DI), X4
  MOVUPS 96(DI), X6
  MOVAPS X4, X7
  UNPCKLPD X6, X7
  MOVUPS X7, 16(AX)
  MOVAPS X4, X7
  UNPCKHPD X6, X7
  MOVUPS X7, 48(AX)
	MOVUPS 16(DI), X1
  MOVUPS 48(DI), X3
  MOVAPS X1, X7
  UNPCKLPD X3, X7
  MOVUPS X7, 64(AX)
  MOVAPS X1, X7
  UNPCKHPD X3, X7
  MOVUPS X7, 96(AX)
	MOVUPS 80(DI), X5
  MOVUPS 112(DI), X0
  MOVAPS X5, X7
  UNPCKLPD X0, X7
  MOVUPS X7, 80(AX)
  MOVAPS X5, X7
  UNPCKHPD X0, X7
  MOVUPS X7, 112(AX)

	MOVB $1, ret+60(FP)
	RET

fwd_ret_false:
	MOVB $0, ret+60(FP)
	RET

// func InverseSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
TEXT ·InverseSSE3Size16Radix16Complex64Asm(SB), NOSPLIT, $16-64
	MOVL dst+0(FP), AX
	MOVL src+12(FP), CX
	MOVL src+16(FP), DX

	CMPL DX, $16
	JNE  inv_ret_false

	CMPL AX, CX
	JNE  inv_use_dst
	MOVL scratch+36(FP), AX
	
inv_use_dst:
	MOVL AX, 0(SP)
	MOVL AX, DI

	MOVL bitrev+48(FP), DX
	XORL SI, SI
inv_bitrev_loop:
	MOVL (DX)(SI*4), BX
  MOVSD (CX)(BX*8), X0
  MOVSD X0, (DI)(SI*8)
  INCL SI
  CMPL SI, $16
  JL inv_bitrev_loop

	// Step 1: Vertical IFFT4 (uses maskNegLoPS like Forward, but ADD/SUB swapped)
	MOVUPS 0(DI), X0
  MOVUPS 32(DI), X2
  MOVUPS 64(DI), X4
  MOVUPS 96(DI), X6
	MOVAPS X0, X1
  ADDPS X4, X1
  SUBPS X4, X0
  MOVAPS X2, X3
  ADDPS X6, X3
  SUBPS X6, X2
	MOVAPS X1, X4
  ADDPS X3, X4
  MOVUPS X4, 0(DI)
  SUBPS X3, X1
  MOVUPS X1, 64(DI)
	MOVAPS X2, X5
  SHUFPS $0xB1, X5, X5
  XORPS ·maskNegLoPS(SB), X5
	MOVAPS X0, X4
  ADDPS X5, X4
  MOVUPS X4, 32(DI)
  SUBPS X5, X0
  MOVUPS X0, 96(DI)

	MOVUPS 16(DI), X0
  MOVUPS 48(DI), X2
  MOVUPS 80(DI), X4
  MOVUPS 112(DI), X6
	MOVAPS X0, X1
  ADDPS X4, X1
  SUBPS X4, X0
  MOVAPS X2, X3
  ADDPS X6, X3
  SUBPS X6, X2
	MOVAPS X1, X4
  ADDPS X3, X4
  MOVUPS X4, 16(DI)
  SUBPS X3, X1
  MOVUPS X1, 80(DI)
	MOVAPS X2, X5
  SHUFPS $0xB1, X5, X5
  XORPS ·maskNegLoPS(SB), X5
	MOVAPS X0, X4
  ADDPS X5, X4
  MOVUPS X4, 48(DI)
  SUBPS X5, X0
  MOVUPS X0, 112(DI)

	// Step 2: Conjugated Twiddles
	MOVL twiddle+24(FP), BX
	
	MOVUPS 0(BX), X7
  XORPS ·maskNegHiPS(SB), X7
  MOVAPS X7, X4
  SHUFPS $0xA0, X4, X4
  MOVAPS X7, X5
  SHUFPS $0xF5, X5, X5
	MOVUPS 32(DI), X0
  MOVAPS X0, X1
  MULPS X4, X1
	MOVAPS X0, X2
  SHUFPS $0xB1, X2, X2
  MULPS X5, X2
  ADDSUBPS X2, X1
  MOVUPS X1, 32(DI)

	MOVUPS 16(BX), X7
  XORPS ·maskNegHiPS(SB), X7
  MOVAPS X7, X4
  SHUFPS $0xA0, X4, X4
  MOVAPS X7, X5
  SHUFPS $0xF5, X5, X5
	MOVUPS 48(DI), X0
  MOVAPS X0, X1
  MULPS X4, X1
	MOVAPS X0, X2
  SHUFPS $0xB1, X2, X2
  MULPS X5, X2
  ADDSUBPS X2, X1
  MOVUPS X1, 48(DI)

	MOVSD 0(BX), X7
  MOVHPS 16(BX), X7
  XORPS ·maskNegHiPS(SB), X7
  MOVAPS X7, X4
  SHUFPS $0xA0, X4, X4
  MOVAPS X7, X5
  SHUFPS $0xF5, X5, X5
	MOVUPS 64(DI), X0
  MOVAPS X0, X1
  MULPS X4, X1
	MOVAPS X0, X2
  SHUFPS $0xB1, X2, X2
  MULPS X5, X2
  ADDSUBPS X2, X1
  MOVUPS X1, 64(DI)

	MOVSD 32(BX), X7
  MOVHPS 48(BX), X7
  XORPS ·maskNegHiPS(SB), X7
  MOVAPS X7, X4
  SHUFPS $0xA0, X4, X4
  MOVAPS X7, X5
  SHUFPS $0xF5, X5, X5
	MOVUPS 80(DI), X0
  MOVAPS X0, X1
  MULPS X4, X1
	MOVAPS X0, X2
  SHUFPS $0xB1, X2, X2
  MULPS X5, X2
  ADDSUBPS X2, X1
  MOVUPS X1, 80(DI)

	MOVSD 0(BX), X7
  MOVHPS 24(BX), X7
  XORPS ·maskNegHiPS(SB), X7
  MOVAPS X7, X4
  SHUFPS $0xA0, X4, X4
  MOVAPS X7, X5
  SHUFPS $0xF5, X5, X5
	MOVUPS 96(DI), X0
  MOVAPS X0, X1
  MULPS X4, X1
	MOVAPS X0, X2
  SHUFPS $0xB1, X2, X2
  MULPS X5, X2
  ADDSUBPS X2, X1
  MOVUPS X1, 96(DI)

	MOVSD 48(BX), X7
  MOVSD 8(BX), X6
  XORPS X4, X4
  SUBPS X6, X4
  MOVLHPS X4, X7
  XORPS ·maskNegHiPS(SB), X7
  MOVAPS X7, X4
  SHUFPS $0xA0, X4, X4
  MOVAPS X7, X5
  SHUFPS $0xF5, X5, X5
	MOVUPS 112(DI), X0
  MOVAPS X0, X1
  MULPS X4, X1
	MOVAPS X0, X2
  SHUFPS $0xB1, X2, X2
  MULPS X5, X2
  ADDSUBPS X2, X1
  MOVUPS X1, 112(DI)

	// Step 3: Row IFFTs (maskNegLoPS, ADD X13, X12; SUB X14, X12)
	XORL SI, SI
inv_row_loop:
	MOVUPS (DI)(SI*1), X0
  MOVUPS 16(DI)(SI*1), X1
  MOVAPS X0, X2
  ADDPS X1, X2
  SUBPS X1, X0
	MOVAPS X2, X3
  MOVHLPS X2, X3
  MOVAPS X2, X4
  ADDPS X3, X4
  SUBPS X3, X2
	MOVAPS X0, X3
  MOVHLPS X0, X3
  SHUFPS $0xB1, X3, X3
  XORPS ·maskNegLoPS(SB), X3
	MOVAPS X0, X5
  ADDPS X3, X5
  SUBPS X3, X0
	UNPCKLPD X5, X4
  UNPCKLPD X0, X2
  MOVUPS X4, (DI)(SI*1)
  MOVUPS X2, 16(DI)(SI*1)
	ADDL $32, SI
  CMPL SI, $128
  JL inv_row_loop

	// Step 4: Scale and Transpose
	MOVL dst+0(FP), AX
  MOVSS ·sixteenth32(SB), X7
  SHUFPS $0x00, X7, X7
	MOVUPS 0(DI), X0
  MOVUPS 32(DI), X2
  MOVAPS X0, X4
  UNPCKLPD X2, X4
  MULPS X7, X4
  MOVUPS X4, 0(AX)
  MOVAPS X0, X4
  UNPCKHPD X2, X4
  MULPS X7, X4
  MOVUPS X4, 32(AX)
	MOVUPS 64(DI), X1
  MOVUPS 96(DI), X3
  MOVAPS X1, X4
  UNPCKLPD X3, X4
  MULPS X7, X4
  MOVUPS X4, 16(AX)
  MOVAPS X1, X4
  UNPCKHPD X3, X4
  MULPS X7, X4
  MOVUPS X4, 48(AX)
	MOVUPS 16(DI), X0
  MOVUPS 48(DI), X2
  MOVAPS X0, X4
  UNPCKLPD X2, X4
  MULPS X7, X4
  MOVUPS X4, 64(AX)
  MOVAPS X0, X4
  UNPCKHPD X2, X4
  MULPS X7, X4
  MOVUPS X4, 96(AX)
	MOVUPS 80(DI), X1
  MOVUPS 112(DI), X3
  MOVAPS X1, X4
  UNPCKLPD X3, X4
  MULPS X7, X4
  MOVUPS X4, 80(AX)
  MOVAPS X1, X4
  UNPCKHPD X3, X4
  MULPS X7, X4
  MOVUPS X4, 112(AX)

	MOVB $1, ret+60(FP)
	RET

inv_ret_false:
	MOVB $0, ret+60(FP)
	RET

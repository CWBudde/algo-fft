//go:build amd64 && asm && !purego

#include "textflag.h"

// ===========================================================================
// Forward transform, size 8, complex64, radix-8 variant
// ===========================================================================
TEXT ·ForwardSSE2Size8Radix8Complex64Asm(SB), NOSPLIT, $0-121
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $8
	JNE  fwd_ret_false
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   fwd_ret_false
	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   fwd_ret_false
	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   fwd_ret_false
	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   fwd_ret_false

	CMPQ R8, R9
	JNE  fwd_use_dst
	MOVQ R11, R8

fwd_use_dst:
	// Load input
	MOVUPS (R9), X0
	MOVUPS 16(R9), X1
	MOVUPS 32(R9), X2
	MOVUPS 48(R9), X3

	// Stage 1: Sum/Diff (Stride 4)
	MOVAPS X0, X4
	ADDPS  X2, X4    // S0, S1
	MOVAPS X1, X5
	ADDPS  X3, X5    // S2, S3
	MOVAPS X0, X6
	SUBPS  X2, X6    // D0, D1
	MOVAPS X1, X7
	SUBPS  X3, X7    // D2, D3

	// Unpack Sums
	MOVAPS X4, X8
	UNPCKLPD X5, X8  // S0, S2 (for e0, e2)
	MOVAPS X4, X9
	UNPCKHPD X5, X9  // S1, S3 (for o0, o2)

	// Process Sums (Simple Butterfly)
	// e0, e2 from S0, S2
	MOVAPS X8, X10
	SHUFPS $0x4E, X10, X10 // S2, S0
	MOVAPS X8, X0
	ADDPS  X10, X0   // e0 in Low
	MOVAPS X8, X1
	SUBPS  X10, X1   // e2 in Low

	// o0, o2 from S1, S3
	MOVAPS X9, X10
	SHUFPS $0x4E, X10, X10 // S3, S1
	MOVAPS X9, X2
	ADDPS  X10, X2   // o0 in Low
	MOVAPS X9, X3
	SUBPS  X10, X3   // o2 in Low

	// Unpack Diffs
	MOVAPS X6, X8
	UNPCKLPD X7, X8  // D0, D2 (for e1, e3)
	MOVAPS X6, X9
	UNPCKHPD X7, X9  // D1, D3 (for o1, o3)

	// Process Diffs (Rotated Butterfly -i)
	// e1, e3 from D0, D2
	MOVAPS X8, X10
	SHUFPS $0x4E, X10, X10 // D2, D0
	
	// Apply -i to D2, D0
	MOVAPS X10, X11
	SHUFPS $0xB1, X11, X11
	MOVUPS ·maskNegHiPS(SB), X12
	XORPS  X12, X11  // D2*-i, D0*-i
	
	MOVAPS X8, X4
	ADDPS  X11, X4   // e1 in Low
	MOVAPS X8, X5
	SUBPS  X11, X5   // e3 in Low

	// o1, o3 from D1, D3
	MOVAPS X9, X10
	SHUFPS $0x4E, X10, X10 // D3, D1
	
	// Apply -i to D3, D1
	MOVAPS X10, X11
	SHUFPS $0xB1, X11, X11
	XORPS  X12, X11  // D3*-i, D1*-i (X12 has maskNegHiPS)
	
	MOVAPS X9, X6
	ADDPS  X11, X6   // o1 in Low
	MOVAPS X9, X7
	SUBPS  X11, X7   // o3 in Low

	// Pack Results
	// Need:
	// E_lo (e0, e1) -> X0, X4
	// E_hi (e2, e3) -> X1, X5
	// O_lo (o0, o1) -> X2, X6
	// O_hi (o2, o3) -> X3, X7
	
	MOVAPS X0, X13
	UNPCKLPD X4, X13 // E_lo
	
	MOVAPS X1, X14
	UNPCKLPD X5, X14 // E_hi
	
	MOVAPS X2, X15
	UNPCKLPD X6, X15 // O_lo
	
	MOVAPS X3, X8
	UNPCKLPD X7, X8  // O_hi

	// Stage 4: Twiddle
	MOVUPS (R10), X0
	MOVUPS 16(R10), X1

	// Butterfly 1: E_lo(X13), O_lo(X15), W01(X0)
	MOVAPS X0, X2
	SHUFPS $0xA0, X2, X2
	MOVAPS X0, X3
	SHUFPS $0xF5, X3, X3
	
	MOVAPS X15, X4
	MULPS  X2, X4
	MOVAPS X15, X5
	SHUFPS $0xB1, X5, X5
	MULPS  X3, X5
	ADDSUBPS X5, X4
	
	MOVAPS X13, X0
	ADDPS  X4, X0
	MOVAPS X13, X2
	SUBPS  X4, X2

	// Butterfly 2: E_hi(X14), O_hi(X8), W23(X1)
	MOVAPS X1, X6
	SHUFPS $0xA0, X6, X6
	MOVAPS X1, X7
	SHUFPS $0xF5, X7, X7
	
	MOVAPS X8, X9
	MULPS  X6, X9
	MOVAPS X8, X10
	SHUFPS $0xB1, X10, X10
	MULPS  X7, X10
	ADDSUBPS X10, X9
	
	MOVAPS X14, X1
	ADDPS  X9, X1
	MOVAPS X14, X3
	SUBPS  X9, X3

	// Store
	MOVUPS X0, (R8)
	MOVUPS X1, 16(R8)
	MOVUPS X2, 32(R8)
	MOVUPS X3, 48(R8)

	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   fwd_done
	MOVUPS (R8), X0
	MOVUPS X0, (R14)
	MOVUPS 16(R8), X0
	MOVUPS X0, 16(R14)
	MOVUPS 32(R8), X0
	MOVUPS X0, 32(R14)
	MOVUPS 48(R8), X0
	MOVUPS X0, 48(R14)

fwd_done:
	MOVB $1, ret+120(FP)
	RET

fwd_ret_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform
// ===========================================================================
TEXT ·InverseSSE2Size8Radix8Complex64Asm(SB), NOSPLIT, $0-121
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $8
	JNE  inv_ret_false
	MOVQ dst+8(FP), AX
	CMPQ AX, $8
	JL   inv_ret_false
	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8
	JL   inv_ret_false
	MOVQ scratch+80(FP), AX
	CMPQ AX, $8
	JL   inv_ret_false
	MOVQ bitrev+104(FP), AX
	CMPQ AX, $8
	JL   inv_ret_false

	CMPQ R8, R9
	JNE  inv_use_dst
	MOVQ R11, R8

inv_use_dst:
	MOVUPS (R9), X0
	MOVUPS 16(R9), X1
	MOVUPS 32(R9), X2
	MOVUPS 48(R9), X3

	// Stage 1
	MOVAPS X0, X4
	ADDPS  X2, X4
	MOVAPS X1, X5
	ADDPS  X3, X5
	MOVAPS X0, X6
	SUBPS  X2, X6
	MOVAPS X1, X7
	SUBPS  X3, X7

	// Unpack Sums
	MOVAPS X4, X8
	UNPCKLPD X5, X8  // S0, S2
	MOVAPS X4, X9
	UNPCKHPD X5, X9  // S1, S3

	// Process Sums (Simple)
	MOVAPS X8, X10
	SHUFPS $0x4E, X10, X10
	MOVAPS X8, X0
	ADDPS  X10, X0   // e0
	MOVAPS X8, X1
	SUBPS  X10, X1   // e2

	MOVAPS X9, X10
	SHUFPS $0x4E, X10, X10
	MOVAPS X9, X2
	ADDPS  X10, X2   // o0
	MOVAPS X9, X3
	SUBPS  X10, X3   // o2

	// Unpack Diffs
	MOVAPS X6, X8
	UNPCKLPD X7, X8  // D0, D2
	MOVAPS X6, X9
	UNPCKHPD X7, X9  // D1, D3

	// Process Diffs (Rotated +i)
	// e1, e3 from D0, D2
	MOVAPS X8, X10
	SHUFPS $0x4E, X10, X10
	
	MOVAPS X10, X11
	SHUFPS $0xB1, X11, X11
	MOVUPS ·maskNegLoPS(SB), X12
	XORPS  X12, X11  // D2*+i, D0*+i
	
	MOVAPS X8, X4
	ADDPS  X11, X4   // e1
	MOVAPS X8, X5
	SUBPS  X11, X5   // e3

	// o1, o3 from D1, D3
	MOVAPS X9, X10
	SHUFPS $0x4E, X10, X10
	
	MOVAPS X10, X11
	SHUFPS $0xB1, X11, X11
	XORPS  X12, X11  // D3*+i, D1*+i
	
	MOVAPS X9, X6
	ADDPS  X11, X6   // o1
	MOVAPS X9, X7
	SUBPS  X11, X7   // o3

	// Pack
	MOVAPS X0, X13
	UNPCKLPD X4, X13 // E_lo
	MOVAPS X1, X14
	UNPCKLPD X5, X14 // E_hi
	MOVAPS X2, X15
	UNPCKLPD X6, X15 // O_lo
	MOVAPS X3, X8
	UNPCKLPD X7, X8  // O_hi

	// Twiddle (Conj)
	MOVUPS (R10), X0
	MOVUPS 16(R10), X1
	MOVUPS ·maskNegHiPS(SB), X12
	XORPS  X12, X0
	XORPS  X12, X1

	// Butterfly 1
	MOVAPS X0, X2
	SHUFPS $0xA0, X2, X2
	MOVAPS X0, X3
	SHUFPS $0xF5, X3, X3
	
	MOVAPS X15, X4
	MULPS  X2, X4
	MOVAPS X15, X5
	SHUFPS $0xB1, X5, X5
	MULPS  X3, X5
	ADDSUBPS X5, X4
	
	MOVAPS X13, X0
	ADDPS  X4, X0
	MOVAPS X13, X2
	SUBPS  X4, X2

	// Butterfly 2
	MOVAPS X1, X6
	SHUFPS $0xA0, X6, X6
	MOVAPS X1, X7
	SHUFPS $0xF5, X7, X7
	
	MOVAPS X8, X9
	MULPS  X6, X9
	MOVAPS X8, X10
	SHUFPS $0xB1, X10, X10
	MULPS  X7, X10
	ADDSUBPS X10, X9
	
	MOVAPS X14, X1
	ADDPS  X9, X1
	MOVAPS X14, X3
	SUBPS  X9, X3

	// Scale
	MOVSS  ·eighth32(SB), X15
	SHUFPS $0x00, X15, X15
	MULPS  X15, X0
	MULPS  X15, X1
	MULPS  X15, X2
	MULPS  X15, X3

	MOVUPS X0, (R8)
	MOVUPS X1, 16(R8)
	MOVUPS X2, 32(R8)
	MOVUPS X3, 48(R8)

	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   inv_done
	MOVUPS (R8), X0
	MOVUPS X0, (R14)
	MOVUPS 16(R8), X0
	MOVUPS X0, 16(R14)
	MOVUPS 32(R8), X0
	MOVUPS X0, 32(R14)
	MOVUPS 48(R8), X0
	MOVUPS X0, 48(R14)

inv_done:
	MOVB $1, ret+120(FP)
	RET

inv_ret_false:
	MOVB $0, ret+120(FP)
	RET

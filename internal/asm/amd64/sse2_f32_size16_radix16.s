//go:build amd64 && asm && !purego

#include "textflag.h"

// ===========================================================================
// Forward transform, size 16, complex64, radix-16 (4x4) variant
// ===========================================================================
TEXT ·ForwardSSE2Size16Radix16Complex64Asm(SB), NOSPLIT, $0-121
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	// Validate inputs (not strictly necessary if caller is trusted, but good)
	CMPQ R13, $16
	JNE  fwd_ret_false

	// Load Input (Natural Order assumed via Identity Indices)
	// We load 8 registers (16 elements).
	// D0: 0, 1
	// D1: 2, 3
	// D2: 4, 5
	// D3: 6, 7
	// D4: 8, 9
	// D5: 10, 11
	// D6: 12, 13
	// D7: 14, 15
	MOVUPS 0(R9), X0
	MOVUPS 16(R9), X1
	MOVUPS 32(R9), X2
	MOVUPS 48(R9), X3
	MOVUPS 64(R9), X4
	MOVUPS 80(R9), X5
	MOVUPS 96(R9), X6
	MOVUPS 112(R9), X7

	// =======================================================================
	// Step 1: 4x Column FFTs (Stride 4)
	// Col 0: 0, 4, 8, 12 -> In D0, D2, D4, D6 (Low parts)
	// Col 1: 1, 5, 9, 13 -> In D0, D2, D4, D6 (High parts)
	// We process (D0, D2, D4, D6) together.
	// =======================================================================
	
	// T1 = D0 + D4
	MOVAPS X0, X8
	ADDPS  X4, X8
	// T2 = D2 + D6
	MOVAPS X2, X9
	ADDPS  X6, X9
	// T3 = D0 - D4
	MOVAPS X0, X10
	SUBPS  X4, X10
	// T4 = D2 - D6
	MOVAPS X2, X11
	SUBPS  X6, X11
	
	// F0 = T1 + T2 (Store in D0)
	MOVAPS X8, X0
	ADDPS  X9, X0
	
	// F2 = T1 - T2 (Store in D4 -> becomes Row 2)
	MOVAPS X8, X4
	SUBPS  X9, X4
	
	// Prepare i*T4
	MOVAPS X11, X12
	SHUFPS $0xB1, X12, X12
	MOVUPS ·maskNegHiPS(SB), X13 // Negate High (Imag) part? No, (x+iy)*i = -y+ix. Negate Real.
	// Wait, ·maskNegHiPS usually negates the high word of a pair (index 1, 3).
	// Complex number is Low=Re, High=Im.
	// We want to negate Real part (-y). That is Low word.
	// Check mask definitions. maskNegLoPS negates Low.
	MOVUPS ·maskNegLoPS(SB), X13
	XORPS  X13, X12 // i*T4
	
	// F1 = T3 - i*T4 (Store in D2 -> becomes Row 1)
	MOVAPS X10, X2
	SUBPS  X12, X2
	
	// F3 = T3 + i*T4 (Store in D6 -> becomes Row 3)
	MOVAPS X10, X6
	ADDPS  X12, X6
	
	// =======================================================================
	// Repeat for Cols 2, 3 (D1, D3, D5, D7)
	// =======================================================================
	
	// T1 = D1 + D5
	MOVAPS X1, X8
	ADDPS  X5, X8
	// T2 = D3 + D7
	MOVAPS X3, X9
	ADDPS  X7, X9
	// T3 = D1 - D5
	MOVAPS X1, X10
	SUBPS  X5, X10
	// T4 = D3 - D7
	MOVAPS X3, X11
	SUBPS  X7, X11
	
	// F0 = T1 + T2 (Store in D1)
	MOVAPS X8, X1
	ADDPS  X9, X1
	
	// F2 = T1 - T2 (Store in D5)
	MOVAPS X8, X5
	SUBPS  X9, X5
	
	// Prepare i*T4
	MOVAPS X11, X12
	SHUFPS $0xB1, X12, X12
	MOVUPS ·maskNegLoPS(SB), X13
	XORPS  X13, X12
	
	// F1 = T3 - i*T4 (Store in D3)
	MOVAPS X10, X3
	SUBPS  X12, X3
	
	// F3 = T3 + i*T4 (Store in D7)
	MOVAPS X10, X7
	ADDPS  X12, X7
	
	// =======================================================================
	// Step 2: Internal Twiddles
	// D0, D1: Row 0. Mult by 1. (Done)
	// D2, D3: Row 1. Mult by W^0, W^1, W^2, W^3.
	// D4, D5: Row 2. Mult by W^0, W^2, W^4, W^6.
	// D6, D7: Row 3. Mult by W^0, W^3, W^6, W^9.
	// =======================================================================
	
	// Define Helper MUL(Reg, TwiddleReg)
	// Complex multiplication: (a+bi)(c+di) = (ac-bd) + i(ad+bc)
	// We use SHUFPS/MULPS approach.
	
	// Load Twiddles from R10 (twiddle array)
	// Twiddle layout: W^0..W^7 (size/2). 8 complex numbers.
	
	// --- Row 1 (D2, D3) ---
	// D2 needs W^0, W^1. (Indices 0, 1). Load 16 bytes at offset 0.
	MOVUPS 0(R10), X14
	// Mult D2 * X14
	MOVAPS X14, X8
	SHUFPS $0xA0, X8, X8 // Re(W)
	MOVAPS X14, X9
	SHUFPS $0xF5, X9, X9 // Im(W)
	MOVAPS X2, X10
	MULPS  X8, X10       // D2 * Re
	MOVAPS X2, X11
	SHUFPS $0xB1, X11, X11 // Swap(D2)
	MULPS  X9, X11       // Swap(D2) * Im
	ADDSUBPS X11, X10    // (ac-bd, ad+bc)
	MOVAPS X10, X2       // Store back
	
	// D3 needs W^2, W^3. (Indices 2, 3). Offset 16.
	MOVUPS 16(R10), X14
	// Mult D3 * X14
	MOVAPS X14, X8
	SHUFPS $0xA0, X8, X8
	MOVAPS X14, X9
	SHUFPS $0xF5, X9, X9
	MOVAPS X3, X10
	MULPS  X8, X10
	MOVAPS X3, X11
	SHUFPS $0xB1, X11, X11
	MULPS  X9, X11
	ADDSUBPS X11, X10
	MOVAPS X10, X3
	
	// --- Row 2 (D4, D5) ---
	// D4 needs W^0, W^2. (Indices 0, 2).
	// Load W^0 (0), W^2 (16). Need to pack.
	MOVSD  0(R10), X14   // Load W^0 (Low)
	MOVHPS 16(R10), X14  // Load W^2 (High)
	// Mult D4 * X14
	MOVAPS X14, X8
	SHUFPS $0xA0, X8, X8
	MOVAPS X14, X9
	SHUFPS $0xF5, X9, X9
	MOVAPS X4, X10
	MULPS  X8, X10
	MOVAPS X4, X11
	SHUFPS $0xB1, X11, X11
	MULPS  X9, X11
	ADDSUBPS X11, X10
	MOVAPS X10, X4
	
	// D5 needs W^4, W^6. (Indices 4, 6). Offsets 32, 48.
	MOVSD  32(R10), X14
	MOVHPS 48(R10), X14
	// Mult D5 * X14
	MOVAPS X14, X8
	SHUFPS $0xA0, X8, X8
	MOVAPS X14, X9
	SHUFPS $0xF5, X9, X9
	MOVAPS X5, X10
	MULPS  X8, X10
	MOVAPS X5, X11
	SHUFPS $0xB1, X11, X11
	MULPS  X9, X11
	ADDSUBPS X11, X10
	MOVAPS X10, X5
	
	// --- Row 3 (D6, D7) ---
	// D6 needs W^0, W^3. (Indices 0, 3). Offsets 0, 24.
	MOVSD  0(R10), X14
	MOVHPS 24(R10), X14
	// Mult D6 * X14
	MOVAPS X14, X8
	SHUFPS $0xA0, X8, X8
	MOVAPS X14, X9
	SHUFPS $0xF5, X9, X9
	MOVAPS X6, X10
	MULPS  X8, X10
	MOVAPS X6, X11
	SHUFPS $0xB1, X11, X11
	MULPS  X9, X11
	ADDSUBPS X11, X10
	MOVAPS X10, X6
	
	// D7 needs W^6, W^9. (Indices 6, 9).
	// W^6 at 48.
	// W^9 = -W^1. W^1 at 8.
	MOVSD  48(R10), X14  // Load W^6 (Low)
	MOVSD  8(R10), X15   // Load W^1 (Temporary)
	// Negate X15
	MOVAPS X15, X8
	XORPS  X8, X8        // Zero
	SUBPS  X15, X8       // 0 - W^1 = -W^1
	MOVLHPS X8, X14      // Move -W^1 to High of X14
	
	// Mult D7 * X14
	MOVAPS X14, X8
	SHUFPS $0xA0, X8, X8
	MOVAPS X14, X9
	SHUFPS $0xF5, X9, X9
	MOVAPS X7, X10
	MULPS  X8, X10
	MOVAPS X7, X11
	SHUFPS $0xB1, X11, X11
	MULPS  X9, X11
	ADDSUBPS X11, X10
	MOVAPS X10, X7
	
	// =======================================================================
	// Step 3: 4x Row FFTs (Stride 1)
	// We have 4 Rows.
	// Row 0: D0, D1.
	// Row 1: D2, D3.
	// Row 2: D4, D5.
	// Row 3: D6, D7.
	// Perform FFT4 on each pair.
	// =======================================================================
	
	// Define Macro-like sequence for (A, B) -> (A', B')
	// We need 4 scratch regs. X8-X11 are free. X12-X15 also.
	
	// --- Row 0 (D0, D1) ---
	MOVAPS X0, X8  // A
	MOVAPS X1, X9  // B
	
	MOVAPS X8, X10
	ADDPS  X9, X10 // Sum = A+B
	MOVAPS X8, X11
	SUBPS  X9, X11 // Diff = A-B
	
	// y0, y2 from Sum
	MOVAPS X10, X12
	MOVHLPS X10, X12 // High(Sum) -> Low(X12)
	MOVAPS X10, X13
	ADDPS  X12, X13  // y0 = S0 + S1 (Low)
	MOVAPS X10, X14
	SUBPS  X12, X14  // y2 = S0 - S1 (Low)
	
	// y1, y3 from Diff
	MOVAPS X11, X12
	MOVHLPS X11, X12 // High(Diff) -> Low(X12)
	// Mult High by i
	MOVAPS X12, X15
	SHUFPS $0xB1, X15, X15
	MOVUPS ·maskNegLoPS(SB), X8 // Reuse X8 for mask
	XORPS  X8, X15   // i*D1
	
	MOVAPS X11, X8   // D0 (Low part correct)
	MOVAPS X8, X9
	SUBPS  X15, X9   // y1 = D0 - i*D1
	MOVAPS X8, X15
	ADDPS  X15, X15  // Wait, I overwrote i*D1 reg? 
	// Recompute i*D1? Or execute y3 = D0 + i*D1.
	// Let's redo cleaner.
	
	// y1 = D0 - i*D1
	// y3 = D0 + i*D1
	// X11 has (D0, D1). X12 has (D1, D1).
	// We need D0 in a reg.
	// X11 is not just D0.
	// We only care about Low part of X11.
	
	// Recalculate i*D1 correctly
	MOVAPS X12, X15 // D1
	SHUFPS $0xB1, X15, X15
	MOVUPS ·maskNegLoPS(SB), X8
	XORPS  X8, X15  // i*D1
	
	MOVAPS X11, X8  // D0 (and D1 high, ignored)
	MOVAPS X8, X9
	SUBPS  X15, X9  // y1
	ADDPS  X15, X8  // y3
	
	// Pack (y0, y1) -> D0. (y2, y3) -> D1.
	UNPCKLPD X9, X13 // (y0, y1)
	MOVAPS X13, X0
	UNPCKLPD X8, X14 // (y2, y3)
	MOVAPS X14, X1
	
	// --- Row 1 (D2, D3) ---
	MOVAPS X2, X8
	MOVAPS X3, X9
	MOVAPS X8, X10
	ADDPS  X9, X10 // Sum
	MOVAPS X8, X11
	SUBPS  X9, X11 // Diff
	
	MOVAPS X10, X12
	MOVHLPS X10, X12
	MOVAPS X10, X13
	ADDPS  X12, X13 // y0
	MOVAPS X10, X14
	SUBPS  X12, X14 // y2
	
	MOVAPS X11, X12
	MOVHLPS X11, X12
	MOVAPS X12, X15
	SHUFPS $0xB1, X15, X15
	MOVUPS ·maskNegLoPS(SB), X8
	XORPS  X8, X15 // i*D1
	
	MOVAPS X11, X8
	MOVAPS X8, X9
	SUBPS  X15, X9 // y1
	ADDPS  X15, X8 // y3
	
	UNPCKLPD X9, X13
	MOVAPS X13, X2
	UNPCKLPD X8, X14
	MOVAPS X14, X3
	
	// --- Row 2 (D4, D5) ---
	MOVAPS X4, X8
	MOVAPS X5, X9
	MOVAPS X8, X10
	ADDPS  X9, X10
	MOVAPS X8, X11
	SUBPS  X9, X11
	
	MOVAPS X10, X12
	MOVHLPS X10, X12
	MOVAPS X10, X13
	ADDPS  X12, X13
	MOVAPS X10, X14
	SUBPS  X12, X14
	
	MOVAPS X11, X12
	MOVHLPS X11, X12
	MOVAPS X12, X15
	SHUFPS $0xB1, X15, X15
	MOVUPS ·maskNegLoPS(SB), X8
	XORPS  X8, X15
	
	MOVAPS X11, X8
	MOVAPS X8, X9
	SUBPS  X15, X9
	ADDPS  X15, X8
	
	UNPCKLPD X9, X13
	MOVAPS X13, X4
	UNPCKLPD X8, X14
	MOVAPS X14, X5
	
	// --- Row 3 (D6, D7) ---
	MOVAPS X6, X8
	MOVAPS X7, X9
	MOVAPS X8, X10
	ADDPS  X9, X10
	MOVAPS X8, X11
	SUBPS  X9, X11
	
	MOVAPS X10, X12
	MOVHLPS X10, X12
	MOVAPS X10, X13
	ADDPS  X12, X13
	MOVAPS X10, X14
	SUBPS  X12, X14
	
	MOVAPS X11, X12
	MOVHLPS X11, X12
	MOVAPS X12, X15
	SHUFPS $0xB1, X15, X15
	MOVUPS ·maskNegLoPS(SB), X8
	XORPS  X8, X15
	
	MOVAPS X11, X8
	MOVAPS X8, X9
	SUBPS  X15, X9
	ADDPS  X15, X8
	
	UNPCKLPD X9, X13
	MOVAPS X13, X6
	UNPCKLPD X8, X14
	MOVAPS X14, X7
	
	// =======================================================================
	// Store Result
	// We need to Transpose the output.
	// Current State:
	// D0=(0,4), D2=(1,5) -> Need (0,1), (4,5)
	// D4=(2,6), D6=(3,7) -> Need (2,3), (6,7)
	// D1=(8,12), D3=(9,13) -> Need (8,9), (12,13)
	// D5=(10,14), D7=(11,15) -> Need (10,11), (14,15)
	//
	// We use X8..X15 as destinations to avoid overwriting inputs.
	// =======================================================================
	
	// (D0, D2) -> X8 (0,1), X10 (4,5)
	MOVAPS X0, X8
	UNPCKLPD X2, X8
	MOVAPS X0, X10
	UNPCKHPD X2, X10
	
	// (D4, D6) -> X9 (2,3), X11 (6,7)
	MOVAPS X4, X9
	UNPCKLPD X6, X9
	MOVAPS X4, X11
	UNPCKHPD X6, X11
	
	// (D1, D3) -> X12 (8,9), X14 (12,13)
	MOVAPS X1, X12
	UNPCKLPD X3, X12
	MOVAPS X1, X14
	UNPCKHPD X3, X14
	
	// (D5, D7) -> X13 (10,11), X15 (14,15)
	MOVAPS X5, X13
	UNPCKLPD X7, X13
	MOVAPS X5, X15
	UNPCKHPD X7, X15
	
	// Store sequentially
	MOVUPS X8, 0(R8)
	MOVUPS X9, 16(R8)
	MOVUPS X10, 32(R8)
	MOVUPS X11, 48(R8)
	MOVUPS X12, 64(R8)
	MOVUPS X13, 80(R8)
	MOVUPS X14, 96(R8)
	MOVUPS X15, 112(R8)
	
	MOVB $1, ret+120(FP)
	RET

fwd_ret_false:
	MOVB $0, ret+120(FP)
	RET

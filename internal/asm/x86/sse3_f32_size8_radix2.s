//go:build 386 && asm && !purego

// ===========================================================================
// SSE2 Size-8 Radix-2 FFT Kernels for x86/386 (complex64)
// ===========================================================================
//
// This is a port of the AMD64 SSE2 size-8 radix-2 kernels to 32-bit x86.
//
// IMPORTANT: x86 (32-bit) only has XMM0-XMM7 (not XMM8-XMM15 like AMD64)
// This implementation uses stack memory for temporary values when needed.
//
// Radix-2 FFT kernel for size 8.
//
// Stage 1 (radix-2): Four 2-point butterflies (stride 1)
// Stage 2 (radix-2): Four 2-point butterflies (stride 2)
// Stage 3 (radix-2): Four 2-point butterflies (stride 4)
//
// Bit-reversal for n=8: [0, 4, 2, 6, 1, 5, 3, 7]
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 8, complex64, radix-2 variant
// ===========================================================================
TEXT ·ForwardSSE3Size8Radix2Complex64Asm(SB), NOSPLIT, $96-61
	// ----------------------------------------------------------------
	// Stack layout:
	//   SP+0..15:   original dst pointer
	//   SP+16..31:  src pointer
	//   SP+32..47:  twiddle pointer
	//   SP+48..63:  scratch pointer
	//   SP+64..79:  bitrev pointer
	//   SP+80..83:  n
	//   SP+84..95:  temp storage (12 bytes, unused currently)
	// ----------------------------------------------------------------
	MOVL dst+0(FP), DI
	MOVL DI, 0(SP)           // Save original dst
	MOVL src+12(FP), SI
	MOVL SI, 4(SP)
	MOVL twiddle+24(FP), BX
	MOVL BX, 8(SP)
	MOVL scratch+36(FP), DX
	MOVL DX, 12(SP)
	MOVL bitrev+48(FP), BP
	MOVL BP, 16(SP)
	MOVL src+16(FP), AX
	MOVL AX, 20(SP)

	// Verify n == 8
	CMPL AX, $8
	JNE  size8_r2_sse2_386_fwd_return_false

	// Validate all slice lengths >= 8
	MOVL dst+4(FP), CX
	CMPL CX, $8
	JL   size8_r2_sse2_386_fwd_return_false

	MOVL twiddle+28(FP), CX
	CMPL CX, $8
	JL   size8_r2_sse2_386_fwd_return_false

	MOVL scratch+40(FP), CX
	CMPL CX, $8
	JL   size8_r2_sse2_386_fwd_return_false

	MOVL bitrev+52(FP), CX
	CMPL CX, $8
	JL   size8_r2_sse2_386_fwd_return_false

	// Select working buffer
	CMPL DI, SI
	JNE  size8_r2_sse2_386_fwd_use_dst
	MOVL DX, DI              // In-place: use scratch

size8_r2_sse2_386_fwd_use_dst:
	MOVL DI, 24(SP)          // Save working buffer pointer

	// ==================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// ==================================================================
	MOVL 16(SP), BP          // bitrev pointer
	MOVL 4(SP), SI           // src pointer

	MOVL (BP), CX
	MOVSD (SI)(CX*8), X0
	MOVL 4(BP), CX
	MOVSD (SI)(CX*8), X1
	MOVL 8(BP), CX
	MOVSD (SI)(CX*8), X2
	MOVL 12(BP), CX
	MOVSD (SI)(CX*8), X3
	MOVL 16(BP), CX
	MOVSD (SI)(CX*8), X4
	MOVL 20(BP), CX
	MOVSD (SI)(CX*8), X5
	MOVL 24(BP), CX
	MOVSD (SI)(CX*8), X6
	MOVL 28(BP), CX
	MOVSD (SI)(CX*8), X7

	// ==================================================================
	// Stage 1: 4 Radix-2 butterflies, stride 1, no twiddles
	// Using only XMM0-XMM7
	// ==================================================================

	// Butterfly: (X0, X1) - use X6 as temp
	MOVAPS X0, X6
	ADDPS  X1, X0            // X0 = a0
	SUBPS  X1, X6            // X6 = a1
	MOVAPS X6, X1

	// Butterfly: (X2, X3) - use X6 as temp
	MOVAPS X2, X6
	ADDPS  X3, X2            // X2 = a2
	SUBPS  X3, X6            // X6 = a3
	MOVAPS X6, X3

	// Butterfly: (X4, X5) - use X6 as temp
	MOVAPS X4, X6
	ADDPS  X5, X4            // X4 = a4
	SUBPS  X5, X6            // X6 = a5
	MOVAPS X6, X5

	// Butterfly: (X6, X7) - use X1 as temp (a1 not needed yet)
	MOVAPS X6, X1
	ADDPS  X7, X6            // X6 = a6
	SUBPS  X7, X1            // X1 = a7
	MOVAPS X1, X7

	// ==================================================================
	// Stage 2: 4 Radix-2 butterflies, stride 2
	// Butterflies on (X0, X2), (X1, X3), (X4, X6), (X5, X7)
	// Twiddles: 1, -i, 1, -i
	// ==================================================================

	// Load mask for -i multiplication (negate imaginary)
	MOVUPS ·maskNegHiPS(SB), X1  // Will use as temp mask

	// Butterfly (X0, X2) with twiddle 1
	MOVAPS X0, X7            // temp
	ADDPS  X2, X0            // b0 = a0 + a2
	SUBPS  X2, X7            // b2 = a0 - a2
	MOVAPS X7, X2

	// Recover a1 from Stage 1 - it's in... wait, we need to track this differently
	// Let me reorganize to preserve needed values

	// Actually, let's store intermediate results to memory when needed
	MOVL 24(SP), DI          // working buffer

	// Store Stage 1 results temporarily
	MOVSD X0, 28(SP)         // a0
	MOVSD X2, 36(SP)         // a2
	MOVSD X4, 44(SP)         // a4
	MOVSD X6, 52(SP)         // a6

	// Reload a1, a3, a5, a7 from registers (they're in X1, X3, X5, X7 from stage 1)
	// Wait - I overwrote X1 with the mask. Let me restart more carefully.

	// ==================================================================
	// Let me use a cleaner approach: process each stage completely
	// before moving to the next, storing to memory between stages
	// ==================================================================

size8_r2_sse2_386_fwd_restart:
	// Reload bit-reversed data
	MOVL 16(SP), BP
	MOVL 4(SP), SI

	MOVL (BP), CX
	MOVSD (SI)(CX*8), X0
	MOVL 4(BP), CX
	MOVSD (SI)(CX*8), X1
	MOVL 8(BP), CX
	MOVSD (SI)(CX*8), X2
	MOVL 12(BP), CX
	MOVSD (SI)(CX*8), X3
	MOVL 16(BP), CX
	MOVSD (SI)(CX*8), X4
	MOVL 20(BP), CX
	MOVSD (SI)(CX*8), X5
	MOVL 24(BP), CX
	MOVSD (SI)(CX*8), X6
	MOVL 28(BP), CX
	MOVSD (SI)(CX*8), X7

	MOVL 24(SP), DI          // working buffer

	// Stage 1: Store results to memory immediately
	// Butterfly (x0, x1)
	MOVAPS X0, X2            // temp
	ADDPS  X1, X0
	MOVSD  X0, (DI)          // a0
	SUBPS  X1, X2
	MOVSD  X2, 8(DI)         // a1

	// Reload for next butterfly
	MOVL 8(BP), CX
	MOVSD (SI)(CX*8), X0
	MOVL 12(BP), CX
	MOVSD (SI)(CX*8), X1

	// Butterfly (x2, x3)
	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, 16(DI)        // a2
	SUBPS  X1, X2
	MOVSD  X2, 24(DI)        // a3

	// Reload for next butterfly
	MOVL 16(BP), CX
	MOVSD (SI)(CX*8), X0
	MOVL 20(BP), CX
	MOVSD (SI)(CX*8), X1

	// Butterfly (x4, x5)
	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, 32(DI)        // a4
	SUBPS  X1, X2
	MOVSD  X2, 40(DI)        // a5

	// Reload for last butterfly
	MOVL 24(BP), CX
	MOVSD (SI)(CX*8), X0
	MOVL 28(BP), CX
	MOVSD (SI)(CX*8), X1

	// Butterfly (x6, x7)
	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, 48(DI)        // a6
	SUBPS  X1, X2
	MOVSD  X2, 56(DI)        // a7

	// ==================================================================
	// Stage 2: Read from memory, write back
	// ==================================================================

	// Butterfly (a0, a2) with twiddle 1
	MOVSD  (DI), X0          // a0
	MOVSD  16(DI), X1        // a2
	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, (DI)          // b0
	SUBPS  X1, X2
	MOVSD  X2, 16(DI)        // b2

	// Butterfly (a1, a3) with twiddle -i
	MOVSD  8(DI), X0         // a1
	MOVSD  24(DI), X1        // a3
	// t = a3 * (-i) = (im, -re)
	SHUFPS $0xB1, X1, X1     // swap
	MOVUPS ·maskNegHiPS(SB), X3
	XORPS  X3, X1            // negate high
	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, 8(DI)         // b1
	SUBPS  X1, X2
	MOVSD  X2, 24(DI)        // b3

	// Butterfly (a4, a6) with twiddle 1
	MOVSD  32(DI), X0        // a4
	MOVSD  48(DI), X1        // a6
	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, 32(DI)        // b4
	SUBPS  X1, X2
	MOVSD  X2, 48(DI)        // b6

	// Butterfly (a5, a7) with twiddle -i
	MOVSD  40(DI), X0        // a5
	MOVSD  56(DI), X1        // a7
	SHUFPS $0xB1, X1, X1
	XORPS  X3, X1            // X3 still has maskNegHiPS
	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, 40(DI)        // b5
	SUBPS  X1, X2
	MOVSD  X2, 56(DI)        // b7

	// ==================================================================
	// Stage 3: Final stage with twiddle factor multiplications
	// ==================================================================
	MOVL 8(SP), BX           // twiddle pointer

	// Butterfly (b0, b4) with twiddle 1
	MOVSD  (DI), X0          // b0
	MOVSD  32(DI), X1        // b4
	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, (DI)          // y0
	SUBPS  X1, X2
	MOVSD  X2, 32(DI)        // y4

	// Butterfly (b1, b5) with twiddle w1
	MOVSD  8(DI), X0         // b1
	MOVSD  40(DI), X1        // b5
	// Complex multiply: t = w1 * b5
	MOVSD  8(BX), X2         // w1
	MOVAPS X2, X3
	SHUFPS $0x00, X3, X3     // w1.re
	MOVAPS X2, X4
	SHUFPS $0x55, X4, X4     // w1.im
	MOVAPS X1, X5
	SHUFPS $0xB1, X5, X5     // b5 swapped
	MOVAPS X1, X6
	MULPS  X3, X6            // b5 * w1.re
	MULPS  X4, X5            // b5_swapped * w1.im
	ADDSUBPS X5, X6          // complex multiply result
	// Butterfly
	MOVAPS X0, X2
	ADDPS  X6, X0
	MOVSD  X0, 8(DI)         // y1
	SUBPS  X6, X2
	MOVSD  X2, 40(DI)        // y5

	// Butterfly (b2, b6) with twiddle w2
	MOVSD  16(DI), X0        // b2
	MOVSD  48(DI), X1        // b6
	MOVSD  16(BX), X2        // w2
	MOVAPS X2, X3
	SHUFPS $0x00, X3, X3
	MOVAPS X2, X4
	SHUFPS $0x55, X4, X4
	MOVAPS X1, X5
	SHUFPS $0xB1, X5, X5
	MOVAPS X1, X6
	MULPS  X3, X6
	MULPS  X4, X5
	ADDSUBPS X5, X6
	MOVAPS X0, X2
	ADDPS  X6, X0
	MOVSD  X0, 16(DI)        // y2
	SUBPS  X6, X2
	MOVSD  X2, 48(DI)        // y6

	// Butterfly (b3, b7) with twiddle w3
	MOVSD  24(DI), X0        // b3
	MOVSD  56(DI), X1        // b7
	MOVSD  24(BX), X2        // w3
	MOVAPS X2, X3
	SHUFPS $0x00, X3, X3
	MOVAPS X2, X4
	SHUFPS $0x55, X4, X4
	MOVAPS X1, X5
	SHUFPS $0xB1, X5, X5
	MOVAPS X1, X6
	MULPS  X3, X6
	MULPS  X4, X5
	ADDSUBPS X5, X6
	MOVAPS X0, X2
	ADDPS  X6, X0
	MOVSD  X0, 24(DI)        // y3
	SUBPS  X6, X2
	MOVSD  X2, 56(DI)        // y7

	// ==================================================================
	// Copy to dst if needed
	// ==================================================================
	MOVL 0(SP), SI           // original dst
	CMPL DI, SI
	JE   size8_r2_sse2_386_fwd_done

	// Copy 64 bytes (8 × complex64)
	MOVL (DI), AX
	MOVL AX, (SI)
	MOVL 4(DI), AX
	MOVL AX, 4(SI)
	MOVL 8(DI), AX
	MOVL AX, 8(SI)
	MOVL 12(DI), AX
	MOVL AX, 12(SI)
	MOVL 16(DI), AX
	MOVL AX, 16(SI)
	MOVL 20(DI), AX
	MOVL AX, 20(SI)
	MOVL 24(DI), AX
	MOVL AX, 24(SI)
	MOVL 28(DI), AX
	MOVL AX, 28(SI)
	MOVL 32(DI), AX
	MOVL AX, 32(SI)
	MOVL 36(DI), AX
	MOVL AX, 36(SI)
	MOVL 40(DI), AX
	MOVL AX, 40(SI)
	MOVL 44(DI), AX
	MOVL AX, 44(SI)
	MOVL 48(DI), AX
	MOVL AX, 48(SI)
	MOVL 52(DI), AX
	MOVL AX, 52(SI)
	MOVL 56(DI), AX
	MOVL AX, 56(SI)
	MOVL 60(DI), AX
	MOVL AX, 60(SI)

size8_r2_sse2_386_fwd_done:
	MOVB $1, ret+60(FP)
	RET

size8_r2_sse2_386_fwd_return_false:
	MOVB $0, ret+60(FP)
	RET

// ===========================================================================
// Inverse transform, size 8, complex64, radix-2 variant
// ===========================================================================
TEXT ·InverseSSE3Size8Radix2Complex64Asm(SB), NOSPLIT, $96-61
	MOVL dst+0(FP), DI
	MOVL DI, 0(SP)
	MOVL src+12(FP), SI
	MOVL SI, 4(SP)
	MOVL twiddle+24(FP), BX
	MOVL BX, 8(SP)
	MOVL scratch+36(FP), DX
	MOVL DX, 12(SP)
	MOVL bitrev+48(FP), BP
	MOVL BP, 16(SP)
	MOVL src+16(FP), AX
	MOVL AX, 20(SP)

	// Verify n == 8
	CMPL AX, $8
	JNE  size8_r2_sse2_386_inv_return_false

	// Validate slice lengths
	MOVL dst+4(FP), CX
	CMPL CX, $8
	JL   size8_r2_sse2_386_inv_return_false

	MOVL twiddle+28(FP), CX
	CMPL CX, $8
	JL   size8_r2_sse2_386_inv_return_false

	MOVL scratch+40(FP), CX
	CMPL CX, $8
	JL   size8_r2_sse2_386_inv_return_false

	MOVL bitrev+52(FP), CX
	CMPL CX, $8
	JL   size8_r2_sse2_386_inv_return_false

	// Select working buffer
	CMPL DI, SI
	JNE  size8_r2_sse2_386_inv_use_dst
	MOVL DX, DI

size8_r2_sse2_386_inv_use_dst:
	MOVL DI, 24(SP)

	// ==================================================================
	// Bit-reversal and Stage 1
	// ==================================================================
	MOVL 16(SP), BP
	MOVL 4(SP), SI

	MOVL (BP), CX
	MOVSD (SI)(CX*8), X0
	MOVL 4(BP), CX
	MOVSD (SI)(CX*8), X1

	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, (DI)
	SUBPS  X1, X2
	MOVSD  X2, 8(DI)

	MOVL 8(BP), CX
	MOVSD (SI)(CX*8), X0
	MOVL 12(BP), CX
	MOVSD (SI)(CX*8), X1

	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, 16(DI)
	SUBPS  X1, X2
	MOVSD  X2, 24(DI)

	MOVL 16(BP), CX
	MOVSD (SI)(CX*8), X0
	MOVL 20(BP), CX
	MOVSD (SI)(CX*8), X1

	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, 32(DI)
	SUBPS  X1, X2
	MOVSD  X2, 40(DI)

	MOVL 24(BP), CX
	MOVSD (SI)(CX*8), X0
	MOVL 28(BP), CX
	MOVSD (SI)(CX*8), X1

	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, 48(DI)
	SUBPS  X1, X2
	MOVSD  X2, 56(DI)

	// ==================================================================
	// Stage 2: with twiddle i (not -i)
	// ==================================================================

	// Butterfly (a0, a2) with twiddle 1
	MOVSD  (DI), X0
	MOVSD  16(DI), X1
	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, (DI)
	SUBPS  X1, X2
	MOVSD  X2, 16(DI)

	// Butterfly (a1, a3) with twiddle i
	MOVSD  8(DI), X0
	MOVSD  24(DI), X1
	// t = a3 * (i) = (-im, re)
	SHUFPS $0xB1, X1, X1
	MOVUPS ·maskNegLoPS(SB), X3
	XORPS  X3, X1
	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, 8(DI)
	SUBPS  X1, X2
	MOVSD  X2, 24(DI)

	// Butterfly (a4, a6) with twiddle 1
	MOVSD  32(DI), X0
	MOVSD  48(DI), X1
	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, 32(DI)
	SUBPS  X1, X2
	MOVSD  X2, 48(DI)

	// Butterfly (a5, a7) with twiddle i
	MOVSD  40(DI), X0
	MOVSD  56(DI), X1
	SHUFPS $0xB1, X1, X1
	XORPS  X3, X1
	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, 40(DI)
	SUBPS  X1, X2
	MOVSD  X2, 56(DI)

	// ==================================================================
	// Stage 3: with conjugate twiddles
	// ==================================================================
	MOVL 8(SP), BX
	MOVUPS ·maskNegHiPS(SB), X7  // for conjugation

	// Butterfly (b0, b4) with twiddle 1
	MOVSD  (DI), X0
	MOVSD  32(DI), X1
	MOVAPS X0, X2
	ADDPS  X1, X0
	MOVSD  X0, (DI)
	SUBPS  X1, X2
	MOVSD  X2, 32(DI)

	// Butterfly (b1, b5) with conj(w1)
	MOVSD  8(DI), X0
	MOVSD  40(DI), X1
	MOVSD  8(BX), X2
	XORPS  X7, X2            // conjugate
	MOVAPS X2, X3
	SHUFPS $0x00, X3, X3
	MOVAPS X2, X4
	SHUFPS $0x55, X4, X4
	MOVAPS X1, X5
	SHUFPS $0xB1, X5, X5
	MOVAPS X1, X6
	MULPS  X3, X6
	MULPS  X4, X5
	ADDSUBPS X5, X6
	MOVAPS X0, X2
	ADDPS  X6, X0
	MOVSD  X0, 8(DI)
	SUBPS  X6, X2
	MOVSD  X2, 40(DI)

	// Butterfly (b2, b6) with conj(w2)
	MOVSD  16(DI), X0
	MOVSD  48(DI), X1
	MOVSD  16(BX), X2
	XORPS  X7, X2
	MOVAPS X2, X3
	SHUFPS $0x00, X3, X3
	MOVAPS X2, X4
	SHUFPS $0x55, X4, X4
	MOVAPS X1, X5
	SHUFPS $0xB1, X5, X5
	MOVAPS X1, X6
	MULPS  X3, X6
	MULPS  X4, X5
	ADDSUBPS X5, X6
	MOVAPS X0, X2
	ADDPS  X6, X0
	MOVSD  X0, 16(DI)
	SUBPS  X6, X2
	MOVSD  X2, 48(DI)

	// Butterfly (b3, b7) with conj(w3)
	MOVSD  24(DI), X0
	MOVSD  56(DI), X1
	MOVSD  24(BX), X2
	XORPS  X7, X2
	MOVAPS X2, X3
	SHUFPS $0x00, X3, X3
	MOVAPS X2, X4
	SHUFPS $0x55, X4, X4
	MOVAPS X1, X5
	SHUFPS $0xB1, X5, X5
	MOVAPS X1, X6
	MULPS  X3, X6
	MULPS  X4, X5
	ADDSUBPS X5, X6
	MOVAPS X0, X2
	ADDPS  X6, X0
	MOVSD  X0, 24(DI)
	SUBPS  X6, X2
	MOVSD  X2, 56(DI)

	// ==================================================================
	// Apply 1/8 scaling and copy if needed
	// ==================================================================
	MOVSS  ·eighth32(SB), X7
	SHUFPS $0x00, X7, X7

	MOVL 0(SP), SI           // original dst
	CMPL DI, SI
	JE   size8_r2_sse2_386_inv_scale_inplace

	// Scale and copy
	MOVSD  (DI), X0
	MULPS  X7, X0
	MOVSD  X0, (SI)
	MOVSD  8(DI), X0
	MULPS  X7, X0
	MOVSD  X0, 8(SI)
	MOVSD  16(DI), X0
	MULPS  X7, X0
	MOVSD  X0, 16(SI)
	MOVSD  24(DI), X0
	MULPS  X7, X0
	MOVSD  X0, 24(SI)
	MOVSD  32(DI), X0
	MULPS  X7, X0
	MOVSD  X0, 32(SI)
	MOVSD  40(DI), X0
	MULPS  X7, X0
	MOVSD  X0, 40(SI)
	MOVSD  48(DI), X0
	MULPS  X7, X0
	MOVSD  X0, 48(SI)
	MOVSD  56(DI), X0
	MULPS  X7, X0
	MOVSD  X0, 56(SI)
	JMP  size8_r2_sse2_386_inv_done

size8_r2_sse2_386_inv_scale_inplace:
	// Scale in place
	MOVSD  (DI), X0
	MULPS  X7, X0
	MOVSD  X0, (DI)
	MOVSD  8(DI), X0
	MULPS  X7, X0
	MOVSD  X0, 8(DI)
	MOVSD  16(DI), X0
	MULPS  X7, X0
	MOVSD  X0, 16(DI)
	MOVSD  24(DI), X0
	MULPS  X7, X0
	MOVSD  X0, 24(DI)
	MOVSD  32(DI), X0
	MULPS  X7, X0
	MOVSD  X0, 32(DI)
	MOVSD  40(DI), X0
	MULPS  X7, X0
	MOVSD  X0, 40(DI)
	MOVSD  48(DI), X0
	MULPS  X7, X0
	MOVSD  X0, 48(DI)
	MOVSD  56(DI), X0
	MULPS  X7, X0
	MOVSD  X0, 56(DI)

size8_r2_sse2_386_inv_done:
	MOVB $1, ret+60(FP)
	RET

size8_r2_sse2_386_inv_return_false:
	MOVB $0, ret+60(FP)
	RET

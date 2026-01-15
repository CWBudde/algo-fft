//go:build 386 && asm && !purego

// ===========================================================================
// SSE (SSE1) Size-8 Radix-2 FFT Kernels for 386 (complex64)
// ===========================================================================
//
// Fully-unrolled radix-2 FFT kernel for size 8 using only SSE instructions.
// (Compatible with Pentium III and newer, or older CPUs with SSE support).
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
// func ForwardSSESize8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
// ===========================================================================
TEXT ·ForwardSSESize8Radix2Complex64Asm(SB), NOSPLIT, $96-61
	// Stack layout for 386 (8 XMM registers only):
	//   SP+0..3:   original dst pointer
	//   SP+4..7:   src pointer
	//   SP+8..11:  twiddle pointer
	//   SP+12..15: scratch pointer
	//   SP+16..19: bitrev pointer
	//   SP+20..23: n
	//   SP+24..27: working buffer pointer
	//   SP+28..91: temp storage for intermediate values (64 bytes)

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
	JNE  size8_sse_fwd_return_false

	// Validate all slice lengths >= 8
	MOVL dst+4(FP), CX
	CMPL CX, $8
	JL   size8_sse_fwd_return_false

	MOVL twiddle+28(FP), CX
	CMPL CX, $8
	JL   size8_sse_fwd_return_false

	MOVL scratch+40(FP), CX
	CMPL CX, $8
	JL   size8_sse_fwd_return_false

	MOVL bitrev+52(FP), CX
	CMPL CX, $8
	JL   size8_sse_fwd_return_false

	// Select working buffer (use scratch for in-place)
	CMPL DI, SI
	JNE  size8_sse_fwd_use_dst
	MOVL DX, DI

size8_sse_fwd_use_dst:
	MOVL DI, 24(SP)          // working buffer

	// ==================================================================
	// Bit-reversal and Stage 1 combined
	// Load from src with bit-reversal, do first stage butterfly
	// ==================================================================
	MOVL 16(SP), BP
	MOVL 4(SP), SI

	// Butterfly (x0, x1) - indices 0,4 in bit-reversed
	MOVL (BP), CX
	MOVSD (SI)(CX*8), X0     // x0
	MOVL 4(BP), CX
	MOVSD (SI)(CX*8), X1     // x1
	MOVAPS X0, X2
	ADDPS  X1, X0            // a0 = x0 + x1
	SUBPS  X1, X2            // a1 = x0 - x1
	MOVSD  X0, 28(SP)
	MOVSD  X2, 36(SP)

	// Butterfly (x2, x3) - indices 2,6 in bit-reversed
	MOVL 8(BP), CX
	MOVSD (SI)(CX*8), X0     // x2
	MOVL 12(BP), CX
	MOVSD (SI)(CX*8), X1     // x3
	MOVAPS X0, X2
	ADDPS  X1, X0            // a2 = x2 + x3
	SUBPS  X1, X2            // a3 = x2 - x3
	MOVSD  X0, 44(SP)
	MOVSD  X2, 52(SP)

	// Butterfly (x4, x5) - indices 1,5 in bit-reversed
	MOVL 16(BP), CX
	MOVSD (SI)(CX*8), X0     // x4
	MOVL 20(BP), CX
	MOVSD (SI)(CX*8), X1     // x5
	MOVAPS X0, X2
	ADDPS  X1, X0            // a4 = x4 + x5
	SUBPS  X1, X2            // a5 = x4 - x5
	MOVSD  X0, 60(SP)
	MOVSD  X2, 68(SP)

	// Butterfly (x6, x7) - indices 3,7 in bit-reversed
	MOVL 24(BP), CX
	MOVSD (SI)(CX*8), X0     // x6
	MOVL 28(BP), CX
	MOVSD (SI)(CX*8), X1     // x7
	MOVAPS X0, X2
	ADDPS  X1, X0            // a6 = x6 + x7
	SUBPS  X1, X2            // a7 = x6 - x7
	MOVSD  X0, 76(SP)
	MOVSD  X2, 84(SP)

	// ==================================================================
	// Stage 2: 4 Radix-2 butterflies, stride 2
	// Twiddles: 1, -i, 1, -i
	// ==================================================================

	// Butterfly (a0, a2) with twiddle 1
	MOVSD  28(SP), X0        // a0
	MOVSD  44(SP), X1        // a2
	MOVAPS X0, X2
	ADDPS  X1, X0            // b0 = a0 + a2
	SUBPS  X1, X2            // b2 = a0 - a2
	MOVSD  X0, 28(SP)
	MOVSD  X2, 44(SP)

	// Butterfly (a1, a3) with twiddle -i
	// t = a3 * (-i) = (im, -re)
	MOVSD  36(SP), X0        // a1
	MOVSD  52(SP), X1        // a3
	SHUFPS $0xB1, X1, X1     // swap re/im
	MOVUPS ·maskNegHiPS(SB), X3
	XORPS  X3, X1            // negate im -> (im, -re) = a3 * (-i)
	MOVAPS X0, X2
	ADDPS  X1, X0            // b1 = a1 + a3*(-i)
	SUBPS  X1, X2            // b3 = a1 - a3*(-i)
	MOVSD  X0, 36(SP)
	MOVSD  X2, 52(SP)

	// Butterfly (a4, a6) with twiddle 1
	MOVSD  60(SP), X0        // a4
	MOVSD  76(SP), X1        // a6
	MOVAPS X0, X2
	ADDPS  X1, X0            // b4 = a4 + a6
	SUBPS  X1, X2            // b6 = a4 - a6
	MOVSD  X0, 60(SP)
	MOVSD  X2, 76(SP)

	// Butterfly (a5, a7) with twiddle -i
	MOVSD  68(SP), X0        // a5
	MOVSD  84(SP), X1        // a7
	SHUFPS $0xB1, X1, X1
	XORPS  X3, X1            // X3 still has maskNegHiPS
	MOVAPS X0, X2
	ADDPS  X1, X0            // b5 = a5 + a7*(-i)
	SUBPS  X1, X2            // b7 = a5 - a7*(-i)
	MOVSD  X0, 68(SP)
	MOVSD  X2, 84(SP)

	// ==================================================================
	// Stage 3: Final stage with twiddle factor multiplications
	// Twiddles: 1, w1, w2, w3
	// ==================================================================
	MOVL 8(SP), BX           // twiddle pointer
	MOVL 24(SP), DI          // working buffer

	// Butterfly (b0, b4) with twiddle 1
	MOVSD  28(SP), X0        // b0
	MOVSD  60(SP), X1        // b4
	MOVAPS X0, X2
	ADDPS  X1, X0            // y0 = b0 + b4
	SUBPS  X1, X2            // y4 = b0 - b4
	MOVSD  X0, (DI)
	MOVSD  X2, 32(DI)

	// Butterfly (b1, b5) with twiddle w1
	// SSE1 complex multiply: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
	// Without ADDSUBPS, we need separate add and sub
	MOVSD  36(SP), X0        // b1
	MOVSD  68(SP), X1        // b5
	MOVSD  8(BX), X2         // w1 = (c, d)

	// Broadcast c and d
	MOVAPS X2, X3
	SHUFPS $0x00, X3, X3     // X3 = (c, c)
	MOVAPS X2, X4
	SHUFPS $0x55, X4, X4     // X4 = (d, d)

	// X1 = (a, b) = b5
	MOVAPS X1, X5
	MULPS  X3, X5            // X5 = (ac, bc)
	MOVAPS X1, X6
	SHUFPS $0xB1, X6, X6     // X6 = (b, a)
	MULPS  X4, X6            // X6 = (bd, ad)

	// Real part: ac - bd, Imag part: bc + ad
	// Need (ac-bd, bc+ad)
	// X5 = (ac, bc), X6 = (bd, ad)
	MOVAPS X5, X7
	SHUFPS $0xA0, X7, X7     // X7 = (ac, ac)
	MOVAPS X6, X1
	SHUFPS $0xA0, X1, X1     // X1 = (bd, bd)
	SUBPS  X1, X7            // X7 = (ac-bd, ac-bd) - real

	MOVAPS X5, X1
	SHUFPS $0xF5, X1, X1     // X1 = (bc, bc)
	MOVAPS X6, X2
	SHUFPS $0xF5, X2, X2     // X2 = (ad, ad)
	ADDPS  X2, X1            // X1 = (bc+ad, bc+ad) - imag

	// Combine: (real, imag)
	MOVLPS X7, X1            // X1 low = real, high = imag (wrong order)
	SHUFPS $0x08, X1, X7     // X7 = (ac-bd, bc+ad)

	// Butterfly with result
	MOVAPS X0, X2
	ADDPS  X7, X0            // y1 = b1 + t
	SUBPS  X7, X2            // y5 = b1 - t
	MOVSD  X0, 8(DI)
	MOVSD  X2, 40(DI)

	// Butterfly (b2, b6) with twiddle w2
	MOVSD  44(SP), X0        // b2
	MOVSD  76(SP), X1        // b6
	MOVSD  16(BX), X2        // w2

	MOVAPS X2, X3
	SHUFPS $0x00, X3, X3
	MOVAPS X2, X4
	SHUFPS $0x55, X4, X4
	MOVAPS X1, X5
	MULPS  X3, X5
	MOVAPS X1, X6
	SHUFPS $0xB1, X6, X6
	MULPS  X4, X6

	MOVAPS X5, X7
	SHUFPS $0xA0, X7, X7
	MOVAPS X6, X1
	SHUFPS $0xA0, X1, X1
	SUBPS  X1, X7

	MOVAPS X5, X1
	SHUFPS $0xF5, X1, X1
	MOVAPS X6, X2
	SHUFPS $0xF5, X2, X2
	ADDPS  X2, X1

	MOVLPS X7, X1
	SHUFPS $0x08, X1, X7

	MOVAPS X0, X2
	ADDPS  X7, X0
	SUBPS  X7, X2
	MOVSD  X0, 16(DI)
	MOVSD  X2, 48(DI)

	// Butterfly (b3, b7) with twiddle w3
	MOVSD  52(SP), X0        // b3
	MOVSD  84(SP), X1        // b7
	MOVSD  24(BX), X2        // w3

	MOVAPS X2, X3
	SHUFPS $0x00, X3, X3
	MOVAPS X2, X4
	SHUFPS $0x55, X4, X4
	MOVAPS X1, X5
	MULPS  X3, X5
	MOVAPS X1, X6
	SHUFPS $0xB1, X6, X6
	MULPS  X4, X6

	MOVAPS X5, X7
	SHUFPS $0xA0, X7, X7
	MOVAPS X6, X1
	SHUFPS $0xA0, X1, X1
	SUBPS  X1, X7

	MOVAPS X5, X1
	SHUFPS $0xF5, X1, X1
	MOVAPS X6, X2
	SHUFPS $0xF5, X2, X2
	ADDPS  X2, X1

	MOVLPS X7, X1
	SHUFPS $0x08, X1, X7

	MOVAPS X0, X2
	ADDPS  X7, X0
	SUBPS  X7, X2
	MOVSD  X0, 24(DI)
	MOVSD  X2, 56(DI)

	// ==================================================================
	// Copy to dst if needed
	// ==================================================================
	MOVL 0(SP), SI           // original dst
	CMPL DI, SI
	JE   size8_sse_fwd_done

	// Copy 64 bytes
	MOVUPS (DI), X0
	MOVUPS X0, (SI)
	MOVUPS 16(DI), X0
	MOVUPS X0, 16(SI)
	MOVUPS 32(DI), X0
	MOVUPS X0, 32(SI)
	MOVUPS 48(DI), X0
	MOVUPS X0, 48(SI)

size8_sse_fwd_done:
	MOVB $1, ret+60(FP)
	RET

size8_sse_fwd_return_false:
	MOVB $0, ret+60(FP)
	RET

// ===========================================================================
// Inverse transform, size 8, complex64, radix-2 variant
// func InverseSSESize8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
// ===========================================================================
TEXT ·InverseSSESize8Radix2Complex64Asm(SB), NOSPLIT, $96-61
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
	JNE  size8_sse_inv_return_false

	// Validate slice lengths
	MOVL dst+4(FP), CX
	CMPL CX, $8
	JL   size8_sse_inv_return_false

	MOVL twiddle+28(FP), CX
	CMPL CX, $8
	JL   size8_sse_inv_return_false

	MOVL scratch+40(FP), CX
	CMPL CX, $8
	JL   size8_sse_inv_return_false

	MOVL bitrev+52(FP), CX
	CMPL CX, $8
	JL   size8_sse_inv_return_false

	// Select working buffer
	CMPL DI, SI
	JNE  size8_sse_inv_use_dst
	MOVL DX, DI

size8_sse_inv_use_dst:
	MOVL DI, 24(SP)

	// ==================================================================
	// Bit-reversal and Stage 1
	// ==================================================================
	MOVL 16(SP), BP
	MOVL 4(SP), SI

	// Butterfly (x0, x1)
	MOVL (BP), CX
	MOVSD (SI)(CX*8), X0
	MOVL 4(BP), CX
	MOVSD (SI)(CX*8), X1
	MOVAPS X0, X2
	ADDPS  X1, X0
	SUBPS  X1, X2
	MOVSD  X0, 28(SP)
	MOVSD  X2, 36(SP)

	// Butterfly (x2, x3)
	MOVL 8(BP), CX
	MOVSD (SI)(CX*8), X0
	MOVL 12(BP), CX
	MOVSD (SI)(CX*8), X1
	MOVAPS X0, X2
	ADDPS  X1, X0
	SUBPS  X1, X2
	MOVSD  X0, 44(SP)
	MOVSD  X2, 52(SP)

	// Butterfly (x4, x5)
	MOVL 16(BP), CX
	MOVSD (SI)(CX*8), X0
	MOVL 20(BP), CX
	MOVSD (SI)(CX*8), X1
	MOVAPS X0, X2
	ADDPS  X1, X0
	SUBPS  X1, X2
	MOVSD  X0, 60(SP)
	MOVSD  X2, 68(SP)

	// Butterfly (x6, x7)
	MOVL 24(BP), CX
	MOVSD (SI)(CX*8), X0
	MOVL 28(BP), CX
	MOVSD (SI)(CX*8), X1
	MOVAPS X0, X2
	ADDPS  X1, X0
	SUBPS  X1, X2
	MOVSD  X0, 76(SP)
	MOVSD  X2, 84(SP)

	// ==================================================================
	// Stage 2: with twiddle i (not -i for inverse)
	// ==================================================================

	// Butterfly (a0, a2) with twiddle 1
	MOVSD  28(SP), X0
	MOVSD  44(SP), X1
	MOVAPS X0, X2
	ADDPS  X1, X0
	SUBPS  X1, X2
	MOVSD  X0, 28(SP)
	MOVSD  X2, 44(SP)

	// Butterfly (a1, a3) with twiddle i
	// t = a3 * i = (-im, re)
	MOVSD  36(SP), X0
	MOVSD  52(SP), X1
	SHUFPS $0xB1, X1, X1
	MOVUPS ·maskNegLoPS(SB), X3
	XORPS  X3, X1            // negate re -> (-im, re) = a3 * i
	MOVAPS X0, X2
	ADDPS  X1, X0
	SUBPS  X1, X2
	MOVSD  X0, 36(SP)
	MOVSD  X2, 52(SP)

	// Butterfly (a4, a6) with twiddle 1
	MOVSD  60(SP), X0
	MOVSD  76(SP), X1
	MOVAPS X0, X2
	ADDPS  X1, X0
	SUBPS  X1, X2
	MOVSD  X0, 60(SP)
	MOVSD  X2, 76(SP)

	// Butterfly (a5, a7) with twiddle i
	MOVSD  68(SP), X0
	MOVSD  84(SP), X1
	SHUFPS $0xB1, X1, X1
	XORPS  X3, X1
	MOVAPS X0, X2
	ADDPS  X1, X0
	SUBPS  X1, X2
	MOVSD  X0, 68(SP)
	MOVSD  X2, 84(SP)

	// ==================================================================
	// Stage 3: with conjugate twiddles
	// ==================================================================
	MOVL 8(SP), BX
	MOVL 24(SP), DI
	MOVUPS ·maskNegHiPS(SB), X7  // for conjugation

	// Butterfly (b0, b4) with twiddle 1
	MOVSD  28(SP), X0
	MOVSD  60(SP), X1
	MOVAPS X0, X2
	ADDPS  X1, X0
	SUBPS  X1, X2
	MOVSD  X0, (DI)
	MOVSD  X2, 32(DI)

	// Butterfly (b1, b5) with conj(w1)
	MOVSD  36(SP), X0
	MOVSD  68(SP), X1
	MOVSD  8(BX), X2
	XORPS  X7, X2            // conjugate

	MOVAPS X2, X3
	SHUFPS $0x00, X3, X3
	MOVAPS X2, X4
	SHUFPS $0x55, X4, X4
	MOVAPS X1, X5
	MULPS  X3, X5
	MOVAPS X1, X6
	SHUFPS $0xB1, X6, X6
	MULPS  X4, X6

	MOVAPS X5, X7
	SHUFPS $0xA0, X7, X7
	MOVAPS X6, X1
	SHUFPS $0xA0, X1, X1
	SUBPS  X1, X7

	MOVAPS X5, X1
	SHUFPS $0xF5, X1, X1
	MOVAPS X6, X2
	SHUFPS $0xF5, X2, X2
	ADDPS  X2, X1

	MOVLPS X7, X1
	SHUFPS $0x08, X1, X7

	MOVAPS X0, X2
	ADDPS  X7, X0
	SUBPS  X7, X2
	MOVSD  X0, 8(DI)
	MOVSD  X2, 40(DI)

	// Reload mask after using X7
	MOVUPS ·maskNegHiPS(SB), X7

	// Butterfly (b2, b6) with conj(w2)
	MOVSD  44(SP), X0
	MOVSD  76(SP), X1
	MOVSD  16(BX), X2
	XORPS  X7, X2

	MOVAPS X2, X3
	SHUFPS $0x00, X3, X3
	MOVAPS X2, X4
	SHUFPS $0x55, X4, X4
	MOVAPS X1, X5
	MULPS  X3, X5
	MOVAPS X1, X6
	SHUFPS $0xB1, X6, X6
	MULPS  X4, X6

	MOVAPS X5, X7
	SHUFPS $0xA0, X7, X7
	MOVAPS X6, X1
	SHUFPS $0xA0, X1, X1
	SUBPS  X1, X7

	MOVAPS X5, X1
	SHUFPS $0xF5, X1, X1
	MOVAPS X6, X2
	SHUFPS $0xF5, X2, X2
	ADDPS  X2, X1

	MOVLPS X7, X1
	SHUFPS $0x08, X1, X7

	MOVAPS X0, X2
	ADDPS  X7, X0
	SUBPS  X7, X2
	MOVSD  X0, 16(DI)
	MOVSD  X2, 48(DI)

	MOVUPS ·maskNegHiPS(SB), X7

	// Butterfly (b3, b7) with conj(w3)
	MOVSD  52(SP), X0
	MOVSD  84(SP), X1
	MOVSD  24(BX), X2
	XORPS  X7, X2

	MOVAPS X2, X3
	SHUFPS $0x00, X3, X3
	MOVAPS X2, X4
	SHUFPS $0x55, X4, X4
	MOVAPS X1, X5
	MULPS  X3, X5
	MOVAPS X1, X6
	SHUFPS $0xB1, X6, X6
	MULPS  X4, X6

	MOVAPS X5, X7
	SHUFPS $0xA0, X7, X7
	MOVAPS X6, X1
	SHUFPS $0xA0, X1, X1
	SUBPS  X1, X7

	MOVAPS X5, X1
	SHUFPS $0xF5, X1, X1
	MOVAPS X6, X2
	SHUFPS $0xF5, X2, X2
	ADDPS  X2, X1

	MOVLPS X7, X1
	SHUFPS $0x08, X1, X7

	MOVAPS X0, X2
	ADDPS  X7, X0
	SUBPS  X7, X2
	MOVSD  X0, 24(DI)
	MOVSD  X2, 56(DI)

	// ==================================================================
	// Apply 1/8 scaling and copy if needed
	// ==================================================================
	MOVSS  ·eighth32(SB), X7
	SHUFPS $0x00, X7, X7

	MOVL 0(SP), SI           // original dst
	CMPL DI, SI
	JE   size8_sse_inv_scale_inplace

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
	JMP  size8_sse_inv_done

size8_sse_inv_scale_inplace:
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

size8_sse_inv_done:
	MOVB $1, ret+60(FP)
	RET

size8_sse_inv_return_false:
	MOVB $0, ret+60(FP)
	RET

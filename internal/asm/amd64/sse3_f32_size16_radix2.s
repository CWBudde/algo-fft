//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-16 Radix-2 FFT Kernels for AMD64 (complex64)
// ===========================================================================
//
// Radix-2 FFT kernel for size 16.
//
// Stage 1 (radix-2): 8 butterflies (stride 1)
// Stage 2 (radix-2): 8 butterflies (stride 2)
// Stage 3 (radix-2): 8 butterflies (stride 4)
// Stage 4 (radix-2): 8 butterflies (stride 8)
//
// Register Allocation:
//   R8  = work buffer pointer (dst or scratch for in-place)
//   R9  = src pointer
//   R10 = twiddle pointer
//   R11 = scratch pointer
//   R12 = bitrev table pointer
//   R13 = n (size)
//   R14 = dst pointer (for final copy)
//   SI  = current work buffer position
//   CX  = loop counter
//   DX  = bitrev index temp
//   AX  = validation temp
//
//   X0-X7   = data elements
//   X8-X9   = butterfly results (a+b, a-b)
//   X10-X14 = twiddle multiplication temps
//   X15     = mask constant
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 16, complex64, radix-2 variant
// ===========================================================================
TEXT ·ForwardSSE3Size16Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// ===== Parameter Loading =====
	MOVQ dst+0(FP), R8       // R8  = dst slice data pointer
	MOVQ src+24(FP), R9      // R9  = src slice data pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle slice data pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch slice data pointer
	MOVQ src+32(FP), R13     // R13 = src slice length (should be 16)

	// ===== Input Validation =====
	CMPQ R13, $16            // check n == 16
	JNE  size16_r2_sse2_fwd_return_false

	MOVQ dst+8(FP), AX       // dst.len
	CMPQ AX, $16             // dst.len >= 16?
	JL   size16_r2_sse2_fwd_return_false

	MOVQ twiddle+56(FP), AX  // twiddle.len
	CMPQ AX, $16             // twiddle.len >= 16?
	JL   size16_r2_sse2_fwd_return_false

	MOVQ scratch+80(FP), AX  // scratch.len
	CMPQ AX, $16             // scratch.len >= 16?
	JL   size16_r2_sse2_fwd_return_false

	MOVQ $16, AX             // sanity check
	CMPQ AX, $16
	JL   size16_r2_sse2_fwd_return_false

	// ===== Select Working Buffer =====
	// If dst == src (in-place), use scratch as work buffer
	CMPQ R8, R9              // dst == src?
	JNE  size16_r2_sse2_fwd_use_dst
	MOVQ R11, R8             // in-place: work in scratch buffer

size16_r2_sse2_fwd_use_dst:
	// ==================================================================
	// STAGES 1 & 2 (Combined with bit-reversal, fully unrolled)
	// ==================================================================
	// Unrolled loop processing all 16 elements with inlined bit-reversal
	// Stage 1 stride 1: butterflies (0,1), (2,3)
	// Stage 2 stride 2: butterflies (0,2), (1,3) with twiddles [1, -i]
	// Bit-reversal pattern: 0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer write position
	MOVUPS ·maskNegHiPS(SB), X15 // X15 = [0, 0x80000000, 0, 0x80000000] for -i mult

	// ===== Block 0: Load src[0,8,4,12] =====
	MOVSD 0(R9), X0          // X0 = src[0]
	MOVSD 64(R9), X1         // X1 = src[8]
	MOVSD 32(R9), X2         // X2 = src[4]
	MOVSD 96(R9), X3         // X3 = src[12]

	// Stage 1: butterflies (0,8) and (4,12)
	MOVAPS X0, X8
	ADDPS  X1, X8
	MOVAPS X0, X9
	SUBPS  X1, X9
	MOVAPS X8, X0
	MOVAPS X9, X1

	MOVAPS X2, X8
	ADDPS  X3, X8
	MOVAPS X2, X9
	SUBPS  X3, X9
	MOVAPS X8, X2
	MOVAPS X9, X3

	// Stage 2: butterflies (0,4) and (8,12) with w=[1,-i]
	MOVAPS X0, X8
	ADDPS  X2, X8
	MOVAPS X0, X9
	SUBPS  X2, X9
	MOVAPS X8, X0
	MOVAPS X9, X2

	MOVAPS X3, X10
	SHUFPS $0xB1, X10, X10
	XORPS  X15, X10
	MOVAPS X1, X8
	ADDPS  X10, X8
	MOVAPS X1, X9
	SUBPS  X10, X9
	MOVAPS X8, X1
	MOVAPS X9, X3

	// Store block 0
	MOVSD X0, 0(SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)

	// ===== Block 1: Load src[2,10,6,14] =====
	MOVSD 16(R9), X0         // X0 = src[2]
	MOVSD 80(R9), X1         // X1 = src[10]
	MOVSD 48(R9), X2         // X2 = src[6]
	MOVSD 112(R9), X3        // X3 = src[14]

	// Stage 1: butterflies (2,10) and (6,14)
	MOVAPS X0, X8
	ADDPS  X1, X8
	MOVAPS X0, X9
	SUBPS  X1, X9
	MOVAPS X8, X0
	MOVAPS X9, X1

	MOVAPS X2, X8
	ADDPS  X3, X8
	MOVAPS X2, X9
	SUBPS  X3, X9
	MOVAPS X8, X2
	MOVAPS X9, X3

	// Stage 2: butterflies (2,6) and (10,14) with w=[1,-i]
	MOVAPS X0, X8
	ADDPS  X2, X8
	MOVAPS X0, X9
	SUBPS  X2, X9
	MOVAPS X8, X0
	MOVAPS X9, X2

	MOVAPS X3, X10
	SHUFPS $0xB1, X10, X10
	XORPS  X15, X10
	MOVAPS X1, X8
	ADDPS  X10, X8
	MOVAPS X1, X9
	SUBPS  X10, X9
	MOVAPS X8, X1
	MOVAPS X9, X3

	// Store block 1
	MOVSD X0, 32(SI)
	MOVSD X1, 40(SI)
	MOVSD X2, 48(SI)
	MOVSD X3, 56(SI)

	// ===== Block 2: Load src[1,9,5,13] =====
	MOVSD 8(R9), X0          // X0 = src[1]
	MOVSD 72(R9), X1         // X1 = src[9]
	MOVSD 40(R9), X2         // X2 = src[5]
	MOVSD 104(R9), X3        // X3 = src[13]

	// Stage 1: butterflies (1,9) and (5,13)
	MOVAPS X0, X8
	ADDPS  X1, X8
	MOVAPS X0, X9
	SUBPS  X1, X9
	MOVAPS X8, X0
	MOVAPS X9, X1

	MOVAPS X2, X8
	ADDPS  X3, X8
	MOVAPS X2, X9
	SUBPS  X3, X9
	MOVAPS X8, X2
	MOVAPS X9, X3

	// Stage 2: butterflies (1,5) and (9,13) with w=[1,-i]
	MOVAPS X0, X8
	ADDPS  X2, X8
	MOVAPS X0, X9
	SUBPS  X2, X9
	MOVAPS X8, X0
	MOVAPS X9, X2

	MOVAPS X3, X10
	SHUFPS $0xB1, X10, X10
	XORPS  X15, X10
	MOVAPS X1, X8
	ADDPS  X10, X8
	MOVAPS X1, X9
	SUBPS  X10, X9
	MOVAPS X8, X1
	MOVAPS X9, X3

	// Store block 2
	MOVSD X0, 64(SI)
	MOVSD X1, 72(SI)
	MOVSD X2, 80(SI)
	MOVSD X3, 88(SI)

	// ===== Block 3: Load src[3,11,7,15] =====
	MOVSD 24(R9), X0         // X0 = src[3]
	MOVSD 88(R9), X1         // X1 = src[11]
	MOVSD 56(R9), X2         // X2 = src[7]
	MOVSD 120(R9), X3        // X3 = src[15]

	// Stage 1: butterflies (3,11) and (7,15)
	MOVAPS X0, X8
	ADDPS  X1, X8
	MOVAPS X0, X9
	SUBPS  X1, X9
	MOVAPS X8, X0
	MOVAPS X9, X1

	MOVAPS X2, X8
	ADDPS  X3, X8
	MOVAPS X2, X9
	SUBPS  X3, X9
	MOVAPS X8, X2
	MOVAPS X9, X3

	// Stage 2: butterflies (3,7) and (11,15) with w=[1,-i]
	MOVAPS X0, X8
	ADDPS  X2, X8
	MOVAPS X0, X9
	SUBPS  X2, X9
	MOVAPS X8, X0
	MOVAPS X9, X2

	MOVAPS X3, X10
	SHUFPS $0xB1, X10, X10
	XORPS  X15, X10
	MOVAPS X1, X8
	ADDPS  X10, X8
	MOVAPS X1, X9
	SUBPS  X10, X9
	MOVAPS X8, X1
	MOVAPS X9, X3

	// Store block 3
	MOVSD X0, 96(SI)
	MOVSD X1, 104(SI)
	MOVSD X2, 112(SI)
	MOVSD X3, 120(SI)

	// ==================================================================
	// STAGE 3 (Stride 4)
	// ==================================================================
	// Process 2 blocks of 8 elements each
	// Butterflies: (0,4), (1,5), (2,6), (3,7) within each block
	// Twiddles: W^0, W^2, W^4, W^6 (W^4 = -i optimized)
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer (reset to start)
	MOVQ $2, CX              // loop counter: 2 blocks of 8

stage3_loop:
	// ----- Load 8 elements from work buffer (stages 1-2 results) -----
	MOVSD (SI), X0           // X0 = work[0]
	MOVSD 8(SI), X1          // X1 = work[1]
	MOVSD 16(SI), X2         // X2 = work[2]
	MOVSD 24(SI), X3         // X3 = work[3]
	MOVSD 32(SI), X4         // X4 = work[4]
	MOVSD 40(SI), X5         // X5 = work[5]
	MOVSD 48(SI), X6         // X6 = work[6]
	MOVSD 56(SI), X7         // X7 = work[7]

	// ----- Butterfly 0: (X0, X4) with w^0 = 1 -----
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X4, X8            // X8 = X0 + X4
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X4, X9            // X9 = X0 - X4
	MOVAPS X8, X0            // X0 = X0 + X4
	MOVAPS X9, X4            // X4 = X0 - X4

	// ----- Butterfly 1: (X1, X5) with w^2 -----
	// Complex mult: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
	// Using ADDSUBPS: computes [a0-b0, a1+b1, a2-b2, a3+b3]
	MOVSD 16(R10), X10       // X10 = twiddle[2] = w^2
	MOVAPS X10, X11          // X11 = w^2
	SHUFPS $0x00, X11, X11   // X11 = (w.re, w.re, w.re, w.re) - broadcast real
	MOVAPS X10, X12          // X12 = w^2
	SHUFPS $0x55, X12, X12   // X12 = (w.im, w.im, w.im, w.im) - broadcast imag
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = (X5.im, X5.re) - swap for cross mult
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5.re * w.re, X5.im * w.re
	MULPS  X12, X13          // X13 = X5.im * w.im, X5.re * w.im
	ADDSUBPS X13, X14        // X14 = (re*re - im*im, im*re + re*im) = X5 * w^2

	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + X5*w^2
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - X5*w^2
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// ----- Butterfly 2: (X2, X6) with w^4 = -i (optimized) -----
	MOVAPS X6, X10           // X10 = X6
	SHUFPS $0xB1, X10, X10   // X10 = (X6.im, X6.re) - swap
	MOVUPS ·maskNegHiPS(SB), X15 // X15 = mask for negating high
	XORPS  X15, X10          // X10 = (X6.im, -X6.re) = X6 * (-i)

	MOVAPS X2, X8            // X8 = X2
	ADDPS  X10, X8           // X8 = X2 + X6*(-i)
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X10, X9           // X9 = X2 - X6*(-i)
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// ----- Butterfly 3: (X3, X7) with w^6 -----
	MOVSD 48(R10), X10       // X10 = twiddle[6] = w^6
	MOVAPS X10, X11          // X11 = w^6
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^6
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X7, X13           // X13 = X7
	SHUFPS $0xB1, X13, X13   // X13 = (X7.im, X7.re)
	MOVAPS X7, X14           // X14 = X7
	MULPS  X11, X14          // X14 = X7 * w.re
	MULPS  X12, X13          // X13 = X7_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X7 * w^6

	MOVAPS X3, X8            // X8 = X3
	ADDPS  X14, X8           // X8 = X3 + X7*w^6
	MOVAPS X3, X9            // X9 = X3
	SUBPS  X14, X9           // X9 = X3 - X7*w^6
	MOVAPS X8, X3            // X3 = X3 + t
	MOVAPS X9, X7            // X7 = X3 - t

	// ----- Store 8 results back to work buffer -----
	MOVSD X0, (SI)           // work[0] = X0
	MOVSD X1, 8(SI)          // work[1] = X1
	MOVSD X2, 16(SI)         // work[2] = X2
	MOVSD X3, 24(SI)         // work[3] = X3
	MOVSD X4, 32(SI)         // work[4] = X4
	MOVSD X5, 40(SI)         // work[5] = X5
	MOVSD X6, 48(SI)         // work[6] = X6
	MOVSD X7, 56(SI)         // work[7] = X7

	ADDQ $64, SI             // advance by 8 elements (64 bytes)
	DECQ CX                  // decrement loop counter
	JNZ  stage3_loop         // continue if CX != 0

	// ==================================================================
	// STAGE 4 (Stride 8) - Final Stage
	// ==================================================================
	// Single block of 16 elements
	// Butterflies: (0,8), (1,9), (2,10), (3,11), (4,12), (5,13), (6,14), (7,15)
	// Twiddles: W^0..W^7
	// Split into two parts to fit in registers
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer (reset to start)

	// ----- Part 1: k=0..3 -> butterflies (0,8), (1,9), (2,10), (3,11) -----
	MOVSD (SI), X0           // X0 = work[0]
	MOVSD 8(SI), X1          // X1 = work[1]
	MOVSD 16(SI), X2         // X2 = work[2]
	MOVSD 24(SI), X3         // X3 = work[3]
	MOVSD 64(SI), X4         // X4 = work[8]
	MOVSD 72(SI), X5         // X5 = work[9]
	MOVSD 80(SI), X6         // X6 = work[10]
	MOVSD 88(SI), X7         // X7 = work[11]

	// Butterfly k=0: (X0, X4) with w^0 = 1
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X4, X8            // X8 = X0 + X4
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X4, X9            // X9 = X0 - X4
	MOVAPS X8, X0            // X0 = X0 + X4
	MOVAPS X9, X4            // X4 = X0 - X4

	// Butterfly k=1: (X1, X5) with w^1
	MOVSD  8(R10), X10       // X10 = twiddle[1] = w^1
	MOVAPS X10, X11          // X11 = w^1
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^1
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = (X5.im, X5.re)
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5 * w.re
	MULPS  X12, X13          // X13 = X5_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X5 * w^1

	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + t
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - t
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// Butterfly k=2: (X2, X6) with w^2
	MOVSD  16(R10), X10      // X10 = twiddle[2] = w^2
	MOVAPS X10, X11          // X11 = w^2
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^2
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X6, X13           // X13 = X6
	SHUFPS $0xB1, X13, X13   // X13 = (X6.im, X6.re)
	MOVAPS X6, X14           // X14 = X6
	MULPS  X11, X14          // X14 = X6 * w.re
	MULPS  X12, X13          // X13 = X6_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X6 * w^2

	MOVAPS X2, X8            // X8 = X2
	ADDPS  X14, X8           // X8 = X2 + t
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X14, X9           // X9 = X2 - t
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// Butterfly k=3: (X3, X7) with w^3
	MOVSD  24(R10), X10      // X10 = twiddle[3] = w^3
	MOVAPS X10, X11          // X11 = w^3
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^3
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X7, X13           // X13 = X7
	SHUFPS $0xB1, X13, X13   // X13 = (X7.im, X7.re)
	MOVAPS X7, X14           // X14 = X7
	MULPS  X11, X14          // X14 = X7 * w.re
	MULPS  X12, X13          // X13 = X7_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X7 * w^3

	MOVAPS X3, X8            // X8 = X3
	ADDPS  X14, X8           // X8 = X3 + t
	MOVAPS X3, X9            // X9 = X3
	SUBPS  X14, X9           // X9 = X3 - t
	MOVAPS X8, X3            // X3 = X3 + t
	MOVAPS X9, X7            // X7 = X3 - t

	// Store Part 1 results
	MOVSD X0, (SI)           // work[0] = X0
	MOVSD X1, 8(SI)          // work[1] = X1
	MOVSD X2, 16(SI)         // work[2] = X2
	MOVSD X3, 24(SI)         // work[3] = X3
	MOVSD X4, 64(SI)         // work[8] = X4
	MOVSD X5, 72(SI)         // work[9] = X5
	MOVSD X6, 80(SI)         // work[10] = X6
	MOVSD X7, 88(SI)         // work[11] = X7

	// ----- Part 2: k=4..7 -> butterflies (4,12), (5,13), (6,14), (7,15) -----
	MOVSD 32(SI), X0         // X0 = work[4]
	MOVSD 40(SI), X1         // X1 = work[5]
	MOVSD 48(SI), X2         // X2 = work[6]
	MOVSD 56(SI), X3         // X3 = work[7]
	MOVSD 96(SI), X4         // X4 = work[12]
	MOVSD 104(SI), X5        // X5 = work[13]
	MOVSD 112(SI), X6        // X6 = work[14]
	MOVSD 120(SI), X7        // X7 = work[15]

	// Butterfly k=4: (X0, X4) with w^4 = -i (optimized)
	MOVAPS X4, X10           // X10 = X4
	SHUFPS $0xB1, X10, X10   // X10 = (X4.im, X4.re)
	MOVUPS ·maskNegHiPS(SB), X15 // X15 = mask for -i
	XORPS  X15, X10          // X10 = X4 * (-i)

	MOVAPS X0, X8            // X8 = X0
	ADDPS  X10, X8           // X8 = X0 + t
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X10, X9           // X9 = X0 - t
	MOVAPS X8, X0            // X0 = X0 + t
	MOVAPS X9, X4            // X4 = X0 - t

	// Butterfly k=5: (X1, X5) with w^5
	MOVSD  40(R10), X10      // X10 = twiddle[5] = w^5
	MOVAPS X10, X11          // X11 = w^5
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^5
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = (X5.im, X5.re)
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5 * w.re
	MULPS  X12, X13          // X13 = X5_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X5 * w^5

	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + t
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - t
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// Butterfly k=6: (X2, X6) with w^6
	MOVSD  48(R10), X10      // X10 = twiddle[6] = w^6
	MOVAPS X10, X11          // X11 = w^6
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^6
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X6, X13           // X13 = X6
	SHUFPS $0xB1, X13, X13   // X13 = (X6.im, X6.re)
	MOVAPS X6, X14           // X14 = X6
	MULPS  X11, X14          // X14 = X6 * w.re
	MULPS  X12, X13          // X13 = X6_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X6 * w^6

	MOVAPS X2, X8            // X8 = X2
	ADDPS  X14, X8           // X8 = X2 + t
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X14, X9           // X9 = X2 - t
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// Butterfly k=7: (X3, X7) with w^7
	MOVSD  56(R10), X10      // X10 = twiddle[7] = w^7
	MOVAPS X10, X11          // X11 = w^7
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^7
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X7, X13           // X13 = X7
	SHUFPS $0xB1, X13, X13   // X13 = (X7.im, X7.re)
	MOVAPS X7, X14           // X14 = X7
	MULPS  X11, X14          // X14 = X7 * w.re
	MULPS  X12, X13          // X13 = X7_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X7 * w^7

	MOVAPS X3, X8            // X8 = X3
	ADDPS  X14, X8           // X8 = X3 + t
	MOVAPS X3, X9            // X9 = X3
	SUBPS  X14, X9           // X9 = X3 - t
	MOVAPS X8, X3            // X3 = X3 + t
	MOVAPS X9, X7            // X7 = X3 - t

	// Store Part 2 results
	MOVSD X0, 32(SI)         // work[4] = X0
	MOVSD X1, 40(SI)         // work[5] = X1
	MOVSD X2, 48(SI)         // work[6] = X2
	MOVSD X3, 56(SI)         // work[7] = X3
	MOVSD X4, 96(SI)         // work[12] = X4
	MOVSD X5, 104(SI)        // work[13] = X5
	MOVSD X6, 112(SI)        // work[14] = X6
	MOVSD X7, 120(SI)        // work[15] = X7

	// ==================================================================
	// Copy to dst if work buffer != dst (in-place case)
	// ==================================================================
	MOVQ dst+0(FP), R14      // R14 = dst pointer
	CMPQ R8, R14             // work buffer == dst?
	JE   size16_r2_sse2_fwd_done // skip copy if same

	// Copy 16 complex64 elements (128 bytes) using MOVUPS (2 elements per XMM)
	MOVUPS (R8), X0          // X0 = work[0:2]
	MOVUPS X0, (R14)         // dst[0:2] = X0
	MOVUPS 16(R8), X1        // X1 = work[2:4]
	MOVUPS X1, 16(R14)       // dst[2:4] = X1
	MOVUPS 32(R8), X2        // X2 = work[4:6]
	MOVUPS X2, 32(R14)       // dst[4:6] = X2
	MOVUPS 48(R8), X3        // X3 = work[6:8]
	MOVUPS X3, 48(R14)       // dst[6:8] = X3
	MOVUPS 64(R8), X4        // X4 = work[8:10]
	MOVUPS X4, 64(R14)       // dst[8:10] = X4
	MOVUPS 80(R8), X5        // X5 = work[10:12]
	MOVUPS X5, 80(R14)       // dst[10:12] = X5
	MOVUPS 96(R8), X6        // X6 = work[12:14]
	MOVUPS X6, 96(R14)       // dst[12:14] = X6
	MOVUPS 112(R8), X7       // X7 = work[14:16]
	MOVUPS X7, 112(R14)      // dst[14:16] = X7

size16_r2_sse2_fwd_done:
	MOVB $1, ret+96(FP)      // return true
	RET

size16_r2_sse2_fwd_return_false:
	MOVB $0, ret+96(FP)      // return false
	RET

// ===========================================================================
// Inverse transform, size 16, complex64, radix-2 variant
// ===========================================================================
TEXT ·InverseSSE3Size16Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// ===== Parameter Loading =====
	MOVQ dst+0(FP), R8       // R8  = dst slice data pointer
	MOVQ src+24(FP), R9      // R9  = src slice data pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle slice data pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch slice data pointer
	MOVQ src+32(FP), R13     // R13 = src slice length

	// ===== Input Validation =====
	CMPQ R13, $16            // check n == 16
	JNE  size16_r2_sse2_inv_return_false

	MOVQ dst+8(FP), AX       // dst.len
	CMPQ AX, $16             // dst.len >= 16?
	JL   size16_r2_sse2_inv_return_false

	MOVQ twiddle+56(FP), AX  // twiddle.len
	CMPQ AX, $16             // twiddle.len >= 16?
	JL   size16_r2_sse2_inv_return_false

	MOVQ scratch+80(FP), AX  // scratch.len
	CMPQ AX, $16             // scratch.len >= 16?
	JL   size16_r2_sse2_inv_return_false

	MOVQ $16, AX             // sanity check
	CMPQ AX, $16
	JL   size16_r2_sse2_inv_return_false

	// ===== Select Working Buffer =====
	CMPQ R8, R9              // dst == src?
	JNE  size16_r2_sse2_inv_use_dst
	MOVQ R11, R8             // in-place: work in scratch buffer

size16_r2_sse2_inv_use_dst:
	// ==================================================================
	// STAGES 1 & 2 (Combined with bit-reversal, fully unrolled)
	// ==================================================================
	// Inverse uses conjugate twiddles: conj(-i) = i for stage 2
	// Bit-reversal pattern: 0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer write position
	MOVUPS ·maskNegLoPS(SB), X15 // X15 = mask for +i mult (negate low = real)

	// ===== Block 0: Load src[0,8,4,12] =====
	MOVSD 0(R9), X0          // X0 = src[0]
	MOVSD 64(R9), X1         // X1 = src[8]
	MOVSD 32(R9), X2         // X2 = src[4]
	MOVSD 96(R9), X3         // X3 = src[12]

	// Stage 1: butterflies (0,8) and (4,12)
	MOVAPS X0, X8
	ADDPS  X1, X8
	MOVAPS X0, X9
	SUBPS  X1, X9
	MOVAPS X8, X0
	MOVAPS X9, X1

	MOVAPS X2, X8
	ADDPS  X3, X8
	MOVAPS X2, X9
	SUBPS  X3, X9
	MOVAPS X8, X2
	MOVAPS X9, X3

	// Stage 2: butterflies with w=[1,i]
	MOVAPS X0, X8
	ADDPS  X2, X8
	MOVAPS X0, X9
	SUBPS  X2, X9
	MOVAPS X8, X0
	MOVAPS X9, X2

	MOVAPS X3, X10
	SHUFPS $0xB1, X10, X10
	XORPS  X15, X10
	MOVAPS X1, X8
	ADDPS  X10, X8
	MOVAPS X1, X9
	SUBPS  X10, X9
	MOVAPS X8, X1
	MOVAPS X9, X3

	// Store block 0
	MOVSD X0, 0(SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)

	// ===== Block 1: Load src[2,10,6,14] =====
	MOVSD 16(R9), X0         // X0 = src[2]
	MOVSD 80(R9), X1         // X1 = src[10]
	MOVSD 48(R9), X2         // X2 = src[6]
	MOVSD 112(R9), X3        // X3 = src[14]

	// Stage 1: butterflies (2,10) and (6,14)
	MOVAPS X0, X8
	ADDPS  X1, X8
	MOVAPS X0, X9
	SUBPS  X1, X9
	MOVAPS X8, X0
	MOVAPS X9, X1

	MOVAPS X2, X8
	ADDPS  X3, X8
	MOVAPS X2, X9
	SUBPS  X3, X9
	MOVAPS X8, X2
	MOVAPS X9, X3

	// Stage 2: butterflies with w=[1,i]
	MOVAPS X0, X8
	ADDPS  X2, X8
	MOVAPS X0, X9
	SUBPS  X2, X9
	MOVAPS X8, X0
	MOVAPS X9, X2

	MOVAPS X3, X10
	SHUFPS $0xB1, X10, X10
	XORPS  X15, X10
	MOVAPS X1, X8
	ADDPS  X10, X8
	MOVAPS X1, X9
	SUBPS  X10, X9
	MOVAPS X8, X1
	MOVAPS X9, X3

	// Store block 1
	MOVSD X0, 32(SI)
	MOVSD X1, 40(SI)
	MOVSD X2, 48(SI)
	MOVSD X3, 56(SI)

	// ===== Block 2: Load src[1,9,5,13] =====
	MOVSD 8(R9), X0          // X0 = src[1]
	MOVSD 72(R9), X1         // X1 = src[9]
	MOVSD 40(R9), X2         // X2 = src[5]
	MOVSD 104(R9), X3        // X3 = src[13]

	// Stage 1: butterflies (1,9) and (5,13)
	MOVAPS X0, X8
	ADDPS  X1, X8
	MOVAPS X0, X9
	SUBPS  X1, X9
	MOVAPS X8, X0
	MOVAPS X9, X1

	MOVAPS X2, X8
	ADDPS  X3, X8
	MOVAPS X2, X9
	SUBPS  X3, X9
	MOVAPS X8, X2
	MOVAPS X9, X3

	// Stage 2: butterflies with w=[1,i]
	MOVAPS X0, X8
	ADDPS  X2, X8
	MOVAPS X0, X9
	SUBPS  X2, X9
	MOVAPS X8, X0
	MOVAPS X9, X2

	MOVAPS X3, X10
	SHUFPS $0xB1, X10, X10
	XORPS  X15, X10
	MOVAPS X1, X8
	ADDPS  X10, X8
	MOVAPS X1, X9
	SUBPS  X10, X9
	MOVAPS X8, X1
	MOVAPS X9, X3

	// Store block 2
	MOVSD X0, 64(SI)
	MOVSD X1, 72(SI)
	MOVSD X2, 80(SI)
	MOVSD X3, 88(SI)

	// ===== Block 3: Load src[3,11,7,15] =====
	MOVSD 24(R9), X0         // X0 = src[3]
	MOVSD 88(R9), X1         // X1 = src[11]
	MOVSD 56(R9), X2         // X2 = src[7]
	MOVSD 120(R9), X3        // X3 = src[15]

	// Stage 1: butterflies (3,11) and (7,15)
	MOVAPS X0, X8
	ADDPS  X1, X8
	MOVAPS X0, X9
	SUBPS  X1, X9
	MOVAPS X8, X0
	MOVAPS X9, X1

	MOVAPS X2, X8
	ADDPS  X3, X8
	MOVAPS X2, X9
	SUBPS  X3, X9
	MOVAPS X8, X2
	MOVAPS X9, X3

	// Stage 2: butterflies with w=[1,i]
	MOVAPS X0, X8
	ADDPS  X2, X8
	MOVAPS X0, X9
	SUBPS  X2, X9
	MOVAPS X8, X0
	MOVAPS X9, X2

	MOVAPS X3, X10
	SHUFPS $0xB1, X10, X10
	XORPS  X15, X10
	MOVAPS X1, X8
	ADDPS  X10, X8
	MOVAPS X1, X9
	SUBPS  X10, X9
	MOVAPS X8, X1
	MOVAPS X9, X3

	// Store block 3
	MOVSD X0, 96(SI)
	MOVSD X1, 104(SI)
	MOVSD X2, 112(SI)
	MOVSD X3, 120(SI)

	// ==================================================================
	// STAGE 3 (Stride 4) - Inverse uses conjugate twiddles
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer (reset)
	MOVQ $2, CX              // loop counter: 2 blocks
	MOVUPS ·maskNegHiPS(SB), X15 // X15 = mask for conjugation

inv_stage3_loop:
	// ----- Load 8 elements from work buffer -----
	MOVSD (SI), X0           // X0 = work[0]
	MOVSD 8(SI), X1          // X1 = work[1]
	MOVSD 16(SI), X2         // X2 = work[2]
	MOVSD 24(SI), X3         // X3 = work[3]
	MOVSD 32(SI), X4         // X4 = work[4]
	MOVSD 40(SI), X5         // X5 = work[5]
	MOVSD 48(SI), X6         // X6 = work[6]
	MOVSD 56(SI), X7         // X7 = work[7]

	// ----- Butterfly 0: (X0, X4) with conj(w^0) = 1 -----
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X4, X8            // X8 = X0 + X4
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X4, X9            // X9 = X0 - X4
	MOVAPS X8, X0            // X0 = X0 + X4
	MOVAPS X9, X4            // X4 = X0 - X4

	// ----- Butterfly 1: (X1, X5) with conj(w^2) -----
	MOVSD 16(R10), X10       // X10 = twiddle[2] = w^2
	XORPS  X15, X10          // X10 = conj(w^2) - negate imag
	MOVAPS X10, X11          // X11 = conj(w^2)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^2)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im (negated)
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = (X5.im, X5.re)
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5 * conj(w).re
	MULPS  X12, X13          // X13 = X5_swapped * conj(w).im
	ADDSUBPS X13, X14        // X14 = X5 * conj(w^2)

	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + t
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - t
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// ----- Butterfly 2: (X2, X6) with conj(w^4) = conj(-i) = i -----
	MOVAPS X6, X10           // X10 = X6
	SHUFPS $0xB1, X10, X10   // X10 = (X6.im, X6.re)
	MOVUPS ·maskNegLoPS(SB), X14 // X14 = mask for +i (negate low)
	XORPS  X14, X10          // X10 = X6 * i

	MOVAPS X2, X8            // X8 = X2
	ADDPS  X10, X8           // X8 = X2 + t
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X10, X9           // X9 = X2 - t
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// ----- Butterfly 3: (X3, X7) with conj(w^6) -----
	MOVSD 48(R10), X10       // X10 = twiddle[6] = w^6
	XORPS  X15, X10          // X10 = conj(w^6)
	MOVAPS X10, X11          // X11 = conj(w^6)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^6)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X7, X13           // X13 = X7
	SHUFPS $0xB1, X13, X13   // X13 = (X7.im, X7.re)
	MOVAPS X7, X14           // X14 = X7
	MULPS  X11, X14          // X14 = X7 * re
	MULPS  X12, X13          // X13 = X7_swapped * im
	ADDSUBPS X13, X14        // X14 = X7 * conj(w^6)

	MOVAPS X3, X8            // X8 = X3
	ADDPS  X14, X8           // X8 = X3 + t
	MOVAPS X3, X9            // X9 = X3
	SUBPS  X14, X9           // X9 = X3 - t
	MOVAPS X8, X3            // X3 = X3 + t
	MOVAPS X9, X7            // X7 = X3 - t

	// ----- Store 8 results -----
	MOVSD X0, (SI)           // work[0] = X0
	MOVSD X1, 8(SI)          // work[1] = X1
	MOVSD X2, 16(SI)         // work[2] = X2
	MOVSD X3, 24(SI)         // work[3] = X3
	MOVSD X4, 32(SI)         // work[4] = X4
	MOVSD X5, 40(SI)         // work[5] = X5
	MOVSD X6, 48(SI)         // work[6] = X6
	MOVSD X7, 56(SI)         // work[7] = X7

	ADDQ $64, SI             // advance by 8 elements
	DECQ CX                  // decrement counter
	JNZ  inv_stage3_loop     // continue if CX != 0

	// ==================================================================
	// STAGE 4 (Stride 8) - Final Stage with conjugate twiddles
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer (reset)
	MOVUPS ·maskNegHiPS(SB), X15 // X15 = conjugate mask

	// ----- Part 1: k=0..3 -----
	MOVSD (SI), X0           // X0 = work[0]
	MOVSD 8(SI), X1          // X1 = work[1]
	MOVSD 16(SI), X2         // X2 = work[2]
	MOVSD 24(SI), X3         // X3 = work[3]
	MOVSD 64(SI), X4         // X4 = work[8]
	MOVSD 72(SI), X5         // X5 = work[9]
	MOVSD 80(SI), X6         // X6 = work[10]
	MOVSD 88(SI), X7         // X7 = work[11]

	// Butterfly k=0: conj(w^0) = 1
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X4, X8            // X8 = X0 + X4
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X4, X9            // X9 = X0 - X4
	MOVAPS X8, X0            // X0 = X0 + X4
	MOVAPS X9, X4            // X4 = X0 - X4

	// Butterfly k=1: conj(w^1)
	MOVSD  8(R10), X10       // X10 = twiddle[1]
	XORPS  X15, X10          // X10 = conj(w^1)
	MOVAPS X10, X11          // X11 = conj(w^1)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^1)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5 * re
	MULPS  X12, X13          // X13 = X5_swap * im
	ADDSUBPS X13, X14        // X14 = X5 * conj(w^1)

	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + t
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - t
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// Butterfly k=2: conj(w^2)
	MOVSD  16(R10), X10      // X10 = twiddle[2]
	XORPS  X15, X10          // X10 = conj(w^2)
	MOVAPS X10, X11          // X11 = conj(w^2)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^2)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X6, X13           // X13 = X6
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X6, X14           // X14 = X6
	MULPS  X11, X14          // X14 = X6 * re
	MULPS  X12, X13          // X13 = X6_swap * im
	ADDSUBPS X13, X14        // X14 = X6 * conj(w^2)

	MOVAPS X2, X8            // X8 = X2
	ADDPS  X14, X8           // X8 = X2 + t
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X14, X9           // X9 = X2 - t
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// Butterfly k=3: conj(w^3)
	MOVSD  24(R10), X10      // X10 = twiddle[3]
	XORPS  X15, X10          // X10 = conj(w^3)
	MOVAPS X10, X11          // X11 = conj(w^3)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^3)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X7, X13           // X13 = X7
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X7, X14           // X14 = X7
	MULPS  X11, X14          // X14 = X7 * re
	MULPS  X12, X13          // X13 = X7_swap * im
	ADDSUBPS X13, X14        // X14 = X7 * conj(w^3)

	MOVAPS X3, X8            // X8 = X3
	ADDPS  X14, X8           // X8 = X3 + t
	MOVAPS X3, X9            // X9 = X3
	SUBPS  X14, X9           // X9 = X3 - t
	MOVAPS X8, X3            // X3 = X3 + t
	MOVAPS X9, X7            // X7 = X3 - t

	// Store Part 1
	MOVSD X0, (SI)           // work[0] = X0
	MOVSD X1, 8(SI)          // work[1] = X1
	MOVSD X2, 16(SI)         // work[2] = X2
	MOVSD X3, 24(SI)         // work[3] = X3
	MOVSD X4, 64(SI)         // work[8] = X4
	MOVSD X5, 72(SI)         // work[9] = X5
	MOVSD X6, 80(SI)         // work[10] = X6
	MOVSD X7, 88(SI)         // work[11] = X7

	// ----- Part 2: k=4..7 -----
	MOVSD 32(SI), X0         // X0 = work[4]
	MOVSD 40(SI), X1         // X1 = work[5]
	MOVSD 48(SI), X2         // X2 = work[6]
	MOVSD 56(SI), X3         // X3 = work[7]
	MOVSD 96(SI), X4         // X4 = work[12]
	MOVSD 104(SI), X5        // X5 = work[13]
	MOVSD 112(SI), X6        // X6 = work[14]
	MOVSD 120(SI), X7        // X7 = work[15]

	// Butterfly k=4: conj(w^4) = conj(-i) = i
	MOVAPS X4, X10           // X10 = X4
	SHUFPS $0xB1, X10, X10   // X10 = (X4.im, X4.re)
	MOVUPS ·maskNegLoPS(SB), X14 // X14 = mask for +i
	XORPS  X14, X10          // X10 = X4 * i

	MOVAPS X0, X8            // X8 = X0
	ADDPS  X10, X8           // X8 = X0 + t
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X10, X9           // X9 = X0 - t
	MOVAPS X8, X0            // X0 = X0 + t
	MOVAPS X9, X4            // X4 = X0 - t

	// Butterfly k=5: conj(w^5)
	MOVSD  40(R10), X10      // X10 = twiddle[5]
	XORPS  X15, X10          // X10 = conj(w^5)
	MOVAPS X10, X11          // X11 = conj(w^5)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^5)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5 * re
	MULPS  X12, X13          // X13 = X5_swap * im
	ADDSUBPS X13, X14        // X14 = X5 * conj(w^5)

	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + t
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - t
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// Butterfly k=6: conj(w^6)
	MOVSD  48(R10), X10      // X10 = twiddle[6]
	XORPS  X15, X10          // X10 = conj(w^6)
	MOVAPS X10, X11          // X11 = conj(w^6)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^6)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X6, X13           // X13 = X6
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X6, X14           // X14 = X6
	MULPS  X11, X14          // X14 = X6 * re
	MULPS  X12, X13          // X13 = X6_swap * im
	ADDSUBPS X13, X14        // X14 = X6 * conj(w^6)

	MOVAPS X2, X8            // X8 = X2
	ADDPS  X14, X8           // X8 = X2 + t
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X14, X9           // X9 = X2 - t
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// Butterfly k=7: conj(w^7)
	MOVSD  56(R10), X10      // X10 = twiddle[7]
	XORPS  X15, X10          // X10 = conj(w^7)
	MOVAPS X10, X11          // X11 = conj(w^7)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^7)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X7, X13           // X13 = X7
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X7, X14           // X14 = X7
	MULPS  X11, X14          // X14 = X7 * re
	MULPS  X12, X13          // X13 = X7_swap * im
	ADDSUBPS X13, X14        // X14 = X7 * conj(w^7)

	MOVAPS X3, X8            // X8 = X3
	ADDPS  X14, X8           // X8 = X3 + t
	MOVAPS X3, X9            // X9 = X3
	SUBPS  X14, X9           // X9 = X3 - t
	MOVAPS X8, X3            // X3 = X3 + t
	MOVAPS X9, X7            // X7 = X3 - t

	// Store Part 2
	MOVSD X0, 32(SI)         // work[4] = X0
	MOVSD X1, 40(SI)         // work[5] = X1
	MOVSD X2, 48(SI)         // work[6] = X2
	MOVSD X3, 56(SI)         // work[7] = X3
	MOVSD X4, 96(SI)         // work[12] = X4
	MOVSD X5, 104(SI)        // work[13] = X5
	MOVSD X6, 112(SI)        // work[14] = X6
	MOVSD X7, 120(SI)        // work[15] = X7

	// ==================================================================
	// Scale by 1/N (1/16 = 0.0625)
	// ==================================================================
	MOVSS ·sixteenth32(SB), X15 // X15 = 0.0625f
	SHUFPS $0x00, X15, X15   // X15 = broadcast to all lanes

	// Scale all 16 elements (128 bytes = 8 MOVUPS of 16 bytes each)
	MOVUPS (SI), X0          // X0 = work[0:2]
	MULPS X15, X0            // X0 *= 1/16
	MOVUPS X0, (SI)          // work[0:2] = X0
	MOVUPS 16(SI), X1        // X1 = work[2:4]
	MULPS X15, X1            // X1 *= 1/16
	MOVUPS X1, 16(SI)        // work[2:4] = X1
	MOVUPS 32(SI), X2        // X2 = work[4:6]
	MULPS X15, X2            // X2 *= 1/16
	MOVUPS X2, 32(SI)        // work[4:6] = X2
	MOVUPS 48(SI), X3        // X3 = work[6:8]
	MULPS X15, X3            // X3 *= 1/16
	MOVUPS X3, 48(SI)        // work[6:8] = X3
	MOVUPS 64(SI), X4        // X4 = work[8:10]
	MULPS X15, X4            // X4 *= 1/16
	MOVUPS X4, 64(SI)        // work[8:10] = X4
	MOVUPS 80(SI), X5        // X5 = work[10:12]
	MULPS X15, X5            // X5 *= 1/16
	MOVUPS X5, 80(SI)        // work[10:12] = X5
	MOVUPS 96(SI), X6        // X6 = work[12:14]
	MULPS X15, X6            // X6 *= 1/16
	MOVUPS X6, 96(SI)        // work[12:14] = X6
	MOVUPS 112(SI), X7       // X7 = work[14:16]
	MULPS X15, X7            // X7 *= 1/16
	MOVUPS X7, 112(SI)       // work[14:16] = X7

	// ==================================================================
	// Copy to dst if needed
	// ==================================================================
	MOVQ dst+0(FP), R14      // R14 = dst pointer
	CMPQ R8, R14             // work == dst?
	JE   size16_r2_sse2_inv_done // skip copy if same

	MOVUPS (R8), X0          // copy work[0:2]
	MOVUPS X0, (R14)
	MOVUPS 16(R8), X1        // copy work[2:4]
	MOVUPS X1, 16(R14)
	MOVUPS 32(R8), X2        // copy work[4:6]
	MOVUPS X2, 32(R14)
	MOVUPS 48(R8), X3        // copy work[6:8]
	MOVUPS X3, 48(R14)
	MOVUPS 64(R8), X4        // copy work[8:10]
	MOVUPS X4, 64(R14)
	MOVUPS 80(R8), X5        // copy work[10:12]
	MOVUPS X5, 80(R14)
	MOVUPS 96(R8), X6        // copy work[12:14]
	MOVUPS X6, 96(R14)
	MOVUPS 112(R8), X7       // copy work[14:16]
	MOVUPS X7, 112(R14)

size16_r2_sse2_inv_done:
	MOVB $1, ret+96(FP)      // return true
	RET

size16_r2_sse2_inv_return_false:
	MOVB $0, ret+96(FP)      // return false
	RET

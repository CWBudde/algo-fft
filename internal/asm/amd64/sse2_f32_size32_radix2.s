//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-32 Radix-2 FFT Kernels for AMD64 (complex64)
// ===========================================================================
//
// Radix-2 FFT kernel for size 32.
//
// Stage 1 (radix-2): 16 butterflies (stride 1)
// Stage 2 (radix-2): 16 butterflies (stride 2)
// Stage 3 (radix-2): 16 butterflies (stride 4)
// Stage 4 (radix-2): 16 butterflies (stride 8)
// Stage 5 (radix-2): 16 butterflies (stride 16)
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
// Forward transform, size 32, complex64, radix-2
// ===========================================================================
TEXT ·ForwardSSE2Size32Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// ===== Parameter Loading =====
	MOVQ dst+0(FP), R8       // R8  = dst slice data pointer
	MOVQ src+24(FP), R9      // R9  = src slice data pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle slice data pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch slice data pointer
	LEAQ ·bitrevSSE2Size32Radix2(SB), R12  // R12 = bitrev table pointer
	MOVQ src+32(FP), R13     // R13 = src slice length (should be 32)

	// ===== Input Validation =====
	CMPQ R13, $32            // check n == 32
	JNE  size32_r2_sse2_fwd_return_false

	MOVQ dst+8(FP), AX       // dst.len
	CMPQ AX, $32             // dst.len >= 32?
	JL   size32_r2_sse2_fwd_return_false

	MOVQ twiddle+56(FP), AX  // twiddle.len
	CMPQ AX, $32             // twiddle.len >= 32?
	JL   size32_r2_sse2_fwd_return_false

	MOVQ scratch+80(FP), AX  // scratch.len
	CMPQ AX, $32             // scratch.len >= 32?
	JL   size32_r2_sse2_fwd_return_false

	MOVQ $32, AX             // sanity check
	CMPQ AX, $32
	JL   size32_r2_sse2_fwd_return_false

	// ===== Select Working Buffer =====
	CMPQ R8, R9              // dst == src?
	JNE  size32_r2_sse2_fwd_use_dst
	MOVQ R11, R8             // in-place: work in scratch buffer

size32_r2_sse2_fwd_use_dst:
	// ==================================================================
	// STAGES 1 & 2 (Combined with bit-reversal)
	// ==================================================================
	// Process 8 blocks of 4 elements with fused bit-reversal load
	// Stage 1 stride 1: butterflies (0,1), (2,3)
	// Stage 2 stride 2: butterflies (0,2), (1,3) with twiddles [1, -i]
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer write position
	MOVQ $8, CX              // loop counter: 8 blocks of 4 elements
	MOVUPS ·maskNegHiPS(SB), X15 // X15 = mask for -i multiplication

size32_r2_sse2_fwd_stage12_loop:
	// ----- Load 4 elements with bit-reversal -----
	MOVQ (R12), DX           // DX = bitrev[i*4 + 0]
	MOVSD (R9)(DX*8), X0     // X0 = src[bitrev[i*4 + 0]]
	MOVQ 8(R12), DX          // DX = bitrev[i*4 + 1]
	MOVSD (R9)(DX*8), X1     // X1 = src[bitrev[i*4 + 1]]
	MOVQ 16(R12), DX         // DX = bitrev[i*4 + 2]
	MOVSD (R9)(DX*8), X2     // X2 = src[bitrev[i*4 + 2]]
	MOVQ 24(R12), DX         // DX = bitrev[i*4 + 3]
	MOVSD (R9)(DX*8), X3     // X3 = src[bitrev[i*4 + 3]]
	ADDQ $32, R12            // advance bitrev pointer by 4 entries

	// ----- Stage 1: stride 1, twiddle w=1 -----
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X1, X8            // X8 = X0 + X1
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X1, X9            // X9 = X0 - X1
	MOVAPS X8, X0            // X0 = X0 + X1
	MOVAPS X9, X1            // X1 = X0 - X1

	MOVAPS X2, X8            // X8 = X2
	ADDPS  X3, X8            // X8 = X2 + X3
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X3, X9            // X9 = X2 - X3
	MOVAPS X8, X2            // X2 = X2 + X3
	MOVAPS X9, X3            // X3 = X2 - X3

	// ----- Stage 2: stride 2, twiddles [1, -i] -----
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X2, X8            // X8 = X0 + X2
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X2, X9            // X9 = X0 - X2
	MOVAPS X8, X0            // X0 = X0 + X2
	MOVAPS X9, X2            // X2 = X0 - X2

	// Butterfly (X1, X3) with w=-i: t = X3 * (-i) = (im, -re)
	MOVAPS X3, X10           // X10 = X3
	SHUFPS $0xB1, X10, X10   // X10 = (X3.im, X3.re) - swap
	XORPS  X15, X10          // X10 = (X3.im, -X3.re) = X3 * (-i)
	MOVAPS X1, X8            // X8 = X1
	ADDPS  X10, X8           // X8 = X1 + X3*(-i)
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X10, X9           // X9 = X1 - X3*(-i)
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X3            // X3 = X1 - t

	// ----- Store 4 results to work buffer -----
	MOVSD X0, (SI)           // work[i*4 + 0] = X0
	MOVSD X1, 8(SI)          // work[i*4 + 1] = X1
	MOVSD X2, 16(SI)         // work[i*4 + 2] = X2
	MOVSD X3, 24(SI)         // work[i*4 + 3] = X3

	ADDQ $32, SI             // advance work pointer by 4 elements
	DECQ CX                  // decrement loop counter
	JNZ  size32_r2_sse2_fwd_stage12_loop

	// ==================================================================
	// STAGE 3 (Stride 4)
	// ==================================================================
	// Process 4 blocks of 8 elements from work buffer
	// Butterflies: (0,4), (1,5), (2,6), (3,7) within each block
	// Twiddles: W^0, W^4, W^8, W^12 (W^8 = -i optimized)
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer (reset to start)
	MOVQ $4, CX              // loop counter: 4 blocks of 8

size32_r2_sse2_fwd_stage3_loop:
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

	// ----- Butterfly 1: (X1, X5) with w^4 -----
	MOVSD 32(R10), X10       // X10 = twiddle[4] = w^4
	MOVAPS X10, X11          // X11 = w^4
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^4
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = (X5.im, X5.re)
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5 * w.re
	MULPS  X12, X13          // X13 = X5_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X5 * w^4
	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + t
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - t
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// ----- Butterfly 2: (X2, X6) with w^8 = -i (optimized) -----
	MOVAPS X6, X10           // X10 = X6
	SHUFPS $0xB1, X10, X10   // X10 = (X6.im, X6.re)
	XORPS  X15, X10          // X10 = X6 * (-i)
	MOVAPS X2, X8            // X8 = X2
	ADDPS  X10, X8           // X8 = X2 + t
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X10, X9           // X9 = X2 - t
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// ----- Butterfly 3: (X3, X7) with w^12 -----
	MOVSD 96(R10), X10       // X10 = twiddle[12] = w^12
	MOVAPS X10, X11          // X11 = w^12
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^12
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X7, X13           // X13 = X7
	SHUFPS $0xB1, X13, X13   // X13 = (X7.im, X7.re)
	MOVAPS X7, X14           // X14 = X7
	MULPS  X11, X14          // X14 = X7 * w.re
	MULPS  X12, X13          // X13 = X7_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X7 * w^12
	MOVAPS X3, X8            // X8 = X3
	ADDPS  X14, X8           // X8 = X3 + t
	MOVAPS X3, X9            // X9 = X3
	SUBPS  X14, X9           // X9 = X3 - t
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

	ADDQ $64, SI             // advance by 8 elements
	DECQ CX                  // decrement loop counter
	JNZ  size32_r2_sse2_fwd_stage3_loop

	// ==================================================================
	// STAGE 4 (Stride 8)
	// ==================================================================
	// Process 2 blocks of 16 elements from work buffer
	// Twiddles: W^0, W^2, W^4, W^6, W^8, W^10, W^12, W^14
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer (reset to start)
	MOVQ $2, CX              // loop counter: 2 blocks of 16

size32_r2_sse2_fwd_stage4_loop:
	// ----- Part 1: k=0..3 -> butterflies (0,8), (1,9), (2,10), (3,11) -----
	MOVSD (SI), X0           // X0 = work[0]
	MOVSD 8(SI), X1          // X1 = work[1]
	MOVSD 16(SI), X2         // X2 = work[2]
	MOVSD 24(SI), X3         // X3 = work[3]
	MOVSD 64(SI), X4         // X4 = work[8]
	MOVSD 72(SI), X5         // X5 = work[9]
	MOVSD 80(SI), X6         // X6 = work[10]
	MOVSD 88(SI), X7         // X7 = work[11]

	// k=0 (w^0=1)
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X4, X8            // X8 = X0 + X4
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X4, X9            // X9 = X0 - X4
	MOVAPS X8, X0            // X0 = X0 + X4
	MOVAPS X9, X4            // X4 = X0 - X4

	// k=1 (w^2)
	MOVSD 16(R10), X10       // X10 = twiddle[2] = w^2
	MOVAPS X10, X11          // X11 = w^2
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^2
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = (X5.im, X5.re)
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5 * w.re
	MULPS  X12, X13          // X13 = X5_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X5 * w^2
	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + t
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - t
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// k=2 (w^4)
	MOVSD 32(R10), X10       // X10 = twiddle[4] = w^4
	MOVAPS X10, X11          // X11 = w^4
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^4
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X6, X13           // X13 = X6
	SHUFPS $0xB1, X13, X13   // X13 = (X6.im, X6.re)
	MOVAPS X6, X14           // X14 = X6
	MULPS  X11, X14          // X14 = X6 * w.re
	MULPS  X12, X13          // X13 = X6_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X6 * w^4
	MOVAPS X2, X8            // X8 = X2
	ADDPS  X14, X8           // X8 = X2 + t
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X14, X9           // X9 = X2 - t
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// k=3 (w^6)
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

	// ----- Part 2: k=4..7 -> butterflies (4,12), (5,13), (6,14), (7,15) -----
	MOVSD 32(SI), X0         // X0 = work[4]
	MOVSD 40(SI), X1         // X1 = work[5]
	MOVSD 48(SI), X2         // X2 = work[6]
	MOVSD 56(SI), X3         // X3 = work[7]
	MOVSD 96(SI), X4         // X4 = work[12]
	MOVSD 104(SI), X5        // X5 = work[13]
	MOVSD 112(SI), X6        // X6 = work[14]
	MOVSD 120(SI), X7        // X7 = work[15]

	// k=4 (w^8 = -i)
	MOVAPS X4, X10           // X10 = X4
	SHUFPS $0xB1, X10, X10   // X10 = (X4.im, X4.re)
	XORPS  X15, X10          // X10 = X4 * (-i)
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X10, X8           // X8 = X0 + t
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X10, X9           // X9 = X0 - t
	MOVAPS X8, X0            // X0 = X0 + t
	MOVAPS X9, X4            // X4 = X0 - t

	// k=5 (w^10)
	MOVSD 80(R10), X10       // X10 = twiddle[10] = w^10
	MOVAPS X10, X11          // X11 = w^10
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^10
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = (X5.im, X5.re)
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5 * w.re
	MULPS  X12, X13          // X13 = X5_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X5 * w^10
	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + t
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - t
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// k=6 (w^12)
	MOVSD 96(R10), X10       // X10 = twiddle[12] = w^12
	MOVAPS X10, X11          // X11 = w^12
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^12
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X6, X13           // X13 = X6
	SHUFPS $0xB1, X13, X13   // X13 = (X6.im, X6.re)
	MOVAPS X6, X14           // X14 = X6
	MULPS  X11, X14          // X14 = X6 * w.re
	MULPS  X12, X13          // X13 = X6_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X6 * w^12
	MOVAPS X2, X8            // X8 = X2
	ADDPS  X14, X8           // X8 = X2 + t
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X14, X9           // X9 = X2 - t
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// k=7 (w^14)
	MOVSD 112(R10), X10      // X10 = twiddle[14] = w^14
	MOVAPS X10, X11          // X11 = w^14
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^14
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X7, X13           // X13 = X7
	SHUFPS $0xB1, X13, X13   // X13 = (X7.im, X7.re)
	MOVAPS X7, X14           // X14 = X7
	MULPS  X11, X14          // X14 = X7 * w.re
	MULPS  X12, X13          // X13 = X7_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X7 * w^14
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

	ADDQ $128, SI            // advance by 16 elements
	DECQ CX                  // decrement loop counter
	JNZ  size32_r2_sse2_fwd_stage4_loop

	// ==================================================================
	// STAGE 5 (Stride 16) - Final Stage
	// ==================================================================
	// Single block of 32 elements, twiddles W^0..W^15
	// Split into 4 parts of 4 butterflies each
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer (reset to start)

	// ----- Part 1: k=0..3 -> butterflies (0,16), (1,17), (2,18), (3,19) -----
	MOVSD (SI), X0           // X0 = work[0]
	MOVSD 8(SI), X1          // X1 = work[1]
	MOVSD 16(SI), X2         // X2 = work[2]
	MOVSD 24(SI), X3         // X3 = work[3]
	MOVSD 128(SI), X4        // X4 = work[16]
	MOVSD 136(SI), X5        // X5 = work[17]
	MOVSD 144(SI), X6        // X6 = work[18]
	MOVSD 152(SI), X7        // X7 = work[19]

	// k=0 (w^0=1)
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X4, X8            // X8 = X0 + X4
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X4, X9            // X9 = X0 - X4
	MOVAPS X8, X0            // X0 = X0 + X4
	MOVAPS X9, X4            // X4 = X0 - X4

	// k=1 (w^1)
	MOVSD 8(R10), X10        // X10 = twiddle[1] = w^1
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

	// k=2 (w^2)
	MOVSD 16(R10), X10       // X10 = twiddle[2] = w^2
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

	// k=3 (w^3)
	MOVSD 24(R10), X10       // X10 = twiddle[3] = w^3
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

	// Store Part 1
	MOVSD X0, (SI)           // work[0] = X0
	MOVSD X1, 8(SI)          // work[1] = X1
	MOVSD X2, 16(SI)         // work[2] = X2
	MOVSD X3, 24(SI)         // work[3] = X3
	MOVSD X4, 128(SI)        // work[16] = X4
	MOVSD X5, 136(SI)        // work[17] = X5
	MOVSD X6, 144(SI)        // work[18] = X6
	MOVSD X7, 152(SI)        // work[19] = X7

	// ----- Part 2: k=4..7 -> butterflies (4,20), (5,21), (6,22), (7,23) -----
	MOVSD 32(SI), X0         // X0 = work[4]
	MOVSD 40(SI), X1         // X1 = work[5]
	MOVSD 48(SI), X2         // X2 = work[6]
	MOVSD 56(SI), X3         // X3 = work[7]
	MOVSD 160(SI), X4        // X4 = work[20]
	MOVSD 168(SI), X5        // X5 = work[21]
	MOVSD 176(SI), X6        // X6 = work[22]
	MOVSD 184(SI), X7        // X7 = work[23]

	// k=4 (w^4)
	MOVSD 32(R10), X10       // X10 = twiddle[4] = w^4
	MOVAPS X10, X11          // X11 = w^4
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^4
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X4, X13           // X13 = X4
	SHUFPS $0xB1, X13, X13   // X13 = (X4.im, X4.re)
	MOVAPS X4, X14           // X14 = X4
	MULPS  X11, X14          // X14 = X4 * w.re
	MULPS  X12, X13          // X13 = X4_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X4 * w^4
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X14, X8           // X8 = X0 + t
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X14, X9           // X9 = X0 - t
	MOVAPS X8, X0            // X0 = X0 + t
	MOVAPS X9, X4            // X4 = X0 - t

	// k=5 (w^5)
	MOVSD 40(R10), X10       // X10 = twiddle[5] = w^5
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

	// k=6 (w^6)
	MOVSD 48(R10), X10       // X10 = twiddle[6] = w^6
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

	// k=7 (w^7)
	MOVSD 56(R10), X10       // X10 = twiddle[7] = w^7
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

	// Store Part 2
	MOVSD X0, 32(SI)         // work[4] = X0
	MOVSD X1, 40(SI)         // work[5] = X1
	MOVSD X2, 48(SI)         // work[6] = X2
	MOVSD X3, 56(SI)         // work[7] = X3
	MOVSD X4, 160(SI)        // work[20] = X4
	MOVSD X5, 168(SI)        // work[21] = X5
	MOVSD X6, 176(SI)        // work[22] = X6
	MOVSD X7, 184(SI)        // work[23] = X7

	// ----- Part 3: k=8..11 -> butterflies (8,24), (9,25), (10,26), (11,27) -----
	MOVSD 64(SI), X0         // X0 = work[8]
	MOVSD 72(SI), X1         // X1 = work[9]
	MOVSD 80(SI), X2         // X2 = work[10]
	MOVSD 88(SI), X3         // X3 = work[11]
	MOVSD 192(SI), X4        // X4 = work[24]
	MOVSD 200(SI), X5        // X5 = work[25]
	MOVSD 208(SI), X6        // X6 = work[26]
	MOVSD 216(SI), X7        // X7 = work[27]

	// k=8 (w^8 = -i)
	MOVAPS X4, X10           // X10 = X4
	SHUFPS $0xB1, X10, X10   // X10 = (X4.im, X4.re)
	XORPS  X15, X10          // X10 = X4 * (-i)
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X10, X8           // X8 = X0 + t
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X10, X9           // X9 = X0 - t
	MOVAPS X8, X0            // X0 = X0 + t
	MOVAPS X9, X4            // X4 = X0 - t

	// k=9 (w^9)
	MOVSD 72(R10), X10       // X10 = twiddle[9] = w^9
	MOVAPS X10, X11          // X11 = w^9
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^9
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = (X5.im, X5.re)
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5 * w.re
	MULPS  X12, X13          // X13 = X5_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X5 * w^9
	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + t
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - t
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// k=10 (w^10)
	MOVSD 80(R10), X10       // X10 = twiddle[10] = w^10
	MOVAPS X10, X11          // X11 = w^10
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^10
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X6, X13           // X13 = X6
	SHUFPS $0xB1, X13, X13   // X13 = (X6.im, X6.re)
	MOVAPS X6, X14           // X14 = X6
	MULPS  X11, X14          // X14 = X6 * w.re
	MULPS  X12, X13          // X13 = X6_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X6 * w^10
	MOVAPS X2, X8            // X8 = X2
	ADDPS  X14, X8           // X8 = X2 + t
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X14, X9           // X9 = X2 - t
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// k=11 (w^11)
	MOVSD 88(R10), X10       // X10 = twiddle[11] = w^11
	MOVAPS X10, X11          // X11 = w^11
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^11
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X7, X13           // X13 = X7
	SHUFPS $0xB1, X13, X13   // X13 = (X7.im, X7.re)
	MOVAPS X7, X14           // X14 = X7
	MULPS  X11, X14          // X14 = X7 * w.re
	MULPS  X12, X13          // X13 = X7_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X7 * w^11
	MOVAPS X3, X8            // X8 = X3
	ADDPS  X14, X8           // X8 = X3 + t
	MOVAPS X3, X9            // X9 = X3
	SUBPS  X14, X9           // X9 = X3 - t
	MOVAPS X8, X3            // X3 = X3 + t
	MOVAPS X9, X7            // X7 = X3 - t

	// Store Part 3
	MOVSD X0, 64(SI)         // work[8] = X0
	MOVSD X1, 72(SI)         // work[9] = X1
	MOVSD X2, 80(SI)         // work[10] = X2
	MOVSD X3, 88(SI)         // work[11] = X3
	MOVSD X4, 192(SI)        // work[24] = X4
	MOVSD X5, 200(SI)        // work[25] = X5
	MOVSD X6, 208(SI)        // work[26] = X6
	MOVSD X7, 216(SI)        // work[27] = X7

	// ----- Part 4: k=12..15 -> butterflies (12,28), (13,29), (14,30), (15,31) -----
	MOVSD 96(SI), X0         // X0 = work[12]
	MOVSD 104(SI), X1        // X1 = work[13]
	MOVSD 112(SI), X2        // X2 = work[14]
	MOVSD 120(SI), X3        // X3 = work[15]
	MOVSD 224(SI), X4        // X4 = work[28]
	MOVSD 232(SI), X5        // X5 = work[29]
	MOVSD 240(SI), X6        // X6 = work[30]
	MOVSD 248(SI), X7        // X7 = work[31]

	// k=12 (w^12)
	MOVSD 96(R10), X10       // X10 = twiddle[12] = w^12
	MOVAPS X10, X11          // X11 = w^12
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^12
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X4, X13           // X13 = X4
	SHUFPS $0xB1, X13, X13   // X13 = (X4.im, X4.re)
	MOVAPS X4, X14           // X14 = X4
	MULPS  X11, X14          // X14 = X4 * w.re
	MULPS  X12, X13          // X13 = X4_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X4 * w^12
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X14, X8           // X8 = X0 + t
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X14, X9           // X9 = X0 - t
	MOVAPS X8, X0            // X0 = X0 + t
	MOVAPS X9, X4            // X4 = X0 - t

	// k=13 (w^13)
	MOVSD 104(R10), X10      // X10 = twiddle[13] = w^13
	MOVAPS X10, X11          // X11 = w^13
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^13
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = (X5.im, X5.re)
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5 * w.re
	MULPS  X12, X13          // X13 = X5_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X5 * w^13
	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + t
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - t
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// k=14 (w^14)
	MOVSD 112(R10), X10      // X10 = twiddle[14] = w^14
	MOVAPS X10, X11          // X11 = w^14
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^14
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X6, X13           // X13 = X6
	SHUFPS $0xB1, X13, X13   // X13 = (X6.im, X6.re)
	MOVAPS X6, X14           // X14 = X6
	MULPS  X11, X14          // X14 = X6 * w.re
	MULPS  X12, X13          // X13 = X6_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X6 * w^14
	MOVAPS X2, X8            // X8 = X2
	ADDPS  X14, X8           // X8 = X2 + t
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X14, X9           // X9 = X2 - t
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// k=15 (w^15)
	MOVSD 120(R10), X10      // X10 = twiddle[15] = w^15
	MOVAPS X10, X11          // X11 = w^15
	SHUFPS $0x00, X11, X11   // X11 = broadcast w.re
	MOVAPS X10, X12          // X12 = w^15
	SHUFPS $0x55, X12, X12   // X12 = broadcast w.im
	MOVAPS X7, X13           // X13 = X7
	SHUFPS $0xB1, X13, X13   // X13 = (X7.im, X7.re)
	MOVAPS X7, X14           // X14 = X7
	MULPS  X11, X14          // X14 = X7 * w.re
	MULPS  X12, X13          // X13 = X7_swapped * w.im
	ADDSUBPS X13, X14        // X14 = X7 * w^15
	MOVAPS X3, X8            // X8 = X3
	ADDPS  X14, X8           // X8 = X3 + t
	MOVAPS X3, X9            // X9 = X3
	SUBPS  X14, X9           // X9 = X3 - t
	MOVAPS X8, X3            // X3 = X3 + t
	MOVAPS X9, X7            // X7 = X3 - t

	// Store Part 4
	MOVSD X0, 96(SI)         // work[12] = X0
	MOVSD X1, 104(SI)        // work[13] = X1
	MOVSD X2, 112(SI)        // work[14] = X2
	MOVSD X3, 120(SI)        // work[15] = X3
	MOVSD X4, 224(SI)        // work[28] = X4
	MOVSD X5, 232(SI)        // work[29] = X5
	MOVSD X6, 240(SI)        // work[30] = X6
	MOVSD X7, 248(SI)        // work[31] = X7

	// ==================================================================
	// Copy to dst if needed
	// ==================================================================
	MOVQ dst+0(FP), R14      // R14 = dst pointer
	CMPQ R8, R14             // work buffer == dst?
	JE   size32_r2_sse2_fwd_done

	// Copy 32 complex64 elements (256 bytes) using loop
	MOVQ $16, CX             // 16 iterations (16 bytes per XMM)
	MOVQ R8, SI              // source = work buffer
size32_r2_sse2_fwd_copy_loop:
	MOVUPS (SI), X0          // X0 = work[i:i+2]
	MOVUPS X0, (R14)         // dst[i:i+2] = X0
	ADDQ $16, SI             // advance source
	ADDQ $16, R14            // advance dest
	DECQ CX                  // decrement counter
	JNZ size32_r2_sse2_fwd_copy_loop

size32_r2_sse2_fwd_done:
	MOVB $1, ret+96(FP)      // return true
	RET

size32_r2_sse2_fwd_return_false:
	MOVB $0, ret+96(FP)      // return false
	RET

// ===========================================================================
// Inverse transform, size 32, complex64, radix-2
// ===========================================================================
TEXT ·InverseSSE2Size32Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// ===== Parameter Loading =====
	MOVQ dst+0(FP), R8       // R8  = dst slice data pointer
	MOVQ src+24(FP), R9      // R9  = src slice data pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle slice data pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch slice data pointer
	LEAQ ·bitrevSSE2Size32Radix2(SB), R12  // R12 = bitrev table pointer
	MOVQ src+32(FP), R13     // R13 = src slice length

	// ===== Input Validation =====
	CMPQ R13, $32            // check n == 32
	JNE  size32_r2_sse2_inv_return_false

	MOVQ dst+8(FP), AX       // dst.len
	CMPQ AX, $32             // dst.len >= 32?
	JL   size32_r2_sse2_inv_return_false

	MOVQ twiddle+56(FP), AX  // twiddle.len
	CMPQ AX, $32             // twiddle.len >= 32?
	JL   size32_r2_sse2_inv_return_false

	MOVQ scratch+80(FP), AX  // scratch.len
	CMPQ AX, $32             // scratch.len >= 32?
	JL   size32_r2_sse2_inv_return_false

	MOVQ $32, AX             // sanity check
	CMPQ AX, $32
	JL   size32_r2_sse2_inv_return_false

	// ===== Select Working Buffer =====
	CMPQ R8, R9              // dst == src?
	JNE  size32_r2_sse2_inv_use_dst
	MOVQ R11, R8             // in-place: work in scratch buffer

size32_r2_sse2_inv_use_dst:
	// ==================================================================
	// STAGES 1 & 2 (Combined with bit-reversal)
	// ==================================================================
	// Inverse uses conjugate twiddles: conj(-i) = i for stage 2
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer write position
	MOVQ $8, CX              // loop counter: 8 blocks of 4 elements
	MOVUPS ·maskNegLoPS(SB), X15 // X15 = mask for +i multiplication

size32_r2_sse2_inv_stage12_loop:
	// ----- Load 4 elements with bit-reversal -----
	MOVQ (R12), DX           // DX = bitrev index
	MOVSD (R9)(DX*8), X0     // X0 = src[bitrev[i*4 + 0]]
	MOVQ 8(R12), DX          // DX = bitrev index
	MOVSD (R9)(DX*8), X1     // X1 = src[bitrev[i*4 + 1]]
	MOVQ 16(R12), DX         // DX = bitrev index
	MOVSD (R9)(DX*8), X2     // X2 = src[bitrev[i*4 + 2]]
	MOVQ 24(R12), DX         // DX = bitrev index
	MOVSD (R9)(DX*8), X3     // X3 = src[bitrev[i*4 + 3]]
	ADDQ $32, R12            // advance bitrev pointer

	// ----- Stage 1: stride 1, twiddle w=1 -----
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X1, X8            // X8 = X0 + X1
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X1, X9            // X9 = X0 - X1
	MOVAPS X8, X0            // X0 = X0 + X1
	MOVAPS X9, X1            // X1 = X0 - X1

	MOVAPS X2, X8            // X8 = X2
	ADDPS  X3, X8            // X8 = X2 + X3
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X3, X9            // X9 = X2 - X3
	MOVAPS X8, X2            // X2 = X2 + X3
	MOVAPS X9, X3            // X3 = X2 - X3

	// ----- Stage 2: stride 2, twiddles [1, conj(-i)=i] -----
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X2, X8            // X8 = X0 + X2
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X2, X9            // X9 = X0 - X2
	MOVAPS X8, X0            // X0 = X0 + X2
	MOVAPS X9, X2            // X2 = X0 - X2

	// Butterfly (X1, X3) with w=i: t = X3 * i = (-im, re)
	MOVAPS X3, X10           // X10 = X3
	SHUFPS $0xB1, X10, X10   // X10 = (X3.im, X3.re) - swap
	XORPS  X15, X10          // X10 = (-X3.im, X3.re) = X3 * i
	MOVAPS X1, X8            // X8 = X1
	ADDPS  X10, X8           // X8 = X1 + X3*i
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X10, X9           // X9 = X1 - X3*i
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X3            // X3 = X1 - t

	// ----- Store 4 results -----
	MOVSD X0, (SI)           // work[0] = X0
	MOVSD X1, 8(SI)          // work[1] = X1
	MOVSD X2, 16(SI)         // work[2] = X2
	MOVSD X3, 24(SI)         // work[3] = X3

	ADDQ $32, SI             // advance work pointer
	DECQ CX                  // decrement counter
	JNZ  size32_r2_sse2_inv_stage12_loop

	// ==================================================================
	// STAGE 3 (Stride 4) - Inverse uses conjugate twiddles
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer (reset)
	MOVQ $4, CX              // loop counter: 4 blocks
	MOVUPS ·maskNegHiPS(SB), X15 // X15 = mask for conjugation

size32_r2_sse2_inv_stage3_loop:
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

	// ----- Butterfly 1: (X1, X5) with conj(w^4) -----
	MOVSD 32(R10), X10       // X10 = twiddle[4] = w^4
	XORPS  X15, X10          // X10 = conj(w^4)
	MOVAPS X10, X11          // X11 = conj(w^4)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^4)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = (X5.im, X5.re)
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5 * re
	MULPS  X12, X13          // X13 = X5_swapped * im
	ADDSUBPS X13, X14        // X14 = X5 * conj(w^4)
	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + t
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - t
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// ----- Butterfly 2: (X2, X6) with conj(w^8) = i -----
	MOVAPS X6, X10           // X10 = X6
	SHUFPS $0xB1, X10, X10   // X10 = (X6.im, X6.re)
	MOVUPS ·maskNegLoPS(SB), X14 // X14 = mask for +i
	XORPS  X14, X10          // X10 = X6 * i
	MOVAPS X2, X8            // X8 = X2
	ADDPS  X10, X8           // X8 = X2 + t
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X10, X9           // X9 = X2 - t
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// ----- Butterfly 3: (X3, X7) with conj(w^12) -----
	MOVSD 96(R10), X10       // X10 = twiddle[12] = w^12
	XORPS  X15, X10          // X10 = conj(w^12)
	MOVAPS X10, X11          // X11 = conj(w^12)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^12)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X7, X13           // X13 = X7
	SHUFPS $0xB1, X13, X13   // X13 = (X7.im, X7.re)
	MOVAPS X7, X14           // X14 = X7
	MULPS  X11, X14          // X14 = X7 * re
	MULPS  X12, X13          // X13 = X7_swapped * im
	ADDSUBPS X13, X14        // X14 = X7 * conj(w^12)
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
	JNZ  size32_r2_sse2_inv_stage3_loop

	// ==================================================================
	// STAGE 4 (Stride 8) - Inverse uses conjugate twiddles
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer (reset)
	MOVQ $2, CX              // loop counter: 2 blocks

size32_r2_sse2_inv_stage4_loop:
	// ----- Part 1: k=0..3 -----
	MOVSD (SI), X0           // X0 = work[0]
	MOVSD 8(SI), X1          // X1 = work[1]
	MOVSD 16(SI), X2         // X2 = work[2]
	MOVSD 24(SI), X3         // X3 = work[3]
	MOVSD 64(SI), X4         // X4 = work[8]
	MOVSD 72(SI), X5         // X5 = work[9]
	MOVSD 80(SI), X6         // X6 = work[10]
	MOVSD 88(SI), X7         // X7 = work[11]

	// k=0 (w^0=1)
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X4, X8            // X8 = X0 + X4
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X4, X9            // X9 = X0 - X4
	MOVAPS X8, X0            // X0 = X0 + X4
	MOVAPS X9, X4            // X4 = X0 - X4

	// k=1 (conj(w^2))
	MOVSD 16(R10), X10       // X10 = twiddle[2]
	XORPS  X15, X10          // X10 = conj(w^2)
	MOVAPS X10, X11          // X11 = conj(w^2)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^2)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5 * re
	MULPS  X12, X13          // X13 = X5_swap * im
	ADDSUBPS X13, X14        // X14 = X5 * conj(w^2)
	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + t
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - t
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// k=2 (conj(w^4))
	MOVSD 32(R10), X10       // X10 = twiddle[4]
	XORPS  X15, X10          // X10 = conj(w^4)
	MOVAPS X10, X11          // X11 = conj(w^4)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^4)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X6, X13           // X13 = X6
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X6, X14           // X14 = X6
	MULPS  X11, X14          // X14 = X6 * re
	MULPS  X12, X13          // X13 = X6_swap * im
	ADDSUBPS X13, X14        // X14 = X6 * conj(w^4)
	MOVAPS X2, X8            // X8 = X2
	ADDPS  X14, X8           // X8 = X2 + t
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X14, X9           // X9 = X2 - t
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// k=3 (conj(w^6))
	MOVSD 48(R10), X10       // X10 = twiddle[6]
	XORPS  X15, X10          // X10 = conj(w^6)
	MOVAPS X10, X11          // X11 = conj(w^6)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^6)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X7, X13           // X13 = X7
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X7, X14           // X14 = X7
	MULPS  X11, X14          // X14 = X7 * re
	MULPS  X12, X13          // X13 = X7_swap * im
	ADDSUBPS X13, X14        // X14 = X7 * conj(w^6)
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

	// k=4 (conj(w^8)=i)
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

	// k=5 (conj(w^10))
	MOVSD 80(R10), X10       // X10 = twiddle[10]
	XORPS  X15, X10          // X10 = conj(w^10)
	MOVAPS X10, X11          // X11 = conj(w^10)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^10)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5 * re
	MULPS  X12, X13          // X13 = X5_swap * im
	ADDSUBPS X13, X14        // X14 = X5 * conj(w^10)
	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + t
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - t
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// k=6 (conj(w^12))
	MOVSD 96(R10), X10       // X10 = twiddle[12]
	XORPS  X15, X10          // X10 = conj(w^12)
	MOVAPS X10, X11          // X11 = conj(w^12)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^12)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X6, X13           // X13 = X6
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X6, X14           // X14 = X6
	MULPS  X11, X14          // X14 = X6 * re
	MULPS  X12, X13          // X13 = X6_swap * im
	ADDSUBPS X13, X14        // X14 = X6 * conj(w^12)
	MOVAPS X2, X8            // X8 = X2
	ADDPS  X14, X8           // X8 = X2 + t
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X14, X9           // X9 = X2 - t
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// k=7 (conj(w^14))
	MOVSD 112(R10), X10      // X10 = twiddle[14]
	XORPS  X15, X10          // X10 = conj(w^14)
	MOVAPS X10, X11          // X11 = conj(w^14)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^14)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X7, X13           // X13 = X7
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X7, X14           // X14 = X7
	MULPS  X11, X14          // X14 = X7 * re
	MULPS  X12, X13          // X13 = X7_swap * im
	ADDSUBPS X13, X14        // X14 = X7 * conj(w^14)
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

	ADDQ $128, SI            // advance by 16 elements
	DECQ CX                  // decrement counter
	JNZ  size32_r2_sse2_inv_stage4_loop

	// ==================================================================
	// STAGE 5 (Stride 16) - Final Stage with conjugate twiddles
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer (reset)

	// ----- Part 1: k=0..3 -----
	MOVSD (SI), X0           // X0 = work[0]
	MOVSD 8(SI), X1          // X1 = work[1]
	MOVSD 16(SI), X2         // X2 = work[2]
	MOVSD 24(SI), X3         // X3 = work[3]
	MOVSD 128(SI), X4        // X4 = work[16]
	MOVSD 136(SI), X5        // X5 = work[17]
	MOVSD 144(SI), X6        // X6 = work[18]
	MOVSD 152(SI), X7        // X7 = work[19]

	// k=0 (w^0=1)
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X4, X8            // X8 = X0 + X4
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X4, X9            // X9 = X0 - X4
	MOVAPS X8, X0            // X0 = X0 + X4
	MOVAPS X9, X4            // X4 = X0 - X4

	// k=1 (conj(w^1))
	MOVSD 8(R10), X10        // X10 = twiddle[1]
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

	// k=2 (conj(w^2))
	MOVSD 16(R10), X10       // X10 = twiddle[2]
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

	// k=3 (conj(w^3))
	MOVSD 24(R10), X10       // X10 = twiddle[3]
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
	MOVSD X4, 128(SI)        // work[16] = X4
	MOVSD X5, 136(SI)        // work[17] = X5
	MOVSD X6, 144(SI)        // work[18] = X6
	MOVSD X7, 152(SI)        // work[19] = X7

	// ----- Part 2: k=4..7 -----
	MOVSD 32(SI), X0         // X0 = work[4]
	MOVSD 40(SI), X1         // X1 = work[5]
	MOVSD 48(SI), X2         // X2 = work[6]
	MOVSD 56(SI), X3         // X3 = work[7]
	MOVSD 160(SI), X4        // X4 = work[20]
	MOVSD 168(SI), X5        // X5 = work[21]
	MOVSD 176(SI), X6        // X6 = work[22]
	MOVSD 184(SI), X7        // X7 = work[23]

	// k=4 (conj(w^4))
	MOVSD 32(R10), X10       // X10 = twiddle[4]
	XORPS  X15, X10          // X10 = conj(w^4)
	MOVAPS X10, X11          // X11 = conj(w^4)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^4)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X4, X13           // X13 = X4
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X4, X14           // X14 = X4
	MULPS  X11, X14          // X14 = X4 * re
	MULPS  X12, X13          // X13 = X4_swap * im
	ADDSUBPS X13, X14        // X14 = X4 * conj(w^4)
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X14, X8           // X8 = X0 + t
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X14, X9           // X9 = X0 - t
	MOVAPS X8, X0            // X0 = X0 + t
	MOVAPS X9, X4            // X4 = X0 - t

	// k=5 (conj(w^5))
	MOVSD 40(R10), X10       // X10 = twiddle[5]
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

	// k=6 (conj(w^6))
	MOVSD 48(R10), X10       // X10 = twiddle[6]
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

	// k=7 (conj(w^7))
	MOVSD 56(R10), X10       // X10 = twiddle[7]
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
	MOVSD X4, 160(SI)        // work[20] = X4
	MOVSD X5, 168(SI)        // work[21] = X5
	MOVSD X6, 176(SI)        // work[22] = X6
	MOVSD X7, 184(SI)        // work[23] = X7

	// ----- Part 3: k=8..11 -----
	MOVSD 64(SI), X0         // X0 = work[8]
	MOVSD 72(SI), X1         // X1 = work[9]
	MOVSD 80(SI), X2         // X2 = work[10]
	MOVSD 88(SI), X3         // X3 = work[11]
	MOVSD 192(SI), X4        // X4 = work[24]
	MOVSD 200(SI), X5        // X5 = work[25]
	MOVSD 208(SI), X6        // X6 = work[26]
	MOVSD 216(SI), X7        // X7 = work[27]

	// k=8 (conj(w^8)=i)
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

	// k=9 (conj(w^9))
	MOVSD 72(R10), X10       // X10 = twiddle[9]
	XORPS  X15, X10          // X10 = conj(w^9)
	MOVAPS X10, X11          // X11 = conj(w^9)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^9)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5 * re
	MULPS  X12, X13          // X13 = X5_swap * im
	ADDSUBPS X13, X14        // X14 = X5 * conj(w^9)
	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + t
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - t
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// k=10 (conj(w^10))
	MOVSD 80(R10), X10       // X10 = twiddle[10]
	XORPS  X15, X10          // X10 = conj(w^10)
	MOVAPS X10, X11          // X11 = conj(w^10)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^10)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X6, X13           // X13 = X6
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X6, X14           // X14 = X6
	MULPS  X11, X14          // X14 = X6 * re
	MULPS  X12, X13          // X13 = X6_swap * im
	ADDSUBPS X13, X14        // X14 = X6 * conj(w^10)
	MOVAPS X2, X8            // X8 = X2
	ADDPS  X14, X8           // X8 = X2 + t
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X14, X9           // X9 = X2 - t
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// k=11 (conj(w^11))
	MOVSD 88(R10), X10       // X10 = twiddle[11]
	XORPS  X15, X10          // X10 = conj(w^11)
	MOVAPS X10, X11          // X11 = conj(w^11)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^11)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X7, X13           // X13 = X7
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X7, X14           // X14 = X7
	MULPS  X11, X14          // X14 = X7 * re
	MULPS  X12, X13          // X13 = X7_swap * im
	ADDSUBPS X13, X14        // X14 = X7 * conj(w^11)
	MOVAPS X3, X8            // X8 = X3
	ADDPS  X14, X8           // X8 = X3 + t
	MOVAPS X3, X9            // X9 = X3
	SUBPS  X14, X9           // X9 = X3 - t
	MOVAPS X8, X3            // X3 = X3 + t
	MOVAPS X9, X7            // X7 = X3 - t

	// Store Part 3
	MOVSD X0, 64(SI)         // work[8] = X0
	MOVSD X1, 72(SI)         // work[9] = X1
	MOVSD X2, 80(SI)         // work[10] = X2
	MOVSD X3, 88(SI)         // work[11] = X3
	MOVSD X4, 192(SI)        // work[24] = X4
	MOVSD X5, 200(SI)        // work[25] = X5
	MOVSD X6, 208(SI)        // work[26] = X6
	MOVSD X7, 216(SI)        // work[27] = X7

	// ----- Part 4: k=12..15 -----
	MOVSD 96(SI), X0         // X0 = work[12]
	MOVSD 104(SI), X1        // X1 = work[13]
	MOVSD 112(SI), X2        // X2 = work[14]
	MOVSD 120(SI), X3        // X3 = work[15]
	MOVSD 224(SI), X4        // X4 = work[28]
	MOVSD 232(SI), X5        // X5 = work[29]
	MOVSD 240(SI), X6        // X6 = work[30]
	MOVSD 248(SI), X7        // X7 = work[31]

	// k=12 (conj(w^12))
	MOVSD 96(R10), X10       // X10 = twiddle[12]
	XORPS  X15, X10          // X10 = conj(w^12)
	MOVAPS X10, X11          // X11 = conj(w^12)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^12)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X4, X13           // X13 = X4
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X4, X14           // X14 = X4
	MULPS  X11, X14          // X14 = X4 * re
	MULPS  X12, X13          // X13 = X4_swap * im
	ADDSUBPS X13, X14        // X14 = X4 * conj(w^12)
	MOVAPS X0, X8            // X8 = X0
	ADDPS  X14, X8           // X8 = X0 + t
	MOVAPS X0, X9            // X9 = X0
	SUBPS  X14, X9           // X9 = X0 - t
	MOVAPS X8, X0            // X0 = X0 + t
	MOVAPS X9, X4            // X4 = X0 - t

	// k=13 (conj(w^13))
	MOVSD 104(R10), X10      // X10 = twiddle[13]
	XORPS  X15, X10          // X10 = conj(w^13)
	MOVAPS X10, X11          // X11 = conj(w^13)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^13)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X5, X13           // X13 = X5
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X5, X14           // X14 = X5
	MULPS  X11, X14          // X14 = X5 * re
	MULPS  X12, X13          // X13 = X5_swap * im
	ADDSUBPS X13, X14        // X14 = X5 * conj(w^13)
	MOVAPS X1, X8            // X8 = X1
	ADDPS  X14, X8           // X8 = X1 + t
	MOVAPS X1, X9            // X9 = X1
	SUBPS  X14, X9           // X9 = X1 - t
	MOVAPS X8, X1            // X1 = X1 + t
	MOVAPS X9, X5            // X5 = X1 - t

	// k=14 (conj(w^14))
	MOVSD 112(R10), X10      // X10 = twiddle[14]
	XORPS  X15, X10          // X10 = conj(w^14)
	MOVAPS X10, X11          // X11 = conj(w^14)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^14)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X6, X13           // X13 = X6
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X6, X14           // X14 = X6
	MULPS  X11, X14          // X14 = X6 * re
	MULPS  X12, X13          // X13 = X6_swap * im
	ADDSUBPS X13, X14        // X14 = X6 * conj(w^14)
	MOVAPS X2, X8            // X8 = X2
	ADDPS  X14, X8           // X8 = X2 + t
	MOVAPS X2, X9            // X9 = X2
	SUBPS  X14, X9           // X9 = X2 - t
	MOVAPS X8, X2            // X2 = X2 + t
	MOVAPS X9, X6            // X6 = X2 - t

	// k=15 (conj(w^15))
	MOVSD 120(R10), X10      // X10 = twiddle[15]
	XORPS  X15, X10          // X10 = conj(w^15)
	MOVAPS X10, X11          // X11 = conj(w^15)
	SHUFPS $0x00, X11, X11   // X11 = broadcast re
	MOVAPS X10, X12          // X12 = conj(w^15)
	SHUFPS $0x55, X12, X12   // X12 = broadcast im
	MOVAPS X7, X13           // X13 = X7
	SHUFPS $0xB1, X13, X13   // X13 = swapped
	MOVAPS X7, X14           // X14 = X7
	MULPS  X11, X14          // X14 = X7 * re
	MULPS  X12, X13          // X13 = X7_swap * im
	ADDSUBPS X13, X14        // X14 = X7 * conj(w^15)
	MOVAPS X3, X8            // X8 = X3
	ADDPS  X14, X8           // X8 = X3 + t
	MOVAPS X3, X9            // X9 = X3
	SUBPS  X14, X9           // X9 = X3 - t
	MOVAPS X8, X3            // X3 = X3 + t
	MOVAPS X9, X7            // X7 = X3 - t

	// Store Part 4
	MOVSD X0, 96(SI)         // work[12] = X0
	MOVSD X1, 104(SI)        // work[13] = X1
	MOVSD X2, 112(SI)        // work[14] = X2
	MOVSD X3, 120(SI)        // work[15] = X3
	MOVSD X4, 224(SI)        // work[28] = X4
	MOVSD X5, 232(SI)        // work[29] = X5
	MOVSD X6, 240(SI)        // work[30] = X6
	MOVSD X7, 248(SI)        // work[31] = X7

	// ==================================================================
	// Scale by 1/N (1/32 = 0.03125)
	// ==================================================================
	MOVSS ·thirtySecond32(SB), X15 // X15 = 0.03125f
	SHUFPS $0x00, X15, X15   // X15 = broadcast to all lanes

	// Scale all 32 elements using loop
	MOVQ $16, CX             // 16 iterations (2 elements per XMM)
	MOVQ R8, SI              // SI = work buffer
size32_r2_sse2_inv_scale_loop:
	MOVUPS (SI), X0          // X0 = work[i:i+2]
	MULPS X15, X0            // X0 *= 1/32
	MOVUPS X0, (SI)          // work[i:i+2] = X0
	ADDQ $16, SI             // advance pointer
	DECQ CX                  // decrement counter
	JNZ size32_r2_sse2_inv_scale_loop

	// ==================================================================
	// Copy to dst if needed
	// ==================================================================
	MOVQ dst+0(FP), R14      // R14 = dst pointer
	CMPQ R8, R14             // work == dst?
	JE   size32_r2_sse2_inv_done

	MOVQ $16, CX             // 16 iterations
	MOVQ R8, SI              // SI = work buffer
size32_r2_sse2_inv_copy_loop:
	MOVUPS (SI), X0          // X0 = work[i:i+2]
	MOVUPS X0, (R14)         // dst[i:i+2] = X0
	ADDQ $16, SI             // advance source
	ADDQ $16, R14            // advance dest
	DECQ CX                  // decrement counter
	JNZ size32_r2_sse2_inv_copy_loop

size32_r2_sse2_inv_done:
	MOVB $1, ret+96(FP)      // return true
	RET

size32_r2_sse2_inv_return_false:
	MOVB $0, ret+96(FP)      // return false
	RET

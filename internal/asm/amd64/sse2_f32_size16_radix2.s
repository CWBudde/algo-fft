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
// Register Management (Tiled Approach):
// We have 16 XMM registers (X0-X15).
// We process data in blocks to fit within registers + scratch.
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 16, complex64, radix-2 variant
// ===========================================================================
TEXT ·ForwardSSE2Size16Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	LEAQ ·bitrevSSE2Size16Radix2(SB), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 16)

	// Verify n == 16
	CMPQ R13, $16
	JNE  size16_r2_sse2_fwd_return_false

	// Validate all slice lengths >= 16
	MOVQ dst+8(FP), AX
	CMPQ AX, $16
	JL   size16_r2_sse2_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $16
	JL   size16_r2_sse2_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $16
	JL   size16_r2_sse2_fwd_return_false

	MOVQ $16, AX
	CMPQ AX, $16
	JL   size16_r2_sse2_fwd_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size16_r2_sse2_fwd_use_dst
	MOVQ R11, R8             // In-place: use scratch

size16_r2_sse2_fwd_use_dst:
	// ==================================================================
	// Bit-reversal permutation
	// ==================================================================
	XORQ CX, CX
bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVSD (R9)(DX*8), X0
	MOVSD X0, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $16
	JL   bitrev_loop

	// ==================================================================
	// Stage 1 & 2 (Combined)
	// Process 4 elements at a time: (x0,x1,x2,x3), (x4,x5,x6,x7)...
	// Stage 1 stride 1: (0,1), (2,3)
	// Stage 2 stride 2: (0,2), (1,3)
	// ==================================================================
	MOVQ R8, SI              // SI = work buffer
	MOVQ $4, CX              // Loop counter (4 blocks of 4)
	
	// Pre-load constant masks/twiddles for Stage 2
	MOVUPS ·maskNegHiPS(SB), X15 // Negate High mask (for -i)

stage12_loop:
	// Load 4 elements
	MOVSD (SI), X0
	MOVSD 8(SI), X1
	MOVSD 16(SI), X2
	MOVSD 24(SI), X3

	// --- Stage 1 (stride 1, w=1) ---
	// Butterfly (X0, X1)
	MOVAPS X0, X8
	ADDPS  X1, X8
	MOVAPS X0, X9
	SUBPS  X1, X9
	MOVAPS X8, X0
	MOVAPS X9, X1

	// Butterfly (X2, X3)
	MOVAPS X2, X8
	ADDPS  X3, X8
	MOVAPS X2, X9
	SUBPS  X3, X9
	MOVAPS X8, X2
	MOVAPS X9, X3

	// --- Stage 2 (stride 2, w=[1, -i]) ---
	// Butterfly (X0, X2) with w=1
	MOVAPS X0, X8
	ADDPS  X2, X8
	MOVAPS X0, X9
	SUBPS  X2, X9
	MOVAPS X8, X0
	MOVAPS X9, X2

	// Butterfly (X1, X3) with w=-i
	// t = X3 * (-i) = (im, -re)
	MOVAPS X3, X10
	SHUFPS $0xB1, X10, X10   // swap: (re, im) -> (im, re)
	XORPS  X15, X10          // negate high: (im, -re)

	MOVAPS X1, X8
	ADDPS  X10, X8
	MOVAPS X1, X9
	SUBPS  X10, X9
	MOVAPS X8, X1
	MOVAPS X9, X3

	// Store back
	MOVSD X0, (SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)

	ADDQ $32, SI
	DECQ CX
	JNZ  stage12_loop

	// ==================================================================
	// Stage 3 (Stride 4)
	// Blocks of 8: (0,4), (1,5), (2,6), (3,7)
	// Twiddles: 1, w2, w4, w6 (from size 16 indices)
	// Wait, indices:
	//   For size 16, Stage 3 twiddles are W_N^k where k = 0, 2, 4, 6?
	//   Standard DIT:
	//   Stage 1: k=0
	//   Stage 2: k=0, 4 (if N=16) -> indices into twiddle array of size 8?
	//   Let's check twiddle indices.
	//   Twiddle array usually holds W_N^0, W_N^1 ... W_N^(N/2-1)
	//   Size 16, twiddles are W_16^0..W_16^7.
	//   Stage 1 (len 2): w^0
	//   Stage 2 (len 4): w^0, w^4 ? No, standard DIT indices are:
	//     Stage s (butterfly group size 2^s):
	//     k goes 0 to 2^(s-1)-1.
	//     Twiddle index = k * (N/2^s)
	//     Stage 1 (size 2): k=0. Index = 0 * 8 = 0.
	//     Stage 2 (size 4): k=0,1. Index = 0*4=0, 1*4=4.
	//       Wait, my previous manual unroll for size 8 used:
	//       Stage 2 (size 4): w^0, w^2 (for N=8).
	//       Here N=16. So indices should be multiplied by 2 compared to N=8?
	//       Size 16 Stage 2 indices: 0, 4. (Twiddles W_16^0, W_16^4).
	//       W_16^4 = -i. Correct.
	//     Stage 3 (size 8): k=0,1,2,3.
	//       Indices: 0*2=0, 1*2=2, 2*2=4, 3*2=6.
	//       Twiddles: W^0, W^2, W^4, W^6.
	// ==================================================================
	
	MOVQ R8, SI              // SI = work buffer
	MOVQ $2, CX              // Loop counter (2 blocks of 8)

stage3_loop:
	// Load 8 elements: x0..x7 (relative to block)
	MOVSD (SI), X0
	MOVSD 8(SI), X1
	MOVSD 16(SI), X2
	MOVSD 24(SI), X3
	MOVSD 32(SI), X4
	MOVSD 40(SI), X5
	MOVSD 48(SI), X6
	MOVSD 56(SI), X7

	// Butterfly 0 (w^0 = 1): (X0, X4)
	MOVAPS X0, X8
	ADDPS  X4, X8
	MOVAPS X0, X9
	SUBPS  X4, X9
	MOVAPS X8, X0
	MOVAPS X9, X4

	// Butterfly 1 (w^2): (X1, X5)
	// Load w2
	MOVSD 16(R10), X10       // twiddle[2]
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11   // w.re
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12   // w.im
	MOVAPS X5, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X5, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14        // t = X5 * w2

	MOVAPS X1, X8
	ADDPS  X14, X8
	MOVAPS X1, X9
	SUBPS  X14, X9
	MOVAPS X8, X1
	MOVAPS X9, X5

	// Butterfly 2 (w^4 = -i): (X2, X6)
	// Optimization: Use -i multiplication
	MOVAPS X6, X10
	SHUFPS $0xB1, X10, X10
	MOVUPS ·maskNegHiPS(SB), X15
	XORPS  X15, X10          // t = X6 * -i

	MOVAPS X2, X8
	ADDPS  X10, X8
	MOVAPS X2, X9
	SUBPS  X10, X9
	MOVAPS X8, X2
	MOVAPS X9, X6

	// Butterfly 3 (w^6): (X3, X7)
	// Load w6
	MOVSD 48(R10), X10       // twiddle[6]
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X7, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X7, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14        // t = X7 * w6

	MOVAPS X3, X8
	ADDPS  X14, X8
	MOVAPS X3, X9
	SUBPS  X14, X9
	MOVAPS X8, X3
	MOVAPS X9, X7

	// Store back
	MOVSD X0, (SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)
	MOVSD X4, 32(SI)
	MOVSD X5, 40(SI)
	MOVSD X6, 48(SI)
	MOVSD X7, 56(SI)

	ADDQ $64, SI
	DECQ CX
	JNZ  stage3_loop

	// ==================================================================
	// Stage 4 (Stride 8)
	// 1 Block of 16 (entire array).
	// k=0..7. Twiddles W^0..W^7.
	// Split into two halves to save registers:
	// Part 1: k=0..3 -> (0,8), (1,9), (2,10), (3,11) using w0..w3
	// Part 2: k=4..7 -> (4,12), (5,13), (6,14), (7,15) using w4..w7
	// ==================================================================
	MOVQ R8, SI

	// --- Part 1 (k=0..3) ---
	MOVSD (SI), X0           // x0
	MOVSD 8(SI), X1          // x1
	MOVSD 16(SI), X2         // x2
	MOVSD 24(SI), X3         // x3
	MOVSD 64(SI), X4         // x8
	MOVSD 72(SI), X5         // x9
	MOVSD 80(SI), X6         // x10
	MOVSD 88(SI), X7         // x11

	// k=0 (w=1): (X0, X4)
	MOVAPS X0, X8
	ADDPS  X4, X8
	MOVAPS X0, X9
	SUBPS  X4, X9
	MOVAPS X8, X0
	MOVAPS X9, X4

	// k=1 (w1): (X1, X5)
	MOVSD  8(R10), X10       // w1
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X5, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X5, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14        // t

	MOVAPS X1, X8
	ADDPS  X14, X8
	MOVAPS X1, X9
	SUBPS  X14, X9
	MOVAPS X8, X1
	MOVAPS X9, X5

	// k=2 (w2): (X2, X6)
	MOVSD  16(R10), X10      // w2
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X6, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X6, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14        // t

	MOVAPS X2, X8
	ADDPS  X14, X8
	MOVAPS X2, X9
	SUBPS  X14, X9
	MOVAPS X8, X2
	MOVAPS X9, X6

	// k=3 (w3): (X3, X7)
	MOVSD  24(R10), X10      // w3
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X7, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X7, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14        // t

	MOVAPS X3, X8
	ADDPS  X14, X8
	MOVAPS X3, X9
	SUBPS  X14, X9
	MOVAPS X8, X3
	MOVAPS X9, X7

	// Store Part 1
	MOVSD X0, (SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)
	MOVSD X4, 64(SI)
	MOVSD X5, 72(SI)
	MOVSD X6, 80(SI)
	MOVSD X7, 88(SI)

	// --- Part 2 (k=4..7) ---
	MOVSD 32(SI), X0         // x4
	MOVSD 40(SI), X1         // x5
	MOVSD 48(SI), X2         // x6
	MOVSD 56(SI), X3         // x7
	MOVSD 96(SI), X4         // x12
	MOVSD 104(SI), X5        // x13
	MOVSD 112(SI), X6        // x14
	MOVSD 120(SI), X7        // x15

	// k=4 (w4 = -i): (X0, X4)
	MOVAPS X4, X10
	SHUFPS $0xB1, X10, X10
	MOVUPS ·maskNegHiPS(SB), X15
	XORPS  X15, X10

	MOVAPS X0, X8
	ADDPS  X10, X8
	MOVAPS X0, X9
	SUBPS  X10, X9
	MOVAPS X8, X0
	MOVAPS X9, X4

	// k=5 (w5): (X1, X5)
	MOVSD  40(R10), X10      // w5
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X5, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X5, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14

	MOVAPS X1, X8
	ADDPS  X14, X8
	MOVAPS X1, X9
	SUBPS  X14, X9
	MOVAPS X8, X1
	MOVAPS X9, X5

	// k=6 (w6): (X2, X6)
	MOVSD  48(R10), X10      // w6
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X6, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X6, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14

	MOVAPS X2, X8
	ADDPS  X14, X8
	MOVAPS X2, X9
	SUBPS  X14, X9
	MOVAPS X8, X2
	MOVAPS X9, X6

	// k=7 (w7): (X3, X7)
	MOVSD  56(R10), X10      // w7
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X7, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X7, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14

	MOVAPS X3, X8
	ADDPS  X14, X8
	MOVAPS X3, X9
	SUBPS  X14, X9
	MOVAPS X8, X3
	MOVAPS X9, X7

	// Store Part 2
	MOVSD X0, 32(SI)
	MOVSD X1, 40(SI)
	MOVSD X2, 48(SI)
	MOVSD X3, 56(SI)
	MOVSD X4, 96(SI)
	MOVSD X5, 104(SI)
	MOVSD X6, 112(SI)
	MOVSD X7, 120(SI)

	// Copy to dst if needed
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   size16_r2_sse2_fwd_done

	// Copy 16 elements (8 MOVUPS)
	// Actually 8 MOVUPS for complex64 (128bit) copies 2 elements per mov
	// 16 elements = 8 XMM vectors
	MOVUPS (R8), X0
	MOVUPS X0, (R14)
	MOVUPS 16(R8), X1
	MOVUPS X1, 16(R14)
	MOVUPS 32(R8), X2
	MOVUPS X2, 32(R14)
	MOVUPS 48(R8), X3
	MOVUPS X3, 48(R14)
	MOVUPS 64(R8), X4
	MOVUPS X4, 64(R14)
	MOVUPS 80(R8), X5
	MOVUPS X5, 80(R14)
	MOVUPS 96(R8), X6
	MOVUPS X6, 96(R14)
	MOVUPS 112(R8), X7
	MOVUPS X7, 112(R14)

size16_r2_sse2_fwd_done:
	MOVB $1, ret+96(FP)
	RET

size16_r2_sse2_fwd_return_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 16, complex64, radix-2 variant
// ===========================================================================
TEXT ·InverseSSE2Size16Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size16Radix2(SB), R12
	MOVQ src+32(FP), R13

	// Verify n == 16
	CMPQ R13, $16
	JNE  size16_r2_sse2_inv_return_false

	// Validate all slice lengths >= 16
	MOVQ dst+8(FP), AX
	CMPQ AX, $16
	JL   size16_r2_sse2_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $16
	JL   size16_r2_sse2_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $16
	JL   size16_r2_sse2_inv_return_false

	MOVQ $16, AX
	CMPQ AX, $16
	JL   size16_r2_sse2_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size16_r2_sse2_inv_use_dst
	MOVQ R11, R8

size16_r2_sse2_inv_use_dst:
	// Bit-reversal
	XORQ CX, CX
inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVSD (R9)(DX*8), X0
	MOVSD X0, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $16
	JL   inv_bitrev_loop

	// Stage 1 & 2 (Combined)
	MOVQ R8, SI
	MOVQ $4, CX
	MOVUPS ·maskNegLoPS(SB), X15 // Negate Low mask (for i)

inv_stage12_loop:
	MOVSD (SI), X0
	MOVSD 8(SI), X1
	MOVSD 16(SI), X2
	MOVSD 24(SI), X3

	// --- Stage 1 ---
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

	// --- Stage 2 (w=[1, i]) ---
	// Butterfly (X0, X2) w=1
	MOVAPS X0, X8
	ADDPS  X2, X8
	MOVAPS X0, X9
	SUBPS  X2, X9
	MOVAPS X8, X0
	MOVAPS X9, X2

	// Butterfly (X1, X3) w=i
	// t = X3 * i = (-im, re)
	MOVAPS X3, X10
	SHUFPS $0xB1, X10, X10
	XORPS  X15, X10

	MOVAPS X1, X8
	ADDPS  X10, X8
	MOVAPS X1, X9
	SUBPS  X10, X9
	MOVAPS X8, X1
	MOVAPS X9, X3

	MOVSD X0, (SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)

	ADDQ $32, SI
	DECQ CX
	JNZ  inv_stage12_loop

	// Stage 3 (Stride 4)
	MOVQ R8, SI
	MOVQ $2, CX
	MOVUPS ·maskNegHiPS(SB), X15 // Load maskNegHiPS for conjugation

inv_stage3_loop:
	MOVSD (SI), X0
	MOVSD 8(SI), X1
	MOVSD 16(SI), X2
	MOVSD 24(SI), X3
	MOVSD 32(SI), X4
	MOVSD 40(SI), X5
	MOVSD 48(SI), X6
	MOVSD 56(SI), X7

	// Butterfly 0 (w^0=1)
	MOVAPS X0, X8
	ADDPS  X4, X8
	MOVAPS X0, X9
	SUBPS  X4, X9
	MOVAPS X8, X0
	MOVAPS X9, X4

	// Butterfly 1 (conj(w2))
	MOVSD 16(R10), X10
	XORPS  X15, X10          // conjugate
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X5, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X5, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14

	MOVAPS X1, X8
	ADDPS  X14, X8
	MOVAPS X1, X9
	SUBPS  X14, X9
	MOVAPS X8, X1
	MOVAPS X9, X5

	// Butterfly 2 (conj(w4) = conj(-i) = i)
	// t = X6 * i = (-im, re)
	MOVAPS X6, X10
	SHUFPS $0xB1, X10, X10
	MOVUPS ·maskNegLoPS(SB), X14 // Use Lo mask for i
	XORPS  X14, X10

	MOVAPS X2, X8
	ADDPS  X10, X8
	MOVAPS X2, X9
	SUBPS  X10, X9
	MOVAPS X8, X2
	MOVAPS X9, X6

	// Butterfly 3 (conj(w6))
	MOVSD 48(R10), X10
	XORPS  X15, X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X7, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X7, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14

	MOVAPS X3, X8
	ADDPS  X14, X8
	MOVAPS X3, X9
	SUBPS  X14, X9
	MOVAPS X8, X3
	MOVAPS X9, X7

	MOVSD X0, (SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)
	MOVSD X4, 32(SI)
	MOVSD X5, 40(SI)
	MOVSD X6, 48(SI)
	MOVSD X7, 56(SI)

	ADDQ $64, SI
	DECQ CX
	JNZ  inv_stage3_loop

	// Stage 4 (Stride 8)
	MOVQ R8, SI
	MOVUPS ·maskNegHiPS(SB), X15 // Reload conjugate mask

	// --- Part 1 (k=0..3) ---
	MOVSD (SI), X0
	MOVSD 8(SI), X1
	MOVSD 16(SI), X2
	MOVSD 24(SI), X3
	MOVSD 64(SI), X4
	MOVSD 72(SI), X5
	MOVSD 80(SI), X6
	MOVSD 88(SI), X7

	// k=0 (w=1)
	MOVAPS X0, X8
	ADDPS  X4, X8
	MOVAPS X0, X9
	SUBPS  X4, X9
	MOVAPS X8, X0
	MOVAPS X9, X4

	// k=1 (conj(w1))
	MOVSD  8(R10), X10
	XORPS  X15, X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X5, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X5, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14

	MOVAPS X1, X8
	ADDPS  X14, X8
	MOVAPS X1, X9
	SUBPS  X14, X9
	MOVAPS X8, X1
	MOVAPS X9, X5

	// k=2 (conj(w2))
	MOVSD  16(R10), X10
	XORPS  X15, X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X6, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X6, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14

	MOVAPS X2, X8
	ADDPS  X14, X8
	MOVAPS X2, X9
	SUBPS  X14, X9
	MOVAPS X8, X2
	MOVAPS X9, X6

	// k=3 (conj(w3))
	MOVSD  24(R10), X10
	XORPS  X15, X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X7, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X7, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14

	MOVAPS X3, X8
	ADDPS  X14, X8
	MOVAPS X3, X9
	SUBPS  X14, X9
	MOVAPS X8, X3
	MOVAPS X9, X7

	MOVSD X0, (SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)
	MOVSD X4, 64(SI)
	MOVSD X5, 72(SI)
	MOVSD X6, 80(SI)
	MOVSD X7, 88(SI)

	// --- Part 2 (k=4..7) ---
	MOVSD 32(SI), X0
	MOVSD 40(SI), X1
	MOVSD 48(SI), X2
	MOVSD 56(SI), X3
	MOVSD 96(SI), X4
	MOVSD 104(SI), X5
	MOVSD 112(SI), X6
	MOVSD 120(SI), X7

	// k=4 (conj(w4)=conj(-i)=i)
	MOVAPS X4, X10
	SHUFPS $0xB1, X10, X10
	MOVUPS ·maskNegLoPS(SB), X14
	XORPS  X14, X10

	MOVAPS X0, X8
	ADDPS  X10, X8
	MOVAPS X0, X9
	SUBPS  X10, X9
	MOVAPS X8, X0
	MOVAPS X9, X4

	// k=5 (conj(w5))
	MOVSD  40(R10), X10
	XORPS  X15, X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X5, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X5, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14

	MOVAPS X1, X8
	ADDPS  X14, X8
	MOVAPS X1, X9
	SUBPS  X14, X9
	MOVAPS X8, X1
	MOVAPS X9, X5

	// k=6 (conj(w6))
	MOVSD  48(R10), X10
	XORPS  X15, X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X6, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X6, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14

	MOVAPS X2, X8
	ADDPS  X14, X8
	MOVAPS X2, X9
	SUBPS  X14, X9
	MOVAPS X8, X2
	MOVAPS X9, X6

	// k=7 (conj(w7))
	MOVSD  56(R10), X10
	XORPS  X15, X10
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X7, X13
	SHUFPS $0xB1, X13, X13
	MOVAPS X7, X14
	MULPS  X11, X14
	MULPS  X12, X13
	ADDSUBPS X13, X14

	MOVAPS X3, X8
	ADDPS  X14, X8
	MOVAPS X3, X9
	SUBPS  X14, X9
	MOVAPS X8, X3
	MOVAPS X9, X7

	MOVSD X0, 32(SI)
	MOVSD X1, 40(SI)
	MOVSD X2, 48(SI)
	MOVSD X3, 56(SI)
	MOVSD X4, 96(SI)
	MOVSD X5, 104(SI)
	MOVSD X6, 112(SI)
	MOVSD X7, 120(SI)

	// Scale by 1/16
	MOVSS ·sixteenth32(SB), X15
	SHUFPS $0x00, X15, X15   // broadcast
	
	MOVUPS (SI), X0
	MULPS X15, X0
	MOVUPS X0, (SI)
	MOVUPS 16(SI), X1
	MULPS X15, X1
	MOVUPS X1, 16(SI)
	MOVUPS 32(SI), X2
	MULPS X15, X2
	MOVUPS X2, 32(SI)
	MOVUPS 48(SI), X3
	MULPS X15, X3
	MOVUPS X3, 48(SI)
	MOVUPS 64(SI), X4
	MULPS X15, X4
	MOVUPS X4, 64(SI)
	MOVUPS 80(SI), X5
	MULPS X15, X5
	MOVUPS X5, 80(SI)
	MOVUPS 96(SI), X6
	MULPS X15, X6
	MOVUPS X6, 96(SI)
	MOVUPS 112(SI), X7
	MULPS X15, X7
	MOVUPS X7, 112(SI)

	// Copy to dst if needed
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   size16_r2_sse2_inv_done

	MOVUPS (R8), X0
	MOVUPS X0, (R14)
	MOVUPS 16(R8), X1
	MOVUPS X1, 16(R14)
	MOVUPS 32(R8), X2
	MOVUPS X2, 32(R14)
	MOVUPS 48(R8), X3
	MOVUPS X3, 48(R14)
	MOVUPS 64(R8), X4
	MOVUPS X4, 64(R14)
	MOVUPS 80(R8), X5
	MOVUPS X5, 80(R14)
	MOVUPS 96(R8), X6
	MOVUPS X6, 96(R14)
	MOVUPS 112(R8), X7
	MOVUPS X7, 112(R14)

size16_r2_sse2_inv_done:
	MOVB $1, ret+96(FP)
	RET

size16_r2_sse2_inv_return_false:
	MOVB $0, ret+96(FP)
	RET

//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-128 Radix-4-then-2 (2×4) FFT Kernels for AMD64 (complex64)
// ===========================================================================
//
// Size 128 = 2 × 4³, implemented as radix-4-then-2 decomposition:
//   Stage 1: 32 radix-4 butterflies, stride=4, twiddle=1 (no multiply)
//   Stage 2: 8 groups × 4 butterflies, stride=16
//   Stage 3: 2 groups × 16 butterflies, stride=64
//   Stage 4: 32 radix-2 butterflies, stride=128
//
// SSE2 version: No FMA, uses ADDSUBPS for complex multiply.
//
// ===========================================================================

#include "textflag.h"

// Forward transform, size 128, complex64, radix-4-then-2 (2×4) (SSE2)
TEXT ·ForwardSSE3Size128Radix4Then2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	LEAQ ·bitrevSSE2Size128Radix4Then2(SB), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 128)

	// Verify n == 128
	CMPQ R13, $128
	JNE  size128_sse2_r4_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $128
	JL   size128_sse2_r4_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $128
	JL   size128_sse2_r4_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $128
	JL   size128_sse2_r4_return_false

	MOVQ $128, AX
	CMPQ AX, $128
	JL   size128_sse2_r4_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size128_sse2_r4_use_dst
	MOVQ R11, R8             // In-place: use scratch

size128_sse2_r4_use_dst:
	// ==================================================================
	// Bit-reversal permutation
	// ==================================================================
	XORQ CX, CX

size128_sse2_r4_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $128
	JL   size128_sse2_r4_bitrev_loop

size128_sse2_r4_stage1:
	// ==================================================================
	// Stage 1: 32 radix-4 butterflies, stride=4, twiddle=1
	// ==================================================================
	XORQ CX, CX

size128_sse2_r4_stage1_loop:
	CMPQ CX, $128
	JGE  size128_sse2_r4_stage2

	LEAQ (R8)(CX*8), SI
	MOVSD (SI), X0           // a0
	MOVSD 8(SI), X1          // a1
	MOVSD 16(SI), X2         // a2
	MOVSD 24(SI), X3         // a3

	// Radix-4 butterfly (twiddle = 1)
	MOVAPS X0, X4
	ADDPS X2, X4             // t0 = a0 + a2
	MOVAPS X0, X5
	SUBPS X2, X5             // t1 = a0 - a2
	MOVAPS X1, X6
	ADDPS X3, X6             // t2 = a1 + a3
	MOVAPS X1, X7
	SUBPS X3, X7             // t3 = a1 - a3

	// (-i)*t3 = (im, -re) for y1
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8     // swap re/im
	MOVUPS ·maskNegHiPS(SB), X9
	XORPS X9, X8             // negate high float

	// i*t3 = (-im, re) for y3
	MOVAPS X7, X11
	SHUFPS $0xB1, X11, X11
	MOVUPS ·maskNegLoPS(SB), X9
	XORPS X9, X11            // negate low float

	// Final outputs
	MOVAPS X4, X0
	ADDPS X6, X0             // y0 = t0 + t2
	MOVAPS X5, X1
	ADDPS X8, X1             // y1 = t1 + (-i)*t3
	MOVAPS X4, X2
	SUBPS X6, X2             // y2 = t0 - t2
	MOVAPS X5, X3
	ADDPS X11, X3            // y3 = t1 + i*t3

	// Store results
	MOVSD X0, (SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)

	ADDQ $4, CX
	JMP  size128_sse2_r4_stage1_loop

size128_sse2_r4_stage2:
	// ==================================================================
	// Stage 2: 8 groups x 4 butterflies, stride=16
	// ==================================================================
	XORQ BX, BX              // group index (0..7)

size128_sse2_r4_stage2_outer:
	CMPQ BX, $8
	JGE  size128_sse2_r4_stage3

	XORQ DX, DX              // butterfly index (0..3)

size128_sse2_r4_stage2_inner:
	CMPQ DX, $4
	JGE  size128_sse2_r4_stage2_next

	// Calculate indices
	MOVQ BX, SI
	SHLQ $4, SI              // SI = BX * 16
	ADDQ DX, SI              // base index

	MOVQ SI, DI
	ADDQ $4, DI              // idx1
	MOVQ SI, R14
	ADDQ $8, R14             // idx2
	MOVQ SI, R15
	ADDQ $12, R15            // idx3

	// Load twiddle factors: twiddle[DX*8], twiddle[DX*16], twiddle[DX*24]
	MOVQ DX, CX
	SHLQ $3, CX
	MOVSD (R10)(CX*8), X8    // w1

	MOVQ DX, CX
	SHLQ $4, CX
	MOVSD (R10)(CX*8), X9    // w2

	MOVQ DX, CX
	IMULQ $24, CX
	MOVSD (R10)(CX*8), X10   // w3

	// Load data
	MOVSD (R8)(SI*8), X0     // a0
	MOVSD (R8)(DI*8), X1     // a1
	MOVSD (R8)(R14*8), X2    // a2
	MOVSD (R8)(R15*8), X3    // a3

	// Complex multiply a1*w1 (SSE2: no FMA)
	MOVAPS X8, X11
	SHUFPS $0x00, X11, X11   // broadcast real
	MOVAPS X8, X12
	SHUFPS $0x55, X12, X12   // broadcast imag
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13   // swap re/im
	MULPS X12, X13
	MOVAPS X1, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X1

	// Complex multiply a2*w2
	MOVAPS X9, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X9, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X2, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X2, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X2

	// Complex multiply a3*w3
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X3, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X3, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X3

	// Radix-4 butterfly
	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	// (-i)*t3
	MOVAPS X7, X14
	SHUFPS $0xB1, X14, X14
	MOVUPS ·maskNegHiPS(SB), X15
	XORPS X15, X14

	// i*t3
	MOVAPS X7, X12
	SHUFPS $0xB1, X12, X12
	MOVUPS ·maskNegLoPS(SB), X15
	XORPS X15, X12

	// Outputs
	MOVAPS X4, X0
	ADDPS X6, X0
	MOVAPS X5, X1
	ADDPS X14, X1
	MOVAPS X4, X2
	SUBPS X6, X2
	MOVAPS X5, X3
	ADDPS X12, X3

	// Store
	MOVSD X0, (R8)(SI*8)
	MOVSD X1, (R8)(DI*8)
	MOVSD X2, (R8)(R14*8)
	MOVSD X3, (R8)(R15*8)

	INCQ DX
	JMP  size128_sse2_r4_stage2_inner

size128_sse2_r4_stage2_next:
	INCQ BX
	JMP  size128_sse2_r4_stage2_outer

size128_sse2_r4_stage3:
	// ==================================================================
	// Stage 3: 2 groups x 16 butterflies, stride=64
	// ==================================================================
	XORQ BX, BX              // group index (0..1)

size128_sse2_r4_stage3_outer:
	CMPQ BX, $2
	JGE  size128_sse2_r4_stage4

	XORQ DX, DX              // butterfly index (0..15)

size128_sse2_r4_stage3_inner:
	CMPQ DX, $16
	JGE  size128_sse2_r4_stage3_next

	// Calculate indices
	MOVQ BX, SI
	SHLQ $6, SI              // SI = BX * 64
	ADDQ DX, SI              // base index

	MOVQ SI, DI
	ADDQ $16, DI             // idx1
	MOVQ SI, R14
	ADDQ $32, R14            // idx2
	MOVQ SI, R15
	ADDQ $48, R15            // idx3

	// Twiddle factors: twiddle[DX*2], twiddle[DX*4], twiddle[DX*6]
	MOVQ DX, CX
	SHLQ $1, CX
	MOVSD (R10)(CX*8), X8    // w1

	MOVQ DX, CX
	SHLQ $2, CX
	MOVSD (R10)(CX*8), X9    // w2

	MOVQ DX, CX
	IMULQ $6, CX
	MOVSD (R10)(CX*8), X10   // w3

	// Load data
	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1
	MOVSD (R8)(R14*8), X2
	MOVSD (R8)(R15*8), X3

	// Complex multiply a1*w1
	MOVAPS X8, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X8, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X1, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X1

	// Complex multiply a2*w2
	MOVAPS X9, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X9, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X2, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X2, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X2

	// Complex multiply a3*w3
	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X3, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X3, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X3

	// Radix-4 butterfly
	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	MOVAPS X7, X14
	SHUFPS $0xB1, X14, X14
	MOVUPS ·maskNegHiPS(SB), X15
	XORPS X15, X14

	MOVAPS X7, X12
	SHUFPS $0xB1, X12, X12
	MOVUPS ·maskNegLoPS(SB), X15
	XORPS X15, X12

	MOVAPS X4, X0
	ADDPS X6, X0
	MOVAPS X5, X1
	ADDPS X14, X1
	MOVAPS X4, X2
	SUBPS X6, X2
	MOVAPS X5, X3
	ADDPS X12, X3

	MOVSD X0, (R8)(SI*8)
	MOVSD X1, (R8)(DI*8)
	MOVSD X2, (R8)(R14*8)
	MOVSD X3, (R8)(R15*8)

	INCQ DX
	JMP  size128_sse2_r4_stage3_inner

size128_sse2_r4_stage3_next:
	INCQ BX
	JMP  size128_sse2_r4_stage3_outer

size128_sse2_r4_stage4:
	// ==================================================================
	// Stage 4: 32 radix-2 butterflies, stride=128
	// ==================================================================
	XORQ DX, DX              // butterfly index (0..63)

size128_sse2_r4_stage4_loop:
	CMPQ DX, $64
	JGE  size128_sse2_r4_done

	MOVQ DX, SI
	MOVQ DX, DI
	ADDQ $64, DI

	// Twiddle factor
	MOVSD (R10)(DX*8), X8

	// Load data
	MOVSD (R8)(SI*8), X0     // a
	MOVSD (R8)(DI*8), X1     // b

	// Complex multiply b*w
	MOVAPS X8, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X8, X12
	SHUFPS $0x55, X12, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X1, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X1

	// Radix-2 butterfly
	MOVAPS X0, X2
	ADDPS X1, X2             // y0 = a + b'
	MOVAPS X0, X3
	SUBPS X1, X3             // y1 = a - b'

	// Store
	MOVSD X2, (R8)(SI*8)
	MOVSD X3, (R8)(DI*8)

	INCQ DX
	JMP  size128_sse2_r4_stage4_loop

size128_sse2_r4_done:
	// Copy to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size128_sse2_r4_done_direct

	XORQ CX, CX

size128_sse2_r4_copy_loop:
	MOVUPS (R8)(CX*1), X0
	MOVUPS X0, (R9)(CX*1)
	ADDQ $16, CX
	CMPQ CX, $1024           // 128 * 8 bytes
	JL   size128_sse2_r4_copy_loop

size128_sse2_r4_done_direct:
	MOVB $1, ret+96(FP)
	RET

size128_sse2_r4_return_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 128, complex64, radix-4-then-2 (2×4) (SSE2)
// ===========================================================================
TEXT ·InverseSSE3Size128Radix4Then2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size128Radix4Then2(SB), R12
	MOVQ src+32(FP), R13

	// Verify n == 128
	CMPQ R13, $128
	JNE  size128_sse2_r4_inv_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $128
	JL   size128_sse2_r4_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $128
	JL   size128_sse2_r4_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $128
	JL   size128_sse2_r4_inv_return_false

	MOVQ $128, AX
	CMPQ AX, $128
	JL   size128_sse2_r4_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size128_sse2_r4_inv_use_dst
	MOVQ R11, R8

size128_sse2_r4_inv_use_dst:
	// Bit-reversal
	XORQ CX, CX

size128_sse2_r4_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $128
	JL   size128_sse2_r4_inv_bitrev_loop

size128_sse2_r4_inv_stage1:
	// ==================================================================
	// Stage 1: 32 radix-4 butterflies (inverse: swap i/-i)
	// ==================================================================
	XORQ CX, CX

size128_sse2_r4_inv_stage1_loop:
	CMPQ CX, $128
	JGE  size128_sse2_r4_inv_stage2

	LEAQ (R8)(CX*8), SI
	MOVSD (SI), X0
	MOVSD 8(SI), X1
	MOVSD 16(SI), X2
	MOVSD 24(SI), X3

	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	// i*t3 for y1
	MOVAPS X7, X11
	SHUFPS $0xB1, X11, X11
	MOVUPS ·maskNegLoPS(SB), X10
	XORPS X10, X11

	// (-i)*t3 for y3
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	MOVUPS ·maskNegHiPS(SB), X9
	XORPS X9, X8

	MOVAPS X4, X0
	ADDPS X6, X0
	MOVAPS X5, X1
	ADDPS X11, X1            // y1 = t1 + i*t3
	MOVAPS X4, X2
	SUBPS X6, X2
	MOVAPS X5, X3
	ADDPS X8, X3             // y3 = t1 + (-i)*t3

	MOVSD X0, (SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)

	ADDQ $4, CX
	JMP  size128_sse2_r4_inv_stage1_loop

size128_sse2_r4_inv_stage2:
	// ==================================================================
	// Stage 2: 8 groups x 4 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ BX, BX

size128_sse2_r4_inv_stage2_outer:
	CMPQ BX, $8
	JGE  size128_sse2_r4_inv_stage3

	XORQ DX, DX

size128_sse2_r4_inv_stage2_inner:
	CMPQ DX, $4
	JGE  size128_sse2_r4_inv_stage2_next

	MOVQ BX, SI
	SHLQ $4, SI
	ADDQ DX, SI

	MOVQ SI, DI
	ADDQ $4, DI
	MOVQ SI, R14
	ADDQ $8, R14
	MOVQ SI, R15
	ADDQ $12, R15

	MOVQ DX, CX
	SHLQ $3, CX
	MOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $4, CX
	MOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $24, CX
	MOVSD (R10)(CX*8), X10

	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1
	MOVSD (R8)(R14*8), X2
	MOVSD (R8)(R15*8), X3

	// Conjugate complex multiply (negate imag before multiply)
	MOVAPS X8, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X8, X12
	SHUFPS $0x55, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X1, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X1

	MOVAPS X9, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X9, X12
	SHUFPS $0x55, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X2, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X2, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X2

	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X3, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X3, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X3

	// Radix-4 butterfly (inverse)
	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	MOVAPS X7, X12
	SHUFPS $0xB1, X12, X12
	MOVUPS ·maskNegLoPS(SB), X15
	XORPS X15, X12

	MOVAPS X7, X14
	SHUFPS $0xB1, X14, X14
	MOVUPS ·maskNegHiPS(SB), X15
	XORPS X15, X14

	MOVAPS X4, X0
	ADDPS X6, X0
	MOVAPS X5, X1
	ADDPS X12, X1
	MOVAPS X4, X2
	SUBPS X6, X2
	MOVAPS X5, X3
	ADDPS X14, X3

	MOVSD X0, (R8)(SI*8)
	MOVSD X1, (R8)(DI*8)
	MOVSD X2, (R8)(R14*8)
	MOVSD X3, (R8)(R15*8)

	INCQ DX
	JMP  size128_sse2_r4_inv_stage2_inner

size128_sse2_r4_inv_stage2_next:
	INCQ BX
	JMP  size128_sse2_r4_inv_stage2_outer

size128_sse2_r4_inv_stage3:
	// ==================================================================
	// Stage 3: 2 groups x 16 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ BX, BX

size128_sse2_r4_inv_stage3_outer:
	CMPQ BX, $2
	JGE  size128_sse2_r4_inv_stage4

	XORQ DX, DX

size128_sse2_r4_inv_stage3_inner:
	CMPQ DX, $16
	JGE  size128_sse2_r4_inv_stage3_next

	MOVQ BX, SI
	SHLQ $6, SI
	ADDQ DX, SI

	MOVQ SI, DI
	ADDQ $16, DI
	MOVQ SI, R14
	ADDQ $32, R14
	MOVQ SI, R15
	ADDQ $48, R15

	MOVQ DX, CX
	SHLQ $1, CX
	MOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $2, CX
	MOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $6, CX
	MOVSD (R10)(CX*8), X10

	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1
	MOVSD (R8)(R14*8), X2
	MOVSD (R8)(R15*8), X3

	// Conjugate complex multiply
	MOVAPS X8, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X8, X12
	SHUFPS $0x55, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X1, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X1

	MOVAPS X9, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X9, X12
	SHUFPS $0x55, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X2, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X2, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X2

	MOVAPS X10, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X10, X12
	SHUFPS $0x55, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X3, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X3, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X3

	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	MOVAPS X7, X12
	SHUFPS $0xB1, X12, X12
	MOVUPS ·maskNegLoPS(SB), X15
	XORPS X15, X12

	MOVAPS X7, X14
	SHUFPS $0xB1, X14, X14
	MOVUPS ·maskNegHiPS(SB), X15
	XORPS X15, X14

	MOVAPS X4, X0
	ADDPS X6, X0
	MOVAPS X5, X1
	ADDPS X12, X1
	MOVAPS X4, X2
	SUBPS X6, X2
	MOVAPS X5, X3
	ADDPS X14, X3

	MOVSD X0, (R8)(SI*8)
	MOVSD X1, (R8)(DI*8)
	MOVSD X2, (R8)(R14*8)
	MOVSD X3, (R8)(R15*8)

	INCQ DX
	JMP  size128_sse2_r4_inv_stage3_inner

size128_sse2_r4_inv_stage3_next:
	INCQ BX
	JMP  size128_sse2_r4_inv_stage3_outer

size128_sse2_r4_inv_stage4:
	// ==================================================================
	// Stage 4: 32 radix-2 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ DX, DX

size128_sse2_r4_inv_stage4_loop:
	CMPQ DX, $64
	JGE  size128_sse2_r4_inv_scale

	MOVQ DX, SI
	MOVQ DX, DI
	ADDQ $64, DI

	MOVSD (R10)(DX*8), X8

	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1

	// Conjugate complex multiply
	MOVAPS X8, X11
	SHUFPS $0x00, X11, X11
	MOVAPS X8, X12
	SHUFPS $0x55, X12, X12
	XORPS X13, X13
	SUBPS X12, X13
	MOVAPS X13, X12
	MOVAPS X1, X13
	SHUFPS $0xB1, X13, X13
	MULPS X12, X13
	MOVAPS X1, X4
	MULPS X11, X4
	ADDSUBPS X13, X4
	MOVAPS X4, X1

	MOVAPS X0, X2
	ADDPS X1, X2
	MOVAPS X0, X3
	SUBPS X1, X3

	MOVSD X2, (R8)(SI*8)
	MOVSD X3, (R8)(DI*8)

	INCQ DX
	JMP  size128_sse2_r4_inv_stage4_loop

size128_sse2_r4_inv_scale:
	// ==================================================================
	// Apply 1/N scaling (1/128 = 0.0078125)
	// ==================================================================
	// Use 128-bit operations to process 2 complex64 at a time
	MOVSS $0.0078125, X15     // 1/128
	SHUFPS $0x00, X15, X15    // broadcast to all 4 floats
	XORQ CX, CX

size128_sse2_r4_inv_scale_loop:
	MOVUPS (R8)(CX*1), X0     // Load 2 complex64 (16 bytes)
	MULPS X15, X0
	MOVUPS X0, (R8)(CX*1)     // Store 2 complex64
	ADDQ $16, CX
	CMPQ CX, $1024            // 128 * 8 bytes = 1024
	JL   size128_sse2_r4_inv_scale_loop

	// Copy to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size128_sse2_r4_inv_done

	XORQ CX, CX

size128_sse2_r4_inv_copy_loop:
	MOVUPS (R8)(CX*1), X0
	MOVUPS X0, (R9)(CX*1)
	ADDQ $16, CX
	CMPQ CX, $1024
	JL   size128_sse2_r4_inv_copy_loop

size128_sse2_r4_inv_done:
	MOVB $1, ret+96(FP)
	RET

size128_sse2_r4_inv_return_false:
	MOVB $0, ret+96(FP)
	RET

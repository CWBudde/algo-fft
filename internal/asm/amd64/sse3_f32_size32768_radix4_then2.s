//go:build amd64 && !purego

// ===========================================================================
// SSE3 Size-32768 Radix-4-then-2 FFT Kernels for AMD64 (complex64)
// ===========================================================================
//
// Size 32768 = 4^7 x 2, implemented as 7 radix-4 stages plus a final
// radix-2 stage (mixed 2/4 decomposition, matching the AVX2 twin):
//   Stage 1: 8192 radix-4 butterflies, span=4, twiddle = 1
//   Stage 2: 2048 groups x 4 butterflies, span=16, twiddle step=2048
//   Stage 3: 512 groups x 16 butterflies, span=64, twiddle step=512
//   Stage 4: 128 groups x 64 butterflies, span=256, twiddle step=128
//   Stage 5: 32 groups x 256 butterflies, span=1024, twiddle step=32
//   Stage 6: 8 groups x 1024 butterflies, span=4096, twiddle step=8
//   Stage 7: 2 groups x 4096 butterflies, span=16384, twiddle step=2
//   Stage 8: 16384 radix-2 butterflies, span=32768, twiddle step=1
// Input permutation: passed in as a slice argument (bitrev []int) — at
// 32768 entries an embedded table would add 256 KiB to the binary; the
// caller passes internal/kernels' shared table (matching the AVX2 twin).
//
// SSE3 version: no FMA, uses ADDSUBPS for the complex multiply.
// ===========================================================================

#include "textflag.h"

// Forward transform, size 32768, complex64, radix-4-then-2 variant
TEXT ·ForwardSSE3Size32768Radix4Then2Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src_len+32(FP), R13

	// Verify n == 32768
	CMPQ R13, $32768
	JNE  size32768_r4t2_f32_return_false

	// Validate all slice lengths >= 32768
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f32_return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f32_return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f32_return_false

	MOVQ bitrev_len+104(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f32_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size32768_r4t2_f32_use_dst
	MOVQ R11, R8

size32768_r4t2_f32_use_dst:
	// ==================================================================
	// Mixed-digit-reversal permutation
	// ==================================================================
	XORQ CX, CX

size32768_r4t2_f32_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $32768
	JL   size32768_r4t2_f32_bitrev_loop

size32768_r4t2_f32_stage1:
	// ==================================================================
	// Stage 1: 8192 radix-4 butterflies, span=4, twiddle=1
	// ==================================================================
	XORQ CX, CX

size32768_r4t2_f32_stage1_loop:
	CMPQ CX, $32768
	JGE  size32768_r4t2_f32_stage2

	LEAQ (R8)(CX*8), SI
	MOVSD (SI), X0
	MOVSD 8(SI), X1
	MOVSD 16(SI), X2
	MOVSD 24(SI), X3

	// Radix-4 butterfly (twiddle = 1)
	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	// (-i)*t3 = (im, -re)
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	MOVUPS ·maskNegHiPS(SB), X9
	XORPS X9, X8

	// i*t3 = (-im, re)
	MOVAPS X7, X11
	SHUFPS $0xB1, X11, X11
	MOVUPS ·maskNegLoPS(SB), X9
	XORPS X9, X11

	// Final butterfly outputs
	MOVAPS X4, X0
	ADDPS X6, X0
	MOVAPS X5, X1
	ADDPS X8, X1
	MOVAPS X4, X2
	SUBPS X6, X2
	MOVAPS X5, X3
	ADDPS X11, X3

	MOVSD X0, (SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)

	ADDQ $4, CX
	JMP  size32768_r4t2_f32_stage1_loop

size32768_r4t2_f32_stage2:
	// ==================================================================
	// Stage 2: 2048 groups x 4 butterflies, span=16, twiddle step=2048
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f32_stage2_outer:
	CMPQ BX, $2048
	JGE  size32768_r4t2_f32_stage3

	XORQ DX, DX

size32768_r4t2_f32_stage2_inner:
	CMPQ DX, $4
	JGE  size32768_r4t2_f32_stage2_next

	// Calculate indices
	MOVQ BX, SI
	SHLQ $4, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ $4, DI
	MOVQ SI, R14
	ADDQ $8, R14
	MOVQ SI, R15
	ADDQ $12, R15

	// Twiddle factors: twiddle[DX*2048], twiddle[DX*4096], twiddle[DX*6144]
	MOVQ DX, CX
	SHLQ $11, CX
	MOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $12, CX
	MOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $6144, CX
	MOVSD (R10)(CX*8), X10

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
	JMP  size32768_r4t2_f32_stage2_inner

size32768_r4t2_f32_stage2_next:
	INCQ BX
	JMP  size32768_r4t2_f32_stage2_outer

size32768_r4t2_f32_stage3:
	// ==================================================================
	// Stage 3: 512 groups x 16 butterflies, span=64, twiddle step=512
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f32_stage3_outer:
	CMPQ BX, $512
	JGE  size32768_r4t2_f32_stage4

	XORQ DX, DX

size32768_r4t2_f32_stage3_inner:
	CMPQ DX, $16
	JGE  size32768_r4t2_f32_stage3_next

	// Calculate indices
	MOVQ BX, SI
	SHLQ $6, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ $16, DI
	MOVQ SI, R14
	ADDQ $32, R14
	MOVQ SI, R15
	ADDQ $48, R15

	// Twiddle factors: twiddle[DX*512], twiddle[DX*1024], twiddle[DX*1536]
	MOVQ DX, CX
	SHLQ $9, CX
	MOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $10, CX
	MOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $1536, CX
	MOVSD (R10)(CX*8), X10

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
	JMP  size32768_r4t2_f32_stage3_inner

size32768_r4t2_f32_stage3_next:
	INCQ BX
	JMP  size32768_r4t2_f32_stage3_outer

size32768_r4t2_f32_stage4:
	// ==================================================================
	// Stage 4: 128 groups x 64 butterflies, span=256, twiddle step=128
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f32_stage4_outer:
	CMPQ BX, $128
	JGE  size32768_r4t2_f32_stage5

	XORQ DX, DX

size32768_r4t2_f32_stage4_inner:
	CMPQ DX, $64
	JGE  size32768_r4t2_f32_stage4_next

	// Calculate indices
	MOVQ BX, SI
	SHLQ $8, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ $64, DI
	MOVQ SI, R14
	ADDQ $128, R14
	MOVQ SI, R15
	ADDQ $192, R15

	// Twiddle factors: twiddle[DX*128], twiddle[DX*256], twiddle[DX*384]
	MOVQ DX, CX
	SHLQ $7, CX
	MOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $8, CX
	MOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $384, CX
	MOVSD (R10)(CX*8), X10

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
	JMP  size32768_r4t2_f32_stage4_inner

size32768_r4t2_f32_stage4_next:
	INCQ BX
	JMP  size32768_r4t2_f32_stage4_outer

size32768_r4t2_f32_stage5:
	// ==================================================================
	// Stage 5: 32 groups x 256 butterflies, span=1024, twiddle step=32
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f32_stage5_outer:
	CMPQ BX, $32
	JGE  size32768_r4t2_f32_stage6

	XORQ DX, DX

size32768_r4t2_f32_stage5_inner:
	CMPQ DX, $256
	JGE  size32768_r4t2_f32_stage5_next

	// Calculate indices
	MOVQ BX, SI
	SHLQ $10, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ $256, DI
	MOVQ SI, R14
	ADDQ $512, R14
	MOVQ SI, R15
	ADDQ $768, R15

	// Twiddle factors: twiddle[DX*32], twiddle[DX*64], twiddle[DX*96]
	MOVQ DX, CX
	SHLQ $5, CX
	MOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $6, CX
	MOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $96, CX
	MOVSD (R10)(CX*8), X10

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
	JMP  size32768_r4t2_f32_stage5_inner

size32768_r4t2_f32_stage5_next:
	INCQ BX
	JMP  size32768_r4t2_f32_stage5_outer

size32768_r4t2_f32_stage6:
	// ==================================================================
	// Stage 6: 8 groups x 1024 butterflies, span=4096, twiddle step=8
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f32_stage6_outer:
	CMPQ BX, $8
	JGE  size32768_r4t2_f32_stage7

	XORQ DX, DX

size32768_r4t2_f32_stage6_inner:
	CMPQ DX, $1024
	JGE  size32768_r4t2_f32_stage6_next

	// Calculate indices
	MOVQ BX, SI
	SHLQ $12, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ $1024, DI
	MOVQ SI, R14
	ADDQ $2048, R14
	MOVQ SI, R15
	ADDQ $3072, R15

	// Twiddle factors: twiddle[DX*8], twiddle[DX*16], twiddle[DX*24]
	MOVQ DX, CX
	SHLQ $3, CX
	MOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $4, CX
	MOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $24, CX
	MOVSD (R10)(CX*8), X10

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
	JMP  size32768_r4t2_f32_stage6_inner

size32768_r4t2_f32_stage6_next:
	INCQ BX
	JMP  size32768_r4t2_f32_stage6_outer

size32768_r4t2_f32_stage7:
	// ==================================================================
	// Stage 7: 2 groups x 4096 butterflies, span=16384, twiddle step=2
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f32_stage7_outer:
	CMPQ BX, $2
	JGE  size32768_r4t2_f32_stage8

	XORQ DX, DX

size32768_r4t2_f32_stage7_inner:
	CMPQ DX, $4096
	JGE  size32768_r4t2_f32_stage7_next

	// Calculate indices
	MOVQ BX, SI
	SHLQ $14, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ $4096, DI
	MOVQ SI, R14
	ADDQ $8192, R14
	MOVQ SI, R15
	ADDQ $12288, R15

	// Twiddle factors: twiddle[DX*2], twiddle[DX*4], twiddle[DX*6]
	MOVQ DX, CX
	SHLQ $1, CX
	MOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $2, CX
	MOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $6, CX
	MOVSD (R10)(CX*8), X10

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
	JMP  size32768_r4t2_f32_stage7_inner

size32768_r4t2_f32_stage7_next:
	INCQ BX
	JMP  size32768_r4t2_f32_stage7_outer

size32768_r4t2_f32_stage8:
	// ==================================================================
	// Stage 8: 16384 radix-2 butterflies, span=32768, twiddle step=1
	// ==================================================================
	XORQ DX, DX

size32768_r4t2_f32_stage8_loop:
	CMPQ DX, $16384
	JGE  size32768_r4t2_f32_done

	MOVQ DX, SI
	MOVQ DX, DI
	ADDQ $16384, DI

	// Twiddle factor: twiddle[DX]
	MOVSD (R10)(DX*8), X8

	// Load data
	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1

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
	ADDPS X1, X2
	MOVAPS X0, X3
	SUBPS X1, X3

	MOVSD X2, (R8)(SI*8)
	MOVSD X3, (R8)(DI*8)

	INCQ DX
	JMP  size32768_r4t2_f32_stage8_loop

size32768_r4t2_f32_done:
	// Copy to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size32768_r4t2_f32_done_direct

	XORQ CX, CX

size32768_r4t2_f32_copy_loop:
	MOVUPS (R8)(CX*1), X0
	MOVUPS X0, (R9)(CX*1)
	ADDQ $16, CX
	CMPQ CX, $262144
	JL   size32768_r4t2_f32_copy_loop

size32768_r4t2_f32_done_direct:
	MOVB $1, ret+120(FP)
	RET

size32768_r4t2_f32_return_false:
	MOVB $0, ret+120(FP)
	RET

// Inverse transform, size 32768, complex64, radix-4-then-2 variant
TEXT ·InverseSSE3Size32768Radix4Then2Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src_len+32(FP), R13

	// Verify n == 32768
	CMPQ R13, $32768
	JNE  size32768_r4t2_f32_inv_return_false

	// Validate all slice lengths >= 32768
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f32_inv_return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f32_inv_return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f32_inv_return_false

	MOVQ bitrev_len+104(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f32_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size32768_r4t2_f32_inv_use_dst
	MOVQ R11, R8

size32768_r4t2_f32_inv_use_dst:
	// ==================================================================
	// Mixed-digit-reversal permutation
	// ==================================================================
	XORQ CX, CX

size32768_r4t2_f32_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $32768
	JL   size32768_r4t2_f32_inv_bitrev_loop

size32768_r4t2_f32_inv_stage1:
	// ==================================================================
	// Stage 1: 8192 radix-4 butterflies (inverse: swap i/-i)
	// ==================================================================
	XORQ CX, CX

size32768_r4t2_f32_inv_stage1_loop:
	CMPQ CX, $32768
	JGE  size32768_r4t2_f32_inv_stage2

	LEAQ (R8)(CX*8), SI
	MOVSD (SI), X0
	MOVSD 8(SI), X1
	MOVSD 16(SI), X2
	MOVSD 24(SI), X3

	// Radix-4 butterfly (twiddle = 1)
	MOVAPS X0, X4
	ADDPS X2, X4
	MOVAPS X0, X5
	SUBPS X2, X5
	MOVAPS X1, X6
	ADDPS X3, X6
	MOVAPS X1, X7
	SUBPS X3, X7

	// i*t3 = (-im, re)
	MOVAPS X7, X11
	SHUFPS $0xB1, X11, X11
	MOVUPS ·maskNegLoPS(SB), X10
	XORPS X10, X11

	// (-i)*t3 = (im, -re)
	MOVAPS X7, X8
	SHUFPS $0xB1, X8, X8
	MOVUPS ·maskNegHiPS(SB), X9
	XORPS X9, X8

	// Final butterfly outputs
	MOVAPS X4, X0
	ADDPS X6, X0
	MOVAPS X5, X1
	ADDPS X11, X1
	MOVAPS X4, X2
	SUBPS X6, X2
	MOVAPS X5, X3
	ADDPS X8, X3

	MOVSD X0, (SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 16(SI)
	MOVSD X3, 24(SI)

	ADDQ $4, CX
	JMP  size32768_r4t2_f32_inv_stage1_loop

size32768_r4t2_f32_inv_stage2:
	// ==================================================================
	// Stage 2: 2048 groups x 4 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f32_inv_stage2_outer:
	CMPQ BX, $2048
	JGE  size32768_r4t2_f32_inv_stage3

	XORQ DX, DX

size32768_r4t2_f32_inv_stage2_inner:
	CMPQ DX, $4
	JGE  size32768_r4t2_f32_inv_stage2_next

	// Calculate indices
	MOVQ BX, SI
	SHLQ $4, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ $4, DI
	MOVQ SI, R14
	ADDQ $8, R14
	MOVQ SI, R15
	ADDQ $12, R15

	// Twiddle factors: twiddle[DX*2048], twiddle[DX*4096], twiddle[DX*6144]
	MOVQ DX, CX
	SHLQ $11, CX
	MOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $12, CX
	MOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $6144, CX
	MOVSD (R10)(CX*8), X10

	// Load data
	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1
	MOVSD (R8)(R14*8), X2
	MOVSD (R8)(R15*8), X3

	// Conjugate complex multiply a1*conj(w1)
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

	// Conjugate complex multiply a2*conj(w2)
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

	// Conjugate complex multiply a3*conj(w3)
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
	JMP  size32768_r4t2_f32_inv_stage2_inner

size32768_r4t2_f32_inv_stage2_next:
	INCQ BX
	JMP  size32768_r4t2_f32_inv_stage2_outer

size32768_r4t2_f32_inv_stage3:
	// ==================================================================
	// Stage 3: 512 groups x 16 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f32_inv_stage3_outer:
	CMPQ BX, $512
	JGE  size32768_r4t2_f32_inv_stage4

	XORQ DX, DX

size32768_r4t2_f32_inv_stage3_inner:
	CMPQ DX, $16
	JGE  size32768_r4t2_f32_inv_stage3_next

	// Calculate indices
	MOVQ BX, SI
	SHLQ $6, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ $16, DI
	MOVQ SI, R14
	ADDQ $32, R14
	MOVQ SI, R15
	ADDQ $48, R15

	// Twiddle factors: twiddle[DX*512], twiddle[DX*1024], twiddle[DX*1536]
	MOVQ DX, CX
	SHLQ $9, CX
	MOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $10, CX
	MOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $1536, CX
	MOVSD (R10)(CX*8), X10

	// Load data
	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1
	MOVSD (R8)(R14*8), X2
	MOVSD (R8)(R15*8), X3

	// Conjugate complex multiply a1*conj(w1)
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

	// Conjugate complex multiply a2*conj(w2)
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

	// Conjugate complex multiply a3*conj(w3)
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
	JMP  size32768_r4t2_f32_inv_stage3_inner

size32768_r4t2_f32_inv_stage3_next:
	INCQ BX
	JMP  size32768_r4t2_f32_inv_stage3_outer

size32768_r4t2_f32_inv_stage4:
	// ==================================================================
	// Stage 4: 128 groups x 64 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f32_inv_stage4_outer:
	CMPQ BX, $128
	JGE  size32768_r4t2_f32_inv_stage5

	XORQ DX, DX

size32768_r4t2_f32_inv_stage4_inner:
	CMPQ DX, $64
	JGE  size32768_r4t2_f32_inv_stage4_next

	// Calculate indices
	MOVQ BX, SI
	SHLQ $8, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ $64, DI
	MOVQ SI, R14
	ADDQ $128, R14
	MOVQ SI, R15
	ADDQ $192, R15

	// Twiddle factors: twiddle[DX*128], twiddle[DX*256], twiddle[DX*384]
	MOVQ DX, CX
	SHLQ $7, CX
	MOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $8, CX
	MOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $384, CX
	MOVSD (R10)(CX*8), X10

	// Load data
	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1
	MOVSD (R8)(R14*8), X2
	MOVSD (R8)(R15*8), X3

	// Conjugate complex multiply a1*conj(w1)
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

	// Conjugate complex multiply a2*conj(w2)
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

	// Conjugate complex multiply a3*conj(w3)
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
	JMP  size32768_r4t2_f32_inv_stage4_inner

size32768_r4t2_f32_inv_stage4_next:
	INCQ BX
	JMP  size32768_r4t2_f32_inv_stage4_outer

size32768_r4t2_f32_inv_stage5:
	// ==================================================================
	// Stage 5: 32 groups x 256 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f32_inv_stage5_outer:
	CMPQ BX, $32
	JGE  size32768_r4t2_f32_inv_stage6

	XORQ DX, DX

size32768_r4t2_f32_inv_stage5_inner:
	CMPQ DX, $256
	JGE  size32768_r4t2_f32_inv_stage5_next

	// Calculate indices
	MOVQ BX, SI
	SHLQ $10, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ $256, DI
	MOVQ SI, R14
	ADDQ $512, R14
	MOVQ SI, R15
	ADDQ $768, R15

	// Twiddle factors: twiddle[DX*32], twiddle[DX*64], twiddle[DX*96]
	MOVQ DX, CX
	SHLQ $5, CX
	MOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $6, CX
	MOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $96, CX
	MOVSD (R10)(CX*8), X10

	// Load data
	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1
	MOVSD (R8)(R14*8), X2
	MOVSD (R8)(R15*8), X3

	// Conjugate complex multiply a1*conj(w1)
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

	// Conjugate complex multiply a2*conj(w2)
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

	// Conjugate complex multiply a3*conj(w3)
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
	JMP  size32768_r4t2_f32_inv_stage5_inner

size32768_r4t2_f32_inv_stage5_next:
	INCQ BX
	JMP  size32768_r4t2_f32_inv_stage5_outer

size32768_r4t2_f32_inv_stage6:
	// ==================================================================
	// Stage 6: 8 groups x 1024 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f32_inv_stage6_outer:
	CMPQ BX, $8
	JGE  size32768_r4t2_f32_inv_stage7

	XORQ DX, DX

size32768_r4t2_f32_inv_stage6_inner:
	CMPQ DX, $1024
	JGE  size32768_r4t2_f32_inv_stage6_next

	// Calculate indices
	MOVQ BX, SI
	SHLQ $12, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ $1024, DI
	MOVQ SI, R14
	ADDQ $2048, R14
	MOVQ SI, R15
	ADDQ $3072, R15

	// Twiddle factors: twiddle[DX*8], twiddle[DX*16], twiddle[DX*24]
	MOVQ DX, CX
	SHLQ $3, CX
	MOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $4, CX
	MOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $24, CX
	MOVSD (R10)(CX*8), X10

	// Load data
	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1
	MOVSD (R8)(R14*8), X2
	MOVSD (R8)(R15*8), X3

	// Conjugate complex multiply a1*conj(w1)
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

	// Conjugate complex multiply a2*conj(w2)
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

	// Conjugate complex multiply a3*conj(w3)
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
	JMP  size32768_r4t2_f32_inv_stage6_inner

size32768_r4t2_f32_inv_stage6_next:
	INCQ BX
	JMP  size32768_r4t2_f32_inv_stage6_outer

size32768_r4t2_f32_inv_stage7:
	// ==================================================================
	// Stage 7: 2 groups x 4096 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f32_inv_stage7_outer:
	CMPQ BX, $2
	JGE  size32768_r4t2_f32_inv_stage8

	XORQ DX, DX

size32768_r4t2_f32_inv_stage7_inner:
	CMPQ DX, $4096
	JGE  size32768_r4t2_f32_inv_stage7_next

	// Calculate indices
	MOVQ BX, SI
	SHLQ $14, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ $4096, DI
	MOVQ SI, R14
	ADDQ $8192, R14
	MOVQ SI, R15
	ADDQ $12288, R15

	// Twiddle factors: twiddle[DX*2], twiddle[DX*4], twiddle[DX*6]
	MOVQ DX, CX
	SHLQ $1, CX
	MOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $2, CX
	MOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $6, CX
	MOVSD (R10)(CX*8), X10

	// Load data
	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1
	MOVSD (R8)(R14*8), X2
	MOVSD (R8)(R15*8), X3

	// Conjugate complex multiply a1*conj(w1)
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

	// Conjugate complex multiply a2*conj(w2)
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

	// Conjugate complex multiply a3*conj(w3)
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
	JMP  size32768_r4t2_f32_inv_stage7_inner

size32768_r4t2_f32_inv_stage7_next:
	INCQ BX
	JMP  size32768_r4t2_f32_inv_stage7_outer

size32768_r4t2_f32_inv_stage8:
	// ==================================================================
	// Stage 8: 16384 radix-2 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ DX, DX

size32768_r4t2_f32_inv_stage8_loop:
	CMPQ DX, $16384
	JGE  size32768_r4t2_f32_inv_scale

	MOVQ DX, SI
	MOVQ DX, DI
	ADDQ $16384, DI

	// Twiddle factor: twiddle[DX]
	MOVSD (R10)(DX*8), X8

	// Load data
	MOVSD (R8)(SI*8), X0
	MOVSD (R8)(DI*8), X1

	// Conjugate complex multiply b*conj(w)
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

	// Radix-2 butterfly
	MOVAPS X0, X2
	ADDPS X1, X2
	MOVAPS X0, X3
	SUBPS X1, X3

	MOVSD X2, (R8)(SI*8)
	MOVSD X3, (R8)(DI*8)

	INCQ DX
	JMP  size32768_r4t2_f32_inv_stage8_loop

size32768_r4t2_f32_inv_scale:
	// Scale by 1/32768
	MOVSS ·oneThirtyTwoThousandSevenHundredSixtyEighth32(SB), X15
	SHUFPS $0x00, X15, X15
	XORQ CX, CX

size32768_r4t2_f32_inv_scale_loop:
	MOVUPS (R8)(CX*1), X0
	MULPS X15, X0
	MOVUPS X0, (R8)(CX*1)
	ADDQ $16, CX
	CMPQ CX, $262144
	JL   size32768_r4t2_f32_inv_scale_loop

	// Copy to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size32768_r4t2_f32_inv_done

	XORQ CX, CX

size32768_r4t2_f32_inv_copy_loop:
	MOVUPS (R8)(CX*1), X0
	MOVUPS X0, (R9)(CX*1)
	ADDQ $16, CX
	CMPQ CX, $262144
	JL   size32768_r4t2_f32_inv_copy_loop

size32768_r4t2_f32_inv_done:
	MOVB $1, ret+120(FP)
	RET

size32768_r4t2_f32_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

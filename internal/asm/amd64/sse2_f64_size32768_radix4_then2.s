//go:build amd64 && !purego

// ===========================================================================
// SSE2 Size-32768 Radix-4-then-2 FFT Kernels for AMD64 (complex128)
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
// ===========================================================================

#include "textflag.h"

// Forward transform, size 32768, complex128, radix-4-then-2 variant
TEXT ·ForwardSSE2Size32768Radix4Then2Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src_len+32(FP), R13

	// Verify n == 32768
	CMPQ R13, $32768
	JNE  size32768_r4t2_f64_return_false

	// Validate all slice lengths >= 32768
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f64_return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f64_return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f64_return_false

	MOVQ bitrev_len+104(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f64_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size32768_r4t2_f64_use_dst
	MOVQ R11, R8

size32768_r4t2_f64_use_dst:
	// ==================================================================
	// Mixed-digit-reversal permutation
	// ==================================================================
	XORQ CX, CX

size32768_r4t2_f64_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X0
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	INCQ CX
	CMPQ CX, $32768
	JL   size32768_r4t2_f64_bitrev_loop

size32768_r4t2_f64_stage1:
	// ==================================================================
	// Stage 1: 8192 radix-4 butterflies, span=4
	// No twiddle factors needed (all 1)
	// ==================================================================
	MOVQ R8, SI
	MOVQ $8192, CX
	MOVUPS ·maskNegHiPD(SB), X15

size32768_r4t2_f64_stage1_loop:
	MOVUPD (SI), X0
	MOVUPD 16(SI), X1
	MOVUPD 32(SI), X2
	MOVUPD 48(SI), X3

	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// (-i)*t3 = (im, -re)
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD X15, X8

	// Final butterfly outputs
	MOVAPD X4, X0
	ADDPD X6, X0
	MOVAPD X5, X1
	ADDPD X8, X1
	MOVAPD X4, X2
	SUBPD X6, X2
	MOVAPD X5, X3
	SUBPD X8, X3

	MOVUPD X0, (SI)
	MOVUPD X1, 16(SI)
	MOVUPD X2, 32(SI)
	MOVUPD X3, 48(SI)

	ADDQ $64, SI
	DECQ CX
	JNZ  size32768_r4t2_f64_stage1_loop

size32768_r4t2_f64_stage2:
	// ==================================================================
	// Stage 2: 2048 groups x 4 butterflies, span=16, twiddle step=2048
	// ==================================================================
	MOVUPS ·maskNegLoPD(SB), X14
	MOVUPS ·maskNegHiPD(SB), X15
	XORQ BX, BX

size32768_r4t2_f64_stage2_outer:
	CMPQ BX, $2048
	JGE  size32768_r4t2_f64_stage3

	XORQ DX, DX

size32768_r4t2_f64_stage2_loop:
	CMPQ DX, $4
	JGE  size32768_r4t2_f64_stage2_next_group

	// Twiddle factors: twiddle[DX*2048], twiddle[DX*4096], twiddle[DX*6144]
	MOVQ DX, AX
	IMULQ $32768, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ DX, AX
	IMULQ $65536, AX
	MOVUPD (R10)(AX*1), X9

	MOVQ DX, AX
	IMULQ $98304, AX
	MOVUPD (R10)(AX*1), X10

	// Load data
	MOVQ BX, R13
	IMULQ $256, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 64(SI), DI
	LEAQ 128(SI), R14
	LEAQ 192(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R14), X2
	MOVUPD (R15), X3

	// Complex multiply a1*w1
	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X1

	// Complex multiply a2*w2
	MOVAPD X2, X4
	UNPCKLPD X4, X4
	MULPD X9, X4
	MOVAPD X2, X5
	UNPCKHPD X5, X5
	MOVAPD X9, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X2

	// Complex multiply a3*w3
	MOVAPD X3, X4
	UNPCKLPD X4, X4
	MULPD X10, X4
	MOVAPD X3, X5
	UNPCKHPD X5, X5
	MOVAPD X10, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X3

	// Radix-4 butterfly
	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// (-i)*t3
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD X15, X8

	MOVAPD X4, X0
	ADDPD X6, X0
	MOVAPD X5, X1
	ADDPD X8, X1
	MOVAPD X4, X2
	SUBPD X6, X2
	MOVAPD X5, X3
	SUBPD X8, X3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R14)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  size32768_r4t2_f64_stage2_loop

size32768_r4t2_f64_stage2_next_group:
	INCQ BX
	JMP  size32768_r4t2_f64_stage2_outer

size32768_r4t2_f64_stage3:
	// ==================================================================
	// Stage 3: 512 groups x 16 butterflies, span=64, twiddle step=512
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f64_stage3_outer:
	CMPQ BX, $512
	JGE  size32768_r4t2_f64_stage4

	XORQ DX, DX

size32768_r4t2_f64_stage3_loop:
	CMPQ DX, $16
	JGE  size32768_r4t2_f64_stage3_next_group

	// Twiddle factors: twiddle[DX*512], twiddle[DX*1024], twiddle[DX*1536]
	MOVQ DX, AX
	IMULQ $8192, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ DX, AX
	IMULQ $16384, AX
	MOVUPD (R10)(AX*1), X9

	MOVQ DX, AX
	IMULQ $24576, AX
	MOVUPD (R10)(AX*1), X10

	// Load data
	MOVQ BX, R13
	IMULQ $1024, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 256(SI), DI
	LEAQ 512(SI), R14
	LEAQ 768(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R14), X2
	MOVUPD (R15), X3

	// Complex multiply a1*w1
	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X1

	// Complex multiply a2*w2
	MOVAPD X2, X4
	UNPCKLPD X4, X4
	MULPD X9, X4
	MOVAPD X2, X5
	UNPCKHPD X5, X5
	MOVAPD X9, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X2

	// Complex multiply a3*w3
	MOVAPD X3, X4
	UNPCKLPD X4, X4
	MULPD X10, X4
	MOVAPD X3, X5
	UNPCKHPD X5, X5
	MOVAPD X10, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X3

	// Radix-4 butterfly
	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// (-i)*t3
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD X15, X8

	MOVAPD X4, X0
	ADDPD X6, X0
	MOVAPD X5, X1
	ADDPD X8, X1
	MOVAPD X4, X2
	SUBPD X6, X2
	MOVAPD X5, X3
	SUBPD X8, X3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R14)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  size32768_r4t2_f64_stage3_loop

size32768_r4t2_f64_stage3_next_group:
	INCQ BX
	JMP  size32768_r4t2_f64_stage3_outer

size32768_r4t2_f64_stage4:
	// ==================================================================
	// Stage 4: 128 groups x 64 butterflies, span=256, twiddle step=128
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f64_stage4_outer:
	CMPQ BX, $128
	JGE  size32768_r4t2_f64_stage5

	XORQ DX, DX

size32768_r4t2_f64_stage4_loop:
	CMPQ DX, $64
	JGE  size32768_r4t2_f64_stage4_next_group

	// Twiddle factors: twiddle[DX*128], twiddle[DX*256], twiddle[DX*384]
	MOVQ DX, AX
	IMULQ $2048, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ DX, AX
	IMULQ $4096, AX
	MOVUPD (R10)(AX*1), X9

	MOVQ DX, AX
	IMULQ $6144, AX
	MOVUPD (R10)(AX*1), X10

	// Load data
	MOVQ BX, R13
	IMULQ $4096, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 1024(SI), DI
	LEAQ 2048(SI), R14
	LEAQ 3072(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R14), X2
	MOVUPD (R15), X3

	// Complex multiply a1*w1
	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X1

	// Complex multiply a2*w2
	MOVAPD X2, X4
	UNPCKLPD X4, X4
	MULPD X9, X4
	MOVAPD X2, X5
	UNPCKHPD X5, X5
	MOVAPD X9, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X2

	// Complex multiply a3*w3
	MOVAPD X3, X4
	UNPCKLPD X4, X4
	MULPD X10, X4
	MOVAPD X3, X5
	UNPCKHPD X5, X5
	MOVAPD X10, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X3

	// Radix-4 butterfly
	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// (-i)*t3
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD X15, X8

	MOVAPD X4, X0
	ADDPD X6, X0
	MOVAPD X5, X1
	ADDPD X8, X1
	MOVAPD X4, X2
	SUBPD X6, X2
	MOVAPD X5, X3
	SUBPD X8, X3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R14)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  size32768_r4t2_f64_stage4_loop

size32768_r4t2_f64_stage4_next_group:
	INCQ BX
	JMP  size32768_r4t2_f64_stage4_outer

size32768_r4t2_f64_stage5:
	// ==================================================================
	// Stage 5: 32 groups x 256 butterflies, span=1024, twiddle step=32
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f64_stage5_outer:
	CMPQ BX, $32
	JGE  size32768_r4t2_f64_stage6

	XORQ DX, DX

size32768_r4t2_f64_stage5_loop:
	CMPQ DX, $256
	JGE  size32768_r4t2_f64_stage5_next_group

	// Twiddle factors: twiddle[DX*32], twiddle[DX*64], twiddle[DX*96]
	MOVQ DX, AX
	IMULQ $512, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ DX, AX
	IMULQ $1024, AX
	MOVUPD (R10)(AX*1), X9

	MOVQ DX, AX
	IMULQ $1536, AX
	MOVUPD (R10)(AX*1), X10

	// Load data
	MOVQ BX, R13
	IMULQ $16384, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 4096(SI), DI
	LEAQ 8192(SI), R14
	LEAQ 12288(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R14), X2
	MOVUPD (R15), X3

	// Complex multiply a1*w1
	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X1

	// Complex multiply a2*w2
	MOVAPD X2, X4
	UNPCKLPD X4, X4
	MULPD X9, X4
	MOVAPD X2, X5
	UNPCKHPD X5, X5
	MOVAPD X9, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X2

	// Complex multiply a3*w3
	MOVAPD X3, X4
	UNPCKLPD X4, X4
	MULPD X10, X4
	MOVAPD X3, X5
	UNPCKHPD X5, X5
	MOVAPD X10, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X3

	// Radix-4 butterfly
	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// (-i)*t3
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD X15, X8

	MOVAPD X4, X0
	ADDPD X6, X0
	MOVAPD X5, X1
	ADDPD X8, X1
	MOVAPD X4, X2
	SUBPD X6, X2
	MOVAPD X5, X3
	SUBPD X8, X3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R14)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  size32768_r4t2_f64_stage5_loop

size32768_r4t2_f64_stage5_next_group:
	INCQ BX
	JMP  size32768_r4t2_f64_stage5_outer

size32768_r4t2_f64_stage6:
	// ==================================================================
	// Stage 6: 8 groups x 1024 butterflies, span=4096, twiddle step=8
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f64_stage6_outer:
	CMPQ BX, $8
	JGE  size32768_r4t2_f64_stage7

	XORQ DX, DX

size32768_r4t2_f64_stage6_loop:
	CMPQ DX, $1024
	JGE  size32768_r4t2_f64_stage6_next_group

	// Twiddle factors: twiddle[DX*8], twiddle[DX*16], twiddle[DX*24]
	MOVQ DX, AX
	IMULQ $128, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ DX, AX
	IMULQ $256, AX
	MOVUPD (R10)(AX*1), X9

	MOVQ DX, AX
	IMULQ $384, AX
	MOVUPD (R10)(AX*1), X10

	// Load data
	MOVQ BX, R13
	IMULQ $65536, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 16384(SI), DI
	LEAQ 32768(SI), R14
	LEAQ 49152(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R14), X2
	MOVUPD (R15), X3

	// Complex multiply a1*w1
	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X1

	// Complex multiply a2*w2
	MOVAPD X2, X4
	UNPCKLPD X4, X4
	MULPD X9, X4
	MOVAPD X2, X5
	UNPCKHPD X5, X5
	MOVAPD X9, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X2

	// Complex multiply a3*w3
	MOVAPD X3, X4
	UNPCKLPD X4, X4
	MULPD X10, X4
	MOVAPD X3, X5
	UNPCKHPD X5, X5
	MOVAPD X10, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X3

	// Radix-4 butterfly
	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// (-i)*t3
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD X15, X8

	MOVAPD X4, X0
	ADDPD X6, X0
	MOVAPD X5, X1
	ADDPD X8, X1
	MOVAPD X4, X2
	SUBPD X6, X2
	MOVAPD X5, X3
	SUBPD X8, X3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R14)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  size32768_r4t2_f64_stage6_loop

size32768_r4t2_f64_stage6_next_group:
	INCQ BX
	JMP  size32768_r4t2_f64_stage6_outer

size32768_r4t2_f64_stage7:
	// ==================================================================
	// Stage 7: 2 groups x 4096 butterflies, span=16384, twiddle step=2
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f64_stage7_outer:
	CMPQ BX, $2
	JGE  size32768_r4t2_f64_stage8

	XORQ DX, DX

size32768_r4t2_f64_stage7_loop:
	CMPQ DX, $4096
	JGE  size32768_r4t2_f64_stage7_next_group

	// Twiddle factors: twiddle[DX*2], twiddle[DX*4], twiddle[DX*6]
	MOVQ DX, AX
	IMULQ $32, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ DX, AX
	IMULQ $64, AX
	MOVUPD (R10)(AX*1), X9

	MOVQ DX, AX
	IMULQ $96, AX
	MOVUPD (R10)(AX*1), X10

	// Load data
	MOVQ BX, R13
	IMULQ $262144, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 65536(SI), DI
	LEAQ 131072(SI), R14
	LEAQ 196608(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R14), X2
	MOVUPD (R15), X3

	// Complex multiply a1*w1
	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X1

	// Complex multiply a2*w2
	MOVAPD X2, X4
	UNPCKLPD X4, X4
	MULPD X9, X4
	MOVAPD X2, X5
	UNPCKHPD X5, X5
	MOVAPD X9, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X2

	// Complex multiply a3*w3
	MOVAPD X3, X4
	UNPCKLPD X4, X4
	MULPD X10, X4
	MOVAPD X3, X5
	UNPCKHPD X5, X5
	MOVAPD X10, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X3

	// Radix-4 butterfly
	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// (-i)*t3
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD X15, X8

	MOVAPD X4, X0
	ADDPD X6, X0
	MOVAPD X5, X1
	ADDPD X8, X1
	MOVAPD X4, X2
	SUBPD X6, X2
	MOVAPD X5, X3
	SUBPD X8, X3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R14)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  size32768_r4t2_f64_stage7_loop

size32768_r4t2_f64_stage7_next_group:
	INCQ BX
	JMP  size32768_r4t2_f64_stage7_outer

size32768_r4t2_f64_stage8:
	// ==================================================================
	// Stage 8: 16384 radix-2 butterflies, span=32768, twiddle step=1
	// ==================================================================
	XORQ DX, DX

size32768_r4t2_f64_stage8_loop:
	CMPQ DX, $16384
	JGE  size32768_r4t2_f64_done

	// Twiddle factor: twiddle[DX]
	MOVQ DX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	// Load data
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R8)(SI*1), SI
	MOVUPD (SI), X0
	MOVUPD 262144(SI), X1

	// Complex multiply b*w
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2
	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X14, X4
	ADDPD X4, X2

	// Radix-2 butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 262144(SI)

	INCQ DX
	JMP  size32768_r4t2_f64_stage8_loop

size32768_r4t2_f64_done:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size32768_r4t2_f64_done_direct

	XORQ CX, CX

size32768_r4t2_f64_copy_loop:
	MOVUPD (R8)(CX*1), X0
	MOVUPD X0, (R9)(CX*1)
	ADDQ $16, CX
	CMPQ CX, $524288
	JL   size32768_r4t2_f64_copy_loop

size32768_r4t2_f64_done_direct:
	MOVB $1, ret+120(FP)
	RET

size32768_r4t2_f64_return_false:
	MOVB $0, ret+120(FP)
	RET

// Inverse transform, size 32768, complex128, radix-4-then-2 variant
TEXT ·InverseSSE2Size32768Radix4Then2Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src_len+32(FP), R13

	// Verify n == 32768
	CMPQ R13, $32768
	JNE  size32768_r4t2_f64_inv_return_false

	// Validate all slice lengths >= 32768
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f64_inv_return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f64_inv_return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f64_inv_return_false

	MOVQ bitrev_len+104(FP), AX
	CMPQ AX, $32768
	JL   size32768_r4t2_f64_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size32768_r4t2_f64_inv_use_dst
	MOVQ R11, R8

size32768_r4t2_f64_inv_use_dst:
	// ==================================================================
	// Mixed-digit-reversal permutation
	// ==================================================================
	XORQ CX, CX

size32768_r4t2_f64_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X0
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	INCQ CX
	CMPQ CX, $32768
	JL   size32768_r4t2_f64_inv_bitrev_loop

size32768_r4t2_f64_inv_stage1:
	// ==================================================================
	// Stage 1: 8192 radix-4 butterflies, span=4 (inverse uses +i)
	// ==================================================================
	MOVQ R8, SI
	MOVQ $8192, CX
	MOVUPS ·maskNegLoPD(SB), X14

size32768_r4t2_f64_inv_stage1_loop:
	MOVUPD (SI), X0
	MOVUPD 16(SI), X1
	MOVUPD 32(SI), X2
	MOVUPD 48(SI), X3

	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// i*t3 = (-im, re)
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD X14, X8

	MOVAPD X4, X0
	ADDPD X6, X0
	MOVAPD X5, X1
	ADDPD X8, X1
	MOVAPD X4, X2
	SUBPD X6, X2
	MOVAPD X5, X3
	SUBPD X8, X3

	MOVUPD X0, (SI)
	MOVUPD X1, 16(SI)
	MOVUPD X2, 32(SI)
	MOVUPD X3, 48(SI)

	ADDQ $64, SI
	DECQ CX
	JNZ  size32768_r4t2_f64_inv_stage1_loop

size32768_r4t2_f64_inv_stage2:
	// ==================================================================
	// Stage 2: 2048 groups x 4 butterflies, span=16, twiddle step=2048
	// ==================================================================
	MOVUPS ·maskNegLoPD(SB), X14
	MOVUPS ·maskNegHiPD(SB), X15
	XORQ BX, BX

size32768_r4t2_f64_inv_stage2_outer:
	CMPQ BX, $2048
	JGE  size32768_r4t2_f64_inv_stage3

	XORQ DX, DX

size32768_r4t2_f64_inv_stage2_loop:
	CMPQ DX, $4
	JGE  size32768_r4t2_f64_inv_stage2_next_group

	// Twiddle factors (conjugated)
	MOVQ DX, AX
	IMULQ $32768, AX
	MOVUPD (R10)(AX*1), X8
	XORPD X15, X8

	MOVQ DX, AX
	IMULQ $65536, AX
	MOVUPD (R10)(AX*1), X9
	XORPD X15, X9

	MOVQ DX, AX
	IMULQ $98304, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X15, X10

	// Load data
	MOVQ BX, R13
	IMULQ $256, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 64(SI), DI
	LEAQ 128(SI), R12
	LEAQ 192(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R12), X2
	MOVUPD (R15), X3

	// Complex multiply a1*w1
	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X1

	// Complex multiply a2*w2
	MOVAPD X2, X4
	UNPCKLPD X4, X4
	MULPD X9, X4
	MOVAPD X2, X5
	UNPCKHPD X5, X5
	MOVAPD X9, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X2

	// Complex multiply a3*w3
	MOVAPD X3, X4
	UNPCKLPD X4, X4
	MULPD X10, X4
	MOVAPD X3, X5
	UNPCKHPD X5, X5
	MOVAPD X10, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X3

	// Radix-4 butterfly (inverse)
	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// i*t3
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD X14, X8

	MOVAPD X4, X0
	ADDPD X6, X0
	MOVAPD X5, X1
	ADDPD X8, X1
	MOVAPD X4, X2
	SUBPD X6, X2
	MOVAPD X5, X3
	SUBPD X8, X3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R12)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  size32768_r4t2_f64_inv_stage2_loop

size32768_r4t2_f64_inv_stage2_next_group:
	INCQ BX
	JMP  size32768_r4t2_f64_inv_stage2_outer

size32768_r4t2_f64_inv_stage3:
	// ==================================================================
	// Stage 3: 512 groups x 16 butterflies, span=64, twiddle step=512
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f64_inv_stage3_outer:
	CMPQ BX, $512
	JGE  size32768_r4t2_f64_inv_stage4

	XORQ DX, DX

size32768_r4t2_f64_inv_stage3_loop:
	CMPQ DX, $16
	JGE  size32768_r4t2_f64_inv_stage3_next_group

	// Twiddle factors (conjugated)
	MOVQ DX, AX
	IMULQ $8192, AX
	MOVUPD (R10)(AX*1), X8
	XORPD X15, X8

	MOVQ DX, AX
	IMULQ $16384, AX
	MOVUPD (R10)(AX*1), X9
	XORPD X15, X9

	MOVQ DX, AX
	IMULQ $24576, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X15, X10

	// Load data
	MOVQ BX, R13
	IMULQ $1024, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 256(SI), DI
	LEAQ 512(SI), R12
	LEAQ 768(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R12), X2
	MOVUPD (R15), X3

	// Complex multiply a1*w1
	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X1

	// Complex multiply a2*w2
	MOVAPD X2, X4
	UNPCKLPD X4, X4
	MULPD X9, X4
	MOVAPD X2, X5
	UNPCKHPD X5, X5
	MOVAPD X9, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X2

	// Complex multiply a3*w3
	MOVAPD X3, X4
	UNPCKLPD X4, X4
	MULPD X10, X4
	MOVAPD X3, X5
	UNPCKHPD X5, X5
	MOVAPD X10, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X3

	// Radix-4 butterfly (inverse)
	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// i*t3
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD X14, X8

	MOVAPD X4, X0
	ADDPD X6, X0
	MOVAPD X5, X1
	ADDPD X8, X1
	MOVAPD X4, X2
	SUBPD X6, X2
	MOVAPD X5, X3
	SUBPD X8, X3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R12)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  size32768_r4t2_f64_inv_stage3_loop

size32768_r4t2_f64_inv_stage3_next_group:
	INCQ BX
	JMP  size32768_r4t2_f64_inv_stage3_outer

size32768_r4t2_f64_inv_stage4:
	// ==================================================================
	// Stage 4: 128 groups x 64 butterflies, span=256, twiddle step=128
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f64_inv_stage4_outer:
	CMPQ BX, $128
	JGE  size32768_r4t2_f64_inv_stage5

	XORQ DX, DX

size32768_r4t2_f64_inv_stage4_loop:
	CMPQ DX, $64
	JGE  size32768_r4t2_f64_inv_stage4_next_group

	// Twiddle factors (conjugated)
	MOVQ DX, AX
	IMULQ $2048, AX
	MOVUPD (R10)(AX*1), X8
	XORPD X15, X8

	MOVQ DX, AX
	IMULQ $4096, AX
	MOVUPD (R10)(AX*1), X9
	XORPD X15, X9

	MOVQ DX, AX
	IMULQ $6144, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X15, X10

	// Load data
	MOVQ BX, R13
	IMULQ $4096, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 1024(SI), DI
	LEAQ 2048(SI), R12
	LEAQ 3072(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R12), X2
	MOVUPD (R15), X3

	// Complex multiply a1*w1
	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X1

	// Complex multiply a2*w2
	MOVAPD X2, X4
	UNPCKLPD X4, X4
	MULPD X9, X4
	MOVAPD X2, X5
	UNPCKHPD X5, X5
	MOVAPD X9, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X2

	// Complex multiply a3*w3
	MOVAPD X3, X4
	UNPCKLPD X4, X4
	MULPD X10, X4
	MOVAPD X3, X5
	UNPCKHPD X5, X5
	MOVAPD X10, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X3

	// Radix-4 butterfly (inverse)
	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// i*t3
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD X14, X8

	MOVAPD X4, X0
	ADDPD X6, X0
	MOVAPD X5, X1
	ADDPD X8, X1
	MOVAPD X4, X2
	SUBPD X6, X2
	MOVAPD X5, X3
	SUBPD X8, X3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R12)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  size32768_r4t2_f64_inv_stage4_loop

size32768_r4t2_f64_inv_stage4_next_group:
	INCQ BX
	JMP  size32768_r4t2_f64_inv_stage4_outer

size32768_r4t2_f64_inv_stage5:
	// ==================================================================
	// Stage 5: 32 groups x 256 butterflies, span=1024, twiddle step=32
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f64_inv_stage5_outer:
	CMPQ BX, $32
	JGE  size32768_r4t2_f64_inv_stage6

	XORQ DX, DX

size32768_r4t2_f64_inv_stage5_loop:
	CMPQ DX, $256
	JGE  size32768_r4t2_f64_inv_stage5_next_group

	// Twiddle factors (conjugated)
	MOVQ DX, AX
	IMULQ $512, AX
	MOVUPD (R10)(AX*1), X8
	XORPD X15, X8

	MOVQ DX, AX
	IMULQ $1024, AX
	MOVUPD (R10)(AX*1), X9
	XORPD X15, X9

	MOVQ DX, AX
	IMULQ $1536, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X15, X10

	// Load data
	MOVQ BX, R13
	IMULQ $16384, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 4096(SI), DI
	LEAQ 8192(SI), R12
	LEAQ 12288(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R12), X2
	MOVUPD (R15), X3

	// Complex multiply a1*w1
	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X1

	// Complex multiply a2*w2
	MOVAPD X2, X4
	UNPCKLPD X4, X4
	MULPD X9, X4
	MOVAPD X2, X5
	UNPCKHPD X5, X5
	MOVAPD X9, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X2

	// Complex multiply a3*w3
	MOVAPD X3, X4
	UNPCKLPD X4, X4
	MULPD X10, X4
	MOVAPD X3, X5
	UNPCKHPD X5, X5
	MOVAPD X10, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X3

	// Radix-4 butterfly (inverse)
	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// i*t3
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD X14, X8

	MOVAPD X4, X0
	ADDPD X6, X0
	MOVAPD X5, X1
	ADDPD X8, X1
	MOVAPD X4, X2
	SUBPD X6, X2
	MOVAPD X5, X3
	SUBPD X8, X3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R12)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  size32768_r4t2_f64_inv_stage5_loop

size32768_r4t2_f64_inv_stage5_next_group:
	INCQ BX
	JMP  size32768_r4t2_f64_inv_stage5_outer

size32768_r4t2_f64_inv_stage6:
	// ==================================================================
	// Stage 6: 8 groups x 1024 butterflies, span=4096, twiddle step=8
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f64_inv_stage6_outer:
	CMPQ BX, $8
	JGE  size32768_r4t2_f64_inv_stage7

	XORQ DX, DX

size32768_r4t2_f64_inv_stage6_loop:
	CMPQ DX, $1024
	JGE  size32768_r4t2_f64_inv_stage6_next_group

	// Twiddle factors (conjugated)
	MOVQ DX, AX
	IMULQ $128, AX
	MOVUPD (R10)(AX*1), X8
	XORPD X15, X8

	MOVQ DX, AX
	IMULQ $256, AX
	MOVUPD (R10)(AX*1), X9
	XORPD X15, X9

	MOVQ DX, AX
	IMULQ $384, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X15, X10

	// Load data
	MOVQ BX, R13
	IMULQ $65536, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 16384(SI), DI
	LEAQ 32768(SI), R12
	LEAQ 49152(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R12), X2
	MOVUPD (R15), X3

	// Complex multiply a1*w1
	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X1

	// Complex multiply a2*w2
	MOVAPD X2, X4
	UNPCKLPD X4, X4
	MULPD X9, X4
	MOVAPD X2, X5
	UNPCKHPD X5, X5
	MOVAPD X9, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X2

	// Complex multiply a3*w3
	MOVAPD X3, X4
	UNPCKLPD X4, X4
	MULPD X10, X4
	MOVAPD X3, X5
	UNPCKHPD X5, X5
	MOVAPD X10, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X3

	// Radix-4 butterfly (inverse)
	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// i*t3
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD X14, X8

	MOVAPD X4, X0
	ADDPD X6, X0
	MOVAPD X5, X1
	ADDPD X8, X1
	MOVAPD X4, X2
	SUBPD X6, X2
	MOVAPD X5, X3
	SUBPD X8, X3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R12)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  size32768_r4t2_f64_inv_stage6_loop

size32768_r4t2_f64_inv_stage6_next_group:
	INCQ BX
	JMP  size32768_r4t2_f64_inv_stage6_outer

size32768_r4t2_f64_inv_stage7:
	// ==================================================================
	// Stage 7: 2 groups x 4096 butterflies, span=16384, twiddle step=2
	// ==================================================================
	XORQ BX, BX

size32768_r4t2_f64_inv_stage7_outer:
	CMPQ BX, $2
	JGE  size32768_r4t2_f64_inv_stage8

	XORQ DX, DX

size32768_r4t2_f64_inv_stage7_loop:
	CMPQ DX, $4096
	JGE  size32768_r4t2_f64_inv_stage7_next_group

	// Twiddle factors (conjugated)
	MOVQ DX, AX
	IMULQ $32, AX
	MOVUPD (R10)(AX*1), X8
	XORPD X15, X8

	MOVQ DX, AX
	IMULQ $64, AX
	MOVUPD (R10)(AX*1), X9
	XORPD X15, X9

	MOVQ DX, AX
	IMULQ $96, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X15, X10

	// Load data
	MOVQ BX, R13
	IMULQ $262144, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 65536(SI), DI
	LEAQ 131072(SI), R12
	LEAQ 196608(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R12), X2
	MOVUPD (R15), X3

	// Complex multiply a1*w1
	MOVAPD X1, X4
	UNPCKLPD X4, X4
	MULPD X8, X4
	MOVAPD X1, X5
	UNPCKHPD X5, X5
	MOVAPD X8, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X1

	// Complex multiply a2*w2
	MOVAPD X2, X4
	UNPCKLPD X4, X4
	MULPD X9, X4
	MOVAPD X2, X5
	UNPCKHPD X5, X5
	MOVAPD X9, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X2

	// Complex multiply a3*w3
	MOVAPD X3, X4
	UNPCKLPD X4, X4
	MULPD X10, X4
	MOVAPD X3, X5
	UNPCKHPD X5, X5
	MOVAPD X10, X6
	SHUFPD $1, X6, X6
	MULPD X5, X6
	XORPD X14, X6
	ADDPD X6, X4
	MOVAPD X4, X3

	// Radix-4 butterfly (inverse)
	MOVAPD X0, X4
	ADDPD X2, X4
	MOVAPD X0, X5
	SUBPD X2, X5
	MOVAPD X1, X6
	ADDPD X3, X6
	MOVAPD X1, X7
	SUBPD X3, X7

	// i*t3
	MOVAPD X7, X8
	SHUFPD $1, X8, X8
	XORPD X14, X8

	MOVAPD X4, X0
	ADDPD X6, X0
	MOVAPD X5, X1
	ADDPD X8, X1
	MOVAPD X4, X2
	SUBPD X6, X2
	MOVAPD X5, X3
	SUBPD X8, X3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R12)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  size32768_r4t2_f64_inv_stage7_loop

size32768_r4t2_f64_inv_stage7_next_group:
	INCQ BX
	JMP  size32768_r4t2_f64_inv_stage7_outer

size32768_r4t2_f64_inv_stage8:
	// ==================================================================
	// Stage 8: 16384 radix-2 butterflies, span=32768 (conjugated twiddles)
	// ==================================================================
	XORQ DX, DX

size32768_r4t2_f64_inv_stage8_loop:
	CMPQ DX, $16384
	JGE  size32768_r4t2_f64_inv_scale

	// Twiddle factor (conjugated): twiddle[DX]
	MOVQ DX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X15, X10

	// Load data
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R8)(SI*1), SI
	MOVUPD (SI), X0
	MOVUPD 262144(SI), X1

	// Complex multiply b*w
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2
	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X14, X4
	ADDPD X4, X2

	// Radix-2 butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 262144(SI)

	INCQ DX
	JMP  size32768_r4t2_f64_inv_stage8_loop

size32768_r4t2_f64_inv_scale:
	// Scale by 1/32768 and copy to dst
	MOVSD ·oneThirtyTwoThousandSevenHundredSixtyEighth64(SB), X15
	SHUFPD $0, X15, X15
	MOVQ $32768, CX
	MOVQ R8, SI
	MOVQ R14, DI

size32768_r4t2_f64_inv_scale_copy:
	MOVUPD (SI), X0
	MULPD X15, X0
	MOVUPD X0, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	DECQ CX
	JNZ  size32768_r4t2_f64_inv_scale_copy

	MOVB $1, ret+120(FP)
	RET

size32768_r4t2_f64_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

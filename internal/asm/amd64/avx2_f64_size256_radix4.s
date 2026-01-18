//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-256 Radix-4 FFT Kernels for AMD64 (complex128)
// ===========================================================================
//
// Size 256 = 4^4, so the radix-4 algorithm uses 4 stages:
//   Stage 1: 64 butterflies, stride=4, twiddle = 1
//   Stage 2: 16 groups x 4 butterflies, stride=16, twiddle step=16
//   Stage 3: 4 groups x 16 butterflies, stride=64, twiddle step=4
//   Stage 4: 1 group x 64 butterflies, twiddle step=1
// ===========================================================================

#include "textflag.h"

// Forward transform, size 256, complex128, radix-4 variant
TEXT ·ForwardAVX2Size256Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size256Radix4(SB), R12
	MOVQ src+32(FP), R13

	// Verify n == 256
	CMPQ R13, $256
	JNE  size256_r4_f64_return_false

	// Validate all slice lengths >= 256
	MOVQ dst+8(FP), AX
	CMPQ AX, $256
	JL   size256_r4_f64_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $256
	JL   size256_r4_f64_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $256
	JL   size256_r4_f64_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size256_r4_f64_use_dst
	MOVQ R11, R8

size256_r4_f64_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 bit-reversal)
	// ==================================================================
	XORQ CX, CX

size256_r4_f64_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	SHLQ $4, DX
	VMOVUPD (R9)(DX*1), X0
	MOVQ CX, AX
	SHLQ $4, AX
	VMOVUPD X0, (R8)(AX*1)
	INCQ CX
	CMPQ CX, $256
	JL   size256_r4_f64_bitrev_loop

size256_r4_f64_stage1:
	// ==================================================================
	// Stage 1: 64 radix-4 butterflies, stride=4
	// No twiddle factors needed (all 1)
	// ==================================================================
	MOVQ R8, SI
	MOVQ $64, CX
	VMOVUPS ·maskNegHiPD(SB), X15

size256_r4_f64_stage1_loop:
	VMOVUPD (SI), X0
	VMOVUPD 16(SI), X1
	VMOVUPD 32(SI), X2
	VMOVUPD 48(SI), X3

	VMOVAPD X0, X4
	VADDPD X2, X4, X4
	VMOVAPD X0, X5
	VSUBPD X2, X5, X5
	VMOVAPD X1, X6
	VADDPD X3, X6, X6
	VMOVAPD X1, X7
	VSUBPD X3, X7, X7

	// (-i)*t3 = (im, -re)
	VMOVAPD X7, X8
	VSHUFPD $1, X8, X8, X8
	VXORPD X15, X8, X8

	// Final butterfly outputs
	VMOVAPD X4, X0
	VADDPD X6, X0, X0
	VMOVAPD X5, X1
	VADDPD X8, X1, X1
	VMOVAPD X4, X2
	VSUBPD X6, X2, X2
	VMOVAPD X5, X3
	VSUBPD X8, X3, X3

	VMOVUPD X0, (SI)
	VMOVUPD X1, 16(SI)
	VMOVUPD X2, 32(SI)
	VMOVUPD X3, 48(SI)

	ADDQ $64, SI
	DECQ CX
	JNZ  size256_r4_f64_stage1_loop

size256_r4_f64_stage2:
	// ==================================================================
	// Stage 2: 16 groups x 4 butterflies, stride=16, twiddle step=16
	// ==================================================================
	VMOVUPS ·maskNegLoPD(SB), X14
	VMOVUPS ·maskNegHiPD(SB), X15
	XORQ BX, BX

size256_r4_f64_stage2_outer:
	CMPQ BX, $16
	JGE  size256_r4_f64_stage3

	XORQ DX, DX

size256_r4_f64_stage2_loop:
	CMPQ DX, $4
	JGE  size256_r4_f64_stage2_next_group

	// Twiddle factors: twiddle[DX*16], twiddle[DX*32], twiddle[DX*48]
	MOVQ DX, AX
	SHLQ $4, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X8

	MOVQ DX, AX
	SHLQ $5, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X9

	MOVQ DX, AX
	IMULQ $48, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X10

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

	VMOVUPD (SI), X0
	VMOVUPD (DI), X1
	VMOVUPD (R14), X2
	VMOVUPD (R15), X3

	// Complex multiply a1*w1
	VMOVAPD X1, X4
	UNPCKLPD X4, X4
	VMULPD X8, X4, X4
	VMOVAPD X1, X5
	UNPCKHPD X5, X5
	VMOVAPD X8, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X1

	// Complex multiply a2*w2
	VMOVAPD X2, X4
	UNPCKLPD X4, X4
	VMULPD X9, X4, X4
	VMOVAPD X2, X5
	UNPCKHPD X5, X5
	VMOVAPD X9, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X2

	// Complex multiply a3*w3
	VMOVAPD X3, X4
	UNPCKLPD X4, X4
	VMULPD X10, X4, X4
	VMOVAPD X3, X5
	UNPCKHPD X5, X5
	VMOVAPD X10, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X3

	// Radix-4 butterfly
	VMOVAPD X0, X4
	VADDPD X2, X4, X4
	VMOVAPD X0, X5
	VSUBPD X2, X5, X5
	VMOVAPD X1, X6
	VADDPD X3, X6, X6
	VMOVAPD X1, X7
	VSUBPD X3, X7, X7

	// (-i)*t3
	VMOVAPD X7, X8
	VSHUFPD $1, X8, X8, X8
	VXORPD X15, X8, X8

	VMOVAPD X4, X0
	VADDPD X6, X0, X0
	VMOVAPD X5, X1
	VADDPD X8, X1, X1
	VMOVAPD X4, X2
	VSUBPD X6, X2, X2
	VMOVAPD X5, X3
	VSUBPD X8, X3, X3

	VMOVUPD X0, (SI)
	VMOVUPD X1, (DI)
	VMOVUPD X2, (R14)
	VMOVUPD X3, (R15)

	INCQ DX
	JMP  size256_r4_f64_stage2_loop

size256_r4_f64_stage2_next_group:
	INCQ BX
	JMP  size256_r4_f64_stage2_outer

size256_r4_f64_stage3:
	// ==================================================================
	// Stage 3: 4 groups x 16 butterflies, stride=64, twiddle step=4
	// ==================================================================
	XORQ BX, BX

size256_r4_f64_stage3_outer:
	CMPQ BX, $4
	JGE  size256_r4_f64_stage4

	XORQ DX, DX

size256_r4_f64_stage3_loop:
	CMPQ DX, $16
	JGE  size256_r4_f64_stage3_next_group

	// Twiddle factors: twiddle[DX*4], twiddle[DX*8], twiddle[DX*12]
	MOVQ DX, AX
	SHLQ $2, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X8

	MOVQ DX, AX
	SHLQ $3, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X9

	MOVQ DX, AX
	IMULQ $12, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X10

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

	VMOVUPD (SI), X0
	VMOVUPD (DI), X1
	VMOVUPD (R14), X2
	VMOVUPD (R15), X3

	// Complex multiply a1*w1
	VMOVAPD X1, X4
	UNPCKLPD X4, X4
	VMULPD X8, X4, X4
	VMOVAPD X1, X5
	UNPCKHPD X5, X5
	VMOVAPD X8, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X1

	// Complex multiply a2*w2
	VMOVAPD X2, X4
	UNPCKLPD X4, X4
	VMULPD X9, X4, X4
	VMOVAPD X2, X5
	UNPCKHPD X5, X5
	VMOVAPD X9, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X2

	// Complex multiply a3*w3
	VMOVAPD X3, X4
	UNPCKLPD X4, X4
	VMULPD X10, X4, X4
	VMOVAPD X3, X5
	UNPCKHPD X5, X5
	VMOVAPD X10, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X3

	// Radix-4 butterfly
	VMOVAPD X0, X4
	VADDPD X2, X4, X4
	VMOVAPD X0, X5
	VSUBPD X2, X5, X5
	VMOVAPD X1, X6
	VADDPD X3, X6, X6
	VMOVAPD X1, X7
	VSUBPD X3, X7, X7

	// (-i)*t3
	VMOVAPD X7, X8
	VSHUFPD $1, X8, X8, X8
	VXORPD X15, X8, X8

	VMOVAPD X4, X0
	VADDPD X6, X0, X0
	VMOVAPD X5, X1
	VADDPD X8, X1, X1
	VMOVAPD X4, X2
	VSUBPD X6, X2, X2
	VMOVAPD X5, X3
	VSUBPD X8, X3, X3

	VMOVUPD X0, (SI)
	VMOVUPD X1, (DI)
	VMOVUPD X2, (R14)
	VMOVUPD X3, (R15)

	INCQ DX
	JMP  size256_r4_f64_stage3_loop

size256_r4_f64_stage3_next_group:
	INCQ BX
	JMP  size256_r4_f64_stage3_outer

size256_r4_f64_stage4:
	// ==================================================================
	// Stage 4: 1 group x 64 butterflies, twiddle step=1
	// ==================================================================
	XORQ DX, DX

size256_r4_f64_stage4_loop:
	CMPQ DX, $64
	JGE  size256_r4_f64_done

	// Twiddle factors: twiddle[DX], twiddle[2*DX], twiddle[3*DX]
	MOVQ DX, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X8

	MOVQ DX, AX
	SHLQ $1, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X9

	MOVQ DX, AX
	IMULQ $3, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X10

	// Load data
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R8)(SI*1), SI
	LEAQ 1024(SI), DI
	LEAQ 2048(SI), R14
	LEAQ 3072(SI), R15

	VMOVUPD (SI), X0
	VMOVUPD (DI), X1
	VMOVUPD (R14), X2
	VMOVUPD (R15), X3

	// Complex multiply a1*w1
	VMOVAPD X1, X4
	UNPCKLPD X4, X4
	VMULPD X8, X4, X4
	VMOVAPD X1, X5
	UNPCKHPD X5, X5
	VMOVAPD X8, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X1

	// Complex multiply a2*w2
	VMOVAPD X2, X4
	UNPCKLPD X4, X4
	VMULPD X9, X4, X4
	VMOVAPD X2, X5
	UNPCKHPD X5, X5
	VMOVAPD X9, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X2

	// Complex multiply a3*w3
	VMOVAPD X3, X4
	UNPCKLPD X4, X4
	VMULPD X10, X4, X4
	VMOVAPD X3, X5
	UNPCKHPD X5, X5
	VMOVAPD X10, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X3

	// Radix-4 butterfly
	VMOVAPD X0, X4
	VADDPD X2, X4, X4
	VMOVAPD X0, X5
	VSUBPD X2, X5, X5
	VMOVAPD X1, X6
	VADDPD X3, X6, X6
	VMOVAPD X1, X7
	VSUBPD X3, X7, X7

	// (-i)*t3
	VMOVAPD X7, X8
	VSHUFPD $1, X8, X8, X8
	VXORPD X15, X8, X8

	VMOVAPD X4, X0
	VADDPD X6, X0, X0
	VMOVAPD X5, X1
	VADDPD X8, X1, X1
	VMOVAPD X4, X2
	VSUBPD X6, X2, X2
	VMOVAPD X5, X3
	VSUBPD X8, X3, X3

	VMOVUPD X0, (SI)
	VMOVUPD X1, (DI)
	VMOVUPD X2, (R14)
	VMOVUPD X3, (R15)

	INCQ DX
	JMP  size256_r4_f64_stage4_loop

size256_r4_f64_done:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size256_r4_f64_done_direct

	XORQ CX, CX

size256_r4_f64_copy_loop:
	VMOVUPD (R8)(CX*1), X0
	VMOVUPD X0, (R9)(CX*1)
	ADDQ $16, CX
	CMPQ CX, $4096
	JL   size256_r4_f64_copy_loop

size256_r4_f64_done_direct:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size256_r4_f64_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 256, complex128, radix-4 variant
TEXT ·InverseAVX2Size256Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size256Radix4(SB), R12
	MOVQ src+32(FP), R13

	// Verify n == 256
	CMPQ R13, $256
	JNE  size256_r4_f64_inv_return_false

	// Validate all slice lengths >= 256
	MOVQ dst+8(FP), AX
	CMPQ AX, $256
	JL   size256_r4_f64_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $256
	JL   size256_r4_f64_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $256
	JL   size256_r4_f64_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size256_r4_f64_inv_use_dst
	MOVQ R11, R8

size256_r4_f64_inv_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 bit-reversal)
	// ==================================================================
	XORQ CX, CX

size256_r4_f64_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	SHLQ $4, DX
	VMOVUPD (R9)(DX*1), X0
	MOVQ CX, AX
	SHLQ $4, AX
	VMOVUPD X0, (R8)(AX*1)
	INCQ CX
	CMPQ CX, $256
	JL   size256_r4_f64_inv_bitrev_loop

size256_r4_f64_inv_stage1:
	// ==================================================================
	// Stage 1: 64 radix-4 butterflies, stride=4 (inverse uses +i)
	// ==================================================================
	MOVQ R8, SI
	MOVQ $64, CX
	VMOVUPS ·maskNegLoPD(SB), X14

size256_r4_f64_inv_stage1_loop:
	VMOVUPD (SI), X0
	VMOVUPD 16(SI), X1
	VMOVUPD 32(SI), X2
	VMOVUPD 48(SI), X3

	VMOVAPD X0, X4
	VADDPD X2, X4, X4
	VMOVAPD X0, X5
	VSUBPD X2, X5, X5
	VMOVAPD X1, X6
	VADDPD X3, X6, X6
	VMOVAPD X1, X7
	VSUBPD X3, X7, X7

	// i*t3 = (-im, re)
	VMOVAPD X7, X8
	VSHUFPD $1, X8, X8, X8
	VXORPD X14, X8, X8

	VMOVAPD X4, X0
	VADDPD X6, X0, X0
	VMOVAPD X5, X1
	VADDPD X8, X1, X1
	VMOVAPD X4, X2
	VSUBPD X6, X2, X2
	VMOVAPD X5, X3
	VSUBPD X8, X3, X3

	VMOVUPD X0, (SI)
	VMOVUPD X1, 16(SI)
	VMOVUPD X2, 32(SI)
	VMOVUPD X3, 48(SI)

	ADDQ $64, SI
	DECQ CX
	JNZ  size256_r4_f64_inv_stage1_loop

size256_r4_f64_inv_stage2:
	// ==================================================================
	// Stage 2: 16 groups x 4 butterflies, stride=16, twiddle step=16
	// ==================================================================
	VMOVUPS ·maskNegLoPD(SB), X14
	VMOVUPS ·maskNegHiPD(SB), X15
	XORQ BX, BX

size256_r4_f64_inv_stage2_outer:
	CMPQ BX, $16
	JGE  size256_r4_f64_inv_stage3

	XORQ DX, DX

size256_r4_f64_inv_stage2_loop:
	CMPQ DX, $4
	JGE  size256_r4_f64_inv_stage2_next_group

	// Twiddle factors (conjugated)
	MOVQ DX, AX
	SHLQ $4, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X8
	VXORPD X15, X8, X8

	MOVQ DX, AX
	SHLQ $5, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X9
	VXORPD X15, X9, X9

	MOVQ DX, AX
	IMULQ $48, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X10
	VXORPD X15, X10, X10

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

	VMOVUPD (SI), X0
	VMOVUPD (DI), X1
	VMOVUPD (R12), X2
	VMOVUPD (R15), X3

	// Complex multiply a1*w1
	VMOVAPD X1, X4
	UNPCKLPD X4, X4
	VMULPD X8, X4, X4
	VMOVAPD X1, X5
	UNPCKHPD X5, X5
	VMOVAPD X8, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X1

	// Complex multiply a2*w2
	VMOVAPD X2, X4
	UNPCKLPD X4, X4
	VMULPD X9, X4, X4
	VMOVAPD X2, X5
	UNPCKHPD X5, X5
	VMOVAPD X9, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X2

	// Complex multiply a3*w3
	VMOVAPD X3, X4
	UNPCKLPD X4, X4
	VMULPD X10, X4, X4
	VMOVAPD X3, X5
	UNPCKHPD X5, X5
	VMOVAPD X10, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X3

	// Radix-4 butterfly (inverse)
	VMOVAPD X0, X4
	VADDPD X2, X4, X4
	VMOVAPD X0, X5
	VSUBPD X2, X5, X5
	VMOVAPD X1, X6
	VADDPD X3, X6, X6
	VMOVAPD X1, X7
	VSUBPD X3, X7, X7

	// i*t3
	VMOVAPD X7, X8
	VSHUFPD $1, X8, X8, X8
	VXORPD X14, X8, X8

	VMOVAPD X4, X0
	VADDPD X6, X0, X0
	VMOVAPD X5, X1
	VADDPD X8, X1, X1
	VMOVAPD X4, X2
	VSUBPD X6, X2, X2
	VMOVAPD X5, X3
	VSUBPD X8, X3, X3

	VMOVUPD X0, (SI)
	VMOVUPD X1, (DI)
	VMOVUPD X2, (R12)
	VMOVUPD X3, (R15)

	INCQ DX
	JMP  size256_r4_f64_inv_stage2_loop

size256_r4_f64_inv_stage2_next_group:
	INCQ BX
	JMP  size256_r4_f64_inv_stage2_outer

size256_r4_f64_inv_stage3:
	// ==================================================================
	// Stage 3: 4 groups x 16 butterflies, stride=64, twiddle step=4
	// ==================================================================
	XORQ BX, BX

size256_r4_f64_inv_stage3_outer:
	CMPQ BX, $4
	JGE  size256_r4_f64_inv_stage4

	XORQ DX, DX

size256_r4_f64_inv_stage3_loop:
	CMPQ DX, $16
	JGE  size256_r4_f64_inv_stage3_next_group

	// Twiddle factors (conjugated)
	MOVQ DX, AX
	SHLQ $2, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X8
	VXORPD X15, X8, X8

	MOVQ DX, AX
	SHLQ $3, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X9
	VXORPD X15, X9, X9

	MOVQ DX, AX
	IMULQ $12, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X10
	VXORPD X15, X10, X10

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

	VMOVUPD (SI), X0
	VMOVUPD (DI), X1
	VMOVUPD (R12), X2
	VMOVUPD (R15), X3

	// Complex multiply a1*w1
	VMOVAPD X1, X4
	UNPCKLPD X4, X4
	VMULPD X8, X4, X4
	VMOVAPD X1, X5
	UNPCKHPD X5, X5
	VMOVAPD X8, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X1

	// Complex multiply a2*w2
	VMOVAPD X2, X4
	UNPCKLPD X4, X4
	VMULPD X9, X4, X4
	VMOVAPD X2, X5
	UNPCKHPD X5, X5
	VMOVAPD X9, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X2

	// Complex multiply a3*w3
	VMOVAPD X3, X4
	UNPCKLPD X4, X4
	VMULPD X10, X4, X4
	VMOVAPD X3, X5
	UNPCKHPD X5, X5
	VMOVAPD X10, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X3

	// Radix-4 butterfly (inverse)
	VMOVAPD X0, X4
	VADDPD X2, X4, X4
	VMOVAPD X0, X5
	VSUBPD X2, X5, X5
	VMOVAPD X1, X6
	VADDPD X3, X6, X6
	VMOVAPD X1, X7
	VSUBPD X3, X7, X7

	// i*t3
	VMOVAPD X7, X8
	VSHUFPD $1, X8, X8, X8
	VXORPD X14, X8, X8

	VMOVAPD X4, X0
	VADDPD X6, X0, X0
	VMOVAPD X5, X1
	VADDPD X8, X1, X1
	VMOVAPD X4, X2
	VSUBPD X6, X2, X2
	VMOVAPD X5, X3
	VSUBPD X8, X3, X3

	VMOVUPD X0, (SI)
	VMOVUPD X1, (DI)
	VMOVUPD X2, (R12)
	VMOVUPD X3, (R15)

	INCQ DX
	JMP  size256_r4_f64_inv_stage3_loop

size256_r4_f64_inv_stage3_next_group:
	INCQ BX
	JMP  size256_r4_f64_inv_stage3_outer

size256_r4_f64_inv_stage4:
	// ==================================================================
	// Stage 4: 1 group x 64 butterflies, twiddle step=1
	// ==================================================================
	XORQ DX, DX

size256_r4_f64_inv_stage4_loop:
	CMPQ DX, $64
	JGE  size256_r4_f64_inv_scale

	// Twiddle factors (conjugated)
	MOVQ DX, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X8
	VXORPD X15, X8, X8

	MOVQ DX, AX
	SHLQ $1, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X9
	VXORPD X15, X9, X9

	MOVQ DX, AX
	IMULQ $3, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X10
	VXORPD X15, X10, X10

	// Load data
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R8)(SI*1), SI
	LEAQ 1024(SI), DI
	LEAQ 2048(SI), R12
	LEAQ 3072(SI), R15

	VMOVUPD (SI), X0
	VMOVUPD (DI), X1
	VMOVUPD (R12), X2
	VMOVUPD (R15), X3

	// Complex multiply a1*w1
	VMOVAPD X1, X4
	UNPCKLPD X4, X4
	VMULPD X8, X4, X4
	VMOVAPD X1, X5
	UNPCKHPD X5, X5
	VMOVAPD X8, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X1

	// Complex multiply a2*w2
	VMOVAPD X2, X4
	UNPCKLPD X4, X4
	VMULPD X9, X4, X4
	VMOVAPD X2, X5
	UNPCKHPD X5, X5
	VMOVAPD X9, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X2

	// Complex multiply a3*w3
	VMOVAPD X3, X4
	UNPCKLPD X4, X4
	VMULPD X10, X4, X4
	VMOVAPD X3, X5
	UNPCKHPD X5, X5
	VMOVAPD X10, X6
	VSHUFPD $1, X6, X6, X6
	VMULPD X5, X6, X6
	VXORPD X14, X6, X6
	VADDPD X6, X4, X4
	VMOVAPD X4, X3

	// Radix-4 butterfly (inverse)
	VMOVAPD X0, X4
	VADDPD X2, X4, X4
	VMOVAPD X0, X5
	VSUBPD X2, X5, X5
	VMOVAPD X1, X6
	VADDPD X3, X6, X6
	VMOVAPD X1, X7
	VSUBPD X3, X7, X7

	// i*t3
	VMOVAPD X7, X8
	VSHUFPD $1, X8, X8, X8
	VXORPD X14, X8, X8

	VMOVAPD X4, X0
	VADDPD X6, X0, X0
	VMOVAPD X5, X1
	VADDPD X8, X1, X1
	VMOVAPD X4, X2
	VSUBPD X6, X2, X2
	VMOVAPD X5, X3
	VSUBPD X8, X3, X3

	VMOVUPD X0, (SI)
	VMOVUPD X1, (DI)
	VMOVUPD X2, (R12)
	VMOVUPD X3, (R15)

	INCQ DX
	JMP  size256_r4_f64_inv_stage4_loop

size256_r4_f64_inv_scale:
	// Scale by 1/256 and copy to dst
	MOVSD ·twoFiftySixth64(SB), X15
	VSHUFPD $0, X15, X15, X15
	MOVQ $256, CX
	MOVQ R8, SI
	MOVQ R14, DI

size256_r4_f64_inv_scale_copy:
	VMOVUPD (SI), X0
	VMULPD X15, X0, X0
	VMOVUPD X0, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	DECQ CX
	JNZ  size256_r4_f64_inv_scale_copy

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size256_r4_f64_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

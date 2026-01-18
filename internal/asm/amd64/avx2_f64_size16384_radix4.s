
//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-16384 Radix-4 FFT Kernel for AMD64 (complex128)
// ===========================================================================
//
// Algorithm: Radix-4 Decimation-in-Time (DIT) FFT
// Stages: 7 (log₄(16384) = 7)
//
// Stage structure:
//   Stage 1: 4096 groups × 1 butterfly, stride=4,     no twiddle
//   Stage 2: 1024 groups × 4 butterflies, stride=16,    twiddle step=1024
//   Stage 3: 256 groups  × 16 butterflies, stride=64,   twiddle step=256
//   Stage 4: 64 groups   × 64 butterflies, stride=256,  twiddle step=64
//   Stage 5: 16 groups   × 256 butterflies, stride=1024, twiddle step=16
//   Stage 6: 4 groups    × 1024 butterflies, stride=4096, twiddle step=4
//   Stage 7: 1 group     × 4096 butterflies, stride=16384, twiddle step=1
//
// ===========================================================================

#include "textflag.h"

TEXT ·ForwardAVX2Size16384Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 16384)
	LEAQ ·bitrev16384_r4(SB), R12

	// Verify n == 16384
	CMPQ R13, $16384
	JNE  r4_16384_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $16384
	JL   r4_16384_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $16384
	JL   r4_16384_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $16384
	JL   r4_16384_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  r4_16384_use_dst
	MOVQ R11, R8             // In-place: use scratch

r4_16384_use_dst:
	// ==================================================================
	// Bit-reversal permutation
	// ==================================================================
	XORQ CX, CX              // CX = i = 0

r4_16384_bitrev_loop:
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	MOVQ DX, AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X0    // load src[bitrev[i]]
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)    // work[i] = src[bitrev[i]]
	INCQ CX
	CMPQ CX, $16384
	JL   r4_16384_bitrev_loop

r4_16384_stage1:
	// ==================================================================
	// Stage 1: 4096 groups, stride=4
	// ==================================================================

	XORQ CX, CX              // CX = base offset

r4_16384_stage1_loop:
	CMPQ CX, $16384
	JGE  r4_16384_stage2

	MOVQ CX, AX
	SHLQ $4, AX
	LEAQ (R8)(AX*1), SI
	MOVUPD (SI), X0          // a0
	MOVUPD 16(SI), X1        // a1
	MOVUPD 32(SI), X2        // a2
	MOVUPD 48(SI), X3        // a3

	VADDPD X0, X2, X4        // t0
	VSUBPD X2, X0, X5        // t1
	VADDPD X1, X3, X6        // t2
	VSUBPD X3, X1, X7        // t3

	// (-i)*t3
	VSHUFPD $0x1, X7, X7, X8
	VXORPD ·maskNegHiPD(SB), X8, X8

	// i*t3
	VSHUFPD $0x1, X7, X7, X11
	VXORPD ·maskNegLoPD(SB), X11, X11

	VADDPD X4, X6, X0        // y0
	VADDPD X5, X8, X1        // y1
	VSUBPD X6, X4, X2        // y2
	VADDPD X5, X11, X3       // y3

	MOVUPD X0, (SI)
	MOVUPD X1, 16(SI)
	MOVUPD X2, 32(SI)
	MOVUPD X3, 48(SI)

	ADDQ $4, CX
	JMP  r4_16384_stage1_loop

r4_16384_stage2:
	// ==================================================================
	// Stage 2: 1024 groups, 4 butterflies
	// Twiddle step = 1024
	// ==================================================================

	XORQ CX, CX

r4_16384_stage2_outer:
	CMPQ CX, $16384
	JGE  r4_16384_stage3

	XORQ DX, DX

r4_16384_stage2_inner:
	CMPQ DX, $4
	JGE  r4_16384_stage2_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 4(BX), SI
	LEAQ 8(BX), DI
	LEAQ 12(BX), R14

	// Twiddles: j*1024
	MOVQ DX, R15
	SHLQ $10, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ R15, R13
	SHLQ $1, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X9

	ADDQ R13, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X0
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X1
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X2
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X3

	// Complex multiply
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X3, X13
	VMOVAPD X13, X3

	VADDPD X0, X2, X4
	VSUBPD X2, X0, X5
	VADDPD X1, X3, X6
	VSUBPD X3, X1, X7

	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegHiPD(SB), X14, X14

	VSHUFPD $0x1, X7, X7, X12
	VXORPD ·maskNegLoPD(SB), X12, X12

	VADDPD X4, X6, X0
	VADDPD X5, X14, X1
	VSUBPD X6, X4, X2
	VADDPD X5, X12, X3

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD X1, (R8)(AX*1)
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD X2, (R8)(AX*1)
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD X3, (R8)(AX*1)

	INCQ DX
	JMP  r4_16384_stage2_inner

r4_16384_stage2_next:
	ADDQ $16, CX
	JMP  r4_16384_stage2_outer


r4_16384_stage3:
	// ==================================================================
	// Stage 3: 256 groups, 16 butterflies
	// Twiddle step = 256
	// ==================================================================

	XORQ CX, CX

r4_16384_stage3_outer:
	CMPQ CX, $16384
	JGE  r4_16384_stage4

	XORQ DX, DX

r4_16384_stage3_inner:
	CMPQ DX, $16
	JGE  r4_16384_stage3_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 16(BX), SI
	LEAQ 32(BX), DI
	LEAQ 48(BX), R14

	// Twiddles: j*256
	MOVQ DX, R15
	SHLQ $8, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ R15, R13
	SHLQ $1, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X9

	ADDQ R13, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X0
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X1
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X2
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X3

	// Complex multiply
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X3, X13
	VMOVAPD X13, X3

	VADDPD X0, X2, X4
	VSUBPD X2, X0, X5
	VADDPD X1, X3, X6
	VSUBPD X3, X1, X7

	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegHiPD(SB), X14, X14

	VSHUFPD $0x1, X7, X7, X12
	VXORPD ·maskNegLoPD(SB), X12, X12

	VADDPD X4, X6, X0
	VADDPD X5, X14, X1
	VSUBPD X6, X4, X2
	VADDPD X5, X12, X3

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD X1, (R8)(AX*1)
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD X2, (R8)(AX*1)
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD X3, (R8)(AX*1)

	INCQ DX
	JMP  r4_16384_stage3_inner

r4_16384_stage3_next:
	ADDQ $64, CX
	JMP  r4_16384_stage3_outer


r4_16384_stage4:
	// ==================================================================
	// Stage 4: 64 groups, 64 butterflies
	// Twiddle step = 64
	// ==================================================================

	XORQ CX, CX

r4_16384_stage4_outer:
	CMPQ CX, $16384
	JGE  r4_16384_stage5

	XORQ DX, DX

r4_16384_stage4_inner:
	CMPQ DX, $64
	JGE  r4_16384_stage4_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 64(BX), SI
	LEAQ 128(BX), DI
	LEAQ 192(BX), R14

	// Twiddles: j*64
	MOVQ DX, R15
	SHLQ $6, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ R15, R13
	SHLQ $1, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X9

	ADDQ R13, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X0
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X1
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X2
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X3

	// Complex multiply
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X3, X13
	VMOVAPD X13, X3

	VADDPD X0, X2, X4
	VSUBPD X2, X0, X5
	VADDPD X1, X3, X6
	VSUBPD X3, X1, X7

	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegHiPD(SB), X14, X14

	VSHUFPD $0x1, X7, X7, X12
	VXORPD ·maskNegLoPD(SB), X12, X12

	VADDPD X4, X6, X0
	VADDPD X5, X14, X1
	VSUBPD X6, X4, X2
	VADDPD X5, X12, X3

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD X1, (R8)(AX*1)
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD X2, (R8)(AX*1)
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD X3, (R8)(AX*1)

	INCQ DX
	JMP  r4_16384_stage4_inner

r4_16384_stage4_next:
	ADDQ $256, CX
	JMP  r4_16384_stage4_outer


r4_16384_stage5:
	// ==================================================================
	// Stage 5: 16 groups, 256 butterflies
	// Twiddle step = 16
	// ==================================================================

	XORQ CX, CX

r4_16384_stage5_outer:
	CMPQ CX, $16384
	JGE  r4_16384_stage6

	XORQ DX, DX

r4_16384_stage5_inner:
	CMPQ DX, $256
	JGE  r4_16384_stage5_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 256(BX), SI
	LEAQ 512(BX), DI
	LEAQ 768(BX), R14

	// Twiddles: j*16
	MOVQ DX, R15
	SHLQ $4, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ R15, R13
	SHLQ $1, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X9

	ADDQ R13, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X0
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X1
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X2
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X3

	// Complex multiply
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X3, X13
	VMOVAPD X13, X3

	VADDPD X0, X2, X4
	VSUBPD X2, X0, X5
	VADDPD X1, X3, X6
	VSUBPD X3, X1, X7

	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegHiPD(SB), X14, X14

	VSHUFPD $0x1, X7, X7, X12
	VXORPD ·maskNegLoPD(SB), X12, X12

	VADDPD X4, X6, X0
	VADDPD X5, X14, X1
	VSUBPD X6, X4, X2
	VADDPD X5, X12, X3

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD X1, (R8)(AX*1)
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD X2, (R8)(AX*1)
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD X3, (R8)(AX*1)

	INCQ DX
	JMP  r4_16384_stage5_inner

r4_16384_stage5_next:
	ADDQ $1024, CX
	JMP  r4_16384_stage5_outer


r4_16384_stage6:
	// ==================================================================
	// Stage 6: 4 groups, 1024 butterflies
	// Twiddle step = 4
	// ==================================================================

	XORQ CX, CX

r4_16384_stage6_outer:
	CMPQ CX, $16384
	JGE  r4_16384_stage7

	XORQ DX, DX

r4_16384_stage6_inner:
	CMPQ DX, $1024
	JGE  r4_16384_stage6_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 1024(BX), SI
	LEAQ 2048(BX), DI
	LEAQ 3072(BX), R14

	// Twiddles: j*4
	MOVQ DX, R15
	SHLQ $2, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ R15, R13
	SHLQ $1, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X9

	ADDQ R13, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X0
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X1
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X2
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X3

	// Complex multiply
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X3, X13
	VMOVAPD X13, X3

	VADDPD X0, X2, X4
	VSUBPD X2, X0, X5
	VADDPD X1, X3, X6
	VSUBPD X3, X1, X7

	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegHiPD(SB), X14, X14

	VSHUFPD $0x1, X7, X7, X12
	VXORPD ·maskNegLoPD(SB), X12, X12

	VADDPD X4, X6, X0
	VADDPD X5, X14, X1
	VSUBPD X6, X4, X2
	VADDPD X5, X12, X3

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD X1, (R8)(AX*1)
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD X2, (R8)(AX*1)
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD X3, (R8)(AX*1)

	INCQ DX
	JMP  r4_16384_stage6_inner

r4_16384_stage6_next:
	ADDQ $4096, CX
	JMP  r4_16384_stage6_outer

r4_16384_stage7:
	// ==================================================================
	// Stage 7: 1 group, 4096 butterflies
	// Twiddle step = 1
	// ==================================================================

	XORQ DX, DX

r4_16384_stage7_loop:
	CMPQ DX, $4096
	JGE  r4_16384_done

	MOVQ DX, BX
	LEAQ 4096(DX), SI
	LEAQ 8192(DX), DI
	LEAQ 12288(DX), R14

	// Twiddles: j*1
	MOVQ DX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ DX, R15
	SHLQ $1, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X9

	ADDQ DX, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X0
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X1
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X2
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X3

	// Complex multiply
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X3, X13
	VMOVAPD X13, X3

	VADDPD X0, X2, X4
	VSUBPD X2, X0, X5
	VADDPD X1, X3, X6
	VSUBPD X3, X1, X7

	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegHiPD(SB), X14, X14

	VSHUFPD $0x1, X7, X7, X12
	VXORPD ·maskNegLoPD(SB), X12, X12

	VADDPD X4, X6, X0
	VADDPD X5, X14, X1
	VSUBPD X6, X4, X2
	VADDPD X5, X12, X3

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD X1, (R8)(AX*1)
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD X2, (R8)(AX*1)
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD X3, (R8)(AX*1)

	INCQ DX
	JMP  r4_16384_stage7_loop


r4_16384_done:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r4_16384_ret

	XORQ CX, CX

r4_16384_copy_loop:
	VMOVUPD (R8)(CX*1), Y0
	VMOVUPD 32(R8)(CX*1), Y1
	VMOVUPD Y0, (R9)(CX*1)
	VMOVUPD Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $262144
	JL   r4_16384_copy_loop

r4_16384_ret:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

r4_16384_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform
// ===========================================================================
TEXT ·InverseAVX2Size16384Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13     // n
	LEAQ ·bitrev16384_r4(SB), R12

	CMPQ R13, $16384
	JNE  r4_16384_inv_return_false

	MOVQ dst+8(FP), AX
	CMPQ AX, $16384
	JL   r4_16384_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $16384
	JL   r4_16384_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $16384
	JL   r4_16384_inv_return_false

	CMPQ R8, R9
	JNE  r4_16384_inv_use_dst
	MOVQ R11, R8

r4_16384_inv_use_dst:
	// Bit-reversal permutation
	XORQ CX, CX
r4_16384_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ DX, AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X0
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	INCQ CX
	CMPQ CX, $16384
	JL   r4_16384_inv_bitrev_loop

r4_16384_inv_stage1:
	// ==================================================================
	// Stage 1: 4096 groups, stride=4
	// ==================================================================
	XORQ CX, CX

r4_16384_inv_stage1_loop:
	CMPQ CX, $16384
	JGE  r4_16384_inv_stage2

	MOVQ CX, AX
	SHLQ $4, AX
	LEAQ (R8)(AX*1), SI
	MOVUPD (SI), X0
	MOVUPD 16(SI), X1
	MOVUPD 32(SI), X2
	MOVUPD 48(SI), X3

	VADDPD X0, X2, X4
	VSUBPD X2, X0, X5
	VADDPD X1, X3, X6
	VSUBPD X3, X1, X7

	// i*t3
	VSHUFPD $0x1, X7, X7, X8
	VXORPD ·maskNegLoPD(SB), X8, X8

	// (-i)*t3
	VSHUFPD $0x1, X7, X7, X11
	VXORPD ·maskNegHiPD(SB), X11, X11

	VADDPD X4, X6, X0
	VADDPD X5, X8, X1
	VSUBPD X6, X4, X2
	VADDPD X5, X11, X3

	MOVUPD X0, (SI)
	MOVUPD X1, 16(SI)
	MOVUPD X2, 32(SI)
	MOVUPD X3, 48(SI)

	ADDQ $4, CX
	JMP  r4_16384_inv_stage1_loop

r4_16384_inv_stage2:
	// ==================================================================
	// Stage 2: 1024 groups, 4 butterflies (conjugated twiddles)
	// Twiddle step = 1024
	// ==================================================================

	XORQ CX, CX

r4_16384_inv_stage2_outer:
	CMPQ CX, $16384
	JGE  r4_16384_inv_stage3

	XORQ DX, DX

r4_16384_inv_stage2_inner:
	CMPQ DX, $4
	JGE  r4_16384_inv_stage2_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 4(BX), SI
	LEAQ 8(BX), DI
	LEAQ 12(BX), R14

	// Twiddles: j*1024
	MOVQ DX, R15
	SHLQ $10, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ R15, R13
	SHLQ $1, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X9

	ADDQ R13, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X0
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X1
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X2
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X3

	// Conjugated complex multiply
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X3, X13
	VMOVAPD X13, X3

	VADDPD X0, X2, X4
	VSUBPD X2, X0, X5
	VADDPD X1, X3, X6
	VSUBPD X3, X1, X7

	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegLoPD(SB), X14, X14

	VSHUFPD $0x1, X7, X7, X12
	VXORPD ·maskNegHiPD(SB), X12, X12

	VADDPD X4, X6, X0
	VADDPD X5, X14, X1
	VSUBPD X6, X4, X2
	VADDPD X5, X12, X3

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD X1, (R8)(AX*1)
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD X2, (R8)(AX*1)
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD X3, (R8)(AX*1)

	INCQ DX
	JMP  r4_16384_inv_stage2_inner

r4_16384_inv_stage2_next:
	ADDQ $16, CX
	JMP  r4_16384_inv_stage2_outer


r4_16384_inv_stage3:
	// ==================================================================
	// Stage 3: 256 groups, 16 butterflies (conjugated twiddles)
	// Twiddle step = 256
	// ==================================================================

	XORQ CX, CX

r4_16384_inv_stage3_outer:
	CMPQ CX, $16384
	JGE  r4_16384_inv_stage4

	XORQ DX, DX

r4_16384_inv_stage3_inner:
	CMPQ DX, $16
	JGE  r4_16384_inv_stage3_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 16(BX), SI
	LEAQ 32(BX), DI
	LEAQ 48(BX), R14

	// Twiddles: j*256
	MOVQ DX, R15
	SHLQ $8, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ R15, R13
	SHLQ $1, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X9

	ADDQ R13, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X0
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X1
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X2
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X3

	// Conjugated complex multiply
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X3, X13
	VMOVAPD X13, X3

	VADDPD X0, X2, X4
	VSUBPD X2, X0, X5
	VADDPD X1, X3, X6
	VSUBPD X3, X1, X7

	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegLoPD(SB), X14, X14

	VSHUFPD $0x1, X7, X7, X12
	VXORPD ·maskNegHiPD(SB), X12, X12

	VADDPD X4, X6, X0
	VADDPD X5, X14, X1
	VSUBPD X6, X4, X2
	VADDPD X5, X12, X3

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD X1, (R8)(AX*1)
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD X2, (R8)(AX*1)
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD X3, (R8)(AX*1)

	INCQ DX
	JMP  r4_16384_inv_stage3_inner

r4_16384_inv_stage3_next:
	ADDQ $64, CX
	JMP  r4_16384_inv_stage3_outer


r4_16384_inv_stage4:
	// ==================================================================
	// Stage 4: 64 groups, 64 butterflies (conjugated twiddles)
	// Twiddle step = 64
	// ==================================================================

	XORQ CX, CX

r4_16384_inv_stage4_outer:
	CMPQ CX, $16384
	JGE  r4_16384_inv_stage5

	XORQ DX, DX

r4_16384_inv_stage4_inner:
	CMPQ DX, $64
	JGE  r4_16384_inv_stage4_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 64(BX), SI
	LEAQ 128(BX), DI
	LEAQ 192(BX), R14

	// Twiddles: j*64
	MOVQ DX, R15
	SHLQ $6, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ R15, R13
	SHLQ $1, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X9

	ADDQ R13, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X0
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X1
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X2
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X3

	// Conjugated complex multiply
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X3, X13
	VMOVAPD X13, X3

	VADDPD X0, X2, X4
	VSUBPD X2, X0, X5
	VADDPD X1, X3, X6
	VSUBPD X3, X1, X7

	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegLoPD(SB), X14, X14

	VSHUFPD $0x1, X7, X7, X12
	VXORPD ·maskNegHiPD(SB), X12, X12

	VADDPD X4, X6, X0
	VADDPD X5, X14, X1
	VSUBPD X6, X4, X2
	VADDPD X5, X12, X3

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD X1, (R8)(AX*1)
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD X2, (R8)(AX*1)
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD X3, (R8)(AX*1)

	INCQ DX
	JMP  r4_16384_inv_stage4_inner

r4_16384_inv_stage4_next:
	ADDQ $256, CX
	JMP  r4_16384_inv_stage4_outer


r4_16384_inv_stage5:
	// ==================================================================
	// Stage 5: 16 groups, 256 butterflies (conjugated twiddles)
	// Twiddle step = 16
	// ==================================================================

	XORQ CX, CX

r4_16384_inv_stage5_outer:
	CMPQ CX, $16384
	JGE  r4_16384_inv_stage6

	XORQ DX, DX

r4_16384_inv_stage5_inner:
	CMPQ DX, $256
	JGE  r4_16384_inv_stage5_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 256(BX), SI
	LEAQ 512(BX), DI
	LEAQ 768(BX), R14

	// Twiddles: j*16
	MOVQ DX, R15
	SHLQ $4, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ R15, R13
	SHLQ $1, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X9

	ADDQ R13, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X0
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X1
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X2
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X3

	// Conjugated complex multiply
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X3, X13
	VMOVAPD X13, X3

	VADDPD X0, X2, X4
	VSUBPD X2, X0, X5
	VADDPD X1, X3, X6
	VSUBPD X3, X1, X7

	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegLoPD(SB), X14, X14

	VSHUFPD $0x1, X7, X7, X12
	VXORPD ·maskNegHiPD(SB), X12, X12

	VADDPD X4, X6, X0
	VADDPD X5, X14, X1
	VSUBPD X6, X4, X2
	VADDPD X5, X12, X3

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD X1, (R8)(AX*1)
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD X2, (R8)(AX*1)
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD X3, (R8)(AX*1)

	INCQ DX
	JMP  r4_16384_inv_stage5_inner

r4_16384_inv_stage5_next:
	ADDQ $1024, CX
	JMP  r4_16384_inv_stage5_outer


r4_16384_inv_stage6:
	// ==================================================================
	// Stage 6: 4 groups, 1024 butterflies (conjugated twiddles)
	// Twiddle step = 4
	// ==================================================================

	XORQ CX, CX

r4_16384_inv_stage6_outer:
	CMPQ CX, $16384
	JGE  r4_16384_inv_stage7

	XORQ DX, DX

r4_16384_inv_stage6_inner:
	CMPQ DX, $1024
	JGE  r4_16384_inv_stage6_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 1024(BX), SI
	LEAQ 2048(BX), DI
	LEAQ 3072(BX), R14

	// Twiddles: j*4
	MOVQ DX, R15
	SHLQ $2, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ R15, R13
	SHLQ $1, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X9

	ADDQ R13, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X0
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X1
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X2
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X3

	// Conjugated complex multiply
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X3, X13
	VMOVAPD X13, X3

	VADDPD X0, X2, X4
	VSUBPD X2, X0, X5
	VADDPD X1, X3, X6
	VSUBPD X3, X1, X7

	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegLoPD(SB), X14, X14

	VSHUFPD $0x1, X7, X7, X12
	VXORPD ·maskNegHiPD(SB), X12, X12

	VADDPD X4, X6, X0
	VADDPD X5, X14, X1
	VSUBPD X6, X4, X2
	VADDPD X5, X12, X3

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD X1, (R8)(AX*1)
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD X2, (R8)(AX*1)
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD X3, (R8)(AX*1)

	INCQ DX
	JMP  r4_16384_inv_stage6_inner

r4_16384_inv_stage6_next:
	ADDQ $4096, CX
	JMP  r4_16384_inv_stage6_outer

r4_16384_inv_stage7:
	// ==================================================================
	// Stage 7: 1 group, 4096 butterflies (conjugated twiddles)
	// Twiddle step = 1
	// ==================================================================

	XORQ DX, DX

r4_16384_inv_stage7_loop:
	CMPQ DX, $4096
	JGE  r4_16384_inv_scale

	MOVQ DX, BX
	LEAQ 4096(DX), SI
	LEAQ 8192(DX), DI
	LEAQ 12288(DX), R14

	// Twiddles: j*1
	MOVQ DX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8

	MOVQ DX, R15
	SHLQ $1, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X9

	ADDQ DX, R15
	MOVQ R15, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X0
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X1
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X2
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X3

	// Conjugated complex multiply
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X3, X13
	VMOVAPD X13, X3

	VADDPD X0, X2, X4
	VSUBPD X2, X0, X5
	VADDPD X1, X3, X6
	VSUBPD X3, X1, X7

	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegLoPD(SB), X14, X14

	VSHUFPD $0x1, X7, X7, X12
	VXORPD ·maskNegHiPD(SB), X12, X12

	VADDPD X4, X6, X0
	VADDPD X5, X14, X1
	VSUBPD X6, X4, X2
	VADDPD X5, X12, X3

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD X1, (R8)(AX*1)
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD X2, (R8)(AX*1)
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD X3, (R8)(AX*1)

	INCQ DX
	JMP  r4_16384_inv_stage7_loop


r4_16384_inv_scale:
	// 1/16384 scaling
	MOVSD ·oneSixteenThousandThreeHundredEightyFourth64(SB), X8
	VBROADCASTSD X8, Y8

	XORQ CX, CX
r4_16384_inv_scale_loop:
	VMOVUPD (R8)(CX*1), Y0
	VMOVUPD 32(R8)(CX*1), Y1
	VMULPD Y8, Y0, Y0
	VMULPD Y8, Y1, Y1
	VMOVUPD Y0, (R8)(CX*1)
	VMOVUPD Y1, 32(R8)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $262144
	JL   r4_16384_inv_scale_loop

	// Copy if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r4_16384_inv_done

	XORQ CX, CX
r4_16384_inv_copy_loop:
	VMOVUPD (R8)(CX*1), Y0
	VMOVUPD 32(R8)(CX*1), Y1
	VMOVUPD Y0, (R9)(CX*1)
	VMOVUPD Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $262144
	JL   r4_16384_inv_copy_loop

r4_16384_inv_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

r4_16384_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

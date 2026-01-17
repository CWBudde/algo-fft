//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-8192 Mixed-Radix-2/4 FFT Kernels for AMD64
// ===========================================================================
//
// This file contains a mixed-radix-2/4 DIT FFT optimized for size 8192.
// Stages:
//   - Stage 1-6: radix-4 (6 stages)
//   - Stage 7: radix-2 (final combine)
//
// Mixed-radix bit-reversal indices are required for stage 1.
//
// ===========================================================================

#include "textflag.h"

TEXT ·ForwardAVX2Size8192Mixed24Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 8192)
	LEAQ ·bitrev8192_m24(SB), R12

	// Verify n == 8192
	CMPQ R13, $8192
	JNE  m24_8192_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $8192
	JL   m24_8192_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8192
	JL   m24_8192_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8192
	JL   m24_8192_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  m24_8192_use_dst
	MOVQ R11, R8             // In-place: use scratch

m24_8192_use_dst:
	// ==================================================================
	// Stage 1: 2048 radix-4 butterflies with mixed-radix bit-reversal
	// ==================================================================
	XORQ CX, CX              // CX = base offset

m24_8192_stage1_loop:
	CMPQ CX, $8192
	JGE  m24_8192_stage2

	// Load bit-reversed indices
	MOVQ (R12)(CX*8), DX
	MOVQ 8(R12)(CX*8), SI
	MOVQ 16(R12)(CX*8), DI
	MOVQ 24(R12)(CX*8), R14

	// Load input values
	MOVQ DX, AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X0
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X1
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X2
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X3

	// Radix-4 butterfly
	VADDPD X0, X2, X4
	VSUBPD X2, X0, X5
	VADDPD X1, X3, X6
	VSUBPD X3, X1, X7

	// (-i)*t3 for y1
	VSHUFPD $0x1, X7, X7, X8
	VXORPD ·maskNegHiPD(SB), X8, X8

	// i*t3 for y3
	VSHUFPD $0x1, X7, X7, X11
	VXORPD ·maskNegLoPD(SB), X11, X11

	VADDPD X4, X6, X0
	VADDPD X5, X8, X1
	VSUBPD X6, X4, X2
	VADDPD X5, X11, X3

	// Store outputs to work buffer
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X1, 16(R8)(AX*1)
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X2, 32(R8)(AX*1)
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X3, 48(R8)(AX*1)

	ADDQ $4, CX
	JMP  m24_8192_stage1_loop

m24_8192_stage2:
	// ==================================================================
	// Stage 2: 512 groups, each with 4 butterflies
	// Twiddle step = 512
	// ==================================================================
	XORQ CX, CX

m24_8192_stage2_outer:
	CMPQ CX, $8192
	JGE  m24_8192_stage3

	XORQ DX, DX

m24_8192_stage2_inner:
	CMPQ DX, $4
	JGE  m24_8192_stage2_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 4(BX), SI
	LEAQ 8(BX), DI
	LEAQ 12(BX), R14

	// Twiddles: j*512, 2*j*512, 3*j*512
	MOVQ DX, R15
	SHLQ $9, R15
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

	// Complex multiply a1*w1, a2*w2, a3*w3
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
	JMP  m24_8192_stage2_inner

m24_8192_stage2_next:
	ADDQ $16, CX
	JMP  m24_8192_stage2_outer

m24_8192_stage3:
	// ==================================================================
	// Stage 3: 128 groups, each with 16 butterflies
	// Twiddle step = 128
	// ==================================================================
	XORQ CX, CX

m24_8192_stage3_outer:
	CMPQ CX, $8192
	JGE  m24_8192_stage4

	XORQ DX, DX

m24_8192_stage3_inner:
	CMPQ DX, $16
	JGE  m24_8192_stage3_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 16(BX), SI
	LEAQ 32(BX), DI
	LEAQ 48(BX), R14

	MOVQ DX, R15
	SHLQ $7, R15
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
	JMP  m24_8192_stage3_inner

m24_8192_stage3_next:
	ADDQ $64, CX
	JMP  m24_8192_stage3_outer

m24_8192_stage4:
	// ==================================================================
	// Stage 4: 32 groups, each with 64 butterflies
	// Twiddle step = 32
	// ==================================================================
	XORQ CX, CX

m24_8192_stage4_outer:
	CMPQ CX, $8192
	JGE  m24_8192_stage5

	XORQ DX, DX

m24_8192_stage4_inner:
	CMPQ DX, $64
	JGE  m24_8192_stage4_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 64(BX), SI
	LEAQ 128(BX), DI
	LEAQ 192(BX), R14

	MOVQ DX, R15
	SHLQ $5, R15
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
	JMP  m24_8192_stage4_inner

m24_8192_stage4_next:
	ADDQ $256, CX
	JMP  m24_8192_stage4_outer

m24_8192_stage5:
	// ==================================================================
	// Stage 5: 8 groups, each with 256 butterflies
	// Twiddle step = 8
	// ==================================================================
	XORQ CX, CX

m24_8192_stage5_outer:
	CMPQ CX, $8192
	JGE  m24_8192_stage6

	XORQ DX, DX

m24_8192_stage5_inner:
	CMPQ DX, $256
	JGE  m24_8192_stage5_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 256(BX), SI
	LEAQ 512(BX), DI
	LEAQ 768(BX), R14

	MOVQ DX, R15
	SHLQ $3, R15
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
	JMP  m24_8192_stage5_inner

m24_8192_stage5_next:
	ADDQ $1024, CX
	JMP  m24_8192_stage5_outer

m24_8192_stage6:
	// ==================================================================
	// Stage 6: 2 groups, each with 1024 butterflies
	// Twiddle step = 2
	// ==================================================================
	XORQ CX, CX

m24_8192_stage6_outer:
	CMPQ CX, $8192
	JGE  m24_8192_stage7

	XORQ DX, DX

m24_8192_stage6_inner:
	CMPQ DX, $1024
	JGE  m24_8192_stage6_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 1024(BX), SI
	LEAQ 2048(BX), DI
	LEAQ 3072(BX), R14

	MOVQ DX, R15
	SHLQ $1, R15
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
	JMP  m24_8192_stage6_inner

m24_8192_stage6_next:
	ADDQ $4096, CX
	JMP  m24_8192_stage6_outer

m24_8192_stage7:
	// ==================================================================
	// Stage 7: radix-2 final stage
	// ==================================================================
	XORQ CX, CX

m24_8192_stage7_loop:
	CMPQ CX, $4096
	JGE  m24_8192_forward_done

	MOVQ CX, BX
	LEAQ 4096(BX), SI

	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8   // twiddle
	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X0    // a
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X1    // b

	// b = b * w
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X1, X13
	VMOVAPD X13, X1

	VADDPD X0, X1, X2
	VSUBPD X1, X0, X3

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD X2, (R8)(AX*1)
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD X3, (R8)(AX*1)

	INCQ CX
	JMP  m24_8192_stage7_loop

m24_8192_forward_done:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   m24_8192_forward_ret

	XORQ CX, CX

m24_8192_forward_copy_loop:
	VMOVUPD (R8)(CX*1), Y0
	VMOVUPD 32(R8)(CX*1), Y1
	VMOVUPD Y0, (R9)(CX*1)
	VMOVUPD Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $131072
	JL   m24_8192_forward_copy_loop

m24_8192_forward_ret:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

m24_8192_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 8192, complex128, mixed-radix-2/4
// ===========================================================================
TEXT ·InverseAVX2Size8192Mixed24Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 8192)
	LEAQ ·bitrev8192_m24(SB), R12

	// Verify n == 8192
	CMPQ R13, $8192
	JNE  m24_8192_inv_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $8192
	JL   m24_8192_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $8192
	JL   m24_8192_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8192
	JL   m24_8192_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  m24_8192_inv_use_dst
	MOVQ R11, R8             // In-place: use scratch

m24_8192_inv_use_dst:
	// ==================================================================
	// Stage 1: 2048 radix-4 butterflies with mixed-radix bit-reversal
	// ==================================================================
	XORQ CX, CX

m24_8192_inv_stage1_loop:
	CMPQ CX, $8192
	JGE  m24_8192_inv_stage2

	MOVQ (R12)(CX*8), DX
	MOVQ 8(R12)(CX*8), SI
	MOVQ 16(R12)(CX*8), DI
	MOVQ 24(R12)(CX*8), R14

	MOVQ DX, AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X0
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X1
	MOVQ DI, AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X2
	MOVQ R14, AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X3

	VADDPD X0, X2, X4
	VSUBPD X2, X0, X5
	VADDPD X1, X3, X6
	VSUBPD X3, X1, X7

	// i*t3 for y1
	VSHUFPD $0x1, X7, X7, X8
	VXORPD ·maskNegLoPD(SB), X8, X8

	// (-i)*t3 for y3
	VSHUFPD $0x1, X7, X7, X11
	VXORPD ·maskNegHiPD(SB), X11, X11

	VADDPD X4, X6, X0
	VADDPD X5, X8, X1
	VSUBPD X6, X4, X2
	VADDPD X5, X11, X3

	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X1, 16(R8)(AX*1)
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X2, 32(R8)(AX*1)
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X3, 48(R8)(AX*1)

	ADDQ $4, CX
	JMP  m24_8192_inv_stage1_loop

m24_8192_inv_stage2:
	// ==================================================================
	// Stage 2: 512 groups, each with 4 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ CX, CX

m24_8192_inv_stage2_outer:
	CMPQ CX, $8192
	JGE  m24_8192_inv_stage3

	XORQ DX, DX

m24_8192_inv_stage2_inner:
	CMPQ DX, $4
	JGE  m24_8192_inv_stage2_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 4(BX), SI
	LEAQ 8(BX), DI
	LEAQ 12(BX), R14

	MOVQ DX, R15
	SHLQ $9, R15
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
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
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
	JMP  m24_8192_inv_stage2_inner

m24_8192_inv_stage2_next:
	ADDQ $16, CX
	JMP  m24_8192_inv_stage2_outer

m24_8192_inv_stage3:
	// ==================================================================
	// Stage 3: 128 groups, each with 16 butterflies
	// ==================================================================
	XORQ CX, CX

m24_8192_inv_stage3_outer:
	CMPQ CX, $8192
	JGE  m24_8192_inv_stage4

	XORQ DX, DX

m24_8192_inv_stage3_inner:
	CMPQ DX, $16
	JGE  m24_8192_inv_stage3_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 16(BX), SI
	LEAQ 32(BX), DI
	LEAQ 48(BX), R14

	MOVQ DX, R15
	SHLQ $7, R15
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

	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
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
	JMP  m24_8192_inv_stage3_inner

m24_8192_inv_stage3_next:
	ADDQ $64, CX
	JMP  m24_8192_inv_stage3_outer

m24_8192_inv_stage4:
	// ==================================================================
	// Stage 4: 32 groups, each with 64 butterflies
	// ==================================================================
	XORQ CX, CX

m24_8192_inv_stage4_outer:
	CMPQ CX, $8192
	JGE  m24_8192_inv_stage5

	XORQ DX, DX

m24_8192_inv_stage4_inner:
	CMPQ DX, $64
	JGE  m24_8192_inv_stage4_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 64(BX), SI
	LEAQ 128(BX), DI
	LEAQ 192(BX), R14

	MOVQ DX, R15
	SHLQ $5, R15
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

	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
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
	JMP  m24_8192_inv_stage4_inner

m24_8192_inv_stage4_next:
	ADDQ $256, CX
	JMP  m24_8192_inv_stage4_outer

m24_8192_inv_stage5:
	// ==================================================================
	// Stage 5: 8 groups, each with 256 butterflies
	// ==================================================================
	XORQ CX, CX

m24_8192_inv_stage5_outer:
	CMPQ CX, $8192
	JGE  m24_8192_inv_stage6

	XORQ DX, DX

m24_8192_inv_stage5_inner:
	CMPQ DX, $256
	JGE  m24_8192_inv_stage5_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 256(BX), SI
	LEAQ 512(BX), DI
	LEAQ 768(BX), R14

	MOVQ DX, R15
	SHLQ $3, R15
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

	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
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
	JMP  m24_8192_inv_stage5_inner

m24_8192_inv_stage5_next:
	ADDQ $1024, CX
	JMP  m24_8192_inv_stage5_outer

m24_8192_inv_stage6:
	// ==================================================================
	// Stage 6: 2 groups, each with 1024 butterflies
	// ==================================================================
	XORQ CX, CX

m24_8192_inv_stage6_outer:
	CMPQ CX, $8192
	JGE  m24_8192_inv_stage7

	XORQ DX, DX

m24_8192_inv_stage6_inner:
	CMPQ DX, $1024
	JGE  m24_8192_inv_stage6_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 1024(BX), SI
	LEAQ 2048(BX), DI
	LEAQ 3072(BX), R14

	MOVQ DX, R15
	SHLQ $1, R15
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

	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1

	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2

	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
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
	JMP  m24_8192_inv_stage6_inner

m24_8192_inv_stage6_next:
	ADDQ $4096, CX
	JMP  m24_8192_inv_stage6_outer

m24_8192_inv_stage7:
	// ==================================================================
	// Stage 7: radix-2 final stage (conjugated twiddles)
	// ==================================================================
	XORQ CX, CX

m24_8192_inv_stage7_loop:
	CMPQ CX, $4096
	JGE  m24_8192_inv_scale

	MOVQ CX, BX
	LEAQ 4096(BX), SI

	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8
	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X0
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD (R8)(AX*1), X1

	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1

	VADDPD X0, X1, X2
	VSUBPD X1, X0, X3

	MOVQ BX, AX
	SHLQ $4, AX
	MOVUPD X2, (R8)(AX*1)
	MOVQ SI, AX
	SHLQ $4, AX
	MOVUPD X3, (R8)(AX*1)

	INCQ CX
	JMP  m24_8192_inv_stage7_loop

m24_8192_inv_scale:
	// ==================================================================
	// Apply 1/N scaling for inverse transform
	// ==================================================================
	MOVSD ·eightThousandOneHundredThirtySecond64(SB), X8
	VBROADCASTSD X8, Y8

	XORQ CX, CX

m24_8192_inv_scale_loop:
	VMOVUPD (R8)(CX*1), Y0
	VMOVUPD 32(R8)(CX*1), Y1
	VMULPD Y8, Y0, Y0
	VMULPD Y8, Y1, Y1
	VMOVUPD Y0, (R8)(CX*1)
	VMOVUPD Y1, 32(R8)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $131072
	JL   m24_8192_inv_scale_loop

	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   m24_8192_inv_done

	XORQ CX, CX

m24_8192_inv_copy_loop:
	VMOVUPD (R8)(CX*1), Y0
	VMOVUPD 32(R8)(CX*1), Y1
	VMOVUPD Y0, (R9)(CX*1)
	VMOVUPD Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $131072
	JL   m24_8192_inv_copy_loop

m24_8192_inv_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

m24_8192_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

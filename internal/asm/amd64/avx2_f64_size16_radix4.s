//go:build amd64 && fft_asm && !purego

// ===========================================================================
// AVX2 Size-16 Radix-4 FFT Kernels for AMD64 (complex128)
// ===========================================================================

#include "textflag.h"

// Forward transform, size 16, complex64, radix-4 variant
TEXT ·forwardAVX2Size16Radix4Complex64Asm(SB), NOSPLIT, $0-121
TEXT ·forwardAVX2Size16Radix4Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // dst pointer
	MOVQ src+24(FP), R9      // src pointer
	MOVQ twiddle+48(FP), R10 // twiddle pointer
	MOVQ scratch+72(FP), R11 // scratch pointer
	MOVQ bitrev+96(FP), R12  // bitrev pointer
	MOVQ src+32(FP), R13     // n (should be 16)

	CMPQ R13, $16
	JNE  size16_r4_128_return_false

	// Validate all slice lengths >= 16
	MOVQ dst+8(FP), AX
	CMPQ AX, $16
	JL   size16_r4_128_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $16
	JL   size16_r4_128_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $16
	JL   size16_r4_128_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $16
	JL   size16_r4_128_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size16_r4_128_use_dst
	MOVQ R11, R8             // In-place: use scratch

size16_r4_128_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 bit-reversal)
	// ==================================================================
	XORQ CX, CX

size16_r4_128_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	// src[DX]
	MOVQ DX, AX
	SHLQ $1, AX
	MOVUPD (R9)(AX*8), X0
	// work[CX]
	MOVQ CX, AX
	SHLQ $1, AX
	MOVUPD X0, (R8)(AX*8)

	INCQ CX
	CMPQ CX, $16
	JL   size16_r4_128_bitrev_loop

size16_r4_128_stage1:
	// ==================================================================
	// Stage 1: 4 radix-4 butterflies, stride=4
	// ==================================================================
	XORQ CX, CX

size16_r4_128_stage1_loop:
	CMPQ CX, $16
	JGE  size16_r4_128_stage2

	MOVQ CX, AX
	SHLQ $1, AX
	LEAQ (R8)(AX*8), SI

	MOVUPD (SI), X0
	MOVUPD 16(SI), X1
	MOVUPD 32(SI), X2
	MOVUPD 48(SI), X3

	VADDPD X2, X0, X4
	VSUBPD X2, X0, X5
	VADDPD X3, X1, X6
	VSUBPD X3, X1, X7

	// (-i)*t3: (im, -re) = swap(t3) with sign toggle on high lane
	VSHUFPD $0x1, X7, X7, X8
	VXORPD ·maskNegHiPD(SB), X8, X8

	// i*t3: (-im, re) = swap(t3) with sign toggle on low lane
	VSHUFPD $0x1, X7, X7, X11
	VXORPD ·maskNegLoPD(SB), X11, X11

	VADDPD X6, X4, X0
	VADDPD X8, X5, X1
	VSUBPD X6, X4, X2
	VADDPD X11, X5, X3

	MOVUPD X0, (SI)
	MOVUPD X1, 16(SI)
	MOVUPD X2, 32(SI)
	MOVUPD X3, 48(SI)

	ADDQ $4, CX
	JMP  size16_r4_128_stage1_loop

size16_r4_128_stage2:
	// ==================================================================
	// Stage 2: 1 group, 4 butterflies
	// Twiddle step = 1
	// ==================================================================
	XORQ DX, DX

size16_r4_128_stage2_loop:
	CMPQ DX, $4
	JGE  size16_r4_128_done

	MOVQ DX, BX
	LEAQ 4(DX), SI
	LEAQ 8(DX), DI
	LEAQ 12(DX), R14

	// Twiddle factors: twiddle[j], twiddle[2*j], twiddle[3*j]
	MOVQ DX, R15
	// w1 = twiddle[j]
	MOVQ R15, AX
	SHLQ $1, AX
	MOVUPD (R10)(AX*8), X8
	// w2 = twiddle[2*j]
	MOVQ R15, AX
	SHLQ $1, AX              // AX = 2*j
	SHLQ $1, AX              // AX = (2*j) << 1 = 4*j (scaled for *8)
	MOVUPD (R10)(AX*8), X9
	// w3 = twiddle[3*j]
	MOVQ R15, AX
	SHLQ $1, AX              // AX = 2*j
	ADDQ R15, AX             // AX = 3*j
	SHLQ $1, AX              // AX = 6*j (scaled for *8)
	MOVUPD (R10)(AX*8), X10

	// Load a0..a3
	MOVQ BX, AX
	SHLQ $1, AX
	MOVUPD (R8)(AX*8), X0
	MOVQ SI, AX
	SHLQ $1, AX
	MOVUPD (R8)(AX*8), X1
	MOVQ DI, AX
	SHLQ $1, AX
	MOVUPD (R8)(AX*8), X2
	MOVQ R14, AX
	SHLQ $1, AX
	MOVUPD (R8)(AX*8), X3

	// Complex multiply a1*w1
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X1, X13
	VMOVAPD X13, X1

	// Complex multiply a2*w2
	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X2, X13
	VMOVAPD X13, X2

	// Complex multiply a3*w3
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X3, X13
	VMOVAPD X13, X3

	VADDPD X2, X0, X4
	VSUBPD X2, X0, X5
	VADDPD X3, X1, X6
	VSUBPD X3, X1, X7

	// (-i)*t3
	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegHiPD(SB), X14, X14

	// i*t3
	VSHUFPD $0x1, X7, X7, X12
	VXORPD ·maskNegLoPD(SB), X12, X12

	VADDPD X6, X4, X0
	VADDPD X14, X5, X1
	VSUBPD X6, X4, X2
	VADDPD X12, X5, X3

	// Store results
	MOVQ BX, AX
	SHLQ $1, AX
	MOVUPD X0, (R8)(AX*8)
	MOVQ SI, AX
	SHLQ $1, AX
	MOVUPD X1, (R8)(AX*8)
	MOVQ DI, AX
	SHLQ $1, AX
	MOVUPD X2, (R8)(AX*8)
	MOVQ R14, AX
	SHLQ $1, AX
	MOVUPD X3, (R8)(AX*8)

	INCQ DX
	JMP  size16_r4_128_stage2_loop

size16_r4_128_done:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size16_r4_128_done_direct

	XORQ CX, CX

size16_r4_128_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $256
	JL   size16_r4_128_copy_loop

size16_r4_128_done_direct:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size16_r4_128_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET

// Inverse transform, size 16, complex128, radix-4 variant
TEXT ·inverseAVX2Size16Radix4Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $16
	JNE  size16_r4_inv_128_return_false

	// Validate all slice lengths >= 16
	MOVQ dst+8(FP), AX
	CMPQ AX, $16
	JL   size16_r4_inv_128_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $16
	JL   size16_r4_inv_128_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $16
	JL   size16_r4_inv_128_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $16
	JL   size16_r4_inv_128_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size16_r4_inv_128_use_dst
	MOVQ R11, R8

size16_r4_inv_128_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 bit-reversal)
	// ==================================================================
	XORQ CX, CX

size16_r4_inv_128_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ DX, AX
	SHLQ $1, AX
	MOVUPD (R9)(AX*8), X0
	MOVQ CX, AX
	SHLQ $1, AX
	MOVUPD X0, (R8)(AX*8)
	INCQ CX
	CMPQ CX, $16
	JL   size16_r4_inv_128_bitrev_loop

size16_r4_inv_128_stage1:
	// ==================================================================
	// Stage 1: 4 radix-4 butterflies, stride=4
	// ==================================================================
	XORQ CX, CX

size16_r4_inv_128_stage1_loop:
	CMPQ CX, $16
	JGE  size16_r4_inv_128_stage2

	MOVQ CX, AX
	SHLQ $1, AX
	LEAQ (R8)(AX*8), SI

	MOVUPD (SI), X0
	MOVUPD 16(SI), X1
	MOVUPD 32(SI), X2
	MOVUPD 48(SI), X3

	VADDPD X2, X0, X4
	VSUBPD X2, X0, X5
	VADDPD X3, X1, X6
	VSUBPD X3, X1, X7

	// (-i)*t3
	VSHUFPD $0x1, X7, X7, X8
	VXORPD ·maskNegHiPD(SB), X8, X8

	// i*t3
	VSHUFPD $0x1, X7, X7, X11
	VXORPD ·maskNegLoPD(SB), X11, X11

	VADDPD X6, X4, X0
	VADDPD X11, X5, X1
	VSUBPD X6, X4, X2
	VADDPD X8, X5, X3

	MOVUPD X0, (SI)
	MOVUPD X1, 16(SI)
	MOVUPD X2, 32(SI)
	MOVUPD X3, 48(SI)

	ADDQ $4, CX
	JMP  size16_r4_inv_128_stage1_loop

size16_r4_inv_128_stage2:
	// ==================================================================
	// Stage 2: 1 group, 4 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ DX, DX

size16_r4_inv_128_stage2_loop:
	CMPQ DX, $4
	JGE  size16_r4_inv_128_scale

	MOVQ DX, BX
	LEAQ 4(DX), SI
	LEAQ 8(DX), DI
	LEAQ 12(DX), R14

	// Twiddle factors: twiddle[j], twiddle[2*j], twiddle[3*j]
	MOVQ DX, R15
	// w1
	MOVQ R15, AX
	SHLQ $1, AX
	MOVUPD (R10)(AX*8), X8
	// w2
	MOVQ R15, AX
	SHLQ $1, AX
	SHLQ $1, AX
	MOVUPD (R10)(AX*8), X9
	// w3
	MOVQ R15, AX
	SHLQ $1, AX
	ADDQ R15, AX
	SHLQ $1, AX
	MOVUPD (R10)(AX*8), X10

	// Load a0..a3
	MOVQ BX, AX
	SHLQ $1, AX
	MOVUPD (R8)(AX*8), X0
	MOVQ SI, AX
	SHLQ $1, AX
	MOVUPD (R8)(AX*8), X1
	MOVQ DI, AX
	SHLQ $1, AX
	MOVUPD (R8)(AX*8), X2
	MOVQ R14, AX
	SHLQ $1, AX
	MOVUPD (R8)(AX*8), X3

	// Conjugate complex multiply a1*w1
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1

	// Conjugate complex multiply a2*w2
	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2

	// Conjugate complex multiply a3*w3
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X3, X13
	VMOVAPD X13, X3

	VADDPD X2, X0, X4
	VSUBPD X2, X0, X5
	VADDPD X3, X1, X6
	VSUBPD X3, X1, X7

	// (-i)*t3
	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegHiPD(SB), X14, X14

	// i*t3
	VSHUFPD $0x1, X7, X7, X12
	VXORPD ·maskNegLoPD(SB), X12, X12

	VADDPD X6, X4, X0
	VADDPD X12, X5, X1
	VSUBPD X6, X4, X2
	VADDPD X14, X5, X3

	// Store
	MOVQ BX, AX
	SHLQ $1, AX
	MOVUPD X0, (R8)(AX*8)
	MOVQ SI, AX
	SHLQ $1, AX
	MOVUPD X1, (R8)(AX*8)
	MOVQ DI, AX
	SHLQ $1, AX
	MOVUPD X2, (R8)(AX*8)
	MOVQ R14, AX
	SHLQ $1, AX
	MOVUPD X3, (R8)(AX*8)

	INCQ DX
	JMP  size16_r4_inv_128_stage2_loop

size16_r4_inv_128_scale:
	// ==================================================================
	// Apply 1/N scaling for inverse transform (1/16)
	// ==================================================================
	MOVQ ·sixteenth64(SB), AX // float64(1.0/16.0)
	VMOVQ AX, X8
	VMOVDDUP X8, X8              // X8 = [s, s]

	XORQ CX, CX

size16_r4_inv_128_scale_loop:
	CMPQ CX, $16
	JGE  size16_r4_inv_128_copy

	MOVQ CX, AX
	SHLQ $1, AX
	MOVUPD (R8)(AX*8), X0
	VMULPD X8, X0, X0
	MOVUPD X0, (R8)(AX*8)

	INCQ CX
	JMP  size16_r4_inv_128_scale_loop

size16_r4_inv_128_copy:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size16_r4_inv_128_done

	XORQ CX, CX

size16_r4_inv_128_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $256
	JL   size16_r4_inv_128_copy_loop

size16_r4_inv_128_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size16_r4_inv_128_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET

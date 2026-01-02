//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-16 Radix-4 FFT Kernels for AMD64
// ===========================================================================
//
// This file contains radix-4 DIT FFT kernels optimized for size 16.
// Size 16 = 4^2, so the radix-4 algorithm uses 2 stages:
//   Stage 1: 4 butterflies, stride=4, twiddle = 1
//   Stage 2: 1 group, 4 butterflies, twiddle step=1
//
// ===========================================================================

#include "textflag.h"

// Forward transform, size 16, complex64, radix-4 variant
TEXT ·ForwardAVX2Size16Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 16)

	// Verify n == 16
	CMPQ R13, $16
	JNE  size16_r4_return_false

	// Validate all slice lengths >= 16
	MOVQ dst+8(FP), AX
	CMPQ AX, $16
	JL   size16_r4_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $16
	JL   size16_r4_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $16
	JL   size16_r4_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $16
	JL   size16_r4_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size16_r4_use_dst
	MOVQ R11, R8             // In-place: use scratch

size16_r4_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 bit-reversal)
	// ==================================================================
	XORQ CX, CX

size16_r4_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $16
	JL   size16_r4_bitrev_loop

size16_r4_stage1:
	// ==================================================================
	// Stage 1: 4 radix-4 butterflies, stride=4
	// ==================================================================
	XORQ CX, CX

size16_r4_stage1_loop:
	CMPQ CX, $16
	JGE  size16_r4_stage2

	LEAQ (R8)(CX*8), SI
	VMOVSD (SI), X0
	VMOVSD 8(SI), X1
	VMOVSD 16(SI), X2
	VMOVSD 24(SI), X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	// (-i)*t3
	VPERMILPS $0xB1, X7, X8
	VXORPS X9, X9, X9
	VSUBPS X8, X9, X10
	VBLENDPS $0x02, X10, X8, X8

	// i*t3
	VPERMILPS $0xB1, X7, X11
	VSUBPS X11, X9, X10
	VBLENDPS $0x01, X10, X11, X11

	VADDPS X4, X6, X0
	VADDPS X5, X8, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X11, X3

	VMOVSD X0, (SI)
	VMOVSD X1, 8(SI)
	VMOVSD X2, 16(SI)
	VMOVSD X3, 24(SI)

	ADDQ $4, CX
	JMP  size16_r4_stage1_loop

size16_r4_stage2:
	// ==================================================================
	// Stage 2: 1 group, 4 butterflies
	// Twiddle step = 1
	// ==================================================================
	XORQ DX, DX

size16_r4_stage2_loop:
	CMPQ DX, $4
	JGE  size16_r4_done

	MOVQ DX, BX
	LEAQ 4(DX), SI
	LEAQ 8(DX), DI
	LEAQ 12(DX), R14

	// Twiddle factors: twiddle[j], twiddle[2*j], twiddle[3*j]
	MOVQ DX, R15
	VMOVSD (R10)(R15*8), X8
	SHLQ $1, R15
	VMOVSD (R10)(R15*8), X9
	ADDQ DX, R15
	VMOVSD (R10)(R15*8), X10

	VMOVSD (R8)(BX*8), X0
	VMOVSD (R8)(SI*8), X1
	VMOVSD (R8)(DI*8), X2
	VMOVSD (R8)(R14*8), X3

	// Complex multiply a1*w1, a2*w2, a3*w3
	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X2, X13
	VMOVAPS X13, X2

	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X3, X13
	VMOVAPS X13, X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	// (-i)*t3
	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14

	// i*t3
	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x01, X11, X12, X12

	VADDPS X4, X6, X0
	VADDPS X5, X14, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X12, X3

	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R14*8)

	INCQ DX
	JMP  size16_r4_stage2_loop

size16_r4_done:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size16_r4_done_direct

	XORQ CX, CX

size16_r4_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $128
	JL   size16_r4_copy_loop

size16_r4_done_direct:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size16_r4_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET

// Inverse transform, size 16, complex64, radix-4 variant
TEXT ·InverseAVX2Size16Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	// Verify n == 16
	CMPQ R13, $16
	JNE  size16_r4_inv_return_false

	// Validate all slice lengths >= 16
	MOVQ dst+8(FP), AX
	CMPQ AX, $16
	JL   size16_r4_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $16
	JL   size16_r4_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $16
	JL   size16_r4_inv_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $16
	JL   size16_r4_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size16_r4_inv_use_dst
	MOVQ R11, R8

size16_r4_inv_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 bit-reversal)
	// ==================================================================
	XORQ CX, CX

size16_r4_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $16
	JL   size16_r4_inv_bitrev_loop

size16_r4_inv_stage1:
	// ==================================================================
	// Stage 1: 4 radix-4 butterflies, stride=4
	// ==================================================================
	XORQ CX, CX

size16_r4_inv_stage1_loop:
	CMPQ CX, $16
	JGE  size16_r4_inv_stage2

	LEAQ (R8)(CX*8), SI
	VMOVSD (SI), X0
	VMOVSD 8(SI), X1
	VMOVSD 16(SI), X2
	VMOVSD 24(SI), X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	// (-i)*t3
	VPERMILPS $0xB1, X7, X8
	VXORPS X9, X9, X9
	VSUBPS X8, X9, X10
	VBLENDPS $0x02, X10, X8, X8

	// i*t3
	VPERMILPS $0xB1, X7, X11
	VSUBPS X11, X9, X10
	VBLENDPS $0x01, X10, X11, X11

	VADDPS X4, X6, X0
	VADDPS X5, X11, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X8, X3

	VMOVSD X0, (SI)
	VMOVSD X1, 8(SI)
	VMOVSD X2, 16(SI)
	VMOVSD X3, 24(SI)

	ADDQ $4, CX
	JMP  size16_r4_inv_stage1_loop

size16_r4_inv_stage2:
	// ==================================================================
	// Stage 2: 1 group, 4 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ DX, DX

size16_r4_inv_stage2_loop:
	CMPQ DX, $4
	JGE  size16_r4_inv_scale

	MOVQ DX, BX
	LEAQ 4(DX), SI
	LEAQ 8(DX), DI
	LEAQ 12(DX), R14

	// Twiddle factors: twiddle[j], twiddle[2*j], twiddle[3*j]
	MOVQ DX, R15
	VMOVSD (R10)(R15*8), X8
	SHLQ $1, R15
	VMOVSD (R10)(R15*8), X9
	ADDQ DX, R15
	VMOVSD (R10)(R15*8), X10

	VMOVSD (R8)(BX*8), X0
	VMOVSD (R8)(SI*8), X1
	VMOVSD (R8)(DI*8), X2
	VMOVSD (R8)(R14*8), X3

	// Conjugate complex multiply
	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMSUBADD231PS X11, X1, X13
	VMOVAPS X13, X1

	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMSUBADD231PS X11, X2, X13
	VMOVAPS X13, X2

	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMSUBADD231PS X11, X3, X13
	VMOVAPS X13, X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	// (-i)*t3
	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14

	// i*t3
	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x01, X11, X12, X12

	VADDPS X4, X6, X0
	VADDPS X5, X12, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X14, X3

	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R14*8)

	INCQ DX
	JMP  size16_r4_inv_stage2_loop

size16_r4_inv_scale:
	// ==================================================================
	// Apply 1/N scaling for inverse transform (1/16)
	// ==================================================================
	MOVL ·sixteenth32(SB), AX         // 1/16 = 0.0625
	MOVD AX, X8
	VBROADCASTSS X8, Y8

	XORQ CX, CX

size16_r4_inv_scale_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMULPS Y8, Y0, Y0
	VMOVUPS Y0, (R8)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $128
	JL   size16_r4_inv_scale_loop

	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size16_r4_inv_done

	XORQ CX, CX

size16_r4_inv_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $128
	JL   size16_r4_inv_copy_loop

size16_r4_inv_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size16_r4_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET

// Forward transform, size 16, complex128, radix-4 variant

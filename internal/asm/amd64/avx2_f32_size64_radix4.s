//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-64 Radix-4 FFT Kernels for AMD64
// ===========================================================================
//
// Size 64 = 4^3, so the radix-4 algorithm uses 3 stages:
//   Stage 1: 16 butterflies, stride=4,  twiddle = 1
//   Stage 2: 4 groups × 4 butterflies, stride=16, twiddle step=4
//   Stage 3: 1 group × 16 butterflies, twiddle step=1
//
// ===========================================================================

#include "textflag.h"

// Forward transform, size 64, complex64, radix-4 variant
TEXT ·ForwardAVX2Size64Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 64)

	// Verify n == 64
	CMPQ R13, $64
	JNE  r4_64_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $64
	JL   r4_64_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $64
	JL   r4_64_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $64
	JL   r4_64_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  r4_64_use_dst
	MOVQ R11, R8             // In-place: use scratch

r4_64_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 bit-reversal)
	// ==================================================================
	LEAQ ·bitrev64_r4(SB), R12
	XORQ CX, CX

r4_64_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $64
	JL   r4_64_bitrev_loop

r4_64_stage1:
	// ==================================================================
	// Stage 1: 16 radix-4 butterflies, stride=4
	// ==================================================================
	XORQ CX, CX

r4_64_stage1_loop:
	CMPQ CX, $64
	JGE  r4_64_stage2

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
	JMP  r4_64_stage1_loop

r4_64_stage2:
	// ==================================================================
	// Stage 2: 4 groups × 4 butterflies, twiddle step=4
	// ==================================================================
	XORQ CX, CX

r4_64_stage2_outer:
	CMPQ CX, $64
	JGE  r4_64_stage3

	XORQ DX, DX

r4_64_stage2_inner:
	CMPQ DX, $4
	JGE  r4_64_stage2_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 4(BX), SI
	LEAQ 8(BX), DI
	LEAQ 12(BX), R14

	// Twiddle factors: w1 = twiddle[j*4], w2 = twiddle[2*j*4], w3 = twiddle[3*j*4]
	MOVQ DX, R15
	SHLQ $2, R15             // j*4
	VMOVSD (R10)(R15*8), X8

	MOVQ R15, R13
	SHLQ $1, R15             // 2*j*4
	VMOVSD (R10)(R15*8), X9

	ADDQ R13, R15            // 3*j*4
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
	JMP  r4_64_stage2_inner

r4_64_stage2_next:
	ADDQ $16, CX
	JMP  r4_64_stage2_outer

r4_64_stage3:
	// ==================================================================
	// Stage 3: 1 group × 16 butterflies, twiddle step=1
	// ==================================================================
	XORQ DX, DX

r4_64_stage3_loop:
	CMPQ DX, $16
	JGE  r4_64_done

	MOVQ DX, BX
	LEAQ 16(DX), SI
	LEAQ 32(DX), DI
	LEAQ 48(DX), R14

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
	JMP  r4_64_stage3_loop

r4_64_done:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r4_64_done_direct

	XORQ CX, CX

r4_64_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $512
	JL   r4_64_copy_loop

r4_64_done_direct:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

r4_64_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 64, complex64, radix-4
// ===========================================================================
TEXT ·InverseAVX2Size64Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 64)

	// Verify n == 64
	CMPQ R13, $64
	JNE  r4_64_inv_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $64
	JL   r4_64_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $64
	JL   r4_64_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $64
	JL   r4_64_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  r4_64_inv_use_dst
	MOVQ R11, R8             // In-place: use scratch

r4_64_inv_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 bit-reversal)
	// ==================================================================
	LEAQ ·bitrev64_r4(SB), R12
	XORQ CX, CX

r4_64_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $64
	JL   r4_64_inv_bitrev_loop

r4_64_inv_stage1:
	// ==================================================================
	// Stage 1: 16 radix-4 butterflies, stride=4
	// ==================================================================
	XORQ CX, CX

r4_64_inv_stage1_loop:
	CMPQ CX, $64
	JGE  r4_64_inv_stage2

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
	JMP  r4_64_inv_stage1_loop

r4_64_inv_stage2:
	// ==================================================================
	// Stage 2: 4 groups × 4 butterflies, twiddle step=4 (conjugated)
	// ==================================================================
	XORQ CX, CX

r4_64_inv_stage2_outer:
	CMPQ CX, $64
	JGE  r4_64_inv_stage3

	XORQ DX, DX

r4_64_inv_stage2_inner:
	CMPQ DX, $4
	JGE  r4_64_inv_stage2_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 4(BX), SI
	LEAQ 8(BX), DI
	LEAQ 12(BX), R14

	// Twiddle factors: w1 = twiddle[j*4], w2 = twiddle[2*j*4], w3 = twiddle[3*j*4]
	MOVQ DX, R15
	SHLQ $2, R15
	VMOVSD (R10)(R15*8), X8

	MOVQ R15, R13
	SHLQ $1, R15
	VMOVSD (R10)(R15*8), X9

	ADDQ R13, R15
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
	JMP  r4_64_inv_stage2_inner

r4_64_inv_stage2_next:
	ADDQ $16, CX
	JMP  r4_64_inv_stage2_outer

r4_64_inv_stage3:
	// ==================================================================
	// Stage 3: 1 group × 16 butterflies, twiddle step=1 (conjugated)
	// ==================================================================
	XORQ DX, DX

r4_64_inv_stage3_loop:
	CMPQ DX, $16
	JGE  r4_64_inv_scale

	MOVQ DX, BX
	LEAQ 16(DX), SI
	LEAQ 32(DX), DI
	LEAQ 48(DX), R14

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
	JMP  r4_64_inv_stage3_loop

r4_64_inv_scale:
	// ==================================================================
	// Apply 1/N scaling for inverse transform (1/64)
	// ==================================================================
	MOVL ·sixtyFourth32(SB), AX         // 1/64 = 0.015625
	MOVD AX, X8
	VBROADCASTSS X8, Y8

	XORQ CX, CX

r4_64_inv_scale_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMULPS Y8, Y0, Y0
	VMOVUPS Y0, (R8)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $512
	JL   r4_64_inv_scale_loop

	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r4_64_inv_done

	XORQ CX, CX

r4_64_inv_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $512
	JL   r4_64_inv_copy_loop

r4_64_inv_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

r4_64_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// DATA SECTION: Bit-reversal indices for size 64 radix-4
// ===========================================================================

DATA ·bitrev64_r4+0(SB)/8, $0
DATA ·bitrev64_r4+8(SB)/8, $16
DATA ·bitrev64_r4+16(SB)/8, $32
DATA ·bitrev64_r4+24(SB)/8, $48
DATA ·bitrev64_r4+32(SB)/8, $4
DATA ·bitrev64_r4+40(SB)/8, $20
DATA ·bitrev64_r4+48(SB)/8, $36
DATA ·bitrev64_r4+56(SB)/8, $52
DATA ·bitrev64_r4+64(SB)/8, $8
DATA ·bitrev64_r4+72(SB)/8, $24
DATA ·bitrev64_r4+80(SB)/8, $40
DATA ·bitrev64_r4+88(SB)/8, $56
DATA ·bitrev64_r4+96(SB)/8, $12
DATA ·bitrev64_r4+104(SB)/8, $28
DATA ·bitrev64_r4+112(SB)/8, $44
DATA ·bitrev64_r4+120(SB)/8, $60
DATA ·bitrev64_r4+128(SB)/8, $1
DATA ·bitrev64_r4+136(SB)/8, $17
DATA ·bitrev64_r4+144(SB)/8, $33
DATA ·bitrev64_r4+152(SB)/8, $49
DATA ·bitrev64_r4+160(SB)/8, $5
DATA ·bitrev64_r4+168(SB)/8, $21
DATA ·bitrev64_r4+176(SB)/8, $37
DATA ·bitrev64_r4+184(SB)/8, $53
DATA ·bitrev64_r4+192(SB)/8, $9
DATA ·bitrev64_r4+200(SB)/8, $25
DATA ·bitrev64_r4+208(SB)/8, $41
DATA ·bitrev64_r4+216(SB)/8, $57
DATA ·bitrev64_r4+224(SB)/8, $13
DATA ·bitrev64_r4+232(SB)/8, $29
DATA ·bitrev64_r4+240(SB)/8, $45
DATA ·bitrev64_r4+248(SB)/8, $61
DATA ·bitrev64_r4+256(SB)/8, $2
DATA ·bitrev64_r4+264(SB)/8, $18
DATA ·bitrev64_r4+272(SB)/8, $34
DATA ·bitrev64_r4+280(SB)/8, $50
DATA ·bitrev64_r4+288(SB)/8, $6
DATA ·bitrev64_r4+296(SB)/8, $22
DATA ·bitrev64_r4+304(SB)/8, $38
DATA ·bitrev64_r4+312(SB)/8, $54
DATA ·bitrev64_r4+320(SB)/8, $10
DATA ·bitrev64_r4+328(SB)/8, $26
DATA ·bitrev64_r4+336(SB)/8, $42
DATA ·bitrev64_r4+344(SB)/8, $58
DATA ·bitrev64_r4+352(SB)/8, $14
DATA ·bitrev64_r4+360(SB)/8, $30
DATA ·bitrev64_r4+368(SB)/8, $46
DATA ·bitrev64_r4+376(SB)/8, $62
DATA ·bitrev64_r4+384(SB)/8, $3
DATA ·bitrev64_r4+392(SB)/8, $19
DATA ·bitrev64_r4+400(SB)/8, $35
DATA ·bitrev64_r4+408(SB)/8, $51
DATA ·bitrev64_r4+416(SB)/8, $7
DATA ·bitrev64_r4+424(SB)/8, $23
DATA ·bitrev64_r4+432(SB)/8, $39
DATA ·bitrev64_r4+440(SB)/8, $55
DATA ·bitrev64_r4+448(SB)/8, $11
DATA ·bitrev64_r4+456(SB)/8, $27
DATA ·bitrev64_r4+464(SB)/8, $43
DATA ·bitrev64_r4+472(SB)/8, $59
DATA ·bitrev64_r4+480(SB)/8, $15
DATA ·bitrev64_r4+488(SB)/8, $31
DATA ·bitrev64_r4+496(SB)/8, $47
DATA ·bitrev64_r4+504(SB)/8, $63
GLOBL ·bitrev64_r4(SB), RODATA, $512

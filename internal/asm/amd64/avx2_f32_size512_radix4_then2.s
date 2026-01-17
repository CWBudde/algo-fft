//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-512 Radix-4-then-2 FFT Kernels for AMD64
// ===========================================================================
//
// This file contains a radix-4-then-2 DIT FFT optimized for size 512.
// Stages:
//   - Stage 1-4: radix-4 (4 stages)
//   - Stage 5: radix-2 (final combine)
//
// Mixed-radix bit-reversal indices are required for stage 1.
//
// ===========================================================================

#include "textflag.h"

TEXT ·ForwardAVX2Size512Radix4Then2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 512)
	LEAQ ·bitrev512_m24(SB), R12

	// Verify n == 512
	CMPQ R13, $512
	JNE  m24_512_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $512
	JL   m24_512_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $512
	JL   m24_512_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $512
	JL   m24_512_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  m24_512_use_dst
	MOVQ R11, R8             // In-place: use scratch

m24_512_use_dst:
	// ==================================================================
	// Stage 1: 128 radix-4 butterflies with radix-4-then-2 bit-reversal
	// ==================================================================
	XORQ CX, CX              // CX = base offset

m24_512_stage1_loop:
	CMPQ CX, $512
	JGE  m24_512_stage2

	// Load bit-reversed indices
	MOVQ (R12)(CX*8), DX
	MOVQ 8(R12)(CX*8), SI
	MOVQ 16(R12)(CX*8), DI
	MOVQ 24(R12)(CX*8), R14

	// Load input values
	VMOVSD (R9)(DX*8), X0
	VMOVSD (R9)(SI*8), X1
	VMOVSD (R9)(DI*8), X2
	VMOVSD (R9)(R14*8), X3

	// Radix-4 butterfly
	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	// (-i)*t3 for y1
	VPERMILPS $0xB1, X7, X8
	VXORPS X9, X9, X9
	VSUBPS X8, X9, X10
	VBLENDPS $0x02, X10, X8, X8

	// i*t3 for y3
	VPERMILPS $0xB1, X7, X11
	VSUBPS X11, X9, X10
	VBLENDPS $0x01, X10, X11, X11

	VADDPS X4, X6, X0
	VADDPS X5, X8, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X11, X3

	// Store outputs to work buffer
	VMOVSD X0, (R8)(CX*8)
	VMOVSD X1, 8(R8)(CX*8)
	VMOVSD X2, 16(R8)(CX*8)
	VMOVSD X3, 24(R8)(CX*8)

	ADDQ $4, CX
	JMP  m24_512_stage1_loop

m24_512_stage2:
	// ==================================================================
	// Stage 2: 32 groups, each with 4 butterflies
	// Twiddle step = 32
	// ==================================================================
	XORQ CX, CX

m24_512_stage2_outer:
	CMPQ CX, $512
	JGE  m24_512_stage3

	XORQ DX, DX

m24_512_stage2_inner:
	CMPQ DX, $4
	JGE  m24_512_stage2_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 4(BX), SI
	LEAQ 8(BX), DI
	LEAQ 12(BX), R14

	// Twiddles: j*32, 2*j*32, 3*j*32
	MOVQ DX, R15
	SHLQ $5, R15
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

	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14

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
	JMP  m24_512_stage2_inner

m24_512_stage2_next:
	ADDQ $16, CX
	JMP  m24_512_stage2_outer

m24_512_stage3:
	// ==================================================================
	// Stage 3: 8 groups, each with 16 butterflies
	// Twiddle step = 8
	// ==================================================================
	XORQ CX, CX

m24_512_stage3_outer:
	CMPQ CX, $512
	JGE  m24_512_stage4

	XORQ DX, DX

m24_512_stage3_inner:
	CMPQ DX, $16
	JGE  m24_512_stage3_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 16(BX), SI
	LEAQ 32(BX), DI
	LEAQ 48(BX), R14

	MOVQ DX, R15
	SHLQ $3, R15
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

	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14

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
	JMP  m24_512_stage3_inner

m24_512_stage3_next:
	ADDQ $64, CX
	JMP  m24_512_stage3_outer

m24_512_stage4:
	// ==================================================================
	// Stage 4: 2 groups, each with 64 butterflies
	// Twiddle step = 2
	// ==================================================================
	XORQ CX, CX

m24_512_stage4_outer:
	CMPQ CX, $512
	JGE  m24_512_stage5

	XORQ DX, DX

m24_512_stage4_inner:
	CMPQ DX, $64
	JGE  m24_512_stage4_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 64(BX), SI
	LEAQ 128(BX), DI
	LEAQ 192(BX), R14

	MOVQ DX, R15
	SHLQ $1, R15
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

	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14

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
	JMP  m24_512_stage4_inner

m24_512_stage4_next:
	ADDQ $256, CX
	JMP  m24_512_stage4_outer

m24_512_stage5:
	// ==================================================================
	// Stage 5: radix-2 final stage
	// ==================================================================
	XORQ CX, CX

m24_512_stage5_loop:
	CMPQ CX, $256
	JGE  m24_512_forward_done

	MOVQ CX, BX
	LEAQ 256(BX), SI

	VMOVSD (R10)(CX*8), X8   // twiddle
	VMOVSD (R8)(BX*8), X0    // a
	VMOVSD (R8)(SI*8), X1    // b

	// b = b * w
	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	VADDPS X0, X1, X2
	VSUBPS X1, X0, X3

	VMOVSD X2, (R8)(BX*8)
	VMOVSD X3, (R8)(SI*8)

	INCQ CX
	JMP  m24_512_stage5_loop

m24_512_forward_done:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   m24_512_forward_ret

	XORQ CX, CX

m24_512_forward_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMOVUPS Y0, (R9)(CX*1)
	VMOVUPS Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $4096
	JL   m24_512_forward_copy_loop

m24_512_forward_ret:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

m24_512_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 512, complex64, radix-4-then-2
// ===========================================================================
TEXT ·InverseAVX2Size512Radix4Then2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 512)
	LEAQ ·bitrev512_m24(SB), R12

	// Verify n == 512
	CMPQ R13, $512
	JNE  m24_512_inv_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $512
	JL   m24_512_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $512
	JL   m24_512_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $512
	JL   m24_512_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  m24_512_inv_use_dst
	MOVQ R11, R8             // In-place: use scratch

m24_512_inv_use_dst:
	// ==================================================================
	// Stage 1: 128 radix-4 butterflies with radix-4-then-2 bit-reversal
	// ==================================================================
	XORQ CX, CX

m24_512_inv_stage1_loop:
	CMPQ CX, $512
	JGE  m24_512_inv_stage2

	MOVQ (R12)(CX*8), DX
	MOVQ 8(R12)(CX*8), SI
	MOVQ 16(R12)(CX*8), DI
	MOVQ 24(R12)(CX*8), R14

	VMOVSD (R9)(DX*8), X0
	VMOVSD (R9)(SI*8), X1
	VMOVSD (R9)(DI*8), X2
	VMOVSD (R9)(R14*8), X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	// i*t3 for y1
	VPERMILPS $0xB1, X7, X8
	VXORPS X9, X9, X9
	VSUBPS X8, X9, X10
	VBLENDPS $0x01, X10, X8, X8

	// (-i)*t3 for y3
	VPERMILPS $0xB1, X7, X11
	VSUBPS X11, X9, X10
	VBLENDPS $0x02, X10, X11, X11

	VADDPS X4, X6, X0
	VADDPS X5, X8, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X11, X3

	VMOVSD X0, (R8)(CX*8)
	VMOVSD X1, 8(R8)(CX*8)
	VMOVSD X2, 16(R8)(CX*8)
	VMOVSD X3, 24(R8)(CX*8)

	ADDQ $4, CX
	JMP  m24_512_inv_stage1_loop

m24_512_inv_stage2:
	// ==================================================================
	// Stage 2: 32 groups, each with 4 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ CX, CX

m24_512_inv_stage2_outer:
	CMPQ CX, $512
	JGE  m24_512_inv_stage3

	XORQ DX, DX

m24_512_inv_stage2_inner:
	CMPQ DX, $4
	JGE  m24_512_inv_stage2_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 4(BX), SI
	LEAQ 8(BX), DI
	LEAQ 12(BX), R14

	MOVQ DX, R15
	SHLQ $5, R15
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

	// Conjugated complex multiply
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

	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x01, X11, X14, X14

	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x02, X11, X12, X12

	VADDPS X4, X6, X0
	VADDPS X5, X14, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X12, X3

	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R14*8)

	INCQ DX
	JMP  m24_512_inv_stage2_inner

m24_512_inv_stage2_next:
	ADDQ $16, CX
	JMP  m24_512_inv_stage2_outer

m24_512_inv_stage3:
	// ==================================================================
	// Stage 3: 8 groups, each with 16 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ CX, CX

m24_512_inv_stage3_outer:
	CMPQ CX, $512
	JGE  m24_512_inv_stage4

	XORQ DX, DX

m24_512_inv_stage3_inner:
	CMPQ DX, $16
	JGE  m24_512_inv_stage3_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 16(BX), SI
	LEAQ 32(BX), DI
	LEAQ 48(BX), R14

	MOVQ DX, R15
	SHLQ $3, R15
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

	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x01, X11, X14, X14

	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x02, X11, X12, X12

	VADDPS X4, X6, X0
	VADDPS X5, X14, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X12, X3

	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R14*8)

	INCQ DX
	JMP  m24_512_inv_stage3_inner

m24_512_inv_stage3_next:
	ADDQ $64, CX
	JMP  m24_512_inv_stage3_outer

m24_512_inv_stage4:
	// ==================================================================
	// Stage 4: 2 groups, each with 64 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ CX, CX

m24_512_inv_stage4_outer:
	CMPQ CX, $512
	JGE  m24_512_inv_stage5

	XORQ DX, DX

m24_512_inv_stage4_inner:
	CMPQ DX, $64
	JGE  m24_512_inv_stage4_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 64(BX), SI
	LEAQ 128(BX), DI
	LEAQ 192(BX), R14

	MOVQ DX, R15
	SHLQ $1, R15
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

	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x01, X11, X14, X14

	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x02, X11, X12, X12

	VADDPS X4, X6, X0
	VADDPS X5, X14, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X12, X3

	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R14*8)

	INCQ DX
	JMP  m24_512_inv_stage4_inner

m24_512_inv_stage4_next:
	ADDQ $256, CX
	JMP  m24_512_inv_stage4_outer

m24_512_inv_stage5:
	// ==================================================================
	// Stage 5: radix-2 final stage (conjugated twiddles)
	// ==================================================================
	XORQ CX, CX

m24_512_inv_stage5_loop:
	CMPQ CX, $256
	JGE  m24_512_inv_scale

	MOVQ CX, BX
	LEAQ 256(BX), SI

	VMOVSD (R10)(CX*8), X8
	VMOVSD (R8)(BX*8), X0
	VMOVSD (R8)(SI*8), X1

	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMSUBADD231PS X11, X1, X13
	VMOVAPS X13, X1

	VADDPS X0, X1, X2
	VSUBPS X1, X0, X3

	VMOVSD X2, (R8)(BX*8)
	VMOVSD X3, (R8)(SI*8)

	INCQ CX
	JMP  m24_512_inv_stage5_loop

m24_512_inv_scale:
	// ==================================================================
	// Apply 1/N scaling for inverse transform
	// ==================================================================
	MOVL ·fiveHundredTwelfth32(SB), AX
	MOVD AX, X8
	VBROADCASTSS X8, Y8

	XORQ CX, CX

m24_512_inv_scale_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, (R8)(CX*1)
	VMOVUPS Y1, 32(R8)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $4096
	JL   m24_512_inv_scale_loop

	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   m24_512_inv_done

	XORQ CX, CX

m24_512_inv_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMOVUPS Y0, (R9)(CX*1)
	VMOVUPS Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $4096
	JL   m24_512_inv_copy_loop

m24_512_inv_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

m24_512_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET


DATA ·bitrev512_m24+0(SB)/8, $0
DATA ·bitrev512_m24+8(SB)/8, $128
DATA ·bitrev512_m24+16(SB)/8, $256
DATA ·bitrev512_m24+24(SB)/8, $384
DATA ·bitrev512_m24+32(SB)/8, $32
DATA ·bitrev512_m24+40(SB)/8, $160
DATA ·bitrev512_m24+48(SB)/8, $288
DATA ·bitrev512_m24+56(SB)/8, $416
DATA ·bitrev512_m24+64(SB)/8, $64
DATA ·bitrev512_m24+72(SB)/8, $192
DATA ·bitrev512_m24+80(SB)/8, $320
DATA ·bitrev512_m24+88(SB)/8, $448
DATA ·bitrev512_m24+96(SB)/8, $96
DATA ·bitrev512_m24+104(SB)/8, $224
DATA ·bitrev512_m24+112(SB)/8, $352
DATA ·bitrev512_m24+120(SB)/8, $480
DATA ·bitrev512_m24+128(SB)/8, $8
DATA ·bitrev512_m24+136(SB)/8, $136
DATA ·bitrev512_m24+144(SB)/8, $264
DATA ·bitrev512_m24+152(SB)/8, $392
DATA ·bitrev512_m24+160(SB)/8, $40
DATA ·bitrev512_m24+168(SB)/8, $168
DATA ·bitrev512_m24+176(SB)/8, $296
DATA ·bitrev512_m24+184(SB)/8, $424
DATA ·bitrev512_m24+192(SB)/8, $72
DATA ·bitrev512_m24+200(SB)/8, $200
DATA ·bitrev512_m24+208(SB)/8, $328
DATA ·bitrev512_m24+216(SB)/8, $456
DATA ·bitrev512_m24+224(SB)/8, $104
DATA ·bitrev512_m24+232(SB)/8, $232
DATA ·bitrev512_m24+240(SB)/8, $360
DATA ·bitrev512_m24+248(SB)/8, $488
DATA ·bitrev512_m24+256(SB)/8, $16
DATA ·bitrev512_m24+264(SB)/8, $144
DATA ·bitrev512_m24+272(SB)/8, $272
DATA ·bitrev512_m24+280(SB)/8, $400
DATA ·bitrev512_m24+288(SB)/8, $48
DATA ·bitrev512_m24+296(SB)/8, $176
DATA ·bitrev512_m24+304(SB)/8, $304
DATA ·bitrev512_m24+312(SB)/8, $432
DATA ·bitrev512_m24+320(SB)/8, $80
DATA ·bitrev512_m24+328(SB)/8, $208
DATA ·bitrev512_m24+336(SB)/8, $336
DATA ·bitrev512_m24+344(SB)/8, $464
DATA ·bitrev512_m24+352(SB)/8, $112
DATA ·bitrev512_m24+360(SB)/8, $240
DATA ·bitrev512_m24+368(SB)/8, $368
DATA ·bitrev512_m24+376(SB)/8, $496
DATA ·bitrev512_m24+384(SB)/8, $24
DATA ·bitrev512_m24+392(SB)/8, $152
DATA ·bitrev512_m24+400(SB)/8, $280
DATA ·bitrev512_m24+408(SB)/8, $408
DATA ·bitrev512_m24+416(SB)/8, $56
DATA ·bitrev512_m24+424(SB)/8, $184
DATA ·bitrev512_m24+432(SB)/8, $312
DATA ·bitrev512_m24+440(SB)/8, $440
DATA ·bitrev512_m24+448(SB)/8, $88
DATA ·bitrev512_m24+456(SB)/8, $216
DATA ·bitrev512_m24+464(SB)/8, $344
DATA ·bitrev512_m24+472(SB)/8, $472
DATA ·bitrev512_m24+480(SB)/8, $120
DATA ·bitrev512_m24+488(SB)/8, $248
DATA ·bitrev512_m24+496(SB)/8, $376
DATA ·bitrev512_m24+504(SB)/8, $504
DATA ·bitrev512_m24+512(SB)/8, $2
DATA ·bitrev512_m24+520(SB)/8, $130
DATA ·bitrev512_m24+528(SB)/8, $258
DATA ·bitrev512_m24+536(SB)/8, $386
DATA ·bitrev512_m24+544(SB)/8, $34
DATA ·bitrev512_m24+552(SB)/8, $162
DATA ·bitrev512_m24+560(SB)/8, $290
DATA ·bitrev512_m24+568(SB)/8, $418
DATA ·bitrev512_m24+576(SB)/8, $66
DATA ·bitrev512_m24+584(SB)/8, $194
DATA ·bitrev512_m24+592(SB)/8, $322
DATA ·bitrev512_m24+600(SB)/8, $450
DATA ·bitrev512_m24+608(SB)/8, $98
DATA ·bitrev512_m24+616(SB)/8, $226
DATA ·bitrev512_m24+624(SB)/8, $354
DATA ·bitrev512_m24+632(SB)/8, $482
DATA ·bitrev512_m24+640(SB)/8, $10
DATA ·bitrev512_m24+648(SB)/8, $138
DATA ·bitrev512_m24+656(SB)/8, $266
DATA ·bitrev512_m24+664(SB)/8, $394
DATA ·bitrev512_m24+672(SB)/8, $42
DATA ·bitrev512_m24+680(SB)/8, $170
DATA ·bitrev512_m24+688(SB)/8, $298
DATA ·bitrev512_m24+696(SB)/8, $426
DATA ·bitrev512_m24+704(SB)/8, $74
DATA ·bitrev512_m24+712(SB)/8, $202
DATA ·bitrev512_m24+720(SB)/8, $330
DATA ·bitrev512_m24+728(SB)/8, $458
DATA ·bitrev512_m24+736(SB)/8, $106
DATA ·bitrev512_m24+744(SB)/8, $234
DATA ·bitrev512_m24+752(SB)/8, $362
DATA ·bitrev512_m24+760(SB)/8, $490
DATA ·bitrev512_m24+768(SB)/8, $18
DATA ·bitrev512_m24+776(SB)/8, $146
DATA ·bitrev512_m24+784(SB)/8, $274
DATA ·bitrev512_m24+792(SB)/8, $402
DATA ·bitrev512_m24+800(SB)/8, $50
DATA ·bitrev512_m24+808(SB)/8, $178
DATA ·bitrev512_m24+816(SB)/8, $306
DATA ·bitrev512_m24+824(SB)/8, $434
DATA ·bitrev512_m24+832(SB)/8, $82
DATA ·bitrev512_m24+840(SB)/8, $210
DATA ·bitrev512_m24+848(SB)/8, $338
DATA ·bitrev512_m24+856(SB)/8, $466
DATA ·bitrev512_m24+864(SB)/8, $114
DATA ·bitrev512_m24+872(SB)/8, $242
DATA ·bitrev512_m24+880(SB)/8, $370
DATA ·bitrev512_m24+888(SB)/8, $498
DATA ·bitrev512_m24+896(SB)/8, $26
DATA ·bitrev512_m24+904(SB)/8, $154
DATA ·bitrev512_m24+912(SB)/8, $282
DATA ·bitrev512_m24+920(SB)/8, $410
DATA ·bitrev512_m24+928(SB)/8, $58
DATA ·bitrev512_m24+936(SB)/8, $186
DATA ·bitrev512_m24+944(SB)/8, $314
DATA ·bitrev512_m24+952(SB)/8, $442
DATA ·bitrev512_m24+960(SB)/8, $90
DATA ·bitrev512_m24+968(SB)/8, $218
DATA ·bitrev512_m24+976(SB)/8, $346
DATA ·bitrev512_m24+984(SB)/8, $474
DATA ·bitrev512_m24+992(SB)/8, $122
DATA ·bitrev512_m24+1000(SB)/8, $250
DATA ·bitrev512_m24+1008(SB)/8, $378
DATA ·bitrev512_m24+1016(SB)/8, $506
DATA ·bitrev512_m24+1024(SB)/8, $4
DATA ·bitrev512_m24+1032(SB)/8, $132
DATA ·bitrev512_m24+1040(SB)/8, $260
DATA ·bitrev512_m24+1048(SB)/8, $388
DATA ·bitrev512_m24+1056(SB)/8, $36
DATA ·bitrev512_m24+1064(SB)/8, $164
DATA ·bitrev512_m24+1072(SB)/8, $292
DATA ·bitrev512_m24+1080(SB)/8, $420
DATA ·bitrev512_m24+1088(SB)/8, $68
DATA ·bitrev512_m24+1096(SB)/8, $196
DATA ·bitrev512_m24+1104(SB)/8, $324
DATA ·bitrev512_m24+1112(SB)/8, $452
DATA ·bitrev512_m24+1120(SB)/8, $100
DATA ·bitrev512_m24+1128(SB)/8, $228
DATA ·bitrev512_m24+1136(SB)/8, $356
DATA ·bitrev512_m24+1144(SB)/8, $484
DATA ·bitrev512_m24+1152(SB)/8, $12
DATA ·bitrev512_m24+1160(SB)/8, $140
DATA ·bitrev512_m24+1168(SB)/8, $268
DATA ·bitrev512_m24+1176(SB)/8, $396
DATA ·bitrev512_m24+1184(SB)/8, $44
DATA ·bitrev512_m24+1192(SB)/8, $172
DATA ·bitrev512_m24+1200(SB)/8, $300
DATA ·bitrev512_m24+1208(SB)/8, $428
DATA ·bitrev512_m24+1216(SB)/8, $76
DATA ·bitrev512_m24+1224(SB)/8, $204
DATA ·bitrev512_m24+1232(SB)/8, $332
DATA ·bitrev512_m24+1240(SB)/8, $460
DATA ·bitrev512_m24+1248(SB)/8, $108
DATA ·bitrev512_m24+1256(SB)/8, $236
DATA ·bitrev512_m24+1264(SB)/8, $364
DATA ·bitrev512_m24+1272(SB)/8, $492
DATA ·bitrev512_m24+1280(SB)/8, $20
DATA ·bitrev512_m24+1288(SB)/8, $148
DATA ·bitrev512_m24+1296(SB)/8, $276
DATA ·bitrev512_m24+1304(SB)/8, $404
DATA ·bitrev512_m24+1312(SB)/8, $52
DATA ·bitrev512_m24+1320(SB)/8, $180
DATA ·bitrev512_m24+1328(SB)/8, $308
DATA ·bitrev512_m24+1336(SB)/8, $436
DATA ·bitrev512_m24+1344(SB)/8, $84
DATA ·bitrev512_m24+1352(SB)/8, $212
DATA ·bitrev512_m24+1360(SB)/8, $340
DATA ·bitrev512_m24+1368(SB)/8, $468
DATA ·bitrev512_m24+1376(SB)/8, $116
DATA ·bitrev512_m24+1384(SB)/8, $244
DATA ·bitrev512_m24+1392(SB)/8, $372
DATA ·bitrev512_m24+1400(SB)/8, $500
DATA ·bitrev512_m24+1408(SB)/8, $28
DATA ·bitrev512_m24+1416(SB)/8, $156
DATA ·bitrev512_m24+1424(SB)/8, $284
DATA ·bitrev512_m24+1432(SB)/8, $412
DATA ·bitrev512_m24+1440(SB)/8, $60
DATA ·bitrev512_m24+1448(SB)/8, $188
DATA ·bitrev512_m24+1456(SB)/8, $316
DATA ·bitrev512_m24+1464(SB)/8, $444
DATA ·bitrev512_m24+1472(SB)/8, $92
DATA ·bitrev512_m24+1480(SB)/8, $220
DATA ·bitrev512_m24+1488(SB)/8, $348
DATA ·bitrev512_m24+1496(SB)/8, $476
DATA ·bitrev512_m24+1504(SB)/8, $124
DATA ·bitrev512_m24+1512(SB)/8, $252
DATA ·bitrev512_m24+1520(SB)/8, $380
DATA ·bitrev512_m24+1528(SB)/8, $508
DATA ·bitrev512_m24+1536(SB)/8, $6
DATA ·bitrev512_m24+1544(SB)/8, $134
DATA ·bitrev512_m24+1552(SB)/8, $262
DATA ·bitrev512_m24+1560(SB)/8, $390
DATA ·bitrev512_m24+1568(SB)/8, $38
DATA ·bitrev512_m24+1576(SB)/8, $166
DATA ·bitrev512_m24+1584(SB)/8, $294
DATA ·bitrev512_m24+1592(SB)/8, $422
DATA ·bitrev512_m24+1600(SB)/8, $70
DATA ·bitrev512_m24+1608(SB)/8, $198
DATA ·bitrev512_m24+1616(SB)/8, $326
DATA ·bitrev512_m24+1624(SB)/8, $454
DATA ·bitrev512_m24+1632(SB)/8, $102
DATA ·bitrev512_m24+1640(SB)/8, $230
DATA ·bitrev512_m24+1648(SB)/8, $358
DATA ·bitrev512_m24+1656(SB)/8, $486
DATA ·bitrev512_m24+1664(SB)/8, $14
DATA ·bitrev512_m24+1672(SB)/8, $142
DATA ·bitrev512_m24+1680(SB)/8, $270
DATA ·bitrev512_m24+1688(SB)/8, $398
DATA ·bitrev512_m24+1696(SB)/8, $46
DATA ·bitrev512_m24+1704(SB)/8, $174
DATA ·bitrev512_m24+1712(SB)/8, $302
DATA ·bitrev512_m24+1720(SB)/8, $430
DATA ·bitrev512_m24+1728(SB)/8, $78
DATA ·bitrev512_m24+1736(SB)/8, $206
DATA ·bitrev512_m24+1744(SB)/8, $334
DATA ·bitrev512_m24+1752(SB)/8, $462
DATA ·bitrev512_m24+1760(SB)/8, $110
DATA ·bitrev512_m24+1768(SB)/8, $238
DATA ·bitrev512_m24+1776(SB)/8, $366
DATA ·bitrev512_m24+1784(SB)/8, $494
DATA ·bitrev512_m24+1792(SB)/8, $22
DATA ·bitrev512_m24+1800(SB)/8, $150
DATA ·bitrev512_m24+1808(SB)/8, $278
DATA ·bitrev512_m24+1816(SB)/8, $406
DATA ·bitrev512_m24+1824(SB)/8, $54
DATA ·bitrev512_m24+1832(SB)/8, $182
DATA ·bitrev512_m24+1840(SB)/8, $310
DATA ·bitrev512_m24+1848(SB)/8, $438
DATA ·bitrev512_m24+1856(SB)/8, $86
DATA ·bitrev512_m24+1864(SB)/8, $214
DATA ·bitrev512_m24+1872(SB)/8, $342
DATA ·bitrev512_m24+1880(SB)/8, $470
DATA ·bitrev512_m24+1888(SB)/8, $118
DATA ·bitrev512_m24+1896(SB)/8, $246
DATA ·bitrev512_m24+1904(SB)/8, $374
DATA ·bitrev512_m24+1912(SB)/8, $502
DATA ·bitrev512_m24+1920(SB)/8, $30
DATA ·bitrev512_m24+1928(SB)/8, $158
DATA ·bitrev512_m24+1936(SB)/8, $286
DATA ·bitrev512_m24+1944(SB)/8, $414
DATA ·bitrev512_m24+1952(SB)/8, $62
DATA ·bitrev512_m24+1960(SB)/8, $190
DATA ·bitrev512_m24+1968(SB)/8, $318
DATA ·bitrev512_m24+1976(SB)/8, $446
DATA ·bitrev512_m24+1984(SB)/8, $94
DATA ·bitrev512_m24+1992(SB)/8, $222
DATA ·bitrev512_m24+2000(SB)/8, $350
DATA ·bitrev512_m24+2008(SB)/8, $478
DATA ·bitrev512_m24+2016(SB)/8, $126
DATA ·bitrev512_m24+2024(SB)/8, $254
DATA ·bitrev512_m24+2032(SB)/8, $382
DATA ·bitrev512_m24+2040(SB)/8, $510
DATA ·bitrev512_m24+2048(SB)/8, $1
DATA ·bitrev512_m24+2056(SB)/8, $129
DATA ·bitrev512_m24+2064(SB)/8, $257
DATA ·bitrev512_m24+2072(SB)/8, $385
DATA ·bitrev512_m24+2080(SB)/8, $33
DATA ·bitrev512_m24+2088(SB)/8, $161
DATA ·bitrev512_m24+2096(SB)/8, $289
DATA ·bitrev512_m24+2104(SB)/8, $417
DATA ·bitrev512_m24+2112(SB)/8, $65
DATA ·bitrev512_m24+2120(SB)/8, $193
DATA ·bitrev512_m24+2128(SB)/8, $321
DATA ·bitrev512_m24+2136(SB)/8, $449
DATA ·bitrev512_m24+2144(SB)/8, $97
DATA ·bitrev512_m24+2152(SB)/8, $225
DATA ·bitrev512_m24+2160(SB)/8, $353
DATA ·bitrev512_m24+2168(SB)/8, $481
DATA ·bitrev512_m24+2176(SB)/8, $9
DATA ·bitrev512_m24+2184(SB)/8, $137
DATA ·bitrev512_m24+2192(SB)/8, $265
DATA ·bitrev512_m24+2200(SB)/8, $393
DATA ·bitrev512_m24+2208(SB)/8, $41
DATA ·bitrev512_m24+2216(SB)/8, $169
DATA ·bitrev512_m24+2224(SB)/8, $297
DATA ·bitrev512_m24+2232(SB)/8, $425
DATA ·bitrev512_m24+2240(SB)/8, $73
DATA ·bitrev512_m24+2248(SB)/8, $201
DATA ·bitrev512_m24+2256(SB)/8, $329
DATA ·bitrev512_m24+2264(SB)/8, $457
DATA ·bitrev512_m24+2272(SB)/8, $105
DATA ·bitrev512_m24+2280(SB)/8, $233
DATA ·bitrev512_m24+2288(SB)/8, $361
DATA ·bitrev512_m24+2296(SB)/8, $489
DATA ·bitrev512_m24+2304(SB)/8, $17
DATA ·bitrev512_m24+2312(SB)/8, $145
DATA ·bitrev512_m24+2320(SB)/8, $273
DATA ·bitrev512_m24+2328(SB)/8, $401
DATA ·bitrev512_m24+2336(SB)/8, $49
DATA ·bitrev512_m24+2344(SB)/8, $177
DATA ·bitrev512_m24+2352(SB)/8, $305
DATA ·bitrev512_m24+2360(SB)/8, $433
DATA ·bitrev512_m24+2368(SB)/8, $81
DATA ·bitrev512_m24+2376(SB)/8, $209
DATA ·bitrev512_m24+2384(SB)/8, $337
DATA ·bitrev512_m24+2392(SB)/8, $465
DATA ·bitrev512_m24+2400(SB)/8, $113
DATA ·bitrev512_m24+2408(SB)/8, $241
DATA ·bitrev512_m24+2416(SB)/8, $369
DATA ·bitrev512_m24+2424(SB)/8, $497
DATA ·bitrev512_m24+2432(SB)/8, $25
DATA ·bitrev512_m24+2440(SB)/8, $153
DATA ·bitrev512_m24+2448(SB)/8, $281
DATA ·bitrev512_m24+2456(SB)/8, $409
DATA ·bitrev512_m24+2464(SB)/8, $57
DATA ·bitrev512_m24+2472(SB)/8, $185
DATA ·bitrev512_m24+2480(SB)/8, $313
DATA ·bitrev512_m24+2488(SB)/8, $441
DATA ·bitrev512_m24+2496(SB)/8, $89
DATA ·bitrev512_m24+2504(SB)/8, $217
DATA ·bitrev512_m24+2512(SB)/8, $345
DATA ·bitrev512_m24+2520(SB)/8, $473
DATA ·bitrev512_m24+2528(SB)/8, $121
DATA ·bitrev512_m24+2536(SB)/8, $249
DATA ·bitrev512_m24+2544(SB)/8, $377
DATA ·bitrev512_m24+2552(SB)/8, $505
DATA ·bitrev512_m24+2560(SB)/8, $3
DATA ·bitrev512_m24+2568(SB)/8, $131
DATA ·bitrev512_m24+2576(SB)/8, $259
DATA ·bitrev512_m24+2584(SB)/8, $387
DATA ·bitrev512_m24+2592(SB)/8, $35
DATA ·bitrev512_m24+2600(SB)/8, $163
DATA ·bitrev512_m24+2608(SB)/8, $291
DATA ·bitrev512_m24+2616(SB)/8, $419
DATA ·bitrev512_m24+2624(SB)/8, $67
DATA ·bitrev512_m24+2632(SB)/8, $195
DATA ·bitrev512_m24+2640(SB)/8, $323
DATA ·bitrev512_m24+2648(SB)/8, $451
DATA ·bitrev512_m24+2656(SB)/8, $99
DATA ·bitrev512_m24+2664(SB)/8, $227
DATA ·bitrev512_m24+2672(SB)/8, $355
DATA ·bitrev512_m24+2680(SB)/8, $483
DATA ·bitrev512_m24+2688(SB)/8, $11
DATA ·bitrev512_m24+2696(SB)/8, $139
DATA ·bitrev512_m24+2704(SB)/8, $267
DATA ·bitrev512_m24+2712(SB)/8, $395
DATA ·bitrev512_m24+2720(SB)/8, $43
DATA ·bitrev512_m24+2728(SB)/8, $171
DATA ·bitrev512_m24+2736(SB)/8, $299
DATA ·bitrev512_m24+2744(SB)/8, $427
DATA ·bitrev512_m24+2752(SB)/8, $75
DATA ·bitrev512_m24+2760(SB)/8, $203
DATA ·bitrev512_m24+2768(SB)/8, $331
DATA ·bitrev512_m24+2776(SB)/8, $459
DATA ·bitrev512_m24+2784(SB)/8, $107
DATA ·bitrev512_m24+2792(SB)/8, $235
DATA ·bitrev512_m24+2800(SB)/8, $363
DATA ·bitrev512_m24+2808(SB)/8, $491
DATA ·bitrev512_m24+2816(SB)/8, $19
DATA ·bitrev512_m24+2824(SB)/8, $147
DATA ·bitrev512_m24+2832(SB)/8, $275
DATA ·bitrev512_m24+2840(SB)/8, $403
DATA ·bitrev512_m24+2848(SB)/8, $51
DATA ·bitrev512_m24+2856(SB)/8, $179
DATA ·bitrev512_m24+2864(SB)/8, $307
DATA ·bitrev512_m24+2872(SB)/8, $435
DATA ·bitrev512_m24+2880(SB)/8, $83
DATA ·bitrev512_m24+2888(SB)/8, $211
DATA ·bitrev512_m24+2896(SB)/8, $339
DATA ·bitrev512_m24+2904(SB)/8, $467
DATA ·bitrev512_m24+2912(SB)/8, $115
DATA ·bitrev512_m24+2920(SB)/8, $243
DATA ·bitrev512_m24+2928(SB)/8, $371
DATA ·bitrev512_m24+2936(SB)/8, $499
DATA ·bitrev512_m24+2944(SB)/8, $27
DATA ·bitrev512_m24+2952(SB)/8, $155
DATA ·bitrev512_m24+2960(SB)/8, $283
DATA ·bitrev512_m24+2968(SB)/8, $411
DATA ·bitrev512_m24+2976(SB)/8, $59
DATA ·bitrev512_m24+2984(SB)/8, $187
DATA ·bitrev512_m24+2992(SB)/8, $315
DATA ·bitrev512_m24+3000(SB)/8, $443
DATA ·bitrev512_m24+3008(SB)/8, $91
DATA ·bitrev512_m24+3016(SB)/8, $219
DATA ·bitrev512_m24+3024(SB)/8, $347
DATA ·bitrev512_m24+3032(SB)/8, $475
DATA ·bitrev512_m24+3040(SB)/8, $123
DATA ·bitrev512_m24+3048(SB)/8, $251
DATA ·bitrev512_m24+3056(SB)/8, $379
DATA ·bitrev512_m24+3064(SB)/8, $507
DATA ·bitrev512_m24+3072(SB)/8, $5
DATA ·bitrev512_m24+3080(SB)/8, $133
DATA ·bitrev512_m24+3088(SB)/8, $261
DATA ·bitrev512_m24+3096(SB)/8, $389
DATA ·bitrev512_m24+3104(SB)/8, $37
DATA ·bitrev512_m24+3112(SB)/8, $165
DATA ·bitrev512_m24+3120(SB)/8, $293
DATA ·bitrev512_m24+3128(SB)/8, $421
DATA ·bitrev512_m24+3136(SB)/8, $69
DATA ·bitrev512_m24+3144(SB)/8, $197
DATA ·bitrev512_m24+3152(SB)/8, $325
DATA ·bitrev512_m24+3160(SB)/8, $453
DATA ·bitrev512_m24+3168(SB)/8, $101
DATA ·bitrev512_m24+3176(SB)/8, $229
DATA ·bitrev512_m24+3184(SB)/8, $357
DATA ·bitrev512_m24+3192(SB)/8, $485
DATA ·bitrev512_m24+3200(SB)/8, $13
DATA ·bitrev512_m24+3208(SB)/8, $141
DATA ·bitrev512_m24+3216(SB)/8, $269
DATA ·bitrev512_m24+3224(SB)/8, $397
DATA ·bitrev512_m24+3232(SB)/8, $45
DATA ·bitrev512_m24+3240(SB)/8, $173
DATA ·bitrev512_m24+3248(SB)/8, $301
DATA ·bitrev512_m24+3256(SB)/8, $429
DATA ·bitrev512_m24+3264(SB)/8, $77
DATA ·bitrev512_m24+3272(SB)/8, $205
DATA ·bitrev512_m24+3280(SB)/8, $333
DATA ·bitrev512_m24+3288(SB)/8, $461
DATA ·bitrev512_m24+3296(SB)/8, $109
DATA ·bitrev512_m24+3304(SB)/8, $237
DATA ·bitrev512_m24+3312(SB)/8, $365
DATA ·bitrev512_m24+3320(SB)/8, $493
DATA ·bitrev512_m24+3328(SB)/8, $21
DATA ·bitrev512_m24+3336(SB)/8, $149
DATA ·bitrev512_m24+3344(SB)/8, $277
DATA ·bitrev512_m24+3352(SB)/8, $405
DATA ·bitrev512_m24+3360(SB)/8, $53
DATA ·bitrev512_m24+3368(SB)/8, $181
DATA ·bitrev512_m24+3376(SB)/8, $309
DATA ·bitrev512_m24+3384(SB)/8, $437
DATA ·bitrev512_m24+3392(SB)/8, $85
DATA ·bitrev512_m24+3400(SB)/8, $213
DATA ·bitrev512_m24+3408(SB)/8, $341
DATA ·bitrev512_m24+3416(SB)/8, $469
DATA ·bitrev512_m24+3424(SB)/8, $117
DATA ·bitrev512_m24+3432(SB)/8, $245
DATA ·bitrev512_m24+3440(SB)/8, $373
DATA ·bitrev512_m24+3448(SB)/8, $501
DATA ·bitrev512_m24+3456(SB)/8, $29
DATA ·bitrev512_m24+3464(SB)/8, $157
DATA ·bitrev512_m24+3472(SB)/8, $285
DATA ·bitrev512_m24+3480(SB)/8, $413
DATA ·bitrev512_m24+3488(SB)/8, $61
DATA ·bitrev512_m24+3496(SB)/8, $189
DATA ·bitrev512_m24+3504(SB)/8, $317
DATA ·bitrev512_m24+3512(SB)/8, $445
DATA ·bitrev512_m24+3520(SB)/8, $93
DATA ·bitrev512_m24+3528(SB)/8, $221
DATA ·bitrev512_m24+3536(SB)/8, $349
DATA ·bitrev512_m24+3544(SB)/8, $477
DATA ·bitrev512_m24+3552(SB)/8, $125
DATA ·bitrev512_m24+3560(SB)/8, $253
DATA ·bitrev512_m24+3568(SB)/8, $381
DATA ·bitrev512_m24+3576(SB)/8, $509
DATA ·bitrev512_m24+3584(SB)/8, $7
DATA ·bitrev512_m24+3592(SB)/8, $135
DATA ·bitrev512_m24+3600(SB)/8, $263
DATA ·bitrev512_m24+3608(SB)/8, $391
DATA ·bitrev512_m24+3616(SB)/8, $39
DATA ·bitrev512_m24+3624(SB)/8, $167
DATA ·bitrev512_m24+3632(SB)/8, $295
DATA ·bitrev512_m24+3640(SB)/8, $423
DATA ·bitrev512_m24+3648(SB)/8, $71
DATA ·bitrev512_m24+3656(SB)/8, $199
DATA ·bitrev512_m24+3664(SB)/8, $327
DATA ·bitrev512_m24+3672(SB)/8, $455
DATA ·bitrev512_m24+3680(SB)/8, $103
DATA ·bitrev512_m24+3688(SB)/8, $231
DATA ·bitrev512_m24+3696(SB)/8, $359
DATA ·bitrev512_m24+3704(SB)/8, $487
DATA ·bitrev512_m24+3712(SB)/8, $15
DATA ·bitrev512_m24+3720(SB)/8, $143
DATA ·bitrev512_m24+3728(SB)/8, $271
DATA ·bitrev512_m24+3736(SB)/8, $399
DATA ·bitrev512_m24+3744(SB)/8, $47
DATA ·bitrev512_m24+3752(SB)/8, $175
DATA ·bitrev512_m24+3760(SB)/8, $303
DATA ·bitrev512_m24+3768(SB)/8, $431
DATA ·bitrev512_m24+3776(SB)/8, $79
DATA ·bitrev512_m24+3784(SB)/8, $207
DATA ·bitrev512_m24+3792(SB)/8, $335
DATA ·bitrev512_m24+3800(SB)/8, $463
DATA ·bitrev512_m24+3808(SB)/8, $111
DATA ·bitrev512_m24+3816(SB)/8, $239
DATA ·bitrev512_m24+3824(SB)/8, $367
DATA ·bitrev512_m24+3832(SB)/8, $495
DATA ·bitrev512_m24+3840(SB)/8, $23
DATA ·bitrev512_m24+3848(SB)/8, $151
DATA ·bitrev512_m24+3856(SB)/8, $279
DATA ·bitrev512_m24+3864(SB)/8, $407
DATA ·bitrev512_m24+3872(SB)/8, $55
DATA ·bitrev512_m24+3880(SB)/8, $183
DATA ·bitrev512_m24+3888(SB)/8, $311
DATA ·bitrev512_m24+3896(SB)/8, $439
DATA ·bitrev512_m24+3904(SB)/8, $87
DATA ·bitrev512_m24+3912(SB)/8, $215
DATA ·bitrev512_m24+3920(SB)/8, $343
DATA ·bitrev512_m24+3928(SB)/8, $471
DATA ·bitrev512_m24+3936(SB)/8, $119
DATA ·bitrev512_m24+3944(SB)/8, $247
DATA ·bitrev512_m24+3952(SB)/8, $375
DATA ·bitrev512_m24+3960(SB)/8, $503
DATA ·bitrev512_m24+3968(SB)/8, $31
DATA ·bitrev512_m24+3976(SB)/8, $159
DATA ·bitrev512_m24+3984(SB)/8, $287
DATA ·bitrev512_m24+3992(SB)/8, $415
DATA ·bitrev512_m24+4000(SB)/8, $63
DATA ·bitrev512_m24+4008(SB)/8, $191
DATA ·bitrev512_m24+4016(SB)/8, $319
DATA ·bitrev512_m24+4024(SB)/8, $447
DATA ·bitrev512_m24+4032(SB)/8, $95
DATA ·bitrev512_m24+4040(SB)/8, $223
DATA ·bitrev512_m24+4048(SB)/8, $351
DATA ·bitrev512_m24+4056(SB)/8, $479
DATA ·bitrev512_m24+4064(SB)/8, $127
DATA ·bitrev512_m24+4072(SB)/8, $255
DATA ·bitrev512_m24+4080(SB)/8, $383
DATA ·bitrev512_m24+4088(SB)/8, $511
GLOBL ·bitrev512_m24(SB), RODATA, $4096

//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-1024 Radix-4 FFT Kernel for AMD64
// ===========================================================================
//
// Algorithm: Radix-4 Decimation-in-Time (DIT) FFT
// Stages: 5 (log₄(1024) = 5)
//
// Stage structure:
//   Stage 1: 256 groups × 1 butterfly, stride=4,   no twiddle (W^0 = 1)
//   Stage 2: 64 groups × 4 butterflies, stride=16, twiddle step=64
//   Stage 3: 16 groups × 16 butterflies, stride=64, twiddle step=16
//   Stage 4: 4 groups × 64 butterflies, stride=256, twiddle step=4
//   Stage 5: 1 group × 256 butterflies, stride=1024, twiddle step=1
//
// ===========================================================================

#include "textflag.h"

TEXT ·ForwardAVX2Size1024Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 1024)

	// Verify n == 1024
	CMPQ R13, $1024
	JNE  r4_1024_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $1024
	JL   r4_1024_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $1024
	JL   r4_1024_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $1024
	JL   r4_1024_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $1024
	JL   r4_1024_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  r4_1024_use_dst
	MOVQ R11, R8             // In-place: use scratch

r4_1024_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 bit-reversal)
	// ==================================================================
	XORQ CX, CX              // CX = i = 0

r4_1024_bitrev_loop:
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[i]]
	MOVQ AX, (R8)(CX*8)      // work[i] = src[bitrev[i]]
	INCQ CX
	CMPQ CX, $1024
	JL   r4_1024_bitrev_loop

r4_1024_stage1:
	// ==================================================================
	// Stage 1: 256 groups, stride=4
	// ==================================================================

	XORQ CX, CX              // CX = base offset

r4_1024_stage1_loop:
	CMPQ CX, $1024
	JGE  r4_1024_stage2

	LEAQ (R8)(CX*8), SI
	VMOVSD (SI), X0          // a0
	VMOVSD 8(SI), X1         // a1
	VMOVSD 16(SI), X2        // a2
	VMOVSD 24(SI), X3        // a3

	VADDPS X0, X2, X4        // t0
	VSUBPS X2, X0, X5        // t1
	VADDPS X1, X3, X6        // t2
	VSUBPS X3, X1, X7        // t3

	// (-i)*t3
	VPERMILPS $0xB1, X7, X8
	VXORPS X9, X9, X9
	VSUBPS X8, X9, X10
	VBLENDPS $0x02, X10, X8, X8

	// i*t3
	VPERMILPS $0xB1, X7, X11
	VSUBPS X11, X9, X10
	VBLENDPS $0x01, X10, X11, X11

	VADDPS X4, X6, X0        // y0
	VADDPS X5, X8, X1        // y1
	VSUBPS X6, X4, X2        // y2
	VADDPS X5, X11, X3       // y3

	VMOVSD X0, (SI)
	VMOVSD X1, 8(SI)
	VMOVSD X2, 16(SI)
	VMOVSD X3, 24(SI)

	ADDQ $4, CX
	JMP  r4_1024_stage1_loop

r4_1024_stage2:
	// ==================================================================
	// Stage 2: 64 groups, 4 butterflies
	// Twiddle step = 64
	// ==================================================================

	XORQ CX, CX              // CX = base offset

r4_1024_stage2_outer:
	CMPQ CX, $1024
	JGE  r4_1024_stage3

	XORQ DX, DX              // DX = j

r4_1024_stage2_inner:
	CMPQ DX, $4
	JGE  r4_1024_stage2_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 4(BX), SI
	LEAQ 8(BX), DI
	LEAQ 12(BX), R14

	// Twiddles: j*64, 2*j*64, 3*j*64
	MOVQ DX, R15
	SHLQ $6, R15             // j*64
	VMOVSD (R10)(R15*8), X8

	MOVQ R15, R13
	SHLQ $1, R15             // 2*j*64
	VMOVSD (R10)(R15*8), X9

	ADDQ R13, R15            // 3*j*64
	VMOVSD (R10)(R15*8), X10

	VMOVSD (R8)(BX*8), X0
	VMOVSD (R8)(SI*8), X1
	VMOVSD (R8)(DI*8), X2
	VMOVSD (R8)(R14*8), X3

	// Complex multiply
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
	JMP  r4_1024_stage2_inner

r4_1024_stage2_next:
	ADDQ $16, CX
	JMP  r4_1024_stage2_outer

r4_1024_stage3:
	// ==================================================================
	// Stage 3: 16 groups, 16 butterflies
	// Twiddle step = 16
	// ==================================================================

	XORQ CX, CX

r4_1024_stage3_outer:
	CMPQ CX, $1024
	JGE  r4_1024_stage4

	XORQ DX, DX

r4_1024_stage3_inner:
	CMPQ DX, $16
	JGE  r4_1024_stage3_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 16(BX), SI
	LEAQ 32(BX), DI
	LEAQ 48(BX), R14

	// Twiddles: j*16
	MOVQ DX, R15
	SHLQ $4, R15
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
	JMP  r4_1024_stage3_inner

r4_1024_stage3_next:
	ADDQ $64, CX
	JMP  r4_1024_stage3_outer

r4_1024_stage4:
	// ==================================================================
	// Stage 4: 4 groups, 64 butterflies
	// Twiddle step = 4
	// ==================================================================

	XORQ CX, CX

r4_1024_stage4_outer:
	CMPQ CX, $1024
	JGE  r4_1024_stage5

	XORQ DX, DX

r4_1024_stage4_inner:
	CMPQ DX, $64
	JGE  r4_1024_stage4_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 64(BX), SI
	LEAQ 128(BX), DI
	LEAQ 192(BX), R14

	// Twiddles: j*4
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
	JMP  r4_1024_stage4_inner

r4_1024_stage4_next:
	ADDQ $256, CX
	JMP  r4_1024_stage4_outer

r4_1024_stage5:
	// ==================================================================
	// Stage 5: 1 group, 256 butterflies
	// Twiddle step = 1
	// ==================================================================

	XORQ DX, DX

r4_1024_stage5_loop:
	CMPQ DX, $256
	JGE  r4_1024_done

	MOVQ DX, BX
	LEAQ 256(DX), SI
	LEAQ 512(DX), DI
	LEAQ 768(DX), R14

	// Twiddles: j*1
	VMOVSD (R10)(DX*8), X8   // w1

	MOVQ DX, R15
	SHLQ $1, R15             // 2*j
	VMOVSD (R10)(R15*8), X9  // w2

	ADDQ DX, R15             // 3*j
	VMOVSD (R10)(R15*8), X10 // w3

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
	JMP  r4_1024_stage5_loop

r4_1024_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

r4_1024_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform
// ===========================================================================
TEXT ·InverseAVX2Size1024Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13     // n

	CMPQ R13, $1024
	JNE  r4_1024_inv_return_false

	MOVQ dst+8(FP), AX
	CMPQ AX, $1024
	JL   r4_1024_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $1024
	JL   r4_1024_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $1024
	JL   r4_1024_inv_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $1024
	JL   r4_1024_inv_return_false

	CMPQ R8, R9
	JNE  r4_1024_inv_use_dst
	MOVQ R11, R8

r4_1024_inv_use_dst:
	// Bit-reversal
	XORQ CX, CX
r4_1024_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $1024
	JL   r4_1024_inv_bitrev_loop

r4_1024_inv_stage1:
	XORQ CX, CX
r4_1024_inv_stage1_loop:
	CMPQ CX, $1024
	JGE  r4_1024_inv_stage2

	LEAQ (R8)(CX*8), SI
	VMOVSD (SI), X0
	VMOVSD 8(SI), X1
	VMOVSD 16(SI), X2
	VMOVSD 24(SI), X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	VPERMILPS $0xB1, X7, X8
	VXORPS X9, X9, X9
	VSUBPS X8, X9, X10
	VBLENDPS $0x02, X10, X8, X8  // (-i)*t3

	VPERMILPS $0xB1, X7, X11
	VSUBPS X11, X9, X10
	VBLENDPS $0x01, X10, X11, X11 // i*t3

	VADDPS X4, X6, X0
	VADDPS X5, X11, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X8, X3

	VMOVSD X0, (SI)
	VMOVSD X1, 8(SI)
	VMOVSD X2, 16(SI)
	VMOVSD X3, 24(SI)

	ADDQ $4, CX
	JMP  r4_1024_inv_stage1_loop

r4_1024_inv_stage2:
	XORQ CX, CX
r4_1024_inv_stage2_outer:
	CMPQ CX, $1024
	JGE  r4_1024_inv_stage3

	XORQ DX, DX
r4_1024_inv_stage2_inner:
	CMPQ DX, $4
	JGE  r4_1024_inv_stage2_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 4(BX), SI
	LEAQ 8(BX), DI
	LEAQ 12(BX), R14

	MOVQ DX, R15
	SHLQ $6, R15
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
	VBLENDPS $0x02, X11, X14, X14

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
	JMP  r4_1024_inv_stage2_inner

r4_1024_inv_stage2_next:
	ADDQ $16, CX
	JMP  r4_1024_inv_stage2_outer

r4_1024_inv_stage3:
	XORQ CX, CX
r4_1024_inv_stage3_outer:
	CMPQ CX, $1024
	JGE  r4_1024_inv_stage4

	XORQ DX, DX
r4_1024_inv_stage3_inner:
	CMPQ DX, $16
	JGE  r4_1024_inv_stage3_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 16(BX), SI
	LEAQ 32(BX), DI
	LEAQ 48(BX), R14

	MOVQ DX, R15
	SHLQ $4, R15
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
	VBLENDPS $0x02, X11, X14, X14

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
	JMP  r4_1024_inv_stage3_inner

r4_1024_inv_stage3_next:
	ADDQ $64, CX
	JMP  r4_1024_inv_stage3_outer

r4_1024_inv_stage4:
	XORQ CX, CX
r4_1024_inv_stage4_outer:
	CMPQ CX, $1024
	JGE  r4_1024_inv_stage5

	XORQ DX, DX
r4_1024_inv_stage4_inner:
	CMPQ DX, $64
	JGE  r4_1024_inv_stage4_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 64(BX), SI
	LEAQ 128(BX), DI
	LEAQ 192(BX), R14

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
	VBLENDPS $0x02, X11, X14, X14

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
	JMP  r4_1024_inv_stage4_inner

r4_1024_inv_stage4_next:
	ADDQ $256, CX
	JMP  r4_1024_inv_stage4_outer

r4_1024_inv_stage5:
	XORQ DX, DX
r4_1024_inv_stage5_loop:
	CMPQ DX, $256
	JGE  r4_1024_inv_scale

	MOVQ DX, BX
	LEAQ 256(DX), SI
	LEAQ 512(DX), DI
	LEAQ 768(DX), R14

	VMOVSD (R10)(DX*8), X8
	MOVQ DX, R15
	SHLQ $1, R15
	VMOVSD (R10)(R15*8), X9
	ADDQ DX, R15
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
	VBLENDPS $0x02, X11, X14, X14

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
	JMP  r4_1024_inv_stage5_loop

r4_1024_inv_scale:
	// 1/1024 = 0.0009765625
	MOVL ·oneThousandTwentyFourth32(SB), AX
	MOVD AX, X8
	VBROADCASTSS X8, Y8

	XORQ CX, CX
r4_1024_inv_scale_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, (R8)(CX*1)
	VMOVUPS Y1, 32(R8)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $8192  // 1024 * 8 bytes = 8192 bytes
	JL   r4_1024_inv_scale_loop

	// Copy if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r4_1024_inv_done

	XORQ CX, CX
r4_1024_inv_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMOVUPS Y0, (R9)(CX*1)
	VMOVUPS Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $8192
	JL   r4_1024_inv_copy_loop

r4_1024_inv_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

r4_1024_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET

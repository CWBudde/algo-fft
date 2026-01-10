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

TEXT ·ForwardAVX2Size1024Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 1024)
	LEAQ ·bitrev1024_r4(SB), R12

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
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r4_1024_ret

	XORQ CX, CX

r4_1024_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMOVUPS Y0, (R9)(CX*1)
	VMOVUPS Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $8192
	JL   r4_1024_copy_loop

r4_1024_ret:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

r4_1024_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform
// ===========================================================================
TEXT ·InverseAVX2Size1024Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13     // n
	LEAQ ·bitrev1024_r4(SB), R12

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
	MOVB $1, ret+96(FP)
	RET

r4_1024_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET


DATA ·bitrev1024_r4+0(SB)/8, $0
DATA ·bitrev1024_r4+8(SB)/8, $256
DATA ·bitrev1024_r4+16(SB)/8, $512
DATA ·bitrev1024_r4+24(SB)/8, $768
DATA ·bitrev1024_r4+32(SB)/8, $64
DATA ·bitrev1024_r4+40(SB)/8, $320
DATA ·bitrev1024_r4+48(SB)/8, $576
DATA ·bitrev1024_r4+56(SB)/8, $832
DATA ·bitrev1024_r4+64(SB)/8, $128
DATA ·bitrev1024_r4+72(SB)/8, $384
DATA ·bitrev1024_r4+80(SB)/8, $640
DATA ·bitrev1024_r4+88(SB)/8, $896
DATA ·bitrev1024_r4+96(SB)/8, $192
DATA ·bitrev1024_r4+104(SB)/8, $448
DATA ·bitrev1024_r4+112(SB)/8, $704
DATA ·bitrev1024_r4+120(SB)/8, $960
DATA ·bitrev1024_r4+128(SB)/8, $16
DATA ·bitrev1024_r4+136(SB)/8, $272
DATA ·bitrev1024_r4+144(SB)/8, $528
DATA ·bitrev1024_r4+152(SB)/8, $784
DATA ·bitrev1024_r4+160(SB)/8, $80
DATA ·bitrev1024_r4+168(SB)/8, $336
DATA ·bitrev1024_r4+176(SB)/8, $592
DATA ·bitrev1024_r4+184(SB)/8, $848
DATA ·bitrev1024_r4+192(SB)/8, $144
DATA ·bitrev1024_r4+200(SB)/8, $400
DATA ·bitrev1024_r4+208(SB)/8, $656
DATA ·bitrev1024_r4+216(SB)/8, $912
DATA ·bitrev1024_r4+224(SB)/8, $208
DATA ·bitrev1024_r4+232(SB)/8, $464
DATA ·bitrev1024_r4+240(SB)/8, $720
DATA ·bitrev1024_r4+248(SB)/8, $976
DATA ·bitrev1024_r4+256(SB)/8, $32
DATA ·bitrev1024_r4+264(SB)/8, $288
DATA ·bitrev1024_r4+272(SB)/8, $544
DATA ·bitrev1024_r4+280(SB)/8, $800
DATA ·bitrev1024_r4+288(SB)/8, $96
DATA ·bitrev1024_r4+296(SB)/8, $352
DATA ·bitrev1024_r4+304(SB)/8, $608
DATA ·bitrev1024_r4+312(SB)/8, $864
DATA ·bitrev1024_r4+320(SB)/8, $160
DATA ·bitrev1024_r4+328(SB)/8, $416
DATA ·bitrev1024_r4+336(SB)/8, $672
DATA ·bitrev1024_r4+344(SB)/8, $928
DATA ·bitrev1024_r4+352(SB)/8, $224
DATA ·bitrev1024_r4+360(SB)/8, $480
DATA ·bitrev1024_r4+368(SB)/8, $736
DATA ·bitrev1024_r4+376(SB)/8, $992
DATA ·bitrev1024_r4+384(SB)/8, $48
DATA ·bitrev1024_r4+392(SB)/8, $304
DATA ·bitrev1024_r4+400(SB)/8, $560
DATA ·bitrev1024_r4+408(SB)/8, $816
DATA ·bitrev1024_r4+416(SB)/8, $112
DATA ·bitrev1024_r4+424(SB)/8, $368
DATA ·bitrev1024_r4+432(SB)/8, $624
DATA ·bitrev1024_r4+440(SB)/8, $880
DATA ·bitrev1024_r4+448(SB)/8, $176
DATA ·bitrev1024_r4+456(SB)/8, $432
DATA ·bitrev1024_r4+464(SB)/8, $688
DATA ·bitrev1024_r4+472(SB)/8, $944
DATA ·bitrev1024_r4+480(SB)/8, $240
DATA ·bitrev1024_r4+488(SB)/8, $496
DATA ·bitrev1024_r4+496(SB)/8, $752
DATA ·bitrev1024_r4+504(SB)/8, $1008
DATA ·bitrev1024_r4+512(SB)/8, $4
DATA ·bitrev1024_r4+520(SB)/8, $260
DATA ·bitrev1024_r4+528(SB)/8, $516
DATA ·bitrev1024_r4+536(SB)/8, $772
DATA ·bitrev1024_r4+544(SB)/8, $68
DATA ·bitrev1024_r4+552(SB)/8, $324
DATA ·bitrev1024_r4+560(SB)/8, $580
DATA ·bitrev1024_r4+568(SB)/8, $836
DATA ·bitrev1024_r4+576(SB)/8, $132
DATA ·bitrev1024_r4+584(SB)/8, $388
DATA ·bitrev1024_r4+592(SB)/8, $644
DATA ·bitrev1024_r4+600(SB)/8, $900
DATA ·bitrev1024_r4+608(SB)/8, $196
DATA ·bitrev1024_r4+616(SB)/8, $452
DATA ·bitrev1024_r4+624(SB)/8, $708
DATA ·bitrev1024_r4+632(SB)/8, $964
DATA ·bitrev1024_r4+640(SB)/8, $20
DATA ·bitrev1024_r4+648(SB)/8, $276
DATA ·bitrev1024_r4+656(SB)/8, $532
DATA ·bitrev1024_r4+664(SB)/8, $788
DATA ·bitrev1024_r4+672(SB)/8, $84
DATA ·bitrev1024_r4+680(SB)/8, $340
DATA ·bitrev1024_r4+688(SB)/8, $596
DATA ·bitrev1024_r4+696(SB)/8, $852
DATA ·bitrev1024_r4+704(SB)/8, $148
DATA ·bitrev1024_r4+712(SB)/8, $404
DATA ·bitrev1024_r4+720(SB)/8, $660
DATA ·bitrev1024_r4+728(SB)/8, $916
DATA ·bitrev1024_r4+736(SB)/8, $212
DATA ·bitrev1024_r4+744(SB)/8, $468
DATA ·bitrev1024_r4+752(SB)/8, $724
DATA ·bitrev1024_r4+760(SB)/8, $980
DATA ·bitrev1024_r4+768(SB)/8, $36
DATA ·bitrev1024_r4+776(SB)/8, $292
DATA ·bitrev1024_r4+784(SB)/8, $548
DATA ·bitrev1024_r4+792(SB)/8, $804
DATA ·bitrev1024_r4+800(SB)/8, $100
DATA ·bitrev1024_r4+808(SB)/8, $356
DATA ·bitrev1024_r4+816(SB)/8, $612
DATA ·bitrev1024_r4+824(SB)/8, $868
DATA ·bitrev1024_r4+832(SB)/8, $164
DATA ·bitrev1024_r4+840(SB)/8, $420
DATA ·bitrev1024_r4+848(SB)/8, $676
DATA ·bitrev1024_r4+856(SB)/8, $932
DATA ·bitrev1024_r4+864(SB)/8, $228
DATA ·bitrev1024_r4+872(SB)/8, $484
DATA ·bitrev1024_r4+880(SB)/8, $740
DATA ·bitrev1024_r4+888(SB)/8, $996
DATA ·bitrev1024_r4+896(SB)/8, $52
DATA ·bitrev1024_r4+904(SB)/8, $308
DATA ·bitrev1024_r4+912(SB)/8, $564
DATA ·bitrev1024_r4+920(SB)/8, $820
DATA ·bitrev1024_r4+928(SB)/8, $116
DATA ·bitrev1024_r4+936(SB)/8, $372
DATA ·bitrev1024_r4+944(SB)/8, $628
DATA ·bitrev1024_r4+952(SB)/8, $884
DATA ·bitrev1024_r4+960(SB)/8, $180
DATA ·bitrev1024_r4+968(SB)/8, $436
DATA ·bitrev1024_r4+976(SB)/8, $692
DATA ·bitrev1024_r4+984(SB)/8, $948
DATA ·bitrev1024_r4+992(SB)/8, $244
DATA ·bitrev1024_r4+1000(SB)/8, $500
DATA ·bitrev1024_r4+1008(SB)/8, $756
DATA ·bitrev1024_r4+1016(SB)/8, $1012
DATA ·bitrev1024_r4+1024(SB)/8, $8
DATA ·bitrev1024_r4+1032(SB)/8, $264
DATA ·bitrev1024_r4+1040(SB)/8, $520
DATA ·bitrev1024_r4+1048(SB)/8, $776
DATA ·bitrev1024_r4+1056(SB)/8, $72
DATA ·bitrev1024_r4+1064(SB)/8, $328
DATA ·bitrev1024_r4+1072(SB)/8, $584
DATA ·bitrev1024_r4+1080(SB)/8, $840
DATA ·bitrev1024_r4+1088(SB)/8, $136
DATA ·bitrev1024_r4+1096(SB)/8, $392
DATA ·bitrev1024_r4+1104(SB)/8, $648
DATA ·bitrev1024_r4+1112(SB)/8, $904
DATA ·bitrev1024_r4+1120(SB)/8, $200
DATA ·bitrev1024_r4+1128(SB)/8, $456
DATA ·bitrev1024_r4+1136(SB)/8, $712
DATA ·bitrev1024_r4+1144(SB)/8, $968
DATA ·bitrev1024_r4+1152(SB)/8, $24
DATA ·bitrev1024_r4+1160(SB)/8, $280
DATA ·bitrev1024_r4+1168(SB)/8, $536
DATA ·bitrev1024_r4+1176(SB)/8, $792
DATA ·bitrev1024_r4+1184(SB)/8, $88
DATA ·bitrev1024_r4+1192(SB)/8, $344
DATA ·bitrev1024_r4+1200(SB)/8, $600
DATA ·bitrev1024_r4+1208(SB)/8, $856
DATA ·bitrev1024_r4+1216(SB)/8, $152
DATA ·bitrev1024_r4+1224(SB)/8, $408
DATA ·bitrev1024_r4+1232(SB)/8, $664
DATA ·bitrev1024_r4+1240(SB)/8, $920
DATA ·bitrev1024_r4+1248(SB)/8, $216
DATA ·bitrev1024_r4+1256(SB)/8, $472
DATA ·bitrev1024_r4+1264(SB)/8, $728
DATA ·bitrev1024_r4+1272(SB)/8, $984
DATA ·bitrev1024_r4+1280(SB)/8, $40
DATA ·bitrev1024_r4+1288(SB)/8, $296
DATA ·bitrev1024_r4+1296(SB)/8, $552
DATA ·bitrev1024_r4+1304(SB)/8, $808
DATA ·bitrev1024_r4+1312(SB)/8, $104
DATA ·bitrev1024_r4+1320(SB)/8, $360
DATA ·bitrev1024_r4+1328(SB)/8, $616
DATA ·bitrev1024_r4+1336(SB)/8, $872
DATA ·bitrev1024_r4+1344(SB)/8, $168
DATA ·bitrev1024_r4+1352(SB)/8, $424
DATA ·bitrev1024_r4+1360(SB)/8, $680
DATA ·bitrev1024_r4+1368(SB)/8, $936
DATA ·bitrev1024_r4+1376(SB)/8, $232
DATA ·bitrev1024_r4+1384(SB)/8, $488
DATA ·bitrev1024_r4+1392(SB)/8, $744
DATA ·bitrev1024_r4+1400(SB)/8, $1000
DATA ·bitrev1024_r4+1408(SB)/8, $56
DATA ·bitrev1024_r4+1416(SB)/8, $312
DATA ·bitrev1024_r4+1424(SB)/8, $568
DATA ·bitrev1024_r4+1432(SB)/8, $824
DATA ·bitrev1024_r4+1440(SB)/8, $120
DATA ·bitrev1024_r4+1448(SB)/8, $376
DATA ·bitrev1024_r4+1456(SB)/8, $632
DATA ·bitrev1024_r4+1464(SB)/8, $888
DATA ·bitrev1024_r4+1472(SB)/8, $184
DATA ·bitrev1024_r4+1480(SB)/8, $440
DATA ·bitrev1024_r4+1488(SB)/8, $696
DATA ·bitrev1024_r4+1496(SB)/8, $952
DATA ·bitrev1024_r4+1504(SB)/8, $248
DATA ·bitrev1024_r4+1512(SB)/8, $504
DATA ·bitrev1024_r4+1520(SB)/8, $760
DATA ·bitrev1024_r4+1528(SB)/8, $1016
DATA ·bitrev1024_r4+1536(SB)/8, $12
DATA ·bitrev1024_r4+1544(SB)/8, $268
DATA ·bitrev1024_r4+1552(SB)/8, $524
DATA ·bitrev1024_r4+1560(SB)/8, $780
DATA ·bitrev1024_r4+1568(SB)/8, $76
DATA ·bitrev1024_r4+1576(SB)/8, $332
DATA ·bitrev1024_r4+1584(SB)/8, $588
DATA ·bitrev1024_r4+1592(SB)/8, $844
DATA ·bitrev1024_r4+1600(SB)/8, $140
DATA ·bitrev1024_r4+1608(SB)/8, $396
DATA ·bitrev1024_r4+1616(SB)/8, $652
DATA ·bitrev1024_r4+1624(SB)/8, $908
DATA ·bitrev1024_r4+1632(SB)/8, $204
DATA ·bitrev1024_r4+1640(SB)/8, $460
DATA ·bitrev1024_r4+1648(SB)/8, $716
DATA ·bitrev1024_r4+1656(SB)/8, $972
DATA ·bitrev1024_r4+1664(SB)/8, $28
DATA ·bitrev1024_r4+1672(SB)/8, $284
DATA ·bitrev1024_r4+1680(SB)/8, $540
DATA ·bitrev1024_r4+1688(SB)/8, $796
DATA ·bitrev1024_r4+1696(SB)/8, $92
DATA ·bitrev1024_r4+1704(SB)/8, $348
DATA ·bitrev1024_r4+1712(SB)/8, $604
DATA ·bitrev1024_r4+1720(SB)/8, $860
DATA ·bitrev1024_r4+1728(SB)/8, $156
DATA ·bitrev1024_r4+1736(SB)/8, $412
DATA ·bitrev1024_r4+1744(SB)/8, $668
DATA ·bitrev1024_r4+1752(SB)/8, $924
DATA ·bitrev1024_r4+1760(SB)/8, $220
DATA ·bitrev1024_r4+1768(SB)/8, $476
DATA ·bitrev1024_r4+1776(SB)/8, $732
DATA ·bitrev1024_r4+1784(SB)/8, $988
DATA ·bitrev1024_r4+1792(SB)/8, $44
DATA ·bitrev1024_r4+1800(SB)/8, $300
DATA ·bitrev1024_r4+1808(SB)/8, $556
DATA ·bitrev1024_r4+1816(SB)/8, $812
DATA ·bitrev1024_r4+1824(SB)/8, $108
DATA ·bitrev1024_r4+1832(SB)/8, $364
DATA ·bitrev1024_r4+1840(SB)/8, $620
DATA ·bitrev1024_r4+1848(SB)/8, $876
DATA ·bitrev1024_r4+1856(SB)/8, $172
DATA ·bitrev1024_r4+1864(SB)/8, $428
DATA ·bitrev1024_r4+1872(SB)/8, $684
DATA ·bitrev1024_r4+1880(SB)/8, $940
DATA ·bitrev1024_r4+1888(SB)/8, $236
DATA ·bitrev1024_r4+1896(SB)/8, $492
DATA ·bitrev1024_r4+1904(SB)/8, $748
DATA ·bitrev1024_r4+1912(SB)/8, $1004
DATA ·bitrev1024_r4+1920(SB)/8, $60
DATA ·bitrev1024_r4+1928(SB)/8, $316
DATA ·bitrev1024_r4+1936(SB)/8, $572
DATA ·bitrev1024_r4+1944(SB)/8, $828
DATA ·bitrev1024_r4+1952(SB)/8, $124
DATA ·bitrev1024_r4+1960(SB)/8, $380
DATA ·bitrev1024_r4+1968(SB)/8, $636
DATA ·bitrev1024_r4+1976(SB)/8, $892
DATA ·bitrev1024_r4+1984(SB)/8, $188
DATA ·bitrev1024_r4+1992(SB)/8, $444
DATA ·bitrev1024_r4+2000(SB)/8, $700
DATA ·bitrev1024_r4+2008(SB)/8, $956
DATA ·bitrev1024_r4+2016(SB)/8, $252
DATA ·bitrev1024_r4+2024(SB)/8, $508
DATA ·bitrev1024_r4+2032(SB)/8, $764
DATA ·bitrev1024_r4+2040(SB)/8, $1020
DATA ·bitrev1024_r4+2048(SB)/8, $1
DATA ·bitrev1024_r4+2056(SB)/8, $257
DATA ·bitrev1024_r4+2064(SB)/8, $513
DATA ·bitrev1024_r4+2072(SB)/8, $769
DATA ·bitrev1024_r4+2080(SB)/8, $65
DATA ·bitrev1024_r4+2088(SB)/8, $321
DATA ·bitrev1024_r4+2096(SB)/8, $577
DATA ·bitrev1024_r4+2104(SB)/8, $833
DATA ·bitrev1024_r4+2112(SB)/8, $129
DATA ·bitrev1024_r4+2120(SB)/8, $385
DATA ·bitrev1024_r4+2128(SB)/8, $641
DATA ·bitrev1024_r4+2136(SB)/8, $897
DATA ·bitrev1024_r4+2144(SB)/8, $193
DATA ·bitrev1024_r4+2152(SB)/8, $449
DATA ·bitrev1024_r4+2160(SB)/8, $705
DATA ·bitrev1024_r4+2168(SB)/8, $961
DATA ·bitrev1024_r4+2176(SB)/8, $17
DATA ·bitrev1024_r4+2184(SB)/8, $273
DATA ·bitrev1024_r4+2192(SB)/8, $529
DATA ·bitrev1024_r4+2200(SB)/8, $785
DATA ·bitrev1024_r4+2208(SB)/8, $81
DATA ·bitrev1024_r4+2216(SB)/8, $337
DATA ·bitrev1024_r4+2224(SB)/8, $593
DATA ·bitrev1024_r4+2232(SB)/8, $849
DATA ·bitrev1024_r4+2240(SB)/8, $145
DATA ·bitrev1024_r4+2248(SB)/8, $401
DATA ·bitrev1024_r4+2256(SB)/8, $657
DATA ·bitrev1024_r4+2264(SB)/8, $913
DATA ·bitrev1024_r4+2272(SB)/8, $209
DATA ·bitrev1024_r4+2280(SB)/8, $465
DATA ·bitrev1024_r4+2288(SB)/8, $721
DATA ·bitrev1024_r4+2296(SB)/8, $977
DATA ·bitrev1024_r4+2304(SB)/8, $33
DATA ·bitrev1024_r4+2312(SB)/8, $289
DATA ·bitrev1024_r4+2320(SB)/8, $545
DATA ·bitrev1024_r4+2328(SB)/8, $801
DATA ·bitrev1024_r4+2336(SB)/8, $97
DATA ·bitrev1024_r4+2344(SB)/8, $353
DATA ·bitrev1024_r4+2352(SB)/8, $609
DATA ·bitrev1024_r4+2360(SB)/8, $865
DATA ·bitrev1024_r4+2368(SB)/8, $161
DATA ·bitrev1024_r4+2376(SB)/8, $417
DATA ·bitrev1024_r4+2384(SB)/8, $673
DATA ·bitrev1024_r4+2392(SB)/8, $929
DATA ·bitrev1024_r4+2400(SB)/8, $225
DATA ·bitrev1024_r4+2408(SB)/8, $481
DATA ·bitrev1024_r4+2416(SB)/8, $737
DATA ·bitrev1024_r4+2424(SB)/8, $993
DATA ·bitrev1024_r4+2432(SB)/8, $49
DATA ·bitrev1024_r4+2440(SB)/8, $305
DATA ·bitrev1024_r4+2448(SB)/8, $561
DATA ·bitrev1024_r4+2456(SB)/8, $817
DATA ·bitrev1024_r4+2464(SB)/8, $113
DATA ·bitrev1024_r4+2472(SB)/8, $369
DATA ·bitrev1024_r4+2480(SB)/8, $625
DATA ·bitrev1024_r4+2488(SB)/8, $881
DATA ·bitrev1024_r4+2496(SB)/8, $177
DATA ·bitrev1024_r4+2504(SB)/8, $433
DATA ·bitrev1024_r4+2512(SB)/8, $689
DATA ·bitrev1024_r4+2520(SB)/8, $945
DATA ·bitrev1024_r4+2528(SB)/8, $241
DATA ·bitrev1024_r4+2536(SB)/8, $497
DATA ·bitrev1024_r4+2544(SB)/8, $753
DATA ·bitrev1024_r4+2552(SB)/8, $1009
DATA ·bitrev1024_r4+2560(SB)/8, $5
DATA ·bitrev1024_r4+2568(SB)/8, $261
DATA ·bitrev1024_r4+2576(SB)/8, $517
DATA ·bitrev1024_r4+2584(SB)/8, $773
DATA ·bitrev1024_r4+2592(SB)/8, $69
DATA ·bitrev1024_r4+2600(SB)/8, $325
DATA ·bitrev1024_r4+2608(SB)/8, $581
DATA ·bitrev1024_r4+2616(SB)/8, $837
DATA ·bitrev1024_r4+2624(SB)/8, $133
DATA ·bitrev1024_r4+2632(SB)/8, $389
DATA ·bitrev1024_r4+2640(SB)/8, $645
DATA ·bitrev1024_r4+2648(SB)/8, $901
DATA ·bitrev1024_r4+2656(SB)/8, $197
DATA ·bitrev1024_r4+2664(SB)/8, $453
DATA ·bitrev1024_r4+2672(SB)/8, $709
DATA ·bitrev1024_r4+2680(SB)/8, $965
DATA ·bitrev1024_r4+2688(SB)/8, $21
DATA ·bitrev1024_r4+2696(SB)/8, $277
DATA ·bitrev1024_r4+2704(SB)/8, $533
DATA ·bitrev1024_r4+2712(SB)/8, $789
DATA ·bitrev1024_r4+2720(SB)/8, $85
DATA ·bitrev1024_r4+2728(SB)/8, $341
DATA ·bitrev1024_r4+2736(SB)/8, $597
DATA ·bitrev1024_r4+2744(SB)/8, $853
DATA ·bitrev1024_r4+2752(SB)/8, $149
DATA ·bitrev1024_r4+2760(SB)/8, $405
DATA ·bitrev1024_r4+2768(SB)/8, $661
DATA ·bitrev1024_r4+2776(SB)/8, $917
DATA ·bitrev1024_r4+2784(SB)/8, $213
DATA ·bitrev1024_r4+2792(SB)/8, $469
DATA ·bitrev1024_r4+2800(SB)/8, $725
DATA ·bitrev1024_r4+2808(SB)/8, $981
DATA ·bitrev1024_r4+2816(SB)/8, $37
DATA ·bitrev1024_r4+2824(SB)/8, $293
DATA ·bitrev1024_r4+2832(SB)/8, $549
DATA ·bitrev1024_r4+2840(SB)/8, $805
DATA ·bitrev1024_r4+2848(SB)/8, $101
DATA ·bitrev1024_r4+2856(SB)/8, $357
DATA ·bitrev1024_r4+2864(SB)/8, $613
DATA ·bitrev1024_r4+2872(SB)/8, $869
DATA ·bitrev1024_r4+2880(SB)/8, $165
DATA ·bitrev1024_r4+2888(SB)/8, $421
DATA ·bitrev1024_r4+2896(SB)/8, $677
DATA ·bitrev1024_r4+2904(SB)/8, $933
DATA ·bitrev1024_r4+2912(SB)/8, $229
DATA ·bitrev1024_r4+2920(SB)/8, $485
DATA ·bitrev1024_r4+2928(SB)/8, $741
DATA ·bitrev1024_r4+2936(SB)/8, $997
DATA ·bitrev1024_r4+2944(SB)/8, $53
DATA ·bitrev1024_r4+2952(SB)/8, $309
DATA ·bitrev1024_r4+2960(SB)/8, $565
DATA ·bitrev1024_r4+2968(SB)/8, $821
DATA ·bitrev1024_r4+2976(SB)/8, $117
DATA ·bitrev1024_r4+2984(SB)/8, $373
DATA ·bitrev1024_r4+2992(SB)/8, $629
DATA ·bitrev1024_r4+3000(SB)/8, $885
DATA ·bitrev1024_r4+3008(SB)/8, $181
DATA ·bitrev1024_r4+3016(SB)/8, $437
DATA ·bitrev1024_r4+3024(SB)/8, $693
DATA ·bitrev1024_r4+3032(SB)/8, $949
DATA ·bitrev1024_r4+3040(SB)/8, $245
DATA ·bitrev1024_r4+3048(SB)/8, $501
DATA ·bitrev1024_r4+3056(SB)/8, $757
DATA ·bitrev1024_r4+3064(SB)/8, $1013
DATA ·bitrev1024_r4+3072(SB)/8, $9
DATA ·bitrev1024_r4+3080(SB)/8, $265
DATA ·bitrev1024_r4+3088(SB)/8, $521
DATA ·bitrev1024_r4+3096(SB)/8, $777
DATA ·bitrev1024_r4+3104(SB)/8, $73
DATA ·bitrev1024_r4+3112(SB)/8, $329
DATA ·bitrev1024_r4+3120(SB)/8, $585
DATA ·bitrev1024_r4+3128(SB)/8, $841
DATA ·bitrev1024_r4+3136(SB)/8, $137
DATA ·bitrev1024_r4+3144(SB)/8, $393
DATA ·bitrev1024_r4+3152(SB)/8, $649
DATA ·bitrev1024_r4+3160(SB)/8, $905
DATA ·bitrev1024_r4+3168(SB)/8, $201
DATA ·bitrev1024_r4+3176(SB)/8, $457
DATA ·bitrev1024_r4+3184(SB)/8, $713
DATA ·bitrev1024_r4+3192(SB)/8, $969
DATA ·bitrev1024_r4+3200(SB)/8, $25
DATA ·bitrev1024_r4+3208(SB)/8, $281
DATA ·bitrev1024_r4+3216(SB)/8, $537
DATA ·bitrev1024_r4+3224(SB)/8, $793
DATA ·bitrev1024_r4+3232(SB)/8, $89
DATA ·bitrev1024_r4+3240(SB)/8, $345
DATA ·bitrev1024_r4+3248(SB)/8, $601
DATA ·bitrev1024_r4+3256(SB)/8, $857
DATA ·bitrev1024_r4+3264(SB)/8, $153
DATA ·bitrev1024_r4+3272(SB)/8, $409
DATA ·bitrev1024_r4+3280(SB)/8, $665
DATA ·bitrev1024_r4+3288(SB)/8, $921
DATA ·bitrev1024_r4+3296(SB)/8, $217
DATA ·bitrev1024_r4+3304(SB)/8, $473
DATA ·bitrev1024_r4+3312(SB)/8, $729
DATA ·bitrev1024_r4+3320(SB)/8, $985
DATA ·bitrev1024_r4+3328(SB)/8, $41
DATA ·bitrev1024_r4+3336(SB)/8, $297
DATA ·bitrev1024_r4+3344(SB)/8, $553
DATA ·bitrev1024_r4+3352(SB)/8, $809
DATA ·bitrev1024_r4+3360(SB)/8, $105
DATA ·bitrev1024_r4+3368(SB)/8, $361
DATA ·bitrev1024_r4+3376(SB)/8, $617
DATA ·bitrev1024_r4+3384(SB)/8, $873
DATA ·bitrev1024_r4+3392(SB)/8, $169
DATA ·bitrev1024_r4+3400(SB)/8, $425
DATA ·bitrev1024_r4+3408(SB)/8, $681
DATA ·bitrev1024_r4+3416(SB)/8, $937
DATA ·bitrev1024_r4+3424(SB)/8, $233
DATA ·bitrev1024_r4+3432(SB)/8, $489
DATA ·bitrev1024_r4+3440(SB)/8, $745
DATA ·bitrev1024_r4+3448(SB)/8, $1001
DATA ·bitrev1024_r4+3456(SB)/8, $57
DATA ·bitrev1024_r4+3464(SB)/8, $313
DATA ·bitrev1024_r4+3472(SB)/8, $569
DATA ·bitrev1024_r4+3480(SB)/8, $825
DATA ·bitrev1024_r4+3488(SB)/8, $121
DATA ·bitrev1024_r4+3496(SB)/8, $377
DATA ·bitrev1024_r4+3504(SB)/8, $633
DATA ·bitrev1024_r4+3512(SB)/8, $889
DATA ·bitrev1024_r4+3520(SB)/8, $185
DATA ·bitrev1024_r4+3528(SB)/8, $441
DATA ·bitrev1024_r4+3536(SB)/8, $697
DATA ·bitrev1024_r4+3544(SB)/8, $953
DATA ·bitrev1024_r4+3552(SB)/8, $249
DATA ·bitrev1024_r4+3560(SB)/8, $505
DATA ·bitrev1024_r4+3568(SB)/8, $761
DATA ·bitrev1024_r4+3576(SB)/8, $1017
DATA ·bitrev1024_r4+3584(SB)/8, $13
DATA ·bitrev1024_r4+3592(SB)/8, $269
DATA ·bitrev1024_r4+3600(SB)/8, $525
DATA ·bitrev1024_r4+3608(SB)/8, $781
DATA ·bitrev1024_r4+3616(SB)/8, $77
DATA ·bitrev1024_r4+3624(SB)/8, $333
DATA ·bitrev1024_r4+3632(SB)/8, $589
DATA ·bitrev1024_r4+3640(SB)/8, $845
DATA ·bitrev1024_r4+3648(SB)/8, $141
DATA ·bitrev1024_r4+3656(SB)/8, $397
DATA ·bitrev1024_r4+3664(SB)/8, $653
DATA ·bitrev1024_r4+3672(SB)/8, $909
DATA ·bitrev1024_r4+3680(SB)/8, $205
DATA ·bitrev1024_r4+3688(SB)/8, $461
DATA ·bitrev1024_r4+3696(SB)/8, $717
DATA ·bitrev1024_r4+3704(SB)/8, $973
DATA ·bitrev1024_r4+3712(SB)/8, $29
DATA ·bitrev1024_r4+3720(SB)/8, $285
DATA ·bitrev1024_r4+3728(SB)/8, $541
DATA ·bitrev1024_r4+3736(SB)/8, $797
DATA ·bitrev1024_r4+3744(SB)/8, $93
DATA ·bitrev1024_r4+3752(SB)/8, $349
DATA ·bitrev1024_r4+3760(SB)/8, $605
DATA ·bitrev1024_r4+3768(SB)/8, $861
DATA ·bitrev1024_r4+3776(SB)/8, $157
DATA ·bitrev1024_r4+3784(SB)/8, $413
DATA ·bitrev1024_r4+3792(SB)/8, $669
DATA ·bitrev1024_r4+3800(SB)/8, $925
DATA ·bitrev1024_r4+3808(SB)/8, $221
DATA ·bitrev1024_r4+3816(SB)/8, $477
DATA ·bitrev1024_r4+3824(SB)/8, $733
DATA ·bitrev1024_r4+3832(SB)/8, $989
DATA ·bitrev1024_r4+3840(SB)/8, $45
DATA ·bitrev1024_r4+3848(SB)/8, $301
DATA ·bitrev1024_r4+3856(SB)/8, $557
DATA ·bitrev1024_r4+3864(SB)/8, $813
DATA ·bitrev1024_r4+3872(SB)/8, $109
DATA ·bitrev1024_r4+3880(SB)/8, $365
DATA ·bitrev1024_r4+3888(SB)/8, $621
DATA ·bitrev1024_r4+3896(SB)/8, $877
DATA ·bitrev1024_r4+3904(SB)/8, $173
DATA ·bitrev1024_r4+3912(SB)/8, $429
DATA ·bitrev1024_r4+3920(SB)/8, $685
DATA ·bitrev1024_r4+3928(SB)/8, $941
DATA ·bitrev1024_r4+3936(SB)/8, $237
DATA ·bitrev1024_r4+3944(SB)/8, $493
DATA ·bitrev1024_r4+3952(SB)/8, $749
DATA ·bitrev1024_r4+3960(SB)/8, $1005
DATA ·bitrev1024_r4+3968(SB)/8, $61
DATA ·bitrev1024_r4+3976(SB)/8, $317
DATA ·bitrev1024_r4+3984(SB)/8, $573
DATA ·bitrev1024_r4+3992(SB)/8, $829
DATA ·bitrev1024_r4+4000(SB)/8, $125
DATA ·bitrev1024_r4+4008(SB)/8, $381
DATA ·bitrev1024_r4+4016(SB)/8, $637
DATA ·bitrev1024_r4+4024(SB)/8, $893
DATA ·bitrev1024_r4+4032(SB)/8, $189
DATA ·bitrev1024_r4+4040(SB)/8, $445
DATA ·bitrev1024_r4+4048(SB)/8, $701
DATA ·bitrev1024_r4+4056(SB)/8, $957
DATA ·bitrev1024_r4+4064(SB)/8, $253
DATA ·bitrev1024_r4+4072(SB)/8, $509
DATA ·bitrev1024_r4+4080(SB)/8, $765
DATA ·bitrev1024_r4+4088(SB)/8, $1021
DATA ·bitrev1024_r4+4096(SB)/8, $2
DATA ·bitrev1024_r4+4104(SB)/8, $258
DATA ·bitrev1024_r4+4112(SB)/8, $514
DATA ·bitrev1024_r4+4120(SB)/8, $770
DATA ·bitrev1024_r4+4128(SB)/8, $66
DATA ·bitrev1024_r4+4136(SB)/8, $322
DATA ·bitrev1024_r4+4144(SB)/8, $578
DATA ·bitrev1024_r4+4152(SB)/8, $834
DATA ·bitrev1024_r4+4160(SB)/8, $130
DATA ·bitrev1024_r4+4168(SB)/8, $386
DATA ·bitrev1024_r4+4176(SB)/8, $642
DATA ·bitrev1024_r4+4184(SB)/8, $898
DATA ·bitrev1024_r4+4192(SB)/8, $194
DATA ·bitrev1024_r4+4200(SB)/8, $450
DATA ·bitrev1024_r4+4208(SB)/8, $706
DATA ·bitrev1024_r4+4216(SB)/8, $962
DATA ·bitrev1024_r4+4224(SB)/8, $18
DATA ·bitrev1024_r4+4232(SB)/8, $274
DATA ·bitrev1024_r4+4240(SB)/8, $530
DATA ·bitrev1024_r4+4248(SB)/8, $786
DATA ·bitrev1024_r4+4256(SB)/8, $82
DATA ·bitrev1024_r4+4264(SB)/8, $338
DATA ·bitrev1024_r4+4272(SB)/8, $594
DATA ·bitrev1024_r4+4280(SB)/8, $850
DATA ·bitrev1024_r4+4288(SB)/8, $146
DATA ·bitrev1024_r4+4296(SB)/8, $402
DATA ·bitrev1024_r4+4304(SB)/8, $658
DATA ·bitrev1024_r4+4312(SB)/8, $914
DATA ·bitrev1024_r4+4320(SB)/8, $210
DATA ·bitrev1024_r4+4328(SB)/8, $466
DATA ·bitrev1024_r4+4336(SB)/8, $722
DATA ·bitrev1024_r4+4344(SB)/8, $978
DATA ·bitrev1024_r4+4352(SB)/8, $34
DATA ·bitrev1024_r4+4360(SB)/8, $290
DATA ·bitrev1024_r4+4368(SB)/8, $546
DATA ·bitrev1024_r4+4376(SB)/8, $802
DATA ·bitrev1024_r4+4384(SB)/8, $98
DATA ·bitrev1024_r4+4392(SB)/8, $354
DATA ·bitrev1024_r4+4400(SB)/8, $610
DATA ·bitrev1024_r4+4408(SB)/8, $866
DATA ·bitrev1024_r4+4416(SB)/8, $162
DATA ·bitrev1024_r4+4424(SB)/8, $418
DATA ·bitrev1024_r4+4432(SB)/8, $674
DATA ·bitrev1024_r4+4440(SB)/8, $930
DATA ·bitrev1024_r4+4448(SB)/8, $226
DATA ·bitrev1024_r4+4456(SB)/8, $482
DATA ·bitrev1024_r4+4464(SB)/8, $738
DATA ·bitrev1024_r4+4472(SB)/8, $994
DATA ·bitrev1024_r4+4480(SB)/8, $50
DATA ·bitrev1024_r4+4488(SB)/8, $306
DATA ·bitrev1024_r4+4496(SB)/8, $562
DATA ·bitrev1024_r4+4504(SB)/8, $818
DATA ·bitrev1024_r4+4512(SB)/8, $114
DATA ·bitrev1024_r4+4520(SB)/8, $370
DATA ·bitrev1024_r4+4528(SB)/8, $626
DATA ·bitrev1024_r4+4536(SB)/8, $882
DATA ·bitrev1024_r4+4544(SB)/8, $178
DATA ·bitrev1024_r4+4552(SB)/8, $434
DATA ·bitrev1024_r4+4560(SB)/8, $690
DATA ·bitrev1024_r4+4568(SB)/8, $946
DATA ·bitrev1024_r4+4576(SB)/8, $242
DATA ·bitrev1024_r4+4584(SB)/8, $498
DATA ·bitrev1024_r4+4592(SB)/8, $754
DATA ·bitrev1024_r4+4600(SB)/8, $1010
DATA ·bitrev1024_r4+4608(SB)/8, $6
DATA ·bitrev1024_r4+4616(SB)/8, $262
DATA ·bitrev1024_r4+4624(SB)/8, $518
DATA ·bitrev1024_r4+4632(SB)/8, $774
DATA ·bitrev1024_r4+4640(SB)/8, $70
DATA ·bitrev1024_r4+4648(SB)/8, $326
DATA ·bitrev1024_r4+4656(SB)/8, $582
DATA ·bitrev1024_r4+4664(SB)/8, $838
DATA ·bitrev1024_r4+4672(SB)/8, $134
DATA ·bitrev1024_r4+4680(SB)/8, $390
DATA ·bitrev1024_r4+4688(SB)/8, $646
DATA ·bitrev1024_r4+4696(SB)/8, $902
DATA ·bitrev1024_r4+4704(SB)/8, $198
DATA ·bitrev1024_r4+4712(SB)/8, $454
DATA ·bitrev1024_r4+4720(SB)/8, $710
DATA ·bitrev1024_r4+4728(SB)/8, $966
DATA ·bitrev1024_r4+4736(SB)/8, $22
DATA ·bitrev1024_r4+4744(SB)/8, $278
DATA ·bitrev1024_r4+4752(SB)/8, $534
DATA ·bitrev1024_r4+4760(SB)/8, $790
DATA ·bitrev1024_r4+4768(SB)/8, $86
DATA ·bitrev1024_r4+4776(SB)/8, $342
DATA ·bitrev1024_r4+4784(SB)/8, $598
DATA ·bitrev1024_r4+4792(SB)/8, $854
DATA ·bitrev1024_r4+4800(SB)/8, $150
DATA ·bitrev1024_r4+4808(SB)/8, $406
DATA ·bitrev1024_r4+4816(SB)/8, $662
DATA ·bitrev1024_r4+4824(SB)/8, $918
DATA ·bitrev1024_r4+4832(SB)/8, $214
DATA ·bitrev1024_r4+4840(SB)/8, $470
DATA ·bitrev1024_r4+4848(SB)/8, $726
DATA ·bitrev1024_r4+4856(SB)/8, $982
DATA ·bitrev1024_r4+4864(SB)/8, $38
DATA ·bitrev1024_r4+4872(SB)/8, $294
DATA ·bitrev1024_r4+4880(SB)/8, $550
DATA ·bitrev1024_r4+4888(SB)/8, $806
DATA ·bitrev1024_r4+4896(SB)/8, $102
DATA ·bitrev1024_r4+4904(SB)/8, $358
DATA ·bitrev1024_r4+4912(SB)/8, $614
DATA ·bitrev1024_r4+4920(SB)/8, $870
DATA ·bitrev1024_r4+4928(SB)/8, $166
DATA ·bitrev1024_r4+4936(SB)/8, $422
DATA ·bitrev1024_r4+4944(SB)/8, $678
DATA ·bitrev1024_r4+4952(SB)/8, $934
DATA ·bitrev1024_r4+4960(SB)/8, $230
DATA ·bitrev1024_r4+4968(SB)/8, $486
DATA ·bitrev1024_r4+4976(SB)/8, $742
DATA ·bitrev1024_r4+4984(SB)/8, $998
DATA ·bitrev1024_r4+4992(SB)/8, $54
DATA ·bitrev1024_r4+5000(SB)/8, $310
DATA ·bitrev1024_r4+5008(SB)/8, $566
DATA ·bitrev1024_r4+5016(SB)/8, $822
DATA ·bitrev1024_r4+5024(SB)/8, $118
DATA ·bitrev1024_r4+5032(SB)/8, $374
DATA ·bitrev1024_r4+5040(SB)/8, $630
DATA ·bitrev1024_r4+5048(SB)/8, $886
DATA ·bitrev1024_r4+5056(SB)/8, $182
DATA ·bitrev1024_r4+5064(SB)/8, $438
DATA ·bitrev1024_r4+5072(SB)/8, $694
DATA ·bitrev1024_r4+5080(SB)/8, $950
DATA ·bitrev1024_r4+5088(SB)/8, $246
DATA ·bitrev1024_r4+5096(SB)/8, $502
DATA ·bitrev1024_r4+5104(SB)/8, $758
DATA ·bitrev1024_r4+5112(SB)/8, $1014
DATA ·bitrev1024_r4+5120(SB)/8, $10
DATA ·bitrev1024_r4+5128(SB)/8, $266
DATA ·bitrev1024_r4+5136(SB)/8, $522
DATA ·bitrev1024_r4+5144(SB)/8, $778
DATA ·bitrev1024_r4+5152(SB)/8, $74
DATA ·bitrev1024_r4+5160(SB)/8, $330
DATA ·bitrev1024_r4+5168(SB)/8, $586
DATA ·bitrev1024_r4+5176(SB)/8, $842
DATA ·bitrev1024_r4+5184(SB)/8, $138
DATA ·bitrev1024_r4+5192(SB)/8, $394
DATA ·bitrev1024_r4+5200(SB)/8, $650
DATA ·bitrev1024_r4+5208(SB)/8, $906
DATA ·bitrev1024_r4+5216(SB)/8, $202
DATA ·bitrev1024_r4+5224(SB)/8, $458
DATA ·bitrev1024_r4+5232(SB)/8, $714
DATA ·bitrev1024_r4+5240(SB)/8, $970
DATA ·bitrev1024_r4+5248(SB)/8, $26
DATA ·bitrev1024_r4+5256(SB)/8, $282
DATA ·bitrev1024_r4+5264(SB)/8, $538
DATA ·bitrev1024_r4+5272(SB)/8, $794
DATA ·bitrev1024_r4+5280(SB)/8, $90
DATA ·bitrev1024_r4+5288(SB)/8, $346
DATA ·bitrev1024_r4+5296(SB)/8, $602
DATA ·bitrev1024_r4+5304(SB)/8, $858
DATA ·bitrev1024_r4+5312(SB)/8, $154
DATA ·bitrev1024_r4+5320(SB)/8, $410
DATA ·bitrev1024_r4+5328(SB)/8, $666
DATA ·bitrev1024_r4+5336(SB)/8, $922
DATA ·bitrev1024_r4+5344(SB)/8, $218
DATA ·bitrev1024_r4+5352(SB)/8, $474
DATA ·bitrev1024_r4+5360(SB)/8, $730
DATA ·bitrev1024_r4+5368(SB)/8, $986
DATA ·bitrev1024_r4+5376(SB)/8, $42
DATA ·bitrev1024_r4+5384(SB)/8, $298
DATA ·bitrev1024_r4+5392(SB)/8, $554
DATA ·bitrev1024_r4+5400(SB)/8, $810
DATA ·bitrev1024_r4+5408(SB)/8, $106
DATA ·bitrev1024_r4+5416(SB)/8, $362
DATA ·bitrev1024_r4+5424(SB)/8, $618
DATA ·bitrev1024_r4+5432(SB)/8, $874
DATA ·bitrev1024_r4+5440(SB)/8, $170
DATA ·bitrev1024_r4+5448(SB)/8, $426
DATA ·bitrev1024_r4+5456(SB)/8, $682
DATA ·bitrev1024_r4+5464(SB)/8, $938
DATA ·bitrev1024_r4+5472(SB)/8, $234
DATA ·bitrev1024_r4+5480(SB)/8, $490
DATA ·bitrev1024_r4+5488(SB)/8, $746
DATA ·bitrev1024_r4+5496(SB)/8, $1002
DATA ·bitrev1024_r4+5504(SB)/8, $58
DATA ·bitrev1024_r4+5512(SB)/8, $314
DATA ·bitrev1024_r4+5520(SB)/8, $570
DATA ·bitrev1024_r4+5528(SB)/8, $826
DATA ·bitrev1024_r4+5536(SB)/8, $122
DATA ·bitrev1024_r4+5544(SB)/8, $378
DATA ·bitrev1024_r4+5552(SB)/8, $634
DATA ·bitrev1024_r4+5560(SB)/8, $890
DATA ·bitrev1024_r4+5568(SB)/8, $186
DATA ·bitrev1024_r4+5576(SB)/8, $442
DATA ·bitrev1024_r4+5584(SB)/8, $698
DATA ·bitrev1024_r4+5592(SB)/8, $954
DATA ·bitrev1024_r4+5600(SB)/8, $250
DATA ·bitrev1024_r4+5608(SB)/8, $506
DATA ·bitrev1024_r4+5616(SB)/8, $762
DATA ·bitrev1024_r4+5624(SB)/8, $1018
DATA ·bitrev1024_r4+5632(SB)/8, $14
DATA ·bitrev1024_r4+5640(SB)/8, $270
DATA ·bitrev1024_r4+5648(SB)/8, $526
DATA ·bitrev1024_r4+5656(SB)/8, $782
DATA ·bitrev1024_r4+5664(SB)/8, $78
DATA ·bitrev1024_r4+5672(SB)/8, $334
DATA ·bitrev1024_r4+5680(SB)/8, $590
DATA ·bitrev1024_r4+5688(SB)/8, $846
DATA ·bitrev1024_r4+5696(SB)/8, $142
DATA ·bitrev1024_r4+5704(SB)/8, $398
DATA ·bitrev1024_r4+5712(SB)/8, $654
DATA ·bitrev1024_r4+5720(SB)/8, $910
DATA ·bitrev1024_r4+5728(SB)/8, $206
DATA ·bitrev1024_r4+5736(SB)/8, $462
DATA ·bitrev1024_r4+5744(SB)/8, $718
DATA ·bitrev1024_r4+5752(SB)/8, $974
DATA ·bitrev1024_r4+5760(SB)/8, $30
DATA ·bitrev1024_r4+5768(SB)/8, $286
DATA ·bitrev1024_r4+5776(SB)/8, $542
DATA ·bitrev1024_r4+5784(SB)/8, $798
DATA ·bitrev1024_r4+5792(SB)/8, $94
DATA ·bitrev1024_r4+5800(SB)/8, $350
DATA ·bitrev1024_r4+5808(SB)/8, $606
DATA ·bitrev1024_r4+5816(SB)/8, $862
DATA ·bitrev1024_r4+5824(SB)/8, $158
DATA ·bitrev1024_r4+5832(SB)/8, $414
DATA ·bitrev1024_r4+5840(SB)/8, $670
DATA ·bitrev1024_r4+5848(SB)/8, $926
DATA ·bitrev1024_r4+5856(SB)/8, $222
DATA ·bitrev1024_r4+5864(SB)/8, $478
DATA ·bitrev1024_r4+5872(SB)/8, $734
DATA ·bitrev1024_r4+5880(SB)/8, $990
DATA ·bitrev1024_r4+5888(SB)/8, $46
DATA ·bitrev1024_r4+5896(SB)/8, $302
DATA ·bitrev1024_r4+5904(SB)/8, $558
DATA ·bitrev1024_r4+5912(SB)/8, $814
DATA ·bitrev1024_r4+5920(SB)/8, $110
DATA ·bitrev1024_r4+5928(SB)/8, $366
DATA ·bitrev1024_r4+5936(SB)/8, $622
DATA ·bitrev1024_r4+5944(SB)/8, $878
DATA ·bitrev1024_r4+5952(SB)/8, $174
DATA ·bitrev1024_r4+5960(SB)/8, $430
DATA ·bitrev1024_r4+5968(SB)/8, $686
DATA ·bitrev1024_r4+5976(SB)/8, $942
DATA ·bitrev1024_r4+5984(SB)/8, $238
DATA ·bitrev1024_r4+5992(SB)/8, $494
DATA ·bitrev1024_r4+6000(SB)/8, $750
DATA ·bitrev1024_r4+6008(SB)/8, $1006
DATA ·bitrev1024_r4+6016(SB)/8, $62
DATA ·bitrev1024_r4+6024(SB)/8, $318
DATA ·bitrev1024_r4+6032(SB)/8, $574
DATA ·bitrev1024_r4+6040(SB)/8, $830
DATA ·bitrev1024_r4+6048(SB)/8, $126
DATA ·bitrev1024_r4+6056(SB)/8, $382
DATA ·bitrev1024_r4+6064(SB)/8, $638
DATA ·bitrev1024_r4+6072(SB)/8, $894
DATA ·bitrev1024_r4+6080(SB)/8, $190
DATA ·bitrev1024_r4+6088(SB)/8, $446
DATA ·bitrev1024_r4+6096(SB)/8, $702
DATA ·bitrev1024_r4+6104(SB)/8, $958
DATA ·bitrev1024_r4+6112(SB)/8, $254
DATA ·bitrev1024_r4+6120(SB)/8, $510
DATA ·bitrev1024_r4+6128(SB)/8, $766
DATA ·bitrev1024_r4+6136(SB)/8, $1022
DATA ·bitrev1024_r4+6144(SB)/8, $3
DATA ·bitrev1024_r4+6152(SB)/8, $259
DATA ·bitrev1024_r4+6160(SB)/8, $515
DATA ·bitrev1024_r4+6168(SB)/8, $771
DATA ·bitrev1024_r4+6176(SB)/8, $67
DATA ·bitrev1024_r4+6184(SB)/8, $323
DATA ·bitrev1024_r4+6192(SB)/8, $579
DATA ·bitrev1024_r4+6200(SB)/8, $835
DATA ·bitrev1024_r4+6208(SB)/8, $131
DATA ·bitrev1024_r4+6216(SB)/8, $387
DATA ·bitrev1024_r4+6224(SB)/8, $643
DATA ·bitrev1024_r4+6232(SB)/8, $899
DATA ·bitrev1024_r4+6240(SB)/8, $195
DATA ·bitrev1024_r4+6248(SB)/8, $451
DATA ·bitrev1024_r4+6256(SB)/8, $707
DATA ·bitrev1024_r4+6264(SB)/8, $963
DATA ·bitrev1024_r4+6272(SB)/8, $19
DATA ·bitrev1024_r4+6280(SB)/8, $275
DATA ·bitrev1024_r4+6288(SB)/8, $531
DATA ·bitrev1024_r4+6296(SB)/8, $787
DATA ·bitrev1024_r4+6304(SB)/8, $83
DATA ·bitrev1024_r4+6312(SB)/8, $339
DATA ·bitrev1024_r4+6320(SB)/8, $595
DATA ·bitrev1024_r4+6328(SB)/8, $851
DATA ·bitrev1024_r4+6336(SB)/8, $147
DATA ·bitrev1024_r4+6344(SB)/8, $403
DATA ·bitrev1024_r4+6352(SB)/8, $659
DATA ·bitrev1024_r4+6360(SB)/8, $915
DATA ·bitrev1024_r4+6368(SB)/8, $211
DATA ·bitrev1024_r4+6376(SB)/8, $467
DATA ·bitrev1024_r4+6384(SB)/8, $723
DATA ·bitrev1024_r4+6392(SB)/8, $979
DATA ·bitrev1024_r4+6400(SB)/8, $35
DATA ·bitrev1024_r4+6408(SB)/8, $291
DATA ·bitrev1024_r4+6416(SB)/8, $547
DATA ·bitrev1024_r4+6424(SB)/8, $803
DATA ·bitrev1024_r4+6432(SB)/8, $99
DATA ·bitrev1024_r4+6440(SB)/8, $355
DATA ·bitrev1024_r4+6448(SB)/8, $611
DATA ·bitrev1024_r4+6456(SB)/8, $867
DATA ·bitrev1024_r4+6464(SB)/8, $163
DATA ·bitrev1024_r4+6472(SB)/8, $419
DATA ·bitrev1024_r4+6480(SB)/8, $675
DATA ·bitrev1024_r4+6488(SB)/8, $931
DATA ·bitrev1024_r4+6496(SB)/8, $227
DATA ·bitrev1024_r4+6504(SB)/8, $483
DATA ·bitrev1024_r4+6512(SB)/8, $739
DATA ·bitrev1024_r4+6520(SB)/8, $995
DATA ·bitrev1024_r4+6528(SB)/8, $51
DATA ·bitrev1024_r4+6536(SB)/8, $307
DATA ·bitrev1024_r4+6544(SB)/8, $563
DATA ·bitrev1024_r4+6552(SB)/8, $819
DATA ·bitrev1024_r4+6560(SB)/8, $115
DATA ·bitrev1024_r4+6568(SB)/8, $371
DATA ·bitrev1024_r4+6576(SB)/8, $627
DATA ·bitrev1024_r4+6584(SB)/8, $883
DATA ·bitrev1024_r4+6592(SB)/8, $179
DATA ·bitrev1024_r4+6600(SB)/8, $435
DATA ·bitrev1024_r4+6608(SB)/8, $691
DATA ·bitrev1024_r4+6616(SB)/8, $947
DATA ·bitrev1024_r4+6624(SB)/8, $243
DATA ·bitrev1024_r4+6632(SB)/8, $499
DATA ·bitrev1024_r4+6640(SB)/8, $755
DATA ·bitrev1024_r4+6648(SB)/8, $1011
DATA ·bitrev1024_r4+6656(SB)/8, $7
DATA ·bitrev1024_r4+6664(SB)/8, $263
DATA ·bitrev1024_r4+6672(SB)/8, $519
DATA ·bitrev1024_r4+6680(SB)/8, $775
DATA ·bitrev1024_r4+6688(SB)/8, $71
DATA ·bitrev1024_r4+6696(SB)/8, $327
DATA ·bitrev1024_r4+6704(SB)/8, $583
DATA ·bitrev1024_r4+6712(SB)/8, $839
DATA ·bitrev1024_r4+6720(SB)/8, $135
DATA ·bitrev1024_r4+6728(SB)/8, $391
DATA ·bitrev1024_r4+6736(SB)/8, $647
DATA ·bitrev1024_r4+6744(SB)/8, $903
DATA ·bitrev1024_r4+6752(SB)/8, $199
DATA ·bitrev1024_r4+6760(SB)/8, $455
DATA ·bitrev1024_r4+6768(SB)/8, $711
DATA ·bitrev1024_r4+6776(SB)/8, $967
DATA ·bitrev1024_r4+6784(SB)/8, $23
DATA ·bitrev1024_r4+6792(SB)/8, $279
DATA ·bitrev1024_r4+6800(SB)/8, $535
DATA ·bitrev1024_r4+6808(SB)/8, $791
DATA ·bitrev1024_r4+6816(SB)/8, $87
DATA ·bitrev1024_r4+6824(SB)/8, $343
DATA ·bitrev1024_r4+6832(SB)/8, $599
DATA ·bitrev1024_r4+6840(SB)/8, $855
DATA ·bitrev1024_r4+6848(SB)/8, $151
DATA ·bitrev1024_r4+6856(SB)/8, $407
DATA ·bitrev1024_r4+6864(SB)/8, $663
DATA ·bitrev1024_r4+6872(SB)/8, $919
DATA ·bitrev1024_r4+6880(SB)/8, $215
DATA ·bitrev1024_r4+6888(SB)/8, $471
DATA ·bitrev1024_r4+6896(SB)/8, $727
DATA ·bitrev1024_r4+6904(SB)/8, $983
DATA ·bitrev1024_r4+6912(SB)/8, $39
DATA ·bitrev1024_r4+6920(SB)/8, $295
DATA ·bitrev1024_r4+6928(SB)/8, $551
DATA ·bitrev1024_r4+6936(SB)/8, $807
DATA ·bitrev1024_r4+6944(SB)/8, $103
DATA ·bitrev1024_r4+6952(SB)/8, $359
DATA ·bitrev1024_r4+6960(SB)/8, $615
DATA ·bitrev1024_r4+6968(SB)/8, $871
DATA ·bitrev1024_r4+6976(SB)/8, $167
DATA ·bitrev1024_r4+6984(SB)/8, $423
DATA ·bitrev1024_r4+6992(SB)/8, $679
DATA ·bitrev1024_r4+7000(SB)/8, $935
DATA ·bitrev1024_r4+7008(SB)/8, $231
DATA ·bitrev1024_r4+7016(SB)/8, $487
DATA ·bitrev1024_r4+7024(SB)/8, $743
DATA ·bitrev1024_r4+7032(SB)/8, $999
DATA ·bitrev1024_r4+7040(SB)/8, $55
DATA ·bitrev1024_r4+7048(SB)/8, $311
DATA ·bitrev1024_r4+7056(SB)/8, $567
DATA ·bitrev1024_r4+7064(SB)/8, $823
DATA ·bitrev1024_r4+7072(SB)/8, $119
DATA ·bitrev1024_r4+7080(SB)/8, $375
DATA ·bitrev1024_r4+7088(SB)/8, $631
DATA ·bitrev1024_r4+7096(SB)/8, $887
DATA ·bitrev1024_r4+7104(SB)/8, $183
DATA ·bitrev1024_r4+7112(SB)/8, $439
DATA ·bitrev1024_r4+7120(SB)/8, $695
DATA ·bitrev1024_r4+7128(SB)/8, $951
DATA ·bitrev1024_r4+7136(SB)/8, $247
DATA ·bitrev1024_r4+7144(SB)/8, $503
DATA ·bitrev1024_r4+7152(SB)/8, $759
DATA ·bitrev1024_r4+7160(SB)/8, $1015
DATA ·bitrev1024_r4+7168(SB)/8, $11
DATA ·bitrev1024_r4+7176(SB)/8, $267
DATA ·bitrev1024_r4+7184(SB)/8, $523
DATA ·bitrev1024_r4+7192(SB)/8, $779
DATA ·bitrev1024_r4+7200(SB)/8, $75
DATA ·bitrev1024_r4+7208(SB)/8, $331
DATA ·bitrev1024_r4+7216(SB)/8, $587
DATA ·bitrev1024_r4+7224(SB)/8, $843
DATA ·bitrev1024_r4+7232(SB)/8, $139
DATA ·bitrev1024_r4+7240(SB)/8, $395
DATA ·bitrev1024_r4+7248(SB)/8, $651
DATA ·bitrev1024_r4+7256(SB)/8, $907
DATA ·bitrev1024_r4+7264(SB)/8, $203
DATA ·bitrev1024_r4+7272(SB)/8, $459
DATA ·bitrev1024_r4+7280(SB)/8, $715
DATA ·bitrev1024_r4+7288(SB)/8, $971
DATA ·bitrev1024_r4+7296(SB)/8, $27
DATA ·bitrev1024_r4+7304(SB)/8, $283
DATA ·bitrev1024_r4+7312(SB)/8, $539
DATA ·bitrev1024_r4+7320(SB)/8, $795
DATA ·bitrev1024_r4+7328(SB)/8, $91
DATA ·bitrev1024_r4+7336(SB)/8, $347
DATA ·bitrev1024_r4+7344(SB)/8, $603
DATA ·bitrev1024_r4+7352(SB)/8, $859
DATA ·bitrev1024_r4+7360(SB)/8, $155
DATA ·bitrev1024_r4+7368(SB)/8, $411
DATA ·bitrev1024_r4+7376(SB)/8, $667
DATA ·bitrev1024_r4+7384(SB)/8, $923
DATA ·bitrev1024_r4+7392(SB)/8, $219
DATA ·bitrev1024_r4+7400(SB)/8, $475
DATA ·bitrev1024_r4+7408(SB)/8, $731
DATA ·bitrev1024_r4+7416(SB)/8, $987
DATA ·bitrev1024_r4+7424(SB)/8, $43
DATA ·bitrev1024_r4+7432(SB)/8, $299
DATA ·bitrev1024_r4+7440(SB)/8, $555
DATA ·bitrev1024_r4+7448(SB)/8, $811
DATA ·bitrev1024_r4+7456(SB)/8, $107
DATA ·bitrev1024_r4+7464(SB)/8, $363
DATA ·bitrev1024_r4+7472(SB)/8, $619
DATA ·bitrev1024_r4+7480(SB)/8, $875
DATA ·bitrev1024_r4+7488(SB)/8, $171
DATA ·bitrev1024_r4+7496(SB)/8, $427
DATA ·bitrev1024_r4+7504(SB)/8, $683
DATA ·bitrev1024_r4+7512(SB)/8, $939
DATA ·bitrev1024_r4+7520(SB)/8, $235
DATA ·bitrev1024_r4+7528(SB)/8, $491
DATA ·bitrev1024_r4+7536(SB)/8, $747
DATA ·bitrev1024_r4+7544(SB)/8, $1003
DATA ·bitrev1024_r4+7552(SB)/8, $59
DATA ·bitrev1024_r4+7560(SB)/8, $315
DATA ·bitrev1024_r4+7568(SB)/8, $571
DATA ·bitrev1024_r4+7576(SB)/8, $827
DATA ·bitrev1024_r4+7584(SB)/8, $123
DATA ·bitrev1024_r4+7592(SB)/8, $379
DATA ·bitrev1024_r4+7600(SB)/8, $635
DATA ·bitrev1024_r4+7608(SB)/8, $891
DATA ·bitrev1024_r4+7616(SB)/8, $187
DATA ·bitrev1024_r4+7624(SB)/8, $443
DATA ·bitrev1024_r4+7632(SB)/8, $699
DATA ·bitrev1024_r4+7640(SB)/8, $955
DATA ·bitrev1024_r4+7648(SB)/8, $251
DATA ·bitrev1024_r4+7656(SB)/8, $507
DATA ·bitrev1024_r4+7664(SB)/8, $763
DATA ·bitrev1024_r4+7672(SB)/8, $1019
DATA ·bitrev1024_r4+7680(SB)/8, $15
DATA ·bitrev1024_r4+7688(SB)/8, $271
DATA ·bitrev1024_r4+7696(SB)/8, $527
DATA ·bitrev1024_r4+7704(SB)/8, $783
DATA ·bitrev1024_r4+7712(SB)/8, $79
DATA ·bitrev1024_r4+7720(SB)/8, $335
DATA ·bitrev1024_r4+7728(SB)/8, $591
DATA ·bitrev1024_r4+7736(SB)/8, $847
DATA ·bitrev1024_r4+7744(SB)/8, $143
DATA ·bitrev1024_r4+7752(SB)/8, $399
DATA ·bitrev1024_r4+7760(SB)/8, $655
DATA ·bitrev1024_r4+7768(SB)/8, $911
DATA ·bitrev1024_r4+7776(SB)/8, $207
DATA ·bitrev1024_r4+7784(SB)/8, $463
DATA ·bitrev1024_r4+7792(SB)/8, $719
DATA ·bitrev1024_r4+7800(SB)/8, $975
DATA ·bitrev1024_r4+7808(SB)/8, $31
DATA ·bitrev1024_r4+7816(SB)/8, $287
DATA ·bitrev1024_r4+7824(SB)/8, $543
DATA ·bitrev1024_r4+7832(SB)/8, $799
DATA ·bitrev1024_r4+7840(SB)/8, $95
DATA ·bitrev1024_r4+7848(SB)/8, $351
DATA ·bitrev1024_r4+7856(SB)/8, $607
DATA ·bitrev1024_r4+7864(SB)/8, $863
DATA ·bitrev1024_r4+7872(SB)/8, $159
DATA ·bitrev1024_r4+7880(SB)/8, $415
DATA ·bitrev1024_r4+7888(SB)/8, $671
DATA ·bitrev1024_r4+7896(SB)/8, $927
DATA ·bitrev1024_r4+7904(SB)/8, $223
DATA ·bitrev1024_r4+7912(SB)/8, $479
DATA ·bitrev1024_r4+7920(SB)/8, $735
DATA ·bitrev1024_r4+7928(SB)/8, $991
DATA ·bitrev1024_r4+7936(SB)/8, $47
DATA ·bitrev1024_r4+7944(SB)/8, $303
DATA ·bitrev1024_r4+7952(SB)/8, $559
DATA ·bitrev1024_r4+7960(SB)/8, $815
DATA ·bitrev1024_r4+7968(SB)/8, $111
DATA ·bitrev1024_r4+7976(SB)/8, $367
DATA ·bitrev1024_r4+7984(SB)/8, $623
DATA ·bitrev1024_r4+7992(SB)/8, $879
DATA ·bitrev1024_r4+8000(SB)/8, $175
DATA ·bitrev1024_r4+8008(SB)/8, $431
DATA ·bitrev1024_r4+8016(SB)/8, $687
DATA ·bitrev1024_r4+8024(SB)/8, $943
DATA ·bitrev1024_r4+8032(SB)/8, $239
DATA ·bitrev1024_r4+8040(SB)/8, $495
DATA ·bitrev1024_r4+8048(SB)/8, $751
DATA ·bitrev1024_r4+8056(SB)/8, $1007
DATA ·bitrev1024_r4+8064(SB)/8, $63
DATA ·bitrev1024_r4+8072(SB)/8, $319
DATA ·bitrev1024_r4+8080(SB)/8, $575
DATA ·bitrev1024_r4+8088(SB)/8, $831
DATA ·bitrev1024_r4+8096(SB)/8, $127
DATA ·bitrev1024_r4+8104(SB)/8, $383
DATA ·bitrev1024_r4+8112(SB)/8, $639
DATA ·bitrev1024_r4+8120(SB)/8, $895
DATA ·bitrev1024_r4+8128(SB)/8, $191
DATA ·bitrev1024_r4+8136(SB)/8, $447
DATA ·bitrev1024_r4+8144(SB)/8, $703
DATA ·bitrev1024_r4+8152(SB)/8, $959
DATA ·bitrev1024_r4+8160(SB)/8, $255
DATA ·bitrev1024_r4+8168(SB)/8, $511
DATA ·bitrev1024_r4+8176(SB)/8, $767
DATA ·bitrev1024_r4+8184(SB)/8, $1023
GLOBL ·bitrev1024_r4(SB), RODATA, $8192

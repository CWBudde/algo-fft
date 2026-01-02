//go:build amd64 && fft_asm && !purego

// ===========================================================================
// AVX2 Size-128 Radix-4 FFT Kernels for AMD64 (complex64)
// ===========================================================================
//
// Size 128 = 2 x 4^3, implemented as:
//   Stage 1: 32 radix-4 butterflies, stride=4, twiddle=1 (no multiply)
//   Stage 2: 8 groups x 4 butterflies, stride=16
//   Stage 3: 2 groups x 16 butterflies, stride=64
//   Stage 4: 32 radix-2 butterflies, stride=128
//
// This reduces from 7 stages (radix-2) to 4 stages (mixed radix).
//
// ===========================================================================

#include "textflag.h"

// Forward transform, size 128, complex64, radix-4 variant
TEXT ·ForwardAVX2Size128Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 128)

	// Verify n == 128
	CMPQ R13, $128
	JNE  size128_r4_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $128
	JL   size128_r4_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $128
	JL   size128_r4_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $128
	JL   size128_r4_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $128
	JL   size128_r4_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size128_r4_use_dst
	MOVQ R11, R8             // In-place: use scratch

size128_r4_use_dst:
	// ==================================================================
	// Bit-reversal permutation (copy src[bitrev[i]] -> work[i])
	// ==================================================================
	XORQ CX, CX

size128_r4_bitrev_loop:
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[i]] (8 bytes = complex64)
	MOVQ AX, (R8)(CX*8)      // work[i] = src[bitrev[i]]
	INCQ CX
	CMPQ CX, $128
	JL   size128_r4_bitrev_loop

size128_r4_stage1:
	// ==================================================================
	// Stage 1: 32 radix-4 butterflies, stride=4, twiddle=1
	// Process groups of 4 elements: [0,1,2,3], [4,5,6,7], ...
	// ==================================================================
	XORQ CX, CX              // CX = base index (0, 4, 8, ..., 124)

size128_r4_stage1_loop:
	CMPQ CX, $128
	JGE  size128_r4_stage2

	// Load 4 complex64 values
	LEAQ (R8)(CX*8), SI
	VMOVSD (SI), X0          // a0
	VMOVSD 8(SI), X1         // a1
	VMOVSD 16(SI), X2        // a2
	VMOVSD 24(SI), X3        // a3

	// Radix-4 butterfly (twiddle = 1)
	// t0 = a0 + a2, t1 = a0 - a2, t2 = a1 + a3, t3 = a1 - a3
	VADDPS X0, X2, X4        // t0 = a0 + a2
	VSUBPS X2, X0, X5        // t1 = a0 - a2
	VADDPS X1, X3, X6        // t2 = a1 + a3
	VSUBPS X3, X1, X7        // t3 = a1 - a3

	// (-i)*t3 = (im, -re) for y1 output
	VPERMILPS $0xB1, X7, X8  // swap re/im
	VXORPS X9, X9, X9
	VSUBPS X8, X9, X10
	VBLENDPS $0x02, X10, X8, X8  // X8 = (im, -re)

	// i*t3 = (-im, re) for y3 output
	VPERMILPS $0xB1, X7, X11
	VSUBPS X11, X9, X10
	VBLENDPS $0x01, X10, X11, X11  // X11 = (-im, re)

	// Final outputs
	VADDPS X4, X6, X0        // y0 = t0 + t2
	VADDPS X5, X8, X1        // y1 = t1 + (-i)*t3
	VSUBPS X6, X4, X2        // y2 = t0 - t2
	VADDPS X5, X11, X3       // y3 = t1 + i*t3

	// Store results
	VMOVSD X0, (SI)
	VMOVSD X1, 8(SI)
	VMOVSD X2, 16(SI)
	VMOVSD X3, 24(SI)

	ADDQ $4, CX
	JMP  size128_r4_stage1_loop

size128_r4_stage2:
	// ==================================================================
	// Stage 2: 8 groups x 4 butterflies, stride=16
	// Group g processes: indices [g*16 + j], [g*16 + j + 4], [g*16 + j + 8], [g*16 + j + 12]
	// for j = 0,1,2,3
	// ==================================================================
	XORQ BX, BX              // BX = group index (0..7)

size128_r4_stage2_outer:
	CMPQ BX, $8
	JGE  size128_r4_stage3

	XORQ DX, DX              // DX = butterfly index within group (0..3)

size128_r4_stage2_inner:
	CMPQ DX, $4
	JGE  size128_r4_stage2_next_group

	// Calculate base index: BX*16 + DX
	MOVQ BX, SI
	SHLQ $4, SI              // SI = BX * 16
	ADDQ DX, SI              // SI = base index

	// Load indices for 4-point butterfly
	// idx0 = SI, idx1 = SI+4, idx2 = SI+8, idx3 = SI+12
	MOVQ SI, DI
	ADDQ $4, DI              // idx1
	MOVQ SI, R14
	ADDQ $8, R14             // idx2
	MOVQ SI, R15
	ADDQ $12, R15            // idx3

	// Load twiddle factors: twiddle[DX*8], twiddle[DX*16], twiddle[DX*24]
	// For stage 2 with stride 16: twiddle indices are j*8, 2*j*8, 3*j*8
	MOVQ DX, CX
	SHLQ $3, CX              // CX = DX * 8
	VMOVSD (R10)(CX*8), X8   // w1 = twiddle[DX*8]

	MOVQ DX, CX
	SHLQ $4, CX              // CX = DX * 16
	VMOVSD (R10)(CX*8), X9   // w2 = twiddle[DX*16]

	MOVQ DX, CX
	IMULQ $24, CX            // CX = DX * 24
	VMOVSD (R10)(CX*8), X10  // w3 = twiddle[DX*24]

	// Load data
	VMOVSD (R8)(SI*8), X0    // a0
	VMOVSD (R8)(DI*8), X1    // a1
	VMOVSD (R8)(R14*8), X2   // a2
	VMOVSD (R8)(R15*8), X3   // a3

	// Complex multiply a1*w1
	VMOVSLDUP X8, X11        // broadcast w1.real
	VMOVSHDUP X8, X12        // broadcast w1.imag
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	// Complex multiply a2*w2
	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X2, X13
	VMOVAPS X13, X2

	// Complex multiply a3*w3
	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X3, X13
	VMOVAPS X13, X3

	// Radix-4 butterfly
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

	// Outputs
	VADDPS X4, X6, X0
	VADDPS X5, X14, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X12, X3

	// Store
	VMOVSD X0, (R8)(SI*8)
	VMOVSD X1, (R8)(DI*8)
	VMOVSD X2, (R8)(R14*8)
	VMOVSD X3, (R8)(R15*8)

	INCQ DX
	JMP  size128_r4_stage2_inner

size128_r4_stage2_next_group:
	INCQ BX
	JMP  size128_r4_stage2_outer

size128_r4_stage3:
	// ==================================================================
	// Stage 3: 2 groups x 16 butterflies, stride=64
	// Group g processes: [g*64 + j], [g*64 + j + 16], [g*64 + j + 32], [g*64 + j + 48]
	// for j = 0..15
	// ==================================================================
	XORQ BX, BX              // BX = group index (0..1)

size128_r4_stage3_outer:
	CMPQ BX, $2
	JGE  size128_r4_stage4

	XORQ DX, DX              // DX = butterfly index (0..15)

size128_r4_stage3_inner:
	CMPQ DX, $16
	JGE  size128_r4_stage3_next_group

	// Calculate base index: BX*64 + DX
	MOVQ BX, SI
	SHLQ $6, SI              // SI = BX * 64
	ADDQ DX, SI              // SI = base index

	// Load indices
	MOVQ SI, DI
	ADDQ $16, DI             // idx1
	MOVQ SI, R14
	ADDQ $32, R14            // idx2
	MOVQ SI, R15
	ADDQ $48, R15            // idx3

	// Twiddle factors: twiddle[DX*2], twiddle[DX*4], twiddle[DX*6]
	MOVQ DX, CX
	SHLQ $1, CX              // CX = DX * 2
	VMOVSD (R10)(CX*8), X8   // w1

	MOVQ DX, CX
	SHLQ $2, CX              // CX = DX * 4
	VMOVSD (R10)(CX*8), X9   // w2

	MOVQ DX, CX
	IMULQ $6, CX             // CX = DX * 6
	VMOVSD (R10)(CX*8), X10  // w3

	// Load data
	VMOVSD (R8)(SI*8), X0
	VMOVSD (R8)(DI*8), X1
	VMOVSD (R8)(R14*8), X2
	VMOVSD (R8)(R15*8), X3

	// Complex multiply a1*w1
	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	// Complex multiply a2*w2
	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X2, X13
	VMOVAPS X13, X2

	// Complex multiply a3*w3
	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X3, X13
	VMOVAPS X13, X3

	// Radix-4 butterfly
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

	VMOVSD X0, (R8)(SI*8)
	VMOVSD X1, (R8)(DI*8)
	VMOVSD X2, (R8)(R14*8)
	VMOVSD X3, (R8)(R15*8)

	INCQ DX
	JMP  size128_r4_stage3_inner

size128_r4_stage3_next_group:
	INCQ BX
	JMP  size128_r4_stage3_outer

size128_r4_stage4:
	// ==================================================================
	// Stage 4: 32 radix-2 butterflies, stride=128
	// Pairs: [j, j+64] for j = 0..63
	// ==================================================================
	XORQ DX, DX              // DX = butterfly index (0..63)

size128_r4_stage4_loop:
	CMPQ DX, $64
	JGE  size128_r4_done

	// Indices: DX and DX+64
	MOVQ DX, SI
	MOVQ DX, DI
	ADDQ $64, DI

	// Twiddle factor: twiddle[DX]
	VMOVSD (R10)(DX*8), X8

	// Load data
	VMOVSD (R8)(SI*8), X0    // a
	VMOVSD (R8)(DI*8), X1    // b

	// Complex multiply b*w
	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1          // b' = b * w

	// Radix-2 butterfly
	VADDPS X0, X1, X2        // y0 = a + b'
	VSUBPS X1, X0, X3        // y1 = a - b'

	// Store
	VMOVSD X2, (R8)(SI*8)
	VMOVSD X3, (R8)(DI*8)

	INCQ DX
	JMP  size128_r4_stage4_loop

size128_r4_done:
	// Copy results to dst if we used scratch buffer
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size128_r4_done_direct

	XORQ CX, CX

size128_r4_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $1024           // 128 * 8 bytes
	JL   size128_r4_copy_loop

size128_r4_done_direct:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size128_r4_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 128, complex64, radix-4
// ===========================================================================
TEXT ·InverseAVX2Size128Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	// Verify n == 128
	CMPQ R13, $128
	JNE  size128_r4_inv_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $128
	JL   size128_r4_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $128
	JL   size128_r4_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $128
	JL   size128_r4_inv_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $128
	JL   size128_r4_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size128_r4_inv_use_dst
	MOVQ R11, R8

size128_r4_inv_use_dst:
	// Bit-reversal permutation
	XORQ CX, CX

size128_r4_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	CMPQ CX, $128
	JL   size128_r4_inv_bitrev_loop

size128_r4_inv_stage1:
	// ==================================================================
	// Stage 1: 32 radix-4 butterflies, twiddle=1 (inverse: swap i/-i)
	// ==================================================================
	XORQ CX, CX

size128_r4_inv_stage1_loop:
	CMPQ CX, $128
	JGE  size128_r4_inv_stage2

	LEAQ (R8)(CX*8), SI
	VMOVSD (SI), X0
	VMOVSD 8(SI), X1
	VMOVSD 16(SI), X2
	VMOVSD 24(SI), X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	// For inverse: use i*t3 for y1, (-i)*t3 for y3 (swapped from forward)
	// i*t3 = (-im, re)
	VPERMILPS $0xB1, X7, X11
	VXORPS X9, X9, X9
	VSUBPS X11, X9, X10
	VBLENDPS $0x01, X10, X11, X11

	// (-i)*t3 = (im, -re)
	VPERMILPS $0xB1, X7, X8
	VSUBPS X8, X9, X10
	VBLENDPS $0x02, X10, X8, X8

	VADDPS X4, X6, X0
	VADDPS X5, X11, X1       // y1 = t1 + i*t3
	VSUBPS X6, X4, X2
	VADDPS X5, X8, X3        // y3 = t1 + (-i)*t3

	VMOVSD X0, (SI)
	VMOVSD X1, 8(SI)
	VMOVSD X2, 16(SI)
	VMOVSD X3, 24(SI)

	ADDQ $4, CX
	JMP  size128_r4_inv_stage1_loop

size128_r4_inv_stage2:
	// ==================================================================
	// Stage 2: 8 groups x 4 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ BX, BX

size128_r4_inv_stage2_outer:
	CMPQ BX, $8
	JGE  size128_r4_inv_stage3

	XORQ DX, DX

size128_r4_inv_stage2_inner:
	CMPQ DX, $4
	JGE  size128_r4_inv_stage2_next

	MOVQ BX, SI
	SHLQ $4, SI
	ADDQ DX, SI

	MOVQ SI, DI
	ADDQ $4, DI
	MOVQ SI, R14
	ADDQ $8, R14
	MOVQ SI, R15
	ADDQ $12, R15

	// Load twiddle factors
	MOVQ DX, CX
	SHLQ $3, CX
	VMOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $4, CX
	VMOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $24, CX
	VMOVSD (R10)(CX*8), X10

	// Load data
	VMOVSD (R8)(SI*8), X0
	VMOVSD (R8)(DI*8), X1
	VMOVSD (R8)(R14*8), X2
	VMOVSD (R8)(R15*8), X3

	// Conjugate complex multiply: use VFMSUBADD instead of VFMADDSUB
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

	// Radix-4 butterfly (inverse)
	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	// i*t3 for y1
	VPERMILPS $0xB1, X7, X12
	VXORPS X15, X15, X15
	VSUBPS X12, X15, X11
	VBLENDPS $0x01, X11, X12, X12

	// (-i)*t3 for y3
	VPERMILPS $0xB1, X7, X14
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14

	VADDPS X4, X6, X0
	VADDPS X5, X12, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X14, X3

	VMOVSD X0, (R8)(SI*8)
	VMOVSD X1, (R8)(DI*8)
	VMOVSD X2, (R8)(R14*8)
	VMOVSD X3, (R8)(R15*8)

	INCQ DX
	JMP  size128_r4_inv_stage2_inner

size128_r4_inv_stage2_next:
	INCQ BX
	JMP  size128_r4_inv_stage2_outer

size128_r4_inv_stage3:
	// ==================================================================
	// Stage 3: 2 groups x 16 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ BX, BX

size128_r4_inv_stage3_outer:
	CMPQ BX, $2
	JGE  size128_r4_inv_stage4

	XORQ DX, DX

size128_r4_inv_stage3_inner:
	CMPQ DX, $16
	JGE  size128_r4_inv_stage3_next

	MOVQ BX, SI
	SHLQ $6, SI
	ADDQ DX, SI

	MOVQ SI, DI
	ADDQ $16, DI
	MOVQ SI, R14
	ADDQ $32, R14
	MOVQ SI, R15
	ADDQ $48, R15

	MOVQ DX, CX
	SHLQ $1, CX
	VMOVSD (R10)(CX*8), X8

	MOVQ DX, CX
	SHLQ $2, CX
	VMOVSD (R10)(CX*8), X9

	MOVQ DX, CX
	IMULQ $6, CX
	VMOVSD (R10)(CX*8), X10

	VMOVSD (R8)(SI*8), X0
	VMOVSD (R8)(DI*8), X1
	VMOVSD (R8)(R14*8), X2
	VMOVSD (R8)(R15*8), X3

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

	VPERMILPS $0xB1, X7, X12
	VXORPS X15, X15, X15
	VSUBPS X12, X15, X11
	VBLENDPS $0x01, X11, X12, X12

	VPERMILPS $0xB1, X7, X14
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14

	VADDPS X4, X6, X0
	VADDPS X5, X12, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X14, X3

	VMOVSD X0, (R8)(SI*8)
	VMOVSD X1, (R8)(DI*8)
	VMOVSD X2, (R8)(R14*8)
	VMOVSD X3, (R8)(R15*8)

	INCQ DX
	JMP  size128_r4_inv_stage3_inner

size128_r4_inv_stage3_next:
	INCQ BX
	JMP  size128_r4_inv_stage3_outer

size128_r4_inv_stage4:
	// ==================================================================
	// Stage 4: 32 radix-2 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ DX, DX

size128_r4_inv_stage4_loop:
	CMPQ DX, $64
	JGE  size128_r4_inv_scale

	MOVQ DX, SI
	MOVQ DX, DI
	ADDQ $64, DI

	VMOVSD (R10)(DX*8), X8

	VMOVSD (R8)(SI*8), X0
	VMOVSD (R8)(DI*8), X1

	// Conjugate complex multiply
	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMSUBADD231PS X11, X1, X13
	VMOVAPS X13, X1

	VADDPS X0, X1, X2
	VSUBPS X1, X0, X3

	VMOVSD X2, (R8)(SI*8)
	VMOVSD X3, (R8)(DI*8)

	INCQ DX
	JMP  size128_r4_inv_stage4_loop

size128_r4_inv_scale:
	// ==================================================================
	// Apply 1/N scaling (1/128 = 0.0078125)
	// ==================================================================
	MOVL ·oneTwentyEighth32(SB), AX     // 1/128 = 0.0078125 in float32
	MOVD AX, X8
	VBROADCASTSS X8, Y8

	XORQ CX, CX

size128_r4_inv_scale_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMULPS Y8, Y0, Y0
	VMOVUPS Y0, (R8)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $1024
	JL   size128_r4_inv_scale_loop

	// Copy to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size128_r4_inv_done

	XORQ CX, CX

size128_r4_inv_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $1024
	JL   size128_r4_inv_copy_loop

size128_r4_inv_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size128_r4_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET

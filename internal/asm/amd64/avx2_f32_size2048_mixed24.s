//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-2048 Mixed-Radix-2/4 FFT Kernels for AMD64
// ===========================================================================
//
// This file contains a mixed-radix-2/4 DIT FFT optimized for size 2048.
// Stages:
//   - Stage 1-5: radix-4 (5 stages)
//   - Stage 6: radix-2 (final combine)
//
// Mixed-radix bit-reversal indices are required for stage 1.
//
// ===========================================================================

#include "textflag.h"

TEXT ·ForwardAVX2Size2048Mixed24Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 2048)

	// Verify n == 2048
	CMPQ R13, $2048
	JNE  m24_2048_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $2048
	JL   m24_2048_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $2048
	JL   m24_2048_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $2048
	JL   m24_2048_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $2048
	JL   m24_2048_return_false

	// Select working buffer
	CMPQ R8, R9              // Check if dst == src (in-place)
	JNE  m24_2048_use_dst    // If not equal, use dst directly
	MOVQ R11, R8             // In-place: use scratch as working buffer

m24_2048_use_dst:
	// ==================================================================
	// Stage 1: 512 radix-4 butterflies with mixed-radix bit-reversal
	// ==================================================================
	XORQ CX, CX              // Initialize base offset CX = 0

m24_2048_stage1_loop:
	CMPQ CX, $2048           // Check if CX >= 2048
	JGE  m24_2048_stage2     // If yes, go to stage 2

	// Load bit-reversed indices for mixed-radix
	MOVQ (R12)(CX*8), DX     // DX = bitrev[CX] (index 0)
	MOVQ 8(R12)(CX*8), SI    // SI = bitrev[CX+1] (index 1)
	MOVQ 16(R12)(CX*8), DI   // DI = bitrev[CX+2] (index 2)
	MOVQ 24(R12)(CX*8), R14  // R14 = bitrev[CX+3] (index 3)

	// Load input values from src using bit-reversed indices
	VMOVSD (R9)(DX*8), X0    // X0 = src[bitrev[CX]]
	VMOVSD (R9)(SI*8), X1    // X1 = src[bitrev[CX+1]]
	VMOVSD (R9)(DI*8), X2    // X2 = src[bitrev[CX+2]]
	VMOVSD (R9)(R14*8), X3   // X3 = src[bitrev[CX+3]]

	// Radix-4 butterfly (no twiddles in stage 1)
	VADDPS X0, X2, X4        // X4 = X0 + X2
	VSUBPS X2, X0, X5        // X5 = X0 - X2
	VADDPS X1, X3, X6        // X6 = X1 + X3
	VSUBPS X3, X1, X7        // X7 = X1 - X3

	// Compute -j * X7 (multiply by -i)
	VPERMILPS $0xB1, X7, X8  // X8 = permute X7 (imag, real, imag, real)
	VXORPS X9, X9, X9        // X9 = zero vector
	VSUBPS X8, X9, X10       // X10 = 0 - X8 = -X8
	VBLENDPS $0x02, X10, X8, X8 // X8 = (X8[0], X10[1], X8[2], X8[3]) = (real, -imag, imag, real)

	// Compute j * X7 (multiply by i)
	VPERMILPS $0xB1, X7, X11 // X11 = permute X7
	VSUBPS X11, X9, X10      // X10 = 0 - X11 = -X11
	VBLENDPS $0x01, X10, X11, X11 // X11 = (X10[0], X11[1], X11[2], X11[3]) = (-imag, real, -imag, real)

	// Final radix-4 outputs
	VADDPS X4, X6, X0        // X0 = X4 + X6 (output 0)
	VADDPS X5, X8, X1        // X1 = X5 + (-j * X7) (output 1)
	VSUBPS X6, X4, X2        // X2 = X4 - X6 (output 2)
	VADDPS X5, X11, X3       // X3 = X5 + (j * X7) (output 3)

	// Store outputs to work buffer in order
	VMOVSD X0, (R8)(CX*8)    // Store output 0
	VMOVSD X1, 8(R8)(CX*8)   // Store output 1
	VMOVSD X2, 16(R8)(CX*8)  // Store output 2
	VMOVSD X3, 24(R8)(CX*8)  // Store output 3

	ADDQ $4, CX              // Advance CX by 4 (processed 4 complex numbers)
	JMP  m24_2048_stage1_loop // Loop back

m24_2048_stage2:
	// ==================================================================
	// Stage 2: 128 groups, each with 4 butterflies
	// Twiddle step = 128
	// ==================================================================
	XORQ CX, CX

m24_2048_stage2_outer:
	CMPQ CX, $2048
	JGE  m24_2048_stage3

	XORQ DX, DX

m24_2048_stage2_inner:
	CMPQ DX, $4
	JGE  m24_2048_stage2_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 4(BX), SI
	LEAQ 8(BX), DI
	LEAQ 12(BX), R14

	// Twiddles: j*128, 2*j*128, 3*j*128 (stage 2, twiddle step = 128)
	MOVQ DX, R15             // R15 = j
	SHLQ $7, R15             // R15 = j * 128
	VMOVSD (R10)(R15*8), X8  // Load W^{j*128} into X8

	MOVQ R15, R13            // R13 = j*128
	SHLQ $1, R15             // R15 = j*256
	VMOVSD (R10)(R15*8), X9  // Load W^{j*256} into X9

	ADDQ R13, R15            // R15 = j*128 + j*256 = j*384
	VMOVSD (R10)(R15*8), X10 // Load W^{j*384} into X10

	VMOVSD (R8)(BX*8), X0    // Load input 0
	VMOVSD (R8)(SI*8), X1    // Load input 1
	VMOVSD (R8)(DI*8), X2    // Load input 2
	VMOVSD (R8)(R14*8), X3   // Load input 3

	// Complex multiply X1 *= W^{j*128}
	VMOVSLDUP X8, X11        // Duplicate real parts of W^{j*128}
	VMOVSHDUP X8, X12        // Duplicate imag parts of W^{j*128}
	VSHUFPS $0xB1, X1, X1, X13 // Swap real/imag of X1
	VMULPS X12, X13, X13     // imag_W * swapped X1
	VFMADDSUB231PS X11, X1, X13 // Fused: X13 ± X11*X1
	VMOVAPS X13, X1          // Store to X1

	// Complex multiply X2 *= W^{j*256}
	VMOVSLDUP X9, X11        // Duplicate real parts of W^{j*256}
	VMOVSHDUP X9, X12        // Duplicate imag parts of W^{j*256}
	VSHUFPS $0xB1, X2, X2, X13 // Swap real/imag of X2
	VMULPS X12, X13, X13     // imag_W * swapped X2
	VFMADDSUB231PS X11, X2, X13 // Fused
	VMOVAPS X13, X2          // Store to X2

	// Complex multiply X3 *= W^{j*384}
	VMOVSLDUP X10, X11       // Duplicate real parts of W^{j*384}
	VMOVSHDUP X10, X12       // Duplicate imag parts of W^{j*384}
	VSHUFPS $0xB1, X3, X3, X13 // Swap real/imag of X3
	VMULPS X12, X13, X13     // imag_W * swapped X3
	VFMADDSUB231PS X11, X3, X13 // Fused
	VMOVAPS X13, X3          // Store to X3

	// Radix-4 butterfly intermediates
	VADDPS X0, X2, X4        // X4 = X0 + X2
	VSUBPS X2, X0, X5        // X5 = X0 - X2
	VADDPS X1, X3, X6        // X6 = X1 + X3
	VSUBPS X3, X1, X7        // X7 = X1 - X3

	// Compute j * X7
	VPERMILPS $0xB1, X7, X14 // Permute X7 (imag, real, imag, real)
	VXORPS X15, X15, X15     // X15 = 0
	VSUBPS X14, X15, X11     // X11 = -X14
	VBLENDPS $0x02, X11, X14, X14 // X14 = (real, -imag, imag, real)

	// Compute -j * X7
	VPERMILPS $0xB1, X7, X12 // Permute X7
	VSUBPS X12, X15, X11     // X11 = -X12
	VBLENDPS $0x01, X11, X12, X12 // X12 = (-imag, real, -imag, real)

	// Final outputs
	VADDPS X4, X6, X0        // X0 = X4 + X6
	VADDPS X5, X14, X1       // X1 = X5 + j*X7
	VSUBPS X6, X4, X2        // X2 = X4 - X6
	VADDPS X5, X12, X3       // X3 = X5 - j*X7

	VMOVSD X0, (R8)(BX*8)    // Store output 0
	VMOVSD X1, (R8)(SI*8)    // Store output 1
	VMOVSD X2, (R8)(DI*8)    // Store output 2
	VMOVSD X3, (R8)(R14*8)   // Store output 3

	INCQ DX                  // DX++
	JMP  m24_2048_stage2_inner // Loop inner

m24_2048_stage2_next:
	ADDQ $16, CX             // Advance CX by 16
	JMP  m24_2048_stage2_outer // Loop outer

m24_2048_stage3:
	// ==================================================================
	// Stage 3: 32 groups, each with 16 butterflies
	// Twiddle step = 32
	// ==================================================================
	XORQ CX, CX

m24_2048_stage3_outer:
	CMPQ CX, $2048
	JGE  m24_2048_stage4

	XORQ DX, DX

m24_2048_stage3_inner:
	CMPQ DX, $16
	JGE  m24_2048_stage3_next

	// Indices: BX = CX + DX, SI = BX + 16, DI = BX + 32, R14 = BX + 48
	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 16(BX), SI
	LEAQ 32(BX), DI
	LEAQ 48(BX), R14

	// Twiddles: j*32, 2*j*32, 3*j*32 (stage 3, twiddle step = 32)
	MOVQ DX, R15             // R15 = j
	SHLQ $5, R15             // R15 = j * 32
	VMOVSD (R10)(R15*8), X8  // Load W^{j*32} into X8

	MOVQ R15, R13            // R13 = j*32
	SHLQ $1, R15             // R15 = j*64
	VMOVSD (R10)(R15*8), X9  // Load W^{j*64} into X9

	ADDQ R13, R15            // R15 = j*32 + j*64 = j*96
	VMOVSD (R10)(R15*8), X10 // Load W^{j*96} into X10

	VMOVSD (R8)(BX*8), X0    // Load input 0
	VMOVSD (R8)(SI*8), X1    // Load input 1
	VMOVSD (R8)(DI*8), X2    // Load input 2
	VMOVSD (R8)(R14*8), X3   // Load input 3

	// Complex multiply X1 *= W^{j*32}
	VMOVSLDUP X8, X11        // Duplicate real parts of W^{j*32}
	VMOVSHDUP X8, X12        // Duplicate imag parts of W^{j*32}
	VSHUFPS $0xB1, X1, X1, X13 // Swap real/imag of X1
	VMULPS X12, X13, X13     // imag_W * swapped X1
	VFMADDSUB231PS X11, X1, X13 // Fused: X13 ± X11*X1
	VMOVAPS X13, X1          // Store to X1

	VMOVSLDUP X9, X11        // Duplicate real parts of W^{j*64}
	VMOVSHDUP X9, X12        // Duplicate imag parts of W^{j*64}
	VSHUFPS $0xB1, X2, X2, X13 // Swap real/imag of X2
	VMULPS X12, X13, X13     // imag_W * swapped X2
	VFMADDSUB231PS X11, X2, X13 // Fused
	VMOVAPS X13, X2          // Store to X2

	VMOVSLDUP X10, X11       // Duplicate real parts of W^{j*96}
	VMOVSHDUP X10, X12       // Duplicate imag parts of W^{j*96}
	VSHUFPS $0xB1, X3, X3, X13 // Swap real/imag of X3
	VMULPS X12, X13, X13     // imag_W * swapped X3
	VFMADDSUB231PS X11, X3, X13 // Fused
	VMOVAPS X13, X3          // Store to X3

	// Radix-4 butterfly intermediates
	VADDPS X0, X2, X4        // X4 = X0 + X2
	VSUBPS X2, X0, X5        // X5 = X0 - X2
	VADDPS X1, X3, X6        // X6 = X1 + X3
	VSUBPS X3, X1, X7        // X7 = X1 - X3

	// Compute j * X7
	VPERMILPS $0xB1, X7, X14 // Permute X7 (imag, real, imag, real)
	VXORPS X15, X15, X15     // X15 = 0
	VSUBPS X14, X15, X11     // X11 = -X14
	VBLENDPS $0x02, X11, X14, X14 // X14 = (real, -imag, imag, real)

	// Compute -j * X7
	VPERMILPS $0xB1, X7, X12 // Permute X7
	VSUBPS X12, X15, X11     // X11 = -X12
	VBLENDPS $0x01, X11, X12, X12 // X12 = (-imag, real, -imag, real)

	// Final outputs
	VADDPS X4, X6, X0        // X0 = X4 + X6
	VADDPS X5, X14, X1       // X1 = X5 + j*X7
	VSUBPS X6, X4, X2        // X2 = X4 - X6
	VADDPS X5, X12, X3       // X3 = X5 - j*X7

	VMOVSD X0, (R8)(BX*8)    // Store output 0
	VMOVSD X1, (R8)(SI*8)    // Store output 1
	VMOVSD X2, (R8)(DI*8)    // Store output 2
	VMOVSD X3, (R8)(R14*8)   // Store output 3

	INCQ DX                  // DX++
	JMP  m24_2048_stage3_inner // Loop inner

m24_2048_stage3_next:
	ADDQ $64, CX             // Advance CX by 64
	JMP  m24_2048_stage3_outer // Loop outer

m24_2048_stage4:
	// ==================================================================
	// Stage 4: 8 groups, each with 64 butterflies
	// Twiddle step = 8
	// ==================================================================
	XORQ CX, CX

m24_2048_stage4_outer:
	CMPQ CX, $2048
	JGE  m24_2048_stage5

	XORQ DX, DX

m24_2048_stage4_inner:
	CMPQ DX, $64
	JGE  m24_2048_stage4_next

	// Indices: BX = CX + DX, SI = BX + 64, DI = BX + 128, R14 = BX + 192
	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 64(BX), SI
	LEAQ 128(BX), DI
	LEAQ 192(BX), R14

	// Twiddles: j*8, 2*j*8, 3*j*8 (stage 4, twiddle step = 8)
	MOVQ DX, R15             // R15 = j
	SHLQ $3, R15             // R15 = j * 8
	VMOVSD (R10)(R15*8), X8  // Load W^{j*8} into X8

	MOVQ R15, R13            // R13 = j*8
	SHLQ $1, R15             // R15 = j*16
	VMOVSD (R10)(R15*8), X9  // Load W^{j*16} into X9

	ADDQ R13, R15            // R15 = j*8 + j*16 = j*24
	VMOVSD (R10)(R15*8), X10 // Load W^{j*24} into X10

	VMOVSD (R8)(BX*8), X0    // Load input 0
	VMOVSD (R8)(SI*8), X1    // Load input 1
	VMOVSD (R8)(DI*8), X2    // Load input 2
	VMOVSD (R8)(R14*8), X3   // Load input 3

	// Complex multiply X1 *= W^{j*8}
	VMOVSLDUP X8, X11        // Duplicate real parts of W^{j*8}
	VMOVSHDUP X8, X12        // Duplicate imag parts of W^{j*8}
	VSHUFPS $0xB1, X1, X1, X13 // Swap real/imag of X1
	VMULPS X12, X13, X13     // imag_W * swapped X1
	VFMADDSUB231PS X11, X1, X13 // Fused: X13 ± X11*X1
	VMOVAPS X13, X1          // Store to X1

	VMOVSLDUP X9, X11        // Duplicate real parts of W^{j*16}
	VMOVSHDUP X9, X12        // Duplicate imag parts of W^{j*16}
	VSHUFPS $0xB1, X2, X2, X13 // Swap real/imag of X2
	VMULPS X12, X13, X13     // imag_W * swapped X2
	VFMADDSUB231PS X11, X2, X13 // Fused
	VMOVAPS X13, X2          // Store to X2

	VMOVSLDUP X10, X11       // Duplicate real parts of W^{j*24}
	VMOVSHDUP X10, X12       // Duplicate imag parts of W^{j*24}
	VSHUFPS $0xB1, X3, X3, X13 // Swap real/imag of X3
	VMULPS X12, X13, X13     // imag_W * swapped X3
	VFMADDSUB231PS X11, X3, X13 // Fused
	VMOVAPS X13, X3          // Store to X3

	// Radix-4 butterfly intermediates
	VADDPS X0, X2, X4        // X4 = X0 + X2
	VSUBPS X2, X0, X5        // X5 = X0 - X2
	VADDPS X1, X3, X6        // X6 = X1 + X3
	VSUBPS X3, X1, X7        // X7 = X1 - X3

	// Compute j * X7
	VPERMILPS $0xB1, X7, X14 // Permute X7 (imag, real, imag, real)
	VXORPS X15, X15, X15     // X15 = 0
	VSUBPS X14, X15, X11     // X11 = -X14
	VBLENDPS $0x02, X11, X14, X14 // X14 = (real, -imag, imag, real)

	// Compute -j * X7
	VPERMILPS $0xB1, X7, X12 // Permute X7
	VSUBPS X12, X15, X11     // X11 = -X12
	VBLENDPS $0x01, X11, X12, X12 // X12 = (-imag, real, -imag, real)

	// Final outputs
	VADDPS X4, X6, X0        // X0 = X4 + X6
	VADDPS X5, X14, X1       // X1 = X5 + j*X7
	VSUBPS X6, X4, X2        // X2 = X4 - X6
	VADDPS X5, X12, X3       // X3 = X5 - j*X7

	VMOVSD X0, (R8)(BX*8)    // Store output 0
	VMOVSD X1, (R8)(SI*8)    // Store output 1
	VMOVSD X2, (R8)(DI*8)    // Store output 2
	VMOVSD X3, (R8)(R14*8)   // Store output 3

	INCQ DX                  // DX++
	JMP  m24_2048_stage4_inner // Loop inner

m24_2048_stage4_next:
	ADDQ $256, CX            // Advance CX by 256
	JMP  m24_2048_stage4_outer // Loop outer

m24_2048_stage5:
	// ==================================================================
	// Stage 5: 2 groups, each with 256 butterflies
	// Twiddle step = 2
	// ==================================================================
	XORQ CX, CX

m24_2048_stage5_outer:
	CMPQ CX, $2048
	JGE  m24_2048_stage6

	XORQ DX, DX

m24_2048_stage5_inner:
	CMPQ DX, $256
	JGE  m24_2048_stage5_next

	// Indices: BX = CX + DX, SI = BX + 256, DI = BX + 512, R14 = BX + 768
	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 256(BX), SI
	LEAQ 512(BX), DI
	LEAQ 768(BX), R14

	// Twiddles: j*2, 2*j*2, 3*j*2 (stage 5, twiddle step = 2)
	MOVQ DX, R15             // R15 = j
	SHLQ $1, R15             // R15 = j * 2
	VMOVSD (R10)(R15*8), X8  // Load W^{j*2} into X8

	MOVQ R15, R13            // R13 = j*2
	SHLQ $1, R15             // R15 = j*4
	VMOVSD (R10)(R15*8), X9  // Load W^{j*4} into X9

	ADDQ R13, R15            // R15 = j*2 + j*4 = j*6
	VMOVSD (R10)(R15*8), X10 // Load W^{j*6} into X10

	VMOVSD (R8)(BX*8), X0    // Load input 0
	VMOVSD (R8)(SI*8), X1    // Load input 1
	VMOVSD (R8)(DI*8), X2    // Load input 2
	VMOVSD (R8)(R14*8), X3   // Load input 3

	// Complex multiply X1 *= W^{j*2}
	VMOVSLDUP X8, X11        // Duplicate real parts of W^{j*2}
	VMOVSHDUP X8, X12        // Duplicate imag parts of W^{j*2}
	VSHUFPS $0xB1, X1, X1, X13 // Swap real/imag of X1
	VMULPS X12, X13, X13     // imag_W * swapped X1
	VFMADDSUB231PS X11, X1, X13 // Fused: X13 ± X11*X1
	VMOVAPS X13, X1          // Store to X1

	VMOVSLDUP X9, X11        // Duplicate real parts of W^{j*4}
	VMOVSHDUP X9, X12        // Duplicate imag parts of W^{j*4}
	VSHUFPS $0xB1, X2, X2, X13 // Swap real/imag of X2
	VMULPS X12, X13, X13     // imag_W * swapped X2
	VFMADDSUB231PS X11, X2, X13 // Fused
	VMOVAPS X13, X2          // Store to X2

	VMOVSLDUP X10, X11       // Duplicate real parts of W^{j*6}
	VMOVSHDUP X10, X12       // Duplicate imag parts of W^{j*6}
	VSHUFPS $0xB1, X3, X3, X13 // Swap real/imag of X3
	VMULPS X12, X13, X13     // imag_W * swapped X3
	VFMADDSUB231PS X11, X3, X13 // Fused
	VMOVAPS X13, X3          // Store to X3

	// Radix-4 butterfly intermediates
	VADDPS X0, X2, X4        // X4 = X0 + X2
	VSUBPS X2, X0, X5        // X5 = X0 - X2
	VADDPS X1, X3, X6        // X6 = X1 + X3
	VSUBPS X3, X1, X7        // X7 = X1 - X3

	// Compute j * X7
	VPERMILPS $0xB1, X7, X14 // Permute X7 (imag, real, imag, real)
	VXORPS X15, X15, X15     // X15 = 0
	VSUBPS X14, X15, X11     // X11 = -X14
	VBLENDPS $0x02, X11, X14, X14 // X14 = (real, -imag, imag, real)

	// Compute -j * X7
	VPERMILPS $0xB1, X7, X12 // Permute X7
	VSUBPS X12, X15, X11     // X11 = -X12
	VBLENDPS $0x01, X11, X12, X12 // X12 = (-imag, real, -imag, real)

	// Final outputs
	VADDPS X4, X6, X0        // X0 = X4 + X6
	VADDPS X5, X14, X1       // X1 = X5 + j*X7
	VSUBPS X6, X4, X2        // X2 = X4 - X6
	VADDPS X5, X12, X3       // X3 = X5 - j*X7

	VMOVSD X0, (R8)(BX*8)    // Store output 0
	VMOVSD X1, (R8)(SI*8)    // Store output 1
	VMOVSD X2, (R8)(DI*8)    // Store output 2
	VMOVSD X3, (R8)(R14*8)   // Store output 3

	INCQ DX                  // DX++
	JMP  m24_2048_stage5_inner // Loop inner

m24_2048_stage5_next:
	ADDQ $1024, CX           // Advance CX by 1024
	JMP  m24_2048_stage5_outer // Loop outer

m24_2048_stage6:
	// ==================================================================
	// Stage 6: radix-2 final stage
	// 1024 pairs of radix-2 butterflies with twiddles
	// ==================================================================
	XORQ CX, CX

m24_2048_stage6_loop:
	CMPQ CX, $1024
	JGE  m24_2048_forward_done

	// Indices: BX = CX, SI = BX + 1024
	MOVQ CX, BX
	LEAQ 1024(BX), SI

	VMOVSD (R10)(CX*8), X8   // Load twiddle W^CX
	VMOVSD (R8)(BX*8), X0    // Load input a
	VMOVSD (R8)(SI*8), X1    // Load input b

	// Complex multiply b *= W^CX
	VMOVSLDUP X8, X11        // Duplicate real parts of W^CX
	VMOVSHDUP X8, X12        // Duplicate imag parts of W^CX
	VSHUFPS $0xB1, X1, X1, X13 // Swap real/imag of b
	VMULPS X12, X13, X13     // imag_W * swapped b
	VFMADDSUB231PS X11, X1, X13 // Fused: X13 ± X11*b
	VMOVAPS X13, X1          // Store to b

	// Radix-2 butterfly: a' = a + b, b' = a - b
	VADDPS X0, X1, X2        // X2 = a + b
	VSUBPS X1, X0, X3        // X3 = a - b (note: VSUBPS b, a → dst = a - b)

	VMOVSD X2, (R8)(BX*8)    // Store a'
	VMOVSD X3, (R8)(SI*8)    // Store b'

	INCQ CX                  // CX++
	JMP  m24_2048_stage6_loop // Loop

m24_2048_forward_done:
	// Copy results to dst if needed (out-of-place transform)
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   m24_2048_forward_ret // In-place: no copy needed

	XORQ CX, CX

m24_2048_forward_copy_loop:
	// Copy 64 bytes (16 complex64 values) per iteration
	VMOVUPS (R8)(CX*1), Y0    // Load 32 bytes from scratch
	VMOVUPS 32(R8)(CX*1), Y1  // Load next 32 bytes from scratch
	VMOVUPS Y0, (R9)(CX*1)    // Store to dst
	VMOVUPS Y1, 32(R9)(CX*1)  // Store to dst
	ADDQ $64, CX              // Advance by 64 bytes
	CMPQ CX, $16384           // 2048 * 8 = 16384 bytes total
	JL   m24_2048_forward_copy_loop

m24_2048_forward_ret:
	VZEROUPPER                // Clear upper YMM registers for compatibility
	MOVB $1, ret+120(FP)      // Return true
	RET

m24_2048_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)      // Return false
	RET

// ===========================================================================
// Inverse transform, size 2048, complex64, mixed-radix-2/4
// ===========================================================================
TEXT ·InverseAVX2Size2048Mixed24Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 2048)

	// Verify n == 2048
	CMPQ R13, $2048
	JNE  m24_2048_inv_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $2048
	JL   m24_2048_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $2048
	JL   m24_2048_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $2048
	JL   m24_2048_inv_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $2048
	JL   m24_2048_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  m24_2048_inv_use_dst
	MOVQ R11, R8             // In-place: use scratch

m24_2048_inv_use_dst:
	// ==================================================================
	// Stage 1: 512 radix-4 butterflies with mixed-radix bit-reversal
	// ==================================================================
	XORQ CX, CX

m24_2048_inv_stage1_loop:
	CMPQ CX, $2048
	JGE  m24_2048_inv_stage2

	// Load bit-reversal indices for mixed-radix
	MOVQ (R12)(CX*8), DX     // Index 0
	MOVQ 8(R12)(CX*8), SI    // Index 1
	MOVQ 16(R12)(CX*8), DI   // Index 2
	MOVQ 24(R12)(CX*8), R14  // Index 3

	VMOVSD (R9)(DX*8), X0    // Load input 0
	VMOVSD (R9)(SI*8), X1    // Load input 1
	VMOVSD (R9)(DI*8), X2    // Load input 2
	VMOVSD (R9)(R14*8), X3   // Load input 3

	// Radix-4 butterfly intermediates (no twiddles in stage 1)
	VADDPS X0, X2, X4        // X4 = X0 + X2
	VSUBPS X2, X0, X5        // X5 = X0 - X2
	VADDPS X1, X3, X6        // X6 = X1 + X3
	VSUBPS X3, X1, X7        // X7 = X1 - X3

	// Compute i * X7 (for inverse: note the different blend pattern)
	VPERMILPS $0xB1, X7, X8  // Permute X7 (imag, real, imag, real)
	VXORPS X9, X9, X9        // X9 = 0
	VSUBPS X8, X9, X10       // X10 = -X8
	VBLENDPS $0x01, X10, X8, X8 // X8 = (-imag, real, -imag, real) = i * X7

	// Compute -i * X7
	VPERMILPS $0xB1, X7, X11 // Permute X7
	VSUBPS X11, X9, X10      // X10 = -X11
	VBLENDPS $0x02, X10, X11, X11 // X11 = (real, -imag, imag, real) = -i * X7

	// Final outputs
	VADDPS X4, X6, X0        // X0 = X4 + X6
	VADDPS X5, X8, X1        // X1 = X5 + i*X7
	VSUBPS X6, X4, X2        // X2 = X4 - X6
	VADDPS X5, X11, X3       // X3 = X5 - i*X7

	VMOVSD X0, (R8)(CX*8)    // Store output 0
	VMOVSD X1, 8(R8)(CX*8)   // Store output 1
	VMOVSD X2, 16(R8)(CX*8)  // Store output 2
	VMOVSD X3, 24(R8)(CX*8)  // Store output 3

	ADDQ $4, CX              // CX += 4
	JMP  m24_2048_inv_stage1_loop
	VSUBPS X6, X4, X2
	VADDPS X5, X11, X3

	VMOVSD X0, (R8)(CX*8)
	VMOVSD X1, 8(R8)(CX*8)
	VMOVSD X2, 16(R8)(CX*8)
	VMOVSD X3, 24(R8)(CX*8)

	ADDQ $4, CX
	JMP  m24_2048_inv_stage1_loop

m24_2048_inv_stage2:
	// ==================================================================
	// Stage 2: 128 groups, each with 4 butterflies (conjugated twiddles)
	// ==================================================================
	XORQ CX, CX

m24_2048_inv_stage2_outer:
	CMPQ CX, $2048
	JGE  m24_2048_inv_stage3

	XORQ DX, DX

m24_2048_inv_stage2_inner:
	CMPQ DX, $4
	JGE  m24_2048_inv_stage2_next

	// Indices: BX = CX + DX, SI = BX + 4, DI = BX + 8, R14 = BX + 12
	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 4(BX), SI
	LEAQ 8(BX), DI
	LEAQ 12(BX), R14

	// Twiddles: j*128, 2*j*128, 3*j*128 (stage 2, twiddle step = 128)
	MOVQ DX, R15             // R15 = j
	SHLQ $7, R15             // R15 = j * 128
	VMOVSD (R10)(R15*8), X8  // Load W^{j*128} into X8

	MOVQ R15, R13            // R13 = j*128
	SHLQ $1, R15             // R15 = j*256
	VMOVSD (R10)(R15*8), X9  // Load W^{j*256} into X9

	ADDQ R13, R15            // R15 = j*128 + j*256 = j*384
	VMOVSD (R10)(R15*8), X10 // Load W^{j*384} into X10

	VMOVSD (R8)(BX*8), X0    // Load input 0
	VMOVSD (R8)(SI*8), X1    // Load input 1
	VMOVSD (R8)(DI*8), X2    // Load input 2
	VMOVSD (R8)(R14*8), X3   // Load input 3

	// Conjugated complex multiply X1 *= conj(W^{j*128})
	VMOVSLDUP X8, X11        // Duplicate real parts of W^{j*128}
	VMOVSHDUP X8, X12        // Duplicate imag parts of W^{j*128}
	VSHUFPS $0xB1, X1, X1, X13 // Swap real/imag of X1
	VMULPS X12, X13, X13     // imag_W * swapped X1
	VFMSUBADD231PS X11, X1, X13 // Fused conjugate: X13 ± X11*X1 (note: FMSUBADD vs FMADD)
	VMOVAPS X13, X1          // Store to X1

	VMOVSLDUP X9, X11        // Duplicate real parts of W^{j*256}
	VMOVSHDUP X9, X12        // Duplicate imag parts of W^{j*256}
	VSHUFPS $0xB1, X2, X2, X13 // Swap real/imag of X2
	VMULPS X12, X13, X13     // imag_W * swapped X2
	VFMSUBADD231PS X11, X2, X13 // Fused conjugate
	VMOVAPS X13, X2          // Store to X2

	VMOVSLDUP X10, X11       // Duplicate real parts of W^{j*384}
	VMOVSHDUP X10, X12       // Duplicate imag parts of W^{j*384}
	VSHUFPS $0xB1, X3, X3, X13 // Swap real/imag of X3
	VMULPS X12, X13, X13     // imag_W * swapped X3
	VFMSUBADD231PS X11, X3, X13 // Fused conjugate
	VMOVAPS X13, X3          // Store to X3

	// Radix-4 butterfly intermediates
	VADDPS X0, X2, X4        // X4 = X0 + X2
	VSUBPS X2, X0, X5        // X5 = X0 - X2
	VADDPS X1, X3, X6        // X6 = X1 + X3
	VSUBPS X3, X1, X7        // X7 = X1 - X3

	// Compute i * X7 (inverse uses same pattern as forward for this part)
	VPERMILPS $0xB1, X7, X14 // Permute X7 (imag, real, imag, real)
	VXORPS X15, X15, X15     // X15 = 0
	VSUBPS X14, X15, X11     // X11 = -X14
	VBLENDPS $0x01, X11, X14, X14 // X14 = (-imag, real, -imag, real) = i * X7

	// Compute -i * X7
	VPERMILPS $0xB1, X7, X12 // Permute X7
	VSUBPS X12, X15, X11     // X11 = -X12
	VBLENDPS $0x02, X11, X12, X12 // X12 = (real, -imag, imag, real) = -i * X7

	// Final outputs
	VADDPS X4, X6, X0        // X0 = X4 + X6
	VADDPS X5, X14, X1       // X1 = X5 + i*X7
	VSUBPS X6, X4, X2        // X2 = X4 - X6
	VADDPS X5, X12, X3       // X3 = X5 - i*X7

	VMOVSD X0, (R8)(BX*8)    // Store output 0
	VMOVSD X1, (R8)(SI*8)    // Store output 1
	VMOVSD X2, (R8)(DI*8)    // Store output 2
	VMOVSD X3, (R8)(R14*8)   // Store output 3

	INCQ DX                  // DX++
	JMP  m24_2048_inv_stage2_inner // Loop inner

m24_2048_inv_stage2_next:
	ADDQ $16, CX             // Advance CX by 16
	JMP  m24_2048_inv_stage2_outer // Loop outer

m24_2048_inv_stage3:
	// ==================================================================
	// Stage 3: 32 groups, each with 16 butterflies
	// ==================================================================
	XORQ CX, CX

m24_2048_inv_stage3_outer:
	CMPQ CX, $2048
	JGE  m24_2048_inv_stage4

	XORQ DX, DX

m24_2048_inv_stage3_inner:
	CMPQ DX, $16
	JGE  m24_2048_inv_stage3_next

	// Indices: BX = CX + DX, SI = BX + 16, DI = BX + 32, R14 = BX + 48
	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 16(BX), SI
	LEAQ 32(BX), DI
	LEAQ 48(BX), R14

	// Twiddles: j*32, 2*j*32, 3*j*32 (stage 3, twiddle step = 32)
	MOVQ DX, R15             // R15 = j
	SHLQ $5, R15             // R15 = j * 32
	VMOVSD (R10)(R15*8), X8  // Load W^{j*32} into X8

	MOVQ R15, R13            // R13 = j*32
	SHLQ $1, R15             // R15 = j*64
	VMOVSD (R10)(R15*8), X9  // Load W^{j*64} into X9

	ADDQ R13, R15            // R15 = j*32 + j*64 = j*96
	VMOVSD (R10)(R15*8), X10 // Load W^{j*96} into X10

	VMOVSD (R8)(BX*8), X0    // Load input 0
	VMOVSD (R8)(SI*8), X1    // Load input 1
	VMOVSD (R8)(DI*8), X2    // Load input 2
	VMOVSD (R8)(R14*8), X3   // Load input 3

	// Conjugated complex multiply X1 *= conj(W^{j*32})
	VMOVSLDUP X8, X11        // Duplicate real parts of W^{j*32}
	VMOVSHDUP X8, X12        // Duplicate imag parts of W^{j*32}
	VSHUFPS $0xB1, X1, X1, X13 // Swap real/imag of X1
	VMULPS X12, X13, X13     // imag_W * swapped X1
	VFMSUBADD231PS X11, X1, X13 // Fused conjugate
	VMOVAPS X13, X1          // Store to X1

	VMOVSLDUP X9, X11        // Duplicate real parts of W^{j*64}
	VMOVSHDUP X9, X12        // Duplicate imag parts of W^{j*64}
	VSHUFPS $0xB1, X2, X2, X13 // Swap real/imag of X2
	VMULPS X12, X13, X13     // imag_W * swapped X2
	VFMSUBADD231PS X11, X2, X13 // Fused conjugate
	VMOVAPS X13, X2          // Store to X2

	VMOVSLDUP X10, X11       // Duplicate real parts of W^{j*96}
	VMOVSHDUP X10, X12       // Duplicate imag parts of W^{j*96}
	VSHUFPS $0xB1, X3, X3, X13 // Swap real/imag of X3
	VMULPS X12, X13, X13     // imag_W * swapped X3
	VFMSUBADD231PS X11, X3, X13 // Fused conjugate
	VMOVAPS X13, X3          // Store to X3

	// Radix-4 butterfly intermediates
	VADDPS X0, X2, X4        // X4 = X0 + X2
	VSUBPS X2, X0, X5        // X5 = X0 - X2
	VADDPS X1, X3, X6        // X6 = X1 + X3
	VSUBPS X3, X1, X7        // X7 = X1 - X3

	// Compute i * X7
	VPERMILPS $0xB1, X7, X14 // Permute X7 (imag, real, imag, real)
	VXORPS X15, X15, X15     // X15 = 0
	VSUBPS X14, X15, X11     // X11 = -X14
	VBLENDPS $0x01, X11, X14, X14 // X14 = (-imag, real, -imag, real) = i * X7

	// Compute -i * X7
	VPERMILPS $0xB1, X7, X12 // Permute X7
	VSUBPS X12, X15, X11     // X11 = -X12
	VBLENDPS $0x02, X11, X12, X12 // X12 = (real, -imag, imag, real) = -i * X7

	// Final outputs
	VADDPS X4, X6, X0        // X0 = X4 + X6
	VADDPS X5, X14, X1       // X1 = X5 + i*X7
	VSUBPS X6, X4, X2        // X2 = X4 - X6
	VADDPS X5, X12, X3       // X3 = X5 - i*X7

	VMOVSD X0, (R8)(BX*8)    // Store output 0
	VMOVSD X1, (R8)(SI*8)    // Store output 1
	VMOVSD X2, (R8)(DI*8)    // Store output 2
	VMOVSD X3, (R8)(R14*8)   // Store output 3

	INCQ DX                  // DX++
	JMP  m24_2048_inv_stage3_inner // Loop inner

m24_2048_inv_stage3_next:
	ADDQ $64, CX             // Advance CX by 64
	JMP  m24_2048_inv_stage3_outer // Loop outer

m24_2048_inv_stage4:
	// ==================================================================
	// Stage 4: 8 groups, each with 64 butterflies
	// ==================================================================
	XORQ CX, CX

m24_2048_inv_stage4_outer:
	CMPQ CX, $2048
	JGE  m24_2048_inv_stage5

	XORQ DX, DX

m24_2048_inv_stage4_inner:
	CMPQ DX, $64
	JGE  m24_2048_inv_stage4_next

	// Indices: BX = CX + DX, SI = BX + 64, DI = BX + 128, R14 = BX + 192
	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 64(BX), SI
	LEAQ 128(BX), DI
	LEAQ 192(BX), R14

	// Twiddles: j*8, 2*j*8, 3*j*8 (stage 4, twiddle step = 8)
	MOVQ DX, R15             // R15 = j
	SHLQ $3, R15             // R15 = j * 8
	VMOVSD (R10)(R15*8), X8  // Load W^{j*8} into X8

	MOVQ R15, R13            // R13 = j*8
	SHLQ $1, R15             // R15 = j*16
	VMOVSD (R10)(R15*8), X9  // Load W^{j*16} into X9

	ADDQ R13, R15            // R15 = j*8 + j*16 = j*24
	VMOVSD (R10)(R15*8), X10 // Load W^{j*24} into X10

	VMOVSD (R8)(BX*8), X0    // Load input 0
	VMOVSD (R8)(SI*8), X1    // Load input 1
	VMOVSD (R8)(DI*8), X2    // Load input 2
	VMOVSD (R8)(R14*8), X3   // Load input 3

	// Conjugated complex multiply X1 *= conj(W^{j*8})
	VMOVSLDUP X8, X11        // Duplicate real parts of W^{j*8}
	VMOVSHDUP X8, X12        // Duplicate imag parts of W^{j*8}
	VSHUFPS $0xB1, X1, X1, X13 // Swap real/imag of X1
	VMULPS X12, X13, X13     // imag_W * swapped X1
	VFMSUBADD231PS X11, X1, X13 // Fused conjugate
	VMOVAPS X13, X1          // Store to X1

	VMOVSLDUP X9, X11        // Duplicate real parts of W^{j*16}
	VMOVSHDUP X9, X12        // Duplicate imag parts of W^{j*16}
	VSHUFPS $0xB1, X2, X2, X13 // Swap real/imag of X2
	VMULPS X12, X13, X13     // imag_W * swapped X2
	VFMSUBADD231PS X11, X2, X13 // Fused conjugate
	VMOVAPS X13, X2          // Store to X2

	VMOVSLDUP X10, X11       // Duplicate real parts of W^{j*24}
	VMOVSHDUP X10, X12       // Duplicate imag parts of W^{j*24}
	VSHUFPS $0xB1, X3, X3, X13 // Swap real/imag of X3
	VMULPS X12, X13, X13     // imag_W * swapped X3
	VFMSUBADD231PS X11, X3, X13 // Fused conjugate
	VMOVAPS X13, X3          // Store to X3

	// Radix-4 butterfly intermediates
	VADDPS X0, X2, X4        // X4 = X0 + X2
	VSUBPS X2, X0, X5        // X5 = X0 - X2
	VADDPS X1, X3, X6        // X6 = X1 + X3
	VSUBPS X3, X1, X7        // X7 = X1 - X3

	// Compute i * X7
	VPERMILPS $0xB1, X7, X14 // Permute X7 (imag, real, imag, real)
	VXORPS X15, X15, X15     // X15 = 0
	VSUBPS X14, X15, X11     // X11 = -X14
	VBLENDPS $0x01, X11, X14, X14 // X14 = (-imag, real, -imag, real) = i * X7

	// Compute -i * X7
	VPERMILPS $0xB1, X7, X12 // Permute X7
	VSUBPS X12, X15, X11     // X11 = -X12
	VBLENDPS $0x02, X11, X12, X12 // X12 = (real, -imag, imag, real) = -i * X7

	// Final outputs
	VADDPS X4, X6, X0        // X0 = X4 + X6
	VADDPS X5, X14, X1       // X1 = X5 + i*X7
	VSUBPS X6, X4, X2        // X2 = X4 - X6
	VADDPS X5, X12, X3       // X3 = X5 - i*X7

	VMOVSD X0, (R8)(BX*8)    // Store output 0
	VMOVSD X1, (R8)(SI*8)    // Store output 1
	VMOVSD X2, (R8)(DI*8)    // Store output 2
	VMOVSD X3, (R8)(R14*8)   // Store output 3

	INCQ DX                  // DX++
	JMP  m24_2048_inv_stage4_inner // Loop inner

m24_2048_inv_stage4_next:
	ADDQ $256, CX            // Advance CX by 256
	JMP  m24_2048_inv_stage4_outer // Loop outer

m24_2048_inv_stage5:
	// ==================================================================
	// Stage 5: 2 groups, each with 256 butterflies
	// ==================================================================
	XORQ CX, CX

m24_2048_inv_stage5_outer:
	CMPQ CX, $2048
	JGE  m24_2048_inv_stage6

	XORQ DX, DX

m24_2048_inv_stage5_inner:
	CMPQ DX, $256
	JGE  m24_2048_inv_stage5_next

	// Indices: BX = CX + DX, SI = BX + 256, DI = BX + 512, R14 = BX + 768
	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 256(BX), SI
	LEAQ 512(BX), DI
	LEAQ 768(BX), R14

	// Twiddles: j*2, 2*j*2, 3*j*2 (stage 5, twiddle step = 2)
	MOVQ DX, R15             // R15 = j
	SHLQ $1, R15             // R15 = j * 2
	VMOVSD (R10)(R15*8), X8  // Load W^{j*2} into X8

	MOVQ R15, R13            // R13 = j*2
	SHLQ $1, R15             // R15 = j*4
	VMOVSD (R10)(R15*8), X9  // Load W^{j*4} into X9

	ADDQ R13, R15            // R15 = j*2 + j*4 = j*6
	VMOVSD (R10)(R15*8), X10 // Load W^{j*6} into X10

	VMOVSD (R8)(BX*8), X0    // Load input 0
	VMOVSD (R8)(SI*8), X1    // Load input 1
	VMOVSD (R8)(DI*8), X2    // Load input 2
	VMOVSD (R8)(R14*8), X3   // Load input 3

	// Conjugated complex multiply X1 *= conj(W^{j*2})
	VMOVSLDUP X8, X11        // Duplicate real parts of W^{j*2}
	VMOVSHDUP X8, X12        // Duplicate imag parts of W^{j*2}
	VSHUFPS $0xB1, X1, X1, X13 // Swap real/imag of X1
	VMULPS X12, X13, X13     // imag_W * swapped X1
	VFMSUBADD231PS X11, X1, X13 // Fused conjugate
	VMOVAPS X13, X1          // Store to X1

	VMOVSLDUP X9, X11        // Duplicate real parts of W^{j*4}
	VMOVSHDUP X9, X12        // Duplicate imag parts of W^{j*4}
	VSHUFPS $0xB1, X2, X2, X13 // Swap real/imag of X2
	VMULPS X12, X13, X13     // imag_W * swapped X2
	VFMSUBADD231PS X11, X2, X13 // Fused conjugate
	VMOVAPS X13, X2          // Store to X2

	VMOVSLDUP X10, X11       // Duplicate real parts of W^{j*6}
	VMOVSHDUP X10, X12       // Duplicate imag parts of W^{j*6}
	VSHUFPS $0xB1, X3, X3, X13 // Swap real/imag of X3
	VMULPS X12, X13, X13     // imag_W * swapped X3
	VFMSUBADD231PS X11, X3, X13 // Fused conjugate
	VMOVAPS X13, X3          // Store to X3

	// Radix-4 butterfly intermediates
	VADDPS X0, X2, X4        // X4 = X0 + X2
	VSUBPS X2, X0, X5        // X5 = X0 - X2
	VADDPS X1, X3, X6        // X6 = X1 + X3
	VSUBPS X3, X1, X7        // X7 = X1 - X3

	// Compute i * X7
	VPERMILPS $0xB1, X7, X14 // Permute X7 (imag, real, imag, real)
	VXORPS X15, X15, X15     // X15 = 0
	VSUBPS X14, X15, X11     // X11 = -X14
	VBLENDPS $0x01, X11, X14, X14 // X14 = (-imag, real, -imag, real) = i * X7

	// Compute -i * X7
	VPERMILPS $0xB1, X7, X12 // Permute X7
	VSUBPS X12, X15, X11     // X11 = -X12
	VBLENDPS $0x02, X11, X12, X12 // X12 = (real, -imag, imag, real) = -i * X7

	// Final outputs
	VADDPS X4, X6, X0        // X0 = X4 + X6
	VADDPS X5, X14, X1       // X1 = X5 + i*X7
	VSUBPS X6, X4, X2        // X2 = X4 - X6
	VADDPS X5, X12, X3       // X3 = X5 - i*X7

	VMOVSD X0, (R8)(BX*8)    // Store output 0
	VMOVSD X1, (R8)(SI*8)    // Store output 1
	VMOVSD X2, (R8)(DI*8)    // Store output 2
	VMOVSD X3, (R8)(R14*8)   // Store output 3

	INCQ DX                  // DX++
	JMP  m24_2048_inv_stage5_inner // Loop inner

m24_2048_inv_stage5_next:
	ADDQ $1024, CX           // Advance CX by 1024
	JMP  m24_2048_inv_stage5_outer // Loop outer

m24_2048_inv_stage6:
	// ==================================================================
	// Stage 6: radix-2 final stage (conjugated twiddles)
	// ==================================================================
	XORQ CX, CX

m24_2048_inv_stage6_loop:
	CMPQ CX, $1024
	JGE  m24_2048_inv_scale

	// Indices: BX = CX, SI = BX + 1024
	MOVQ CX, BX
	LEAQ 1024(BX), SI

	VMOVSD (R10)(CX*8), X8   // Load twiddle W^CX
	VMOVSD (R8)(BX*8), X0    // Load input a
	VMOVSD (R8)(SI*8), X1    // Load input b

	// Conjugated complex multiply b *= conj(W^CX)
	VMOVSLDUP X8, X11        // Duplicate real parts of W^CX
	VMOVSHDUP X8, X12        // Duplicate imag parts of W^CX
	VSHUFPS $0xB1, X1, X1, X13 // Swap real/imag of b
	VMULPS X12, X13, X13     // imag_W * swapped b
	VFMSUBADD231PS X11, X1, X13 // Fused conjugate: X13 ± X11*b
	VMOVAPS X13, X1          // Store to b

	// Radix-2 butterfly: a' = a + b, b' = a - b
	VADDPS X0, X1, X2        // X2 = a + b
	VSUBPS X1, X0, X3        // X3 = a - b (note: VSUBPS b, a → dst = a - b)

	VMOVSD X2, (R8)(BX*8)    // Store a'
	VMOVSD X3, (R8)(SI*8)    // Store b'

	INCQ CX                  // CX++
	JMP  m24_2048_inv_stage6_loop

m24_2048_inv_scale:
	// ==================================================================
	// Apply 1/N scaling for inverse transform (N=2048, so 1/2048)
	// ==================================================================
	MOVL ·twoThousandFortyEighth32(SB), AX // Load 1/2048 as float32
	MOVD AX, X8                           // Move to XMM register
	VBROADCASTSS X8, Y8                   // Broadcast to YMM register

	XORQ CX, CX

m24_2048_inv_scale_loop:
	// Scale 64 bytes (16 complex64 values) per iteration
	VMOVUPS (R8)(CX*1), Y0    // Load 32 bytes from working buffer
	VMOVUPS 32(R8)(CX*1), Y1  // Load next 32 bytes
	VMULPS Y8, Y0, Y0         // Multiply by 1/2048
	VMULPS Y8, Y1, Y1         // Multiply by 1/2048
	VMOVUPS Y0, (R8)(CX*1)    // Store back
	VMOVUPS Y1, 32(R8)(CX*1)  // Store back
	ADDQ $64, CX              // Advance by 64 bytes
	CMPQ CX, $16384           // 2048 * 8 = 16384 bytes total
	JL   m24_2048_inv_scale_loop
	
	// My loop:
	VMOVUPS Y0, (R8)(CX*1)
	VMOVUPS Y1, 32(R8)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $16384
	JL   m24_2048_inv_scale_loop

	// Copy results to dst if needed (out-of-place inverse transform)
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   m24_2048_inv_done    // In-place: no copy needed

	XORQ CX, CX

m24_2048_inv_copy_loop:
	// Copy 64 bytes (16 complex64 values) per iteration
	VMOVUPS (R8)(CX*1), Y0    // Load 32 bytes from working buffer
	VMOVUPS 32(R8)(CX*1), Y1  // Load next 32 bytes
	VMOVUPS Y0, (R9)(CX*1)    // Store to dst
	VMOVUPS Y1, 32(R9)(CX*1)  // Store to dst
	ADDQ $64, CX              // Advance by 64 bytes
	CMPQ CX, $16384           // 2048 * 8 = 16384 bytes total
	JL   m24_2048_inv_copy_loop

m24_2048_inv_done:
	VZEROUPPER                // Clear upper YMM registers for compatibility
	MOVB $1, ret+120(FP)      // Return true
	RET

m24_2048_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)      // Return false
	RET

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

TEXT ·ForwardAVX2Size2048Mixed24Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 2048)
	LEAQ ·bitrev2048_m24(SB), R12

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
	MOVB $1, ret+96(FP)       // Return true
	RET

m24_2048_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)       // Return false
	RET

// ===========================================================================
// Inverse transform, size 2048, complex64, mixed-radix-2/4
// ===========================================================================
TEXT ·InverseAVX2Size2048Mixed24Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 2048)
	LEAQ ·bitrev2048_m24(SB), R12

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
	MOVB $1, ret+96(FP)       // Return true
	RET

m24_2048_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)       // Return false
	RET


DATA ·bitrev2048_m24+0(SB)/8, $0
DATA ·bitrev2048_m24+8(SB)/8, $512
DATA ·bitrev2048_m24+16(SB)/8, $1024
DATA ·bitrev2048_m24+24(SB)/8, $1536
DATA ·bitrev2048_m24+32(SB)/8, $128
DATA ·bitrev2048_m24+40(SB)/8, $640
DATA ·bitrev2048_m24+48(SB)/8, $1152
DATA ·bitrev2048_m24+56(SB)/8, $1664
DATA ·bitrev2048_m24+64(SB)/8, $256
DATA ·bitrev2048_m24+72(SB)/8, $768
DATA ·bitrev2048_m24+80(SB)/8, $1280
DATA ·bitrev2048_m24+88(SB)/8, $1792
DATA ·bitrev2048_m24+96(SB)/8, $384
DATA ·bitrev2048_m24+104(SB)/8, $896
DATA ·bitrev2048_m24+112(SB)/8, $1408
DATA ·bitrev2048_m24+120(SB)/8, $1920
DATA ·bitrev2048_m24+128(SB)/8, $32
DATA ·bitrev2048_m24+136(SB)/8, $544
DATA ·bitrev2048_m24+144(SB)/8, $1056
DATA ·bitrev2048_m24+152(SB)/8, $1568
DATA ·bitrev2048_m24+160(SB)/8, $160
DATA ·bitrev2048_m24+168(SB)/8, $672
DATA ·bitrev2048_m24+176(SB)/8, $1184
DATA ·bitrev2048_m24+184(SB)/8, $1696
DATA ·bitrev2048_m24+192(SB)/8, $288
DATA ·bitrev2048_m24+200(SB)/8, $800
DATA ·bitrev2048_m24+208(SB)/8, $1312
DATA ·bitrev2048_m24+216(SB)/8, $1824
DATA ·bitrev2048_m24+224(SB)/8, $416
DATA ·bitrev2048_m24+232(SB)/8, $928
DATA ·bitrev2048_m24+240(SB)/8, $1440
DATA ·bitrev2048_m24+248(SB)/8, $1952
DATA ·bitrev2048_m24+256(SB)/8, $64
DATA ·bitrev2048_m24+264(SB)/8, $576
DATA ·bitrev2048_m24+272(SB)/8, $1088
DATA ·bitrev2048_m24+280(SB)/8, $1600
DATA ·bitrev2048_m24+288(SB)/8, $192
DATA ·bitrev2048_m24+296(SB)/8, $704
DATA ·bitrev2048_m24+304(SB)/8, $1216
DATA ·bitrev2048_m24+312(SB)/8, $1728
DATA ·bitrev2048_m24+320(SB)/8, $320
DATA ·bitrev2048_m24+328(SB)/8, $832
DATA ·bitrev2048_m24+336(SB)/8, $1344
DATA ·bitrev2048_m24+344(SB)/8, $1856
DATA ·bitrev2048_m24+352(SB)/8, $448
DATA ·bitrev2048_m24+360(SB)/8, $960
DATA ·bitrev2048_m24+368(SB)/8, $1472
DATA ·bitrev2048_m24+376(SB)/8, $1984
DATA ·bitrev2048_m24+384(SB)/8, $96
DATA ·bitrev2048_m24+392(SB)/8, $608
DATA ·bitrev2048_m24+400(SB)/8, $1120
DATA ·bitrev2048_m24+408(SB)/8, $1632
DATA ·bitrev2048_m24+416(SB)/8, $224
DATA ·bitrev2048_m24+424(SB)/8, $736
DATA ·bitrev2048_m24+432(SB)/8, $1248
DATA ·bitrev2048_m24+440(SB)/8, $1760
DATA ·bitrev2048_m24+448(SB)/8, $352
DATA ·bitrev2048_m24+456(SB)/8, $864
DATA ·bitrev2048_m24+464(SB)/8, $1376
DATA ·bitrev2048_m24+472(SB)/8, $1888
DATA ·bitrev2048_m24+480(SB)/8, $480
DATA ·bitrev2048_m24+488(SB)/8, $992
DATA ·bitrev2048_m24+496(SB)/8, $1504
DATA ·bitrev2048_m24+504(SB)/8, $2016
DATA ·bitrev2048_m24+512(SB)/8, $8
DATA ·bitrev2048_m24+520(SB)/8, $520
DATA ·bitrev2048_m24+528(SB)/8, $1032
DATA ·bitrev2048_m24+536(SB)/8, $1544
DATA ·bitrev2048_m24+544(SB)/8, $136
DATA ·bitrev2048_m24+552(SB)/8, $648
DATA ·bitrev2048_m24+560(SB)/8, $1160
DATA ·bitrev2048_m24+568(SB)/8, $1672
DATA ·bitrev2048_m24+576(SB)/8, $264
DATA ·bitrev2048_m24+584(SB)/8, $776
DATA ·bitrev2048_m24+592(SB)/8, $1288
DATA ·bitrev2048_m24+600(SB)/8, $1800
DATA ·bitrev2048_m24+608(SB)/8, $392
DATA ·bitrev2048_m24+616(SB)/8, $904
DATA ·bitrev2048_m24+624(SB)/8, $1416
DATA ·bitrev2048_m24+632(SB)/8, $1928
DATA ·bitrev2048_m24+640(SB)/8, $40
DATA ·bitrev2048_m24+648(SB)/8, $552
DATA ·bitrev2048_m24+656(SB)/8, $1064
DATA ·bitrev2048_m24+664(SB)/8, $1576
DATA ·bitrev2048_m24+672(SB)/8, $168
DATA ·bitrev2048_m24+680(SB)/8, $680
DATA ·bitrev2048_m24+688(SB)/8, $1192
DATA ·bitrev2048_m24+696(SB)/8, $1704
DATA ·bitrev2048_m24+704(SB)/8, $296
DATA ·bitrev2048_m24+712(SB)/8, $808
DATA ·bitrev2048_m24+720(SB)/8, $1320
DATA ·bitrev2048_m24+728(SB)/8, $1832
DATA ·bitrev2048_m24+736(SB)/8, $424
DATA ·bitrev2048_m24+744(SB)/8, $936
DATA ·bitrev2048_m24+752(SB)/8, $1448
DATA ·bitrev2048_m24+760(SB)/8, $1960
DATA ·bitrev2048_m24+768(SB)/8, $72
DATA ·bitrev2048_m24+776(SB)/8, $584
DATA ·bitrev2048_m24+784(SB)/8, $1096
DATA ·bitrev2048_m24+792(SB)/8, $1608
DATA ·bitrev2048_m24+800(SB)/8, $200
DATA ·bitrev2048_m24+808(SB)/8, $712
DATA ·bitrev2048_m24+816(SB)/8, $1224
DATA ·bitrev2048_m24+824(SB)/8, $1736
DATA ·bitrev2048_m24+832(SB)/8, $328
DATA ·bitrev2048_m24+840(SB)/8, $840
DATA ·bitrev2048_m24+848(SB)/8, $1352
DATA ·bitrev2048_m24+856(SB)/8, $1864
DATA ·bitrev2048_m24+864(SB)/8, $456
DATA ·bitrev2048_m24+872(SB)/8, $968
DATA ·bitrev2048_m24+880(SB)/8, $1480
DATA ·bitrev2048_m24+888(SB)/8, $1992
DATA ·bitrev2048_m24+896(SB)/8, $104
DATA ·bitrev2048_m24+904(SB)/8, $616
DATA ·bitrev2048_m24+912(SB)/8, $1128
DATA ·bitrev2048_m24+920(SB)/8, $1640
DATA ·bitrev2048_m24+928(SB)/8, $232
DATA ·bitrev2048_m24+936(SB)/8, $744
DATA ·bitrev2048_m24+944(SB)/8, $1256
DATA ·bitrev2048_m24+952(SB)/8, $1768
DATA ·bitrev2048_m24+960(SB)/8, $360
DATA ·bitrev2048_m24+968(SB)/8, $872
DATA ·bitrev2048_m24+976(SB)/8, $1384
DATA ·bitrev2048_m24+984(SB)/8, $1896
DATA ·bitrev2048_m24+992(SB)/8, $488
DATA ·bitrev2048_m24+1000(SB)/8, $1000
DATA ·bitrev2048_m24+1008(SB)/8, $1512
DATA ·bitrev2048_m24+1016(SB)/8, $2024
DATA ·bitrev2048_m24+1024(SB)/8, $16
DATA ·bitrev2048_m24+1032(SB)/8, $528
DATA ·bitrev2048_m24+1040(SB)/8, $1040
DATA ·bitrev2048_m24+1048(SB)/8, $1552
DATA ·bitrev2048_m24+1056(SB)/8, $144
DATA ·bitrev2048_m24+1064(SB)/8, $656
DATA ·bitrev2048_m24+1072(SB)/8, $1168
DATA ·bitrev2048_m24+1080(SB)/8, $1680
DATA ·bitrev2048_m24+1088(SB)/8, $272
DATA ·bitrev2048_m24+1096(SB)/8, $784
DATA ·bitrev2048_m24+1104(SB)/8, $1296
DATA ·bitrev2048_m24+1112(SB)/8, $1808
DATA ·bitrev2048_m24+1120(SB)/8, $400
DATA ·bitrev2048_m24+1128(SB)/8, $912
DATA ·bitrev2048_m24+1136(SB)/8, $1424
DATA ·bitrev2048_m24+1144(SB)/8, $1936
DATA ·bitrev2048_m24+1152(SB)/8, $48
DATA ·bitrev2048_m24+1160(SB)/8, $560
DATA ·bitrev2048_m24+1168(SB)/8, $1072
DATA ·bitrev2048_m24+1176(SB)/8, $1584
DATA ·bitrev2048_m24+1184(SB)/8, $176
DATA ·bitrev2048_m24+1192(SB)/8, $688
DATA ·bitrev2048_m24+1200(SB)/8, $1200
DATA ·bitrev2048_m24+1208(SB)/8, $1712
DATA ·bitrev2048_m24+1216(SB)/8, $304
DATA ·bitrev2048_m24+1224(SB)/8, $816
DATA ·bitrev2048_m24+1232(SB)/8, $1328
DATA ·bitrev2048_m24+1240(SB)/8, $1840
DATA ·bitrev2048_m24+1248(SB)/8, $432
DATA ·bitrev2048_m24+1256(SB)/8, $944
DATA ·bitrev2048_m24+1264(SB)/8, $1456
DATA ·bitrev2048_m24+1272(SB)/8, $1968
DATA ·bitrev2048_m24+1280(SB)/8, $80
DATA ·bitrev2048_m24+1288(SB)/8, $592
DATA ·bitrev2048_m24+1296(SB)/8, $1104
DATA ·bitrev2048_m24+1304(SB)/8, $1616
DATA ·bitrev2048_m24+1312(SB)/8, $208
DATA ·bitrev2048_m24+1320(SB)/8, $720
DATA ·bitrev2048_m24+1328(SB)/8, $1232
DATA ·bitrev2048_m24+1336(SB)/8, $1744
DATA ·bitrev2048_m24+1344(SB)/8, $336
DATA ·bitrev2048_m24+1352(SB)/8, $848
DATA ·bitrev2048_m24+1360(SB)/8, $1360
DATA ·bitrev2048_m24+1368(SB)/8, $1872
DATA ·bitrev2048_m24+1376(SB)/8, $464
DATA ·bitrev2048_m24+1384(SB)/8, $976
DATA ·bitrev2048_m24+1392(SB)/8, $1488
DATA ·bitrev2048_m24+1400(SB)/8, $2000
DATA ·bitrev2048_m24+1408(SB)/8, $112
DATA ·bitrev2048_m24+1416(SB)/8, $624
DATA ·bitrev2048_m24+1424(SB)/8, $1136
DATA ·bitrev2048_m24+1432(SB)/8, $1648
DATA ·bitrev2048_m24+1440(SB)/8, $240
DATA ·bitrev2048_m24+1448(SB)/8, $752
DATA ·bitrev2048_m24+1456(SB)/8, $1264
DATA ·bitrev2048_m24+1464(SB)/8, $1776
DATA ·bitrev2048_m24+1472(SB)/8, $368
DATA ·bitrev2048_m24+1480(SB)/8, $880
DATA ·bitrev2048_m24+1488(SB)/8, $1392
DATA ·bitrev2048_m24+1496(SB)/8, $1904
DATA ·bitrev2048_m24+1504(SB)/8, $496
DATA ·bitrev2048_m24+1512(SB)/8, $1008
DATA ·bitrev2048_m24+1520(SB)/8, $1520
DATA ·bitrev2048_m24+1528(SB)/8, $2032
DATA ·bitrev2048_m24+1536(SB)/8, $24
DATA ·bitrev2048_m24+1544(SB)/8, $536
DATA ·bitrev2048_m24+1552(SB)/8, $1048
DATA ·bitrev2048_m24+1560(SB)/8, $1560
DATA ·bitrev2048_m24+1568(SB)/8, $152
DATA ·bitrev2048_m24+1576(SB)/8, $664
DATA ·bitrev2048_m24+1584(SB)/8, $1176
DATA ·bitrev2048_m24+1592(SB)/8, $1688
DATA ·bitrev2048_m24+1600(SB)/8, $280
DATA ·bitrev2048_m24+1608(SB)/8, $792
DATA ·bitrev2048_m24+1616(SB)/8, $1304
DATA ·bitrev2048_m24+1624(SB)/8, $1816
DATA ·bitrev2048_m24+1632(SB)/8, $408
DATA ·bitrev2048_m24+1640(SB)/8, $920
DATA ·bitrev2048_m24+1648(SB)/8, $1432
DATA ·bitrev2048_m24+1656(SB)/8, $1944
DATA ·bitrev2048_m24+1664(SB)/8, $56
DATA ·bitrev2048_m24+1672(SB)/8, $568
DATA ·bitrev2048_m24+1680(SB)/8, $1080
DATA ·bitrev2048_m24+1688(SB)/8, $1592
DATA ·bitrev2048_m24+1696(SB)/8, $184
DATA ·bitrev2048_m24+1704(SB)/8, $696
DATA ·bitrev2048_m24+1712(SB)/8, $1208
DATA ·bitrev2048_m24+1720(SB)/8, $1720
DATA ·bitrev2048_m24+1728(SB)/8, $312
DATA ·bitrev2048_m24+1736(SB)/8, $824
DATA ·bitrev2048_m24+1744(SB)/8, $1336
DATA ·bitrev2048_m24+1752(SB)/8, $1848
DATA ·bitrev2048_m24+1760(SB)/8, $440
DATA ·bitrev2048_m24+1768(SB)/8, $952
DATA ·bitrev2048_m24+1776(SB)/8, $1464
DATA ·bitrev2048_m24+1784(SB)/8, $1976
DATA ·bitrev2048_m24+1792(SB)/8, $88
DATA ·bitrev2048_m24+1800(SB)/8, $600
DATA ·bitrev2048_m24+1808(SB)/8, $1112
DATA ·bitrev2048_m24+1816(SB)/8, $1624
DATA ·bitrev2048_m24+1824(SB)/8, $216
DATA ·bitrev2048_m24+1832(SB)/8, $728
DATA ·bitrev2048_m24+1840(SB)/8, $1240
DATA ·bitrev2048_m24+1848(SB)/8, $1752
DATA ·bitrev2048_m24+1856(SB)/8, $344
DATA ·bitrev2048_m24+1864(SB)/8, $856
DATA ·bitrev2048_m24+1872(SB)/8, $1368
DATA ·bitrev2048_m24+1880(SB)/8, $1880
DATA ·bitrev2048_m24+1888(SB)/8, $472
DATA ·bitrev2048_m24+1896(SB)/8, $984
DATA ·bitrev2048_m24+1904(SB)/8, $1496
DATA ·bitrev2048_m24+1912(SB)/8, $2008
DATA ·bitrev2048_m24+1920(SB)/8, $120
DATA ·bitrev2048_m24+1928(SB)/8, $632
DATA ·bitrev2048_m24+1936(SB)/8, $1144
DATA ·bitrev2048_m24+1944(SB)/8, $1656
DATA ·bitrev2048_m24+1952(SB)/8, $248
DATA ·bitrev2048_m24+1960(SB)/8, $760
DATA ·bitrev2048_m24+1968(SB)/8, $1272
DATA ·bitrev2048_m24+1976(SB)/8, $1784
DATA ·bitrev2048_m24+1984(SB)/8, $376
DATA ·bitrev2048_m24+1992(SB)/8, $888
DATA ·bitrev2048_m24+2000(SB)/8, $1400
DATA ·bitrev2048_m24+2008(SB)/8, $1912
DATA ·bitrev2048_m24+2016(SB)/8, $504
DATA ·bitrev2048_m24+2024(SB)/8, $1016
DATA ·bitrev2048_m24+2032(SB)/8, $1528
DATA ·bitrev2048_m24+2040(SB)/8, $2040
DATA ·bitrev2048_m24+2048(SB)/8, $2
DATA ·bitrev2048_m24+2056(SB)/8, $514
DATA ·bitrev2048_m24+2064(SB)/8, $1026
DATA ·bitrev2048_m24+2072(SB)/8, $1538
DATA ·bitrev2048_m24+2080(SB)/8, $130
DATA ·bitrev2048_m24+2088(SB)/8, $642
DATA ·bitrev2048_m24+2096(SB)/8, $1154
DATA ·bitrev2048_m24+2104(SB)/8, $1666
DATA ·bitrev2048_m24+2112(SB)/8, $258
DATA ·bitrev2048_m24+2120(SB)/8, $770
DATA ·bitrev2048_m24+2128(SB)/8, $1282
DATA ·bitrev2048_m24+2136(SB)/8, $1794
DATA ·bitrev2048_m24+2144(SB)/8, $386
DATA ·bitrev2048_m24+2152(SB)/8, $898
DATA ·bitrev2048_m24+2160(SB)/8, $1410
DATA ·bitrev2048_m24+2168(SB)/8, $1922
DATA ·bitrev2048_m24+2176(SB)/8, $34
DATA ·bitrev2048_m24+2184(SB)/8, $546
DATA ·bitrev2048_m24+2192(SB)/8, $1058
DATA ·bitrev2048_m24+2200(SB)/8, $1570
DATA ·bitrev2048_m24+2208(SB)/8, $162
DATA ·bitrev2048_m24+2216(SB)/8, $674
DATA ·bitrev2048_m24+2224(SB)/8, $1186
DATA ·bitrev2048_m24+2232(SB)/8, $1698
DATA ·bitrev2048_m24+2240(SB)/8, $290
DATA ·bitrev2048_m24+2248(SB)/8, $802
DATA ·bitrev2048_m24+2256(SB)/8, $1314
DATA ·bitrev2048_m24+2264(SB)/8, $1826
DATA ·bitrev2048_m24+2272(SB)/8, $418
DATA ·bitrev2048_m24+2280(SB)/8, $930
DATA ·bitrev2048_m24+2288(SB)/8, $1442
DATA ·bitrev2048_m24+2296(SB)/8, $1954
DATA ·bitrev2048_m24+2304(SB)/8, $66
DATA ·bitrev2048_m24+2312(SB)/8, $578
DATA ·bitrev2048_m24+2320(SB)/8, $1090
DATA ·bitrev2048_m24+2328(SB)/8, $1602
DATA ·bitrev2048_m24+2336(SB)/8, $194
DATA ·bitrev2048_m24+2344(SB)/8, $706
DATA ·bitrev2048_m24+2352(SB)/8, $1218
DATA ·bitrev2048_m24+2360(SB)/8, $1730
DATA ·bitrev2048_m24+2368(SB)/8, $322
DATA ·bitrev2048_m24+2376(SB)/8, $834
DATA ·bitrev2048_m24+2384(SB)/8, $1346
DATA ·bitrev2048_m24+2392(SB)/8, $1858
DATA ·bitrev2048_m24+2400(SB)/8, $450
DATA ·bitrev2048_m24+2408(SB)/8, $962
DATA ·bitrev2048_m24+2416(SB)/8, $1474
DATA ·bitrev2048_m24+2424(SB)/8, $1986
DATA ·bitrev2048_m24+2432(SB)/8, $98
DATA ·bitrev2048_m24+2440(SB)/8, $610
DATA ·bitrev2048_m24+2448(SB)/8, $1122
DATA ·bitrev2048_m24+2456(SB)/8, $1634
DATA ·bitrev2048_m24+2464(SB)/8, $226
DATA ·bitrev2048_m24+2472(SB)/8, $738
DATA ·bitrev2048_m24+2480(SB)/8, $1250
DATA ·bitrev2048_m24+2488(SB)/8, $1762
DATA ·bitrev2048_m24+2496(SB)/8, $354
DATA ·bitrev2048_m24+2504(SB)/8, $866
DATA ·bitrev2048_m24+2512(SB)/8, $1378
DATA ·bitrev2048_m24+2520(SB)/8, $1890
DATA ·bitrev2048_m24+2528(SB)/8, $482
DATA ·bitrev2048_m24+2536(SB)/8, $994
DATA ·bitrev2048_m24+2544(SB)/8, $1506
DATA ·bitrev2048_m24+2552(SB)/8, $2018
DATA ·bitrev2048_m24+2560(SB)/8, $10
DATA ·bitrev2048_m24+2568(SB)/8, $522
DATA ·bitrev2048_m24+2576(SB)/8, $1034
DATA ·bitrev2048_m24+2584(SB)/8, $1546
DATA ·bitrev2048_m24+2592(SB)/8, $138
DATA ·bitrev2048_m24+2600(SB)/8, $650
DATA ·bitrev2048_m24+2608(SB)/8, $1162
DATA ·bitrev2048_m24+2616(SB)/8, $1674
DATA ·bitrev2048_m24+2624(SB)/8, $266
DATA ·bitrev2048_m24+2632(SB)/8, $778
DATA ·bitrev2048_m24+2640(SB)/8, $1290
DATA ·bitrev2048_m24+2648(SB)/8, $1802
DATA ·bitrev2048_m24+2656(SB)/8, $394
DATA ·bitrev2048_m24+2664(SB)/8, $906
DATA ·bitrev2048_m24+2672(SB)/8, $1418
DATA ·bitrev2048_m24+2680(SB)/8, $1930
DATA ·bitrev2048_m24+2688(SB)/8, $42
DATA ·bitrev2048_m24+2696(SB)/8, $554
DATA ·bitrev2048_m24+2704(SB)/8, $1066
DATA ·bitrev2048_m24+2712(SB)/8, $1578
DATA ·bitrev2048_m24+2720(SB)/8, $170
DATA ·bitrev2048_m24+2728(SB)/8, $682
DATA ·bitrev2048_m24+2736(SB)/8, $1194
DATA ·bitrev2048_m24+2744(SB)/8, $1706
DATA ·bitrev2048_m24+2752(SB)/8, $298
DATA ·bitrev2048_m24+2760(SB)/8, $810
DATA ·bitrev2048_m24+2768(SB)/8, $1322
DATA ·bitrev2048_m24+2776(SB)/8, $1834
DATA ·bitrev2048_m24+2784(SB)/8, $426
DATA ·bitrev2048_m24+2792(SB)/8, $938
DATA ·bitrev2048_m24+2800(SB)/8, $1450
DATA ·bitrev2048_m24+2808(SB)/8, $1962
DATA ·bitrev2048_m24+2816(SB)/8, $74
DATA ·bitrev2048_m24+2824(SB)/8, $586
DATA ·bitrev2048_m24+2832(SB)/8, $1098
DATA ·bitrev2048_m24+2840(SB)/8, $1610
DATA ·bitrev2048_m24+2848(SB)/8, $202
DATA ·bitrev2048_m24+2856(SB)/8, $714
DATA ·bitrev2048_m24+2864(SB)/8, $1226
DATA ·bitrev2048_m24+2872(SB)/8, $1738
DATA ·bitrev2048_m24+2880(SB)/8, $330
DATA ·bitrev2048_m24+2888(SB)/8, $842
DATA ·bitrev2048_m24+2896(SB)/8, $1354
DATA ·bitrev2048_m24+2904(SB)/8, $1866
DATA ·bitrev2048_m24+2912(SB)/8, $458
DATA ·bitrev2048_m24+2920(SB)/8, $970
DATA ·bitrev2048_m24+2928(SB)/8, $1482
DATA ·bitrev2048_m24+2936(SB)/8, $1994
DATA ·bitrev2048_m24+2944(SB)/8, $106
DATA ·bitrev2048_m24+2952(SB)/8, $618
DATA ·bitrev2048_m24+2960(SB)/8, $1130
DATA ·bitrev2048_m24+2968(SB)/8, $1642
DATA ·bitrev2048_m24+2976(SB)/8, $234
DATA ·bitrev2048_m24+2984(SB)/8, $746
DATA ·bitrev2048_m24+2992(SB)/8, $1258
DATA ·bitrev2048_m24+3000(SB)/8, $1770
DATA ·bitrev2048_m24+3008(SB)/8, $362
DATA ·bitrev2048_m24+3016(SB)/8, $874
DATA ·bitrev2048_m24+3024(SB)/8, $1386
DATA ·bitrev2048_m24+3032(SB)/8, $1898
DATA ·bitrev2048_m24+3040(SB)/8, $490
DATA ·bitrev2048_m24+3048(SB)/8, $1002
DATA ·bitrev2048_m24+3056(SB)/8, $1514
DATA ·bitrev2048_m24+3064(SB)/8, $2026
DATA ·bitrev2048_m24+3072(SB)/8, $18
DATA ·bitrev2048_m24+3080(SB)/8, $530
DATA ·bitrev2048_m24+3088(SB)/8, $1042
DATA ·bitrev2048_m24+3096(SB)/8, $1554
DATA ·bitrev2048_m24+3104(SB)/8, $146
DATA ·bitrev2048_m24+3112(SB)/8, $658
DATA ·bitrev2048_m24+3120(SB)/8, $1170
DATA ·bitrev2048_m24+3128(SB)/8, $1682
DATA ·bitrev2048_m24+3136(SB)/8, $274
DATA ·bitrev2048_m24+3144(SB)/8, $786
DATA ·bitrev2048_m24+3152(SB)/8, $1298
DATA ·bitrev2048_m24+3160(SB)/8, $1810
DATA ·bitrev2048_m24+3168(SB)/8, $402
DATA ·bitrev2048_m24+3176(SB)/8, $914
DATA ·bitrev2048_m24+3184(SB)/8, $1426
DATA ·bitrev2048_m24+3192(SB)/8, $1938
DATA ·bitrev2048_m24+3200(SB)/8, $50
DATA ·bitrev2048_m24+3208(SB)/8, $562
DATA ·bitrev2048_m24+3216(SB)/8, $1074
DATA ·bitrev2048_m24+3224(SB)/8, $1586
DATA ·bitrev2048_m24+3232(SB)/8, $178
DATA ·bitrev2048_m24+3240(SB)/8, $690
DATA ·bitrev2048_m24+3248(SB)/8, $1202
DATA ·bitrev2048_m24+3256(SB)/8, $1714
DATA ·bitrev2048_m24+3264(SB)/8, $306
DATA ·bitrev2048_m24+3272(SB)/8, $818
DATA ·bitrev2048_m24+3280(SB)/8, $1330
DATA ·bitrev2048_m24+3288(SB)/8, $1842
DATA ·bitrev2048_m24+3296(SB)/8, $434
DATA ·bitrev2048_m24+3304(SB)/8, $946
DATA ·bitrev2048_m24+3312(SB)/8, $1458
DATA ·bitrev2048_m24+3320(SB)/8, $1970
DATA ·bitrev2048_m24+3328(SB)/8, $82
DATA ·bitrev2048_m24+3336(SB)/8, $594
DATA ·bitrev2048_m24+3344(SB)/8, $1106
DATA ·bitrev2048_m24+3352(SB)/8, $1618
DATA ·bitrev2048_m24+3360(SB)/8, $210
DATA ·bitrev2048_m24+3368(SB)/8, $722
DATA ·bitrev2048_m24+3376(SB)/8, $1234
DATA ·bitrev2048_m24+3384(SB)/8, $1746
DATA ·bitrev2048_m24+3392(SB)/8, $338
DATA ·bitrev2048_m24+3400(SB)/8, $850
DATA ·bitrev2048_m24+3408(SB)/8, $1362
DATA ·bitrev2048_m24+3416(SB)/8, $1874
DATA ·bitrev2048_m24+3424(SB)/8, $466
DATA ·bitrev2048_m24+3432(SB)/8, $978
DATA ·bitrev2048_m24+3440(SB)/8, $1490
DATA ·bitrev2048_m24+3448(SB)/8, $2002
DATA ·bitrev2048_m24+3456(SB)/8, $114
DATA ·bitrev2048_m24+3464(SB)/8, $626
DATA ·bitrev2048_m24+3472(SB)/8, $1138
DATA ·bitrev2048_m24+3480(SB)/8, $1650
DATA ·bitrev2048_m24+3488(SB)/8, $242
DATA ·bitrev2048_m24+3496(SB)/8, $754
DATA ·bitrev2048_m24+3504(SB)/8, $1266
DATA ·bitrev2048_m24+3512(SB)/8, $1778
DATA ·bitrev2048_m24+3520(SB)/8, $370
DATA ·bitrev2048_m24+3528(SB)/8, $882
DATA ·bitrev2048_m24+3536(SB)/8, $1394
DATA ·bitrev2048_m24+3544(SB)/8, $1906
DATA ·bitrev2048_m24+3552(SB)/8, $498
DATA ·bitrev2048_m24+3560(SB)/8, $1010
DATA ·bitrev2048_m24+3568(SB)/8, $1522
DATA ·bitrev2048_m24+3576(SB)/8, $2034
DATA ·bitrev2048_m24+3584(SB)/8, $26
DATA ·bitrev2048_m24+3592(SB)/8, $538
DATA ·bitrev2048_m24+3600(SB)/8, $1050
DATA ·bitrev2048_m24+3608(SB)/8, $1562
DATA ·bitrev2048_m24+3616(SB)/8, $154
DATA ·bitrev2048_m24+3624(SB)/8, $666
DATA ·bitrev2048_m24+3632(SB)/8, $1178
DATA ·bitrev2048_m24+3640(SB)/8, $1690
DATA ·bitrev2048_m24+3648(SB)/8, $282
DATA ·bitrev2048_m24+3656(SB)/8, $794
DATA ·bitrev2048_m24+3664(SB)/8, $1306
DATA ·bitrev2048_m24+3672(SB)/8, $1818
DATA ·bitrev2048_m24+3680(SB)/8, $410
DATA ·bitrev2048_m24+3688(SB)/8, $922
DATA ·bitrev2048_m24+3696(SB)/8, $1434
DATA ·bitrev2048_m24+3704(SB)/8, $1946
DATA ·bitrev2048_m24+3712(SB)/8, $58
DATA ·bitrev2048_m24+3720(SB)/8, $570
DATA ·bitrev2048_m24+3728(SB)/8, $1082
DATA ·bitrev2048_m24+3736(SB)/8, $1594
DATA ·bitrev2048_m24+3744(SB)/8, $186
DATA ·bitrev2048_m24+3752(SB)/8, $698
DATA ·bitrev2048_m24+3760(SB)/8, $1210
DATA ·bitrev2048_m24+3768(SB)/8, $1722
DATA ·bitrev2048_m24+3776(SB)/8, $314
DATA ·bitrev2048_m24+3784(SB)/8, $826
DATA ·bitrev2048_m24+3792(SB)/8, $1338
DATA ·bitrev2048_m24+3800(SB)/8, $1850
DATA ·bitrev2048_m24+3808(SB)/8, $442
DATA ·bitrev2048_m24+3816(SB)/8, $954
DATA ·bitrev2048_m24+3824(SB)/8, $1466
DATA ·bitrev2048_m24+3832(SB)/8, $1978
DATA ·bitrev2048_m24+3840(SB)/8, $90
DATA ·bitrev2048_m24+3848(SB)/8, $602
DATA ·bitrev2048_m24+3856(SB)/8, $1114
DATA ·bitrev2048_m24+3864(SB)/8, $1626
DATA ·bitrev2048_m24+3872(SB)/8, $218
DATA ·bitrev2048_m24+3880(SB)/8, $730
DATA ·bitrev2048_m24+3888(SB)/8, $1242
DATA ·bitrev2048_m24+3896(SB)/8, $1754
DATA ·bitrev2048_m24+3904(SB)/8, $346
DATA ·bitrev2048_m24+3912(SB)/8, $858
DATA ·bitrev2048_m24+3920(SB)/8, $1370
DATA ·bitrev2048_m24+3928(SB)/8, $1882
DATA ·bitrev2048_m24+3936(SB)/8, $474
DATA ·bitrev2048_m24+3944(SB)/8, $986
DATA ·bitrev2048_m24+3952(SB)/8, $1498
DATA ·bitrev2048_m24+3960(SB)/8, $2010
DATA ·bitrev2048_m24+3968(SB)/8, $122
DATA ·bitrev2048_m24+3976(SB)/8, $634
DATA ·bitrev2048_m24+3984(SB)/8, $1146
DATA ·bitrev2048_m24+3992(SB)/8, $1658
DATA ·bitrev2048_m24+4000(SB)/8, $250
DATA ·bitrev2048_m24+4008(SB)/8, $762
DATA ·bitrev2048_m24+4016(SB)/8, $1274
DATA ·bitrev2048_m24+4024(SB)/8, $1786
DATA ·bitrev2048_m24+4032(SB)/8, $378
DATA ·bitrev2048_m24+4040(SB)/8, $890
DATA ·bitrev2048_m24+4048(SB)/8, $1402
DATA ·bitrev2048_m24+4056(SB)/8, $1914
DATA ·bitrev2048_m24+4064(SB)/8, $506
DATA ·bitrev2048_m24+4072(SB)/8, $1018
DATA ·bitrev2048_m24+4080(SB)/8, $1530
DATA ·bitrev2048_m24+4088(SB)/8, $2042
DATA ·bitrev2048_m24+4096(SB)/8, $4
DATA ·bitrev2048_m24+4104(SB)/8, $516
DATA ·bitrev2048_m24+4112(SB)/8, $1028
DATA ·bitrev2048_m24+4120(SB)/8, $1540
DATA ·bitrev2048_m24+4128(SB)/8, $132
DATA ·bitrev2048_m24+4136(SB)/8, $644
DATA ·bitrev2048_m24+4144(SB)/8, $1156
DATA ·bitrev2048_m24+4152(SB)/8, $1668
DATA ·bitrev2048_m24+4160(SB)/8, $260
DATA ·bitrev2048_m24+4168(SB)/8, $772
DATA ·bitrev2048_m24+4176(SB)/8, $1284
DATA ·bitrev2048_m24+4184(SB)/8, $1796
DATA ·bitrev2048_m24+4192(SB)/8, $388
DATA ·bitrev2048_m24+4200(SB)/8, $900
DATA ·bitrev2048_m24+4208(SB)/8, $1412
DATA ·bitrev2048_m24+4216(SB)/8, $1924
DATA ·bitrev2048_m24+4224(SB)/8, $36
DATA ·bitrev2048_m24+4232(SB)/8, $548
DATA ·bitrev2048_m24+4240(SB)/8, $1060
DATA ·bitrev2048_m24+4248(SB)/8, $1572
DATA ·bitrev2048_m24+4256(SB)/8, $164
DATA ·bitrev2048_m24+4264(SB)/8, $676
DATA ·bitrev2048_m24+4272(SB)/8, $1188
DATA ·bitrev2048_m24+4280(SB)/8, $1700
DATA ·bitrev2048_m24+4288(SB)/8, $292
DATA ·bitrev2048_m24+4296(SB)/8, $804
DATA ·bitrev2048_m24+4304(SB)/8, $1316
DATA ·bitrev2048_m24+4312(SB)/8, $1828
DATA ·bitrev2048_m24+4320(SB)/8, $420
DATA ·bitrev2048_m24+4328(SB)/8, $932
DATA ·bitrev2048_m24+4336(SB)/8, $1444
DATA ·bitrev2048_m24+4344(SB)/8, $1956
DATA ·bitrev2048_m24+4352(SB)/8, $68
DATA ·bitrev2048_m24+4360(SB)/8, $580
DATA ·bitrev2048_m24+4368(SB)/8, $1092
DATA ·bitrev2048_m24+4376(SB)/8, $1604
DATA ·bitrev2048_m24+4384(SB)/8, $196
DATA ·bitrev2048_m24+4392(SB)/8, $708
DATA ·bitrev2048_m24+4400(SB)/8, $1220
DATA ·bitrev2048_m24+4408(SB)/8, $1732
DATA ·bitrev2048_m24+4416(SB)/8, $324
DATA ·bitrev2048_m24+4424(SB)/8, $836
DATA ·bitrev2048_m24+4432(SB)/8, $1348
DATA ·bitrev2048_m24+4440(SB)/8, $1860
DATA ·bitrev2048_m24+4448(SB)/8, $452
DATA ·bitrev2048_m24+4456(SB)/8, $964
DATA ·bitrev2048_m24+4464(SB)/8, $1476
DATA ·bitrev2048_m24+4472(SB)/8, $1988
DATA ·bitrev2048_m24+4480(SB)/8, $100
DATA ·bitrev2048_m24+4488(SB)/8, $612
DATA ·bitrev2048_m24+4496(SB)/8, $1124
DATA ·bitrev2048_m24+4504(SB)/8, $1636
DATA ·bitrev2048_m24+4512(SB)/8, $228
DATA ·bitrev2048_m24+4520(SB)/8, $740
DATA ·bitrev2048_m24+4528(SB)/8, $1252
DATA ·bitrev2048_m24+4536(SB)/8, $1764
DATA ·bitrev2048_m24+4544(SB)/8, $356
DATA ·bitrev2048_m24+4552(SB)/8, $868
DATA ·bitrev2048_m24+4560(SB)/8, $1380
DATA ·bitrev2048_m24+4568(SB)/8, $1892
DATA ·bitrev2048_m24+4576(SB)/8, $484
DATA ·bitrev2048_m24+4584(SB)/8, $996
DATA ·bitrev2048_m24+4592(SB)/8, $1508
DATA ·bitrev2048_m24+4600(SB)/8, $2020
DATA ·bitrev2048_m24+4608(SB)/8, $12
DATA ·bitrev2048_m24+4616(SB)/8, $524
DATA ·bitrev2048_m24+4624(SB)/8, $1036
DATA ·bitrev2048_m24+4632(SB)/8, $1548
DATA ·bitrev2048_m24+4640(SB)/8, $140
DATA ·bitrev2048_m24+4648(SB)/8, $652
DATA ·bitrev2048_m24+4656(SB)/8, $1164
DATA ·bitrev2048_m24+4664(SB)/8, $1676
DATA ·bitrev2048_m24+4672(SB)/8, $268
DATA ·bitrev2048_m24+4680(SB)/8, $780
DATA ·bitrev2048_m24+4688(SB)/8, $1292
DATA ·bitrev2048_m24+4696(SB)/8, $1804
DATA ·bitrev2048_m24+4704(SB)/8, $396
DATA ·bitrev2048_m24+4712(SB)/8, $908
DATA ·bitrev2048_m24+4720(SB)/8, $1420
DATA ·bitrev2048_m24+4728(SB)/8, $1932
DATA ·bitrev2048_m24+4736(SB)/8, $44
DATA ·bitrev2048_m24+4744(SB)/8, $556
DATA ·bitrev2048_m24+4752(SB)/8, $1068
DATA ·bitrev2048_m24+4760(SB)/8, $1580
DATA ·bitrev2048_m24+4768(SB)/8, $172
DATA ·bitrev2048_m24+4776(SB)/8, $684
DATA ·bitrev2048_m24+4784(SB)/8, $1196
DATA ·bitrev2048_m24+4792(SB)/8, $1708
DATA ·bitrev2048_m24+4800(SB)/8, $300
DATA ·bitrev2048_m24+4808(SB)/8, $812
DATA ·bitrev2048_m24+4816(SB)/8, $1324
DATA ·bitrev2048_m24+4824(SB)/8, $1836
DATA ·bitrev2048_m24+4832(SB)/8, $428
DATA ·bitrev2048_m24+4840(SB)/8, $940
DATA ·bitrev2048_m24+4848(SB)/8, $1452
DATA ·bitrev2048_m24+4856(SB)/8, $1964
DATA ·bitrev2048_m24+4864(SB)/8, $76
DATA ·bitrev2048_m24+4872(SB)/8, $588
DATA ·bitrev2048_m24+4880(SB)/8, $1100
DATA ·bitrev2048_m24+4888(SB)/8, $1612
DATA ·bitrev2048_m24+4896(SB)/8, $204
DATA ·bitrev2048_m24+4904(SB)/8, $716
DATA ·bitrev2048_m24+4912(SB)/8, $1228
DATA ·bitrev2048_m24+4920(SB)/8, $1740
DATA ·bitrev2048_m24+4928(SB)/8, $332
DATA ·bitrev2048_m24+4936(SB)/8, $844
DATA ·bitrev2048_m24+4944(SB)/8, $1356
DATA ·bitrev2048_m24+4952(SB)/8, $1868
DATA ·bitrev2048_m24+4960(SB)/8, $460
DATA ·bitrev2048_m24+4968(SB)/8, $972
DATA ·bitrev2048_m24+4976(SB)/8, $1484
DATA ·bitrev2048_m24+4984(SB)/8, $1996
DATA ·bitrev2048_m24+4992(SB)/8, $108
DATA ·bitrev2048_m24+5000(SB)/8, $620
DATA ·bitrev2048_m24+5008(SB)/8, $1132
DATA ·bitrev2048_m24+5016(SB)/8, $1644
DATA ·bitrev2048_m24+5024(SB)/8, $236
DATA ·bitrev2048_m24+5032(SB)/8, $748
DATA ·bitrev2048_m24+5040(SB)/8, $1260
DATA ·bitrev2048_m24+5048(SB)/8, $1772
DATA ·bitrev2048_m24+5056(SB)/8, $364
DATA ·bitrev2048_m24+5064(SB)/8, $876
DATA ·bitrev2048_m24+5072(SB)/8, $1388
DATA ·bitrev2048_m24+5080(SB)/8, $1900
DATA ·bitrev2048_m24+5088(SB)/8, $492
DATA ·bitrev2048_m24+5096(SB)/8, $1004
DATA ·bitrev2048_m24+5104(SB)/8, $1516
DATA ·bitrev2048_m24+5112(SB)/8, $2028
DATA ·bitrev2048_m24+5120(SB)/8, $20
DATA ·bitrev2048_m24+5128(SB)/8, $532
DATA ·bitrev2048_m24+5136(SB)/8, $1044
DATA ·bitrev2048_m24+5144(SB)/8, $1556
DATA ·bitrev2048_m24+5152(SB)/8, $148
DATA ·bitrev2048_m24+5160(SB)/8, $660
DATA ·bitrev2048_m24+5168(SB)/8, $1172
DATA ·bitrev2048_m24+5176(SB)/8, $1684
DATA ·bitrev2048_m24+5184(SB)/8, $276
DATA ·bitrev2048_m24+5192(SB)/8, $788
DATA ·bitrev2048_m24+5200(SB)/8, $1300
DATA ·bitrev2048_m24+5208(SB)/8, $1812
DATA ·bitrev2048_m24+5216(SB)/8, $404
DATA ·bitrev2048_m24+5224(SB)/8, $916
DATA ·bitrev2048_m24+5232(SB)/8, $1428
DATA ·bitrev2048_m24+5240(SB)/8, $1940
DATA ·bitrev2048_m24+5248(SB)/8, $52
DATA ·bitrev2048_m24+5256(SB)/8, $564
DATA ·bitrev2048_m24+5264(SB)/8, $1076
DATA ·bitrev2048_m24+5272(SB)/8, $1588
DATA ·bitrev2048_m24+5280(SB)/8, $180
DATA ·bitrev2048_m24+5288(SB)/8, $692
DATA ·bitrev2048_m24+5296(SB)/8, $1204
DATA ·bitrev2048_m24+5304(SB)/8, $1716
DATA ·bitrev2048_m24+5312(SB)/8, $308
DATA ·bitrev2048_m24+5320(SB)/8, $820
DATA ·bitrev2048_m24+5328(SB)/8, $1332
DATA ·bitrev2048_m24+5336(SB)/8, $1844
DATA ·bitrev2048_m24+5344(SB)/8, $436
DATA ·bitrev2048_m24+5352(SB)/8, $948
DATA ·bitrev2048_m24+5360(SB)/8, $1460
DATA ·bitrev2048_m24+5368(SB)/8, $1972
DATA ·bitrev2048_m24+5376(SB)/8, $84
DATA ·bitrev2048_m24+5384(SB)/8, $596
DATA ·bitrev2048_m24+5392(SB)/8, $1108
DATA ·bitrev2048_m24+5400(SB)/8, $1620
DATA ·bitrev2048_m24+5408(SB)/8, $212
DATA ·bitrev2048_m24+5416(SB)/8, $724
DATA ·bitrev2048_m24+5424(SB)/8, $1236
DATA ·bitrev2048_m24+5432(SB)/8, $1748
DATA ·bitrev2048_m24+5440(SB)/8, $340
DATA ·bitrev2048_m24+5448(SB)/8, $852
DATA ·bitrev2048_m24+5456(SB)/8, $1364
DATA ·bitrev2048_m24+5464(SB)/8, $1876
DATA ·bitrev2048_m24+5472(SB)/8, $468
DATA ·bitrev2048_m24+5480(SB)/8, $980
DATA ·bitrev2048_m24+5488(SB)/8, $1492
DATA ·bitrev2048_m24+5496(SB)/8, $2004
DATA ·bitrev2048_m24+5504(SB)/8, $116
DATA ·bitrev2048_m24+5512(SB)/8, $628
DATA ·bitrev2048_m24+5520(SB)/8, $1140
DATA ·bitrev2048_m24+5528(SB)/8, $1652
DATA ·bitrev2048_m24+5536(SB)/8, $244
DATA ·bitrev2048_m24+5544(SB)/8, $756
DATA ·bitrev2048_m24+5552(SB)/8, $1268
DATA ·bitrev2048_m24+5560(SB)/8, $1780
DATA ·bitrev2048_m24+5568(SB)/8, $372
DATA ·bitrev2048_m24+5576(SB)/8, $884
DATA ·bitrev2048_m24+5584(SB)/8, $1396
DATA ·bitrev2048_m24+5592(SB)/8, $1908
DATA ·bitrev2048_m24+5600(SB)/8, $500
DATA ·bitrev2048_m24+5608(SB)/8, $1012
DATA ·bitrev2048_m24+5616(SB)/8, $1524
DATA ·bitrev2048_m24+5624(SB)/8, $2036
DATA ·bitrev2048_m24+5632(SB)/8, $28
DATA ·bitrev2048_m24+5640(SB)/8, $540
DATA ·bitrev2048_m24+5648(SB)/8, $1052
DATA ·bitrev2048_m24+5656(SB)/8, $1564
DATA ·bitrev2048_m24+5664(SB)/8, $156
DATA ·bitrev2048_m24+5672(SB)/8, $668
DATA ·bitrev2048_m24+5680(SB)/8, $1180
DATA ·bitrev2048_m24+5688(SB)/8, $1692
DATA ·bitrev2048_m24+5696(SB)/8, $284
DATA ·bitrev2048_m24+5704(SB)/8, $796
DATA ·bitrev2048_m24+5712(SB)/8, $1308
DATA ·bitrev2048_m24+5720(SB)/8, $1820
DATA ·bitrev2048_m24+5728(SB)/8, $412
DATA ·bitrev2048_m24+5736(SB)/8, $924
DATA ·bitrev2048_m24+5744(SB)/8, $1436
DATA ·bitrev2048_m24+5752(SB)/8, $1948
DATA ·bitrev2048_m24+5760(SB)/8, $60
DATA ·bitrev2048_m24+5768(SB)/8, $572
DATA ·bitrev2048_m24+5776(SB)/8, $1084
DATA ·bitrev2048_m24+5784(SB)/8, $1596
DATA ·bitrev2048_m24+5792(SB)/8, $188
DATA ·bitrev2048_m24+5800(SB)/8, $700
DATA ·bitrev2048_m24+5808(SB)/8, $1212
DATA ·bitrev2048_m24+5816(SB)/8, $1724
DATA ·bitrev2048_m24+5824(SB)/8, $316
DATA ·bitrev2048_m24+5832(SB)/8, $828
DATA ·bitrev2048_m24+5840(SB)/8, $1340
DATA ·bitrev2048_m24+5848(SB)/8, $1852
DATA ·bitrev2048_m24+5856(SB)/8, $444
DATA ·bitrev2048_m24+5864(SB)/8, $956
DATA ·bitrev2048_m24+5872(SB)/8, $1468
DATA ·bitrev2048_m24+5880(SB)/8, $1980
DATA ·bitrev2048_m24+5888(SB)/8, $92
DATA ·bitrev2048_m24+5896(SB)/8, $604
DATA ·bitrev2048_m24+5904(SB)/8, $1116
DATA ·bitrev2048_m24+5912(SB)/8, $1628
DATA ·bitrev2048_m24+5920(SB)/8, $220
DATA ·bitrev2048_m24+5928(SB)/8, $732
DATA ·bitrev2048_m24+5936(SB)/8, $1244
DATA ·bitrev2048_m24+5944(SB)/8, $1756
DATA ·bitrev2048_m24+5952(SB)/8, $348
DATA ·bitrev2048_m24+5960(SB)/8, $860
DATA ·bitrev2048_m24+5968(SB)/8, $1372
DATA ·bitrev2048_m24+5976(SB)/8, $1884
DATA ·bitrev2048_m24+5984(SB)/8, $476
DATA ·bitrev2048_m24+5992(SB)/8, $988
DATA ·bitrev2048_m24+6000(SB)/8, $1500
DATA ·bitrev2048_m24+6008(SB)/8, $2012
DATA ·bitrev2048_m24+6016(SB)/8, $124
DATA ·bitrev2048_m24+6024(SB)/8, $636
DATA ·bitrev2048_m24+6032(SB)/8, $1148
DATA ·bitrev2048_m24+6040(SB)/8, $1660
DATA ·bitrev2048_m24+6048(SB)/8, $252
DATA ·bitrev2048_m24+6056(SB)/8, $764
DATA ·bitrev2048_m24+6064(SB)/8, $1276
DATA ·bitrev2048_m24+6072(SB)/8, $1788
DATA ·bitrev2048_m24+6080(SB)/8, $380
DATA ·bitrev2048_m24+6088(SB)/8, $892
DATA ·bitrev2048_m24+6096(SB)/8, $1404
DATA ·bitrev2048_m24+6104(SB)/8, $1916
DATA ·bitrev2048_m24+6112(SB)/8, $508
DATA ·bitrev2048_m24+6120(SB)/8, $1020
DATA ·bitrev2048_m24+6128(SB)/8, $1532
DATA ·bitrev2048_m24+6136(SB)/8, $2044
DATA ·bitrev2048_m24+6144(SB)/8, $6
DATA ·bitrev2048_m24+6152(SB)/8, $518
DATA ·bitrev2048_m24+6160(SB)/8, $1030
DATA ·bitrev2048_m24+6168(SB)/8, $1542
DATA ·bitrev2048_m24+6176(SB)/8, $134
DATA ·bitrev2048_m24+6184(SB)/8, $646
DATA ·bitrev2048_m24+6192(SB)/8, $1158
DATA ·bitrev2048_m24+6200(SB)/8, $1670
DATA ·bitrev2048_m24+6208(SB)/8, $262
DATA ·bitrev2048_m24+6216(SB)/8, $774
DATA ·bitrev2048_m24+6224(SB)/8, $1286
DATA ·bitrev2048_m24+6232(SB)/8, $1798
DATA ·bitrev2048_m24+6240(SB)/8, $390
DATA ·bitrev2048_m24+6248(SB)/8, $902
DATA ·bitrev2048_m24+6256(SB)/8, $1414
DATA ·bitrev2048_m24+6264(SB)/8, $1926
DATA ·bitrev2048_m24+6272(SB)/8, $38
DATA ·bitrev2048_m24+6280(SB)/8, $550
DATA ·bitrev2048_m24+6288(SB)/8, $1062
DATA ·bitrev2048_m24+6296(SB)/8, $1574
DATA ·bitrev2048_m24+6304(SB)/8, $166
DATA ·bitrev2048_m24+6312(SB)/8, $678
DATA ·bitrev2048_m24+6320(SB)/8, $1190
DATA ·bitrev2048_m24+6328(SB)/8, $1702
DATA ·bitrev2048_m24+6336(SB)/8, $294
DATA ·bitrev2048_m24+6344(SB)/8, $806
DATA ·bitrev2048_m24+6352(SB)/8, $1318
DATA ·bitrev2048_m24+6360(SB)/8, $1830
DATA ·bitrev2048_m24+6368(SB)/8, $422
DATA ·bitrev2048_m24+6376(SB)/8, $934
DATA ·bitrev2048_m24+6384(SB)/8, $1446
DATA ·bitrev2048_m24+6392(SB)/8, $1958
DATA ·bitrev2048_m24+6400(SB)/8, $70
DATA ·bitrev2048_m24+6408(SB)/8, $582
DATA ·bitrev2048_m24+6416(SB)/8, $1094
DATA ·bitrev2048_m24+6424(SB)/8, $1606
DATA ·bitrev2048_m24+6432(SB)/8, $198
DATA ·bitrev2048_m24+6440(SB)/8, $710
DATA ·bitrev2048_m24+6448(SB)/8, $1222
DATA ·bitrev2048_m24+6456(SB)/8, $1734
DATA ·bitrev2048_m24+6464(SB)/8, $326
DATA ·bitrev2048_m24+6472(SB)/8, $838
DATA ·bitrev2048_m24+6480(SB)/8, $1350
DATA ·bitrev2048_m24+6488(SB)/8, $1862
DATA ·bitrev2048_m24+6496(SB)/8, $454
DATA ·bitrev2048_m24+6504(SB)/8, $966
DATA ·bitrev2048_m24+6512(SB)/8, $1478
DATA ·bitrev2048_m24+6520(SB)/8, $1990
DATA ·bitrev2048_m24+6528(SB)/8, $102
DATA ·bitrev2048_m24+6536(SB)/8, $614
DATA ·bitrev2048_m24+6544(SB)/8, $1126
DATA ·bitrev2048_m24+6552(SB)/8, $1638
DATA ·bitrev2048_m24+6560(SB)/8, $230
DATA ·bitrev2048_m24+6568(SB)/8, $742
DATA ·bitrev2048_m24+6576(SB)/8, $1254
DATA ·bitrev2048_m24+6584(SB)/8, $1766
DATA ·bitrev2048_m24+6592(SB)/8, $358
DATA ·bitrev2048_m24+6600(SB)/8, $870
DATA ·bitrev2048_m24+6608(SB)/8, $1382
DATA ·bitrev2048_m24+6616(SB)/8, $1894
DATA ·bitrev2048_m24+6624(SB)/8, $486
DATA ·bitrev2048_m24+6632(SB)/8, $998
DATA ·bitrev2048_m24+6640(SB)/8, $1510
DATA ·bitrev2048_m24+6648(SB)/8, $2022
DATA ·bitrev2048_m24+6656(SB)/8, $14
DATA ·bitrev2048_m24+6664(SB)/8, $526
DATA ·bitrev2048_m24+6672(SB)/8, $1038
DATA ·bitrev2048_m24+6680(SB)/8, $1550
DATA ·bitrev2048_m24+6688(SB)/8, $142
DATA ·bitrev2048_m24+6696(SB)/8, $654
DATA ·bitrev2048_m24+6704(SB)/8, $1166
DATA ·bitrev2048_m24+6712(SB)/8, $1678
DATA ·bitrev2048_m24+6720(SB)/8, $270
DATA ·bitrev2048_m24+6728(SB)/8, $782
DATA ·bitrev2048_m24+6736(SB)/8, $1294
DATA ·bitrev2048_m24+6744(SB)/8, $1806
DATA ·bitrev2048_m24+6752(SB)/8, $398
DATA ·bitrev2048_m24+6760(SB)/8, $910
DATA ·bitrev2048_m24+6768(SB)/8, $1422
DATA ·bitrev2048_m24+6776(SB)/8, $1934
DATA ·bitrev2048_m24+6784(SB)/8, $46
DATA ·bitrev2048_m24+6792(SB)/8, $558
DATA ·bitrev2048_m24+6800(SB)/8, $1070
DATA ·bitrev2048_m24+6808(SB)/8, $1582
DATA ·bitrev2048_m24+6816(SB)/8, $174
DATA ·bitrev2048_m24+6824(SB)/8, $686
DATA ·bitrev2048_m24+6832(SB)/8, $1198
DATA ·bitrev2048_m24+6840(SB)/8, $1710
DATA ·bitrev2048_m24+6848(SB)/8, $302
DATA ·bitrev2048_m24+6856(SB)/8, $814
DATA ·bitrev2048_m24+6864(SB)/8, $1326
DATA ·bitrev2048_m24+6872(SB)/8, $1838
DATA ·bitrev2048_m24+6880(SB)/8, $430
DATA ·bitrev2048_m24+6888(SB)/8, $942
DATA ·bitrev2048_m24+6896(SB)/8, $1454
DATA ·bitrev2048_m24+6904(SB)/8, $1966
DATA ·bitrev2048_m24+6912(SB)/8, $78
DATA ·bitrev2048_m24+6920(SB)/8, $590
DATA ·bitrev2048_m24+6928(SB)/8, $1102
DATA ·bitrev2048_m24+6936(SB)/8, $1614
DATA ·bitrev2048_m24+6944(SB)/8, $206
DATA ·bitrev2048_m24+6952(SB)/8, $718
DATA ·bitrev2048_m24+6960(SB)/8, $1230
DATA ·bitrev2048_m24+6968(SB)/8, $1742
DATA ·bitrev2048_m24+6976(SB)/8, $334
DATA ·bitrev2048_m24+6984(SB)/8, $846
DATA ·bitrev2048_m24+6992(SB)/8, $1358
DATA ·bitrev2048_m24+7000(SB)/8, $1870
DATA ·bitrev2048_m24+7008(SB)/8, $462
DATA ·bitrev2048_m24+7016(SB)/8, $974
DATA ·bitrev2048_m24+7024(SB)/8, $1486
DATA ·bitrev2048_m24+7032(SB)/8, $1998
DATA ·bitrev2048_m24+7040(SB)/8, $110
DATA ·bitrev2048_m24+7048(SB)/8, $622
DATA ·bitrev2048_m24+7056(SB)/8, $1134
DATA ·bitrev2048_m24+7064(SB)/8, $1646
DATA ·bitrev2048_m24+7072(SB)/8, $238
DATA ·bitrev2048_m24+7080(SB)/8, $750
DATA ·bitrev2048_m24+7088(SB)/8, $1262
DATA ·bitrev2048_m24+7096(SB)/8, $1774
DATA ·bitrev2048_m24+7104(SB)/8, $366
DATA ·bitrev2048_m24+7112(SB)/8, $878
DATA ·bitrev2048_m24+7120(SB)/8, $1390
DATA ·bitrev2048_m24+7128(SB)/8, $1902
DATA ·bitrev2048_m24+7136(SB)/8, $494
DATA ·bitrev2048_m24+7144(SB)/8, $1006
DATA ·bitrev2048_m24+7152(SB)/8, $1518
DATA ·bitrev2048_m24+7160(SB)/8, $2030
DATA ·bitrev2048_m24+7168(SB)/8, $22
DATA ·bitrev2048_m24+7176(SB)/8, $534
DATA ·bitrev2048_m24+7184(SB)/8, $1046
DATA ·bitrev2048_m24+7192(SB)/8, $1558
DATA ·bitrev2048_m24+7200(SB)/8, $150
DATA ·bitrev2048_m24+7208(SB)/8, $662
DATA ·bitrev2048_m24+7216(SB)/8, $1174
DATA ·bitrev2048_m24+7224(SB)/8, $1686
DATA ·bitrev2048_m24+7232(SB)/8, $278
DATA ·bitrev2048_m24+7240(SB)/8, $790
DATA ·bitrev2048_m24+7248(SB)/8, $1302
DATA ·bitrev2048_m24+7256(SB)/8, $1814
DATA ·bitrev2048_m24+7264(SB)/8, $406
DATA ·bitrev2048_m24+7272(SB)/8, $918
DATA ·bitrev2048_m24+7280(SB)/8, $1430
DATA ·bitrev2048_m24+7288(SB)/8, $1942
DATA ·bitrev2048_m24+7296(SB)/8, $54
DATA ·bitrev2048_m24+7304(SB)/8, $566
DATA ·bitrev2048_m24+7312(SB)/8, $1078
DATA ·bitrev2048_m24+7320(SB)/8, $1590
DATA ·bitrev2048_m24+7328(SB)/8, $182
DATA ·bitrev2048_m24+7336(SB)/8, $694
DATA ·bitrev2048_m24+7344(SB)/8, $1206
DATA ·bitrev2048_m24+7352(SB)/8, $1718
DATA ·bitrev2048_m24+7360(SB)/8, $310
DATA ·bitrev2048_m24+7368(SB)/8, $822
DATA ·bitrev2048_m24+7376(SB)/8, $1334
DATA ·bitrev2048_m24+7384(SB)/8, $1846
DATA ·bitrev2048_m24+7392(SB)/8, $438
DATA ·bitrev2048_m24+7400(SB)/8, $950
DATA ·bitrev2048_m24+7408(SB)/8, $1462
DATA ·bitrev2048_m24+7416(SB)/8, $1974
DATA ·bitrev2048_m24+7424(SB)/8, $86
DATA ·bitrev2048_m24+7432(SB)/8, $598
DATA ·bitrev2048_m24+7440(SB)/8, $1110
DATA ·bitrev2048_m24+7448(SB)/8, $1622
DATA ·bitrev2048_m24+7456(SB)/8, $214
DATA ·bitrev2048_m24+7464(SB)/8, $726
DATA ·bitrev2048_m24+7472(SB)/8, $1238
DATA ·bitrev2048_m24+7480(SB)/8, $1750
DATA ·bitrev2048_m24+7488(SB)/8, $342
DATA ·bitrev2048_m24+7496(SB)/8, $854
DATA ·bitrev2048_m24+7504(SB)/8, $1366
DATA ·bitrev2048_m24+7512(SB)/8, $1878
DATA ·bitrev2048_m24+7520(SB)/8, $470
DATA ·bitrev2048_m24+7528(SB)/8, $982
DATA ·bitrev2048_m24+7536(SB)/8, $1494
DATA ·bitrev2048_m24+7544(SB)/8, $2006
DATA ·bitrev2048_m24+7552(SB)/8, $118
DATA ·bitrev2048_m24+7560(SB)/8, $630
DATA ·bitrev2048_m24+7568(SB)/8, $1142
DATA ·bitrev2048_m24+7576(SB)/8, $1654
DATA ·bitrev2048_m24+7584(SB)/8, $246
DATA ·bitrev2048_m24+7592(SB)/8, $758
DATA ·bitrev2048_m24+7600(SB)/8, $1270
DATA ·bitrev2048_m24+7608(SB)/8, $1782
DATA ·bitrev2048_m24+7616(SB)/8, $374
DATA ·bitrev2048_m24+7624(SB)/8, $886
DATA ·bitrev2048_m24+7632(SB)/8, $1398
DATA ·bitrev2048_m24+7640(SB)/8, $1910
DATA ·bitrev2048_m24+7648(SB)/8, $502
DATA ·bitrev2048_m24+7656(SB)/8, $1014
DATA ·bitrev2048_m24+7664(SB)/8, $1526
DATA ·bitrev2048_m24+7672(SB)/8, $2038
DATA ·bitrev2048_m24+7680(SB)/8, $30
DATA ·bitrev2048_m24+7688(SB)/8, $542
DATA ·bitrev2048_m24+7696(SB)/8, $1054
DATA ·bitrev2048_m24+7704(SB)/8, $1566
DATA ·bitrev2048_m24+7712(SB)/8, $158
DATA ·bitrev2048_m24+7720(SB)/8, $670
DATA ·bitrev2048_m24+7728(SB)/8, $1182
DATA ·bitrev2048_m24+7736(SB)/8, $1694
DATA ·bitrev2048_m24+7744(SB)/8, $286
DATA ·bitrev2048_m24+7752(SB)/8, $798
DATA ·bitrev2048_m24+7760(SB)/8, $1310
DATA ·bitrev2048_m24+7768(SB)/8, $1822
DATA ·bitrev2048_m24+7776(SB)/8, $414
DATA ·bitrev2048_m24+7784(SB)/8, $926
DATA ·bitrev2048_m24+7792(SB)/8, $1438
DATA ·bitrev2048_m24+7800(SB)/8, $1950
DATA ·bitrev2048_m24+7808(SB)/8, $62
DATA ·bitrev2048_m24+7816(SB)/8, $574
DATA ·bitrev2048_m24+7824(SB)/8, $1086
DATA ·bitrev2048_m24+7832(SB)/8, $1598
DATA ·bitrev2048_m24+7840(SB)/8, $190
DATA ·bitrev2048_m24+7848(SB)/8, $702
DATA ·bitrev2048_m24+7856(SB)/8, $1214
DATA ·bitrev2048_m24+7864(SB)/8, $1726
DATA ·bitrev2048_m24+7872(SB)/8, $318
DATA ·bitrev2048_m24+7880(SB)/8, $830
DATA ·bitrev2048_m24+7888(SB)/8, $1342
DATA ·bitrev2048_m24+7896(SB)/8, $1854
DATA ·bitrev2048_m24+7904(SB)/8, $446
DATA ·bitrev2048_m24+7912(SB)/8, $958
DATA ·bitrev2048_m24+7920(SB)/8, $1470
DATA ·bitrev2048_m24+7928(SB)/8, $1982
DATA ·bitrev2048_m24+7936(SB)/8, $94
DATA ·bitrev2048_m24+7944(SB)/8, $606
DATA ·bitrev2048_m24+7952(SB)/8, $1118
DATA ·bitrev2048_m24+7960(SB)/8, $1630
DATA ·bitrev2048_m24+7968(SB)/8, $222
DATA ·bitrev2048_m24+7976(SB)/8, $734
DATA ·bitrev2048_m24+7984(SB)/8, $1246
DATA ·bitrev2048_m24+7992(SB)/8, $1758
DATA ·bitrev2048_m24+8000(SB)/8, $350
DATA ·bitrev2048_m24+8008(SB)/8, $862
DATA ·bitrev2048_m24+8016(SB)/8, $1374
DATA ·bitrev2048_m24+8024(SB)/8, $1886
DATA ·bitrev2048_m24+8032(SB)/8, $478
DATA ·bitrev2048_m24+8040(SB)/8, $990
DATA ·bitrev2048_m24+8048(SB)/8, $1502
DATA ·bitrev2048_m24+8056(SB)/8, $2014
DATA ·bitrev2048_m24+8064(SB)/8, $126
DATA ·bitrev2048_m24+8072(SB)/8, $638
DATA ·bitrev2048_m24+8080(SB)/8, $1150
DATA ·bitrev2048_m24+8088(SB)/8, $1662
DATA ·bitrev2048_m24+8096(SB)/8, $254
DATA ·bitrev2048_m24+8104(SB)/8, $766
DATA ·bitrev2048_m24+8112(SB)/8, $1278
DATA ·bitrev2048_m24+8120(SB)/8, $1790
DATA ·bitrev2048_m24+8128(SB)/8, $382
DATA ·bitrev2048_m24+8136(SB)/8, $894
DATA ·bitrev2048_m24+8144(SB)/8, $1406
DATA ·bitrev2048_m24+8152(SB)/8, $1918
DATA ·bitrev2048_m24+8160(SB)/8, $510
DATA ·bitrev2048_m24+8168(SB)/8, $1022
DATA ·bitrev2048_m24+8176(SB)/8, $1534
DATA ·bitrev2048_m24+8184(SB)/8, $2046
DATA ·bitrev2048_m24+8192(SB)/8, $1
DATA ·bitrev2048_m24+8200(SB)/8, $513
DATA ·bitrev2048_m24+8208(SB)/8, $1025
DATA ·bitrev2048_m24+8216(SB)/8, $1537
DATA ·bitrev2048_m24+8224(SB)/8, $129
DATA ·bitrev2048_m24+8232(SB)/8, $641
DATA ·bitrev2048_m24+8240(SB)/8, $1153
DATA ·bitrev2048_m24+8248(SB)/8, $1665
DATA ·bitrev2048_m24+8256(SB)/8, $257
DATA ·bitrev2048_m24+8264(SB)/8, $769
DATA ·bitrev2048_m24+8272(SB)/8, $1281
DATA ·bitrev2048_m24+8280(SB)/8, $1793
DATA ·bitrev2048_m24+8288(SB)/8, $385
DATA ·bitrev2048_m24+8296(SB)/8, $897
DATA ·bitrev2048_m24+8304(SB)/8, $1409
DATA ·bitrev2048_m24+8312(SB)/8, $1921
DATA ·bitrev2048_m24+8320(SB)/8, $33
DATA ·bitrev2048_m24+8328(SB)/8, $545
DATA ·bitrev2048_m24+8336(SB)/8, $1057
DATA ·bitrev2048_m24+8344(SB)/8, $1569
DATA ·bitrev2048_m24+8352(SB)/8, $161
DATA ·bitrev2048_m24+8360(SB)/8, $673
DATA ·bitrev2048_m24+8368(SB)/8, $1185
DATA ·bitrev2048_m24+8376(SB)/8, $1697
DATA ·bitrev2048_m24+8384(SB)/8, $289
DATA ·bitrev2048_m24+8392(SB)/8, $801
DATA ·bitrev2048_m24+8400(SB)/8, $1313
DATA ·bitrev2048_m24+8408(SB)/8, $1825
DATA ·bitrev2048_m24+8416(SB)/8, $417
DATA ·bitrev2048_m24+8424(SB)/8, $929
DATA ·bitrev2048_m24+8432(SB)/8, $1441
DATA ·bitrev2048_m24+8440(SB)/8, $1953
DATA ·bitrev2048_m24+8448(SB)/8, $65
DATA ·bitrev2048_m24+8456(SB)/8, $577
DATA ·bitrev2048_m24+8464(SB)/8, $1089
DATA ·bitrev2048_m24+8472(SB)/8, $1601
DATA ·bitrev2048_m24+8480(SB)/8, $193
DATA ·bitrev2048_m24+8488(SB)/8, $705
DATA ·bitrev2048_m24+8496(SB)/8, $1217
DATA ·bitrev2048_m24+8504(SB)/8, $1729
DATA ·bitrev2048_m24+8512(SB)/8, $321
DATA ·bitrev2048_m24+8520(SB)/8, $833
DATA ·bitrev2048_m24+8528(SB)/8, $1345
DATA ·bitrev2048_m24+8536(SB)/8, $1857
DATA ·bitrev2048_m24+8544(SB)/8, $449
DATA ·bitrev2048_m24+8552(SB)/8, $961
DATA ·bitrev2048_m24+8560(SB)/8, $1473
DATA ·bitrev2048_m24+8568(SB)/8, $1985
DATA ·bitrev2048_m24+8576(SB)/8, $97
DATA ·bitrev2048_m24+8584(SB)/8, $609
DATA ·bitrev2048_m24+8592(SB)/8, $1121
DATA ·bitrev2048_m24+8600(SB)/8, $1633
DATA ·bitrev2048_m24+8608(SB)/8, $225
DATA ·bitrev2048_m24+8616(SB)/8, $737
DATA ·bitrev2048_m24+8624(SB)/8, $1249
DATA ·bitrev2048_m24+8632(SB)/8, $1761
DATA ·bitrev2048_m24+8640(SB)/8, $353
DATA ·bitrev2048_m24+8648(SB)/8, $865
DATA ·bitrev2048_m24+8656(SB)/8, $1377
DATA ·bitrev2048_m24+8664(SB)/8, $1889
DATA ·bitrev2048_m24+8672(SB)/8, $481
DATA ·bitrev2048_m24+8680(SB)/8, $993
DATA ·bitrev2048_m24+8688(SB)/8, $1505
DATA ·bitrev2048_m24+8696(SB)/8, $2017
DATA ·bitrev2048_m24+8704(SB)/8, $9
DATA ·bitrev2048_m24+8712(SB)/8, $521
DATA ·bitrev2048_m24+8720(SB)/8, $1033
DATA ·bitrev2048_m24+8728(SB)/8, $1545
DATA ·bitrev2048_m24+8736(SB)/8, $137
DATA ·bitrev2048_m24+8744(SB)/8, $649
DATA ·bitrev2048_m24+8752(SB)/8, $1161
DATA ·bitrev2048_m24+8760(SB)/8, $1673
DATA ·bitrev2048_m24+8768(SB)/8, $265
DATA ·bitrev2048_m24+8776(SB)/8, $777
DATA ·bitrev2048_m24+8784(SB)/8, $1289
DATA ·bitrev2048_m24+8792(SB)/8, $1801
DATA ·bitrev2048_m24+8800(SB)/8, $393
DATA ·bitrev2048_m24+8808(SB)/8, $905
DATA ·bitrev2048_m24+8816(SB)/8, $1417
DATA ·bitrev2048_m24+8824(SB)/8, $1929
DATA ·bitrev2048_m24+8832(SB)/8, $41
DATA ·bitrev2048_m24+8840(SB)/8, $553
DATA ·bitrev2048_m24+8848(SB)/8, $1065
DATA ·bitrev2048_m24+8856(SB)/8, $1577
DATA ·bitrev2048_m24+8864(SB)/8, $169
DATA ·bitrev2048_m24+8872(SB)/8, $681
DATA ·bitrev2048_m24+8880(SB)/8, $1193
DATA ·bitrev2048_m24+8888(SB)/8, $1705
DATA ·bitrev2048_m24+8896(SB)/8, $297
DATA ·bitrev2048_m24+8904(SB)/8, $809
DATA ·bitrev2048_m24+8912(SB)/8, $1321
DATA ·bitrev2048_m24+8920(SB)/8, $1833
DATA ·bitrev2048_m24+8928(SB)/8, $425
DATA ·bitrev2048_m24+8936(SB)/8, $937
DATA ·bitrev2048_m24+8944(SB)/8, $1449
DATA ·bitrev2048_m24+8952(SB)/8, $1961
DATA ·bitrev2048_m24+8960(SB)/8, $73
DATA ·bitrev2048_m24+8968(SB)/8, $585
DATA ·bitrev2048_m24+8976(SB)/8, $1097
DATA ·bitrev2048_m24+8984(SB)/8, $1609
DATA ·bitrev2048_m24+8992(SB)/8, $201
DATA ·bitrev2048_m24+9000(SB)/8, $713
DATA ·bitrev2048_m24+9008(SB)/8, $1225
DATA ·bitrev2048_m24+9016(SB)/8, $1737
DATA ·bitrev2048_m24+9024(SB)/8, $329
DATA ·bitrev2048_m24+9032(SB)/8, $841
DATA ·bitrev2048_m24+9040(SB)/8, $1353
DATA ·bitrev2048_m24+9048(SB)/8, $1865
DATA ·bitrev2048_m24+9056(SB)/8, $457
DATA ·bitrev2048_m24+9064(SB)/8, $969
DATA ·bitrev2048_m24+9072(SB)/8, $1481
DATA ·bitrev2048_m24+9080(SB)/8, $1993
DATA ·bitrev2048_m24+9088(SB)/8, $105
DATA ·bitrev2048_m24+9096(SB)/8, $617
DATA ·bitrev2048_m24+9104(SB)/8, $1129
DATA ·bitrev2048_m24+9112(SB)/8, $1641
DATA ·bitrev2048_m24+9120(SB)/8, $233
DATA ·bitrev2048_m24+9128(SB)/8, $745
DATA ·bitrev2048_m24+9136(SB)/8, $1257
DATA ·bitrev2048_m24+9144(SB)/8, $1769
DATA ·bitrev2048_m24+9152(SB)/8, $361
DATA ·bitrev2048_m24+9160(SB)/8, $873
DATA ·bitrev2048_m24+9168(SB)/8, $1385
DATA ·bitrev2048_m24+9176(SB)/8, $1897
DATA ·bitrev2048_m24+9184(SB)/8, $489
DATA ·bitrev2048_m24+9192(SB)/8, $1001
DATA ·bitrev2048_m24+9200(SB)/8, $1513
DATA ·bitrev2048_m24+9208(SB)/8, $2025
DATA ·bitrev2048_m24+9216(SB)/8, $17
DATA ·bitrev2048_m24+9224(SB)/8, $529
DATA ·bitrev2048_m24+9232(SB)/8, $1041
DATA ·bitrev2048_m24+9240(SB)/8, $1553
DATA ·bitrev2048_m24+9248(SB)/8, $145
DATA ·bitrev2048_m24+9256(SB)/8, $657
DATA ·bitrev2048_m24+9264(SB)/8, $1169
DATA ·bitrev2048_m24+9272(SB)/8, $1681
DATA ·bitrev2048_m24+9280(SB)/8, $273
DATA ·bitrev2048_m24+9288(SB)/8, $785
DATA ·bitrev2048_m24+9296(SB)/8, $1297
DATA ·bitrev2048_m24+9304(SB)/8, $1809
DATA ·bitrev2048_m24+9312(SB)/8, $401
DATA ·bitrev2048_m24+9320(SB)/8, $913
DATA ·bitrev2048_m24+9328(SB)/8, $1425
DATA ·bitrev2048_m24+9336(SB)/8, $1937
DATA ·bitrev2048_m24+9344(SB)/8, $49
DATA ·bitrev2048_m24+9352(SB)/8, $561
DATA ·bitrev2048_m24+9360(SB)/8, $1073
DATA ·bitrev2048_m24+9368(SB)/8, $1585
DATA ·bitrev2048_m24+9376(SB)/8, $177
DATA ·bitrev2048_m24+9384(SB)/8, $689
DATA ·bitrev2048_m24+9392(SB)/8, $1201
DATA ·bitrev2048_m24+9400(SB)/8, $1713
DATA ·bitrev2048_m24+9408(SB)/8, $305
DATA ·bitrev2048_m24+9416(SB)/8, $817
DATA ·bitrev2048_m24+9424(SB)/8, $1329
DATA ·bitrev2048_m24+9432(SB)/8, $1841
DATA ·bitrev2048_m24+9440(SB)/8, $433
DATA ·bitrev2048_m24+9448(SB)/8, $945
DATA ·bitrev2048_m24+9456(SB)/8, $1457
DATA ·bitrev2048_m24+9464(SB)/8, $1969
DATA ·bitrev2048_m24+9472(SB)/8, $81
DATA ·bitrev2048_m24+9480(SB)/8, $593
DATA ·bitrev2048_m24+9488(SB)/8, $1105
DATA ·bitrev2048_m24+9496(SB)/8, $1617
DATA ·bitrev2048_m24+9504(SB)/8, $209
DATA ·bitrev2048_m24+9512(SB)/8, $721
DATA ·bitrev2048_m24+9520(SB)/8, $1233
DATA ·bitrev2048_m24+9528(SB)/8, $1745
DATA ·bitrev2048_m24+9536(SB)/8, $337
DATA ·bitrev2048_m24+9544(SB)/8, $849
DATA ·bitrev2048_m24+9552(SB)/8, $1361
DATA ·bitrev2048_m24+9560(SB)/8, $1873
DATA ·bitrev2048_m24+9568(SB)/8, $465
DATA ·bitrev2048_m24+9576(SB)/8, $977
DATA ·bitrev2048_m24+9584(SB)/8, $1489
DATA ·bitrev2048_m24+9592(SB)/8, $2001
DATA ·bitrev2048_m24+9600(SB)/8, $113
DATA ·bitrev2048_m24+9608(SB)/8, $625
DATA ·bitrev2048_m24+9616(SB)/8, $1137
DATA ·bitrev2048_m24+9624(SB)/8, $1649
DATA ·bitrev2048_m24+9632(SB)/8, $241
DATA ·bitrev2048_m24+9640(SB)/8, $753
DATA ·bitrev2048_m24+9648(SB)/8, $1265
DATA ·bitrev2048_m24+9656(SB)/8, $1777
DATA ·bitrev2048_m24+9664(SB)/8, $369
DATA ·bitrev2048_m24+9672(SB)/8, $881
DATA ·bitrev2048_m24+9680(SB)/8, $1393
DATA ·bitrev2048_m24+9688(SB)/8, $1905
DATA ·bitrev2048_m24+9696(SB)/8, $497
DATA ·bitrev2048_m24+9704(SB)/8, $1009
DATA ·bitrev2048_m24+9712(SB)/8, $1521
DATA ·bitrev2048_m24+9720(SB)/8, $2033
DATA ·bitrev2048_m24+9728(SB)/8, $25
DATA ·bitrev2048_m24+9736(SB)/8, $537
DATA ·bitrev2048_m24+9744(SB)/8, $1049
DATA ·bitrev2048_m24+9752(SB)/8, $1561
DATA ·bitrev2048_m24+9760(SB)/8, $153
DATA ·bitrev2048_m24+9768(SB)/8, $665
DATA ·bitrev2048_m24+9776(SB)/8, $1177
DATA ·bitrev2048_m24+9784(SB)/8, $1689
DATA ·bitrev2048_m24+9792(SB)/8, $281
DATA ·bitrev2048_m24+9800(SB)/8, $793
DATA ·bitrev2048_m24+9808(SB)/8, $1305
DATA ·bitrev2048_m24+9816(SB)/8, $1817
DATA ·bitrev2048_m24+9824(SB)/8, $409
DATA ·bitrev2048_m24+9832(SB)/8, $921
DATA ·bitrev2048_m24+9840(SB)/8, $1433
DATA ·bitrev2048_m24+9848(SB)/8, $1945
DATA ·bitrev2048_m24+9856(SB)/8, $57
DATA ·bitrev2048_m24+9864(SB)/8, $569
DATA ·bitrev2048_m24+9872(SB)/8, $1081
DATA ·bitrev2048_m24+9880(SB)/8, $1593
DATA ·bitrev2048_m24+9888(SB)/8, $185
DATA ·bitrev2048_m24+9896(SB)/8, $697
DATA ·bitrev2048_m24+9904(SB)/8, $1209
DATA ·bitrev2048_m24+9912(SB)/8, $1721
DATA ·bitrev2048_m24+9920(SB)/8, $313
DATA ·bitrev2048_m24+9928(SB)/8, $825
DATA ·bitrev2048_m24+9936(SB)/8, $1337
DATA ·bitrev2048_m24+9944(SB)/8, $1849
DATA ·bitrev2048_m24+9952(SB)/8, $441
DATA ·bitrev2048_m24+9960(SB)/8, $953
DATA ·bitrev2048_m24+9968(SB)/8, $1465
DATA ·bitrev2048_m24+9976(SB)/8, $1977
DATA ·bitrev2048_m24+9984(SB)/8, $89
DATA ·bitrev2048_m24+9992(SB)/8, $601
DATA ·bitrev2048_m24+10000(SB)/8, $1113
DATA ·bitrev2048_m24+10008(SB)/8, $1625
DATA ·bitrev2048_m24+10016(SB)/8, $217
DATA ·bitrev2048_m24+10024(SB)/8, $729
DATA ·bitrev2048_m24+10032(SB)/8, $1241
DATA ·bitrev2048_m24+10040(SB)/8, $1753
DATA ·bitrev2048_m24+10048(SB)/8, $345
DATA ·bitrev2048_m24+10056(SB)/8, $857
DATA ·bitrev2048_m24+10064(SB)/8, $1369
DATA ·bitrev2048_m24+10072(SB)/8, $1881
DATA ·bitrev2048_m24+10080(SB)/8, $473
DATA ·bitrev2048_m24+10088(SB)/8, $985
DATA ·bitrev2048_m24+10096(SB)/8, $1497
DATA ·bitrev2048_m24+10104(SB)/8, $2009
DATA ·bitrev2048_m24+10112(SB)/8, $121
DATA ·bitrev2048_m24+10120(SB)/8, $633
DATA ·bitrev2048_m24+10128(SB)/8, $1145
DATA ·bitrev2048_m24+10136(SB)/8, $1657
DATA ·bitrev2048_m24+10144(SB)/8, $249
DATA ·bitrev2048_m24+10152(SB)/8, $761
DATA ·bitrev2048_m24+10160(SB)/8, $1273
DATA ·bitrev2048_m24+10168(SB)/8, $1785
DATA ·bitrev2048_m24+10176(SB)/8, $377
DATA ·bitrev2048_m24+10184(SB)/8, $889
DATA ·bitrev2048_m24+10192(SB)/8, $1401
DATA ·bitrev2048_m24+10200(SB)/8, $1913
DATA ·bitrev2048_m24+10208(SB)/8, $505
DATA ·bitrev2048_m24+10216(SB)/8, $1017
DATA ·bitrev2048_m24+10224(SB)/8, $1529
DATA ·bitrev2048_m24+10232(SB)/8, $2041
DATA ·bitrev2048_m24+10240(SB)/8, $3
DATA ·bitrev2048_m24+10248(SB)/8, $515
DATA ·bitrev2048_m24+10256(SB)/8, $1027
DATA ·bitrev2048_m24+10264(SB)/8, $1539
DATA ·bitrev2048_m24+10272(SB)/8, $131
DATA ·bitrev2048_m24+10280(SB)/8, $643
DATA ·bitrev2048_m24+10288(SB)/8, $1155
DATA ·bitrev2048_m24+10296(SB)/8, $1667
DATA ·bitrev2048_m24+10304(SB)/8, $259
DATA ·bitrev2048_m24+10312(SB)/8, $771
DATA ·bitrev2048_m24+10320(SB)/8, $1283
DATA ·bitrev2048_m24+10328(SB)/8, $1795
DATA ·bitrev2048_m24+10336(SB)/8, $387
DATA ·bitrev2048_m24+10344(SB)/8, $899
DATA ·bitrev2048_m24+10352(SB)/8, $1411
DATA ·bitrev2048_m24+10360(SB)/8, $1923
DATA ·bitrev2048_m24+10368(SB)/8, $35
DATA ·bitrev2048_m24+10376(SB)/8, $547
DATA ·bitrev2048_m24+10384(SB)/8, $1059
DATA ·bitrev2048_m24+10392(SB)/8, $1571
DATA ·bitrev2048_m24+10400(SB)/8, $163
DATA ·bitrev2048_m24+10408(SB)/8, $675
DATA ·bitrev2048_m24+10416(SB)/8, $1187
DATA ·bitrev2048_m24+10424(SB)/8, $1699
DATA ·bitrev2048_m24+10432(SB)/8, $291
DATA ·bitrev2048_m24+10440(SB)/8, $803
DATA ·bitrev2048_m24+10448(SB)/8, $1315
DATA ·bitrev2048_m24+10456(SB)/8, $1827
DATA ·bitrev2048_m24+10464(SB)/8, $419
DATA ·bitrev2048_m24+10472(SB)/8, $931
DATA ·bitrev2048_m24+10480(SB)/8, $1443
DATA ·bitrev2048_m24+10488(SB)/8, $1955
DATA ·bitrev2048_m24+10496(SB)/8, $67
DATA ·bitrev2048_m24+10504(SB)/8, $579
DATA ·bitrev2048_m24+10512(SB)/8, $1091
DATA ·bitrev2048_m24+10520(SB)/8, $1603
DATA ·bitrev2048_m24+10528(SB)/8, $195
DATA ·bitrev2048_m24+10536(SB)/8, $707
DATA ·bitrev2048_m24+10544(SB)/8, $1219
DATA ·bitrev2048_m24+10552(SB)/8, $1731
DATA ·bitrev2048_m24+10560(SB)/8, $323
DATA ·bitrev2048_m24+10568(SB)/8, $835
DATA ·bitrev2048_m24+10576(SB)/8, $1347
DATA ·bitrev2048_m24+10584(SB)/8, $1859
DATA ·bitrev2048_m24+10592(SB)/8, $451
DATA ·bitrev2048_m24+10600(SB)/8, $963
DATA ·bitrev2048_m24+10608(SB)/8, $1475
DATA ·bitrev2048_m24+10616(SB)/8, $1987
DATA ·bitrev2048_m24+10624(SB)/8, $99
DATA ·bitrev2048_m24+10632(SB)/8, $611
DATA ·bitrev2048_m24+10640(SB)/8, $1123
DATA ·bitrev2048_m24+10648(SB)/8, $1635
DATA ·bitrev2048_m24+10656(SB)/8, $227
DATA ·bitrev2048_m24+10664(SB)/8, $739
DATA ·bitrev2048_m24+10672(SB)/8, $1251
DATA ·bitrev2048_m24+10680(SB)/8, $1763
DATA ·bitrev2048_m24+10688(SB)/8, $355
DATA ·bitrev2048_m24+10696(SB)/8, $867
DATA ·bitrev2048_m24+10704(SB)/8, $1379
DATA ·bitrev2048_m24+10712(SB)/8, $1891
DATA ·bitrev2048_m24+10720(SB)/8, $483
DATA ·bitrev2048_m24+10728(SB)/8, $995
DATA ·bitrev2048_m24+10736(SB)/8, $1507
DATA ·bitrev2048_m24+10744(SB)/8, $2019
DATA ·bitrev2048_m24+10752(SB)/8, $11
DATA ·bitrev2048_m24+10760(SB)/8, $523
DATA ·bitrev2048_m24+10768(SB)/8, $1035
DATA ·bitrev2048_m24+10776(SB)/8, $1547
DATA ·bitrev2048_m24+10784(SB)/8, $139
DATA ·bitrev2048_m24+10792(SB)/8, $651
DATA ·bitrev2048_m24+10800(SB)/8, $1163
DATA ·bitrev2048_m24+10808(SB)/8, $1675
DATA ·bitrev2048_m24+10816(SB)/8, $267
DATA ·bitrev2048_m24+10824(SB)/8, $779
DATA ·bitrev2048_m24+10832(SB)/8, $1291
DATA ·bitrev2048_m24+10840(SB)/8, $1803
DATA ·bitrev2048_m24+10848(SB)/8, $395
DATA ·bitrev2048_m24+10856(SB)/8, $907
DATA ·bitrev2048_m24+10864(SB)/8, $1419
DATA ·bitrev2048_m24+10872(SB)/8, $1931
DATA ·bitrev2048_m24+10880(SB)/8, $43
DATA ·bitrev2048_m24+10888(SB)/8, $555
DATA ·bitrev2048_m24+10896(SB)/8, $1067
DATA ·bitrev2048_m24+10904(SB)/8, $1579
DATA ·bitrev2048_m24+10912(SB)/8, $171
DATA ·bitrev2048_m24+10920(SB)/8, $683
DATA ·bitrev2048_m24+10928(SB)/8, $1195
DATA ·bitrev2048_m24+10936(SB)/8, $1707
DATA ·bitrev2048_m24+10944(SB)/8, $299
DATA ·bitrev2048_m24+10952(SB)/8, $811
DATA ·bitrev2048_m24+10960(SB)/8, $1323
DATA ·bitrev2048_m24+10968(SB)/8, $1835
DATA ·bitrev2048_m24+10976(SB)/8, $427
DATA ·bitrev2048_m24+10984(SB)/8, $939
DATA ·bitrev2048_m24+10992(SB)/8, $1451
DATA ·bitrev2048_m24+11000(SB)/8, $1963
DATA ·bitrev2048_m24+11008(SB)/8, $75
DATA ·bitrev2048_m24+11016(SB)/8, $587
DATA ·bitrev2048_m24+11024(SB)/8, $1099
DATA ·bitrev2048_m24+11032(SB)/8, $1611
DATA ·bitrev2048_m24+11040(SB)/8, $203
DATA ·bitrev2048_m24+11048(SB)/8, $715
DATA ·bitrev2048_m24+11056(SB)/8, $1227
DATA ·bitrev2048_m24+11064(SB)/8, $1739
DATA ·bitrev2048_m24+11072(SB)/8, $331
DATA ·bitrev2048_m24+11080(SB)/8, $843
DATA ·bitrev2048_m24+11088(SB)/8, $1355
DATA ·bitrev2048_m24+11096(SB)/8, $1867
DATA ·bitrev2048_m24+11104(SB)/8, $459
DATA ·bitrev2048_m24+11112(SB)/8, $971
DATA ·bitrev2048_m24+11120(SB)/8, $1483
DATA ·bitrev2048_m24+11128(SB)/8, $1995
DATA ·bitrev2048_m24+11136(SB)/8, $107
DATA ·bitrev2048_m24+11144(SB)/8, $619
DATA ·bitrev2048_m24+11152(SB)/8, $1131
DATA ·bitrev2048_m24+11160(SB)/8, $1643
DATA ·bitrev2048_m24+11168(SB)/8, $235
DATA ·bitrev2048_m24+11176(SB)/8, $747
DATA ·bitrev2048_m24+11184(SB)/8, $1259
DATA ·bitrev2048_m24+11192(SB)/8, $1771
DATA ·bitrev2048_m24+11200(SB)/8, $363
DATA ·bitrev2048_m24+11208(SB)/8, $875
DATA ·bitrev2048_m24+11216(SB)/8, $1387
DATA ·bitrev2048_m24+11224(SB)/8, $1899
DATA ·bitrev2048_m24+11232(SB)/8, $491
DATA ·bitrev2048_m24+11240(SB)/8, $1003
DATA ·bitrev2048_m24+11248(SB)/8, $1515
DATA ·bitrev2048_m24+11256(SB)/8, $2027
DATA ·bitrev2048_m24+11264(SB)/8, $19
DATA ·bitrev2048_m24+11272(SB)/8, $531
DATA ·bitrev2048_m24+11280(SB)/8, $1043
DATA ·bitrev2048_m24+11288(SB)/8, $1555
DATA ·bitrev2048_m24+11296(SB)/8, $147
DATA ·bitrev2048_m24+11304(SB)/8, $659
DATA ·bitrev2048_m24+11312(SB)/8, $1171
DATA ·bitrev2048_m24+11320(SB)/8, $1683
DATA ·bitrev2048_m24+11328(SB)/8, $275
DATA ·bitrev2048_m24+11336(SB)/8, $787
DATA ·bitrev2048_m24+11344(SB)/8, $1299
DATA ·bitrev2048_m24+11352(SB)/8, $1811
DATA ·bitrev2048_m24+11360(SB)/8, $403
DATA ·bitrev2048_m24+11368(SB)/8, $915
DATA ·bitrev2048_m24+11376(SB)/8, $1427
DATA ·bitrev2048_m24+11384(SB)/8, $1939
DATA ·bitrev2048_m24+11392(SB)/8, $51
DATA ·bitrev2048_m24+11400(SB)/8, $563
DATA ·bitrev2048_m24+11408(SB)/8, $1075
DATA ·bitrev2048_m24+11416(SB)/8, $1587
DATA ·bitrev2048_m24+11424(SB)/8, $179
DATA ·bitrev2048_m24+11432(SB)/8, $691
DATA ·bitrev2048_m24+11440(SB)/8, $1203
DATA ·bitrev2048_m24+11448(SB)/8, $1715
DATA ·bitrev2048_m24+11456(SB)/8, $307
DATA ·bitrev2048_m24+11464(SB)/8, $819
DATA ·bitrev2048_m24+11472(SB)/8, $1331
DATA ·bitrev2048_m24+11480(SB)/8, $1843
DATA ·bitrev2048_m24+11488(SB)/8, $435
DATA ·bitrev2048_m24+11496(SB)/8, $947
DATA ·bitrev2048_m24+11504(SB)/8, $1459
DATA ·bitrev2048_m24+11512(SB)/8, $1971
DATA ·bitrev2048_m24+11520(SB)/8, $83
DATA ·bitrev2048_m24+11528(SB)/8, $595
DATA ·bitrev2048_m24+11536(SB)/8, $1107
DATA ·bitrev2048_m24+11544(SB)/8, $1619
DATA ·bitrev2048_m24+11552(SB)/8, $211
DATA ·bitrev2048_m24+11560(SB)/8, $723
DATA ·bitrev2048_m24+11568(SB)/8, $1235
DATA ·bitrev2048_m24+11576(SB)/8, $1747
DATA ·bitrev2048_m24+11584(SB)/8, $339
DATA ·bitrev2048_m24+11592(SB)/8, $851
DATA ·bitrev2048_m24+11600(SB)/8, $1363
DATA ·bitrev2048_m24+11608(SB)/8, $1875
DATA ·bitrev2048_m24+11616(SB)/8, $467
DATA ·bitrev2048_m24+11624(SB)/8, $979
DATA ·bitrev2048_m24+11632(SB)/8, $1491
DATA ·bitrev2048_m24+11640(SB)/8, $2003
DATA ·bitrev2048_m24+11648(SB)/8, $115
DATA ·bitrev2048_m24+11656(SB)/8, $627
DATA ·bitrev2048_m24+11664(SB)/8, $1139
DATA ·bitrev2048_m24+11672(SB)/8, $1651
DATA ·bitrev2048_m24+11680(SB)/8, $243
DATA ·bitrev2048_m24+11688(SB)/8, $755
DATA ·bitrev2048_m24+11696(SB)/8, $1267
DATA ·bitrev2048_m24+11704(SB)/8, $1779
DATA ·bitrev2048_m24+11712(SB)/8, $371
DATA ·bitrev2048_m24+11720(SB)/8, $883
DATA ·bitrev2048_m24+11728(SB)/8, $1395
DATA ·bitrev2048_m24+11736(SB)/8, $1907
DATA ·bitrev2048_m24+11744(SB)/8, $499
DATA ·bitrev2048_m24+11752(SB)/8, $1011
DATA ·bitrev2048_m24+11760(SB)/8, $1523
DATA ·bitrev2048_m24+11768(SB)/8, $2035
DATA ·bitrev2048_m24+11776(SB)/8, $27
DATA ·bitrev2048_m24+11784(SB)/8, $539
DATA ·bitrev2048_m24+11792(SB)/8, $1051
DATA ·bitrev2048_m24+11800(SB)/8, $1563
DATA ·bitrev2048_m24+11808(SB)/8, $155
DATA ·bitrev2048_m24+11816(SB)/8, $667
DATA ·bitrev2048_m24+11824(SB)/8, $1179
DATA ·bitrev2048_m24+11832(SB)/8, $1691
DATA ·bitrev2048_m24+11840(SB)/8, $283
DATA ·bitrev2048_m24+11848(SB)/8, $795
DATA ·bitrev2048_m24+11856(SB)/8, $1307
DATA ·bitrev2048_m24+11864(SB)/8, $1819
DATA ·bitrev2048_m24+11872(SB)/8, $411
DATA ·bitrev2048_m24+11880(SB)/8, $923
DATA ·bitrev2048_m24+11888(SB)/8, $1435
DATA ·bitrev2048_m24+11896(SB)/8, $1947
DATA ·bitrev2048_m24+11904(SB)/8, $59
DATA ·bitrev2048_m24+11912(SB)/8, $571
DATA ·bitrev2048_m24+11920(SB)/8, $1083
DATA ·bitrev2048_m24+11928(SB)/8, $1595
DATA ·bitrev2048_m24+11936(SB)/8, $187
DATA ·bitrev2048_m24+11944(SB)/8, $699
DATA ·bitrev2048_m24+11952(SB)/8, $1211
DATA ·bitrev2048_m24+11960(SB)/8, $1723
DATA ·bitrev2048_m24+11968(SB)/8, $315
DATA ·bitrev2048_m24+11976(SB)/8, $827
DATA ·bitrev2048_m24+11984(SB)/8, $1339
DATA ·bitrev2048_m24+11992(SB)/8, $1851
DATA ·bitrev2048_m24+12000(SB)/8, $443
DATA ·bitrev2048_m24+12008(SB)/8, $955
DATA ·bitrev2048_m24+12016(SB)/8, $1467
DATA ·bitrev2048_m24+12024(SB)/8, $1979
DATA ·bitrev2048_m24+12032(SB)/8, $91
DATA ·bitrev2048_m24+12040(SB)/8, $603
DATA ·bitrev2048_m24+12048(SB)/8, $1115
DATA ·bitrev2048_m24+12056(SB)/8, $1627
DATA ·bitrev2048_m24+12064(SB)/8, $219
DATA ·bitrev2048_m24+12072(SB)/8, $731
DATA ·bitrev2048_m24+12080(SB)/8, $1243
DATA ·bitrev2048_m24+12088(SB)/8, $1755
DATA ·bitrev2048_m24+12096(SB)/8, $347
DATA ·bitrev2048_m24+12104(SB)/8, $859
DATA ·bitrev2048_m24+12112(SB)/8, $1371
DATA ·bitrev2048_m24+12120(SB)/8, $1883
DATA ·bitrev2048_m24+12128(SB)/8, $475
DATA ·bitrev2048_m24+12136(SB)/8, $987
DATA ·bitrev2048_m24+12144(SB)/8, $1499
DATA ·bitrev2048_m24+12152(SB)/8, $2011
DATA ·bitrev2048_m24+12160(SB)/8, $123
DATA ·bitrev2048_m24+12168(SB)/8, $635
DATA ·bitrev2048_m24+12176(SB)/8, $1147
DATA ·bitrev2048_m24+12184(SB)/8, $1659
DATA ·bitrev2048_m24+12192(SB)/8, $251
DATA ·bitrev2048_m24+12200(SB)/8, $763
DATA ·bitrev2048_m24+12208(SB)/8, $1275
DATA ·bitrev2048_m24+12216(SB)/8, $1787
DATA ·bitrev2048_m24+12224(SB)/8, $379
DATA ·bitrev2048_m24+12232(SB)/8, $891
DATA ·bitrev2048_m24+12240(SB)/8, $1403
DATA ·bitrev2048_m24+12248(SB)/8, $1915
DATA ·bitrev2048_m24+12256(SB)/8, $507
DATA ·bitrev2048_m24+12264(SB)/8, $1019
DATA ·bitrev2048_m24+12272(SB)/8, $1531
DATA ·bitrev2048_m24+12280(SB)/8, $2043
DATA ·bitrev2048_m24+12288(SB)/8, $5
DATA ·bitrev2048_m24+12296(SB)/8, $517
DATA ·bitrev2048_m24+12304(SB)/8, $1029
DATA ·bitrev2048_m24+12312(SB)/8, $1541
DATA ·bitrev2048_m24+12320(SB)/8, $133
DATA ·bitrev2048_m24+12328(SB)/8, $645
DATA ·bitrev2048_m24+12336(SB)/8, $1157
DATA ·bitrev2048_m24+12344(SB)/8, $1669
DATA ·bitrev2048_m24+12352(SB)/8, $261
DATA ·bitrev2048_m24+12360(SB)/8, $773
DATA ·bitrev2048_m24+12368(SB)/8, $1285
DATA ·bitrev2048_m24+12376(SB)/8, $1797
DATA ·bitrev2048_m24+12384(SB)/8, $389
DATA ·bitrev2048_m24+12392(SB)/8, $901
DATA ·bitrev2048_m24+12400(SB)/8, $1413
DATA ·bitrev2048_m24+12408(SB)/8, $1925
DATA ·bitrev2048_m24+12416(SB)/8, $37
DATA ·bitrev2048_m24+12424(SB)/8, $549
DATA ·bitrev2048_m24+12432(SB)/8, $1061
DATA ·bitrev2048_m24+12440(SB)/8, $1573
DATA ·bitrev2048_m24+12448(SB)/8, $165
DATA ·bitrev2048_m24+12456(SB)/8, $677
DATA ·bitrev2048_m24+12464(SB)/8, $1189
DATA ·bitrev2048_m24+12472(SB)/8, $1701
DATA ·bitrev2048_m24+12480(SB)/8, $293
DATA ·bitrev2048_m24+12488(SB)/8, $805
DATA ·bitrev2048_m24+12496(SB)/8, $1317
DATA ·bitrev2048_m24+12504(SB)/8, $1829
DATA ·bitrev2048_m24+12512(SB)/8, $421
DATA ·bitrev2048_m24+12520(SB)/8, $933
DATA ·bitrev2048_m24+12528(SB)/8, $1445
DATA ·bitrev2048_m24+12536(SB)/8, $1957
DATA ·bitrev2048_m24+12544(SB)/8, $69
DATA ·bitrev2048_m24+12552(SB)/8, $581
DATA ·bitrev2048_m24+12560(SB)/8, $1093
DATA ·bitrev2048_m24+12568(SB)/8, $1605
DATA ·bitrev2048_m24+12576(SB)/8, $197
DATA ·bitrev2048_m24+12584(SB)/8, $709
DATA ·bitrev2048_m24+12592(SB)/8, $1221
DATA ·bitrev2048_m24+12600(SB)/8, $1733
DATA ·bitrev2048_m24+12608(SB)/8, $325
DATA ·bitrev2048_m24+12616(SB)/8, $837
DATA ·bitrev2048_m24+12624(SB)/8, $1349
DATA ·bitrev2048_m24+12632(SB)/8, $1861
DATA ·bitrev2048_m24+12640(SB)/8, $453
DATA ·bitrev2048_m24+12648(SB)/8, $965
DATA ·bitrev2048_m24+12656(SB)/8, $1477
DATA ·bitrev2048_m24+12664(SB)/8, $1989
DATA ·bitrev2048_m24+12672(SB)/8, $101
DATA ·bitrev2048_m24+12680(SB)/8, $613
DATA ·bitrev2048_m24+12688(SB)/8, $1125
DATA ·bitrev2048_m24+12696(SB)/8, $1637
DATA ·bitrev2048_m24+12704(SB)/8, $229
DATA ·bitrev2048_m24+12712(SB)/8, $741
DATA ·bitrev2048_m24+12720(SB)/8, $1253
DATA ·bitrev2048_m24+12728(SB)/8, $1765
DATA ·bitrev2048_m24+12736(SB)/8, $357
DATA ·bitrev2048_m24+12744(SB)/8, $869
DATA ·bitrev2048_m24+12752(SB)/8, $1381
DATA ·bitrev2048_m24+12760(SB)/8, $1893
DATA ·bitrev2048_m24+12768(SB)/8, $485
DATA ·bitrev2048_m24+12776(SB)/8, $997
DATA ·bitrev2048_m24+12784(SB)/8, $1509
DATA ·bitrev2048_m24+12792(SB)/8, $2021
DATA ·bitrev2048_m24+12800(SB)/8, $13
DATA ·bitrev2048_m24+12808(SB)/8, $525
DATA ·bitrev2048_m24+12816(SB)/8, $1037
DATA ·bitrev2048_m24+12824(SB)/8, $1549
DATA ·bitrev2048_m24+12832(SB)/8, $141
DATA ·bitrev2048_m24+12840(SB)/8, $653
DATA ·bitrev2048_m24+12848(SB)/8, $1165
DATA ·bitrev2048_m24+12856(SB)/8, $1677
DATA ·bitrev2048_m24+12864(SB)/8, $269
DATA ·bitrev2048_m24+12872(SB)/8, $781
DATA ·bitrev2048_m24+12880(SB)/8, $1293
DATA ·bitrev2048_m24+12888(SB)/8, $1805
DATA ·bitrev2048_m24+12896(SB)/8, $397
DATA ·bitrev2048_m24+12904(SB)/8, $909
DATA ·bitrev2048_m24+12912(SB)/8, $1421
DATA ·bitrev2048_m24+12920(SB)/8, $1933
DATA ·bitrev2048_m24+12928(SB)/8, $45
DATA ·bitrev2048_m24+12936(SB)/8, $557
DATA ·bitrev2048_m24+12944(SB)/8, $1069
DATA ·bitrev2048_m24+12952(SB)/8, $1581
DATA ·bitrev2048_m24+12960(SB)/8, $173
DATA ·bitrev2048_m24+12968(SB)/8, $685
DATA ·bitrev2048_m24+12976(SB)/8, $1197
DATA ·bitrev2048_m24+12984(SB)/8, $1709
DATA ·bitrev2048_m24+12992(SB)/8, $301
DATA ·bitrev2048_m24+13000(SB)/8, $813
DATA ·bitrev2048_m24+13008(SB)/8, $1325
DATA ·bitrev2048_m24+13016(SB)/8, $1837
DATA ·bitrev2048_m24+13024(SB)/8, $429
DATA ·bitrev2048_m24+13032(SB)/8, $941
DATA ·bitrev2048_m24+13040(SB)/8, $1453
DATA ·bitrev2048_m24+13048(SB)/8, $1965
DATA ·bitrev2048_m24+13056(SB)/8, $77
DATA ·bitrev2048_m24+13064(SB)/8, $589
DATA ·bitrev2048_m24+13072(SB)/8, $1101
DATA ·bitrev2048_m24+13080(SB)/8, $1613
DATA ·bitrev2048_m24+13088(SB)/8, $205
DATA ·bitrev2048_m24+13096(SB)/8, $717
DATA ·bitrev2048_m24+13104(SB)/8, $1229
DATA ·bitrev2048_m24+13112(SB)/8, $1741
DATA ·bitrev2048_m24+13120(SB)/8, $333
DATA ·bitrev2048_m24+13128(SB)/8, $845
DATA ·bitrev2048_m24+13136(SB)/8, $1357
DATA ·bitrev2048_m24+13144(SB)/8, $1869
DATA ·bitrev2048_m24+13152(SB)/8, $461
DATA ·bitrev2048_m24+13160(SB)/8, $973
DATA ·bitrev2048_m24+13168(SB)/8, $1485
DATA ·bitrev2048_m24+13176(SB)/8, $1997
DATA ·bitrev2048_m24+13184(SB)/8, $109
DATA ·bitrev2048_m24+13192(SB)/8, $621
DATA ·bitrev2048_m24+13200(SB)/8, $1133
DATA ·bitrev2048_m24+13208(SB)/8, $1645
DATA ·bitrev2048_m24+13216(SB)/8, $237
DATA ·bitrev2048_m24+13224(SB)/8, $749
DATA ·bitrev2048_m24+13232(SB)/8, $1261
DATA ·bitrev2048_m24+13240(SB)/8, $1773
DATA ·bitrev2048_m24+13248(SB)/8, $365
DATA ·bitrev2048_m24+13256(SB)/8, $877
DATA ·bitrev2048_m24+13264(SB)/8, $1389
DATA ·bitrev2048_m24+13272(SB)/8, $1901
DATA ·bitrev2048_m24+13280(SB)/8, $493
DATA ·bitrev2048_m24+13288(SB)/8, $1005
DATA ·bitrev2048_m24+13296(SB)/8, $1517
DATA ·bitrev2048_m24+13304(SB)/8, $2029
DATA ·bitrev2048_m24+13312(SB)/8, $21
DATA ·bitrev2048_m24+13320(SB)/8, $533
DATA ·bitrev2048_m24+13328(SB)/8, $1045
DATA ·bitrev2048_m24+13336(SB)/8, $1557
DATA ·bitrev2048_m24+13344(SB)/8, $149
DATA ·bitrev2048_m24+13352(SB)/8, $661
DATA ·bitrev2048_m24+13360(SB)/8, $1173
DATA ·bitrev2048_m24+13368(SB)/8, $1685
DATA ·bitrev2048_m24+13376(SB)/8, $277
DATA ·bitrev2048_m24+13384(SB)/8, $789
DATA ·bitrev2048_m24+13392(SB)/8, $1301
DATA ·bitrev2048_m24+13400(SB)/8, $1813
DATA ·bitrev2048_m24+13408(SB)/8, $405
DATA ·bitrev2048_m24+13416(SB)/8, $917
DATA ·bitrev2048_m24+13424(SB)/8, $1429
DATA ·bitrev2048_m24+13432(SB)/8, $1941
DATA ·bitrev2048_m24+13440(SB)/8, $53
DATA ·bitrev2048_m24+13448(SB)/8, $565
DATA ·bitrev2048_m24+13456(SB)/8, $1077
DATA ·bitrev2048_m24+13464(SB)/8, $1589
DATA ·bitrev2048_m24+13472(SB)/8, $181
DATA ·bitrev2048_m24+13480(SB)/8, $693
DATA ·bitrev2048_m24+13488(SB)/8, $1205
DATA ·bitrev2048_m24+13496(SB)/8, $1717
DATA ·bitrev2048_m24+13504(SB)/8, $309
DATA ·bitrev2048_m24+13512(SB)/8, $821
DATA ·bitrev2048_m24+13520(SB)/8, $1333
DATA ·bitrev2048_m24+13528(SB)/8, $1845
DATA ·bitrev2048_m24+13536(SB)/8, $437
DATA ·bitrev2048_m24+13544(SB)/8, $949
DATA ·bitrev2048_m24+13552(SB)/8, $1461
DATA ·bitrev2048_m24+13560(SB)/8, $1973
DATA ·bitrev2048_m24+13568(SB)/8, $85
DATA ·bitrev2048_m24+13576(SB)/8, $597
DATA ·bitrev2048_m24+13584(SB)/8, $1109
DATA ·bitrev2048_m24+13592(SB)/8, $1621
DATA ·bitrev2048_m24+13600(SB)/8, $213
DATA ·bitrev2048_m24+13608(SB)/8, $725
DATA ·bitrev2048_m24+13616(SB)/8, $1237
DATA ·bitrev2048_m24+13624(SB)/8, $1749
DATA ·bitrev2048_m24+13632(SB)/8, $341
DATA ·bitrev2048_m24+13640(SB)/8, $853
DATA ·bitrev2048_m24+13648(SB)/8, $1365
DATA ·bitrev2048_m24+13656(SB)/8, $1877
DATA ·bitrev2048_m24+13664(SB)/8, $469
DATA ·bitrev2048_m24+13672(SB)/8, $981
DATA ·bitrev2048_m24+13680(SB)/8, $1493
DATA ·bitrev2048_m24+13688(SB)/8, $2005
DATA ·bitrev2048_m24+13696(SB)/8, $117
DATA ·bitrev2048_m24+13704(SB)/8, $629
DATA ·bitrev2048_m24+13712(SB)/8, $1141
DATA ·bitrev2048_m24+13720(SB)/8, $1653
DATA ·bitrev2048_m24+13728(SB)/8, $245
DATA ·bitrev2048_m24+13736(SB)/8, $757
DATA ·bitrev2048_m24+13744(SB)/8, $1269
DATA ·bitrev2048_m24+13752(SB)/8, $1781
DATA ·bitrev2048_m24+13760(SB)/8, $373
DATA ·bitrev2048_m24+13768(SB)/8, $885
DATA ·bitrev2048_m24+13776(SB)/8, $1397
DATA ·bitrev2048_m24+13784(SB)/8, $1909
DATA ·bitrev2048_m24+13792(SB)/8, $501
DATA ·bitrev2048_m24+13800(SB)/8, $1013
DATA ·bitrev2048_m24+13808(SB)/8, $1525
DATA ·bitrev2048_m24+13816(SB)/8, $2037
DATA ·bitrev2048_m24+13824(SB)/8, $29
DATA ·bitrev2048_m24+13832(SB)/8, $541
DATA ·bitrev2048_m24+13840(SB)/8, $1053
DATA ·bitrev2048_m24+13848(SB)/8, $1565
DATA ·bitrev2048_m24+13856(SB)/8, $157
DATA ·bitrev2048_m24+13864(SB)/8, $669
DATA ·bitrev2048_m24+13872(SB)/8, $1181
DATA ·bitrev2048_m24+13880(SB)/8, $1693
DATA ·bitrev2048_m24+13888(SB)/8, $285
DATA ·bitrev2048_m24+13896(SB)/8, $797
DATA ·bitrev2048_m24+13904(SB)/8, $1309
DATA ·bitrev2048_m24+13912(SB)/8, $1821
DATA ·bitrev2048_m24+13920(SB)/8, $413
DATA ·bitrev2048_m24+13928(SB)/8, $925
DATA ·bitrev2048_m24+13936(SB)/8, $1437
DATA ·bitrev2048_m24+13944(SB)/8, $1949
DATA ·bitrev2048_m24+13952(SB)/8, $61
DATA ·bitrev2048_m24+13960(SB)/8, $573
DATA ·bitrev2048_m24+13968(SB)/8, $1085
DATA ·bitrev2048_m24+13976(SB)/8, $1597
DATA ·bitrev2048_m24+13984(SB)/8, $189
DATA ·bitrev2048_m24+13992(SB)/8, $701
DATA ·bitrev2048_m24+14000(SB)/8, $1213
DATA ·bitrev2048_m24+14008(SB)/8, $1725
DATA ·bitrev2048_m24+14016(SB)/8, $317
DATA ·bitrev2048_m24+14024(SB)/8, $829
DATA ·bitrev2048_m24+14032(SB)/8, $1341
DATA ·bitrev2048_m24+14040(SB)/8, $1853
DATA ·bitrev2048_m24+14048(SB)/8, $445
DATA ·bitrev2048_m24+14056(SB)/8, $957
DATA ·bitrev2048_m24+14064(SB)/8, $1469
DATA ·bitrev2048_m24+14072(SB)/8, $1981
DATA ·bitrev2048_m24+14080(SB)/8, $93
DATA ·bitrev2048_m24+14088(SB)/8, $605
DATA ·bitrev2048_m24+14096(SB)/8, $1117
DATA ·bitrev2048_m24+14104(SB)/8, $1629
DATA ·bitrev2048_m24+14112(SB)/8, $221
DATA ·bitrev2048_m24+14120(SB)/8, $733
DATA ·bitrev2048_m24+14128(SB)/8, $1245
DATA ·bitrev2048_m24+14136(SB)/8, $1757
DATA ·bitrev2048_m24+14144(SB)/8, $349
DATA ·bitrev2048_m24+14152(SB)/8, $861
DATA ·bitrev2048_m24+14160(SB)/8, $1373
DATA ·bitrev2048_m24+14168(SB)/8, $1885
DATA ·bitrev2048_m24+14176(SB)/8, $477
DATA ·bitrev2048_m24+14184(SB)/8, $989
DATA ·bitrev2048_m24+14192(SB)/8, $1501
DATA ·bitrev2048_m24+14200(SB)/8, $2013
DATA ·bitrev2048_m24+14208(SB)/8, $125
DATA ·bitrev2048_m24+14216(SB)/8, $637
DATA ·bitrev2048_m24+14224(SB)/8, $1149
DATA ·bitrev2048_m24+14232(SB)/8, $1661
DATA ·bitrev2048_m24+14240(SB)/8, $253
DATA ·bitrev2048_m24+14248(SB)/8, $765
DATA ·bitrev2048_m24+14256(SB)/8, $1277
DATA ·bitrev2048_m24+14264(SB)/8, $1789
DATA ·bitrev2048_m24+14272(SB)/8, $381
DATA ·bitrev2048_m24+14280(SB)/8, $893
DATA ·bitrev2048_m24+14288(SB)/8, $1405
DATA ·bitrev2048_m24+14296(SB)/8, $1917
DATA ·bitrev2048_m24+14304(SB)/8, $509
DATA ·bitrev2048_m24+14312(SB)/8, $1021
DATA ·bitrev2048_m24+14320(SB)/8, $1533
DATA ·bitrev2048_m24+14328(SB)/8, $2045
DATA ·bitrev2048_m24+14336(SB)/8, $7
DATA ·bitrev2048_m24+14344(SB)/8, $519
DATA ·bitrev2048_m24+14352(SB)/8, $1031
DATA ·bitrev2048_m24+14360(SB)/8, $1543
DATA ·bitrev2048_m24+14368(SB)/8, $135
DATA ·bitrev2048_m24+14376(SB)/8, $647
DATA ·bitrev2048_m24+14384(SB)/8, $1159
DATA ·bitrev2048_m24+14392(SB)/8, $1671
DATA ·bitrev2048_m24+14400(SB)/8, $263
DATA ·bitrev2048_m24+14408(SB)/8, $775
DATA ·bitrev2048_m24+14416(SB)/8, $1287
DATA ·bitrev2048_m24+14424(SB)/8, $1799
DATA ·bitrev2048_m24+14432(SB)/8, $391
DATA ·bitrev2048_m24+14440(SB)/8, $903
DATA ·bitrev2048_m24+14448(SB)/8, $1415
DATA ·bitrev2048_m24+14456(SB)/8, $1927
DATA ·bitrev2048_m24+14464(SB)/8, $39
DATA ·bitrev2048_m24+14472(SB)/8, $551
DATA ·bitrev2048_m24+14480(SB)/8, $1063
DATA ·bitrev2048_m24+14488(SB)/8, $1575
DATA ·bitrev2048_m24+14496(SB)/8, $167
DATA ·bitrev2048_m24+14504(SB)/8, $679
DATA ·bitrev2048_m24+14512(SB)/8, $1191
DATA ·bitrev2048_m24+14520(SB)/8, $1703
DATA ·bitrev2048_m24+14528(SB)/8, $295
DATA ·bitrev2048_m24+14536(SB)/8, $807
DATA ·bitrev2048_m24+14544(SB)/8, $1319
DATA ·bitrev2048_m24+14552(SB)/8, $1831
DATA ·bitrev2048_m24+14560(SB)/8, $423
DATA ·bitrev2048_m24+14568(SB)/8, $935
DATA ·bitrev2048_m24+14576(SB)/8, $1447
DATA ·bitrev2048_m24+14584(SB)/8, $1959
DATA ·bitrev2048_m24+14592(SB)/8, $71
DATA ·bitrev2048_m24+14600(SB)/8, $583
DATA ·bitrev2048_m24+14608(SB)/8, $1095
DATA ·bitrev2048_m24+14616(SB)/8, $1607
DATA ·bitrev2048_m24+14624(SB)/8, $199
DATA ·bitrev2048_m24+14632(SB)/8, $711
DATA ·bitrev2048_m24+14640(SB)/8, $1223
DATA ·bitrev2048_m24+14648(SB)/8, $1735
DATA ·bitrev2048_m24+14656(SB)/8, $327
DATA ·bitrev2048_m24+14664(SB)/8, $839
DATA ·bitrev2048_m24+14672(SB)/8, $1351
DATA ·bitrev2048_m24+14680(SB)/8, $1863
DATA ·bitrev2048_m24+14688(SB)/8, $455
DATA ·bitrev2048_m24+14696(SB)/8, $967
DATA ·bitrev2048_m24+14704(SB)/8, $1479
DATA ·bitrev2048_m24+14712(SB)/8, $1991
DATA ·bitrev2048_m24+14720(SB)/8, $103
DATA ·bitrev2048_m24+14728(SB)/8, $615
DATA ·bitrev2048_m24+14736(SB)/8, $1127
DATA ·bitrev2048_m24+14744(SB)/8, $1639
DATA ·bitrev2048_m24+14752(SB)/8, $231
DATA ·bitrev2048_m24+14760(SB)/8, $743
DATA ·bitrev2048_m24+14768(SB)/8, $1255
DATA ·bitrev2048_m24+14776(SB)/8, $1767
DATA ·bitrev2048_m24+14784(SB)/8, $359
DATA ·bitrev2048_m24+14792(SB)/8, $871
DATA ·bitrev2048_m24+14800(SB)/8, $1383
DATA ·bitrev2048_m24+14808(SB)/8, $1895
DATA ·bitrev2048_m24+14816(SB)/8, $487
DATA ·bitrev2048_m24+14824(SB)/8, $999
DATA ·bitrev2048_m24+14832(SB)/8, $1511
DATA ·bitrev2048_m24+14840(SB)/8, $2023
DATA ·bitrev2048_m24+14848(SB)/8, $15
DATA ·bitrev2048_m24+14856(SB)/8, $527
DATA ·bitrev2048_m24+14864(SB)/8, $1039
DATA ·bitrev2048_m24+14872(SB)/8, $1551
DATA ·bitrev2048_m24+14880(SB)/8, $143
DATA ·bitrev2048_m24+14888(SB)/8, $655
DATA ·bitrev2048_m24+14896(SB)/8, $1167
DATA ·bitrev2048_m24+14904(SB)/8, $1679
DATA ·bitrev2048_m24+14912(SB)/8, $271
DATA ·bitrev2048_m24+14920(SB)/8, $783
DATA ·bitrev2048_m24+14928(SB)/8, $1295
DATA ·bitrev2048_m24+14936(SB)/8, $1807
DATA ·bitrev2048_m24+14944(SB)/8, $399
DATA ·bitrev2048_m24+14952(SB)/8, $911
DATA ·bitrev2048_m24+14960(SB)/8, $1423
DATA ·bitrev2048_m24+14968(SB)/8, $1935
DATA ·bitrev2048_m24+14976(SB)/8, $47
DATA ·bitrev2048_m24+14984(SB)/8, $559
DATA ·bitrev2048_m24+14992(SB)/8, $1071
DATA ·bitrev2048_m24+15000(SB)/8, $1583
DATA ·bitrev2048_m24+15008(SB)/8, $175
DATA ·bitrev2048_m24+15016(SB)/8, $687
DATA ·bitrev2048_m24+15024(SB)/8, $1199
DATA ·bitrev2048_m24+15032(SB)/8, $1711
DATA ·bitrev2048_m24+15040(SB)/8, $303
DATA ·bitrev2048_m24+15048(SB)/8, $815
DATA ·bitrev2048_m24+15056(SB)/8, $1327
DATA ·bitrev2048_m24+15064(SB)/8, $1839
DATA ·bitrev2048_m24+15072(SB)/8, $431
DATA ·bitrev2048_m24+15080(SB)/8, $943
DATA ·bitrev2048_m24+15088(SB)/8, $1455
DATA ·bitrev2048_m24+15096(SB)/8, $1967
DATA ·bitrev2048_m24+15104(SB)/8, $79
DATA ·bitrev2048_m24+15112(SB)/8, $591
DATA ·bitrev2048_m24+15120(SB)/8, $1103
DATA ·bitrev2048_m24+15128(SB)/8, $1615
DATA ·bitrev2048_m24+15136(SB)/8, $207
DATA ·bitrev2048_m24+15144(SB)/8, $719
DATA ·bitrev2048_m24+15152(SB)/8, $1231
DATA ·bitrev2048_m24+15160(SB)/8, $1743
DATA ·bitrev2048_m24+15168(SB)/8, $335
DATA ·bitrev2048_m24+15176(SB)/8, $847
DATA ·bitrev2048_m24+15184(SB)/8, $1359
DATA ·bitrev2048_m24+15192(SB)/8, $1871
DATA ·bitrev2048_m24+15200(SB)/8, $463
DATA ·bitrev2048_m24+15208(SB)/8, $975
DATA ·bitrev2048_m24+15216(SB)/8, $1487
DATA ·bitrev2048_m24+15224(SB)/8, $1999
DATA ·bitrev2048_m24+15232(SB)/8, $111
DATA ·bitrev2048_m24+15240(SB)/8, $623
DATA ·bitrev2048_m24+15248(SB)/8, $1135
DATA ·bitrev2048_m24+15256(SB)/8, $1647
DATA ·bitrev2048_m24+15264(SB)/8, $239
DATA ·bitrev2048_m24+15272(SB)/8, $751
DATA ·bitrev2048_m24+15280(SB)/8, $1263
DATA ·bitrev2048_m24+15288(SB)/8, $1775
DATA ·bitrev2048_m24+15296(SB)/8, $367
DATA ·bitrev2048_m24+15304(SB)/8, $879
DATA ·bitrev2048_m24+15312(SB)/8, $1391
DATA ·bitrev2048_m24+15320(SB)/8, $1903
DATA ·bitrev2048_m24+15328(SB)/8, $495
DATA ·bitrev2048_m24+15336(SB)/8, $1007
DATA ·bitrev2048_m24+15344(SB)/8, $1519
DATA ·bitrev2048_m24+15352(SB)/8, $2031
DATA ·bitrev2048_m24+15360(SB)/8, $23
DATA ·bitrev2048_m24+15368(SB)/8, $535
DATA ·bitrev2048_m24+15376(SB)/8, $1047
DATA ·bitrev2048_m24+15384(SB)/8, $1559
DATA ·bitrev2048_m24+15392(SB)/8, $151
DATA ·bitrev2048_m24+15400(SB)/8, $663
DATA ·bitrev2048_m24+15408(SB)/8, $1175
DATA ·bitrev2048_m24+15416(SB)/8, $1687
DATA ·bitrev2048_m24+15424(SB)/8, $279
DATA ·bitrev2048_m24+15432(SB)/8, $791
DATA ·bitrev2048_m24+15440(SB)/8, $1303
DATA ·bitrev2048_m24+15448(SB)/8, $1815
DATA ·bitrev2048_m24+15456(SB)/8, $407
DATA ·bitrev2048_m24+15464(SB)/8, $919
DATA ·bitrev2048_m24+15472(SB)/8, $1431
DATA ·bitrev2048_m24+15480(SB)/8, $1943
DATA ·bitrev2048_m24+15488(SB)/8, $55
DATA ·bitrev2048_m24+15496(SB)/8, $567
DATA ·bitrev2048_m24+15504(SB)/8, $1079
DATA ·bitrev2048_m24+15512(SB)/8, $1591
DATA ·bitrev2048_m24+15520(SB)/8, $183
DATA ·bitrev2048_m24+15528(SB)/8, $695
DATA ·bitrev2048_m24+15536(SB)/8, $1207
DATA ·bitrev2048_m24+15544(SB)/8, $1719
DATA ·bitrev2048_m24+15552(SB)/8, $311
DATA ·bitrev2048_m24+15560(SB)/8, $823
DATA ·bitrev2048_m24+15568(SB)/8, $1335
DATA ·bitrev2048_m24+15576(SB)/8, $1847
DATA ·bitrev2048_m24+15584(SB)/8, $439
DATA ·bitrev2048_m24+15592(SB)/8, $951
DATA ·bitrev2048_m24+15600(SB)/8, $1463
DATA ·bitrev2048_m24+15608(SB)/8, $1975
DATA ·bitrev2048_m24+15616(SB)/8, $87
DATA ·bitrev2048_m24+15624(SB)/8, $599
DATA ·bitrev2048_m24+15632(SB)/8, $1111
DATA ·bitrev2048_m24+15640(SB)/8, $1623
DATA ·bitrev2048_m24+15648(SB)/8, $215
DATA ·bitrev2048_m24+15656(SB)/8, $727
DATA ·bitrev2048_m24+15664(SB)/8, $1239
DATA ·bitrev2048_m24+15672(SB)/8, $1751
DATA ·bitrev2048_m24+15680(SB)/8, $343
DATA ·bitrev2048_m24+15688(SB)/8, $855
DATA ·bitrev2048_m24+15696(SB)/8, $1367
DATA ·bitrev2048_m24+15704(SB)/8, $1879
DATA ·bitrev2048_m24+15712(SB)/8, $471
DATA ·bitrev2048_m24+15720(SB)/8, $983
DATA ·bitrev2048_m24+15728(SB)/8, $1495
DATA ·bitrev2048_m24+15736(SB)/8, $2007
DATA ·bitrev2048_m24+15744(SB)/8, $119
DATA ·bitrev2048_m24+15752(SB)/8, $631
DATA ·bitrev2048_m24+15760(SB)/8, $1143
DATA ·bitrev2048_m24+15768(SB)/8, $1655
DATA ·bitrev2048_m24+15776(SB)/8, $247
DATA ·bitrev2048_m24+15784(SB)/8, $759
DATA ·bitrev2048_m24+15792(SB)/8, $1271
DATA ·bitrev2048_m24+15800(SB)/8, $1783
DATA ·bitrev2048_m24+15808(SB)/8, $375
DATA ·bitrev2048_m24+15816(SB)/8, $887
DATA ·bitrev2048_m24+15824(SB)/8, $1399
DATA ·bitrev2048_m24+15832(SB)/8, $1911
DATA ·bitrev2048_m24+15840(SB)/8, $503
DATA ·bitrev2048_m24+15848(SB)/8, $1015
DATA ·bitrev2048_m24+15856(SB)/8, $1527
DATA ·bitrev2048_m24+15864(SB)/8, $2039
DATA ·bitrev2048_m24+15872(SB)/8, $31
DATA ·bitrev2048_m24+15880(SB)/8, $543
DATA ·bitrev2048_m24+15888(SB)/8, $1055
DATA ·bitrev2048_m24+15896(SB)/8, $1567
DATA ·bitrev2048_m24+15904(SB)/8, $159
DATA ·bitrev2048_m24+15912(SB)/8, $671
DATA ·bitrev2048_m24+15920(SB)/8, $1183
DATA ·bitrev2048_m24+15928(SB)/8, $1695
DATA ·bitrev2048_m24+15936(SB)/8, $287
DATA ·bitrev2048_m24+15944(SB)/8, $799
DATA ·bitrev2048_m24+15952(SB)/8, $1311
DATA ·bitrev2048_m24+15960(SB)/8, $1823
DATA ·bitrev2048_m24+15968(SB)/8, $415
DATA ·bitrev2048_m24+15976(SB)/8, $927
DATA ·bitrev2048_m24+15984(SB)/8, $1439
DATA ·bitrev2048_m24+15992(SB)/8, $1951
DATA ·bitrev2048_m24+16000(SB)/8, $63
DATA ·bitrev2048_m24+16008(SB)/8, $575
DATA ·bitrev2048_m24+16016(SB)/8, $1087
DATA ·bitrev2048_m24+16024(SB)/8, $1599
DATA ·bitrev2048_m24+16032(SB)/8, $191
DATA ·bitrev2048_m24+16040(SB)/8, $703
DATA ·bitrev2048_m24+16048(SB)/8, $1215
DATA ·bitrev2048_m24+16056(SB)/8, $1727
DATA ·bitrev2048_m24+16064(SB)/8, $319
DATA ·bitrev2048_m24+16072(SB)/8, $831
DATA ·bitrev2048_m24+16080(SB)/8, $1343
DATA ·bitrev2048_m24+16088(SB)/8, $1855
DATA ·bitrev2048_m24+16096(SB)/8, $447
DATA ·bitrev2048_m24+16104(SB)/8, $959
DATA ·bitrev2048_m24+16112(SB)/8, $1471
DATA ·bitrev2048_m24+16120(SB)/8, $1983
DATA ·bitrev2048_m24+16128(SB)/8, $95
DATA ·bitrev2048_m24+16136(SB)/8, $607
DATA ·bitrev2048_m24+16144(SB)/8, $1119
DATA ·bitrev2048_m24+16152(SB)/8, $1631
DATA ·bitrev2048_m24+16160(SB)/8, $223
DATA ·bitrev2048_m24+16168(SB)/8, $735
DATA ·bitrev2048_m24+16176(SB)/8, $1247
DATA ·bitrev2048_m24+16184(SB)/8, $1759
DATA ·bitrev2048_m24+16192(SB)/8, $351
DATA ·bitrev2048_m24+16200(SB)/8, $863
DATA ·bitrev2048_m24+16208(SB)/8, $1375
DATA ·bitrev2048_m24+16216(SB)/8, $1887
DATA ·bitrev2048_m24+16224(SB)/8, $479
DATA ·bitrev2048_m24+16232(SB)/8, $991
DATA ·bitrev2048_m24+16240(SB)/8, $1503
DATA ·bitrev2048_m24+16248(SB)/8, $2015
DATA ·bitrev2048_m24+16256(SB)/8, $127
DATA ·bitrev2048_m24+16264(SB)/8, $639
DATA ·bitrev2048_m24+16272(SB)/8, $1151
DATA ·bitrev2048_m24+16280(SB)/8, $1663
DATA ·bitrev2048_m24+16288(SB)/8, $255
DATA ·bitrev2048_m24+16296(SB)/8, $767
DATA ·bitrev2048_m24+16304(SB)/8, $1279
DATA ·bitrev2048_m24+16312(SB)/8, $1791
DATA ·bitrev2048_m24+16320(SB)/8, $383
DATA ·bitrev2048_m24+16328(SB)/8, $895
DATA ·bitrev2048_m24+16336(SB)/8, $1407
DATA ·bitrev2048_m24+16344(SB)/8, $1919
DATA ·bitrev2048_m24+16352(SB)/8, $511
DATA ·bitrev2048_m24+16360(SB)/8, $1023
DATA ·bitrev2048_m24+16368(SB)/8, $1535
DATA ·bitrev2048_m24+16376(SB)/8, $2047
GLOBL ·bitrev2048_m24(SB), RODATA, $16384

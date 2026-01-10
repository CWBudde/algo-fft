//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-256 Radix-4 FFT Kernel for AMD64
// ===========================================================================
//
// This file contains a radix-4 DIT FFT optimized for size 256 using AVX2.
// Unlike the radix-2 approach (8 stages), radix-4 uses only 4 stages.
//
// Algorithm: Radix-4 Decimation-in-Time (DIT) FFT
// Stages: 4 (log₄(256) = 4)
//
// Stage structure:
//   Stage 1: 64 butterflies, stride=4,   no twiddle multiply (W^0 = 1)
//   Stage 2: 16 groups × 4 butterflies, stride=16, twiddle step=16
//   Stage 3: 4 groups × 16 butterflies, stride=64, twiddle step=4
//   Stage 4: 1 group × 64 butterflies, stride=256, twiddle step=1
//
// Radix-4 Butterfly:
//   Input: a0, a1, a2, a3 (pre-multiplied by twiddles W^0, W^k, W^2k, W^3k)
//   t0 = a0 + a2
//   t1 = a0 - a2
//   t2 = a1 + a3
//   t3 = a1 - a3
//   y0 = t0 + t2
//   y2 = t0 - t2
//   y1 = t1 + (-i)*t3
//   y3 = t1 + i*t3
//
// Complex multiply by i:  (a+bi)*i = -b+ai
// Complex multiply by -i: (a+bi)*(-i) = b-ai
//
// ===========================================================================

#include "textflag.h"

TEXT ·ForwardAVX2Size256Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 256)

	// Verify n == 256
	CMPQ R13, $256
	JNE  r4_256_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $256
	JL   r4_256_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $256
	JL   r4_256_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $256
	JL   r4_256_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  r4_256_use_dst
	MOVQ R11, R8             // In-place: use scratch

r4_256_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 bit-reversal)
	// ==================================================================
	LEAQ ·bitrev256_r4(SB), R12
	XORQ CX, CX              // CX = i = 0

r4_256_bitrev_loop:
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i] ([]int = 8 bytes per element)
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[i]] (8 bytes = 1 complex64)
	MOVQ AX, (R8)(CX*8)      // work[i] = src[bitrev[i]]
	INCQ CX
	CMPQ CX, $256
	JL   r4_256_bitrev_loop

r4_256_stage1:
	// ==================================================================
	// Stage 1: 64 radix-4 butterflies, stride=4
	// All twiddle factors are 1 (W^0), so no multiplication needed
	// Process groups of 4 complex values at: 0-3, 4-7, 8-11, ..., 252-255
	// ==================================================================

	XORQ CX, CX              // CX = base offset in elements (not bytes)

r4_256_stage1_loop:
	CMPQ CX, $256
	JGE  r4_256_stage2

	// Load 4 complex64 values: work[base], work[base+1], work[base+2], work[base+3]
	// Each complex64 = 8 bytes, so 4 values = 32 bytes
	LEAQ (R8)(CX*8), SI      // SI = &work[base]
	VMOVSD (SI), X0          // X0 = a0 (work[base])
	VMOVSD 8(SI), X1         // X1 = a1 (work[base+1])
	VMOVSD 16(SI), X2        // X2 = a2 (work[base+2])
	VMOVSD 24(SI), X3        // X3 = a3 (work[base+3])

	// Radix-4 butterfly - compute all outputs before writing
	// t0 = a0 + a2
	VADDPS X0, X2, X4
	// t1 = a0 - a2
	VSUBPS X2, X0, X5
	// t2 = a1 + a3
	VADDPS X1, X3, X6
	// t3 = a1 - a3
	VSUBPS X3, X1, X7

	// Compute (-i)*t3 for y1
	// (-i)*(a+bi) = b-ai: swap to get [b,a], then negate second component
	VPERMILPS $0xB1, X7, X8  // X8 = [t3.i, t3.r] (swap real/imag)
	VXORPS X9, X9, X9
	VSUBPS X8, X9, X10       // X10 = -X8 = [-t3.i, -t3.r]
	VBLENDPS $0x02, X10, X8, X8  // X8 = [t3.i, -t3.r] = [b, -a] = (-i)*t3

	// Compute i*t3 for y3
	// i*(a+bi) = -b+ai: swap to get [b,a], then negate first component
	VPERMILPS $0xB1, X7, X11  // X11 = [t3.i, t3.r]
	VSUBPS X11, X9, X10       // X10 = -X11 = [-t3.i, -t3.r]
	VBLENDPS $0x01, X10, X11, X11  // X11 = [-t3.i, t3.r] = [-b, a] = i*t3

	// Now compute all 4 outputs
	// y0 = t0 + t2
	VADDPS X4, X6, X0
	// y1 = t1 + (-i)*t3
	VADDPS X5, X8, X1
	// y2 = t0 - t2
	VSUBPS X6, X4, X2
	// y3 = t1 + i*t3
	VADDPS X5, X11, X3

	// Write all outputs
	VMOVSD X0, (SI)
	VMOVSD X1, 8(SI)
	VMOVSD X2, 16(SI)
	VMOVSD X3, 24(SI)

	ADDQ $4, CX
	JMP  r4_256_stage1_loop

r4_256_stage2:
	// ==================================================================
	// Stage 2: 16 groups, each with 4 butterflies
	// Twiddle step = 16
	// Groups at base offsets: 0, 16, 32, ..., 240
	// Within each group, process j=0,1,2,3
	// ==================================================================

	XORQ CX, CX              // CX = base offset

r4_256_stage2_outer:
	CMPQ CX, $256
	JGE  r4_256_stage3

	// Process 4 butterflies in this group (j=0,1,2,3)
	XORQ DX, DX              // DX = j

r4_256_stage2_inner:
	CMPQ DX, $4
	JGE  r4_256_stage2_next

	// Calculate indices: idx0 = base+j, idx1 = base+j+4, idx2 = base+j+8, idx3 = base+j+12
	MOVQ CX, BX              // BX = base
	ADDQ DX, BX              // BX = idx0 = base + j
	LEAQ 4(BX), SI           // SI = idx1 = base + j + 4
	LEAQ 8(BX), DI           // DI = idx2 = base + j + 8
	LEAQ 12(BX), R14         // R14 = idx3 = base + j + 12

	// Load twiddle factors: w1 = twiddle[j*16], w2 = twiddle[2*j*16], w3 = twiddle[3*j*16]
	MOVQ DX, R15
	SHLQ $4, R15             // R15 = j*16
	VMOVSD (R10)(R15*8), X8  // X8 = w1

	MOVQ R15, R13            // R13 = j*16
	SHLQ $1, R15             // R15 = 2*j*16
	VMOVSD (R10)(R15*8), X9  // X9 = w2

	ADDQ R13, R15            // R15 = 2*j*16 + j*16 = 3*j*16
	VMOVSD (R10)(R15*8), X10 // X10 = w3

	// Load data
	VMOVSD (R8)(BX*8), X0    // X0 = a0 = work[idx0]
	VMOVSD (R8)(SI*8), X1    // X1 = work[idx1]
	VMOVSD (R8)(DI*8), X2    // X2 = work[idx2]
	VMOVSD (R8)(R14*8), X3   // X3 = work[idx3]

	// Complex multiply: a1 = a1 * w1
	// (a+bi)*(c+di) = (ac-bd)+(ad+bc)i
	// Use FMA: result = a*c + (shuffle(a)*d), with FMADDSUB handling ±
	VMOVSLDUP X8, X11        // X11 = [w1.r, w1.r]
	VMOVSHDUP X8, X12        // X12 = [w1.i, w1.i]
	VSHUFPS $0xB1, X1, X1, X13  // X13 = [a1.i, a1.r]
	VMULPS X12, X13, X13     // X13 = [a1.i*w1.i, a1.r*w1.i]
	VFMADDSUB231PS X11, X1, X13  // X13 = w1 * a1
	VMOVAPS X13, X1

	// Complex multiply: a2 = a2 * w2
	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X2, X13
	VMOVAPS X13, X2

	// Complex multiply: a3 = a3 * w3
	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X3, X13
	VMOVAPS X13, X3

	// Radix-4 butterfly - compute all outputs before writing
	VADDPS X0, X2, X4        // t0
	VSUBPS X2, X0, X5        // t1
	VADDPS X1, X3, X6        // t2
	VSUBPS X3, X1, X7        // t3

	// Compute (-i)*t3 for y1
	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14  // X14 = [t3.i, -t3.r] = (-i)*t3

	// Compute i*t3 for y3
	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x01, X11, X12, X12  // X12 = [-t3.i, t3.r] = i*t3

	// Compute all 4 outputs
	// y0 = t0 + t2
	VADDPS X4, X6, X0
	// y1 = t1 + (-i)*t3
	VADDPS X5, X14, X1
	// y2 = t0 - t2
	VSUBPS X6, X4, X2
	// y3 = t1 + i*t3
	VADDPS X5, X12, X3

	// Write all outputs
	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R14*8)

	INCQ DX
	JMP  r4_256_stage2_inner

r4_256_stage2_next:
	ADDQ $16, CX
	JMP  r4_256_stage2_outer

r4_256_stage3:
	// ==================================================================
	// Stage 3: 4 groups, each with 16 butterflies
	// Twiddle step = 4
	// Groups at base offsets: 0, 64, 128, 192
	// ==================================================================

	XORQ CX, CX              // CX = base offset

r4_256_stage3_outer:
	CMPQ CX, $256
	JGE  r4_256_stage4

	XORQ DX, DX              // DX = j

r4_256_stage3_inner:
	CMPQ DX, $16
	JGE  r4_256_stage3_next

	// Calculate indices
	MOVQ CX, BX
	ADDQ DX, BX              // BX = idx0 = base + j
	LEAQ 16(BX), SI          // SI = idx1 = base + j + 16
	LEAQ 32(BX), DI          // DI = idx2 = base + j + 32
	LEAQ 48(BX), R14         // R14 = idx3 = base + j + 48

	// Twiddle factors: twiddle[j*4], twiddle[2*j*4], twiddle[3*j*4]
	MOVQ DX, R15
	SHLQ $2, R15             // R15 = j*4
	VMOVSD (R10)(R15*8), X8  // X8 = w1

	MOVQ R15, R13            // R13 = j*4
	SHLQ $1, R15             // R15 = 2*j*4
	VMOVSD (R10)(R15*8), X9  // X9 = w2

	ADDQ R13, R15            // R15 = 2*j*4 + j*4 = 3*j*4
	VMOVSD (R10)(R15*8), X10 // X10 = w3

	// Load, multiply, butterfly (same pattern as stage 2)
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

	// Radix-4 butterfly - compute all outputs before writing
	VADDPS X0, X2, X4        // t0
	VSUBPS X2, X0, X5        // t1
	VADDPS X1, X3, X6        // t2
	VSUBPS X3, X1, X7        // t3

	// Compute (-i)*t3 for y1
	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14  // X14 = [t3.i, -t3.r] = (-i)*t3

	// Compute i*t3 for y3
	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x01, X11, X12, X12  // X12 = [-t3.i, t3.r] = i*t3

	// Compute all 4 outputs
	// y0 = t0 + t2
	VADDPS X4, X6, X0
	// y1 = t1 + (-i)*t3
	VADDPS X5, X14, X1
	// y2 = t0 - t2
	VSUBPS X6, X4, X2
	// y3 = t1 + i*t3
	VADDPS X5, X12, X3

	// Write all outputs
	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R14*8)

	INCQ DX
	JMP  r4_256_stage3_inner

r4_256_stage3_next:
	ADDQ $64, CX
	JMP  r4_256_stage3_outer

r4_256_stage4:
	// ==================================================================
	// Stage 4: 1 group, 64 butterflies
	// Twiddle step = 1
	// ==================================================================

	XORQ DX, DX              // DX = j

r4_256_stage4_loop:
	CMPQ DX, $64
	JGE  r4_256_done

	// idx0 = j, idx1 = j+64, idx2 = j+128, idx3 = j+192
	MOVQ DX, BX
	LEAQ 64(DX), SI
	LEAQ 128(DX), DI
	LEAQ 192(DX), R14

	// Twiddle factors: twiddle[j], twiddle[2*j], twiddle[3*j]
	VMOVSD (R10)(DX*8), X8   // w1
	MOVQ DX, R15
	SHLQ $1, R15
	VMOVSD (R10)(R15*8), X9  // w2
	ADDQ DX, R15
	VMOVSD (R10)(R15*8), X10 // w3

	// Load, multiply, butterfly
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

	// Radix-4 butterfly - compute all outputs before writing
	VADDPS X0, X2, X4        // t0
	VSUBPS X2, X0, X5        // t1
	VADDPS X1, X3, X6        // t2
	VSUBPS X3, X1, X7        // t3

	// Compute (-i)*t3 for y1
	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14  // X14 = [t3.i, -t3.r] = (-i)*t3

	// Compute i*t3 for y3
	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x01, X11, X12, X12  // X12 = [-t3.i, t3.r] = i*t3

	// Compute all 4 outputs
	// y0 = t0 + t2
	VADDPS X4, X6, X0
	// y1 = t1 + (-i)*t3
	VADDPS X5, X14, X1
	// y2 = t0 - t2
	VSUBPS X6, X4, X2
	// y3 = t1 + i*t3
	VADDPS X5, X12, X3

	// Write all outputs
	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R14*8)

	INCQ DX
	JMP  r4_256_stage4_loop

r4_256_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

r4_256_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 256, complex64, radix-4
// ===========================================================================
// Same as forward but uses conjugated twiddles (VFMSUBADD), +i for y1 and -i
// for y3, and applies 1/256 scaling.
TEXT ·InverseAVX2Size256Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 256)

	// Verify n == 256
	CMPQ R13, $256
	JNE  r4_256_inv_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $256
	JL   r4_256_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $256
	JL   r4_256_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $256
	JL   r4_256_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  r4_256_inv_use_dst
	MOVQ R11, R8             // In-place: use scratch

r4_256_inv_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 bit-reversal)
	// ==================================================================
	LEAQ ·bitrev256_r4(SB), R12
	XORQ CX, CX              // CX = i = 0

r4_256_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[i]]
	MOVQ AX, (R8)(CX*8)      // work[i] = src[bitrev[i]]
	INCQ CX
	CMPQ CX, $256
	JL   r4_256_inv_bitrev_loop

r4_256_inv_stage1:
	// ==================================================================
	// Stage 1: 64 radix-4 butterflies, stride=4
	// All twiddle factors are 1 (W^0), so no multiplication needed
	// ==================================================================

	XORQ CX, CX              // CX = base offset in elements (not bytes)

r4_256_inv_stage1_loop:
	CMPQ CX, $256
	JGE  r4_256_inv_stage2

	LEAQ (R8)(CX*8), SI      // SI = &work[base]
	VMOVSD (SI), X0          // X0 = a0
	VMOVSD 8(SI), X1         // X1 = a1
	VMOVSD 16(SI), X2        // X2 = a2
	VMOVSD 24(SI), X3        // X3 = a3

	VADDPS X0, X2, X4        // t0
	VSUBPS X2, X0, X5        // t1
	VADDPS X1, X3, X6        // t2
	VSUBPS X3, X1, X7        // t3

	// Compute (-i)*t3
	VPERMILPS $0xB1, X7, X8
	VXORPS X9, X9, X9
	VSUBPS X8, X9, X10
	VBLENDPS $0x02, X10, X8, X8  // X8 = (-i)*t3

	// Compute i*t3
	VPERMILPS $0xB1, X7, X11
	VSUBPS X11, X9, X10
	VBLENDPS $0x01, X10, X11, X11  // X11 = i*t3

	// y0 = t0 + t2
	VADDPS X4, X6, X0
	// y1 = t1 + i*t3
	VADDPS X5, X11, X1
	// y2 = t0 - t2
	VSUBPS X6, X4, X2
	// y3 = t1 + (-i)*t3
	VADDPS X5, X8, X3

	VMOVSD X0, (SI)
	VMOVSD X1, 8(SI)
	VMOVSD X2, 16(SI)
	VMOVSD X3, 24(SI)

	ADDQ $4, CX
	JMP  r4_256_inv_stage1_loop

r4_256_inv_stage2:
	// ==================================================================
	// Stage 2: 16 groups, each with 4 butterflies
	// Twiddle step = 16 (conjugated twiddles)
	// ==================================================================

	XORQ CX, CX              // CX = base offset

r4_256_inv_stage2_outer:
	CMPQ CX, $256
	JGE  r4_256_inv_stage3

	XORQ DX, DX              // DX = j

r4_256_inv_stage2_inner:
	CMPQ DX, $4
	JGE  r4_256_inv_stage2_next

	// idx0 = base+j, idx1 = base+j+4, idx2 = base+j+8, idx3 = base+j+12
	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 4(BX), SI
	LEAQ 8(BX), DI
	LEAQ 12(BX), R14

	// Twiddle factors: w1, w2, w3
	MOVQ DX, R15
	SHLQ $4, R15
	VMOVSD (R10)(R15*8), X8

	MOVQ R15, R13
	SHLQ $1, R15
	VMOVSD (R10)(R15*8), X9

	ADDQ R13, R15
	VMOVSD (R10)(R15*8), X10

	// Load data
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

	VADDPS X0, X2, X4        // t0
	VSUBPS X2, X0, X5        // t1
	VADDPS X1, X3, X6        // t2
	VSUBPS X3, X1, X7        // t3

	// (-i)*t3
	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14

	// i*t3
	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x01, X11, X12, X12

	VADDPS X4, X6, X0        // y0
	VADDPS X5, X12, X1       // y1 = t1 + i*t3
	VSUBPS X6, X4, X2        // y2
	VADDPS X5, X14, X3       // y3 = t1 + (-i)*t3

	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R14*8)

	INCQ DX
	JMP  r4_256_inv_stage2_inner

r4_256_inv_stage2_next:
	ADDQ $16, CX
	JMP  r4_256_inv_stage2_outer

r4_256_inv_stage3:
	// ==================================================================
	// Stage 3: 4 groups, each with 16 butterflies
	// Twiddle step = 4 (conjugated twiddles)
	// ==================================================================

	XORQ CX, CX

r4_256_inv_stage3_outer:
	CMPQ CX, $256
	JGE  r4_256_inv_stage4

	XORQ DX, DX

r4_256_inv_stage3_inner:
	CMPQ DX, $16
	JGE  r4_256_inv_stage3_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 16(BX), SI
	LEAQ 32(BX), DI
	LEAQ 48(BX), R14

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
	JMP  r4_256_inv_stage3_inner

r4_256_inv_stage3_next:
	ADDQ $64, CX
	JMP  r4_256_inv_stage3_outer

r4_256_inv_stage4:
	// ==================================================================
	// Stage 4: 1 group, 64 butterflies
	// Twiddle step = 1 (conjugated twiddles)
	// ==================================================================

	XORQ DX, DX

r4_256_inv_stage4_loop:
	CMPQ DX, $64
	JGE  r4_256_inv_scale

	MOVQ DX, BX
	LEAQ 64(DX), SI
	LEAQ 128(DX), DI
	LEAQ 192(DX), R14

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
	JMP  r4_256_inv_stage4_loop

r4_256_inv_scale:
	// ==================================================================
	// Apply 1/N scaling for inverse transform
	// ==================================================================
	MOVL ·twoFiftySixth32(SB), AX         // 1/256 = 0.00390625
	MOVD AX, X8
	VBROADCASTSS X8, Y8

	XORQ CX, CX

r4_256_inv_scale_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, (R8)(CX*1)
	VMOVUPS Y1, 32(R8)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $2048
	JL   r4_256_inv_scale_loop

	// ==================================================================
	// Copy results to dst if needed
	// ==================================================================
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r4_256_inv_done

	XORQ CX, CX

r4_256_inv_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMOVUPS Y0, (R9)(CX*1)
	VMOVUPS Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $2048
	JL   r4_256_inv_copy_loop

r4_256_inv_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

r4_256_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

GLOBL ·bitrev256_r4(SB), RODATA, $2048
DATA ·bitrev256_r4+0(SB)/8, $0
DATA ·bitrev256_r4+8(SB)/8, $64
DATA ·bitrev256_r4+16(SB)/8, $128
DATA ·bitrev256_r4+24(SB)/8, $192
DATA ·bitrev256_r4+32(SB)/8, $16
DATA ·bitrev256_r4+40(SB)/8, $80
DATA ·bitrev256_r4+48(SB)/8, $144
DATA ·bitrev256_r4+56(SB)/8, $208
DATA ·bitrev256_r4+64(SB)/8, $32
DATA ·bitrev256_r4+72(SB)/8, $96
DATA ·bitrev256_r4+80(SB)/8, $160
DATA ·bitrev256_r4+88(SB)/8, $224
DATA ·bitrev256_r4+96(SB)/8, $48
DATA ·bitrev256_r4+104(SB)/8, $112
DATA ·bitrev256_r4+112(SB)/8, $176
DATA ·bitrev256_r4+120(SB)/8, $240
DATA ·bitrev256_r4+128(SB)/8, $4
DATA ·bitrev256_r4+136(SB)/8, $68
DATA ·bitrev256_r4+144(SB)/8, $132
DATA ·bitrev256_r4+152(SB)/8, $196
DATA ·bitrev256_r4+160(SB)/8, $20
DATA ·bitrev256_r4+168(SB)/8, $84
DATA ·bitrev256_r4+176(SB)/8, $148
DATA ·bitrev256_r4+184(SB)/8, $212
DATA ·bitrev256_r4+192(SB)/8, $36
DATA ·bitrev256_r4+200(SB)/8, $100
DATA ·bitrev256_r4+208(SB)/8, $164
DATA ·bitrev256_r4+216(SB)/8, $228
DATA ·bitrev256_r4+224(SB)/8, $52
DATA ·bitrev256_r4+232(SB)/8, $116
DATA ·bitrev256_r4+240(SB)/8, $180
DATA ·bitrev256_r4+248(SB)/8, $244
DATA ·bitrev256_r4+256(SB)/8, $8
DATA ·bitrev256_r4+264(SB)/8, $72
DATA ·bitrev256_r4+272(SB)/8, $136
DATA ·bitrev256_r4+280(SB)/8, $200
DATA ·bitrev256_r4+288(SB)/8, $24
DATA ·bitrev256_r4+296(SB)/8, $88
DATA ·bitrev256_r4+304(SB)/8, $152
DATA ·bitrev256_r4+312(SB)/8, $216
DATA ·bitrev256_r4+320(SB)/8, $40
DATA ·bitrev256_r4+328(SB)/8, $104
DATA ·bitrev256_r4+336(SB)/8, $168
DATA ·bitrev256_r4+344(SB)/8, $232
DATA ·bitrev256_r4+352(SB)/8, $56
DATA ·bitrev256_r4+360(SB)/8, $120
DATA ·bitrev256_r4+368(SB)/8, $184
DATA ·bitrev256_r4+376(SB)/8, $248
DATA ·bitrev256_r4+384(SB)/8, $12
DATA ·bitrev256_r4+392(SB)/8, $76
DATA ·bitrev256_r4+400(SB)/8, $140
DATA ·bitrev256_r4+408(SB)/8, $204
DATA ·bitrev256_r4+416(SB)/8, $28
DATA ·bitrev256_r4+424(SB)/8, $92
DATA ·bitrev256_r4+432(SB)/8, $156
DATA ·bitrev256_r4+440(SB)/8, $220
DATA ·bitrev256_r4+448(SB)/8, $44
DATA ·bitrev256_r4+456(SB)/8, $108
DATA ·bitrev256_r4+464(SB)/8, $172
DATA ·bitrev256_r4+472(SB)/8, $236
DATA ·bitrev256_r4+480(SB)/8, $60
DATA ·bitrev256_r4+488(SB)/8, $124
DATA ·bitrev256_r4+496(SB)/8, $188
DATA ·bitrev256_r4+504(SB)/8, $252
DATA ·bitrev256_r4+512(SB)/8, $1
DATA ·bitrev256_r4+520(SB)/8, $65
DATA ·bitrev256_r4+528(SB)/8, $129
DATA ·bitrev256_r4+536(SB)/8, $193
DATA ·bitrev256_r4+544(SB)/8, $17
DATA ·bitrev256_r4+552(SB)/8, $81
DATA ·bitrev256_r4+560(SB)/8, $145
DATA ·bitrev256_r4+568(SB)/8, $209
DATA ·bitrev256_r4+576(SB)/8, $33
DATA ·bitrev256_r4+584(SB)/8, $97
DATA ·bitrev256_r4+592(SB)/8, $161
DATA ·bitrev256_r4+600(SB)/8, $225
DATA ·bitrev256_r4+608(SB)/8, $49
DATA ·bitrev256_r4+616(SB)/8, $113
DATA ·bitrev256_r4+624(SB)/8, $177
DATA ·bitrev256_r4+632(SB)/8, $241
DATA ·bitrev256_r4+640(SB)/8, $5
DATA ·bitrev256_r4+648(SB)/8, $69
DATA ·bitrev256_r4+656(SB)/8, $133
DATA ·bitrev256_r4+664(SB)/8, $197
DATA ·bitrev256_r4+672(SB)/8, $21
DATA ·bitrev256_r4+680(SB)/8, $85
DATA ·bitrev256_r4+688(SB)/8, $149
DATA ·bitrev256_r4+696(SB)/8, $213
DATA ·bitrev256_r4+704(SB)/8, $37
DATA ·bitrev256_r4+712(SB)/8, $101
DATA ·bitrev256_r4+720(SB)/8, $165
DATA ·bitrev256_r4+728(SB)/8, $229
DATA ·bitrev256_r4+736(SB)/8, $53
DATA ·bitrev256_r4+744(SB)/8, $117
DATA ·bitrev256_r4+752(SB)/8, $181
DATA ·bitrev256_r4+760(SB)/8, $245
DATA ·bitrev256_r4+768(SB)/8, $9
DATA ·bitrev256_r4+776(SB)/8, $73
DATA ·bitrev256_r4+784(SB)/8, $137
DATA ·bitrev256_r4+792(SB)/8, $201
DATA ·bitrev256_r4+800(SB)/8, $25
DATA ·bitrev256_r4+808(SB)/8, $89
DATA ·bitrev256_r4+816(SB)/8, $153
DATA ·bitrev256_r4+824(SB)/8, $217
DATA ·bitrev256_r4+832(SB)/8, $41
DATA ·bitrev256_r4+840(SB)/8, $105
DATA ·bitrev256_r4+848(SB)/8, $169
DATA ·bitrev256_r4+856(SB)/8, $233
DATA ·bitrev256_r4+864(SB)/8, $57
DATA ·bitrev256_r4+872(SB)/8, $121
DATA ·bitrev256_r4+880(SB)/8, $185
DATA ·bitrev256_r4+888(SB)/8, $249
DATA ·bitrev256_r4+896(SB)/8, $13
DATA ·bitrev256_r4+904(SB)/8, $77
DATA ·bitrev256_r4+912(SB)/8, $141
DATA ·bitrev256_r4+920(SB)/8, $205
DATA ·bitrev256_r4+928(SB)/8, $29
DATA ·bitrev256_r4+936(SB)/8, $93
DATA ·bitrev256_r4+944(SB)/8, $157
DATA ·bitrev256_r4+952(SB)/8, $221
DATA ·bitrev256_r4+960(SB)/8, $45
DATA ·bitrev256_r4+968(SB)/8, $109
DATA ·bitrev256_r4+976(SB)/8, $173
DATA ·bitrev256_r4+984(SB)/8, $237
DATA ·bitrev256_r4+992(SB)/8, $61
DATA ·bitrev256_r4+1000(SB)/8, $125
DATA ·bitrev256_r4+1008(SB)/8, $189
DATA ·bitrev256_r4+1016(SB)/8, $253
DATA ·bitrev256_r4+1024(SB)/8, $2
DATA ·bitrev256_r4+1032(SB)/8, $66
DATA ·bitrev256_r4+1040(SB)/8, $130
DATA ·bitrev256_r4+1048(SB)/8, $194
DATA ·bitrev256_r4+1056(SB)/8, $18
DATA ·bitrev256_r4+1064(SB)/8, $82
DATA ·bitrev256_r4+1072(SB)/8, $146
DATA ·bitrev256_r4+1080(SB)/8, $210
DATA ·bitrev256_r4+1088(SB)/8, $34
DATA ·bitrev256_r4+1096(SB)/8, $98
DATA ·bitrev256_r4+1104(SB)/8, $162
DATA ·bitrev256_r4+1112(SB)/8, $226
DATA ·bitrev256_r4+1120(SB)/8, $50
DATA ·bitrev256_r4+1128(SB)/8, $114
DATA ·bitrev256_r4+1136(SB)/8, $178
DATA ·bitrev256_r4+1144(SB)/8, $242
DATA ·bitrev256_r4+1152(SB)/8, $6
DATA ·bitrev256_r4+1160(SB)/8, $70
DATA ·bitrev256_r4+1168(SB)/8, $134
DATA ·bitrev256_r4+1176(SB)/8, $198
DATA ·bitrev256_r4+1184(SB)/8, $22
DATA ·bitrev256_r4+1192(SB)/8, $86
DATA ·bitrev256_r4+1200(SB)/8, $150
DATA ·bitrev256_r4+1208(SB)/8, $214
DATA ·bitrev256_r4+1216(SB)/8, $38
DATA ·bitrev256_r4+1224(SB)/8, $102
DATA ·bitrev256_r4+1232(SB)/8, $166
DATA ·bitrev256_r4+1240(SB)/8, $230
DATA ·bitrev256_r4+1248(SB)/8, $54
DATA ·bitrev256_r4+1256(SB)/8, $118
DATA ·bitrev256_r4+1264(SB)/8, $182
DATA ·bitrev256_r4+1272(SB)/8, $246
DATA ·bitrev256_r4+1280(SB)/8, $10
DATA ·bitrev256_r4+1288(SB)/8, $74
DATA ·bitrev256_r4+1296(SB)/8, $138
DATA ·bitrev256_r4+1304(SB)/8, $202
DATA ·bitrev256_r4+1312(SB)/8, $26
DATA ·bitrev256_r4+1320(SB)/8, $90
DATA ·bitrev256_r4+1328(SB)/8, $154
DATA ·bitrev256_r4+1336(SB)/8, $218
DATA ·bitrev256_r4+1344(SB)/8, $42
DATA ·bitrev256_r4+1352(SB)/8, $106
DATA ·bitrev256_r4+1360(SB)/8, $170
DATA ·bitrev256_r4+1368(SB)/8, $234
DATA ·bitrev256_r4+1376(SB)/8, $58
DATA ·bitrev256_r4+1384(SB)/8, $122
DATA ·bitrev256_r4+1392(SB)/8, $186
DATA ·bitrev256_r4+1400(SB)/8, $250
DATA ·bitrev256_r4+1408(SB)/8, $14
DATA ·bitrev256_r4+1416(SB)/8, $78
DATA ·bitrev256_r4+1424(SB)/8, $142
DATA ·bitrev256_r4+1432(SB)/8, $206
DATA ·bitrev256_r4+1440(SB)/8, $30
DATA ·bitrev256_r4+1448(SB)/8, $94
DATA ·bitrev256_r4+1456(SB)/8, $158
DATA ·bitrev256_r4+1464(SB)/8, $222
DATA ·bitrev256_r4+1472(SB)/8, $46
DATA ·bitrev256_r4+1480(SB)/8, $110
DATA ·bitrev256_r4+1488(SB)/8, $174
DATA ·bitrev256_r4+1496(SB)/8, $238
DATA ·bitrev256_r4+1504(SB)/8, $62
DATA ·bitrev256_r4+1512(SB)/8, $126
DATA ·bitrev256_r4+1520(SB)/8, $190
DATA ·bitrev256_r4+1528(SB)/8, $254
DATA ·bitrev256_r4+1536(SB)/8, $3
DATA ·bitrev256_r4+1544(SB)/8, $67
DATA ·bitrev256_r4+1552(SB)/8, $131
DATA ·bitrev256_r4+1560(SB)/8, $195
DATA ·bitrev256_r4+1568(SB)/8, $19
DATA ·bitrev256_r4+1576(SB)/8, $83
DATA ·bitrev256_r4+1584(SB)/8, $147
DATA ·bitrev256_r4+1592(SB)/8, $211
DATA ·bitrev256_r4+1600(SB)/8, $35
DATA ·bitrev256_r4+1608(SB)/8, $99
DATA ·bitrev256_r4+1616(SB)/8, $163
DATA ·bitrev256_r4+1624(SB)/8, $227
DATA ·bitrev256_r4+1632(SB)/8, $51
DATA ·bitrev256_r4+1640(SB)/8, $115
DATA ·bitrev256_r4+1648(SB)/8, $179
DATA ·bitrev256_r4+1656(SB)/8, $243
DATA ·bitrev256_r4+1664(SB)/8, $7
DATA ·bitrev256_r4+1672(SB)/8, $71
DATA ·bitrev256_r4+1680(SB)/8, $135
DATA ·bitrev256_r4+1688(SB)/8, $199
DATA ·bitrev256_r4+1696(SB)/8, $23
DATA ·bitrev256_r4+1704(SB)/8, $87
DATA ·bitrev256_r4+1712(SB)/8, $151
DATA ·bitrev256_r4+1720(SB)/8, $215
DATA ·bitrev256_r4+1728(SB)/8, $39
DATA ·bitrev256_r4+1736(SB)/8, $103
DATA ·bitrev256_r4+1744(SB)/8, $167
DATA ·bitrev256_r4+1752(SB)/8, $231
DATA ·bitrev256_r4+1760(SB)/8, $55
DATA ·bitrev256_r4+1768(SB)/8, $119
DATA ·bitrev256_r4+1776(SB)/8, $183
DATA ·bitrev256_r4+1784(SB)/8, $247
DATA ·bitrev256_r4+1792(SB)/8, $11
DATA ·bitrev256_r4+1800(SB)/8, $75
DATA ·bitrev256_r4+1808(SB)/8, $139
DATA ·bitrev256_r4+1816(SB)/8, $203
DATA ·bitrev256_r4+1824(SB)/8, $27
DATA ·bitrev256_r4+1832(SB)/8, $91
DATA ·bitrev256_r4+1840(SB)/8, $155
DATA ·bitrev256_r4+1848(SB)/8, $219
DATA ·bitrev256_r4+1856(SB)/8, $43
DATA ·bitrev256_r4+1864(SB)/8, $107
DATA ·bitrev256_r4+1872(SB)/8, $171
DATA ·bitrev256_r4+1880(SB)/8, $235
DATA ·bitrev256_r4+1888(SB)/8, $59
DATA ·bitrev256_r4+1896(SB)/8, $123
DATA ·bitrev256_r4+1904(SB)/8, $187
DATA ·bitrev256_r4+1912(SB)/8, $251
DATA ·bitrev256_r4+1920(SB)/8, $15
DATA ·bitrev256_r4+1928(SB)/8, $79
DATA ·bitrev256_r4+1936(SB)/8, $143
DATA ·bitrev256_r4+1944(SB)/8, $207
DATA ·bitrev256_r4+1952(SB)/8, $31
DATA ·bitrev256_r4+1960(SB)/8, $95
DATA ·bitrev256_r4+1968(SB)/8, $159
DATA ·bitrev256_r4+1976(SB)/8, $223
DATA ·bitrev256_r4+1984(SB)/8, $47
DATA ·bitrev256_r4+1992(SB)/8, $111
DATA ·bitrev256_r4+2000(SB)/8, $175
DATA ·bitrev256_r4+2008(SB)/8, $239
DATA ·bitrev256_r4+2016(SB)/8, $63
DATA ·bitrev256_r4+2024(SB)/8, $127
DATA ·bitrev256_r4+2032(SB)/8, $191
DATA ·bitrev256_r4+2040(SB)/8, $255

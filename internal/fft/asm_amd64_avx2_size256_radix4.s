//go:build amd64 && fft_asm && !purego

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

TEXT ·forwardAVX2Size256Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
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

	MOVQ bitrev+104(FP), AX
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
	MOVB $1, ret+120(FP)
	RET

r4_256_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 256, complex64, radix-4
// ===========================================================================
// Same as forward but uses conjugated twiddles (VFMSUBADD), +i for y1 and -i
// for y3, and applies 1/256 scaling.
TEXT ·inverseAVX2Size256Radix4Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
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

	MOVQ bitrev+104(FP), AX
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
	MOVL $0x3B800000, AX         // 1/256 = 0.00390625
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
	MOVB $1, ret+120(FP)
	RET

r4_256_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET

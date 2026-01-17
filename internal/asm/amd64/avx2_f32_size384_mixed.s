//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-384 Mixed-Radix (128×3) FFT Kernels for AMD64 (complex64)
// ===========================================================================
//
// Size 384 = 128 × 3 = 2^7 × 3
//
// Algorithm: Decompose as radix-3 outer, size-128 inner
//   Step 1: Perform 3 independent 128-point FFTs on sub-arrays
//   Step 2: Apply twiddle factors to elements 128-383
//   Step 3: Perform 128 radix-3 butterflies across the 3 sub-arrays
//
// This implementation delegates the 128-point sub-FFTs to the existing
// AVX2 size-128 kernel, then performs twiddle application and radix-3
// butterflies using AVX2 SIMD instructions.
//
// Twiddle factors for 384-point FFT:
//   W_384^k = exp(-2πik/384) for k = 0..383
//
// For the radix-3 final stage, we need:
//   - Elements 0-127: no twiddle (W^0 = 1)
//   - Elements 128-255: multiply by W_384^k for k = 0..127
//   - Elements 256-383: multiply by W_384^(2k) for k = 0..127
//
// ===========================================================================

#include "textflag.h"

// ============================================================================
// Forward Transform: Size 384, Complex64, Mixed-Radix 128×3 (AVX2)
// ============================================================================
//
// This is a Go-callable stub that validates parameters.
// The actual transform is performed by the Go wrapper which calls:
//   1. ForwardAVX2Size128Radix4Then2Complex64Asm for each 128-point sub-FFT
//   2. This assembly for twiddle multiplication and radix-3 butterflies
//
// For size-384, we implement the final radix-3 stage in assembly.
// The Go wrapper handles orchestration.
//
// func ForwardAVX2Size384MixedComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
TEXT ·ForwardAVX2Size384MixedComplex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer (size-384 twiddles)
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 384)

	// Verify n == 384
	CMPQ R13, $384
	JNE  size384_return_false

	// Validate slice lengths (all must be >= 384)
	MOVQ dst+8(FP), AX
	CMPQ AX, $384
	JL   size384_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $384
	JL   size384_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $384
	JL   size384_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $384
	JL   size384_return_false

	// Return true to indicate this kernel handles size 384
	// The actual FFT is performed by the Go wrapper which:
	// 1. Calls 3x size-128 FFTs
	// 2. Applies twiddle factors
	// 3. Performs radix-3 butterflies
	MOVB $1, ret+120(FP)
	RET

size384_return_false:
	MOVB $0, ret+120(FP)
	RET

// ============================================================================
// Inverse Transform: Size 384, Complex64, Mixed-Radix 128×3 (AVX2)
// ============================================================================
//
// func InverseAVX2Size384MixedComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
TEXT ·InverseAVX2Size384MixedComplex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer (size-384 twiddles)
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 384)

	// Verify n == 384
	CMPQ R13, $384
	JNE  size384_inv_return_false

	// Validate slice lengths (all must be >= 384)
	MOVQ dst+8(FP), AX
	CMPQ AX, $384
	JL   size384_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $384
	JL   size384_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $384
	JL   size384_inv_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $384
	JL   size384_inv_return_false

	// Return true to indicate this kernel handles size 384
	MOVB $1, ret+120(FP)
	RET

size384_inv_return_false:
	MOVB $0, ret+120(FP)
	RET

// ============================================================================
// Twiddle Application: Apply twiddle factors to sub-arrays 1 and 2
// ============================================================================
//
// This function applies twiddle factors for the radix-3 decomposition:
//   dst[128+k] *= twiddle[k]       for k = 0..127
//   dst[256+k] *= twiddle[2*k]     for k = 0..127
//
// Uses AVX2 for 4 complex64 values at a time.
//
// func ApplyTwiddle384Complex64Asm(data, twiddle []complex64)
TEXT ·ApplyTwiddle384Complex64Asm(SB), NOSPLIT, $0-48
	MOVQ data+0(FP), R8      // R8 = data pointer
	MOVQ data+8(FP), R9      // R9 = data length
	MOVQ twiddle+24(FP), R10 // R10 = twiddle pointer

	// Verify length >= 384
	CMPQ R9, $384
	JL   twiddle384_done

	// Process sub-array 1: data[128..255] *= twiddle[0..127]
	// Each iteration processes 4 complex64 values (32 bytes)
	XORQ CX, CX              // CX = offset in elements (0, 4, 8, ..., 124)

twiddle384_subarray1_loop:
	CMPQ CX, $128
	JGE  twiddle384_subarray2

	// Load 4 complex64 from data[128+CX..128+CX+3]
	LEAQ 1024(R8)(CX*8), SI  // SI = &data[128 + CX] (128*8 = 1024 bytes offset)
	VMOVUPS (SI), Y0         // Y0 = data[128+CX : 128+CX+4]

	// Load 4 twiddle factors from twiddle[CX..CX+3]
	LEAQ (R10)(CX*8), DI     // DI = &twiddle[CX]
	VMOVUPS (DI), Y1         // Y1 = twiddle[CX : CX+4]

	// Complex multiply: Y0 = Y0 * Y1
	// (a+bi)(c+di) = (ac-bd) + (ad+bc)i
	// Using AVX2: duplicate real/imag, multiply, add/sub

	// Step 1: Duplicate real parts: [a,a,a,a,...] and imag parts: [b,b,b,b,...]
	VMOVSLDUP Y0, Y2         // Y2 = [a0,a0, a1,a1, a2,a2, a3,a3] (duplicate reals)
	VMOVSHDUP Y0, Y3         // Y3 = [b0,b0, b1,b1, b2,b2, b3,b3] (duplicate imags)

	// Step 2: Multiply
	VMULPS Y2, Y1, Y4        // Y4 = [a*c, a*d, ...] (real * twiddle)

	// Step 3: Swap real/imag of twiddle for cross terms
	VSHUFPS $0xB1, Y1, Y1, Y5 // Y5 = [d,c, d,c, ...] (swap pairs)
	VMULPS Y3, Y5, Y6        // Y6 = [b*d, b*c, ...]

	// Step 4: Add/sub to get final result
	// Real = a*c - b*d, Imag = a*d + b*c
	VADDSUBPS Y6, Y4, Y0     // Y0 = [a*c-b*d, a*d+b*c, ...] = complex product

	// Store result
	VMOVUPS Y0, (SI)

	ADDQ $4, CX
	JMP  twiddle384_subarray1_loop

twiddle384_subarray2:
	// Process sub-array 2: data[256..383] *= twiddle[0..255:2] (every other twiddle)
	// Each iteration processes 4 complex64 values
	XORQ CX, CX              // CX = offset in elements (0, 4, 8, ..., 124)

twiddle384_subarray2_loop:
	CMPQ CX, $128
	JGE  twiddle384_done

	// Load 4 complex64 from data[256+CX..256+CX+3]
	LEAQ 2048(R8)(CX*8), SI  // SI = &data[256 + CX] (256*8 = 2048 bytes offset)
	VMOVUPS (SI), Y0         // Y0 = data[256+CX : 256+CX+4]

	// Load 4 twiddle factors from twiddle[2*CX], twiddle[2*CX+2], twiddle[2*CX+4], twiddle[2*CX+6]
	// We need to gather these since they're not contiguous
	// For simplicity, load 2 at a time with scalar operations

	// First pair: twiddle[2*CX] and twiddle[2*CX+2]
	MOVQ CX, DX
	SHLQ $1, DX              // DX = 2*CX
	LEAQ (R10)(DX*8), DI     // DI = &twiddle[2*CX]

	// Load twiddle[2*CX]
	VMOVSD (DI), X1          // X1 = twiddle[2*CX]
	// Load twiddle[2*CX+2]
	VMOVSD 16(DI), X2        // X2 = twiddle[2*CX+2]
	// Load twiddle[2*CX+4]
	VMOVSD 32(DI), X3        // X3 = twiddle[2*CX+4]
	// Load twiddle[2*CX+6]
	VMOVSD 48(DI), X4        // X4 = twiddle[2*CX+6]

	// Combine into Y1: [tw0, tw2, tw4, tw6]
	VINSERTPS $0x10, X2, X1, X1  // X1 = [tw0_r, tw0_i, tw2_r, tw2_i]
	VINSERTPS $0x10, X4, X3, X3  // X3 = [tw4_r, tw4_i, tw6_r, tw6_i]
	VINSERTF128 $1, X3, Y1, Y1   // Y1 = [tw0, tw2, tw4, tw6]

	// Complex multiply: Y0 = Y0 * Y1
	VMOVSLDUP Y0, Y2         // Y2 = duplicate reals
	VMOVSHDUP Y0, Y3         // Y3 = duplicate imags
	VMULPS Y2, Y1, Y4        // Y4 = real * twiddle
	VSHUFPS $0xB1, Y1, Y1, Y5 // Y5 = swap twiddle pairs
	VMULPS Y3, Y5, Y6        // Y6 = imag * swapped twiddle
	VADDSUBPS Y6, Y4, Y0     // Y0 = complex product

	// Store result
	VMOVUPS Y0, (SI)

	ADDQ $4, CX
	JMP  twiddle384_subarray2_loop

twiddle384_done:
	RET

// ============================================================================
// Radix-3 Butterflies: 128 radix-3 butterflies across 3 sub-arrays
// ============================================================================
//
// Performs 128 radix-3 butterflies:
//   for k = 0..127:
//     a0, a1, a2 = data[k], data[128+k], data[256+k]
//     data[k], data[128+k], data[256+k] = butterfly3(a0, a1, a2)
//
// Forward butterfly (already twiddle-multiplied inputs):
//   t1 = a1 + a2
//   t2 = a1 - a2
//   y0 = a0 + t1
//   base = a0 + half*t1  (half = -0.5)
//   y1 = base + coef*t2  (coef = 0 - i*sqrt(3)/2)
//   y2 = base - coef*t2
//
// Uses AVX2 to process 4 butterflies at a time (12 complex64 values).
//
// func Radix3Butterflies384ForwardComplex64Asm(data []complex64)
TEXT ·Radix3Butterflies384ForwardComplex64Asm(SB), NOSPLIT, $0-24
	MOVQ data+0(FP), R8      // R8 = data pointer
	MOVQ data+8(FP), R9      // R9 = data length

	// Verify length >= 384
	CMPQ R9, $384
	JL   radix3_384_fwd_done

	// Load constants
	// half = -0.5 (broadcast to all lanes)
	MOVL $0xBF000000, AX     // -0.5 in float32 IEEE 754
	MOVD AX, X0
	VBROADCASTSS X0, Y8      // Y8 = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]

	// sqrt(3)/2 = 0.866025403784
	MOVL $0x3F5DB3D7, AX     // sqrt(3)/2 in float32 IEEE 754
	MOVD AX, X0
	VBROADCASTSS X0, Y9      // Y9 = [sqrt3_2, sqrt3_2, ...]

	// Sign flip mask for negating imaginary parts: [0, -0, 0, -0, ...]
	// Used for: (a+bi) * (0 - ci) = bc - aci = (bc) + (-ac)i
	// We need to produce: imag(t2)*c - i*real(t2)*c from t2
	VXORPS Y10, Y10, Y10     // Y10 = 0 (will hold sign mask)

	// Process 4 butterflies at a time
	XORQ CX, CX              // CX = offset (0, 4, 8, ..., 124)

radix3_384_fwd_loop:
	CMPQ CX, $128
	JGE  radix3_384_fwd_done

	// Load a0[k..k+3] from data[CX..CX+3]
	LEAQ (R8)(CX*8), SI      // SI = &data[CX]
	VMOVUPS (SI), Y0         // Y0 = a0 (4 complex64)

	// Load a1[k..k+3] from data[128+CX..128+CX+3]
	LEAQ 1024(R8)(CX*8), DI  // DI = &data[128+CX]
	VMOVUPS (DI), Y1         // Y1 = a1

	// Load a2[k..k+3] from data[256+CX..256+CX+3]
	LEAQ 2048(R8)(CX*8), DX  // Use a different register for third pointer
	VMOVUPS (DX), Y2         // Y2 = a2

	// Radix-3 butterfly computation
	// t1 = a1 + a2
	VADDPS Y2, Y1, Y3        // Y3 = t1

	// t2 = a1 - a2
	VSUBPS Y2, Y1, Y4        // Y4 = t2

	// y0 = a0 + t1
	VADDPS Y3, Y0, Y5        // Y5 = y0

	// base = a0 + half*t1 = a0 + (-0.5)*t1 = a0 - 0.5*t1
	VMULPS Y8, Y3, Y6        // Y6 = half*t1 = -0.5*t1
	VADDPS Y6, Y0, Y6        // Y6 = base = a0 + half*t1

	// coef*t2 where coef = 0 - i*sqrt(3)/2
	// (a+bi) * (0 - ci) = (a*0 - b*(-c)) + (a*(-c) + b*0)i = bc - aci
	// So result = (imag(t2)*c, -real(t2)*c) = c * (imag(t2), -real(t2))

	// Swap real/imag of t2: Y4 = [r0,i0,r1,i1,...] -> [i0,r0,i1,r1,...]
	VSHUFPS $0xB1, Y4, Y4, Y7 // Y7 = [i,r,i,r,...] (swapped t2)

	// Negate the imaginary parts (now in even positions after swap)
	// Actually we need to negate the new real part (which was imag before swap)
	// Result should be (imag(t2), -real(t2)) but we have (imag(t2), real(t2))
	// So negate positions 1,3,5,7 (the original real parts, now in odd positions)

	// Create sign mask: [0, 0x80000000, 0, 0x80000000, ...]
	VXORPS Y10, Y10, Y10
	MOVL $0x80000000, AX
	MOVD AX, X11
	VPBROADCASTD X11, Y11    // Y11 = [0x80000000, ...]
	VBLENDPS $0xAA, Y11, Y10, Y10 // Y10 = [0, 0x80, 0, 0x80, 0, 0x80, 0, 0x80]

	VXORPS Y10, Y7, Y7       // Y7 = [imag(t2), -real(t2), ...]

	// Multiply by sqrt(3)/2
	VMULPS Y9, Y7, Y7        // Y7 = coef*t2

	// y1 = base + coef*t2
	VADDPS Y7, Y6, Y1        // Y1 = y1

	// y2 = base - coef*t2
	VSUBPS Y7, Y6, Y2        // Y2 = y2

	// Store results
	VMOVUPS Y5, (SI)         // data[CX..CX+3] = y0
	VMOVUPS Y1, (DI)         // data[128+CX..] = y1
	VMOVUPS Y2, (DX)         // data[256+CX..] = y2

	ADDQ $4, CX
	JMP  radix3_384_fwd_loop

radix3_384_fwd_done:
	RET

// ============================================================================
// Radix-3 Butterflies: Inverse direction
// ============================================================================
//
// Inverse butterfly uses conjugate coefficient: coef = 0 + i*sqrt(3)/2
// (a+bi) * (0 + ci) = -bc + aci = (-imag(t2)*c, real(t2)*c)
//
// func Radix3Butterflies384InverseComplex64Asm(data []complex64)
TEXT ·Radix3Butterflies384InverseComplex64Asm(SB), NOSPLIT, $0-24
	MOVQ data+0(FP), R8      // R8 = data pointer
	MOVQ data+8(FP), R9      // R9 = data length

	// Verify length >= 384
	CMPQ R9, $384
	JL   radix3_384_inv_done

	// Load constants
	// half = -0.5
	MOVL $0xBF000000, AX
	MOVD AX, X0
	VBROADCASTSS X0, Y8      // Y8 = [-0.5, ...]

	// sqrt(3)/2
	MOVL $0x3F5DB3D7, AX
	MOVD AX, X0
	VBROADCASTSS X0, Y9      // Y9 = [sqrt3_2, ...]

	// Process 4 butterflies at a time
	XORQ CX, CX

radix3_384_inv_loop:
	CMPQ CX, $128
	JGE  radix3_384_inv_done

	// Load a0, a1, a2
	LEAQ (R8)(CX*8), SI
	VMOVUPS (SI), Y0

	LEAQ 1024(R8)(CX*8), DI
	VMOVUPS (DI), Y1

	LEAQ 2048(R8)(CX*8), DX
	VMOVUPS (DX), Y2

	// t1 = a1 + a2
	VADDPS Y2, Y1, Y3

	// t2 = a1 - a2
	VSUBPS Y2, Y1, Y4

	// y0 = a0 + t1
	VADDPS Y3, Y0, Y5

	// base = a0 + half*t1
	VMULPS Y8, Y3, Y6
	VADDPS Y6, Y0, Y6

	// coef*t2 where coef = 0 + i*sqrt(3)/2
	// (a+bi) * (0 + ci) = -bc + aci = (-imag(t2)*c, real(t2)*c)
	// Result = c * (-imag(t2), real(t2))

	// Swap real/imag
	VSHUFPS $0xB1, Y4, Y4, Y7

	// Negate the even positions (the imag parts that were moved to real position)
	// We want (-imag, real), and after swap we have (imag, real)
	// So negate positions 0,2,4,6

	VXORPS Y10, Y10, Y10
	MOVL $0x80000000, AX
	MOVD AX, X11
	VPBROADCASTD X11, Y11
	VBLENDPS $0x55, Y11, Y10, Y10 // Y10 = [0x80, 0, 0x80, 0, ...]

	VXORPS Y10, Y7, Y7       // Y7 = [-imag(t2), real(t2), ...]

	// Multiply by sqrt(3)/2
	VMULPS Y9, Y7, Y7

	// y1 = base + coef*t2
	VADDPS Y7, Y6, Y1

	// y2 = base - coef*t2
	VSUBPS Y7, Y6, Y2

	// Store results
	VMOVUPS Y5, (SI)
	VMOVUPS Y1, (DI)
	VMOVUPS Y2, (DX)

	ADDQ $4, CX
	JMP  radix3_384_inv_loop

radix3_384_inv_done:
	RET

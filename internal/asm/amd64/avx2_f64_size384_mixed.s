//go:build amd64 && !purego

// ===========================================================================
// AVX2 Size-384 Mixed-Radix (128×3) FFT Kernels for AMD64 (complex128)
// ===========================================================================
//
// Size 384 = 128 × 3 = 2^7 × 3
//
// REACHABILITY -- read before including this file in any size-specific sweep.
// It carries no row in cmd/gencodelets/specs.go, which is the same shape as the
// sixteen size-specific AVX2 files PLAN.md §1.3 is retiring, and it is NOT one
// of them. This file is reached through the Go codelet
// forwardDIT384MixedComplex128 / inverseDIT384MixedComplex128, not through the
// KernelStrategy switch in internal/fft/kernels_amd64_size_specific.go, so the
// §1.3 measurement says nothing about it and deleting it on that result would
// break a live path. The census in docs/IMPLEMENTATION_INVENTORY.md reports its
// symbols live for exactly this reason.
//
// Algorithm: Decompose as radix-3 outer, size-128 inner
//   Step 1: Perform 3 independent 128-point FFTs on sub-arrays
//   Step 2: Apply twiddle factors to elements 128-383
//   Step 3: Perform 128 radix-3 butterflies across the 3 sub-arrays
//
// Twiddle factors for 384-point FFT:
//   W_384^k = exp(-2πik/384) for k = 0..383
//
// ===========================================================================

#include "textflag.h"

// Sign mask that flips the imaginary component of 2 packed complex128, i.e.
// conjugation by VXORPD. Odd float64 lanes carry the float64 sign bit.
GLOBL ·conjMask384f64<>(SB), RODATA|NOPTR, $32
DATA ·conjMask384f64<>+0(SB)/8,  $0x0000000000000000
DATA ·conjMask384f64<>+8(SB)/8,  $0x8000000000000000
DATA ·conjMask384f64<>+16(SB)/8, $0x0000000000000000
DATA ·conjMask384f64<>+24(SB)/8, $0x8000000000000000

// ============================================================================
// Twiddle Application: Apply twiddle factors to sub-arrays 1 and 2
// ============================================================================
//
// func ApplyTwiddle384Complex128Asm(data, twiddle []complex128)
TEXT ·ApplyTwiddle384Complex128Asm(SB), NOSPLIT, $0-48
	MOVQ data+0(FP), R8      // R8 = data pointer
	MOVQ data_len+8(FP), R9      // R9 = data length
	MOVQ twiddle+24(FP), R10 // R10 = twiddle pointer

	// Verify length >= 384
	CMPQ R9, $384
	JL   twiddle384_done

	// Process sub-array 1: data[128..255] *= twiddle[0..127]
	// Each iteration processes 2 complex128 values (32 bytes)
	XORQ CX, CX              // CX = offset in elements

twiddle384_subarray1_loop:
	CMPQ CX, $128
	JGE  twiddle384_subarray2

	// Load 2 complex128 from data[128+CX..128+CX+1]
	// Offset 128*16 = 2048
	MOVQ CX, R14
	SHLQ $4, R14             // R14 = CX * 16 bytes
	LEAQ 2048(R8)(R14*1), SI
	VMOVUPD (SI), Y0         // Y0 = data[128+CX : 128+CX+2]

	// Load 2 twiddle factors from twiddle[CX..CX+1]
	LEAQ (R10)(R14*1), DI
	VMOVUPD (DI), Y1         // Y1 = twiddle[CX : CX+2]

	// Complex multiply: Y0 = Y0 * Y1 (FMA-fused multiply-addsub)
	// using VPERMPD for shuffling doubles in YMM

	// Duplicate reals: [r0, r0, r1, r1]
	// 0xA0 = 10 10 00 00 (indices 2,2,0,0)
	VPERMPD $0xA0, Y0, Y2

	// Duplicate imags: [i0, i0, i1, i1]
	// 0xF5 = 11 11 01 01 (indices 3,3,1,1)
	VPERMPD $0xF5, Y0, Y3

	// Swap twiddle pairs: [ti0, tr0, ti1, tr1]
	// 0xB1 = 10 11 00 01 (indices 2,3,0,1)
	VPERMPD $0xB1, Y1, Y5

	// Multiply imag*swapped_twiddle
	VMULPD Y3, Y5, Y6        // Y6 = [i0*ti0, i0*tr0, i1*ti1, i1*tr1]

	// FMA: Y6 = twiddle*real -/+ Y6
	// Real = a*c - b*d, Imag = a*d + b*c (single rounding step per lane)
	VFMADDSUB231PD Y2, Y1, Y6 // Y6 = [r*tr-i*ti, r*ti+i*tr, ...]

	// Store result
	VMOVUPD Y6, (SI)

	ADDQ $2, CX
	JMP  twiddle384_subarray1_loop

twiddle384_subarray2:
	// Process sub-array 2: data[256..383] *= twiddle[0..255:2]
	XORQ CX, CX

twiddle384_subarray2_loop:
	CMPQ CX, $128
	JGE  twiddle384_done

	// Load 2 complex128 from data[256+CX..256+CX+1]
	// Offset 256*16 = 4096
	MOVQ CX, R14
	SHLQ $4, R14             // R14 = CX * 16 bytes
	LEAQ 4096(R8)(R14*1), SI
	VMOVUPD (SI), Y0

	// Load twiddle factors: twiddle[2*CX] and twiddle[2*CX+2]
	MOVQ CX, DX
	SHLQ $5, DX              // DX = 2 * CX * 16 = CX * 32 bytes
	LEAQ (R10)(DX*1), DI    // DI = &twiddle[2*CX]

	// Load twiddle[2*CX] (16 bytes) into low half of Y1 (using X1)
	VMOVUPD (DI), X1         // X1 = twiddle[2*CX]

	// Load twiddle[2*CX+2] (16 bytes) into upper half of Y1
	// Offset (2*CX+2)*16 - (2*CX)*16 = 32 bytes
	VMOVUPD 32(DI), X2       // X2 = twiddle[2*CX+2]

	VINSERTF128 $1, X2, Y1, Y1 // Y1 = [tw[2*CX], tw[2*CX+2]]

	// Complex multiply (same as above, FMA-fused)
	VPERMPD $0xA0, Y0, Y2    // Reals
	VPERMPD $0xF5, Y0, Y3    // Imags
	VPERMPD $0xB1, Y1, Y5    // Swap Twiddle
	VMULPD Y3, Y5, Y6        // Imag * Swapped
	VFMADDSUB231PD Y2, Y1, Y6 // Y6 = twiddle*real -/+ Y6 = complex product

	// Store result
	VMOVUPD Y6, (SI)

	ADDQ $2, CX
	JMP  twiddle384_subarray2_loop

twiddle384_done:
	VZEROUPPER
	RET

// ============================================================================
// Conjugate Twiddle Application: the inverse-direction twin of the above
// ============================================================================
//
//   dst[128+k] *= conj(twiddle[k])       for k = 0..127
//   dst[256+k] *= conj(twiddle[2*k])     for k = 0..127
//
// Identical to ApplyTwiddle384Complex128Asm except that each loaded twiddle
// vector is conjugated with a single VXORPD against conjMask384f64 before the
// multiply. Conjugating the twiddle is one instruction; conjugating the product
// instead is not, because VFMADDSUB's fixed even-subtract/odd-add pattern gives
// the wrong sign on the imaginary term for a conjugated multiply.
//
// func ApplyConjTwiddle384Complex128Asm(data, twiddle []complex128)
TEXT ·ApplyConjTwiddle384Complex128Asm(SB), NOSPLIT, $0-48
	MOVQ data+0(FP), R8      // R8 = data pointer
	MOVQ data_len+8(FP), R9  // R9 = data length
	MOVQ twiddle+24(FP), R10 // R10 = twiddle pointer

	// Verify length >= 384
	CMPQ R9, $384
	JL   conjtwiddle384_done

	VMOVUPD ·conjMask384f64<>(SB), Y7 // Y7 = imaginary-lane sign mask

	// Process sub-array 1: data[128..255] *= conj(twiddle[0..127])
	XORQ CX, CX

conjtwiddle384_subarray1_loop:
	CMPQ CX, $128
	JGE  conjtwiddle384_subarray2

	MOVQ CX, R14
	SHLQ $4, R14             // R14 = CX * 16 bytes
	LEAQ 2048(R8)(R14*1), SI
	VMOVUPD (SI), Y0         // Y0 = data[128+CX : 128+CX+2]

	LEAQ (R10)(R14*1), DI
	VMOVUPD (DI), Y1         // Y1 = twiddle[CX : CX+2]
	VXORPD Y7, Y1, Y1        // Y1 = conj(twiddle)

	VPERMPD $0xA0, Y0, Y2    // Reals
	VPERMPD $0xF5, Y0, Y3    // Imags
	VPERMPD $0xB1, Y1, Y5    // Swap twiddle
	VMULPD Y3, Y5, Y6        // Imag * swapped
	VFMADDSUB231PD Y2, Y1, Y6 // Complex product

	VMOVUPD Y6, (SI)

	ADDQ $2, CX
	JMP  conjtwiddle384_subarray1_loop

conjtwiddle384_subarray2:
	// Process sub-array 2: data[256..383] *= conj(twiddle[2*k])
	XORQ CX, CX

conjtwiddle384_subarray2_loop:
	CMPQ CX, $128
	JGE  conjtwiddle384_done

	MOVQ CX, R14
	SHLQ $4, R14
	LEAQ 4096(R8)(R14*1), SI
	VMOVUPD (SI), Y0

	MOVQ CX, DX
	SHLQ $5, DX              // DX = 2 * CX * 16 bytes
	LEAQ (R10)(DX*1), DI     // DI = &twiddle[2*CX]

	VMOVUPD (DI), X1         // X1 = twiddle[2*CX]
	VMOVUPD 32(DI), X2       // X2 = twiddle[2*CX+2]
	VINSERTF128 $1, X2, Y1, Y1 // Y1 = [tw[2*CX], tw[2*CX+2]]
	VXORPD Y7, Y1, Y1        // Y1 = conj(twiddle)

	VPERMPD $0xA0, Y0, Y2
	VPERMPD $0xF5, Y0, Y3
	VPERMPD $0xB1, Y1, Y5
	VMULPD Y3, Y5, Y6
	VFMADDSUB231PD Y2, Y1, Y6

	VMOVUPD Y6, (SI)

	ADDQ $2, CX
	JMP  conjtwiddle384_subarray2_loop

conjtwiddle384_done:
	VZEROUPPER
	RET

// ============================================================================
// Radix-3 Butterflies: 128 radix-3 butterflies (complex128)
// ============================================================================
//
// func Radix3Butterflies384ForwardComplex128Asm(data []complex128)
TEXT ·Radix3Butterflies384ForwardComplex128Asm(SB), NOSPLIT, $0-24
	MOVQ data+0(FP), R8      // R8 = data pointer
	MOVQ data_len+8(FP), R9      // R9 = data length

	// Verify length >= 384
	CMPQ R9, $384
	JL   radix3_384_fwd_done

	// Load constants
	// half = -0.5
	MOVQ $0xBFE0000000000000, R15 // -0.5 double
	VMOVQ R15, X0
	VBROADCASTSD X0, Y8      // Y8 = [-0.5, -0.5, -0.5, -0.5]

	// sqrt(3)/2 = 0.8660254037844386
	MOVQ $0x3FEBB67AE8584CAA, R15 // sqrt(3)/2 double
	VMOVQ R15, X0
	VBROADCASTSD X0, Y9      // Y9 = [sqrt3_2, ...]

	// Prepare sign masks for Forward transform
	// Forward: coef = 0 - i*sqrt(3)/2
	// Need to negate indices 1 and 3 (real part after swap)
	// Mask = [0, Sign, 0, Sign]
	MOVQ $0x8000000000000000, R15
	VMOVQ R15, X11
	VPBROADCASTQ X11, Y11    // Y11 = [Sign, Sign, Sign, Sign]
	VXORPD Y10, Y10, Y10     // Y10 = 0
	// Blend: 1010 = 0x0A. Select from Y11 if bit set.
	VBLENDPD $0x0A, Y11, Y10, Y12 // Y12 = [0, Sign, 0, Sign]

	// Process 2 butterflies at a time (32 bytes per array)
	XORQ CX, CX

radix3_384_fwd_loop:
	CMPQ CX, $128
	JGE  radix3_384_fwd_done

	// Load a0 (data[CX])
	MOVQ CX, R14
	SHLQ $4, R14
	LEAQ (R8)(R14*1), SI
	VMOVUPD (SI), Y0         // Y0 = a0

	// Load a1 (data[128+CX])
	// Offset = 128 * 16 = 2048
	LEAQ 2048(R8)(R14*1), DI
	VMOVUPD (DI), Y1         // Y1 = a1

	// Load a2 (data[256+CX])
	// Offset = 256 * 16 = 4096
	LEAQ 4096(R8)(R14*1), DX
	VMOVUPD (DX), Y2         // Y2 = a2

	// t1 = a1 + a2
	VADDPD Y2, Y1, Y3        // Y3 = t1

	// t2 = a1 - a2
	VSUBPD Y2, Y1, Y4        // Y4 = t2

	// y0 = a0 + t1
	VADDPD Y3, Y0, Y5        // Y5 = y0

	// base = a0 + half*t1
	VMULPD Y8, Y3, Y6        // Y6 = half*t1
	VADDPD Y6, Y0, Y6        // Y6 = base

	// coef*t2
	// Swap real/imag of t2: [r0, i0, r1, i1] -> [i0, r0, i1, r1]
	VPERMPD $0xB1, Y4, Y7    // Y7 = swapped t2

	// Apply sign mask (Negate indices 1 and 3)
	// Y12 has signs at 1 and 3.
	VXORPD Y12, Y7, Y7       // Y7 = [i0, -r0, i1, -r1]

	// Multiply by sqrt(3)/2
	VMULPD Y9, Y7, Y7        // Y7 = coef*t2

	// y1 = base + coef*t2
	VADDPD Y7, Y6, Y1        // Y1 = y1

	// y2 = base - coef*t2
	VSUBPD Y7, Y6, Y2        // Y2 = y2

	// Store results
	VMOVUPD Y5, (SI)
	VMOVUPD Y1, (DI)
	VMOVUPD Y2, (DX)

	ADDQ $2, CX
	JMP  radix3_384_fwd_loop

radix3_384_fwd_done:
	VZEROUPPER
	RET

// ============================================================================
// Radix-3 Butterflies: Inverse (complex128)
// ============================================================================
//
// func Radix3Butterflies384InverseComplex128Asm(data []complex128)
TEXT ·Radix3Butterflies384InverseComplex128Asm(SB), NOSPLIT, $0-24
	MOVQ data+0(FP), R8
	MOVQ data_len+8(FP), R9

	CMPQ R9, $384
	JL   radix3_384_inv_done

	// Constants
	MOVQ $0xBFE0000000000000, R15
	VMOVQ R15, X0
	VBROADCASTSD X0, Y8      // Y8 = -0.5

	MOVQ $0x3FEBB67AE8584CAA, R15
	VMOVQ R15, X0
	VBROADCASTSD X0, Y9      // Y9 = sqrt3_2

	// Sign mask for Inverse
	// Inverse: coef = 0 + i*sqrt(3)/2
	// Result = (-imag, real)
	// Swap gives (imag, real)
	// Negate index 0 and 2
	MOVQ $0x8000000000000000, R15
	VMOVQ R15, X11
	VPBROADCASTQ X11, Y11
	VXORPD Y10, Y10, Y10
	// Blend 0101 = 0x05
	VBLENDPD $0x05, Y11, Y10, Y12

	XORQ CX, CX

radix3_384_inv_loop:
	CMPQ CX, $128
	JGE  radix3_384_inv_done

	MOVQ CX, R14
	SHLQ $4, R14

	LEAQ (R8)(R14*1), SI
	VMOVUPD (SI), Y0

	LEAQ 2048(R8)(R14*1), DI
	VMOVUPD (DI), Y1

	LEAQ 4096(R8)(R14*1), DX
	VMOVUPD (DX), Y2

	VADDPD Y2, Y1, Y3        // t1
	VSUBPD Y2, Y1, Y4        // t2

	VADDPD Y3, Y0, Y5        // y0

	VMULPD Y8, Y3, Y6
	VADDPD Y6, Y0, Y6        // base

	// Swap t2
	VPERMPD $0xB1, Y4, Y7

	// Apply sign (Negate 0 and 2)
	VXORPD Y12, Y7, Y7

	// Mul by sqrt3/2
	VMULPD Y9, Y7, Y7

	VADDPD Y7, Y6, Y1        // y1
	VSUBPD Y7, Y6, Y2        // y2

	VMOVUPD Y5, (SI)
	VMOVUPD Y1, (DI)
	VMOVUPD Y2, (DX)

	ADDQ $2, CX
	JMP  radix3_384_inv_loop

radix3_384_inv_done:
	VZEROUPPER
	RET

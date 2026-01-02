//go:build amd64

// ===========================================================================
// AVX2 Complex Array Multiplication for AMD64
// ===========================================================================
//
// This file implements SIMD-accelerated element-wise complex multiplication:
//   dst[i] = a[i] * b[i]  (or dst[i] *= src[i] for in-place)
//
// COMPLEX MULTIPLICATION
// ----------------------
// For complex numbers a = ar + i*ai and b = br + i*bi:
//   (a * b).real = ar*br - ai*bi
//   (a * b).imag = ar*bi + ai*br
//
// Memory layout for complex64 in YMM (4 complex numbers, 32 bytes):
//   [a0.r, a0.i, a1.r, a1.i, a2.r, a2.i, a3.r, a3.i]
//    lane0 lane1 lane2 lane3 lane4 lane5 lane6 lane7
//
// AVX2 Strategy using FMA:
//   1. VMOVSLDUP: broadcast real parts [a.r, a.r, a.r, a.r, ...]
//   2. VMOVSHDUP: broadcast imag parts [a.i, a.i, a.i, a.i, ...]
//   3. Multiply a.r * [b.r, b.i, ...] -> [a.r*b.r, a.r*b.i, ...]
//   4. VSHUFPS 0xB1: swap b pairs -> [b.i, b.r, ...]
//   5. Multiply a.i * [b.i, b.r, ...] -> [a.i*b.i, a.i*b.r, ...]
//   6. VFMADDSUB231PS: result = a.r*b -/+ a.i*b_swap
//      Even lanes (real): subtract -> a.r*b.r - a.i*b.i ✓
//      Odd lanes (imag):  add      -> a.r*b.i + a.i*b.r ✓
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Go Calling Convention - Slice Layout
// ===========================================================================
// Each []T in Go ABI is: ptr (8 bytes) + len (8 bytes) + cap (8 bytes) = 24 bytes
//
// ===========================================================================
// complexMulArrayComplex128AVX2Asm - Element-wise complex128 multiplication
// ===========================================================================
// Computes: dst[i] = a[i] * b[i] for i = 0..n-1
//
// Memory layout for complex128 in YMM (2 complex numbers, 32 bytes):
//   [a0.r, a0.i, a1.r, a1.i]  (4 x float64)
//
// Complex128 uses double precision (float64), so each YMM holds 2 complex numbers.
// ===========================================================================
TEXT ·ComplexMulArrayComplex128AVX2Asm(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), DI       // DI = dst pointer
	MOVQ a+24(FP), SI        // SI = a pointer
	MOVQ b+48(FP), DX        // DX = b pointer
	MOVQ a+32(FP), CX        // CX = n = len(a)

	TESTQ CX, CX
	JZ    cmul128_done

	XORQ AX, AX              // AX = i = 0

cmul128_avx2_loop:
	MOVQ CX, R8
	SUBQ AX, R8
	CMPQ R8, $2              // Process 2 complex128 at a time
	JL   cmul128_scalar

	// Byte offset: i * 16 (complex128 = 16 bytes)
	MOVQ AX, R9
	SHLQ $4, R9              // R9 = i * 16

	// Load 2 complex128 values (32 bytes) from each array
	VMOVUPD (SI)(R9*1), Y0   // Y0 = a[i:i+2]
	VMOVUPD (DX)(R9*1), Y1   // Y1 = b[i:i+2]

	// Complex multiplication for float64
	// Y0 = [a0.r, a0.i, a1.r, a1.i]
	// Y1 = [b0.r, b0.i, b1.r, b1.i]

	// Broadcast a.real: duplicate even positions
	VMOVDDUP Y0, Y2          // Y2 = [a0.r, a0.r, a1.r, a1.r]

	// Broadcast a.imag: need to shuffle
	VPERMILPD $0x0F, Y0, Y3  // Y3 = [a0.i, a0.i, a1.i, a1.i]

	// Swap b pairs: [b.r, b.i] -> [b.i, b.r]
	VPERMILPD $0x05, Y1, Y4  // Y4 = [b0.i, b0.r, b1.i, b1.r]

	// Compute a.i * b_swapped
	VMULPD Y3, Y4, Y4        // Y4 = [a.i*b.i, a.i*b.r, ...]

	// FMA: result = a.r * b -/+ a.i * b_swap
	VFMADDSUB231PD Y2, Y1, Y4 // Y4 = a * b

	VMOVUPD Y4, (DI)(R9*1)   // dst[i:i+2] = result

	ADDQ $2, AX
	JMP  cmul128_avx2_loop

cmul128_scalar:
	CMPQ AX, CX
	JGE  cmul128_done

cmul128_scalar_loop:
	MOVQ AX, R9
	SHLQ $4, R9              // i * 16

	// Load single complex128 (16 bytes)
	VMOVUPD (SI)(R9*1), X0   // X0 = a[i]
	VMOVUPD (DX)(R9*1), X1   // X1 = b[i]

	// Complex multiply for single complex128
	VMOVDDUP X0, X2          // X2 = [a.r, a.r]
	VPERMILPD $0x03, X0, X3  // X3 = [a.i, a.i]
	VPERMILPD $0x01, X1, X4  // X4 = [b.i, b.r]
	VMULPD X3, X4, X4        // X4 = [a.i*b.i, a.i*b.r]
	VFMADDSUB231PD X2, X1, X4 // X4 = a * b

	VMOVUPD X4, (DI)(R9*1)

	INCQ AX
	CMPQ AX, CX
	JL   cmul128_scalar_loop

cmul128_done:
	VZEROUPPER
	RET

// ===========================================================================
// complexMulArrayInPlaceComplex128AVX2Asm - In-place complex128 multiplication
// ===========================================================================
TEXT ·ComplexMulArrayInPlaceComplex128AVX2Asm(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), DI       // DI = dst pointer
	MOVQ src+24(FP), SI      // SI = src pointer
	MOVQ dst+8(FP), CX       // CX = n = len(dst)

	TESTQ CX, CX
	JZ    cmul128ip_done

	XORQ AX, AX

cmul128ip_avx2_loop:
	MOVQ CX, R8
	SUBQ AX, R8
	CMPQ R8, $2
	JL   cmul128ip_scalar

	MOVQ AX, R9
	SHLQ $4, R9

	VMOVUPD (DI)(R9*1), Y0   // dst[i:i+2]
	VMOVUPD (SI)(R9*1), Y1   // src[i:i+2]

	VMOVDDUP Y0, Y2          // dst.r broadcast
	VPERMILPD $0x0F, Y0, Y3  // dst.i broadcast
	VPERMILPD $0x05, Y1, Y4  // src swapped
	VMULPD Y3, Y4, Y4
	VFMADDSUB231PD Y2, Y1, Y4

	VMOVUPD Y4, (DI)(R9*1)

	ADDQ $2, AX
	JMP  cmul128ip_avx2_loop

cmul128ip_scalar:
	CMPQ AX, CX
	JGE  cmul128ip_done

cmul128ip_scalar_loop:
	MOVQ AX, R9
	SHLQ $4, R9

	VMOVUPD (DI)(R9*1), X0   // dst[i]
	VMOVUPD (SI)(R9*1), X1   // src[i]

	VMOVDDUP X0, X2
	VPERMILPD $0x03, X0, X3
	VPERMILPD $0x01, X1, X4
	VMULPD X3, X4, X4
	VFMADDSUB231PD X2, X1, X4

	VMOVUPD X4, (DI)(R9*1)

	INCQ AX
	CMPQ AX, CX
	JL   cmul128ip_scalar_loop

cmul128ip_done:
	VZEROUPPER
	RET

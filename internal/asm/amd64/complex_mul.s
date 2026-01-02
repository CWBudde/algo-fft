//go:build amd64

// ===========================================================================
// AVX2/SSE2 Complex Array Multiplication for AMD64
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
// func complexMulArrayComplex64AVX2Asm(dst, a, b []complex64)
// Stack frame layout (offsets from FP):
//   dst: FP+0  (ptr), FP+8  (len), FP+16 (cap)
//   a:   FP+24 (ptr), FP+32 (len), FP+40 (cap)
//   b:   FP+48 (ptr), FP+56 (len), FP+64 (cap)

// ===========================================================================
// complexMulArrayComplex64AVX2Asm - Element-wise complex64 multiplication
// ===========================================================================
// Computes: dst[i] = a[i] * b[i] for i = 0..n-1
//
// Parameters:
//   dst []complex64 - Output buffer
//   a   []complex64 - First input (defines length n)
//   b   []complex64 - Second input
// ===========================================================================
TEXT ·ComplexMulArrayComplex64AVX2Asm(SB), NOSPLIT, $0-72
	MOVQ dst+0(FP), DI       // DI = dst pointer
	MOVQ a+24(FP), SI        // SI = a pointer
	MOVQ b+48(FP), DX        // DX = b pointer
	MOVQ a+32(FP), CX        // CX = n = len(a)

	// Empty input check
	TESTQ CX, CX
	JZ    cmul64_done

	XORQ AX, AX              // AX = i = 0

cmul64_avx2_loop:
	// Check if 4+ elements remain
	MOVQ CX, R8
	SUBQ AX, R8              // R8 = remaining = n - i
	CMPQ R8, $4
	JL   cmul64_scalar       // Less than 4, finish with scalar

	// Compute byte offset: i * 8 (complex64 = 8 bytes)
	MOVQ AX, R9
	SHLQ $3, R9              // R9 = i * 8

	// Load 4 complex64 values (32 bytes) from each array
	VMOVUPS (SI)(R9*1), Y0   // Y0 = a[i:i+4]
	VMOVUPS (DX)(R9*1), Y1   // Y1 = b[i:i+4]

	// Complex multiplication: dst = a * b
	// Y0 = [a0.r, a0.i, a1.r, a1.i, a2.r, a2.i, a3.r, a3.i]
	// Y1 = [b0.r, b0.i, b1.r, b1.i, b2.r, b2.i, b3.r, b3.i]

	// Step 1: Broadcast a.real parts
	VMOVSLDUP Y0, Y2         // Y2 = [a.r, a.r, a.r, a.r, ...]

	// Step 2: Broadcast a.imag parts
	VMOVSHDUP Y0, Y3         // Y3 = [a.i, a.i, a.i, a.i, ...]

	// Step 3: Swap b pairs for cross-term
	VSHUFPS $0xB1, Y1, Y1, Y4 // Y4 = [b.i, b.r, ...] (swap adjacent)

	// Step 4: Compute a.i * b_swapped
	VMULPS Y3, Y4, Y4        // Y4 = [a.i*b.i, a.i*b.r, ...]

	// Step 5: FMA: result = a.r * b -/+ a.i * b_swap
	// VFMADDSUB231PS: Y4 = Y2 * Y1 -/+ Y4
	//   Even lanes: Y2*Y1 - Y4 = a.r*b.r - a.i*b.i (real part ✓)
	//   Odd lanes:  Y2*Y1 + Y4 = a.r*b.i + a.i*b.r (imag part ✓)
	VFMADDSUB231PS Y2, Y1, Y4 // Y4 = a * b

	// Store result
	VMOVUPS Y4, (DI)(R9*1)   // dst[i:i+4] = result

	ADDQ $4, AX              // i += 4
	JMP  cmul64_avx2_loop

cmul64_scalar:
	// Handle remaining 0-3 elements with scalar SSE
	CMPQ AX, CX
	JGE  cmul64_done

cmul64_scalar_loop:
	// Compute byte offset
	MOVQ AX, R9
	SHLQ $3, R9              // R9 = i * 8

	// Load single complex64 (8 bytes)
	MOVSD (SI)(R9*1), X0     // X0 = a[i]
	MOVSD (DX)(R9*1), X1     // X1 = b[i]

	// Complex multiply using SSE
	MOVSLDUP X0, X2          // X2 = [a.r, a.r]
	MOVSHDUP X0, X3          // X3 = [a.i, a.i]
	MOVAPS X1, X4
	SHUFPS $0xB1, X4, X4     // X4 = [b.i, b.r]
	MULPS X3, X4             // X4 = [a.i*b.i, a.i*b.r]

	// We need: [a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r]
	// SSE3 ADDSUBPS does: even lanes sub, odd lanes add
	MULPS X2, X1             // X1 = [a.r*b.r, a.r*b.i]
	ADDSUBPS X4, X1          // X1 = [a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r]

	// Store result
	MOVSD X1, (DI)(R9*1)     // dst[i] = result

	INCQ AX                  // i++
	CMPQ AX, CX
	JL   cmul64_scalar_loop

cmul64_done:
	VZEROUPPER
	RET

// ===========================================================================
// complexMulArrayInPlaceComplex64AVX2Asm - In-place complex64 multiplication
// ===========================================================================
// Computes: dst[i] *= src[i] for i = 0..n-1
//
// Parameters:
//   dst []complex64 - Buffer to multiply in-place
//   src []complex64 - Multiplier values
// ===========================================================================
TEXT ·ComplexMulArrayInPlaceComplex64AVX2Asm(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), DI       // DI = dst pointer
	MOVQ src+24(FP), SI      // SI = src pointer
	MOVQ dst+8(FP), CX       // CX = n = len(dst)

	TESTQ CX, CX
	JZ    cmul64ip_done

	XORQ AX, AX              // AX = i = 0

cmul64ip_avx2_loop:
	MOVQ CX, R8
	SUBQ AX, R8
	CMPQ R8, $4
	JL   cmul64ip_scalar

	MOVQ AX, R9
	SHLQ $3, R9

	// Load dst and src
	VMOVUPS (DI)(R9*1), Y0   // Y0 = dst[i:i+4]
	VMOVUPS (SI)(R9*1), Y1   // Y1 = src[i:i+4]

	// Complex multiply: dst = dst * src
	VMOVSLDUP Y0, Y2         // Y2 = dst.r broadcast
	VMOVSHDUP Y0, Y3         // Y3 = dst.i broadcast
	VSHUFPS $0xB1, Y1, Y1, Y4 // Y4 = src swapped
	VMULPS Y3, Y4, Y4        // Y4 = dst.i * src_swap
	VFMADDSUB231PS Y2, Y1, Y4 // Y4 = dst * src

	VMOVUPS Y4, (DI)(R9*1)   // dst[i:i+4] = result

	ADDQ $4, AX
	JMP  cmul64ip_avx2_loop

cmul64ip_scalar:
	CMPQ AX, CX
	JGE  cmul64ip_done

cmul64ip_scalar_loop:
	MOVQ AX, R9
	SHLQ $3, R9

	MOVSD (DI)(R9*1), X0     // dst[i]
	MOVSD (SI)(R9*1), X1     // src[i]

	MOVSLDUP X0, X2
	MOVSHDUP X0, X3
	MOVAPS X1, X4
	SHUFPS $0xB1, X4, X4
	MULPS X3, X4
	MULPS X2, X1
	ADDSUBPS X4, X1

	MOVSD X1, (DI)(R9*1)

	INCQ AX
	CMPQ AX, CX
	JL   cmul64ip_scalar_loop

cmul64ip_done:
	VZEROUPPER
	RET

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

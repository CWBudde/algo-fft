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
	MOVQ dst+0(FP), DI       // DI = dst pointer (slice data ptr)
	MOVQ a+24(FP), SI        // SI = a pointer (slice data ptr)
	MOVQ b+48(FP), DX        // DX = b pointer (slice data ptr)
	MOVQ a+32(FP), CX        // CX = n = len(a) (number of complex64 elements)

	// Empty input check
	TESTQ CX, CX             // Test if n == 0
	JZ    cmul64_done        // Jump to done if empty

	XORQ AX, AX              // AX = i = 0 (loop counter initialization)

cmul64_avx2_loop:
	// Check if 4+ elements remain
	MOVQ CX, R8              // Copy n to R8
	SUBQ AX, R8              // R8 = n - i (remaining elements)
	CMPQ R8, $4              // Compare remaining with 4
	JL   cmul64_scalar       // Jump to scalar if < 4 elements remain

	// Compute byte offset: i * 8 (complex64 = 8 bytes)
	MOVQ AX, R9              // Copy index to R9
	SHLQ $3, R9              // R9 = i * 8 (shift left by 3 = multiply by 8)

	// Load 4 complex64 values (32 bytes) from each array
	VMOVUPS (SI)(R9*1), Y0   // Y0 = a[i:i+4] (load 32 bytes unaligned)
	VMOVUPS (DX)(R9*1), Y1   // Y1 = b[i:i+4] (load 32 bytes unaligned)

	// Complex multiplication: dst = a * b
	// Y0 = [a0.r, a0.i, a1.r, a1.i, a2.r, a2.i, a3.r, a3.i]
	// Y1 = [b0.r, b0.i, b1.r, b1.i, b2.r, b2.i, b3.r, b3.i]

	// Step 1: Broadcast a.real parts
	VMOVSLDUP Y0, Y2         // Y2 = [a0.r, a0.r, a1.r, a1.r, ...] - Duplicate low (real) parts

	// Step 2: Broadcast a.imag parts
	VMOVSHDUP Y0, Y3         // Y3 = [a0.i, a0.i, a1.i, a1.i, ...] - Duplicate high (imag) parts

	// Step 3: Swap b pairs for cross-term
	VSHUFPS $0xB1, Y1, Y1, Y4 // Y4 = [b0.i, b0.r, b1.i, b1.r, ...] - Swap adjacent pairs (0xB1)

	// Step 4: Compute a.i * b_swapped
	VMULPS Y3, Y4, Y4        // Y4 = [a0.i*b0.i, a0.i*b0.r, a1.i*b1.i, a1.i*b1.r, ...]

	// Step 5: FMA: result = a.r * b -/+ a.i * b_swap
	// VFMADDSUB231PS: Y4 = Y2 * Y1 -/+ Y4
	//   Even lanes (0,2,4,6): Y2*Y1 - Y4 = a.r*b.r - a.i*b.i (real part ✓)
	//   Odd lanes  (1,3,5,7): Y2*Y1 + Y4 = a.r*b.i + a.i*b.r (imag part ✓)
	VFMADDSUB231PS Y2, Y1, Y4 // Y4 = a * b (fused multiply-add/sub)

	// Store result
	VMOVUPS Y4, (DI)(R9*1)   // dst[i:i+4] = result (store 32 bytes unaligned)

	ADDQ $4, AX              // i += 4 (increment by 4 elements)
	JMP  cmul64_avx2_loop    // Continue AVX2 loop

cmul64_scalar:
	// Handle remaining 0-3 elements with scalar SSE
	CMPQ AX, CX              // Compare i with n
	JGE  cmul64_done         // Jump to done if i >= n (no remaining elements)

cmul64_scalar_loop:
	// Compute byte offset
	MOVQ AX, R9              // Copy index to R9
	SHLQ $3, R9              // R9 = i * 8 (shift left by 3 = multiply by 8)

	// Load single complex64 (8 bytes)
	MOVSD (SI)(R9*1), X0     // X0 = a[i] (load 8 bytes)
	MOVSD (DX)(R9*1), X1     // X1 = b[i] (load 8 bytes)

	// Complex multiply using SSE
	MOVSLDUP X0, X2          // X2 = [a.r, a.r] - Duplicate low (real) parts
	MOVSHDUP X0, X3          // X3 = [a.i, a.i] - Duplicate high (imag) parts
	MOVAPS X1, X4            // Copy b to X4
	SHUFPS $0xB1, X4, X4     // X4 = [b.i, b.r] - Swap adjacent pairs (0xB1 = 10110001)
	MULPS X3, X4             // X4 = a.i * [b.i, b.r] = [a.i*b.i, a.i*b.r]

	// We need: [a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r]
	// SSE3 ADDSUBPS does: even lanes sub, odd lanes add
	MULPS X2, X1             // X1 = a.r * [b.r, b.i] = [a.r*b.r, a.r*b.i]
	ADDSUBPS X4, X1          // X1 = X1 -/+ X4 = [a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r]

	// Store result
	MOVSD X1, (DI)(R9*1)     // dst[i] = result (store 8 bytes)

	INCQ AX                  // i++ (increment element counter)
	CMPQ AX, CX              // Compare i with n
	JL   cmul64_scalar_loop  // Loop if i < n

cmul64_done:
	VZEROUPPER               // Clear upper 128 bits of YMM registers (AVX-SSE transition)
	RET                      // Return to caller

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
	MOVQ dst+0(FP), DI       // DI = dst pointer (slice data ptr)
	MOVQ src+24(FP), SI      // SI = src pointer (slice data ptr)
	MOVQ dst+8(FP), CX       // CX = n = len(dst) (number of complex64 elements)

	TESTQ CX, CX             // Test if n == 0
	JZ    cmul64ip_done      // Jump to done if empty

	XORQ AX, AX              // AX = i = 0 (loop counter initialization)

cmul64ip_avx2_loop:
	MOVQ CX, R8              // Copy n to R8
	SUBQ AX, R8              // R8 = n - i (remaining elements)
	CMPQ R8, $4              // Compare remaining with 4
	JL   cmul64ip_scalar     // Jump to scalar if < 4 elements remain

	MOVQ AX, R9              // Copy index to R9
	SHLQ $3, R9              // R9 = i * 8 (byte offset)

	// Load dst and src (4 complex64 values = 32 bytes each)
	VMOVUPS (DI)(R9*1), Y0   // Y0 = dst[i:i+4] (load 32 bytes unaligned)
	VMOVUPS (SI)(R9*1), Y1   // Y1 = src[i:i+4] (load 32 bytes unaligned)

	// Complex multiply: dst = dst * src
	VMOVSLDUP Y0, Y2         // Y2 = [dst.r, dst.r, ...] - Duplicate real parts
	VMOVSHDUP Y0, Y3         // Y3 = [dst.i, dst.i, ...] - Duplicate imag parts
	VSHUFPS $0xB1, Y1, Y1, Y4 // Y4 = [src.i, src.r, ...] - Swap adjacent pairs
	VMULPS Y3, Y4, Y4        // Y4 = dst.i * src_swapped
	VFMADDSUB231PS Y2, Y1, Y4 // Y4 = dst.r*src -/+ dst.i*src_swap (FMA)

	VMOVUPS Y4, (DI)(R9*1)   // dst[i:i+4] = result (store 32 bytes unaligned)

	ADDQ $4, AX              // i += 4 (increment by 4 elements)
	JMP  cmul64ip_avx2_loop  // Continue AVX2 loop

cmul64ip_scalar:
	CMPQ AX, CX              // Compare i with n
	JGE  cmul64ip_done       // Jump to done if i >= n (no remaining elements)

cmul64ip_scalar_loop:
	MOVQ AX, R9              // Copy index to R9
	SHLQ $3, R9              // R9 = i * 8 (byte offset)

	MOVSD (DI)(R9*1), X0     // X0 = dst[i] (load 8 bytes)
	MOVSD (SI)(R9*1), X1     // X1 = src[i] (load 8 bytes)

	// Complex multiply: dst = dst * src
	MOVSLDUP X0, X2          // X2 = [dst.r, dst.r] - Duplicate low (real) parts
	MOVSHDUP X0, X3          // X3 = [dst.i, dst.i] - Duplicate high (imag) parts
	MOVAPS X1, X4            // Copy src to X4
	SHUFPS $0xB1, X4, X4     // X4 = [src.i, src.r] - Swap adjacent pairs
	MULPS X3, X4             // X4 = dst.i * [src.i, src.r] = [dst.i*src.i, dst.i*src.r]
	MULPS X2, X1             // X1 = dst.r * [src.r, src.i] = [dst.r*src.r, dst.r*src.i]
	ADDSUBPS X4, X1          // X1 = X1 -/+ X4 = [dst.r*src.r - dst.i*src.i, dst.r*src.i + dst.i*src.r]

	MOVSD X1, (DI)(R9*1)     // dst[i] = result (store 8 bytes)

	INCQ AX                  // i++ (increment element counter)
	CMPQ AX, CX              // Compare i with n
	JL   cmul64ip_scalar_loop // Loop if i < n

cmul64ip_done:
	VZEROUPPER               // Clear upper 128 bits of YMM registers (AVX-SSE transition)
	RET                      // Return to caller

//go:build amd64 && fft_asm && !purego

// ===========================================================================
// AMD64 FFT Assembly - Core Utilities and Constants
// ===========================================================================
//
// This file contains shared utilities, constants, and small helpers used by
// the AVX2 and SSE2 FFT implementations.
//
// See asm_amd64_avx2_generic.s for the main FFT algorithm documentation.
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Go Calling Convention - Slice Layout
// ===========================================================================
// Each []T in Go ABI is: ptr (8 bytes) + len (8 bytes) + cap (8 bytes) = 24 bytes
//
// Function signature:
//   func forwardAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
//
// Stack frame layout (offsets from FP):
//   dst:     FP+0   (ptr), FP+8   (len), FP+16  (cap)
//   src:     FP+24  (ptr), FP+32  (len), FP+40  (cap)
//   twiddle: FP+48  (ptr), FP+56  (len), FP+64  (cap)
//   scratch: FP+72  (ptr), FP+80  (len), FP+88  (cap)
//   bitrev:  FP+96  (ptr), FP+104 (len), FP+112 (cap)
//   return:  FP+120 (bool, 1 byte)

// ===========================================================================
// Data Type Sizes
// ===========================================================================
// complex64:  8 bytes  = 4 bytes (float32 real) + 4 bytes (float32 imag)
// complex128: 16 bytes = 8 bytes (float64 real) + 8 bytes (float64 imag)
//
// YMM register (256 bits = 32 bytes):
//   - Holds 4 complex64  values (4 × 8  = 32 bytes)
//   - Holds 2 complex128 values (2 × 16 = 32 bytes)
//
// XMM register (128 bits = 16 bytes):
//   - Holds 2 complex64  values (2 × 8  = 16 bytes)
//   - Holds 1 complex128 value  (1 × 16 = 16 bytes)

// ===========================================================================
// UTILITY FUNCTIONS: Small helpers for specific FFT operations
// ===========================================================================

// asmCopyComplex64 - Copy a single complex64 (8 bytes)
// Parameters: AX = dst ptr, BX = src ptr (passed via registers, not stack)
TEXT ·asmCopyComplex64(SB), NOSPLIT|NOFRAME, $0-16
	MOVQ (BX), CX               // Load 8 bytes (real + imag as single QWORD)
	MOVQ CX, (AX)               // Store to destination
	RET

// ===========================================================================
// CONSTANTS: Single-precision floating-point literals
// ===========================================================================

// half32: Single-precision 0.5 for complex64 inverse scaling (n=2 case)
DATA ·half32+0(SB)/4, $0x3f000000  // 0.5f (IEEE 754 single)
GLOBL ·half32(SB), RODATA|NOPTR, $4

// one32: Single-precision 1.0 (reserved for future use)
DATA ·one32+0(SB)/4, $0x3f800000   // 1.0f (IEEE 754 single)
GLOBL ·one32(SB), RODATA|NOPTR, $4

// ===========================================================================
// asmForward2Complex64 - Specialized size-2 forward FFT for complex64
// ===========================================================================
// Computes the 2-point DFT:
//   X[0] = x[0] + x[1]  (DC component)
//   X[1] = x[0] - x[1]  (Nyquist component)
//
// For n=2, twiddle factor w = e^(-2πi/2) = -1, so:
//   X[k] = sum(x[n] * w^(nk)) gives X[0] = x[0] + x[1], X[1] = x[0] - x[1]
//
// Parameters: AX = dst ptr, BX = src ptr
// Returns: 1 in AX (success)
// ===========================================================================
TEXT ·asmForward2Complex64(SB), NOSPLIT|NOFRAME, $0-16
	// Load both complex64 values component-by-component
	MOVSS (BX), X0              // X0 = x[0].real
	MOVSS 4(BX), X1             // X1 = x[0].imag
	MOVSS 8(BX), X2             // X2 = x[1].real
	MOVSS 12(BX), X3            // X3 = x[1].imag

	// X[0] = x[0] + x[1] (DC component)
	MOVSS X0, X4
	ADDSS X2, X4                // X4 = x[0].real + x[1].real
	MOVSS X1, X5
	ADDSS X3, X5                // X5 = x[0].imag + x[1].imag

	MOVSS X4, (AX)              // Store X[0].real
	MOVSS X5, 4(AX)             // Store X[0].imag

	// X[1] = x[0] - x[1] (Nyquist component)
	MOVSS X0, X6
	SUBSS X2, X6                // X6 = x[0].real - x[1].real
	MOVSS X1, X7
	SUBSS X3, X7                // X7 = x[0].imag - x[1].imag

	MOVSS X6, 8(AX)             // Store X[1].real
	MOVSS X7, 12(AX)            // Store X[1].imag

	MOVL $1, AX                 // Return success
	RET

// ===========================================================================
// asmInverse2Complex64 - Specialized size-2 inverse FFT for complex64
// ===========================================================================
// Computes the 2-point inverse DFT with 1/n normalization:
//   x[0] = (X[0] + X[1]) / 2
//   x[1] = (X[0] - X[1]) / 2
//
// Parameters: AX = dst ptr, BX = src ptr
// Returns: 1 in AX (success)
// ===========================================================================
TEXT ·asmInverse2Complex64(SB), NOSPLIT|NOFRAME, $0-16
	// Load both complex64 values
	MOVSS (BX), X0              // X0 = X[0].real
	MOVSS 4(BX), X1             // X1 = X[0].imag
	MOVSS 8(BX), X2             // X2 = X[1].real
	MOVSS 12(BX), X3            // X3 = X[1].imag

	// x[0] = (X[0] + X[1]) / 2
	MOVSS X0, X4
	ADDSS X2, X4                // X4 = X[0].real + X[1].real
	MOVSS X1, X5
	ADDSS X3, X5                // X5 = X[0].imag + X[1].imag

	MULSS ·half32(SB), X4       // X4 = (X[0].real + X[1].real) * 0.5
	MULSS ·half32(SB), X5       // X5 = (X[0].imag + X[1].imag) * 0.5

	MOVSS X4, (AX)              // Store x[0].real
	MOVSS X5, 4(AX)             // Store x[0].imag

	// x[1] = (X[0] - X[1]) / 2
	MOVSS X0, X6
	SUBSS X2, X6                // X6 = X[0].real - X[1].real
	MOVSS X1, X7
	SUBSS X3, X7                // X7 = X[0].imag - X[1].imag

	MULSS ·half32(SB), X6       // X6 = (X[0].real - X[1].real) * 0.5
	MULSS ·half32(SB), X7       // X7 = (X[0].imag - X[1].imag) * 0.5

	MOVSS X6, 8(AX)             // Store x[1].real
	MOVSS X7, 12(AX)            // Store x[1].imag

	MOVL $1, AX                 // Return success
	RET

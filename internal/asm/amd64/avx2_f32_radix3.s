//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Radix-3 Butterfly (complex64) for AMD64
// ===========================================================================
//
// This file implements AVX2-optimized radix-3 butterfly operations.
// The radix-3 butterfly computes:
//   y0 = a0 + a1 + a2
//   y1 = a0 + half*(a1+a2) + coef*(a1-a2)
//   y2 = a0 + half*(a1+a2) - coef*(a1-a2)
//
// where:
//   half = -0.5 + 0i
//   coef = 0 - i*sqrt(3)/2  (forward)
//   coef = 0 + i*sqrt(3)/2  (inverse)
//
// This implementation processes 4 radix-3 butterflies in parallel using AVX2.
// Each butterfly takes 3 complex64 inputs and produces 3 complex64 outputs.
// With 4 parallel butterflies: 12 complex64 inputs → 12 complex64 outputs.
//
// ===========================================================================

#include "textflag.h"

// Constants for radix-3 butterfly
DATA radix3_half<>+0x00(SB)/4, $0xBF000000  // -0.5 (real)
DATA radix3_half<>+0x04(SB)/4, $0xBF000000  // -0.5 (imag)
DATA radix3_half<>+0x08(SB)/4, $0xBF000000  // -0.5 (real)
DATA radix3_half<>+0x0C(SB)/4, $0xBF000000  // -0.5 (imag)
DATA radix3_half<>+0x10(SB)/4, $0xBF000000  // -0.5 (real)
DATA radix3_half<>+0x14(SB)/4, $0xBF000000  // -0.5 (imag)
DATA radix3_half<>+0x18(SB)/4, $0xBF000000  // -0.5 (real)
DATA radix3_half<>+0x1C(SB)/4, $0xBF000000  // -0.5 (imag)
GLOBL radix3_half<>(SB), RODATA|NOPTR, $32

// Forward coefficient: 0 - i*sqrt(3)/2 = 0 - 0.866025403784i
DATA radix3_coef_fwd<>+0x00(SB)/4, $0x00000000  //  0.0 (real)
DATA radix3_coef_fwd<>+0x04(SB)/4, $0xBF5DB3D7  // -0.866025403784 (imag)
DATA radix3_coef_fwd<>+0x08(SB)/4, $0x00000000  //  0.0 (real)
DATA radix3_coef_fwd<>+0x0C(SB)/4, $0xBF5DB3D7  // -0.866025403784 (imag)
DATA radix3_coef_fwd<>+0x10(SB)/4, $0x00000000  //  0.0 (real)
DATA radix3_coef_fwd<>+0x14(SB)/4, $0xBF5DB3D7  // -0.866025403784 (imag)
DATA radix3_coef_fwd<>+0x18(SB)/4, $0x00000000  //  0.0 (real)
DATA radix3_coef_fwd<>+0x1C(SB)/4, $0xBF5DB3D7  // -0.866025403784 (imag)
GLOBL radix3_coef_fwd<>(SB), RODATA|NOPTR, $32

// Inverse coefficient: 0 + i*sqrt(3)/2 = 0 + 0.866025403784i
DATA radix3_coef_inv<>+0x00(SB)/4, $0x00000000  //  0.0 (real)
DATA radix3_coef_inv<>+0x04(SB)/4, $0x3F5DB3D7  //  0.866025403784 (imag)
DATA radix3_coef_inv<>+0x08(SB)/4, $0x00000000  //  0.0 (real)
DATA radix3_coef_inv<>+0x0C(SB)/4, $0x3F5DB3D7  //  0.866025403784 (imag)
DATA radix3_coef_inv<>+0x10(SB)/4, $0x00000000  //  0.0 (real)
DATA radix3_coef_inv<>+0x14(SB)/4, $0x3F5DB3D7  //  0.866025403784 (imag)
DATA radix3_coef_inv<>+0x18(SB)/4, $0x00000000  //  0.0 (real)
DATA radix3_coef_inv<>+0x1C(SB)/4, $0x3F5DB3D7  //  0.866025403784 (imag)
GLOBL radix3_coef_inv<>(SB), RODATA|NOPTR, $32

// Shuffle masks for complex multiplication
// To multiply (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
// We need to permute and negate values

// Mask to negate imaginary parts (positions 1,3,5,7): [0, -1, 0, -1, ...]
DATA neg_imag_mask<>+0x00(SB)/4, $0x00000000  // 0 (sign bit = 0)
DATA neg_imag_mask<>+0x04(SB)/4, $0x80000000  // -1 (sign bit = 1)
DATA neg_imag_mask<>+0x08(SB)/4, $0x00000000  // 0
DATA neg_imag_mask<>+0x0C(SB)/4, $0x80000000  // -1
DATA neg_imag_mask<>+0x10(SB)/4, $0x00000000  // 0
DATA neg_imag_mask<>+0x14(SB)/4, $0x80000000  // -1
DATA neg_imag_mask<>+0x18(SB)/4, $0x00000000  // 0
DATA neg_imag_mask<>+0x1C(SB)/4, $0x80000000  // -1
GLOBL neg_imag_mask<>(SB), RODATA|NOPTR, $32

// Mask to negate real parts (positions 0,2,4,6): [-1, 0, -1, 0, ...]
DATA neg_real_mask<>+0x00(SB)/4, $0x80000000  // -1 (sign bit = 1)
DATA neg_real_mask<>+0x04(SB)/4, $0x00000000  // 0
DATA neg_real_mask<>+0x08(SB)/4, $0x80000000  // -1
DATA neg_real_mask<>+0x0C(SB)/4, $0x00000000  // 0
DATA neg_real_mask<>+0x10(SB)/4, $0x80000000  // -1
DATA neg_real_mask<>+0x14(SB)/4, $0x00000000  // 0
DATA neg_real_mask<>+0x18(SB)/4, $0x80000000  // -1
DATA neg_real_mask<>+0x1C(SB)/4, $0x00000000  // 0
GLOBL neg_real_mask<>(SB), RODATA|NOPTR, $32

// ===========================================================================
// Forward Butterfly - processes 4 radix-3 butterflies in parallel
// ===========================================================================
// Input:  Y0, Y1, Y2 contain 4 complex64 values each (a0[0..3], a1[0..3], a2[0..3])
// Output: Y0, Y1, Y2 contain 4 complex64 results (y0[0..3], y1[0..3], y2[0..3])
//
// Butterfly3 forward:
//   t1 = a1 + a2
//   t2 = a1 - a2
//   y0 = a0 + t1
//   base = a0 + half*t1  (half = -0.5)
//   y1 = base + coef*t2  (coef = 0 - i*sqrt(3)/2)
//   y2 = base - coef*t2
//
// Note: Complex multiplication by coef = 0 - i*sqrt(3)/2:
//   (a+bi) * (0 - ci) where c = sqrt(3)/2
//   = (a*0 - b*(-c)) + i*(a*(-c) + b*0)
//   = b*c - i*a*c
//   = c*(b - i*a)
//   So: swap real/imag of t2, negate imag, multiply by sqrt(3)/2
// ===========================================================================

// Macro: BUTTERFLY3_FORWARD_AVX2
// Computes 4 parallel radix-3 forward butterflies
// Input:  Y0=a0, Y1=a1, Y2=a2 (each has 4 complex64 values)
// Output: Y0=y0, Y1=y1, Y2=y2
// Clobbers: Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11
#define BUTTERFLY3_FORWARD_AVX2() \
	/* Load constants */ \
	VMOVUPS radix3_half<>(SB), Y8        /* Y8 = [-0.5+0i, ...] */ \
	VMOVUPS neg_imag_mask<>(SB), Y11     /* Y11 = sign flip mask for imaginary */ \
	\
	/* Load sqrt(3)/2 coefficient - need to broadcast */ \
	MOVL $0x3F5DB3D7, AX                 /* AX = sqrt(3)/2 bit pattern */ \
	MOVD AX, X9                          /* Move to XMM */ \
	VBROADCASTSS X9, Y9                  /* Broadcast to all lanes of YMM */ \
	\
	/* t1 = a1 + a2 */ \
	VADDPS Y2, Y1, Y3                    /* Y3 = t1 = a1 + a2 */ \
	\
	/* t2 = a1 - a2 */ \
	VSUBPS Y2, Y1, Y4                    /* Y4 = t2 = a1 - a2 */ \
	\
	/* y0 = a0 + t1 */ \
	VADDPS Y3, Y0, Y5                    /* Y5 = y0 = a0 + t1 */ \
	\
	/* base = a0 + half*t1 */ \
	VMULPS Y8, Y3, Y6                    /* Y6 = half*t1 = -0.5*t1 */ \
	VADDPS Y6, Y0, Y6                    /* Y6 = base = a0 + half*t1 */ \
	\
	/* Complex multiply: coef * t2 = (0 - i*sqrt(3)/2) * t2 */ \
	/* Result = sqrt(3)/2 * (imag(t2) - i*real(t2)) */ \
	VSHUFPS $0xB1, Y4, Y4, Y7            /* Y7 = [imag, real, imag, real, ...] (swap adjacent pairs) */ \
	VXORPS Y11, Y7, Y7                   /* Y7 = [imag, -real, ...] */ \
	VMULPS Y9, Y7, Y7                    /* Y7 = sqrt(3)/2 * [imag, -real] = coef*t2 */ \
	\
	/* y1 = base + coef*t2 */ \
	VADDPS Y7, Y6, Y1                    /* Y1 = y1 = base + coef*t2 */ \
	\
	/* y2 = base - coef*t2 */ \
	VSUBPS Y7, Y6, Y2                    /* Y2 = y2 = base - coef*t2 */ \
	\
	/* Store y0 back to Y0 */ \
	VMOVUPS Y5, Y0                       /* Y0 = y0 */

// ===========================================================================
// Inverse Butterfly - processes 4 radix-3 butterflies in parallel
// ===========================================================================
// Same as forward, but uses conjugate coefficient
// coef = 0 + i*sqrt(3)/2
// (a+bi) * (0 + ci) = -bc + i*ac = c*(-b + i*a)
// So: swap, negate real, multiply by sqrt(3)/2

#define BUTTERFLY3_INVERSE_AVX2() \
	/* Load constants */ \
	VMOVUPS radix3_half<>(SB), Y8        /* Y8 = [-0.5+0i, ...] */ \
	VMOVUPS neg_real_mask<>(SB), Y11     /* Y11 = negate real parts mask */ \
	\
	/* Load sqrt(3)/2 coefficient */ \
	MOVL $0x3F5DB3D7, AX                 /* AX = sqrt(3)/2 bit pattern */ \
	MOVD AX, X9                          /* Move to XMM */ \
	VBROADCASTSS X9, Y9                  /* Broadcast to YMM */ \
	\
	/* t1 = a1 + a2 */ \
	VADDPS Y2, Y1, Y3                    /* Y3 = t1 */ \
	\
	/* t2 = a1 - a2 */ \
	VSUBPS Y2, Y1, Y4                    /* Y4 = t2 */ \
	\
	/* y0 = a0 + t1 */ \
	VADDPS Y3, Y0, Y5                    /* Y5 = y0 */ \
	\
	/* base = a0 + half*t1 */ \
	VMULPS Y8, Y3, Y6                    /* Y6 = half*t1 */ \
	VADDPS Y6, Y0, Y6                    /* Y6 = base */ \
	\
	/* Complex multiply: coef * t2 = (0 + i*sqrt(3)/2) * t2 */ \
	/* Result = sqrt(3)/2 * (-imag(t2) + i*real(t2)) */ \
	VSHUFPS $0xB1, Y4, Y4, Y7            /* Y7 = [imag, real, imag, real, ...] (swap adjacent pairs) */ \
	VXORPS Y11, Y7, Y7                   /* Y7 = [-imag, real, -imag, real, ...] */ \
	VMULPS Y9, Y7, Y7                    /* Y7 = coef*t2 */ \
	\
	/* y1 = base + coef*t2 */ \
	VADDPS Y7, Y6, Y1                    /* Y1 = y1 */ \
	\
	/* y2 = base - coef*t2 */ \
	VSUBPS Y7, Y6, Y2                    /* Y2 = y2 */ \
	\
	/* Store y0 */ \
	VMOVUPS Y5, Y0                       /* Y0 = y0 */

// ===========================================================================
// Function: Butterfly3ForwardAVX2Complex64
// ===========================================================================
// Processes 4 radix-3 forward butterflies in parallel
// Input: a0[0:3], a1[0:3], a2[0:3] (12 complex64 = 96 bytes)
// Output: y0[0:3], y1[0:3], y2[0:3] (12 complex64 = 96 bytes)
//
// func Butterfly3ForwardAVX2Complex64(y0, y1, y2, a0, a1, a2 []complex64)
TEXT ·Butterfly3ForwardAVX2Complex64(SB), NOSPLIT, $0-144
	// Load input pointers
	MOVQ y0+0(FP), R8   // R8 = y0 pointer
	MOVQ y1+24(FP), R9  // R9 = y1 pointer
	MOVQ y2+48(FP), R10 // R10 = y2 pointer
	MOVQ a0+72(FP), R11 // R11 = a0 pointer
	MOVQ a1+96(FP), R12 // R12 = a1 pointer
	MOVQ a2+120(FP), R13 // R13 = a2 pointer

	// Verify all slices have length >= 4
	MOVQ y0+8(FP), AX
	CMPQ AX, $4
	JL   butterfly3_fwd_return

	MOVQ y1+32(FP), AX
	CMPQ AX, $4
	JL   butterfly3_fwd_return

	MOVQ y2+56(FP), AX
	CMPQ AX, $4
	JL   butterfly3_fwd_return

	MOVQ a0+80(FP), AX
	CMPQ AX, $4
	JL   butterfly3_fwd_return

	MOVQ a1+104(FP), AX
	CMPQ AX, $4
	JL   butterfly3_fwd_return

	MOVQ a2+128(FP), AX
	CMPQ AX, $4
	JL   butterfly3_fwd_return

	// Load inputs: 4 complex64 = 32 bytes per register
	VMOVUPS (R11), Y0  // Y0 = a0[0:3] (4 complex64)
	VMOVUPS (R12), Y1  // Y1 = a1[0:3]
	VMOVUPS (R13), Y2  // Y2 = a2[0:3]

	// Perform butterfly
	BUTTERFLY3_FORWARD_AVX2()

	// Store results
	VMOVUPS Y0, (R8)   // y0[0:3]
	VMOVUPS Y1, (R9)   // y1[0:3]
	VMOVUPS Y2, (R10)  // y2[0:3]

butterfly3_fwd_return:
	RET

// ===========================================================================
// Function: Butterfly3InverseAVX2Complex64
// ===========================================================================
// Processes 4 radix-3 inverse butterflies in parallel
//
// func Butterfly3InverseAVX2Complex64(y0, y1, y2, a0, a1, a2 []complex64)
TEXT ·Butterfly3InverseAVX2Complex64(SB), NOSPLIT, $0-144
	// Load input pointers
	MOVQ y0+0(FP), R8
	MOVQ y1+24(FP), R9
	MOVQ y2+48(FP), R10
	MOVQ a0+72(FP), R11
	MOVQ a1+96(FP), R12
	MOVQ a2+120(FP), R13

	// Verify all slices have length >= 4
	MOVQ y0+8(FP), AX
	CMPQ AX, $4
	JL   butterfly3_inv_return

	MOVQ y1+32(FP), AX
	CMPQ AX, $4
	JL   butterfly3_inv_return

	MOVQ y2+56(FP), AX
	CMPQ AX, $4
	JL   butterfly3_inv_return

	MOVQ a0+80(FP), AX
	CMPQ AX, $4
	JL   butterfly3_inv_return

	MOVQ a1+104(FP), AX
	CMPQ AX, $4
	JL   butterfly3_inv_return

	MOVQ a2+128(FP), AX
	CMPQ AX, $4
	JL   butterfly3_inv_return

	// Load inputs
	VMOVUPS (R11), Y0
	VMOVUPS (R12), Y1
	VMOVUPS (R13), Y2

	// Perform butterfly
	BUTTERFLY3_INVERSE_AVX2()

	// Store results
	VMOVUPS Y0, (R8)
	VMOVUPS Y1, (R9)
	VMOVUPS Y2, (R10)

butterfly3_inv_return:
	RET

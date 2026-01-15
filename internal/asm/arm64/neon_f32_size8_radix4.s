//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-8 Mixed-Radix (Radix-4 + Radix-2) FFT Kernels for ARM64
// ===========================================================================
// Decomposition: 8 = 4 × 2
// Stage 1: Two radix-4 butterflies on [0,2,4,6] and [1,3,5,7]
// Stage 2: Four radix-2 butterflies combining pairs with twiddle factors

#include "textflag.h"

// Forward transform, size 8, mixed radix-4/radix-2
TEXT ·ForwardNEONSize8Radix4Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $8, R13
	BNE  neon8r4_return_false

	MOVD dst+8(FP), R0
	CMP  $8, R0
	BLT  neon8r4_return_false

	MOVD twiddle+56(FP), R0
	CMP  $8, R0
	BLT  neon8r4_return_false

	MOVD scratch+80(FP), R0
	CMP  $8, R0
	BLT  neon8r4_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon8r4_use_dst
	MOVD R11, R8

neon8r4_use_dst:
	// Load all 8 complex values from src
	// x[0..7] at offsets 0,8,16,24,32,40,48,56
	FMOVS 0(R9), F0        // x0.re
	FMOVS 4(R9), F1        // x0.im
	FMOVS 8(R9), F2        // x1.re
	FMOVS 12(R9), F3       // x1.im
	FMOVS 16(R9), F4       // x2.re
	FMOVS 20(R9), F5       // x2.im
	FMOVS 24(R9), F6       // x3.re
	FMOVS 28(R9), F7       // x3.im
	FMOVS 32(R9), F8       // x4.re
	FMOVS 36(R9), F9       // x4.im
	FMOVS 40(R9), F10      // x5.re
	FMOVS 44(R9), F11      // x5.im
	FMOVS 48(R9), F12      // x6.re
	FMOVS 52(R9), F13      // x6.im
	FMOVS 56(R9), F14      // x7.re
	FMOVS 60(R9), F15      // x7.im

	// ===== Stage 1: Two radix-4 butterflies =====
	// Radix-4 on evens: x0, x2, x4, x6 -> a0, a1, a2, a3
	// t0 = x0 + x4
	FADDS F8, F0, F16      // t0.re
	FADDS F9, F1, F17      // t0.im
	// t1 = x0 - x4
	FSUBS F8, F0, F18      // t1.re
	FSUBS F9, F1, F19      // t1.im
	// t2 = x2 + x6
	FADDS F12, F4, F20     // t2.re
	FADDS F13, F5, F21     // t2.im
	// t3 = x2 - x6
	FSUBS F12, F4, F22     // t3.re
	FSUBS F13, F5, F23     // t3.im

	// a0 = t0 + t2
	FADDS F20, F16, F0     // a0.re
	FADDS F21, F17, F1     // a0.im
	// a2 = t0 - t2
	FSUBS F20, F16, F4     // a2.re
	FSUBS F21, F17, F5     // a2.im
	// a1 = t1 - j*t3 = t1.re + t3.im, t1.im - t3.re (forward DIT)
	FADDS F23, F18, F24    // a1.re = t1.re + t3.im
	FSUBS F22, F19, F25    // a1.im = t1.im - t3.re
	// a3 = t1 + j*t3 = t1.re - t3.im, t1.im + t3.re
	FSUBS F23, F18, F26    // a3.re = t1.re - t3.im
	FADDS F22, F19, F27    // a3.im = t1.im + t3.re

	// Radix-4 on odds: x1, x3, x5, x7 -> b0, b1, b2, b3
	// t0 = x1 + x5
	FADDS F10, F2, F16     // t0.re
	FADDS F11, F3, F17     // t0.im
	// t1 = x1 - x5
	FSUBS F10, F2, F18     // t1.re
	FSUBS F11, F3, F19     // t1.im
	// t2 = x3 + x7
	FADDS F14, F6, F20     // t2.re
	FADDS F15, F7, F21     // t2.im
	// t3 = x3 - x7
	FSUBS F14, F6, F22     // t3.re
	FSUBS F15, F7, F23     // t3.im

	// b0 = t0 + t2
	FADDS F20, F16, F2     // b0.re
	FADDS F21, F17, F3     // b0.im
	// b2 = t0 - t2
	FSUBS F20, F16, F6     // b2.re
	FSUBS F21, F17, F7     // b2.im
	// b1 = t1 - j*t3 = t1.re + t3.im, t1.im - t3.re (forward DIT)
	FADDS F23, F18, F28    // b1.re = t1.re + t3.im
	FSUBS F22, F19, F29    // b1.im = t1.im - t3.re
	// b3 = t1 + j*t3 = t1.re - t3.im, t1.im + t3.re
	FSUBS F23, F18, F30    // b3.re = t1.re - t3.im
	FADDS F22, F19, F31    // b3.im = t1.im + t3.re

	// ===== Stage 2: Four radix-2 butterflies with twiddles =====
	// Load twiddle factors w1, w2, w3 (w0 = 1)
	FMOVS 8(R10), F16      // w1.re
	FMOVS 12(R10), F17     // w1.im
	FMOVS 16(R10), F18     // w2.re
	FMOVS 20(R10), F19     // w2.im
	FMOVS 24(R10), F20     // w3.re
	FMOVS 28(R10), F21     // w3.im

	// out[0] = a0 + b0, out[4] = a0 - b0
	FADDS F2, F0, F8       // out0.re
	FADDS F3, F1, F9       // out0.im
	FSUBS F2, F0, F10      // out4.re
	FSUBS F3, F1, F11      // out4.im

	// wb1 = w1 * b1
	FMULS F16, F28, F22    // w1.re * b1.re
	FMULS F17, F29, F23    // w1.im * b1.im
	FSUBS F23, F22, F22    // wb1.re
	FMULS F16, F29, F23    // w1.re * b1.im
	FMULS F17, F28, F12    // w1.im * b1.re
	FADDS F12, F23, F23    // wb1.im
	// out[1] = a1 + wb1, out[5] = a1 - wb1
	FADDS F22, F24, F12    // out1.re
	FADDS F23, F25, F13    // out1.im
	FSUBS F22, F24, F14    // out5.re
	FSUBS F23, F25, F15    // out5.im

	// wb2 = w2 * b2
	FMULS F18, F6, F22     // w2.re * b2.re
	FMULS F19, F7, F23     // w2.im * b2.im
	FSUBS F23, F22, F22    // wb2.re
	FMULS F18, F7, F23     // w2.re * b2.im
	FMULS F19, F6, F24     // w2.im * b2.re
	FADDS F24, F23, F23    // wb2.im
	// out[2] = a2 + wb2, out[6] = a2 - wb2
	FADDS F22, F4, F24     // out2.re
	FADDS F23, F5, F25     // out2.im
	FSUBS F22, F4, F4      // out6.re (reuse F4)
	FSUBS F23, F5, F5      // out6.im (reuse F5)

	// wb3 = w3 * b3
	FMULS F20, F30, F22    // w3.re * b3.re
	FMULS F21, F31, F23    // w3.im * b3.im
	FSUBS F23, F22, F22    // wb3.re
	FMULS F20, F31, F23    // w3.re * b3.im
	FMULS F21, F30, F28    // w3.im * b3.re
	FADDS F28, F23, F23    // wb3.im
	// out[3] = a3 + wb3, out[7] = a3 - wb3
	FADDS F22, F26, F28    // out3.re
	FADDS F23, F27, F29    // out3.im
	FSUBS F22, F26, F30    // out7.re
	FSUBS F23, F27, F31    // out7.im

	// Store results
	FMOVS F8, 0(R8)        // out[0]
	FMOVS F9, 4(R8)
	FMOVS F12, 8(R8)       // out[1]
	FMOVS F13, 12(R8)
	FMOVS F24, 16(R8)      // out[2]
	FMOVS F25, 20(R8)
	FMOVS F28, 24(R8)      // out[3]
	FMOVS F29, 28(R8)
	FMOVS F10, 32(R8)      // out[4]
	FMOVS F11, 36(R8)
	FMOVS F14, 40(R8)      // out[5]
	FMOVS F15, 44(R8)
	FMOVS F4, 48(R8)       // out[6]
	FMOVS F5, 52(R8)
	FMOVS F30, 56(R8)      // out[7]
	FMOVS F31, 60(R8)

	CMP  R8, R20
	BEQ  neon8r4_return_true

	MOVD $0, R0
neon8r4_copy_loop:
	CMP  $8, R0
	BGE  neon8r4_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	ADD  R20, R1, R4
	MOVD R3, (R4)
	ADD  $1, R0, R0
	B    neon8r4_copy_loop

neon8r4_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon8r4_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Inverse transform, size 8, mixed radix-4/radix-2
TEXT ·InverseNEONSize8Radix4Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $8, R13
	BNE  neon8r4_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $8, R0
	BLT  neon8r4_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $8, R0
	BLT  neon8r4_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $8, R0
	BLT  neon8r4_inv_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon8r4_inv_use_dst
	MOVD R11, R8

neon8r4_inv_use_dst:
	// Load all 8 complex values from src
	FMOVS 0(R9), F0        // x0.re
	FMOVS 4(R9), F1        // x0.im
	FMOVS 8(R9), F2        // x1.re
	FMOVS 12(R9), F3       // x1.im
	FMOVS 16(R9), F4       // x2.re
	FMOVS 20(R9), F5       // x2.im
	FMOVS 24(R9), F6       // x3.re
	FMOVS 28(R9), F7       // x3.im
	FMOVS 32(R9), F8       // x4.re
	FMOVS 36(R9), F9       // x4.im
	FMOVS 40(R9), F10      // x5.re
	FMOVS 44(R9), F11      // x5.im
	FMOVS 48(R9), F12      // x6.re
	FMOVS 52(R9), F13      // x6.im
	FMOVS 56(R9), F14      // x7.re
	FMOVS 60(R9), F15      // x7.im

	// ===== Stage 1: Two inverse radix-4 butterflies =====
	// Inverse radix-4 on evens: x0, x2, x4, x6 -> a0, a1, a2, a3
	// For inverse, use -j instead of +j
	FADDS F8, F0, F16      // t0.re
	FADDS F9, F1, F17      // t0.im
	FSUBS F8, F0, F18      // t1.re
	FSUBS F9, F1, F19      // t1.im
	FADDS F12, F4, F20     // t2.re
	FADDS F13, F5, F21     // t2.im
	FSUBS F12, F4, F22     // t3.re
	FSUBS F13, F5, F23     // t3.im

	FADDS F20, F16, F0     // a0.re
	FADDS F21, F17, F1     // a0.im
	FSUBS F20, F16, F4     // a2.re
	FSUBS F21, F17, F5     // a2.im
	// a1 = t1 + j*t3 = t1.re - t3.im, t1.im + t3.re (inverse DIT)
	FSUBS F23, F18, F24    // a1.re = t1.re - t3.im
	FADDS F22, F19, F25    // a1.im = t1.im + t3.re
	// a3 = t1 - j*t3 = t1.re + t3.im, t1.im - t3.re
	FADDS F23, F18, F26    // a3.re = t1.re + t3.im
	FSUBS F22, F19, F27    // a3.im = t1.im - t3.re

	// Inverse radix-4 on odds: x1, x3, x5, x7 -> b0, b1, b2, b3
	FADDS F10, F2, F16     // t0.re
	FADDS F11, F3, F17     // t0.im
	FSUBS F10, F2, F18     // t1.re
	FSUBS F11, F3, F19     // t1.im
	FADDS F14, F6, F20     // t2.re
	FADDS F15, F7, F21     // t2.im
	FSUBS F14, F6, F22     // t3.re
	FSUBS F15, F7, F23     // t3.im

	FADDS F20, F16, F2     // b0.re
	FADDS F21, F17, F3     // b0.im
	FSUBS F20, F16, F6     // b2.re
	FSUBS F21, F17, F7     // b2.im
	// b1 = t1 + j*t3 = t1.re - t3.im, t1.im + t3.re (inverse DIT)
	FSUBS F23, F18, F28    // b1.re = t1.re - t3.im
	FADDS F22, F19, F29    // b1.im = t1.im + t3.re
	// b3 = t1 - j*t3 = t1.re + t3.im, t1.im - t3.re
	FADDS F23, F18, F30    // b3.re = t1.re + t3.im
	FSUBS F22, F19, F31    // b3.im = t1.im - t3.re

	// ===== Stage 2: Four radix-2 butterflies with conjugated twiddles =====
	FMOVS 8(R10), F16      // w1.re
	FMOVS 12(R10), F17     // w1.im
	FNEGS F17, F17         // conjugate
	FMOVS 16(R10), F18     // w2.re
	FMOVS 20(R10), F19     // w2.im
	FNEGS F19, F19         // conjugate
	FMOVS 24(R10), F20     // w3.re
	FMOVS 28(R10), F21     // w3.im
	FNEGS F21, F21         // conjugate

	// out[0] = a0 + b0, out[4] = a0 - b0
	FADDS F2, F0, F8       // out0.re
	FADDS F3, F1, F9       // out0.im
	FSUBS F2, F0, F10      // out4.re
	FSUBS F3, F1, F11      // out4.im

	// wb1 = conj(w1) * b1
	FMULS F16, F28, F22
	FMULS F17, F29, F23
	FSUBS F23, F22, F22
	FMULS F16, F29, F23
	FMULS F17, F28, F12
	FADDS F12, F23, F23
	FADDS F22, F24, F12    // out1.re
	FADDS F23, F25, F13    // out1.im
	FSUBS F22, F24, F14    // out5.re
	FSUBS F23, F25, F15    // out5.im

	// wb2 = conj(w2) * b2
	FMULS F18, F6, F22
	FMULS F19, F7, F23
	FSUBS F23, F22, F22
	FMULS F18, F7, F23
	FMULS F19, F6, F24
	FADDS F24, F23, F23
	FADDS F22, F4, F24     // out2.re
	FADDS F23, F5, F25     // out2.im
	FSUBS F22, F4, F4      // out6.re
	FSUBS F23, F5, F5      // out6.im

	// wb3 = conj(w3) * b3
	FMULS F20, F30, F22
	FMULS F21, F31, F23
	FSUBS F23, F22, F22
	FMULS F20, F31, F23
	FMULS F21, F30, F28
	FADDS F28, F23, F23
	FADDS F22, F26, F28    // out3.re
	FADDS F23, F27, F29    // out3.im
	FSUBS F22, F26, F30    // out7.re
	FSUBS F23, F27, F31    // out7.im

	// Store results
	FMOVS F8, 0(R8)
	FMOVS F9, 4(R8)
	FMOVS F12, 8(R8)
	FMOVS F13, 12(R8)
	FMOVS F24, 16(R8)
	FMOVS F25, 20(R8)
	FMOVS F28, 24(R8)
	FMOVS F29, 28(R8)
	FMOVS F10, 32(R8)
	FMOVS F11, 36(R8)
	FMOVS F14, 40(R8)
	FMOVS F15, 44(R8)
	FMOVS F4, 48(R8)
	FMOVS F5, 52(R8)
	FMOVS F30, 56(R8)
	FMOVS F31, 60(R8)

	CMP  R8, R20
	BEQ  neon8r4_inv_scale

	MOVD $0, R0
neon8r4_inv_copy_loop:
	CMP  $8, R0
	BGE  neon8r4_inv_scale
	LSL  $3, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	ADD  R20, R1, R4
	MOVD R3, (R4)
	ADD  $1, R0, R0
	B    neon8r4_inv_copy_loop

neon8r4_inv_scale:
	MOVD $·neonInv8(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon8r4_inv_scale_loop:
	CMP  $8, R0
	BGE  neon8r4_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon8r4_inv_scale_loop

neon8r4_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon8r4_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

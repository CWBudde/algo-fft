//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-32 Mixed-Radix (Radix-4 + Radix-2) FFT Kernels for ARM64 (complex128)
// ===========================================================================
//
// Size 32 = 4 * 4 * 2, mixed-radix algorithm:
//   Stage 1: 8 radix-4 butterflies (no twiddles), stride=4
//   Stage 2: radix-4 with twiddles, size=16, step=2
//   Stage 3: radix-2 with twiddles, size=32, step=1
//
// Each complex128 element is 16 bytes (real f64 + imag f64).
//
// ===========================================================================

#include "textflag.h"

// Note: neonInv32F64 is defined in neon_f64_size32_radix2.s to avoid duplicate symbols

// Forward transform, size 32, complex128, mixed radix (radix-4, radix-4, radix-2)
// func ForwardNEONSize32MixedRadix24Complex128Asm(dst, src, twiddle, scratch []complex128) bool
TEXT ·ForwardNEONSize32MixedRadix24Complex128Asm(SB), NOSPLIT, $0-97
	// Load slice headers from stack
	MOVD dst+0(FP), R8          // R8 = dst.data
	MOVD src+24(FP), R9         // R9 = src.data
	MOVD twiddle+48(FP), R10    // R10 = twiddle.data
	MOVD scratch+72(FP), R11    // R11 = scratch.data
	MOVD src+32(FP), R13        // R13 = src.len (n)

	// Validate n == 32
	CMP  $32, R13               // n must be exactly 32
	BNE  neon32m24f64_return_false

	// Validate dst capacity >= 32
	MOVD dst+8(FP), R0
	CMP  $32, R0
	BLT  neon32m24f64_return_false

	// Validate twiddle capacity >= 32
	MOVD twiddle+56(FP), R0
	CMP  $32, R0
	BLT  neon32m24f64_return_false

	// Validate scratch capacity >= 32
	MOVD scratch+80(FP), R0
	CMP  $32, R0
	BLT  neon32m24f64_return_false

	// Load bit-reversal table address
	MOVD $bitrev_size32_mixed24_f64<>(SB), R12

	// Save original dst pointer
	MOVD R8, R20

	// If src == dst, use scratch as working buffer
	CMP  R8, R9
	BNE  neon32m24f64_use_dst
	MOVD R11, R8

neon32m24f64_use_dst:
	// =========================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =========================================================================
	MOVD $0, R0

neon32m24f64_bitrev_loop:
	CMP  $32, R0
	BGE  neon32m24f64_stage1

	LSL  $3, R0, R1             // offset into bitrev table
	ADD  R12, R1, R1
	MOVD (R1), R2               // bitrev[i]

	LSL  $4, R2, R3             // source offset (16 bytes per complex128)
	ADD  R9, R3, R3
	MOVD (R3), R4               // real
	MOVD 8(R3), R5              // imag

	LSL  $4, R0, R3             // dest offset
	ADD  R8, R3, R3
	MOVD R4, (R3)
	MOVD R5, 8(R3)

	ADD  $1, R0, R0
	B    neon32m24f64_bitrev_loop

neon32m24f64_stage1:
	// =========================================================================
	// Stage 1: 8 radix-4 butterflies (no twiddles)
	// =========================================================================
	MOVD $0, R14                // base index

neon32m24f64_stage1_loop:
	CMP  $32, R14
	BGE  neon32m24f64_stage2

	LSL  $4, R14, R1            // byte offset
	ADD  R8, R1, R1             // &work[base]

	// Load x0,x1,x2,x3
	FMOVD 0(R1), F0             // x0.real
	FMOVD 8(R1), F1             // x0.imag
	FMOVD 16(R1), F2            // x1.real
	FMOVD 24(R1), F3            // x1.imag
	FMOVD 32(R1), F4            // x2.real
	FMOVD 40(R1), F5            // x2.imag
	FMOVD 48(R1), F6            // x3.real
	FMOVD 56(R1), F7            // x3.imag

	// t0 = x0 + x2, t1 = x0 - x2
	FADDD F4, F0, F8
	FADDD F5, F1, F9
	FSUBD F4, F0, F10
	FSUBD F5, F1, F11

	// t2 = x1 + x3, t3 = x1 - x3
	FADDD F6, F2, F12
	FADDD F7, F3, F13
	FSUBD F6, F2, F14
	FSUBD F7, F3, F15

	// b0 = t0 + t2, b2 = t0 - t2
	FADDD F12, F8, F16
	FADDD F13, F9, F17
	FSUBD F12, F8, F18
	FSUBD F13, F9, F19

	// (-i) * t3 => (t3.imag, -t3.real)
	FMOVD F15, F20
	FNEGD F14, F21

	// b1 = t1 + (-i)*t3
	FADDD F20, F10, F22
	FADDD F21, F11, F23

	// i * t3 => (-t3.imag, t3.real)
	FNEGD F15, F24
	FMOVD F14, F25

	// b3 = t1 + i*t3
	FADDD F24, F10, F26
	FADDD F25, F11, F27

	// Store results
	FMOVD F16, 0(R1)
	FMOVD F17, 8(R1)
	FMOVD F22, 16(R1)
	FMOVD F23, 24(R1)
	FMOVD F18, 32(R1)
	FMOVD F19, 40(R1)
	FMOVD F26, 48(R1)
	FMOVD F27, 56(R1)

	ADD  $4, R14, R14
	B    neon32m24f64_stage1_loop

neon32m24f64_stage2:
	// =========================================================================
	// Stage 2: radix-4 with twiddles, size=16, step=2
	// 2 groups: base 0 and 16
	// =========================================================================
	MOVD $0, R14                // base

neon32m24f64_stage2_base:
	CMP  $32, R14
	BGE  neon32m24f64_stage3

	MOVD $0, R15                // j

neon32m24f64_stage2_j:
	CMP  $4, R15
	BGE  neon32m24f64_stage2_next

	// Indices: idx0=base+j, idx1=base+j+4, idx2=base+j+8, idx3=base+j+12
	ADD  R14, R15, R0           // idx0
	ADD  $4, R0, R1             // idx1
	ADD  $8, R0, R2             // idx2
	ADD  $12, R0, R3            // idx3

	// Load twiddles: w1=tw[j*2], w2=tw[j*4], w3=tw[j*6]
	LSL  $1, R15, R4            // j*2
	LSL  $4, R4, R5             // byte offset
	ADD  R10, R5, R5
	FMOVD 0(R5), F0             // w1.real
	FMOVD 8(R5), F1             // w1.imag

	LSL  $2, R15, R4            // j*4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2             // w2.real
	FMOVD 8(R5), F3             // w2.imag

	// j*6 = j*4 + j*2
	LSL  $1, R15, R6            // j*2
	LSL  $2, R15, R4            // j*4
	ADD  R6, R4, R4             // j*6
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4             // w3.real
	FMOVD 8(R5), F5             // w3.imag

	// Load a0, a1, a2, a3
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// a1 = w1 * a1
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	// a2 = w2 * a2
	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	// a3 = w3 * a3
	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly
	FADDD F10, F6, F14          // t0 = a0 + a2
	FADDD F11, F7, F15
	FSUBD F10, F6, F16          // t1 = a0 - a2
	FSUBD F11, F7, F17

	FADDD F12, F8, F18          // t2 = a1 + a3
	FADDD F13, F9, F19
	FSUBD F12, F8, F20          // t3 = a1 - a3
	FSUBD F13, F9, F21

	// b0 = t0 + t2, b2 = t0 - t2
	FADDD F18, F14, F22
	FADDD F19, F15, F23
	FSUBD F18, F14, F24
	FSUBD F19, F15, F25

	// b1 = t1 + (-i)*t3
	FADDD F21, F16, F26         // t1.r + t3.i
	FSUBD F20, F17, F27         // t1.i - t3.r

	// b3 = t1 + i*t3
	FSUBD F21, F16, F28         // t1.r - t3.i
	FADDD F20, F17, F29         // t1.i + t3.r

	// Store results
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon32m24f64_stage2_j

neon32m24f64_stage2_next:
	ADD  $16, R14, R14
	B    neon32m24f64_stage2_base

neon32m24f64_stage3:
	// =========================================================================
	// Stage 3: radix-2 with twiddles, size=32, step=1
	// =========================================================================
	MOVD $0, R0

neon32m24f64_stage3_loop:
	CMP  $16, R0
	BGE  neon32m24f64_done

	ADD  $16, R0, R1            // idx_b = j + 16

	// Load twiddle[j]
	LSL  $4, R0, R2
	ADD  R10, R2, R2
	FMOVD 0(R2), F0             // w.real
	FMOVD 8(R2), F1             // w.imag

	// Load a = work[j]
	LSL  $4, R0, R2
	ADD  R8, R2, R2
	FMOVD 0(R2), F2             // a.real
	FMOVD 8(R2), F3             // a.imag

	// Load b = work[j+16]
	LSL  $4, R1, R2
	ADD  R8, R2, R2
	FMOVD 0(R2), F4             // b.real
	FMOVD 8(R2), F5             // b.imag

	// wb = w * b
	FMULD F0, F4, F6            // w.r * b.r
	FMULD F1, F5, F7            // w.i * b.i
	FSUBD F7, F6, F6            // wb.real
	FMULD F0, F5, F7            // w.r * b.i
	FMULD F1, F4, F8            // w.i * b.r
	FADDD F8, F7, F7            // wb.imag

	// Butterfly: out_a = a + wb, out_b = a - wb
	FADDD F6, F2, F8            // out_a.real
	FADDD F7, F3, F9            // out_a.imag
	FSUBD F6, F2, F10           // out_b.real
	FSUBD F7, F3, F11           // out_b.imag

	// Store
	LSL  $4, R0, R2
	ADD  R8, R2, R2
	FMOVD F8, 0(R2)
	FMOVD F9, 8(R2)

	LSL  $4, R1, R2
	ADD  R8, R2, R2
	FMOVD F10, 0(R2)
	FMOVD F11, 8(R2)

	ADD  $1, R0, R0
	B    neon32m24f64_stage3_loop

neon32m24f64_done:
	// =========================================================================
	// Copy back if we used scratch
	// =========================================================================
	CMP  R8, R20
	BEQ  neon32m24f64_return_true

	MOVD $0, R0
neon32m24f64_copy_loop:
	CMP  $32, R0
	BGE  neon32m24f64_return_true
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon32m24f64_copy_loop

neon32m24f64_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon32m24f64_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 32, complex128, mixed radix
// ===========================================================================
TEXT ·InverseNEONSize32MixedRadix24Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $32, R13
	BNE  neon32m24f64_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $32, R0
	BLT  neon32m24f64_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $32, R0
	BLT  neon32m24f64_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $32, R0
	BLT  neon32m24f64_inv_return_false

	MOVD $bitrev_size32_mixed24_f64<>(SB), R12
	MOVD R8, R20

	CMP  R8, R9
	BNE  neon32m24f64_inv_use_dst
	MOVD R11, R8

neon32m24f64_inv_use_dst:
	// =========================================================================
	// Bit-reversal permutation
	// =========================================================================
	MOVD $0, R0

neon32m24f64_inv_bitrev_loop:
	CMP  $32, R0
	BGE  neon32m24f64_inv_stage1

	LSL  $3, R0, R1
	ADD  R12, R1, R1
	MOVD (R1), R2

	LSL  $4, R2, R3
	ADD  R9, R3, R3
	MOVD (R3), R4
	MOVD 8(R3), R5

	LSL  $4, R0, R3
	ADD  R8, R3, R3
	MOVD R4, (R3)
	MOVD R5, 8(R3)

	ADD  $1, R0, R0
	B    neon32m24f64_inv_bitrev_loop

neon32m24f64_inv_stage1:
	// =========================================================================
	// Stage 1: 8 radix-4 butterflies (no twiddles) - inverse uses +i instead of -i
	// =========================================================================
	MOVD $0, R14

neon32m24f64_inv_stage1_loop:
	CMP  $32, R14
	BGE  neon32m24f64_inv_stage2

	LSL  $4, R14, R1
	ADD  R8, R1, R1

	FMOVD 0(R1), F0
	FMOVD 8(R1), F1
	FMOVD 16(R1), F2
	FMOVD 24(R1), F3
	FMOVD 32(R1), F4
	FMOVD 40(R1), F5
	FMOVD 48(R1), F6
	FMOVD 56(R1), F7

	FADDD F4, F0, F8
	FADDD F5, F1, F9
	FSUBD F4, F0, F10
	FSUBD F5, F1, F11

	FADDD F6, F2, F12
	FADDD F7, F3, F13
	FSUBD F6, F2, F14
	FSUBD F7, F3, F15

	FADDD F12, F8, F16
	FADDD F13, F9, F17
	FSUBD F12, F8, F18
	FSUBD F13, F9, F19

	// For inverse: i * t3 => (-t3.imag, t3.real)
	FNEGD F15, F20
	FMOVD F14, F21

	FADDD F20, F10, F22
	FADDD F21, F11, F23

	// For inverse: (-i) * t3 => (t3.imag, -t3.real)
	FMOVD F15, F24
	FNEGD F14, F25

	FADDD F24, F10, F26
	FADDD F25, F11, F27

	FMOVD F16, 0(R1)
	FMOVD F17, 8(R1)
	FMOVD F22, 16(R1)
	FMOVD F23, 24(R1)
	FMOVD F18, 32(R1)
	FMOVD F19, 40(R1)
	FMOVD F26, 48(R1)
	FMOVD F27, 56(R1)

	ADD  $4, R14, R14
	B    neon32m24f64_inv_stage1_loop

neon32m24f64_inv_stage2:
	// =========================================================================
	// Stage 2: radix-4 with conjugated twiddles
	// =========================================================================
	MOVD $0, R14

neon32m24f64_inv_stage2_base:
	CMP  $32, R14
	BGE  neon32m24f64_inv_stage3

	MOVD $0, R15

neon32m24f64_inv_stage2_j:
	CMP  $4, R15
	BGE  neon32m24f64_inv_stage2_next

	ADD  R14, R15, R0
	ADD  $4, R0, R1
	ADD  $8, R0, R2
	ADD  $12, R0, R3

	// Load twiddles and conjugate
	LSL  $1, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1
	FNEGD F1, F1

	LSL  $2, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3
	FNEGD F3, F3

	LSL  $1, R15, R6
	LSL  $2, R15, R4
	ADD  R6, R4, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5
	FNEGD F5, F5

	// Load values
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// Apply twiddles
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly (inverse)
	FADDD F10, F6, F14
	FADDD F11, F7, F15
	FSUBD F10, F6, F16
	FSUBD F11, F7, F17

	FADDD F12, F8, F18
	FADDD F13, F9, F19
	FSUBD F12, F8, F20
	FSUBD F13, F9, F21

	FADDD F18, F14, F22
	FADDD F19, F15, F23
	FSUBD F18, F14, F24
	FSUBD F19, F15, F25

	// For inverse: i * t3 for b1
	FSUBD F21, F16, F26
	FADDD F20, F17, F27

	// For inverse: (-i) * t3 for b3
	FADDD F21, F16, F28
	FSUBD F20, F17, F29

	// Store
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon32m24f64_inv_stage2_j

neon32m24f64_inv_stage2_next:
	ADD  $16, R14, R14
	B    neon32m24f64_inv_stage2_base

neon32m24f64_inv_stage3:
	// =========================================================================
	// Stage 3: radix-2 with conjugated twiddles
	// =========================================================================
	MOVD $0, R0

neon32m24f64_inv_stage3_loop:
	CMP  $16, R0
	BGE  neon32m24f64_inv_scale

	ADD  $16, R0, R1

	// Load twiddle and conjugate
	LSL  $4, R0, R2
	ADD  R10, R2, R2
	FMOVD 0(R2), F0
	FMOVD 8(R2), F1
	FNEGD F1, F1

	// Load a
	LSL  $4, R0, R2
	ADD  R8, R2, R2
	FMOVD 0(R2), F2
	FMOVD 8(R2), F3

	// Load b
	LSL  $4, R1, R2
	ADD  R8, R2, R2
	FMOVD 0(R2), F4
	FMOVD 8(R2), F5

	// wb = w* * b
	FMULD F0, F4, F6
	FMULD F1, F5, F7
	FSUBD F7, F6, F6
	FMULD F0, F5, F7
	FMULD F1, F4, F8
	FADDD F8, F7, F7

	// Butterfly
	FADDD F6, F2, F8
	FADDD F7, F3, F9
	FSUBD F6, F2, F10
	FSUBD F7, F3, F11

	// Store
	LSL  $4, R0, R2
	ADD  R8, R2, R2
	FMOVD F8, 0(R2)
	FMOVD F9, 8(R2)

	LSL  $4, R1, R2
	ADD  R8, R2, R2
	FMOVD F10, 0(R2)
	FMOVD F11, 8(R2)

	ADD  $1, R0, R0
	B    neon32m24f64_inv_stage3_loop

neon32m24f64_inv_scale:
	// =========================================================================
	// Copy back if we used scratch
	// =========================================================================
	CMP  R8, R20
	BEQ  neon32m24f64_inv_scale_apply

	MOVD $0, R0
neon32m24f64_inv_copy_loop:
	CMP  $32, R0
	BGE  neon32m24f64_inv_scale_apply
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon32m24f64_inv_copy_loop

neon32m24f64_inv_scale_apply:
	// =========================================================================
	// Apply 1/N scaling
	// =========================================================================
	MOVD $·neonInv32F64(SB), R1
	FMOVD (R1), F0              // 1/32
	MOVD $0, R0

neon32m24f64_inv_scale_loop:
	CMP  $32, R0
	BGE  neon32m24f64_inv_return_true
	LSL  $4, R0, R1
	ADD  R20, R1, R1
	FMOVD 0(R1), F2
	FMOVD 8(R1), F3
	FMULD F0, F2, F2
	FMULD F0, F3, F3
	FMOVD F2, 0(R1)
	FMOVD F3, 8(R1)
	ADD  $1, R0, R0
	B    neon32m24f64_inv_scale_loop

neon32m24f64_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon32m24f64_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Data Section - Bit-reversal permutation table for size 32 mixed-radix 2/4
// ===========================================================================
// For mixed-radix 4,4,2 decomposition

DATA bitrev_size32_mixed24_f64<>+0x000(SB)/8, $0
DATA bitrev_size32_mixed24_f64<>+0x008(SB)/8, $8
DATA bitrev_size32_mixed24_f64<>+0x010(SB)/8, $16
DATA bitrev_size32_mixed24_f64<>+0x018(SB)/8, $24
DATA bitrev_size32_mixed24_f64<>+0x020(SB)/8, $2
DATA bitrev_size32_mixed24_f64<>+0x028(SB)/8, $10
DATA bitrev_size32_mixed24_f64<>+0x030(SB)/8, $18
DATA bitrev_size32_mixed24_f64<>+0x038(SB)/8, $26
DATA bitrev_size32_mixed24_f64<>+0x040(SB)/8, $4
DATA bitrev_size32_mixed24_f64<>+0x048(SB)/8, $12
DATA bitrev_size32_mixed24_f64<>+0x050(SB)/8, $20
DATA bitrev_size32_mixed24_f64<>+0x058(SB)/8, $28
DATA bitrev_size32_mixed24_f64<>+0x060(SB)/8, $6
DATA bitrev_size32_mixed24_f64<>+0x068(SB)/8, $14
DATA bitrev_size32_mixed24_f64<>+0x070(SB)/8, $22
DATA bitrev_size32_mixed24_f64<>+0x078(SB)/8, $30
DATA bitrev_size32_mixed24_f64<>+0x080(SB)/8, $1
DATA bitrev_size32_mixed24_f64<>+0x088(SB)/8, $9
DATA bitrev_size32_mixed24_f64<>+0x090(SB)/8, $17
DATA bitrev_size32_mixed24_f64<>+0x098(SB)/8, $25
DATA bitrev_size32_mixed24_f64<>+0x0A0(SB)/8, $3
DATA bitrev_size32_mixed24_f64<>+0x0A8(SB)/8, $11
DATA bitrev_size32_mixed24_f64<>+0x0B0(SB)/8, $19
DATA bitrev_size32_mixed24_f64<>+0x0B8(SB)/8, $27
DATA bitrev_size32_mixed24_f64<>+0x0C0(SB)/8, $5
DATA bitrev_size32_mixed24_f64<>+0x0C8(SB)/8, $13
DATA bitrev_size32_mixed24_f64<>+0x0D0(SB)/8, $21
DATA bitrev_size32_mixed24_f64<>+0x0D8(SB)/8, $29
DATA bitrev_size32_mixed24_f64<>+0x0E0(SB)/8, $7
DATA bitrev_size32_mixed24_f64<>+0x0E8(SB)/8, $15
DATA bitrev_size32_mixed24_f64<>+0x0F0(SB)/8, $23
DATA bitrev_size32_mixed24_f64<>+0x0F8(SB)/8, $31
GLOBL bitrev_size32_mixed24_f64<>(SB), RODATA, $256

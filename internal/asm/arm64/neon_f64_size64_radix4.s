//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-64 Radix-4 FFT Kernels for ARM64 (complex128)
// ===========================================================================
//
// Size 64 = 4^3, radix-4 algorithm uses 3 stages:
//   Stage 1: 16 radix-4 butterflies (no twiddles), stride=4
//   Stage 2: 4 groups of 4 butterflies, twiddle step=4
//   Stage 3: 1 group of 16 butterflies, twiddle step=1
//
// Each complex128 element is 16 bytes (real f64 + imag f64).
//
// Register allocation:
//   R8  = working dst pointer (may point to scratch if src==dst)
//   R9  = src pointer
//   R10 = twiddle pointer
//   R11 = scratch pointer
//   R12 = bit-reversal table pointer
//   R13 = n (size, should be 64)
//   R14 = loop counter for stages
//   R15 = inner loop counter
//   R16 = temporary for index calculations
//   R17 = temporary for index calculations
//   R20 = original dst pointer (preserved)
//
// ===========================================================================

#include "textflag.h"

DATA ·neonInv64R4F64+0(SB)/8, $0x3f90000000000000 // 1/64 = 0.015625
GLOBL ·neonInv64R4F64(SB), RODATA, $8

// Forward transform, size 64, complex128, radix-4 variant
// func ForwardNEONSize64Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool
TEXT ·ForwardNEONSize64Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load slice headers from stack
	MOVD dst+0(FP), R8          // R8 = dst.data
	MOVD src+24(FP), R9         // R9 = src.data
	MOVD twiddle+48(FP), R10    // R10 = twiddle.data
	MOVD scratch+72(FP), R11    // R11 = scratch.data
	MOVD src+32(FP), R13        // R13 = src.len (n)

	// Validate n == 64
	CMP  $64, R13               // n must be exactly 64
	BNE  neon64r4f64_return_false

	// Validate dst capacity >= 64
	MOVD dst+8(FP), R0          // R0 = dst.len
	CMP  $64, R0
	BLT  neon64r4f64_return_false

	// Validate twiddle capacity >= 64
	MOVD twiddle+56(FP), R0     // R0 = twiddle.len
	CMP  $64, R0
	BLT  neon64r4f64_return_false

	// Validate scratch capacity >= 64
	MOVD scratch+80(FP), R0     // R0 = scratch.len
	CMP  $64, R0
	BLT  neon64r4f64_return_false

	// Load bit-reversal table address
	MOVD $bitrev_size64_radix4_f64<>(SB), R12

	// Save original dst pointer
	MOVD R8, R20                // R20 = original dst

	// If src == dst, use scratch as working buffer to avoid aliasing
	CMP  R8, R9                 // compare dst and src pointers
	BNE  neon64r4f64_use_dst    // if different, use dst directly
	MOVD R11, R8                // src==dst: use scratch as working buffer

neon64r4f64_use_dst:
	// =========================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =========================================================================
	MOVD $0, R0                 // R0 = loop counter i

neon64r4f64_bitrev_loop:
	CMP  $64, R0                // while i < 64
	BGE  neon64r4f64_stage1

	LSL  $3, R0, R1             // R1 = i * 8 (offset into bitrev table)
	ADD  R12, R1, R1            // R1 = &bitrev[i]
	MOVD (R1), R2               // R2 = bitrev[i] (source index)

	LSL  $4, R2, R3             // R3 = bitrev[i] * 16 (complex128 is 16 bytes)
	ADD  R9, R3, R3             // R3 = &src[bitrev[i]]
	MOVD (R3), R4               // R4 = src[bitrev[i]].real (as bits)
	MOVD 8(R3), R5              // R5 = src[bitrev[i]].imag (as bits)

	LSL  $4, R0, R3             // R3 = i * 16
	ADD  R8, R3, R3             // R3 = &work[i]
	MOVD R4, (R3)               // work[i].real = src[bitrev[i]].real
	MOVD R5, 8(R3)              // work[i].imag = src[bitrev[i]].imag

	ADD  $1, R0, R0             // i++
	B    neon64r4f64_bitrev_loop

neon64r4f64_stage1:
	// =========================================================================
	// Stage 1: 16 radix-4 butterflies (no twiddles), stride=4
	// Process groups [0..3], [4..7], ..., [60..63]
	// =========================================================================
	MOVD $0, R14                // R14 = base index

neon64r4f64_stage1_loop:
	CMP  $64, R14               // while base < 64
	BGE  neon64r4f64_stage2

	// addr = &work[base]
	LSL  $4, R14, R1            // R1 = base * 16
	ADD  R8, R1, R1             // R1 = &work[base]

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
	FADDD F4, F0, F8            // t0.real = x0.r + x2.r
	FADDD F5, F1, F9            // t0.imag = x0.i + x2.i
	FSUBD F4, F0, F10           // t1.real = x0.r - x2.r
	FSUBD F5, F1, F11           // t1.imag = x0.i - x2.i

	// t2 = x1 + x3, t3 = x1 - x3
	FADDD F6, F2, F12           // t2.real = x1.r + x3.r
	FADDD F7, F3, F13           // t2.imag = x1.i + x3.i
	FSUBD F6, F2, F14           // t3.real = x1.r - x3.r
	FSUBD F7, F3, F15           // t3.imag = x1.i - x3.i

	// b0 = t0 + t2, b2 = t0 - t2
	FADDD F12, F8, F16          // b0.real
	FADDD F13, F9, F17          // b0.imag
	FSUBD F12, F8, F18          // b2.real
	FSUBD F13, F9, F19          // b2.imag

	// (-i) * t3 => (t3.imag, -t3.real) for forward transform
	FMOVD F15, F20              // (-i)*t3.real = t3.imag
	FNEGD F14, F21              // (-i)*t3.imag = -t3.real

	// b1 = t1 + (-i)*t3
	FADDD F20, F10, F22         // b1.real
	FADDD F21, F11, F23         // b1.imag

	// i * t3 => (-t3.imag, t3.real) for b3
	FNEGD F15, F24              // i*t3.real = -t3.imag
	FMOVD F14, F25              // i*t3.imag = t3.real

	// b3 = t1 + i*t3
	FADDD F24, F10, F26         // b3.real
	FADDD F25, F11, F27         // b3.imag

	// Store results
	FMOVD F16, 0(R1)            // work[base+0]
	FMOVD F17, 8(R1)
	FMOVD F22, 16(R1)           // work[base+1]
	FMOVD F23, 24(R1)
	FMOVD F18, 32(R1)           // work[base+2]
	FMOVD F19, 40(R1)
	FMOVD F26, 48(R1)           // work[base+3]
	FMOVD F27, 56(R1)

	ADD  $4, R14, R14           // base += 4
	B    neon64r4f64_stage1_loop

neon64r4f64_stage2:
	// =========================================================================
	// Stage 2: size=16, 4 groups of 4 butterflies each, twiddle step=4
	// Groups start at base 0, 16, 32, 48
	// =========================================================================
	MOVD $0, R14                // R14 = base (group start)

neon64r4f64_stage2_base:
	CMP  $64, R14               // while base < 64
	BGE  neon64r4f64_stage3

	MOVD $0, R15                // R15 = j (butterfly index within group)

neon64r4f64_stage2_j:
	CMP  $4, R15                // while j < 4
	BGE  neon64r4f64_stage2_next

	// Calculate indices: idx0=base+j, idx1=base+j+4, idx2=base+j+8, idx3=base+j+12
	ADD  R14, R15, R0           // R0 = idx0 = base + j
	ADD  $4, R0, R1             // R1 = idx1 = base + j + 4
	ADD  $8, R0, R2             // R2 = idx2 = base + j + 8
	ADD  $12, R0, R3            // R3 = idx3 = base + j + 12

	// Load twiddles: w1=tw[j*4], w2=tw[j*8], w3=tw[j*12]
	LSL  $2, R15, R4            // R4 = j * 4
	LSL  $4, R4, R5             // R5 = j*4 * 16 bytes
	ADD  R10, R5, R5            // R5 = &twiddle[j*4]
	FMOVD 0(R5), F0             // w1.real
	FMOVD 8(R5), F1             // w1.imag

	LSL  $3, R15, R4            // R4 = j * 8
	LSL  $4, R4, R5
	ADD  R10, R5, R5            // R5 = &twiddle[j*8]
	FMOVD 0(R5), F2             // w2.real
	FMOVD 8(R5), F3             // w2.imag

	MOVD $12, R6
	MUL  R15, R6, R4            // R4 = j * 12
	LSL  $4, R4, R5
	ADD  R10, R5, R5            // R5 = &twiddle[j*12]
	FMOVD 0(R5), F4             // w3.real
	FMOVD 8(R5), F5             // w3.imag

	// Load a0, a1, a2, a3
	LSL  $4, R0, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F6             // a0.real
	FMOVD 8(R5), F7             // a0.imag

	LSL  $4, R1, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F8             // a1.real
	FMOVD 8(R5), F9             // a1.imag

	LSL  $4, R2, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F10            // a2.real
	FMOVD 8(R5), F11            // a2.imag

	LSL  $4, R3, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F12            // a3.real
	FMOVD 8(R5), F13            // a3.imag

	// Apply twiddles: a1 *= w1, a2 *= w2, a3 *= w3
	// a1 = w1 * a1
	FMULD F0, F8, F14           // w1.r * a1.r
	FMULD F1, F9, F15           // w1.i * a1.i
	FSUBD F15, F14, F14         // new a1.real
	FMULD F0, F9, F15           // w1.r * a1.i
	FMULD F1, F8, F16           // w1.i * a1.r
	FADDD F16, F15, F15         // new a1.imag
	FMOVD F14, F8               // a1.real
	FMOVD F15, F9               // a1.imag

	// a2 = w2 * a2
	FMULD F2, F10, F14          // w2.r * a2.r
	FMULD F3, F11, F15          // w2.i * a2.i
	FSUBD F15, F14, F14         // new a2.real
	FMULD F2, F11, F15          // w2.r * a2.i
	FMULD F3, F10, F16          // w2.i * a2.r
	FADDD F16, F15, F15         // new a2.imag
	FMOVD F14, F10              // a2.real
	FMOVD F15, F11              // a2.imag

	// a3 = w3 * a3
	FMULD F4, F12, F14          // w3.r * a3.r
	FMULD F5, F13, F15          // w3.i * a3.i
	FSUBD F15, F14, F14         // new a3.real
	FMULD F4, F13, F15          // w3.r * a3.i
	FMULD F5, F12, F16          // w3.i * a3.r
	FADDD F16, F15, F15         // new a3.imag
	FMOVD F14, F12              // a3.real
	FMOVD F15, F13              // a3.imag

	// Radix-4 butterfly
	// t0 = a0 + a2, t1 = a0 - a2
	FADDD F10, F6, F14          // t0.real
	FADDD F11, F7, F15          // t0.imag
	FSUBD F10, F6, F16          // t1.real
	FSUBD F11, F7, F17          // t1.imag

	// t2 = a1 + a3, t3 = a1 - a3
	FADDD F12, F8, F18          // t2.real
	FADDD F13, F9, F19          // t2.imag
	FSUBD F12, F8, F20          // t3.real
	FSUBD F13, F9, F21          // t3.imag

	// b0 = t0 + t2, b2 = t0 - t2
	FADDD F18, F14, F22         // b0.real
	FADDD F19, F15, F23         // b0.imag
	FSUBD F18, F14, F24         // b2.real
	FSUBD F19, F15, F25         // b2.imag

	// (-i) * t3
	FMOVD F21, F26              // (-i)*t3.real = t3.imag
	FNEGD F20, F27              // (-i)*t3.imag = -t3.real

	// b1 = t1 + (-i)*t3
	FADDD F26, F16, F28         // b1.real
	FADDD F27, F17, F29         // b1.imag

	// i * t3
	FNEGD F21, F30              // i*t3.real = -t3.imag
	FMOVD F20, F31              // i*t3.imag = t3.real

	// b3 = t1 + i*t3
	FADDD F30, F16, F20         // b3.real
	FADDD F31, F17, F21         // b3.imag

	// Store results
	LSL  $4, R0, R5
	ADD  R8, R5, R5
	FMOVD F22, 0(R5)            // b0
	FMOVD F23, 8(R5)

	LSL  $4, R1, R5
	ADD  R8, R5, R5
	FMOVD F28, 0(R5)            // b1
	FMOVD F29, 8(R5)

	LSL  $4, R2, R5
	ADD  R8, R5, R5
	FMOVD F24, 0(R5)            // b2
	FMOVD F25, 8(R5)

	LSL  $4, R3, R5
	ADD  R8, R5, R5
	FMOVD F20, 0(R5)            // b3
	FMOVD F21, 8(R5)

	ADD  $1, R15, R15           // j++
	B    neon64r4f64_stage2_j

neon64r4f64_stage2_next:
	ADD  $16, R14, R14          // base += 16
	B    neon64r4f64_stage2_base

neon64r4f64_stage3:
	// =========================================================================
	// Stage 3: size=64, 1 group of 16 butterflies, twiddle step=1
	// =========================================================================
	MOVD $0, R15                // R15 = j

neon64r4f64_stage3_loop:
	CMP  $16, R15               // while j < 16
	BGE  neon64r4f64_done

	// Indices: idx0=j, idx1=j+16, idx2=j+32, idx3=j+48
	MOVD R15, R0                // R0 = idx0 = j
	ADD  $16, R15, R1           // R1 = idx1 = j + 16
	ADD  $32, R15, R2           // R2 = idx2 = j + 32
	ADD  $48, R15, R3           // R3 = idx3 = j + 48

	// Load twiddles: w1=tw[j], w2=tw[2j], w3=tw[3j]
	LSL  $4, R15, R5            // R5 = j * 16
	ADD  R10, R5, R5            // R5 = &twiddle[j]
	FMOVD 0(R5), F0             // w1.real
	FMOVD 8(R5), F1             // w1.imag

	LSL  $1, R15, R4            // R4 = 2j
	LSL  $4, R4, R5
	ADD  R10, R5, R5            // R5 = &twiddle[2j]
	FMOVD 0(R5), F2             // w2.real
	FMOVD 8(R5), F3             // w2.imag

	ADD  R4, R15, R4            // R4 = 3j = 2j + j
	LSL  $4, R4, R5
	ADD  R10, R5, R5            // R5 = &twiddle[3j]
	FMOVD 0(R5), F4             // w3.real
	FMOVD 8(R5), F5             // w3.imag

	// Load a0, a1, a2, a3
	LSL  $4, R0, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F6             // a0.real
	FMOVD 8(R5), F7             // a0.imag

	LSL  $4, R1, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F8             // a1.real
	FMOVD 8(R5), F9             // a1.imag

	LSL  $4, R2, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F10            // a2.real
	FMOVD 8(R5), F11            // a2.imag

	LSL  $4, R3, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F12            // a3.real
	FMOVD 8(R5), F13            // a3.imag

	// Apply twiddles: a1 *= w1, a2 *= w2, a3 *= w3
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
	FADDD F10, F6, F14          // t0.real
	FADDD F11, F7, F15          // t0.imag
	FSUBD F10, F6, F16          // t1.real
	FSUBD F11, F7, F17          // t1.imag

	FADDD F12, F8, F18          // t2.real
	FADDD F13, F9, F19          // t2.imag
	FSUBD F12, F8, F20          // t3.real
	FSUBD F13, F9, F21          // t3.imag

	FADDD F18, F14, F22         // b0.real
	FADDD F19, F15, F23         // b0.imag
	FSUBD F18, F14, F24         // b2.real
	FSUBD F19, F15, F25         // b2.imag

	FMOVD F21, F26              // (-i)*t3.real
	FNEGD F20, F27              // (-i)*t3.imag

	FADDD F26, F16, F28         // b1.real
	FADDD F27, F17, F29         // b1.imag

	FNEGD F21, F30              // i*t3.real
	FMOVD F20, F31              // i*t3.imag

	FADDD F30, F16, F20         // b3.real
	FADDD F31, F17, F21         // b3.imag

	// Store results
	LSL  $4, R0, R5
	ADD  R8, R5, R5
	FMOVD F22, 0(R5)
	FMOVD F23, 8(R5)

	LSL  $4, R1, R5
	ADD  R8, R5, R5
	FMOVD F28, 0(R5)
	FMOVD F29, 8(R5)

	LSL  $4, R2, R5
	ADD  R8, R5, R5
	FMOVD F24, 0(R5)
	FMOVD F25, 8(R5)

	LSL  $4, R3, R5
	ADD  R8, R5, R5
	FMOVD F20, 0(R5)
	FMOVD F21, 8(R5)

	ADD  $1, R15, R15           // j++
	B    neon64r4f64_stage3_loop

neon64r4f64_done:
	// =========================================================================
	// Copy back from scratch to dst if we used scratch buffer
	// =========================================================================
	CMP  R8, R20                // did we use scratch? (R8 != R20)
	BEQ  neon64r4f64_return_true // no, result already in dst

	MOVD $0, R0                 // R0 = loop counter
neon64r4f64_copy_loop:
	CMP  $64, R0                // while i < 64
	BGE  neon64r4f64_return_true
	LSL  $4, R0, R1             // R1 = i * 16
	ADD  R8, R1, R2             // R2 = &scratch[i]
	MOVD (R2), R3               // R3 = scratch[i].real (as bits)
	MOVD 8(R2), R4              // R4 = scratch[i].imag (as bits)
	ADD  R20, R1, R5            // R5 = &dst[i]
	MOVD R3, (R5)               // dst[i].real = scratch[i].real
	MOVD R4, 8(R5)              // dst[i].imag = scratch[i].imag
	ADD  $1, R0, R0             // i++
	B    neon64r4f64_copy_loop

neon64r4f64_return_true:
	MOVD $1, R0                 // return true
	MOVB R0, ret+96(FP)
	RET

neon64r4f64_return_false:
	MOVD $0, R0                 // return false
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 64, complex128, radix-4 variant
// Same as forward but with conjugated twiddles and 1/N scaling
// ===========================================================================
TEXT ·InverseNEONSize64Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load slice headers from stack
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	// Validate n == 64
	CMP  $64, R13
	BNE  neon64r4f64_inv_return_false

	// Validate capacities
	MOVD dst+8(FP), R0
	CMP  $64, R0
	BLT  neon64r4f64_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $64, R0
	BLT  neon64r4f64_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $64, R0
	BLT  neon64r4f64_inv_return_false

	// Load bit-reversal table address
	MOVD $bitrev_size64_radix4_f64<>(SB), R12

	// Save original dst pointer
	MOVD R8, R20

	// Select working buffer
	CMP  R8, R9
	BNE  neon64r4f64_inv_use_dst
	MOVD R11, R8

neon64r4f64_inv_use_dst:
	// =========================================================================
	// Bit-reversal permutation
	// =========================================================================
	MOVD $0, R0

neon64r4f64_inv_bitrev_loop:
	CMP  $64, R0
	BGE  neon64r4f64_inv_stage1

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
	B    neon64r4f64_inv_bitrev_loop

neon64r4f64_inv_stage1:
	// =========================================================================
	// Stage 1: 16 radix-4 butterflies (no twiddles), stride=4
	// For inverse: use +i instead of -i
	// =========================================================================
	MOVD $0, R14

neon64r4f64_inv_stage1_loop:
	CMP  $64, R14
	BGE  neon64r4f64_inv_stage2

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

	// For inverse: i * t3 => (-t3.imag, t3.real) for b1
	FNEGD F15, F20              // i*t3.real = -t3.imag
	FMOVD F14, F21              // i*t3.imag = t3.real

	FADDD F20, F10, F22         // b1 = t1 + i*t3
	FADDD F21, F11, F23

	// For inverse: (-i) * t3 => (t3.imag, -t3.real) for b3
	FMOVD F15, F24              // (-i)*t3.real = t3.imag
	FNEGD F14, F25              // (-i)*t3.imag = -t3.real

	FADDD F24, F10, F26         // b3 = t1 + (-i)*t3
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
	B    neon64r4f64_inv_stage1_loop

neon64r4f64_inv_stage2:
	// =========================================================================
	// Stage 2: size=16, 4 groups, twiddle step=4 (conjugated)
	// =========================================================================
	MOVD $0, R14

neon64r4f64_inv_stage2_base:
	CMP  $64, R14
	BGE  neon64r4f64_inv_stage3

	MOVD $0, R15

neon64r4f64_inv_stage2_j:
	CMP  $4, R15
	BGE  neon64r4f64_inv_stage2_next

	ADD  R14, R15, R0           // idx0
	ADD  $4, R0, R1             // idx1
	ADD  $8, R0, R2             // idx2
	ADD  $12, R0, R3            // idx3

	// Load twiddles and conjugate
	LSL  $2, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1
	FNEGD F1, F1                // conjugate

	LSL  $3, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3
	FNEGD F3, F3                // conjugate

	MOVD $12, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5
	FNEGD F5, F5                // conjugate

	// Load values
	LSL  $4, R0, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F6
	FMOVD 8(R5), F7

	LSL  $4, R1, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F8
	FMOVD 8(R5), F9

	LSL  $4, R2, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F10
	FMOVD 8(R5), F11

	LSL  $4, R3, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F12
	FMOVD 8(R5), F13

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

	// For inverse: i * t3
	FNEGD F21, F26
	FMOVD F20, F27

	FADDD F26, F16, F28
	FADDD F27, F17, F29

	// For inverse: (-i) * t3
	FMOVD F21, F30
	FNEGD F20, F31

	FADDD F30, F16, F20
	FADDD F31, F17, F21

	// Store
	LSL  $4, R0, R5
	ADD  R8, R5, R5
	FMOVD F22, 0(R5)
	FMOVD F23, 8(R5)

	LSL  $4, R1, R5
	ADD  R8, R5, R5
	FMOVD F28, 0(R5)
	FMOVD F29, 8(R5)

	LSL  $4, R2, R5
	ADD  R8, R5, R5
	FMOVD F24, 0(R5)
	FMOVD F25, 8(R5)

	LSL  $4, R3, R5
	ADD  R8, R5, R5
	FMOVD F20, 0(R5)
	FMOVD F21, 8(R5)

	ADD  $1, R15, R15
	B    neon64r4f64_inv_stage2_j

neon64r4f64_inv_stage2_next:
	ADD  $16, R14, R14
	B    neon64r4f64_inv_stage2_base

neon64r4f64_inv_stage3:
	// =========================================================================
	// Stage 3: size=64, 1 group of 16 butterflies (conjugated twiddles)
	// =========================================================================
	MOVD $0, R15

neon64r4f64_inv_stage3_loop:
	CMP  $16, R15
	BGE  neon64r4f64_inv_done

	MOVD R15, R0
	ADD  $16, R15, R1
	ADD  $32, R15, R2
	ADD  $48, R15, R3

	// Load twiddles and conjugate
	LSL  $4, R15, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1
	FNEGD F1, F1

	LSL  $1, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3
	FNEGD F3, F3

	ADD  R4, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5
	FNEGD F5, F5

	// Load values
	LSL  $4, R0, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F6
	FMOVD 8(R5), F7

	LSL  $4, R1, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F8
	FMOVD 8(R5), F9

	LSL  $4, R2, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F10
	FMOVD 8(R5), F11

	LSL  $4, R3, R5
	ADD  R8, R5, R5
	FMOVD 0(R5), F12
	FMOVD 8(R5), F13

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

	FNEGD F21, F26
	FMOVD F20, F27

	FADDD F26, F16, F28
	FADDD F27, F17, F29

	FMOVD F21, F30
	FNEGD F20, F31

	FADDD F30, F16, F20
	FADDD F31, F17, F21

	// Store
	LSL  $4, R0, R5
	ADD  R8, R5, R5
	FMOVD F22, 0(R5)
	FMOVD F23, 8(R5)

	LSL  $4, R1, R5
	ADD  R8, R5, R5
	FMOVD F28, 0(R5)
	FMOVD F29, 8(R5)

	LSL  $4, R2, R5
	ADD  R8, R5, R5
	FMOVD F24, 0(R5)
	FMOVD F25, 8(R5)

	LSL  $4, R3, R5
	ADD  R8, R5, R5
	FMOVD F20, 0(R5)
	FMOVD F21, 8(R5)

	ADD  $1, R15, R15
	B    neon64r4f64_inv_stage3_loop

neon64r4f64_inv_done:
	// =========================================================================
	// Copy back if we used scratch
	// =========================================================================
	CMP  R8, R20
	BEQ  neon64r4f64_inv_scale

	MOVD $0, R0
neon64r4f64_inv_copy_loop:
	CMP  $64, R0
	BGE  neon64r4f64_inv_scale
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon64r4f64_inv_copy_loop

neon64r4f64_inv_scale:
	// =========================================================================
	// Apply 1/N scaling for inverse transform
	// =========================================================================
	MOVD $·neonInv64R4F64(SB), R1
	FMOVD (R1), F0              // F0 = 1/64
	MOVD $0, R0

neon64r4f64_inv_scale_loop:
	CMP  $64, R0
	BGE  neon64r4f64_inv_return_true
	LSL  $4, R0, R1
	ADD  R20, R1, R1
	FMOVD 0(R1), F2
	FMOVD 8(R1), F3
	FMULD F0, F2, F2
	FMULD F0, F3, F3
	FMOVD F2, 0(R1)
	FMOVD F3, 8(R1)
	ADD  $1, R0, R0
	B    neon64r4f64_inv_scale_loop

neon64r4f64_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon64r4f64_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Data Section - Bit-reversal permutation table for size 64 radix-4
// ===========================================================================
// For radix-4 decomposition: reverse 3 base-4 digits
// bitrev[i] = reverse_digits_base4(i, 3)

DATA bitrev_size64_radix4_f64<>+0x000(SB)/8, $0
DATA bitrev_size64_radix4_f64<>+0x008(SB)/8, $16
DATA bitrev_size64_radix4_f64<>+0x010(SB)/8, $32
DATA bitrev_size64_radix4_f64<>+0x018(SB)/8, $48
DATA bitrev_size64_radix4_f64<>+0x020(SB)/8, $4
DATA bitrev_size64_radix4_f64<>+0x028(SB)/8, $20
DATA bitrev_size64_radix4_f64<>+0x030(SB)/8, $36
DATA bitrev_size64_radix4_f64<>+0x038(SB)/8, $52
DATA bitrev_size64_radix4_f64<>+0x040(SB)/8, $8
DATA bitrev_size64_radix4_f64<>+0x048(SB)/8, $24
DATA bitrev_size64_radix4_f64<>+0x050(SB)/8, $40
DATA bitrev_size64_radix4_f64<>+0x058(SB)/8, $56
DATA bitrev_size64_radix4_f64<>+0x060(SB)/8, $12
DATA bitrev_size64_radix4_f64<>+0x068(SB)/8, $28
DATA bitrev_size64_radix4_f64<>+0x070(SB)/8, $44
DATA bitrev_size64_radix4_f64<>+0x078(SB)/8, $60
DATA bitrev_size64_radix4_f64<>+0x080(SB)/8, $1
DATA bitrev_size64_radix4_f64<>+0x088(SB)/8, $17
DATA bitrev_size64_radix4_f64<>+0x090(SB)/8, $33
DATA bitrev_size64_radix4_f64<>+0x098(SB)/8, $49
DATA bitrev_size64_radix4_f64<>+0x0A0(SB)/8, $5
DATA bitrev_size64_radix4_f64<>+0x0A8(SB)/8, $21
DATA bitrev_size64_radix4_f64<>+0x0B0(SB)/8, $37
DATA bitrev_size64_radix4_f64<>+0x0B8(SB)/8, $53
DATA bitrev_size64_radix4_f64<>+0x0C0(SB)/8, $9
DATA bitrev_size64_radix4_f64<>+0x0C8(SB)/8, $25
DATA bitrev_size64_radix4_f64<>+0x0D0(SB)/8, $41
DATA bitrev_size64_radix4_f64<>+0x0D8(SB)/8, $57
DATA bitrev_size64_radix4_f64<>+0x0E0(SB)/8, $13
DATA bitrev_size64_radix4_f64<>+0x0E8(SB)/8, $29
DATA bitrev_size64_radix4_f64<>+0x0F0(SB)/8, $45
DATA bitrev_size64_radix4_f64<>+0x0F8(SB)/8, $61
DATA bitrev_size64_radix4_f64<>+0x100(SB)/8, $2
DATA bitrev_size64_radix4_f64<>+0x108(SB)/8, $18
DATA bitrev_size64_radix4_f64<>+0x110(SB)/8, $34
DATA bitrev_size64_radix4_f64<>+0x118(SB)/8, $50
DATA bitrev_size64_radix4_f64<>+0x120(SB)/8, $6
DATA bitrev_size64_radix4_f64<>+0x128(SB)/8, $22
DATA bitrev_size64_radix4_f64<>+0x130(SB)/8, $38
DATA bitrev_size64_radix4_f64<>+0x138(SB)/8, $54
DATA bitrev_size64_radix4_f64<>+0x140(SB)/8, $10
DATA bitrev_size64_radix4_f64<>+0x148(SB)/8, $26
DATA bitrev_size64_radix4_f64<>+0x150(SB)/8, $42
DATA bitrev_size64_radix4_f64<>+0x158(SB)/8, $58
DATA bitrev_size64_radix4_f64<>+0x160(SB)/8, $14
DATA bitrev_size64_radix4_f64<>+0x168(SB)/8, $30
DATA bitrev_size64_radix4_f64<>+0x170(SB)/8, $46
DATA bitrev_size64_radix4_f64<>+0x178(SB)/8, $62
DATA bitrev_size64_radix4_f64<>+0x180(SB)/8, $3
DATA bitrev_size64_radix4_f64<>+0x188(SB)/8, $19
DATA bitrev_size64_radix4_f64<>+0x190(SB)/8, $35
DATA bitrev_size64_radix4_f64<>+0x198(SB)/8, $51
DATA bitrev_size64_radix4_f64<>+0x1A0(SB)/8, $7
DATA bitrev_size64_radix4_f64<>+0x1A8(SB)/8, $23
DATA bitrev_size64_radix4_f64<>+0x1B0(SB)/8, $39
DATA bitrev_size64_radix4_f64<>+0x1B8(SB)/8, $55
DATA bitrev_size64_radix4_f64<>+0x1C0(SB)/8, $11
DATA bitrev_size64_radix4_f64<>+0x1C8(SB)/8, $27
DATA bitrev_size64_radix4_f64<>+0x1D0(SB)/8, $43
DATA bitrev_size64_radix4_f64<>+0x1D8(SB)/8, $59
DATA bitrev_size64_radix4_f64<>+0x1E0(SB)/8, $15
DATA bitrev_size64_radix4_f64<>+0x1E8(SB)/8, $31
DATA bitrev_size64_radix4_f64<>+0x1F0(SB)/8, $47
DATA bitrev_size64_radix4_f64<>+0x1F8(SB)/8, $63
GLOBL bitrev_size64_radix4_f64<>(SB), RODATA, $512

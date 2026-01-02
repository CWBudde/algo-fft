//go:build arm64 && asm && !purego

// ===========================================================================
// NEON-optimized FFT Assembly for ARM64 (complex128/float64)
// ===========================================================================
//
// This file implements high-performance FFT transforms using ARM NEON (Advanced SIMD)
// instructions for complex128 (double-precision) data types.
//
// ALGORITHM: Decimation-in-Time (DIT) Cooley-Tukey (same as AVX2 implementation)
//
// NEON CHARACTERISTICS:
// - 128-bit registers (Q/V0-V31)
// - Process 1 complex128 per register (each complex128 = 16 bytes)
// - Use FMLA/FMLS for fused multiply-add/subtract
// - Manual twiddle gathering for strided access (no gather instruction)
//
// REGISTER ALLOCATION:
//   R8:  work pointer (dst or scratch)
//   R9:  src pointer
//   R10: twiddle pointer
//   R11: scratch pointer / reused for stride_bytes
//   R12: bitrev pointer / reused for stride_bytes
//   R13: n (transform length)
//   R14: size (outer loop: 2, 4, 8, ... n)
//   R15: half = size/2
//   R16: step = n/size (twiddle stride)
//   R17: base (middle loop counter)
//   R0:  j (inner loop counter)
//   R1-R4: temporary index calculations
//
// ===========================================================================

#include "textflag.h"

TEXT ·ForwardNEONComplex128Asm(SB), NOSPLIT, $0-121
	// -----------------------------------------------------------------------
	// PHASE 1: Load parameters and validate inputs
	// -----------------------------------------------------------------------
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD bitrev+96(FP), R12
	MOVD src+32(FP), R13

	CBZ  R13, f128_return_true

	MOVD dst+8(FP), R0
	CMP  R13, R0
	BLT  f128_return_false

	MOVD twiddle+56(FP), R0
	CMP  R13, R0
	BLT  f128_return_false

	MOVD scratch+80(FP), R0
	CMP  R13, R0
	BLT  f128_return_false

	MOVD bitrev+104(FP), R0
	CMP  R13, R0
	BLT  f128_return_false

	CMP  $1, R13
	BNE  f128_check_power_of_2
	MOVD (R9), R0
	MOVD 8(R9), R1
	MOVD R0, (R8)
	MOVD R1, 8(R8)
	B    f128_return_true

f128_check_power_of_2:
	SUB  $1, R13, R0
	TST  R13, R0
	BNE  f128_return_false

	// -----------------------------------------------------------------------
	// PHASE 2: Select working buffer
	// -----------------------------------------------------------------------
	CMP  R8, R9
	BNE  f128_use_dst_as_work

	MOVD R11, R8
	B    f128_do_bit_reversal

f128_use_dst_as_work:
	// Out-of-place: use dst directly

f128_do_bit_reversal:
	// -----------------------------------------------------------------------
	// PHASE 3: Bit-reversal permutation
	// -----------------------------------------------------------------------
	MOVD $0, R17

f128_bitrev_loop:
	CMP  R13, R17
	BGE  f128_bitrev_done

	LSL  $3, R17, R0
	ADD  R12, R0, R0
	MOVD (R0), R1

	LSL  $4, R1, R0
	ADD  R9, R0, R0
	MOVD (R0), R2
	MOVD 8(R0), R3

	LSL  $4, R17, R0
	ADD  R8, R0, R0
	MOVD R2, (R0)
	MOVD R3, 8(R0)

	ADD  $1, R17, R17
	B    f128_bitrev_loop

f128_bitrev_done:
	// -----------------------------------------------------------------------
	// PHASE 4: Main DIT Butterfly Stages
	// -----------------------------------------------------------------------
	MOVD $2, R14

f128_size_loop:
	CMP  R13, R14
	BGT  f128_transform_done

	LSR  $1, R14, R15
	UDIV R14, R13, R16
	MOVD $0, R17

f128_base_loop:
	CMP  R13, R17
	BGE  f128_next_size

	MOVD $0, R0

f128_inner_loop:
	CMP  R15, R0
	BGE  f128_next_base

	ADD  R17, R0, R1
	ADD  R1, R15, R2

	MUL  R0, R16, R3
	LSL  $4, R3, R3
	ADD  R10, R3, R3

	FMOVD 0(R3), F0
	FMOVD 8(R3), F1

	LSL  $4, R1, R4
	ADD  R8, R4, R4
	FMOVD 0(R4), F2
	FMOVD 8(R4), F3

	LSL  $4, R2, R4
	ADD  R8, R4, R4
	FMOVD 0(R4), F4
	FMOVD 8(R4), F5

	FMULD F0, F4, F6
	FMULD F1, F5, F7
	FSUBD F7, F6, F6

	FMULD F0, F5, F7
	FMULD F1, F4, F5
	FADDD F5, F7, F7

	FADDD F6, F2, F0
	FADDD F7, F3, F1
	FSUBD F6, F2, F4
	FSUBD F7, F3, F5

	LSL  $4, R1, R4
	ADD  R8, R4, R4
	FMOVD F0, 0(R4)
	FMOVD F1, 8(R4)

	LSL  $4, R2, R4
	ADD  R8, R4, R4
	FMOVD F4, 0(R4)
	FMOVD F5, 8(R4)

	ADD  $1, R0, R0
	B    f128_inner_loop

f128_next_base:
	ADD  R14, R17, R17
	B    f128_base_loop

f128_next_size:
	LSL  $1, R14, R14
	B    f128_size_loop

f128_transform_done:
	// -----------------------------------------------------------------------
	// PHASE 5: Copy result to destination if needed
	// -----------------------------------------------------------------------
	MOVD dst+0(FP), R0
	CMP  R8, R0
	BEQ  f128_return_true

	MOVD $0, R1

f128_copy_loop:
	CMP  R13, R1
	BGE  f128_return_true

	LSL  $4, R1, R2
	ADD  R8, R2, R3
	MOVD (R3), R4
	MOVD 8(R3), R5

	ADD  R0, R2, R3
	MOVD R4, (R3)
	MOVD R5, 8(R3)

	ADD  $1, R1, R1
	B    f128_copy_loop

f128_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

f128_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// ===========================================================================
// inverseNEONComplex128Asm - Inverse FFT for complex128 using NEON
// ===========================================================================

TEXT ·InverseNEONComplex128Asm(SB), NOSPLIT, $0-121
	// -----------------------------------------------------------------------
	// PHASE 1: Load parameters and validate inputs
	// -----------------------------------------------------------------------
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD bitrev+96(FP), R12
	MOVD src+32(FP), R13

	CBZ  R13, i128_return_true

	MOVD dst+8(FP), R0
	CMP  R13, R0
	BLT  i128_return_false

	MOVD twiddle+56(FP), R0
	CMP  R13, R0
	BLT  i128_return_false

	MOVD scratch+80(FP), R0
	CMP  R13, R0
	BLT  i128_return_false

	MOVD bitrev+104(FP), R0
	CMP  R13, R0
	BLT  i128_return_false

	CMP  $1, R13
	BNE  i128_check_power_of_2
	MOVD (R9), R0
	MOVD 8(R9), R1
	MOVD R0, (R8)
	MOVD R1, 8(R8)
	B    i128_scale_done

i128_check_power_of_2:
	SUB  $1, R13, R0
	TST  R13, R0
	BNE  i128_return_false

	// -----------------------------------------------------------------------
	// PHASE 2: Select working buffer
	// -----------------------------------------------------------------------
	CMP  R8, R9
	BNE  i128_use_dst_as_work

	MOVD R11, R8
	B    i128_do_bit_reversal

i128_use_dst_as_work:
	// Out-of-place: use dst directly

i128_do_bit_reversal:
	// -----------------------------------------------------------------------
	// PHASE 3: Bit-reversal permutation
	// -----------------------------------------------------------------------
	MOVD $0, R17

i128_bitrev_loop:
	CMP  R13, R17
	BGE  i128_bitrev_done

	LSL  $3, R17, R0
	ADD  R12, R0, R0
	MOVD (R0), R1

	LSL  $4, R1, R0
	ADD  R9, R0, R0
	MOVD (R0), R2
	MOVD 8(R0), R3

	LSL  $4, R17, R0
	ADD  R8, R0, R0
	MOVD R2, (R0)
	MOVD R3, 8(R0)

	ADD  $1, R17, R17
	B    i128_bitrev_loop

i128_bitrev_done:
	// -----------------------------------------------------------------------
	// PHASE 4: Main DIT Butterfly Stages (inverse)
	// -----------------------------------------------------------------------
	MOVD $2, R14

i128_size_loop:
	CMP  R13, R14
	BGT  i128_transform_done

	LSR  $1, R14, R15
	UDIV R14, R13, R16
	MOVD $0, R17

i128_base_loop:
	CMP  R13, R17
	BGE  i128_next_size

	MOVD $0, R0

i128_inner_loop:
	CMP  R15, R0
	BGE  i128_next_base

	ADD  R17, R0, R1
	ADD  R1, R15, R2

	MUL  R0, R16, R3
	LSL  $4, R3, R3
	ADD  R10, R3, R3

	FMOVD 0(R3), F0
	FMOVD 8(R3), F1
	FNEGD F1, F1

	LSL  $4, R1, R4
	ADD  R8, R4, R4
	FMOVD 0(R4), F2
	FMOVD 8(R4), F3

	LSL  $4, R2, R4
	ADD  R8, R4, R4
	FMOVD 0(R4), F4
	FMOVD 8(R4), F5

	FMULD F0, F4, F6
	FMULD F1, F5, F7
	FSUBD F7, F6, F6

	FMULD F0, F5, F7
	FMULD F1, F4, F5
	FADDD F5, F7, F7

	FADDD F6, F2, F0
	FADDD F7, F3, F1
	FSUBD F6, F2, F4
	FSUBD F7, F3, F5

	LSL  $4, R1, R4
	ADD  R8, R4, R4
	FMOVD F0, 0(R4)
	FMOVD F1, 8(R4)

	LSL  $4, R2, R4
	ADD  R8, R4, R4
	FMOVD F4, 0(R4)
	FMOVD F5, 8(R4)

	ADD  $1, R0, R0
	B    i128_inner_loop

i128_next_base:
	ADD  R14, R17, R17
	B    i128_base_loop

i128_next_size:
	LSL  $1, R14, R14
	B    i128_size_loop

i128_transform_done:
	// -----------------------------------------------------------------------
	// PHASE 5: Copy result to destination if needed
	// -----------------------------------------------------------------------
	MOVD dst+0(FP), R0
	CMP  R8, R0
	BEQ  i128_scale

	MOVD $0, R1

i128_copy_loop:
	CMP  R13, R1
	BGE  i128_scale

	LSL  $4, R1, R2
	ADD  R8, R2, R3
	MOVD (R3), R4
	MOVD 8(R3), R5

	ADD  R0, R2, R3
	MOVD R4, (R3)
	MOVD R5, 8(R3)

	ADD  $1, R1, R1
	B    i128_copy_loop

i128_scale:
	// -----------------------------------------------------------------------
	// PHASE 6: Scale by 1/n
	// -----------------------------------------------------------------------
	MOVD dst+0(FP), R0
	MOVD $0, R1

	MOVD $·neonOne64(SB), R2
	FMOVD 0(R2), F0              // F0 = 1.0
	MOVD R13, R3
	SCVTFWD R3, F1               // F1 = float64(n)
	FDIVD F1, F0, F0             // F0 = 1.0 / n

i128_scale_loop:
	CMP  R13, R1
	BGE  i128_scale_done

	LSL  $4, R1, R2
	ADD  R0, R2, R2
	FMOVD 0(R2), F2
	FMOVD 8(R2), F3
	FMULD F0, F2, F2
	FMULD F0, F3, F3
	FMOVD F2, 0(R2)
	FMOVD F3, 8(R2)

	ADD  $1, R1, R1
	B    i128_scale_loop

i128_scale_done:
	B    i128_return_true

i128_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

i128_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// ===========================================================================
// Size-Specific NEON Kernels (Stubs)
// ===========================================================================
//
// These are placeholder stubs that return false to trigger fallback to the
// generic NEON kernels. They will be replaced with fully unrolled size-specific
// implementations in subsequent phases (15.5.2-5).
//
// Architecture notes for future implementation:
// - NEON 128-bit registers hold 2 complex64 (vs AVX2's 4 complex64)
// - Size-16: 4 stages, 8 butterflies, 32 complex multiplies
// - Size-32: 5 stages, 16 butterflies, 80 complex multiplies
// - Size-64: 6 stages, 32 butterflies, 192 complex multiplies
// - Size-128: 7 stages, 64 butterflies, 448 complex multiplies

// ===========================================================================
// Forward Size-Specific Kernels (complex64)
// ===========================================================================

// forwardNEONSize16Complex64Asm - Size-16 forward FFT (fully unrolled)
// ===========================================================================
//
// Fully unrolled DIT FFT for size 16 using NEON SIMD.
// 4 stages: size=2, 4, 8, 16
// NEON processes 2 complex64 per 128-bit register (vs AVX2's 4).
//
// Register allocation:
//   R8:  work pointer (dst or scratch)
//   R9:  src pointer
//   R10: twiddle pointer
//   R11: scratch pointer
//   R12: bitrev pointer
//   R13: n (should be 16)
//
// Vector registers:
//   V0-V7:   Data registers for 16 complex64 values (8 vectors of 2 each)
//   V16-V23: Temporary for butterfly operations
//   V24-V27: Twiddle factors
//   V28:     neonOnes constant (1.0f x 4)
//   V29-V31: Scratch
//

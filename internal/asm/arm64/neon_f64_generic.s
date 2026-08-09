//go:build arm64 && !purego

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
// - 128-bit registers (Q/V0-V31); one .D2 register holds 2 float64, i.e.
//   exactly 2 complex128 once real/imag are split into separate registers
//   (half the per-register width of the complex64 generic kernel's .4S).
// - The butterfly's twiddle multiply and add/sub use VADDF_D2/VSUBF_D2/
//   VMULF_D2/VFMAF_D2/VFMSF_D2 from neon_fp.h, which emit the real vector
//   FADD/FSUB/FMUL/FMLA/FMLS encodings directly via WORD — Go's assembler has
//   no mnemonic for vector FP add/sub/mul. This file previously used plain
//   scalar FMOVD/FADDD/FMULD/FSUBD throughout (no vector instructions at
//   all, despite living among the NEON kernels); see docs/CODELET_BENCHMARKS.md.
// - Two butterflies (j, j+1) are processed per vector iteration when at
//   least 2 remain in the current group. VLD2/VST2 deinterleave two adjacent
//   complex128 directly into separate re/im registers, so — unlike the
//   complex64 kernel — no VUZP/VZIP shuffle is needed to split real from
//   imaginary. Twiddles are loaded contiguously via VLD2 when step==1 (the
//   final stage), or gathered lane-by-lane via VMOV when step>1.
// - Manual twiddle gathering for strided access (no gather instruction).
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
//   R1-R7, R20-R21: temporary index/gather calculations
//
// ===========================================================================

#include "textflag.h"
#include "neon_fp.h"

TEXT ·ForwardNEONComplex128Asm(SB), NOSPLIT, $0-97
	// -----------------------------------------------------------------------
	// PHASE 1: Load parameters and validate inputs
	// -----------------------------------------------------------------------
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src_len+32(FP), R13

	CBZ  R13, f128_return_true

	MOVD dst_len+8(FP), R0
	CMP  R13, R0
	BLT  f128_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  R13, R0
	BLT  f128_return_false

	MOVD scratch_len+80(FP), R0
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

	// R12 = 64 - log2(n), the shift used after RBIT.
	CLZ R13, R12
	ADD $1, R12, R12

	// -----------------------------------------------------------------------
	// PHASE 2: Select working buffer
	// -----------------------------------------------------------------------
	MOVD R8, R20                 // R20 = original dst
	CMP  R8, R9
	BNE  f128_use_dst_as_work

	MOVD R11, R8
	B    f128_do_bit_reversal

f128_use_dst_as_work:
	// Out-of-place: use dst directly

f128_do_bit_reversal:
	// -----------------------------------------------------------------------
	// PHASE 3: Bit-reversal permutation (computed on-the-fly)
	// -----------------------------------------------------------------------
	// For each i in [0, n), compute rev = bitreverse(i, log2(n))
	// and copy src[rev] to work[i]
	MOVD $0, R17                 // R17 = i

f128_bitrev_loop:
	CMP  R13, R17
	BGE  f128_bitrev_done

	RBIT R17, R1
	LSR  R12, R1, R1             // reverse the low log2(n) bits
	LSL  $4, R1, R0              // R0 = rev * 16 (byte offset)
	ADD  R9, R0, R0              // R0 = &src[rev]
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

	SUB  R0, R15, R5             // R5 = remaining = half - j
	CMP  $2, R5
	BLT  f128_scalar_butterfly
	CMP  $1, R16
	BEQ  f128_vector_contig

	// -------------------------------------------------------------------
	// Vectorized gather path: step > 1, at least 2 butterflies remain.
	// Processes (j, j+1) together.
	// -------------------------------------------------------------------
	ADD  R17, R0, R1             // R1 = idx_a
	ADD  R1, R15, R2             // R2 = idx_b

	LSL  $4, R1, R3
	ADD  R8, R3, R3              // R3 = &work[idx_a]
	LSL  $4, R2, R4
	ADD  R8, R4, R4              // R4 = &work[idx_b]

	VLD2 (R3), [V0.D2, V1.D2]    // V0 = [ar0,ar1], V1 = [ai0,ai1]
	VLD2 (R4), [V2.D2, V3.D2]    // V2 = [br0,br1], V3 = [bi0,bi1]

	MUL  R0, R16, R5             // R5 = idx0 = j*step
	ADD  R5, R16, R6             // R6 = idx1 = (j+1)*step
	LSL  $4, R5, R5
	ADD  R10, R5, R5             // R5 = &twiddle[idx0]
	LSL  $4, R6, R6
	ADD  R10, R6, R6             // R6 = &twiddle[idx1]

	MOVD 0(R5), R20              // wr0 bits
	MOVD 8(R5), R21              // wi0 bits
	VMOV R20, V4.D[0]
	VMOV R21, V5.D[0]
	MOVD 0(R6), R20              // wr1 bits
	MOVD 8(R6), R21              // wi1 bits
	VMOV R20, V4.D[1]            // V4 = wr = [wr0, wr1]
	VMOV R21, V5.D[1]            // V5 = wi = [wi0, wi1]

	// wb = w * b: wb.re = br*wr - bi*wi, wb.im = br*wi + bi*wr
	VMULF_D2(2, 4, 6)  // V6 = br*wr
	VFMSF_D2(3, 5, 6)  // V6 -= bi*wi
	VMULF_D2(2, 5, 7)  // V7 = br*wi
	VFMAF_D2(3, 4, 7)  // V7 += bi*wr

	// a' = a + wb, b' = a - wb
	VADDF_D2(0, 6, 10) // a'.re
	VADDF_D2(1, 7, 11) // a'.im
	VSUBF_D2(0, 6, 12) // b'.re
	VSUBF_D2(1, 7, 13) // b'.im

	VST2 [V10.D2, V11.D2], (R3)
	VST2 [V12.D2, V13.D2], (R4)

	ADD  $2, R0, R0              // j += 2
	B    f128_inner_loop

f128_vector_contig:
	// -------------------------------------------------------------------
	// Vectorized contiguous path: step == 1 (final stage), twiddle[j] and
	// twiddle[j+1] are adjacent, so they load directly via VLD2.
	// -------------------------------------------------------------------
	ADD  R17, R0, R1             // R1 = idx_a
	ADD  R1, R15, R2             // R2 = idx_b

	LSL  $4, R1, R3
	ADD  R8, R3, R3              // R3 = &work[idx_a]
	LSL  $4, R2, R4
	ADD  R8, R4, R4              // R4 = &work[idx_b]

	LSL  $4, R0, R5
	ADD  R10, R5, R5             // R5 = &twiddle[j]

	VLD2 (R3), [V0.D2, V1.D2]    // a
	VLD2 (R4), [V2.D2, V3.D2]    // b
	VLD2 (R5), [V4.D2, V5.D2]    // w

	VMULF_D2(2, 4, 6)
	VFMSF_D2(3, 5, 6)  // V6 -= bi*wi
	VMULF_D2(2, 5, 7)
	VFMAF_D2(3, 4, 7)  // V7 += bi*wr

	VADDF_D2(0, 6, 10)
	VADDF_D2(1, 7, 11)
	VSUBF_D2(0, 6, 12)
	VSUBF_D2(1, 7, 13)

	VST2 [V10.D2, V11.D2], (R3)
	VST2 [V12.D2, V13.D2], (R4)

	ADD  $2, R0, R0              // j += 2
	B    f128_inner_loop

f128_scalar_butterfly:
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
	VLD1 (R3), [V0.B16, V1.B16]  // raw 32-byte copy: 2 complex128
	ADD  R0, R2, R3
	VST1 [V0.B16, V1.B16], (R3)

	ADD  $2, R1, R1
	B    f128_copy_loop

f128_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

f128_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// inverseNEONComplex128Asm - Inverse FFT for complex128 using NEON
// ===========================================================================

TEXT ·InverseNEONComplex128Asm(SB), NOSPLIT, $0-97
	// -----------------------------------------------------------------------
	// PHASE 1: Load parameters and validate inputs
	// -----------------------------------------------------------------------
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src_len+32(FP), R13

	CBZ  R13, i128_return_true

	MOVD dst_len+8(FP), R0
	CMP  R13, R0
	BLT  i128_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  R13, R0
	BLT  i128_return_false

	MOVD scratch_len+80(FP), R0
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

	// R12 = 64 - log2(n), the shift used after RBIT.
	CLZ R13, R12
	ADD $1, R12, R12

	// -----------------------------------------------------------------------
	// PHASE 2: Select working buffer
	// -----------------------------------------------------------------------
	MOVD R8, R20                 // R20 = original dst
	CMP  R8, R9
	BNE  i128_use_dst_as_work

	MOVD R11, R8
	B    i128_do_bit_reversal

i128_use_dst_as_work:
	// Out-of-place: use dst directly

i128_do_bit_reversal:
	// -----------------------------------------------------------------------
	// PHASE 3: Bit-reversal permutation (computed on-the-fly)
	// -----------------------------------------------------------------------
	MOVD $0, R17                 // R17 = i

i128_bitrev_loop:
	CMP  R13, R17
	BGE  i128_bitrev_done

	RBIT R17, R1
	LSR  R12, R1, R1             // reverse the low log2(n) bits
	LSL  $4, R1, R0              // R0 = rev * 16 (byte offset)
	ADD  R9, R0, R0              // R0 = &src[rev]
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

	SUB  R0, R15, R5             // R5 = remaining = half - j
	CMP  $2, R5
	BLT  i128_scalar_butterfly
	CMP  $1, R16
	BEQ  i128_vector_contig

	// -------------------------------------------------------------------
	// Vectorized gather path: step > 1, at least 2 butterflies remain.
	// Uses conj(w): wb.re = br*wr + bi*wi, wb.im = bi*wr - br*wi.
	// -------------------------------------------------------------------
	ADD  R17, R0, R1             // R1 = idx_a
	ADD  R1, R15, R2             // R2 = idx_b

	LSL  $4, R1, R3
	ADD  R8, R3, R3              // R3 = &work[idx_a]
	LSL  $4, R2, R4
	ADD  R8, R4, R4              // R4 = &work[idx_b]

	VLD2 (R3), [V0.D2, V1.D2]    // V0 = [ar0,ar1], V1 = [ai0,ai1]
	VLD2 (R4), [V2.D2, V3.D2]    // V2 = [br0,br1], V3 = [bi0,bi1]

	MUL  R0, R16, R5             // R5 = idx0 = j*step
	ADD  R5, R16, R6             // R6 = idx1 = (j+1)*step
	LSL  $4, R5, R5
	ADD  R10, R5, R5             // R5 = &twiddle[idx0]
	LSL  $4, R6, R6
	ADD  R10, R6, R6             // R6 = &twiddle[idx1]

	MOVD 0(R5), R20              // wr0 bits
	MOVD 8(R5), R21              // wi0 bits
	VMOV R20, V4.D[0]
	VMOV R21, V5.D[0]
	MOVD 0(R6), R20              // wr1 bits
	MOVD 8(R6), R21              // wi1 bits
	VMOV R20, V4.D[1]            // V4 = wr = [wr0, wr1]
	VMOV R21, V5.D[1]            // V5 = wi = [wi0, wi1]

	VMULF_D2(2, 4, 6)  // V6 = br*wr
	VFMAF_D2(3, 5, 6)  // V6 += bi*wi
	VMULF_D2(3, 4, 7)  // V7 = bi*wr
	VFMSF_D2(2, 5, 7)  // V7 -= br*wi

	VADDF_D2(0, 6, 10) // a'.re
	VADDF_D2(1, 7, 11) // a'.im
	VSUBF_D2(0, 6, 12) // b'.re
	VSUBF_D2(1, 7, 13) // b'.im

	VST2 [V10.D2, V11.D2], (R3)
	VST2 [V12.D2, V13.D2], (R4)

	ADD  $2, R0, R0              // j += 2
	B    i128_inner_loop

i128_vector_contig:
	ADD  R17, R0, R1             // R1 = idx_a
	ADD  R1, R15, R2             // R2 = idx_b

	LSL  $4, R1, R3
	ADD  R8, R3, R3              // R3 = &work[idx_a]
	LSL  $4, R2, R4
	ADD  R8, R4, R4              // R4 = &work[idx_b]

	LSL  $4, R0, R5
	ADD  R10, R5, R5             // R5 = &twiddle[j]

	VLD2 (R3), [V0.D2, V1.D2]
	VLD2 (R4), [V2.D2, V3.D2]
	VLD2 (R5), [V4.D2, V5.D2]

	VMULF_D2(2, 4, 6)
	VFMAF_D2(3, 5, 6)  // V6 += bi*wi
	VMULF_D2(3, 4, 7)
	VFMSF_D2(2, 5, 7)  // V7 -= br*wi

	VADDF_D2(0, 6, 10)
	VADDF_D2(1, 7, 11)
	VSUBF_D2(0, 6, 12)
	VSUBF_D2(1, 7, 13)

	VST2 [V10.D2, V11.D2], (R3)
	VST2 [V12.D2, V13.D2], (R4)

	ADD  $2, R0, R0              // j += 2
	B    i128_inner_loop

i128_scalar_butterfly:
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
	VLD1 (R3), [V0.B16, V1.B16]  // raw 32-byte copy: 2 complex128
	ADD  R0, R2, R3
	VST1 [V0.B16, V1.B16], (R3)

	ADD  $2, R1, R1
	B    i128_copy_loop

i128_scale:
	// -----------------------------------------------------------------------
	// PHASE 6: Scale by 1/n
	// -----------------------------------------------------------------------
	// Compute the scalar reciprocal, stash it through scratch[0] (already
	// validated to be >= n complex128 long, and free at this point — its
	// contents, if used as the work buffer, were already copied to dst
	// above) and VLD1R-broadcast it into both lanes of V30.
	MOVD dst+0(FP), R0
	MOVD $0, R1

	MOVD  $·neonOne64(SB), R2
	FMOVD 0(R2), F0              // F0 = 1.0
	MOVD  R13, R3
	SCVTFWD R3, F1                // F1 = float64(n)
	FDIVD F1, F0, F0              // F0 = 1.0 / n
	FMOVD F0, 0(R11)              // stash into scratch[0]
	VLD1R (R11), [V30.D2]         // V30 = [1/n, 1/n]

i128_scale_loop:
	CMP  R13, R1
	BGE  i128_scale_done

	LSL  $4, R1, R2
	ADD  R0, R2, R2
	VLD2 (R2), [V0.D2, V1.D2]     // V0 = [re0,re1], V1 = [im0,im1]
	VMULF_D2(0, 30, 0)
	VMULF_D2(1, 30, 1)
	VST2 [V0.D2, V1.D2], (R2)

	ADD  $2, R1, R1
	B    i128_scale_loop

i128_scale_done:
	B    i128_return_true

i128_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

i128_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
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

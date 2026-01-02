//go:build arm64 && asm && !purego

// ===========================================================================
// NEON-optimized FFT Assembly for ARM64 (complex64/float32)
// ===========================================================================
//
// This file implements high-performance FFT transforms using ARM NEON (Advanced SIMD)
// instructions for complex64 (single-precision) data types.
//
// ALGORITHM: Decimation-in-Time (DIT) Cooley-Tukey (same as AVX2 implementation)
//
// NEON CHARACTERISTICS:
// - 128-bit registers (Q/V0-V31)
// - Process 2 complex64 per register (each complex64 = 8 bytes)
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


// ===========================================================================
// forwardNEONComplex64Asm - Forward FFT for complex64 using NEON
// ===========================================================================
//
// func forwardNEONComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
//
// Go calling convention on ARM64:
//   dst_base+0(FP), dst_len+8(FP), dst_cap+16(FP)
//   src_base+24(FP), src_len+32(FP), src_cap+40(FP)
//   twiddle_base+48(FP), twiddle_len+56(FP), twiddle_cap+64(FP)
//   scratch_base+72(FP), scratch_len+80(FP), scratch_cap+88(FP)
//   bitrev_base+96(FP), bitrev_len+104(FP), bitrev_cap+112(FP)
//   return: bool (R0)
//
TEXT ·ForwardNEONComplex64Asm(SB), NOSPLIT, $0-121
	// -----------------------------------------------------------------------
	// PHASE 1: Load parameters and validate inputs
	// -----------------------------------------------------------------------
	// R8  = dst pointer
	MOVD dst+0(FP), R8
	// R9  = src pointer
	MOVD src+24(FP), R9
	// R10 = twiddle pointer
	MOVD twiddle+48(FP), R10
	// R11 = scratch pointer
	MOVD scratch+72(FP), R11
	// R12 = bitrev pointer
	MOVD bitrev+96(FP), R12
	// R13 = n = len(src)
	MOVD src+32(FP), R13

	// Empty input is valid (no-op)
	CBZ  R13, return_true

	// Validate all slice lengths are >= n
	MOVD dst+8(FP), R0
	CMP  R13, R0
	BLT  return_false            // dst too short

	MOVD twiddle+56(FP), R0
	CMP  R13, R0
	BLT  return_false            // twiddle too short

	MOVD scratch+80(FP), R0
	CMP  R13, R0
	BLT  return_false            // scratch too short

	MOVD bitrev+104(FP), R0
	CMP  R13, R0
	BLT  return_false            // bitrev too short

	// Trivial case: n=1, just copy
	CMP  $1, R13
	BNE  check_power_of_2
	MOVD (R9), R0                // Load 8 bytes (complex64)
	MOVD R0, (R8)                // Store to dst
	B    return_true

check_power_of_2:
	// Verify n is power of 2: (n & (n-1)) == 0
	SUB  $1, R13, R0             // R0 = n - 1
	TST  R13, R0                 // Test n & (n-1)
	BNE  return_false            // Not power of 2

	// -----------------------------------------------------------------------
	// PHASE 2: Select working buffer
	// -----------------------------------------------------------------------
	// For in-place transforms (dst == src), use scratch buffer
	CMP  R8, R9
	BNE  use_dst_as_work

	// In-place: use scratch as working buffer
	MOVD R11, R8                 // R8 = work = scratch
	B    do_bit_reversal

use_dst_as_work:
	// Out-of-place: use dst directly

do_bit_reversal:
	// -----------------------------------------------------------------------
	// PHASE 3: Bit-reversal permutation
	// -----------------------------------------------------------------------
	// Reorder input using precomputed bit-reversed indices:
	//   work[i] = src[bitrev[i]]  for i = 0..n-1
	//
	// Algorithm:
	//   for i := 0; i < n; i++ {
	//     j := bitrev[i]
	//     work[i] = src[j]
	//   }
	MOVD $0, R17                 // R17 = i = 0

bitrev_loop:
	CMP  R13, R17                // Compare i with n
	BGE  bitrev_done             // if i >= n, done

	// Load j = bitrev[i]
	// bitrev is []int, each int is 8 bytes on arm64
	LSL  $3, R17, R0             // R0 = i * 8 (byte offset for int array)
	ADD  R12, R0, R0             // R0 = &bitrev[i]
	MOVD (R0), R1                // R1 = j = bitrev[i]

	// Load src[j] (complex64 = 8 bytes)
	LSL  $3, R1, R0              // R0 = j * 8 (byte offset for complex64 array)
	ADD  R9, R0, R0              // R0 = &src[j]
	MOVD (R0), R2                // R2 = src[j] (8 bytes = 1 complex64)

	// Store to work[i]
	LSL  $3, R17, R0             // R0 = i * 8
	ADD  R8, R0, R0              // R0 = &work[i]
	MOVD R2, (R0)                // work[i] = src[j]

	ADD  $1, R17, R17            // i++
	B    bitrev_loop

bitrev_done:
	// -----------------------------------------------------------------------
	// PHASE 4: Main DIT Butterfly Stages
	// -----------------------------------------------------------------------
	// Outer loop: for size = 2, 4, 8, ... up to n
	//   Each iteration doubles the butterfly group size
	//   - size=2:  combine pairs        (n/2 groups of 2)
	//   - size=4:  combine quads        (n/4 groups of 4)
	//   - size=n:  final combination    (1 group of n)
	MOVD $2, R14                 // R14 = size = 2

size_loop:
	CMP  R13, R14                // Compare size with n
	BGT  transform_done          // Done when size > n

	// half = number of butterflies per group = size/2
	LSR  $1, R14, R15            // R15 = half = size >> 1

	// step = twiddle stride = n/size
	// Twiddles are stored for the largest stage (size=n)
	// Smaller stages skip entries: twiddle[j*step]
	UDIV R14, R13, R16           // R16 = step = n / size (dividend in R13, divisor in R14)

	// Middle loop: process each group
	// for base = 0, size, 2*size, ... up to n
	MOVD $0, R17                 // R17 = base = 0

base_loop:
	CMP  R13, R17                // Compare base with n
	BGE  next_size               // All groups processed

	// Inner loop: process butterflies within this group
	// for j = 0; j < half; j++
	MOVD $0, R0                  // R0 = j = 0

inner_loop:
	CMP  R15, R0                 // Compare j with half
	BGE  next_base               // All butterflies in group processed

	SUB  R0, R15, R5             // R5 = remaining = half - j
	CMP  $2, R5
	BLT  scalar_butterfly
	CMP  $1, R16
	BEQ  vector_contig

	// Vectorized gather path for step > 1 and at least 2 butterflies remain.
	// idx_a = base + j, idx_b = idx_a + half
	ADD  R17, R0, R1             // R1 = idx_a
	ADD  R1, R15, R2             // R2 = idx_b

	// ptr_a = work + idx_a*8, ptr_b = work + idx_b*8
	LSL  $3, R1, R3
	ADD  R8, R3, R3              // R3 = &work[idx_a]
	LSL  $3, R2, R4
	ADD  R8, R4, R4              // R4 = &work[idx_b]

	// twiddle indices for j and j+1
	MUL  R0, R16, R5             // R5 = idx0 = j*step
	ADD  R5, R16, R6             // R6 = idx1 = (j+1)*step
	LSL  $3, R5, R5
	ADD  R10, R5, R5             // R5 = &twiddle[idx0]
	LSL  $3, R6, R6
	ADD  R10, R6, R6             // R6 = &twiddle[idx1]

	// Load a, b vectors
	VLD1 (R3), [V0.S4]           // V0 = [ar0, ai0, ar1, ai1]
	VLD1 (R4), [V1.S4]           // V1 = [br0, bi0, br1, bi1]

	// Gather twiddles into V2
	MOVW 0(R5), R7
	MOVW 4(R5), R2
	MOVW 0(R6), R1
	MOVW 4(R6), R5
	VMOV R7, V2.S[0]
	VMOV R2, V2.S[1]
	VMOV R1, V2.S[2]
	VMOV R5, V2.S[3]

	// Deinterleave b and w.
	VUZP1 V1.S4, V1.S4, V3.S4    // V3 = [br0, br1, br0, br1]
	VUZP2 V1.S4, V1.S4, V4.S4    // V4 = [bi0, bi1, bi0, bi1]
	VUZP1 V2.S4, V2.S4, V5.S4    // V5 = [wr0, wr1, wr0, wr1]
	VUZP2 V2.S4, V2.S4, V6.S4    // V6 = [wi0, wi1, wi0, wi1]

	// wb.real = br*wr - bi*wi
	VEOR V7.B16, V7.B16, V7.B16
	VFMLA V3.S4, V5.S4, V7.S4
	VFMLS V4.S4, V6.S4, V7.S4

	// wb.imag = br*wi + bi*wr
	VEOR V8.B16, V8.B16, V8.B16
	VFMLA V3.S4, V6.S4, V8.S4
	VFMLA V4.S4, V5.S4, V8.S4

	// Pack wb and butterfly.
	VZIP1 V8.S4, V7.S4, V9.S4   // V9 = [wb.r0, wb.i0, wb.r1, wb.i1]

	// a' = a + wb, b' = a - wb (use ones vector + VFMLA/VFMLS)
	MOVD $·neonOnes(SB), R6
	VLD1 (R6), [V12.S4]
	VMOV V0.B16, V10.B16
	VFMLA V9.S4, V12.S4, V10.S4
	VMOV V0.B16, V11.B16
	VFMLS V9.S4, V12.S4, V11.S4

	VST1 [V10.S4], (R3)
	VST1 [V11.S4], (R4)

	ADD  $2, R0, R0              // j += 2
	B    inner_loop

vector_contig:
	// idx_a = base + j, idx_b = idx_a + half
	ADD  R17, R0, R1             // R1 = idx_a
	ADD  R1, R15, R2             // R2 = idx_b

	// ptr_a = work + idx_a*8, ptr_b = work + idx_b*8
	LSL  $3, R1, R3
	ADD  R8, R3, R3              // R3 = &work[idx_a]
	LSL  $3, R2, R4
	ADD  R8, R4, R4              // R4 = &work[idx_b]

	// ptr_w = twiddle + j*8 (step == 1)
	LSL  $3, R0, R5
	ADD  R10, R5, R5             // R5 = &twiddle[j]

	// Load a, b, w (2 complex64 each)
	VLD1 (R3), [V0.S4]           // V0 = [ar0, ai0, ar1, ai1]
	VLD1 (R4), [V1.S4]           // V1 = [br0, bi0, br1, bi1]
	VLD1 (R5), [V2.S4]           // V2 = [wr0, wi0, wr1, wi1]

	// Deinterleave b and w.
	VUZP1 V1.S4, V1.S4, V3.S4    // V3 = [br0, br1, br0, br1]
	VUZP2 V1.S4, V1.S4, V4.S4    // V4 = [bi0, bi1, bi0, bi1]
	VUZP1 V2.S4, V2.S4, V5.S4    // V5 = [wr0, wr1, wr0, wr1]
	VUZP2 V2.S4, V2.S4, V6.S4    // V6 = [wi0, wi1, wi0, wi1]

	// wb.real = br*wr - bi*wi
	VEOR V7.B16, V7.B16, V7.B16
	VFMLA V3.S4, V5.S4, V7.S4
	VFMLS V4.S4, V6.S4, V7.S4

	// wb.imag = br*wi + bi*wr
	VEOR V8.B16, V8.B16, V8.B16
	VFMLA V3.S4, V6.S4, V8.S4
	VFMLA V4.S4, V5.S4, V8.S4

	// Pack wb and butterfly.
	VZIP1 V8.S4, V7.S4, V9.S4   // V9 = [wb.r0, wb.i0, wb.r1, wb.i1]

	// a' = a + wb, b' = a - wb (use ones vector + VFMLA/VFMLS)
	MOVD $·neonOnes(SB), R6
	VLD1 (R6), [V12.S4]
	VMOV V0.B16, V10.B16
	VFMLA V9.S4, V12.S4, V10.S4
	VMOV V0.B16, V11.B16
	VFMLS V9.S4, V12.S4, V11.S4

	VST1 [V10.S4], (R3)
	VST1 [V11.S4], (R4)

	ADD  $2, R0, R0              // j += 2
	B    inner_loop

scalar_butterfly:
	// -----------------------------------------------------------------------
	// Scalar Butterfly Computation
	// -----------------------------------------------------------------------
	// Compute indices:
	//   idx_a = base + j
	//   idx_b = base + j + half
	ADD  R17, R0, R1             // R1 = idx_a = base + j
	ADD  R1, R15, R2             // R2 = idx_b = idx_a + half

	// Load twiddle factor w = twiddle[j * step]
	// For size=2, step=n/2 and j=0, so w = twiddle[0] = 1+0i
	MUL  R0, R16, R3             // R3 = tw_idx = j * step
	LSL  $3, R3, R3              // R3 = tw_idx * 8 (byte offset)
	ADD  R10, R3, R3             // R3 = &twiddle[j*step]

	// Load twiddle as two 32-bit floats
	// w_real = F0, w_imag = F1
	FMOVS 0(R3), F0              // F0 = w.real
	FMOVS 4(R3), F1              // F1 = w.imag

	// Load a = work[idx_a] (complex64 = 8 bytes)
	LSL  $3, R1, R4              // R4 = idx_a * 8
	ADD  R8, R4, R4              // R4 = &work[idx_a]
	FMOVS 0(R4), F2              // F2 = a.real
	FMOVS 4(R4), F3              // F3 = a.imag

	// Load b = work[idx_b] (complex64 = 8 bytes)
	LSL  $3, R2, R4              // R4 = idx_b * 8
	ADD  R8, R4, R4              // R4 = &work[idx_b]
	FMOVS 0(R4), F4              // F4 = b.real
	FMOVS 4(R4), F5              // F5 = b.imag

	// Complex multiply: wb = w * b
	// wb.real = w.real * b.real - w.imag * b.imag
	// wb.imag = w.real * b.imag + w.imag * b.real
	FMULS F0, F4, F6             // F6 = w.real * b.real
	FMULS F1, F5, F7             // F7 = w.imag * b.imag
	FSUBS F7, F6, F6             // F6 = wb.real = w.real*b.real - w.imag*b.imag

	FMULS F0, F5, F7             // F7 = w.real * b.imag
	FMULS F1, F4, F5             // F5 = w.imag * b.real
	FADDS F5, F7, F7             // F7 = wb.imag = w.real*b.imag + w.imag*b.real

	// Butterfly operation:
	//   a' = a + wb
	//   b' = a - wb
	FADDS F6, F2, F0             // F0 = a'.real = a.real + wb.real
	FADDS F7, F3, F1             // F1 = a'.imag = a.imag + wb.imag

	FSUBS F6, F2, F4             // F4 = b'.real = a.real - wb.real
	FSUBS F7, F3, F5             // F5 = b'.imag = a.imag - wb.imag

	// Store a' = work[idx_a]
	LSL  $3, R1, R4              // R4 = idx_a * 8
	ADD  R8, R4, R4              // R4 = &work[idx_a]
	FMOVS F0, 0(R4)              // Store a'.real
	FMOVS F1, 4(R4)              // Store a'.imag

	// Store b' = work[idx_b]
	LSL  $3, R2, R4              // R4 = idx_b * 8
	ADD  R8, R4, R4              // R4 = &work[idx_b]
	FMOVS F4, 0(R4)              // Store b'.real
	FMOVS F5, 4(R4)              // Store b'.imag

	// j++
	ADD  $1, R0, R0
	B    inner_loop

next_base:
	// base += size
	ADD  R14, R17, R17
	B    base_loop

next_size:
	// size *= 2
	LSL  $1, R14, R14
	B    size_loop

transform_done:
	// -----------------------------------------------------------------------
	// PHASE 5: Copy result to destination if needed
	// -----------------------------------------------------------------------
	// If we used scratch buffer (in-place transform), copy to dst
	// Check if R8 (work) == dst or scratch
	MOVD dst+0(FP), R0           // R0 = dst
	CMP  R8, R0
	BEQ  return_true             // Already in dst, done

	// Copy from scratch (R8) to dst (R0)
	MOVD $0, R1                  // R1 = i = 0

copy_loop:
	CMP  R13, R1                 // Compare i with n
	BGE  return_true             // Done copying

	LSL  $3, R1, R2              // R2 = i * 8
	ADD  R8, R2, R3              // R3 = &work[i]
	MOVD (R3), R4                // R4 = work[i]

	ADD  R0, R2, R3              // R3 = &dst[i]
	MOVD R4, (R3)                // dst[i] = work[i]

	ADD  $1, R1, R1              // i++
	B    copy_loop

return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// ===========================================================================
// neonComplexMul2Asm - Multiply 2 complex64 values using NEON

// ===========================================================================
// inverseNEONComplex64Asm - Inverse FFT for complex64 using NEON
// ===========================================================================
TEXT ·InverseNEONComplex64Asm(SB), NOSPLIT, $0-121
	// -----------------------------------------------------------------------
	// PHASE 1: Load parameters and validate inputs
	// -----------------------------------------------------------------------
	// R8  = dst pointer
	MOVD dst+0(FP), R8
	// R9  = src pointer
	MOVD src+24(FP), R9
	// R10 = twiddle pointer
	MOVD twiddle+48(FP), R10
	// R11 = scratch pointer
	MOVD scratch+72(FP), R11
	// R12 = bitrev pointer
	MOVD bitrev+96(FP), R12
	// R13 = n = len(src)
	MOVD src+32(FP), R13

	// Empty input is valid (no-op)
	CBZ  R13, inv_return_true

	// Validate all slice lengths are >= n
	MOVD dst+8(FP), R0
	CMP  R13, R0
	BLT  inv_return_false        // dst too short

	MOVD twiddle+56(FP), R0
	CMP  R13, R0
	BLT  inv_return_false        // twiddle too short

	MOVD scratch+80(FP), R0
	CMP  R13, R0
	BLT  inv_return_false        // scratch too short

	MOVD bitrev+104(FP), R0
	CMP  R13, R0
	BLT  inv_return_false        // bitrev too short

	// Trivial case: n=1, just copy
	CMP  $1, R13
	BNE  inv_check_power_of_2
	MOVD (R9), R0                // Load 8 bytes (complex64)
	MOVD R0, (R8)                // Store to dst
	B    inv_scale_done

inv_check_power_of_2:
	// Verify n is power of 2: (n & (n-1)) == 0
	SUB  $1, R13, R0             // R0 = n - 1
	TST  R13, R0                 // Test n & (n-1)
	BNE  inv_return_false        // Not power of 2

	// -----------------------------------------------------------------------
	// PHASE 2: Select working buffer
	// -----------------------------------------------------------------------
	// For in-place transforms (dst == src), use scratch buffer
	CMP  R8, R9
	BNE  inv_use_dst_as_work

	// In-place: use scratch as working buffer
	MOVD R11, R8                 // R8 = work = scratch
	B    inv_do_bit_reversal

inv_use_dst_as_work:
	// Out-of-place: use dst directly

inv_do_bit_reversal:
	// -----------------------------------------------------------------------
	// PHASE 3: Bit-reversal permutation
	// -----------------------------------------------------------------------
	MOVD $0, R17                 // R17 = i = 0

inv_bitrev_loop:
	CMP  R13, R17                // Compare i with n
	BGE  inv_bitrev_done         // if i >= n, done

	// Load j = bitrev[i]
	LSL  $3, R17, R0             // R0 = i * 8 (byte offset for int array)
	ADD  R12, R0, R0             // R0 = &bitrev[i]
	MOVD (R0), R1                // R1 = j = bitrev[i]

	// Load src[j] (complex64 = 8 bytes)
	LSL  $3, R1, R0              // R0 = j * 8 (byte offset for complex64 array)
	ADD  R9, R0, R0              // R0 = &src[j]
	MOVD (R0), R2                // R2 = src[j]

	// Store to work[i]
	LSL  $3, R17, R0             // R0 = i * 8
	ADD  R8, R0, R0              // R0 = &work[i]
	MOVD R2, (R0)                // work[i] = src[j]

	ADD  $1, R17, R17            // i++
	B    inv_bitrev_loop

inv_bitrev_done:
	// -----------------------------------------------------------------------
	// PHASE 4: Main DIT Butterfly Stages (inverse)
	// -----------------------------------------------------------------------
	MOVD $2, R14                 // R14 = size = 2

inv_size_loop:
	CMP  R13, R14                // Compare size with n
	BGT  inv_transform_done      // Done when size > n

	// half = number of butterflies per group = size/2
	LSR  $1, R14, R15            // R15 = half = size >> 1

	// step = twiddle stride = n/size
	UDIV R14, R13, R16           // R16 = step = n / size

	// Middle loop: process each group
	MOVD $0, R17                 // R17 = base = 0

inv_base_loop:
	CMP  R13, R17                // Compare base with n
	BGE  inv_next_size           // All groups processed

	// Inner loop: process butterflies within this group
	MOVD $0, R0                  // R0 = j = 0

inv_inner_loop:
	CMP  R15, R0                 // Compare j with half
	BGE  inv_next_base           // All butterflies in group processed

	SUB  R0, R15, R5             // R5 = remaining = half - j
	CMP  $2, R5
	BLT  inv_scalar_butterfly
	CMP  $1, R16
	BEQ  inv_vector_contig

	// Vectorized gather path for step > 1
	ADD  R17, R0, R1             // R1 = idx_a
	ADD  R1, R15, R2             // R2 = idx_b

	LSL  $3, R1, R3
	ADD  R8, R3, R3              // R3 = &work[idx_a]
	LSL  $3, R2, R4
	ADD  R8, R4, R4              // R4 = &work[idx_b]

	MUL  R0, R16, R5             // R5 = idx0 = j*step
	ADD  R5, R16, R6             // R6 = idx1 = (j+1)*step
	LSL  $3, R5, R5
	ADD  R10, R5, R5             // R5 = &twiddle[idx0]
	LSL  $3, R6, R6
	ADD  R10, R6, R6             // R6 = &twiddle[idx1]

	VLD1 (R3), [V0.S4]
	VLD1 (R4), [V1.S4]

	MOVW 0(R5), R7
	MOVW 4(R5), R2
	MOVW 0(R6), R1
	MOVW 4(R6), R5
	VMOV R7, V2.S[0]
	VMOV R2, V2.S[1]
	VMOV R1, V2.S[2]
	VMOV R5, V2.S[3]

	MOVD $·neonSignImag(SB), R6
	VLD1 (R6), [V13.S4]
	VEOR V13.B16, V2.B16, V2.B16 // conjugate twiddles

	VUZP1 V1.S4, V1.S4, V3.S4
	VUZP2 V1.S4, V1.S4, V4.S4
	VUZP1 V2.S4, V2.S4, V5.S4
	VUZP2 V2.S4, V2.S4, V6.S4

	VEOR V7.B16, V7.B16, V7.B16
	VFMLA V3.S4, V5.S4, V7.S4
	VFMLS V4.S4, V6.S4, V7.S4

	VEOR V8.B16, V8.B16, V8.B16
	VFMLA V3.S4, V6.S4, V8.S4
	VFMLA V4.S4, V5.S4, V8.S4

	VZIP1 V8.S4, V7.S4, V9.S4

	MOVD $·neonOnes(SB), R6
	VLD1 (R6), [V12.S4]
	VMOV V0.B16, V10.B16
	VFMLA V9.S4, V12.S4, V10.S4
	VMOV V0.B16, V11.B16
	VFMLS V9.S4, V12.S4, V11.S4

	VST1 [V10.S4], (R3)
	VST1 [V11.S4], (R4)

	ADD  $2, R0, R0
	B    inv_inner_loop

inv_vector_contig:
	ADD  R17, R0, R1             // R1 = idx_a
	ADD  R1, R15, R2             // R2 = idx_b

	LSL  $3, R1, R3
	ADD  R8, R3, R3              // R3 = &work[idx_a]
	LSL  $3, R2, R4
	ADD  R8, R4, R4              // R4 = &work[idx_b]

	LSL  $3, R0, R5
	ADD  R10, R5, R5             // R5 = &twiddle[j]

	VLD1 (R3), [V0.S4]
	VLD1 (R4), [V1.S4]
	VLD1 (R5), [V2.S4]

	MOVD $·neonSignImag(SB), R6
	VLD1 (R6), [V13.S4]
	VEOR V13.B16, V2.B16, V2.B16 // conjugate twiddles

	VUZP1 V1.S4, V1.S4, V3.S4
	VUZP2 V1.S4, V1.S4, V4.S4
	VUZP1 V2.S4, V2.S4, V5.S4
	VUZP2 V2.S4, V2.S4, V6.S4

	VEOR V7.B16, V7.B16, V7.B16
	VFMLA V3.S4, V5.S4, V7.S4
	VFMLS V4.S4, V6.S4, V7.S4

	VEOR V8.B16, V8.B16, V8.B16
	VFMLA V3.S4, V6.S4, V8.S4
	VFMLA V4.S4, V5.S4, V8.S4

	VZIP1 V8.S4, V7.S4, V9.S4

	MOVD $·neonOnes(SB), R6
	VLD1 (R6), [V12.S4]
	VMOV V0.B16, V10.B16
	VFMLA V9.S4, V12.S4, V10.S4
	VMOV V0.B16, V11.B16
	VFMLS V9.S4, V12.S4, V11.S4

	VST1 [V10.S4], (R3)
	VST1 [V11.S4], (R4)

	ADD  $2, R0, R0
	B    inv_inner_loop

inv_scalar_butterfly:
	// Scalar butterfly
	ADD  R17, R0, R1             // R1 = idx_a = base + j
	ADD  R1, R15, R2             // R2 = idx_b = idx_a + half

	MUL  R0, R16, R3             // R3 = tw_idx = j * step
	LSL  $3, R3, R3
	ADD  R10, R3, R3             // R3 = &twiddle[j*step]

	FMOVS 0(R3), F0              // F0 = w.real
	FMOVS 4(R3), F1              // F1 = w.imag
	FNEGS F1, F1                 // conjugate twiddle

	LSL  $3, R1, R4
	ADD  R8, R4, R4
	FMOVS 0(R4), F2              // a.real
	FMOVS 4(R4), F3              // a.imag

	LSL  $3, R2, R4
	ADD  R8, R4, R4
	FMOVS 0(R4), F4              // b.real
	FMOVS 4(R4), F5              // b.imag

	FMULS F0, F4, F6
	FMULS F1, F5, F7
	FSUBS F7, F6, F6             // wb.real

	FMULS F0, F5, F7
	FMULS F1, F4, F5
	FADDS F5, F7, F7             // wb.imag

	FADDS F6, F2, F0
	FADDS F7, F3, F1
	FSUBS F6, F2, F4
	FSUBS F7, F3, F5

	LSL  $3, R1, R4
	ADD  R8, R4, R4
	FMOVS F0, 0(R4)
	FMOVS F1, 4(R4)

	LSL  $3, R2, R4
	ADD  R8, R4, R4
	FMOVS F4, 0(R4)
	FMOVS F5, 4(R4)

	ADD  $1, R0, R0
	B    inv_inner_loop

inv_next_base:
	ADD  R14, R17, R17
	B    inv_base_loop

inv_next_size:
	LSL  $1, R14, R14
	B    inv_size_loop

inv_transform_done:
	// -----------------------------------------------------------------------
	// PHASE 5: Copy result to destination if needed
	// -----------------------------------------------------------------------
	MOVD dst+0(FP), R0           // R0 = dst
	CMP  R8, R0
	BEQ  inv_scale               // Already in dst

	// Copy from scratch (R8) to dst (R0)
	MOVD $0, R1                  // R1 = i = 0

inv_copy_loop:
	CMP  R13, R1
	BGE  inv_scale

	LSL  $3, R1, R2
	ADD  R8, R2, R3
	MOVD (R3), R4

	ADD  R0, R2, R3
	MOVD R4, (R3)

	ADD  $1, R1, R1
	B    inv_copy_loop

inv_scale:
	// -----------------------------------------------------------------------
	// PHASE 6: Scale by 1/n
	// -----------------------------------------------------------------------
	MOVD dst+0(FP), R0
	MOVD $0, R1

	MOVD $·neonOnes(SB), R2
	FMOVS 0(R2), F0              // F0 = 1.0
	MOVW R13, R3
	SCVTFWS R3, F1               // F1 = float32(n)
	FDIVS F1, F0, F0             // F0 = 1.0 / n

inv_scale_loop:
	CMP  R13, R1
	BGE  inv_scale_done

	LSL  $3, R1, R2
	ADD  R0, R2, R2
	FMOVS 0(R2), F2
	FMOVS 4(R2), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R2)
	FMOVS F3, 4(R2)

	ADD  $1, R1, R1
	B    inv_scale_loop

inv_scale_done:
	B    inv_return_true

inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// ===========================================================================
// forwardNEONComplex128Asm - Forward FFT for complex128 using NEON
// ===========================================================================


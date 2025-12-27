//go:build arm64 && fft_asm && !purego

// ===========================================================================
// NEON-optimized FFT Assembly for ARM64
// ===========================================================================
//
// This file implements high-performance FFT transforms using ARM NEON (Advanced SIMD)
// instructions for both complex64 (single-precision) and complex128 (double-precision).
//
// ALGORITHM: Decimation-in-Time (DIT) Cooley-Tukey (same as AVX2 implementation)
//
// NEON CHARACTERISTICS:
// - 128-bit registers (Q/V0-V31)
// - Process 2 complex64 or 1 complex128 per register
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

DATA ·neonOnes+0(SB)/4, $0x3f800000
DATA ·neonOnes+4(SB)/4, $0x3f800000
DATA ·neonOnes+8(SB)/4, $0x3f800000
DATA ·neonOnes+12(SB)/4, $0x3f800000
GLOBL ·neonOnes(SB), RODATA, $16

DATA ·neonSignImag+0(SB)/4, $0x00000000
DATA ·neonSignImag+4(SB)/4, $0x80000000
DATA ·neonSignImag+8(SB)/4, $0x00000000
DATA ·neonSignImag+12(SB)/4, $0x80000000
GLOBL ·neonSignImag(SB), RODATA, $16

DATA ·neonOne64+0(SB)/8, $0x3ff0000000000000
GLOBL ·neonOne64(SB), RODATA, $8

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
TEXT ·forwardNEONComplex64Asm(SB), NOSPLIT, $0-121
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
// func neonComplexMul2Asm(dst, a, b *complex64)
// a and b each point to 2 complex64 values (4 float32 lanes).
// dst receives the 2 complex64 results: dst[i] = a[i] * b[i].
TEXT ·neonComplexMul2Asm(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD a+8(FP), R1
	MOVD b+16(FP), R2

	// Load a and b as interleaved complex64 values.
	VLD1 (R1), [V0.S4]          // V0 = [ar0, ai0, ar1, ai1]
	VLD1 (R2), [V1.S4]          // V1 = [br0, bi0, br1, bi1]

	// Deinterleave into real/imag vectors (duplicated lanes).
	VUZP1 V0.S4, V0.S4, V2.S4   // V2 = [ar0, ar1, ar0, ar1]
	VUZP2 V0.S4, V0.S4, V3.S4   // V3 = [ai0, ai1, ai0, ai1]
	VUZP1 V1.S4, V1.S4, V4.S4   // V4 = [br0, br1, br0, br1]
	VUZP2 V1.S4, V1.S4, V5.S4   // V5 = [bi0, bi1, bi0, bi1]

	// real = ar*br - ai*bi
	VEOR V6.B16, V6.B16, V6.B16
	VFMLA V2.S4, V4.S4, V6.S4
	VFMLS V3.S4, V5.S4, V6.S4

	// imag = ar*bi + ai*br
	VEOR V7.B16, V7.B16, V7.B16
	VFMLA V2.S4, V5.S4, V7.S4
	VFMLA V3.S4, V4.S4, V7.S4

	// Re-interleave real/imag into complex lanes.
	VZIP1 V7.S4, V6.S4, V0.S4
	VST1 [V0.S4], (R0)
	RET

// ===========================================================================
// inverseNEONComplex64Asm - Inverse FFT for complex64 using NEON
// ===========================================================================
TEXT ·inverseNEONComplex64Asm(SB), NOSPLIT, $0-121
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
TEXT ·forwardNEONComplex128Asm(SB), NOSPLIT, $0-121
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
TEXT ·inverseNEONComplex128Asm(SB), NOSPLIT, $0-121
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
TEXT ·forwardNEONSize16Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVD dst+0(FP), R8           // R8 = dst pointer
	MOVD src+24(FP), R9          // R9 = src pointer
	MOVD twiddle+48(FP), R10     // R10 = twiddle pointer
	MOVD scratch+72(FP), R11     // R11 = scratch pointer
	MOVD bitrev+96(FP), R12      // R12 = bitrev pointer
	MOVD src+32(FP), R13         // R13 = n (should be 16)

	// Verify n == 16
	CMP  $16, R13
	BNE  neon16_return_false

	// Validate all slice lengths >= 16
	MOVD dst+8(FP), R0
	CMP  $16, R0
	BLT  neon16_return_false

	MOVD twiddle+56(FP), R0
	CMP  $16, R0
	BLT  neon16_return_false

	MOVD scratch+80(FP), R0
	CMP  $16, R0
	BLT  neon16_return_false

	MOVD bitrev+104(FP), R0
	CMP  $16, R0
	BLT  neon16_return_false

	// Select working buffer: in-place uses scratch, out-of-place uses dst
	CMP  R8, R9
	BNE  neon16_use_dst

	// In-place: use scratch as working buffer
	MOVD R11, R8
	B    neon16_bitrev

neon16_use_dst:
	// Out-of-place: use dst directly (R8 already set)

neon16_bitrev:
	// =======================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =======================================================================
	// For size 16, bitrev = [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
	// We use the precomputed bitrev slice for correctness.
	// Each complex64 is 8 bytes, each int in bitrev is 8 bytes.

	// Unroll all 16 loads
	MOVD (R12), R0               // R0 = bitrev[0]
	LSL  $3, R0, R0              // byte offset
	ADD  R9, R0, R0              // &src[bitrev[0]]
	MOVD (R0), R1
	MOVD R1, (R8)                // work[0]

	MOVD 8(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 8(R8)               // work[1]

	MOVD 16(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 16(R8)              // work[2]

	MOVD 24(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 24(R8)              // work[3]

	MOVD 32(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 32(R8)              // work[4]

	MOVD 40(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 40(R8)              // work[5]

	MOVD 48(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 48(R8)              // work[6]

	MOVD 56(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 56(R8)              // work[7]

	MOVD 64(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 64(R8)              // work[8]

	MOVD 72(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 72(R8)              // work[9]

	MOVD 80(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 80(R8)              // work[10]

	MOVD 88(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 88(R8)              // work[11]

	MOVD 96(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 96(R8)              // work[12]

	MOVD 104(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 104(R8)             // work[13]

	MOVD 112(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 112(R8)             // work[14]

	MOVD 120(R12), R0
	LSL  $3, R0, R0
	ADD  R9, R0, R0
	MOVD (R0), R1
	MOVD R1, 120(R8)             // work[15]

	// Load neonOnes constant for butterfly operations
	MOVD $·neonOnes(SB), R0
	VLD1 (R0), [V28.S4]          // V28 = [1.0, 1.0, 1.0, 1.0]

	// =======================================================================
	// STAGE 1: size=2, half=1, step=8
	// =======================================================================
	// 8 butterflies with pairs: (0,1), (2,3), (4,5), (6,7),
	//                           (8,9), (10,11), (12,13), (14,15)
	// All use twiddle[0] = (1, 0), so t = b * 1 = b
	// Butterfly: a' = a + b, b' = a - b (no complex multiply needed)

	// Load all 16 complex64 values into 8 NEON registers (2 complex64 each)
	VLD1 (R8), [V0.S4]           // V0 = [work[0], work[1]] = [a0, b0]
	ADD  $16, R8, R1
	VLD1 (R1), [V1.S4]           // V1 = [work[2], work[3]] = [a1, b1]
	ADD  $16, R1, R1
	VLD1 (R1), [V2.S4]           // V2 = [work[4], work[5]] = [a2, b2]
	ADD  $16, R1, R1
	VLD1 (R1), [V3.S4]           // V3 = [work[6], work[7]] = [a3, b3]
	ADD  $16, R1, R1
	VLD1 (R1), [V4.S4]           // V4 = [work[8], work[9]] = [a4, b4]
	ADD  $16, R1, R1
	VLD1 (R1), [V5.S4]           // V5 = [work[10], work[11]] = [a5, b5]
	ADD  $16, R1, R1
	VLD1 (R1), [V6.S4]           // V6 = [work[12], work[13]] = [a6, b6]
	ADD  $16, R1, R1
	VLD1 (R1), [V7.S4]           // V7 = [work[14], work[15]] = [a7, b7]

	// Stage 1: For each register V, we have [a, b] and need [a+b, a-b]
	// Use VEXT to create [b, a], then add/sub using VFMLA/VFMLS with ones vector
	// (Go assembler doesn't support VFADD/VFSUB directly)

	// V0 = [a0.r, a0.i, b0.r, b0.i] -> [a0+b0, a0-b0]
	VEXT $8, V0.B16, V0.B16, V16.B16  // V16 = [b0, a0]
	// V17 = V0 + V16 via VFMLA: dst = dst + a*1
	VMOV V0.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4      // V17 = V0 + V16*1 = a + b
	// V18 = V0 - V16 via VFMLS: dst = dst - a*1
	VMOV V0.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4      // V18 = V0 - V16*1 = a - b
	// Now extract: low 64 bits of V17 = a', low 64 bits of V18 = b'
	// Combine into V0 = [a', b']
	VMOV V17.D[0], V0.D[0]            // V0.D[0] = a' = a0+b0
	VMOV V18.D[0], V0.D[1]            // V0.D[1] = b' = a0-b0

	// Same pattern for V1-V7
	VEXT $8, V1.B16, V1.B16, V16.B16
	VMOV V1.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4
	VMOV V1.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4
	VMOV V17.D[0], V1.D[0]
	VMOV V18.D[0], V1.D[1]

	VEXT $8, V2.B16, V2.B16, V16.B16
	VMOV V2.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4
	VMOV V2.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4
	VMOV V17.D[0], V2.D[0]
	VMOV V18.D[0], V2.D[1]

	VEXT $8, V3.B16, V3.B16, V16.B16
	VMOV V3.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4
	VMOV V3.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4
	VMOV V17.D[0], V3.D[0]
	VMOV V18.D[0], V3.D[1]

	VEXT $8, V4.B16, V4.B16, V16.B16
	VMOV V4.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4
	VMOV V4.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4
	VMOV V17.D[0], V4.D[0]
	VMOV V18.D[0], V4.D[1]

	VEXT $8, V5.B16, V5.B16, V16.B16
	VMOV V5.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4
	VMOV V5.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4
	VMOV V17.D[0], V5.D[0]
	VMOV V18.D[0], V5.D[1]

	VEXT $8, V6.B16, V6.B16, V16.B16
	VMOV V6.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4
	VMOV V6.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4
	VMOV V17.D[0], V6.D[0]
	VMOV V18.D[0], V6.D[1]

	VEXT $8, V7.B16, V7.B16, V16.B16
	VMOV V7.B16, V17.B16
	VFMLA V16.S4, V28.S4, V17.S4
	VMOV V7.B16, V18.B16
	VFMLS V16.S4, V28.S4, V18.S4
	VMOV V17.D[0], V7.D[0]
	VMOV V18.D[0], V7.D[1]

	// =======================================================================
	// STAGE 2: size=4, half=2, step=4
	// =======================================================================
	// 4 groups of 2 butterflies: (0,2),(1,3), (4,6),(5,7), (8,10),(9,11), (12,14),(13,15)
	// j=0 uses twiddle[0]=(1,0), j=1 uses twiddle[4]=(0,-1)
	//
	// Current layout after stage 1:
	// V0 = [d0, d1], V1 = [d2, d3], V2 = [d4, d5], V3 = [d6, d7]
	// V4 = [d8, d9], V5 = [d10, d11], V6 = [d12, d13], V7 = [d14, d15]
	//
	// Need butterflies: (d0,d2), (d1,d3), (d4,d6), etc.
	// This means pairing V0 with V1, V2 with V3, V4 with V5, V6 with V7

	// Load twiddle factors for stage 2: tw[0] and tw[4]
	// Pack into V24 = [tw0, tw4] for the two butterflies per group
	VLD1.P 8(R10), [V24.D1]      // V24.D[0] = twiddle[0] = (1, 0)
	ADD    $24, R10, R0          // R0 = &twiddle[4] (skip tw[1,2,3])
	VLD1   (R0), [V25.D1]        // V25.D[0] = twiddle[4] = (0, -1)
	VMOV   V24.D[0], V24.D[0]    // tw0 in low
	VMOV   V25.D[0], V24.D[1]    // tw4 in high
	// V24 = [tw0, tw4]

	// Reset R10 to twiddle base for later stages
	MOVD twiddle+48(FP), R10

	// Group 1: V0 = [d0, d1] with V1 = [d2, d3]
	// a = [d0, d1], b = [d2, d3], w = [tw0, tw4]
	// t = w * b, a' = a + t, b' = a - t

	// Complex multiply: t = w * b
	// Deinterleave b into real/imag
	VUZP1 V1.S4, V1.S4, V16.S4   // V16 = [br0, br1, br0, br1]
	VUZP2 V1.S4, V1.S4, V17.S4   // V17 = [bi0, bi1, bi0, bi1]
	VUZP1 V24.S4, V24.S4, V18.S4 // V18 = [wr0, wr1, wr0, wr1]
	VUZP2 V24.S4, V24.S4, V19.S4 // V19 = [wi0, wi1, wi0, wi1]

	// t.real = br*wr - bi*wi
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4

	// t.imag = br*wi + bi*wr
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4

	// Reinterleave t
	VZIP1 V21.S4, V20.S4, V22.S4 // V22 = [t0.r, t0.i, t1.r, t1.i] = t

	// Butterfly: a' = a + t, b' = a - t (using VFMLA/VFMLS with ones vector)
	VMOV V0.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4  // V16 = V0 + V22*1 = a'
	VMOV V0.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4  // V17 = V0 - V22*1 = b'
	VMOV V16.B16, V0.B16          // V0 = new [d0, d1]
	VMOV V17.B16, V1.B16          // V1 = new [d2, d3]

	// Group 2: V2 = [d4, d5] with V3 = [d6, d7]
	VUZP1 V3.S4, V3.S4, V16.S4
	VUZP2 V3.S4, V3.S4, V17.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V2.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4
	VMOV V2.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4
	VMOV V16.B16, V2.B16
	VMOV V17.B16, V3.B16

	// Group 3: V4 = [d8, d9] with V5 = [d10, d11]
	VUZP1 V5.S4, V5.S4, V16.S4
	VUZP2 V5.S4, V5.S4, V17.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V4.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4
	VMOV V4.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4
	VMOV V16.B16, V4.B16
	VMOV V17.B16, V5.B16

	// Group 4: V6 = [d12, d13] with V7 = [d14, d15]
	VUZP1 V7.S4, V7.S4, V16.S4
	VUZP2 V7.S4, V7.S4, V17.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V6.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4
	VMOV V6.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4
	VMOV V16.B16, V6.B16
	VMOV V17.B16, V7.B16

	// =======================================================================
	// STAGE 3: size=8, half=4, step=2
	// =======================================================================
	// 2 groups of 4 butterflies:
	//   Group 1: (d0,d4), (d1,d5), (d2,d6), (d3,d7) with tw[0,2,4,6]
	//   Group 2: (d8,d12), (d9,d13), (d10,d14), (d11,d15) with tw[0,2,4,6]
	//
	// Current layout:
	// V0=[d0,d1], V1=[d2,d3], V2=[d4,d5], V3=[d6,d7]
	// V4=[d8,d9], V5=[d10,d11], V6=[d12,d13], V7=[d14,d15]
	//
	// Need butterflies between: (V0,V2), (V1,V3), (V4,V6), (V5,V7)
	// Twiddles: tw[0], tw[2], tw[4], tw[6] in pairs

	// Load twiddle factors for stage 3
	// V24 = [tw0, tw2], V25 = [tw4, tw6]
	VLD1.P 8(R10), [V24.D1]      // tw[0]
	ADD    $8, R10, R0           // skip tw[1]
	VLD1   (R0), [V25.D1]        // tw[2]
	VMOV   V25.D[0], V24.D[1]    // V24 = [tw0, tw2]
	ADD    $8, R0, R0            // R0 = &tw[3]
	ADD    $8, R0, R0            // R0 = &tw[4]
	VLD1.P 8(R0), [V25.D1]       // load tw[4], R0 = &tw[5]
	ADD    $8, R0, R0            // R0 = &tw[6]
	VLD1   (R0), [V26.D1]        // load tw[6]
	VMOV   V26.D[0], V25.D[1]    // V25 = [tw4, tw6]

	// Reset R10 for stage 4
	MOVD twiddle+48(FP), R10

	// Group 1a: V0 = [d0, d1] with V2 = [d4, d5], using V24 = [tw0, tw2]
	VUZP1 V2.S4, V2.S4, V16.S4
	VUZP2 V2.S4, V2.S4, V17.S4
	VUZP1 V24.S4, V24.S4, V18.S4
	VUZP2 V24.S4, V24.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V0.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4
	VMOV V0.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4
	VMOV V16.B16, V0.B16         // V0 = new [d0, d1]
	VMOV V17.B16, V2.B16         // V2 = new [d4, d5]

	// Group 1b: V1 = [d2, d3] with V3 = [d6, d7], using V25 = [tw4, tw6]
	VUZP1 V3.S4, V3.S4, V16.S4
	VUZP2 V3.S4, V3.S4, V17.S4
	VUZP1 V25.S4, V25.S4, V18.S4
	VUZP2 V25.S4, V25.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V1.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4
	VMOV V1.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4
	VMOV V16.B16, V1.B16         // V1 = new [d2, d3]
	VMOV V17.B16, V3.B16         // V3 = new [d6, d7]

	// Reload twiddles (same pattern for group 2)
	VLD1.P 8(R10), [V24.D1]      // load tw[0], R10 = &tw[1]
	ADD    $8, R10, R0           // R0 = &tw[2]
	VLD1   (R0), [V25.D1]        // load tw[2]
	VMOV   V25.D[0], V24.D[1]    // V24 = [tw[0], tw[2]]
	ADD    $8, R0, R0            // R0 = &tw[3]
	ADD    $8, R0, R0            // R0 = &tw[4]
	VLD1.P 8(R0), [V25.D1]       // load tw[4], R0 = &tw[5]
	ADD    $8, R0, R0            // R0 = &tw[6]
	VLD1   (R0), [V26.D1]        // load tw[6]
	VMOV   V26.D[0], V25.D[1]    // V25 = [tw[4], tw[6]]
	MOVD twiddle+48(FP), R10

	// Group 2a: V4 = [d8, d9] with V6 = [d12, d13], using V24 = [tw0, tw2]
	VUZP1 V6.S4, V6.S4, V16.S4
	VUZP2 V6.S4, V6.S4, V17.S4
	VUZP1 V24.S4, V24.S4, V18.S4
	VUZP2 V24.S4, V24.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V4.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4
	VMOV V4.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4
	VMOV V16.B16, V4.B16         // V4 = new [d8, d9]
	VMOV V17.B16, V6.B16         // V6 = new [d12, d13]

	// Group 2b: V5 = [d10, d11] with V7 = [d14, d15], using V25 = [tw4, tw6]
	VUZP1 V7.S4, V7.S4, V16.S4
	VUZP2 V7.S4, V7.S4, V17.S4
	VUZP1 V25.S4, V25.S4, V18.S4
	VUZP2 V25.S4, V25.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V5.B16, V16.B16
	VFMLA V22.S4, V28.S4, V16.S4
	VMOV V5.B16, V17.B16
	VFMLS V22.S4, V28.S4, V17.S4
	VMOV V16.B16, V5.B16         // V5 = new [d10, d11]
	VMOV V17.B16, V7.B16         // V7 = new [d14, d15]

	// =======================================================================
	// STAGE 4: size=16, half=8, step=1
	// =======================================================================
	// 8 butterflies: (d0,d8), (d1,d9), ..., (d7,d15)
	// Twiddles: tw[0..7]
	//
	// Current layout:
	// V0=[d0,d1], V1=[d2,d3], V2=[d4,d5], V3=[d6,d7]
	// V4=[d8,d9], V5=[d10,d11], V6=[d12,d13], V7=[d14,d15]
	//
	// Need butterflies between: (V0,V4), (V1,V5), (V2,V6), (V3,V7)
	// Twiddles: tw[0,1], tw[2,3], tw[4,5], tw[6,7]

	// Load twiddle factors for stage 4
	VLD1 (R10), [V24.S4]         // V24 = [tw0, tw1]
	ADD  $16, R10, R0
	VLD1 (R0), [V25.S4]          // V25 = [tw2, tw3]
	ADD  $16, R0, R0
	VLD1 (R0), [V26.S4]          // V26 = [tw4, tw5]
	ADD  $16, R0, R0
	VLD1 (R0), [V27.S4]          // V27 = [tw6, tw7]

	// Butterfly 1: V0 = [d0, d1] with V4 = [d8, d9], using V24 = [tw0, tw1]
	VUZP1 V4.S4, V4.S4, V16.S4
	VUZP2 V4.S4, V4.S4, V17.S4
	VUZP1 V24.S4, V24.S4, V18.S4
	VUZP2 V24.S4, V24.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V0.B16, V8.B16
	VFMLA V22.S4, V28.S4, V8.S4   // V8 = V0 + V22 = new [d0, d1]
	VMOV V0.B16, V12.B16
	VFMLS V22.S4, V28.S4, V12.S4  // V12 = V0 - V22 = new [d8, d9]

	// Butterfly 2: V1 = [d2, d3] with V5 = [d10, d11], using V25 = [tw2, tw3]
	VUZP1 V5.S4, V5.S4, V16.S4
	VUZP2 V5.S4, V5.S4, V17.S4
	VUZP1 V25.S4, V25.S4, V18.S4
	VUZP2 V25.S4, V25.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V1.B16, V9.B16
	VFMLA V22.S4, V28.S4, V9.S4   // V9 = V1 + V22 = new [d2, d3]
	VMOV V1.B16, V13.B16
	VFMLS V22.S4, V28.S4, V13.S4  // V13 = V1 - V22 = new [d10, d11]

	// Butterfly 3: V2 = [d4, d5] with V6 = [d12, d13], using V26 = [tw4, tw5]
	VUZP1 V6.S4, V6.S4, V16.S4
	VUZP2 V6.S4, V6.S4, V17.S4
	VUZP1 V26.S4, V26.S4, V18.S4
	VUZP2 V26.S4, V26.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V2.B16, V10.B16
	VFMLA V22.S4, V28.S4, V10.S4  // V10 = V2 + V22 = new [d4, d5]
	VMOV V2.B16, V14.B16
	VFMLS V22.S4, V28.S4, V14.S4  // V14 = V2 - V22 = new [d12, d13]

	// Butterfly 4: V3 = [d6, d7] with V7 = [d14, d15], using V27 = [tw6, tw7]
	VUZP1 V7.S4, V7.S4, V16.S4
	VUZP2 V7.S4, V7.S4, V17.S4
	VUZP1 V27.S4, V27.S4, V18.S4
	VUZP2 V27.S4, V27.S4, V19.S4
	VEOR V20.B16, V20.B16, V20.B16
	VFMLA V16.S4, V18.S4, V20.S4
	VFMLS V17.S4, V19.S4, V20.S4
	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.S4, V19.S4, V21.S4
	VFMLA V17.S4, V18.S4, V21.S4
	VZIP1 V21.S4, V20.S4, V22.S4
	VMOV V3.B16, V11.B16
	VFMLA V22.S4, V28.S4, V11.S4  // V11 = V3 + V22 = new [d6, d7]
	VMOV V3.B16, V15.B16
	VFMLS V22.S4, V28.S4, V15.S4  // V15 = V3 - V22 = new [d14, d15]

	// =======================================================================
	// Store final results to dst
	// =======================================================================
	// Results are now in V8-V15:
	// V8=[d0,d1], V9=[d2,d3], V10=[d4,d5], V11=[d6,d7]
	// V12=[d8,d9], V13=[d10,d11], V14=[d12,d13], V15=[d14,d15]

	MOVD dst+0(FP), R9           // R9 = dst pointer
	VST1 [V8.S4], (R9)           // dst[0-1]
	ADD  $16, R9, R9
	VST1 [V9.S4], (R9)           // dst[2-3]
	ADD  $16, R9, R9
	VST1 [V10.S4], (R9)          // dst[4-5]
	ADD  $16, R9, R9
	VST1 [V11.S4], (R9)          // dst[6-7]
	ADD  $16, R9, R9
	VST1 [V12.S4], (R9)          // dst[8-9]
	ADD  $16, R9, R9
	VST1 [V13.S4], (R9)          // dst[10-11]
	ADD  $16, R9, R9
	VST1 [V14.S4], (R9)          // dst[12-13]
	ADD  $16, R9, R9
	VST1 [V15.S4], (R9)          // dst[14-15]

	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon16_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// forwardNEONSize32Complex64Asm - Size-32 forward FFT (stub)
TEXT ·forwardNEONSize32Complex64Asm(SB), NOSPLIT, $0-121
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// forwardNEONSize64Complex64Asm - Size-64 forward FFT (stub)
TEXT ·forwardNEONSize64Complex64Asm(SB), NOSPLIT, $0-121
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// forwardNEONSize128Complex64Asm - Size-128 forward FFT (stub)
TEXT ·forwardNEONSize128Complex64Asm(SB), NOSPLIT, $0-121
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// ===========================================================================
// Inverse Size-Specific Kernels (complex64)
// ===========================================================================

// inverseNEONSize16Complex64Asm - Size-16 inverse FFT (stub)
TEXT ·inverseNEONSize16Complex64Asm(SB), NOSPLIT, $0-121
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// inverseNEONSize32Complex64Asm - Size-32 inverse FFT (stub)
TEXT ·inverseNEONSize32Complex64Asm(SB), NOSPLIT, $0-121
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// inverseNEONSize64Complex64Asm - Size-64 inverse FFT (stub)
TEXT ·inverseNEONSize64Complex64Asm(SB), NOSPLIT, $0-121
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// inverseNEONSize128Complex64Asm - Size-128 inverse FFT (stub)
TEXT ·inverseNEONSize128Complex64Asm(SB), NOSPLIT, $0-121
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

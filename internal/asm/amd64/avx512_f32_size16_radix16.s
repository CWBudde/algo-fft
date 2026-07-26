//go:build amd64 && !purego

// ===========================================================================
// AVX-512 size-16 complex64 FFT codelets - radix-16 (4x4) decomposition
// ===========================================================================
//
// 16 complex64 = 128 bytes = exactly two ZMM registers, so the whole transform
// is register resident: memory is touched only for the two loads, the two
// stores, and the read-only constant tables.
//
// ALGORITHM
// ---------
// This is the same 4x4 Cooley-Tukey radix-16 decomposition as
// avx2_f32_size16_radix16.s (vertical FFT4 -> internal twiddles ->
// horizontal FFT4 -> transpose), but expressed as four radix-2
// decimation-in-frequency layers so that every twiddle multiply becomes a
// single full-width vector multiply and the final transpose collapses into the
// last layer's operand permute:
//
//   layer 1 (half = 8): pairs (i, i+8),  twiddle W16^i        }  vertical FFT4
//   layer 2 (half = 4): pairs (i, i+4),  twiddle W8^i         }  + W16^(r*c)
//   layer 3 (half = 2): pairs (i, i+2),  twiddle W4^i         }  horizontal
//   layer 4 (half = 1): pairs (i, i+1),  twiddle 1            }  FFT4
//
// Layers 1+2 together are exactly the radix-4 butterfly over
// {a[i], a[i+4], a[i+8], a[i+12]} followed by the W16^(k*i) twiddles, i.e. the
// "vertical FFT4 + internal twiddle" steps; layers 3+4 are the radix-4
// butterfly within each group of four, i.e. the "horizontal FFT4".
// DIF leaves the result in digit-reversed order; because layer 4 has a unit
// twiddle its operands can be gathered with the digit-reversal permutation
// already applied, so the transpose costs nothing beyond the two permutes that
// pair up the operands anyway.
//
// LANE BOOKKEEPING (one ZMM = 8 complex64 lanes = 8 qwords)
// ---------------------------------------------------------
//   after layer 1:  Z3 = pos 0..7          Z5 = pos 8..15
//   after layer 2:  Z3 = pos 0,1,2,3, 8,9,10,11    Z5 = pos 4..7, 12..15
//   after layer 3:  Z3 = pos 0,1, 8,9, 4,5, 12,13  Z5 = pos 2,3, 10,11, 6,7, 14,15
//   after layer 4:  Z8 = X[0..7]           Z9 = X[8..15]
// The layer-4 operand permutes pick the even/odd qwords of (Z3, Z5), which
// yields the "a" operands in order pos 0,8,4,12,2,10,6,14 - precisely
// bitrev(0..7) - so the sums land in natural order.
//
// TWIDDLES
// --------
// The twiddle factors of a size-16 transform are fixed, so they are embedded as
// real- and imaginary-duplicated constant vectors instead of being gathered
// from the caller's table with strided loads (which is what costs the AVX2
// kernel most of its shuffle throughput). The values are the float32 roundings
// of exp(-2*pi*i*k/16), bit-identical to internal/math.ComputeTwiddleFactors,
// so results match the table-driven siblings. len(twiddle) is still validated
// so the rejection behaviour matches the AVX2 codelet at this size.
//
// COMPLEX MULTIPLY (per ZMM, 8 complex64)
// ---------------------------------------
//   VSHUFPS $0xB1        x -> [x.i, x.r, ...]
//   VMULPS  w_im         -> [x.i*w.i, x.r*w.i, ...]
//   VFMADDSUB231PS w_re  -> even: x.r*w.r - x.i*w.i,  odd: x.i*w.r + x.r*w.i
// The inverse uses VFMSUBADD231PS with the same constants, which evaluates
// conj(w)*x, and scales the result by 1/16.
//
// Requires AVX512F only; callers gate on cpu.Features.HasAVX512.
// Plan 9 operand order is src, dst: VSUBPS b, a, dst computes dst = a - b.
//
// ===========================================================================

#include "textflag.h"

// ---------------------------------------------------------------------------
// Layer 1 twiddles: W16^j for j = 0..7 (real / imaginary duplicated per lane)
// ---------------------------------------------------------------------------
DATA avx512c64s16_w1_re<>+0(SB)/8, $0x3f8000003f800000  // lane 0: re = 1
DATA avx512c64s16_w1_re<>+8(SB)/8, $0x3f6c835e3f6c835e  // lane 1: re = 0.9238795
DATA avx512c64s16_w1_re<>+16(SB)/8, $0x3f3504f33f3504f3 // lane 2: re = 0.70710677
DATA avx512c64s16_w1_re<>+24(SB)/8, $0x3ec3ef153ec3ef15 // lane 3: re = 0.38268343
DATA avx512c64s16_w1_re<>+32(SB)/8, $0x248d3132248d3132 // lane 4: re = 6.123234e-17
DATA avx512c64s16_w1_re<>+40(SB)/8, $0xbec3ef15bec3ef15 // lane 5: re = -0.38268343
DATA avx512c64s16_w1_re<>+48(SB)/8, $0xbf3504f3bf3504f3 // lane 6: re = -0.70710677
DATA avx512c64s16_w1_re<>+56(SB)/8, $0xbf6c835ebf6c835e // lane 7: re = -0.9238795
GLOBL avx512c64s16_w1_re<>(SB), RODATA|NOPTR, $64

DATA avx512c64s16_w1_im<>+0(SB)/8, $0x8000000080000000  // lane 0: im = -0
DATA avx512c64s16_w1_im<>+8(SB)/8, $0xbec3ef15bec3ef15  // lane 1: im = -0.38268343
DATA avx512c64s16_w1_im<>+16(SB)/8, $0xbf3504f3bf3504f3 // lane 2: im = -0.70710677
DATA avx512c64s16_w1_im<>+24(SB)/8, $0xbf6c835ebf6c835e // lane 3: im = -0.9238795
DATA avx512c64s16_w1_im<>+32(SB)/8, $0xbf800000bf800000 // lane 4: im = -1
DATA avx512c64s16_w1_im<>+40(SB)/8, $0xbf6c835ebf6c835e // lane 5: im = -0.9238795
DATA avx512c64s16_w1_im<>+48(SB)/8, $0xbf3504f3bf3504f3 // lane 6: im = -0.70710677
DATA avx512c64s16_w1_im<>+56(SB)/8, $0xbec3ef15bec3ef15 // lane 7: im = -0.38268343
GLOBL avx512c64s16_w1_im<>(SB), RODATA|NOPTR, $64

// ---------------------------------------------------------------------------
// Layer 2 twiddles: W8^(j mod 4) = W16^(2*(j mod 4)), j = 0..7
// ---------------------------------------------------------------------------
DATA avx512c64s16_w2_re<>+0(SB)/8, $0x3f8000003f800000  // lane 0: re = 1
DATA avx512c64s16_w2_re<>+8(SB)/8, $0x3f3504f33f3504f3  // lane 1: re = 0.70710677
DATA avx512c64s16_w2_re<>+16(SB)/8, $0x248d3132248d3132 // lane 2: re = 6.123234e-17
DATA avx512c64s16_w2_re<>+24(SB)/8, $0xbf3504f3bf3504f3 // lane 3: re = -0.70710677
DATA avx512c64s16_w2_re<>+32(SB)/8, $0x3f8000003f800000 // lane 4: re = 1
DATA avx512c64s16_w2_re<>+40(SB)/8, $0x3f3504f33f3504f3 // lane 5: re = 0.70710677
DATA avx512c64s16_w2_re<>+48(SB)/8, $0x248d3132248d3132 // lane 6: re = 6.123234e-17
DATA avx512c64s16_w2_re<>+56(SB)/8, $0xbf3504f3bf3504f3 // lane 7: re = -0.70710677
GLOBL avx512c64s16_w2_re<>(SB), RODATA|NOPTR, $64

DATA avx512c64s16_w2_im<>+0(SB)/8, $0x8000000080000000  // lane 0: im = -0
DATA avx512c64s16_w2_im<>+8(SB)/8, $0xbf3504f3bf3504f3  // lane 1: im = -0.70710677
DATA avx512c64s16_w2_im<>+16(SB)/8, $0xbf800000bf800000 // lane 2: im = -1
DATA avx512c64s16_w2_im<>+24(SB)/8, $0xbf3504f3bf3504f3 // lane 3: im = -0.70710677
DATA avx512c64s16_w2_im<>+32(SB)/8, $0x8000000080000000 // lane 4: im = -0
DATA avx512c64s16_w2_im<>+40(SB)/8, $0xbf3504f3bf3504f3 // lane 5: im = -0.70710677
DATA avx512c64s16_w2_im<>+48(SB)/8, $0xbf800000bf800000 // lane 6: im = -1
DATA avx512c64s16_w2_im<>+56(SB)/8, $0xbf3504f3bf3504f3 // lane 7: im = -0.70710677
GLOBL avx512c64s16_w2_im<>(SB), RODATA|NOPTR, $64

// ---------------------------------------------------------------------------
// Layer 3 twiddles: W4^(j mod 2) = W16^(4*(j mod 2)), j = 0..7 (1, -i, 1, -i, ...)
// ---------------------------------------------------------------------------
DATA avx512c64s16_w3_re<>+0(SB)/8, $0x3f8000003f800000  // lane 0: re = 1
DATA avx512c64s16_w3_re<>+8(SB)/8, $0x248d3132248d3132  // lane 1: re = 6.123234e-17
DATA avx512c64s16_w3_re<>+16(SB)/8, $0x3f8000003f800000 // lane 2: re = 1
DATA avx512c64s16_w3_re<>+24(SB)/8, $0x248d3132248d3132 // lane 3: re = 6.123234e-17
DATA avx512c64s16_w3_re<>+32(SB)/8, $0x3f8000003f800000 // lane 4: re = 1
DATA avx512c64s16_w3_re<>+40(SB)/8, $0x248d3132248d3132 // lane 5: re = 6.123234e-17
DATA avx512c64s16_w3_re<>+48(SB)/8, $0x3f8000003f800000 // lane 6: re = 1
DATA avx512c64s16_w3_re<>+56(SB)/8, $0x248d3132248d3132 // lane 7: re = 6.123234e-17
GLOBL avx512c64s16_w3_re<>(SB), RODATA|NOPTR, $64

DATA avx512c64s16_w3_im<>+0(SB)/8, $0x8000000080000000  // lane 0: im = -0
DATA avx512c64s16_w3_im<>+8(SB)/8, $0xbf800000bf800000  // lane 1: im = -1
DATA avx512c64s16_w3_im<>+16(SB)/8, $0x8000000080000000 // lane 2: im = -0
DATA avx512c64s16_w3_im<>+24(SB)/8, $0xbf800000bf800000 // lane 3: im = -1
DATA avx512c64s16_w3_im<>+32(SB)/8, $0x8000000080000000 // lane 4: im = -0
DATA avx512c64s16_w3_im<>+40(SB)/8, $0xbf800000bf800000 // lane 5: im = -1
DATA avx512c64s16_w3_im<>+48(SB)/8, $0x8000000080000000 // lane 6: im = -0
DATA avx512c64s16_w3_im<>+56(SB)/8, $0xbf800000bf800000 // lane 7: im = -1
GLOBL avx512c64s16_w3_im<>(SB), RODATA|NOPTR, $64

// ---------------------------------------------------------------------------
// Layer 4 operand permutations (qword indices; 0..7 select the first source,
// 8..15 the second). Picking the even qwords gathers the layer-4 "a" operands
// in digit-reversed order pos 0,8,4,12,2,10,6,14 = bitrev(0..7); the odd
// qwords gather the "b" operands pos 1,9,5,13,3,11,7,15 = bitrev(8..15).
// ---------------------------------------------------------------------------
DATA avx512c64s16_pe<>+0(SB)/8, $0
DATA avx512c64s16_pe<>+8(SB)/8, $2
DATA avx512c64s16_pe<>+16(SB)/8, $4
DATA avx512c64s16_pe<>+24(SB)/8, $6
DATA avx512c64s16_pe<>+32(SB)/8, $8
DATA avx512c64s16_pe<>+40(SB)/8, $10
DATA avx512c64s16_pe<>+48(SB)/8, $12
DATA avx512c64s16_pe<>+56(SB)/8, $14
GLOBL avx512c64s16_pe<>(SB), RODATA|NOPTR, $64

DATA avx512c64s16_po<>+0(SB)/8, $1
DATA avx512c64s16_po<>+8(SB)/8, $3
DATA avx512c64s16_po<>+16(SB)/8, $5
DATA avx512c64s16_po<>+24(SB)/8, $7
DATA avx512c64s16_po<>+32(SB)/8, $9
DATA avx512c64s16_po<>+40(SB)/8, $11
DATA avx512c64s16_po<>+48(SB)/8, $13
DATA avx512c64s16_po<>+56(SB)/8, $15
GLOBL avx512c64s16_po<>(SB), RODATA|NOPTR, $64

// ===========================================================================
// ForwardAVX512Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64) bool
// ===========================================================================
// Frame $0-97:
//   dst     FP+0  ptr, FP+8  len, FP+16 cap
//   src     FP+24 ptr, FP+32 len, FP+40 cap
//   twiddle FP+48 ptr, FP+56 len, FP+64 cap
//   scratch FP+72 ptr, FP+80 len, FP+88 cap
//   ret     FP+96 bool
//
// The transform is register resident, so dst == src (in place) needs no
// scratch: src is fully consumed into Z0/Z1 before the first store.
TEXT ·ForwardAVX512Size16Radix16Complex64Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8      // R8 = destination pointer
	MOVQ src+24(FP), R9     // R9 = source pointer

	MOVQ src_len+32(FP), AX // AX = len(src)
	CMPQ AX, $16            // this codelet handles exactly 16 points
	JNE  fwd16r16_false     // any other length: reject, caller falls back

	MOVQ dst_len+8(FP), AX  // AX = len(dst)
	CMPQ AX, $16
	JL   fwd16r16_false     // destination too short

	MOVQ twiddle_len+56(FP), AX // AX = len(twiddle)
	CMPQ AX, $16
	JL   fwd16r16_false        // twiddle table too short

	MOVQ scratch_len+80(FP), AX // AX = len(scratch)
	CMPQ AX, $16
	JL   fwd16r16_false        // scratch too short

	VMOVUPS 0(R9), Z0  // Z0 = a[0..7]
	VMOVUPS 64(R9), Z1 // Z1 = a[8..15]

	// -----------------------------------------------------------------------
	// Layer 1: half = 8, pairs (i, i+8), twiddle W16^i. Lane aligned across
	// the two registers, so no permute is needed.
	// -----------------------------------------------------------------------
	VSUBPS  Z1, Z0, Z2                            // Z2 = a[i] - a[i+8]
	VADDPS  Z1, Z0, Z3                            // Z3 = a[i] + a[i+8]  -> pos 0..7
	VSHUFPS $0xB1, Z2, Z2, Z4                     // Z4 = swap re/im of the difference
	VMULPS  avx512c64s16_w1_im<>(SB), Z4, Z5      // Z5 = [d.i*w.i, d.r*w.i, ...]
	VFMADDSUB231PS avx512c64s16_w1_re<>(SB), Z2, Z5 // Z5 = W16^i * d       -> pos 8..15

	// -----------------------------------------------------------------------
	// Layer 2: half = 4, pairs (i, i+4) inside each half, twiddle W8^i.
	// Gather the 128-bit lanes so partners line up: Z0 = pos 0-3, 8-11 and
	// Z1 = pos 4-7, 12-15.
	// -----------------------------------------------------------------------
	VSHUFF64X2 $0x44, Z5, Z3, Z0                  // Z0 = [Z3.L0,Z3.L1, Z5.L0,Z5.L1] = pos 0-3, 8-11
	VSHUFF64X2 $0xEE, Z5, Z3, Z1                  // Z1 = [Z3.L2,Z3.L3, Z5.L2,Z5.L3] = pos 4-7, 12-15
	VSUBPS  Z1, Z0, Z2                            // Z2 = a - b
	VADDPS  Z1, Z0, Z3                            // Z3 = a + b            -> pos 0-3, 8-11
	VSHUFPS $0xB1, Z2, Z2, Z4                     // Z4 = swap re/im of the difference
	VMULPS  avx512c64s16_w2_im<>(SB), Z4, Z5      // Z5 = [d.i*w.i, d.r*w.i, ...]
	VFMADDSUB231PS avx512c64s16_w2_re<>(SB), Z2, Z5 // Z5 = W8^i * d        -> pos 4-7, 12-15

	// -----------------------------------------------------------------------
	// Layer 3: half = 2, pairs (i, i+2) inside each group of four,
	// twiddle W4^i = 1, -i alternating.
	// -----------------------------------------------------------------------
	VSHUFF64X2 $0x88, Z5, Z3, Z0                  // Z0 = [Z3.L0,Z3.L2, Z5.L0,Z5.L2] = pos 0,1, 8,9, 4,5, 12,13
	VSHUFF64X2 $0xDD, Z5, Z3, Z1                  // Z1 = [Z3.L1,Z3.L3, Z5.L1,Z5.L3] = pos 2,3, 10,11, 6,7, 14,15
	VSUBPS  Z1, Z0, Z2                            // Z2 = a - b
	VADDPS  Z1, Z0, Z3                            // Z3 = a + b            -> pos 0,1, 8,9, 4,5, 12,13
	VSHUFPS $0xB1, Z2, Z2, Z4                     // Z4 = swap re/im of the difference
	VMULPS  avx512c64s16_w3_im<>(SB), Z4, Z5      // Z5 = [d.i*w.i, d.r*w.i, ...]
	VFMADDSUB231PS avx512c64s16_w3_re<>(SB), Z2, Z5 // Z5 = W4^i * d        -> pos 2,3, 10,11, 6,7, 14,15

	// -----------------------------------------------------------------------
	// Layer 4: half = 1, twiddle 1. Gathering the even/odd qwords of (Z3, Z5)
	// applies the digit reversal at the same time, so the sums and differences
	// land directly in natural order.
	// -----------------------------------------------------------------------
	VMOVUPS  avx512c64s16_pe<>(SB), Z6            // Z6 = even-qword index table
	VPERMI2Q Z5, Z3, Z6                           // Z6 = a operands: pos 0,8,4,12,2,10,6,14
	VMOVUPS  avx512c64s16_po<>(SB), Z7            // Z7 = odd-qword index table
	VPERMI2Q Z5, Z3, Z7                           // Z7 = b operands: pos 1,9,5,13,3,11,7,15
	VADDPS   Z7, Z6, Z8                           // Z8 = X[0..7]
	VSUBPS   Z7, Z6, Z9                           // Z9 = X[8..15]

	VMOVUPS Z8, 0(R8)  // store X[0..7]
	VMOVUPS Z9, 64(R8) // store X[8..15]

	VZEROUPPER          // avoid AVX-SSE transition penalties
	MOVB $1, ret+96(FP) // handled
	RET

fwd16r16_false:
	MOVB $0, ret+96(FP) // not handled, caller falls back
	RET

// ===========================================================================
// InverseAVX512Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64) bool
// ===========================================================================
// Identical to the forward kernel except that every twiddle multiply uses
// VFMSUBADD231PS, which evaluates conj(w)*d with the same constant tables, and
// the result is scaled by 1/16.
TEXT ·InverseAVX512Size16Radix16Complex64Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8      // R8 = destination pointer
	MOVQ src+24(FP), R9     // R9 = source pointer

	MOVQ src_len+32(FP), AX // AX = len(src)
	CMPQ AX, $16            // this codelet handles exactly 16 points
	JNE  inv16r16_false     // any other length: reject, caller falls back

	MOVQ dst_len+8(FP), AX  // AX = len(dst)
	CMPQ AX, $16
	JL   inv16r16_false     // destination too short

	MOVQ twiddle_len+56(FP), AX // AX = len(twiddle)
	CMPQ AX, $16
	JL   inv16r16_false        // twiddle table too short

	MOVQ scratch_len+80(FP), AX // AX = len(scratch)
	CMPQ AX, $16
	JL   inv16r16_false        // scratch too short

	VMOVUPS 0(R9), Z0  // Z0 = a[0..7]
	VMOVUPS 64(R9), Z1 // Z1 = a[8..15]

	// --- Layer 1: half = 8, conjugated twiddle conj(W16^i) -----------------
	VSUBPS  Z1, Z0, Z2                            // Z2 = a[i] - a[i+8]
	VADDPS  Z1, Z0, Z3                            // Z3 = a[i] + a[i+8]  -> pos 0..7
	VSHUFPS $0xB1, Z2, Z2, Z4                     // Z4 = swap re/im of the difference
	VMULPS  avx512c64s16_w1_im<>(SB), Z4, Z5      // Z5 = [d.i*w.i, d.r*w.i, ...]
	VFMSUBADD231PS avx512c64s16_w1_re<>(SB), Z2, Z5 // Z5 = conj(W16^i) * d -> pos 8..15

	// --- Layer 2: half = 4, conjugated twiddle conj(W8^i) ------------------
	VSHUFF64X2 $0x44, Z5, Z3, Z0                  // Z0 = pos 0-3, 8-11
	VSHUFF64X2 $0xEE, Z5, Z3, Z1                  // Z1 = pos 4-7, 12-15
	VSUBPS  Z1, Z0, Z2                            // Z2 = a - b
	VADDPS  Z1, Z0, Z3                            // Z3 = a + b            -> pos 0-3, 8-11
	VSHUFPS $0xB1, Z2, Z2, Z4                     // Z4 = swap re/im of the difference
	VMULPS  avx512c64s16_w2_im<>(SB), Z4, Z5      // Z5 = [d.i*w.i, d.r*w.i, ...]
	VFMSUBADD231PS avx512c64s16_w2_re<>(SB), Z2, Z5 // Z5 = conj(W8^i) * d  -> pos 4-7, 12-15

	// --- Layer 3: half = 2, conjugated twiddle conj(W4^i) = 1, +i, ... -----
	VSHUFF64X2 $0x88, Z5, Z3, Z0                  // Z0 = pos 0,1, 8,9, 4,5, 12,13
	VSHUFF64X2 $0xDD, Z5, Z3, Z1                  // Z1 = pos 2,3, 10,11, 6,7, 14,15
	VSUBPS  Z1, Z0, Z2                            // Z2 = a - b
	VADDPS  Z1, Z0, Z3                            // Z3 = a + b            -> pos 0,1, 8,9, 4,5, 12,13
	VSHUFPS $0xB1, Z2, Z2, Z4                     // Z4 = swap re/im of the difference
	VMULPS  avx512c64s16_w3_im<>(SB), Z4, Z5      // Z5 = [d.i*w.i, d.r*w.i, ...]
	VFMSUBADD231PS avx512c64s16_w3_re<>(SB), Z2, Z5 // Z5 = conj(W4^i) * d  -> pos 2,3, 10,11, 6,7, 14,15

	// --- Layer 4 (twiddle 1) fused with the digit reversal, then 1/16 ------
	VMOVUPS  avx512c64s16_pe<>(SB), Z6            // Z6 = even-qword index table
	VPERMI2Q Z5, Z3, Z6                           // Z6 = a operands: pos 0,8,4,12,2,10,6,14
	VMOVUPS  avx512c64s16_po<>(SB), Z7            // Z7 = odd-qword index table
	VPERMI2Q Z5, Z3, Z7                           // Z7 = b operands: pos 1,9,5,13,3,11,7,15
	VADDPS   Z7, Z6, Z8                           // Z8 = unscaled x[0..7]
	VSUBPS   Z7, Z6, Z9                           // Z9 = unscaled x[8..15]

	VMULPS.BCST ·sixteenth32(SB), Z8, Z8          // Z8 *= 1/16 (embedded broadcast)
	VMULPS.BCST ·sixteenth32(SB), Z9, Z9          // Z9 *= 1/16 (embedded broadcast)

	VMOVUPS Z8, 0(R8)  // store x[0..7]
	VMOVUPS Z9, 64(R8) // store x[8..15]

	VZEROUPPER          // avoid AVX-SSE transition penalties
	MOVB $1, ret+96(FP) // handled
	RET

inv16r16_false:
	MOVB $0, ret+96(FP) // not handled, caller falls back
	RET

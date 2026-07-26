//go:build amd64 && !purego

// ===========================================================================
// AVX-512 Size-8 Radix-8 FFT (complex64) Kernels for AMD64
// ===========================================================================
//
// 8 x complex64 = 64 bytes = exactly one ZMM register, so the whole transform
// is register resident: one load, one store, no memory traffic in between and
// no scratch buffer (in-place works for free).
//
// WHY THIS LOOKS NOTHING LIKE avx2_f32_size8_radix8.s
// ---------------------------------------------------
// With the data in a single register every stage is a cross-lane operation, so
// width buys nothing here. The win has to come from arithmetic reduction, and
// at n = 8 every twiddle is trivial (+/-1, +/-i, +/-(1 +/- i)/sqrt2), so no
// general complex multiply is ever needed:
//
//   - Each of the three radix-2 stages is expressed as ONE signed cross-lane
//     permute pair plus ONE add:  V' = P_a(V, -V) + P_b(V, -V).  VPERMI2PS
//     takes two source tables, so feeding it V and -V lets a single instruction
//     emit an arbitrary permutation with arbitrary per-element sign.  That
//     folds, at zero extra cost, the bit-reversal, the butterfly sign pattern
//     and every multiplication by +/-i into the permute index tables.
//   - The only surviving multiplication is by 1/sqrt2, and it rides along in
//     the final VFMADD231PS as a per-lane coefficient vector.
//
// STRUCTURE (forward, DIT with bit-reversed input)
// -----------------------------------------------
//   bitrev(n=8)      = [0, 4, 2, 6, 1, 5, 3, 7]   (same table as the AVX2 file)
//   stage 1  S1 = [c0, c1, c2, -i*c3, c4, c5, c6, -i*c7]
//                  (bit-reversal + butterfly + stage 2's -i twiddle, folded)
//   stage 2  D  = [d0, d1, d2, d3, d4, d5, -i*d6, -i*d7]
//                  (butterfly + stage 3's -i twiddles, folded)
//   stage 3  y_k     = D_k + c_k * D_{k+4},  y_{k+4} = D_k - c_k * D_{k+4}
//            with c = [1, (1-i)/sqrt2, 1, (1-i)/sqrt2]; the remaining (1-i)
//            is split as 1 + (-i), i.e. two permuted terms, and the 1/sqrt2 is
//            the per-lane coefficient of the closing FMA.
//   Output is in natural order: no final permutation.
//
// The inverse uses the conjugated twiddles (+i and (1+i)/sqrt2) and folds the
// 1/8 normalisation into both the T1 scale and the FMA coefficient vector.
//
// The index tables below were derived and verified numerically against a naive
// DFT-8 (2000 random vectors, both directions) before this file was written.
//
// All instructions used here are AVX512F only (VPERMI2PS, VSHUFF32X4, masked
// VADDPS, VFMADD231PS); callers gate on cpu.Features.HasAVX512.
//
// ===========================================================================

#include "textflag.h"

// ---------------------------------------------------------------------------
// Permute index tables.  Index i < 16 selects float slot i of the first table
// (the value), i >= 16 selects slot i-16 of the second table (its negation).
// Float slot 2k / 2k+1 is the real / imaginary part of complex lane k.
// File-scoped (<>) so parallel work on other sizes cannot collide.
// ---------------------------------------------------------------------------

// permF1A = [0 1 0 1 4 5 5 12 2 3 2 3 6 7 7 14]
DATA permF1A<>+0(SB)/4, $0
DATA permF1A<>+4(SB)/4, $1
DATA permF1A<>+8(SB)/4, $0
DATA permF1A<>+12(SB)/4, $1
DATA permF1A<>+16(SB)/4, $4
DATA permF1A<>+20(SB)/4, $5
DATA permF1A<>+24(SB)/4, $5
DATA permF1A<>+28(SB)/4, $12
DATA permF1A<>+32(SB)/4, $2
DATA permF1A<>+36(SB)/4, $3
DATA permF1A<>+40(SB)/4, $2
DATA permF1A<>+44(SB)/4, $3
DATA permF1A<>+48(SB)/4, $6
DATA permF1A<>+52(SB)/4, $7
DATA permF1A<>+56(SB)/4, $7
DATA permF1A<>+60(SB)/4, $14
GLOBL permF1A<>(SB), RODATA|NOPTR, $64

// permF1B = [8 9 24 25 12 13 29 20 10 11 26 27 14 15 31 22]
DATA permF1B<>+0(SB)/4, $8
DATA permF1B<>+4(SB)/4, $9
DATA permF1B<>+8(SB)/4, $24
DATA permF1B<>+12(SB)/4, $25
DATA permF1B<>+16(SB)/4, $12
DATA permF1B<>+20(SB)/4, $13
DATA permF1B<>+24(SB)/4, $29
DATA permF1B<>+28(SB)/4, $20
DATA permF1B<>+32(SB)/4, $10
DATA permF1B<>+36(SB)/4, $11
DATA permF1B<>+40(SB)/4, $26
DATA permF1B<>+44(SB)/4, $27
DATA permF1B<>+48(SB)/4, $14
DATA permF1B<>+52(SB)/4, $15
DATA permF1B<>+56(SB)/4, $31
DATA permF1B<>+60(SB)/4, $22
GLOBL permF1B<>(SB), RODATA|NOPTR, $64

// permF2A = [0 1 2 3 0 1 2 3 8 9 10 11 9 12 11 14]
DATA permF2A<>+0(SB)/4, $0
DATA permF2A<>+4(SB)/4, $1
DATA permF2A<>+8(SB)/4, $2
DATA permF2A<>+12(SB)/4, $3
DATA permF2A<>+16(SB)/4, $0
DATA permF2A<>+20(SB)/4, $1
DATA permF2A<>+24(SB)/4, $2
DATA permF2A<>+28(SB)/4, $3
DATA permF2A<>+32(SB)/4, $8
DATA permF2A<>+36(SB)/4, $9
DATA permF2A<>+40(SB)/4, $10
DATA permF2A<>+44(SB)/4, $11
DATA permF2A<>+48(SB)/4, $9
DATA permF2A<>+52(SB)/4, $12
DATA permF2A<>+56(SB)/4, $11
DATA permF2A<>+60(SB)/4, $14
GLOBL permF2A<>(SB), RODATA|NOPTR, $64

// permF2B = [4 5 6 7 20 21 22 23 12 13 14 15 29 24 31 26]
DATA permF2B<>+0(SB)/4, $4
DATA permF2B<>+4(SB)/4, $5
DATA permF2B<>+8(SB)/4, $6
DATA permF2B<>+12(SB)/4, $7
DATA permF2B<>+16(SB)/4, $20
DATA permF2B<>+20(SB)/4, $21
DATA permF2B<>+24(SB)/4, $22
DATA permF2B<>+28(SB)/4, $23
DATA permF2B<>+32(SB)/4, $12
DATA permF2B<>+36(SB)/4, $13
DATA permF2B<>+40(SB)/4, $14
DATA permF2B<>+44(SB)/4, $15
DATA permF2B<>+48(SB)/4, $29
DATA permF2B<>+52(SB)/4, $24
DATA permF2B<>+56(SB)/4, $31
DATA permF2B<>+60(SB)/4, $26
GLOBL permF2B<>(SB), RODATA|NOPTR, $64

// permF3T = [0 0 11 26 0 0 15 30 0 0 27 10 0 0 31 14]
// Only complex lanes 1,3,5,7 (mask 0xCCCC) are consumed; the rest are ignored.
DATA permF3T<>+0(SB)/4, $0
DATA permF3T<>+4(SB)/4, $0
DATA permF3T<>+8(SB)/4, $11
DATA permF3T<>+12(SB)/4, $26
DATA permF3T<>+16(SB)/4, $0
DATA permF3T<>+20(SB)/4, $0
DATA permF3T<>+24(SB)/4, $15
DATA permF3T<>+28(SB)/4, $30
DATA permF3T<>+32(SB)/4, $0
DATA permF3T<>+36(SB)/4, $0
DATA permF3T<>+40(SB)/4, $27
DATA permF3T<>+44(SB)/4, $10
DATA permF3T<>+48(SB)/4, $0
DATA permF3T<>+52(SB)/4, $0
DATA permF3T<>+56(SB)/4, $31
DATA permF3T<>+60(SB)/4, $14
GLOBL permF3T<>(SB), RODATA|NOPTR, $64

// permI1A = [0 1 0 1 4 5 13 4 2 3 2 3 6 7 15 6]
DATA permI1A<>+0(SB)/4, $0
DATA permI1A<>+4(SB)/4, $1
DATA permI1A<>+8(SB)/4, $0
DATA permI1A<>+12(SB)/4, $1
DATA permI1A<>+16(SB)/4, $4
DATA permI1A<>+20(SB)/4, $5
DATA permI1A<>+24(SB)/4, $13
DATA permI1A<>+28(SB)/4, $4
DATA permI1A<>+32(SB)/4, $2
DATA permI1A<>+36(SB)/4, $3
DATA permI1A<>+40(SB)/4, $2
DATA permI1A<>+44(SB)/4, $3
DATA permI1A<>+48(SB)/4, $6
DATA permI1A<>+52(SB)/4, $7
DATA permI1A<>+56(SB)/4, $15
DATA permI1A<>+60(SB)/4, $6
GLOBL permI1A<>(SB), RODATA|NOPTR, $64

// permI1B = [8 9 24 25 12 13 21 28 10 11 26 27 14 15 23 30]
DATA permI1B<>+0(SB)/4, $8
DATA permI1B<>+4(SB)/4, $9
DATA permI1B<>+8(SB)/4, $24
DATA permI1B<>+12(SB)/4, $25
DATA permI1B<>+16(SB)/4, $12
DATA permI1B<>+20(SB)/4, $13
DATA permI1B<>+24(SB)/4, $21
DATA permI1B<>+28(SB)/4, $28
DATA permI1B<>+32(SB)/4, $10
DATA permI1B<>+36(SB)/4, $11
DATA permI1B<>+40(SB)/4, $26
DATA permI1B<>+44(SB)/4, $27
DATA permI1B<>+48(SB)/4, $14
DATA permI1B<>+52(SB)/4, $15
DATA permI1B<>+56(SB)/4, $23
DATA permI1B<>+60(SB)/4, $30
GLOBL permI1B<>(SB), RODATA|NOPTR, $64

// permI2A = [0 1 2 3 0 1 2 3 8 9 10 11 13 8 15 10]
DATA permI2A<>+0(SB)/4, $0
DATA permI2A<>+4(SB)/4, $1
DATA permI2A<>+8(SB)/4, $2
DATA permI2A<>+12(SB)/4, $3
DATA permI2A<>+16(SB)/4, $0
DATA permI2A<>+20(SB)/4, $1
DATA permI2A<>+24(SB)/4, $2
DATA permI2A<>+28(SB)/4, $3
DATA permI2A<>+32(SB)/4, $8
DATA permI2A<>+36(SB)/4, $9
DATA permI2A<>+40(SB)/4, $10
DATA permI2A<>+44(SB)/4, $11
DATA permI2A<>+48(SB)/4, $13
DATA permI2A<>+52(SB)/4, $8
DATA permI2A<>+56(SB)/4, $15
DATA permI2A<>+60(SB)/4, $10
GLOBL permI2A<>(SB), RODATA|NOPTR, $64

// permI2B = [4 5 6 7 20 21 22 23 12 13 14 15 25 28 27 30]
DATA permI2B<>+0(SB)/4, $4
DATA permI2B<>+4(SB)/4, $5
DATA permI2B<>+8(SB)/4, $6
DATA permI2B<>+12(SB)/4, $7
DATA permI2B<>+16(SB)/4, $20
DATA permI2B<>+20(SB)/4, $21
DATA permI2B<>+24(SB)/4, $22
DATA permI2B<>+28(SB)/4, $23
DATA permI2B<>+32(SB)/4, $12
DATA permI2B<>+36(SB)/4, $13
DATA permI2B<>+40(SB)/4, $14
DATA permI2B<>+44(SB)/4, $15
DATA permI2B<>+48(SB)/4, $25
DATA permI2B<>+52(SB)/4, $28
DATA permI2B<>+56(SB)/4, $27
DATA permI2B<>+60(SB)/4, $30
GLOBL permI2B<>(SB), RODATA|NOPTR, $64

// permI3T = [0 0 27 10 0 0 31 14 0 0 11 26 0 0 15 30]
DATA permI3T<>+0(SB)/4, $0
DATA permI3T<>+4(SB)/4, $0
DATA permI3T<>+8(SB)/4, $27
DATA permI3T<>+12(SB)/4, $10
DATA permI3T<>+16(SB)/4, $0
DATA permI3T<>+20(SB)/4, $0
DATA permI3T<>+24(SB)/4, $31
DATA permI3T<>+28(SB)/4, $14
DATA permI3T<>+32(SB)/4, $0
DATA permI3T<>+36(SB)/4, $0
DATA permI3T<>+40(SB)/4, $11
DATA permI3T<>+44(SB)/4, $26
DATA permI3T<>+48(SB)/4, $0
DATA permI3T<>+52(SB)/4, $0
DATA permI3T<>+56(SB)/4, $15
DATA permI3T<>+60(SB)/4, $30
GLOBL permI3T<>(SB), RODATA|NOPTR, $64

// ---------------------------------------------------------------------------
// Sign masks and twiddle coefficient vectors.  ZMM-wide (64 byte) variants of
// the scalar constants in core.s, which only exist at 4/16/32 bytes.
// ---------------------------------------------------------------------------

// negAll: float32 sign bit in all 16 slots (builds the -V permute table).
DATA negAll<>+0(SB)/8, $0x8000000080000000
DATA negAll<>+8(SB)/8, $0x8000000080000000
DATA negAll<>+16(SB)/8, $0x8000000080000000
DATA negAll<>+24(SB)/8, $0x8000000080000000
DATA negAll<>+32(SB)/8, $0x8000000080000000
DATA negAll<>+40(SB)/8, $0x8000000080000000
DATA negAll<>+48(SB)/8, $0x8000000080000000
DATA negAll<>+56(SB)/8, $0x8000000080000000
GLOBL negAll<>(SB), RODATA|NOPTR, $64

// negHalf: sign bit in slots 8..15 only (negates complex lanes 4..7, i.e. the
// upper 256 bits) for the last stage's subtract half.
DATA negHalf<>+0(SB)/8, $0x0000000000000000
DATA negHalf<>+8(SB)/8, $0x0000000000000000
DATA negHalf<>+16(SB)/8, $0x0000000000000000
DATA negHalf<>+24(SB)/8, $0x0000000000000000
DATA negHalf<>+32(SB)/8, $0x8000000080000000
DATA negHalf<>+40(SB)/8, $0x8000000080000000
DATA negHalf<>+48(SB)/8, $0x8000000080000000
DATA negHalf<>+56(SB)/8, $0x8000000080000000
GLOBL negHalf<>(SB), RODATA|NOPTR, $64

// coefF: last-stage per-lane coefficient, forward.
// 1.0f (0x3F800000) on complex lanes 0,2,4,6; 1/sqrt2 (0x3F3504F3) on 1,3,5,7.
DATA coefF<>+0(SB)/8, $0x3F8000003F800000
DATA coefF<>+8(SB)/8, $0x3F3504F33F3504F3
DATA coefF<>+16(SB)/8, $0x3F8000003F800000
DATA coefF<>+24(SB)/8, $0x3F3504F33F3504F3
DATA coefF<>+32(SB)/8, $0x3F8000003F800000
DATA coefF<>+40(SB)/8, $0x3F3504F33F3504F3
DATA coefF<>+48(SB)/8, $0x3F8000003F800000
DATA coefF<>+56(SB)/8, $0x3F3504F33F3504F3
GLOBL coefF<>(SB), RODATA|NOPTR, $64

// coefI: the same coefficients scaled by the inverse's 1/8 normalisation.
// 0.125f (0x3E000000) and 1/(8*sqrt2) (0x3DB504F3).
DATA coefI<>+0(SB)/8, $0x3E0000003E000000
DATA coefI<>+8(SB)/8, $0x3DB504F33DB504F3
DATA coefI<>+16(SB)/8, $0x3E0000003E000000
DATA coefI<>+24(SB)/8, $0x3DB504F33DB504F3
DATA coefI<>+32(SB)/8, $0x3E0000003E000000
DATA coefI<>+40(SB)/8, $0x3DB504F33DB504F3
DATA coefI<>+48(SB)/8, $0x3E0000003E000000
DATA coefI<>+56(SB)/8, $0x3DB504F33DB504F3
GLOBL coefI<>(SB), RODATA|NOPTR, $64

// ===========================================================================
// ForwardAVX512Size8Radix8Complex64Asm - forward size-8 radix-8, complex64
// ===========================================================================
TEXT ·ForwardAVX512Size8Radix8Complex64Asm(SB), NOSPLIT, $0-97
	MOVQ src_len+32(FP), R13 // R13 = len(src)
	CMPQ R13, $8             // this codelet handles exactly n == 8
	JNE  s8r8_512_fwd_false

	MOVQ dst_len+8(FP), AX
	CMPQ AX, $8
	JL   s8r8_512_fwd_false // dst too short

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $8
	JL   s8r8_512_fwd_false // twiddle too short (values are hardcoded, but the
	                        // contract is the same as every other size-8 codelet)

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $8
	JL   s8r8_512_fwd_false // scratch too short

	MOVQ dst+0(FP), R8  // R8 = dst pointer
	MOVQ src+24(FP), R9 // R9 = src pointer

	// The transform is register resident, so dst == src (in place) needs no
	// scratch buffer: the single store at the end happens after the last read.

	VMOVUPS permF1A<>(SB), Z10 // stage 1, first operand indices
	VMOVUPS permF1B<>(SB), Z11 // stage 1, second operand indices
	VMOVUPS permF2A<>(SB), Z12 // stage 2, first operand indices
	VMOVUPS permF2B<>(SB), Z13 // stage 2, second operand indices
	VMOVUPS permF3T<>(SB), Z14 // stage 3, third term indices

	MOVL  $0xCCCC, AX // complex lanes 1,3,5,7 = float slots 2,3,6,7,10,11,14,15
	KMOVW AX, K1

	VMOVUPS (R9), Z0 // Z0 = x0..x7, the entire input in one ZMM

	// ---- stage 1: bit-reversal + radix-2 butterfly + stage 2's -i twiddle ----
	VPXORD    negAll<>(SB), Z0, Z1 // Z1 = -Z0 (the negated permute table)
	VPERMI2PS Z1, Z0, Z10          // Z10 = A1 (indices < 16 -> Z0, >= 16 -> Z1)
	VPERMI2PS Z1, Z0, Z11          // Z11 = B1
	VADDPS    Z10, Z11, Z2         // Z2 = S1 = [c0,c1,c2,-i*c3,c4,c5,c6,-i*c7]

	// ---- stage 2: radix-2 butterfly + stage 3's -i twiddles ----
	VPXORD    negAll<>(SB), Z2, Z3 // Z3 = -S1
	VPERMI2PS Z3, Z2, Z12          // Z12 = A2
	VPERMI2PS Z3, Z2, Z13          // Z13 = B2
	VADDPS    Z12, Z13, Z4         // Z4 = D = [d0,d1,d2,d3,d4,d5,-i*d6,-i*d7]

	// ---- stage 3: y_k = D_k + c_k*D_{k+4}, y_{k+4} = D_k - c_k*D_{k+4} ----
	VPXORD     negAll<>(SB), Z4, Z5  // Z5 = -D
	VPERMI2PS  Z5, Z4, Z14           // Z14 = T3 = +/-(-i)*D5, +/-(-i)*D7
	VSHUFF32X4 $0x44, Z4, Z4, Z6     // Z6 = T1 = [D0,D1,D2,D3,D0,D1,D2,D3]
	VSHUFF32X4 $0xEE, Z4, Z4, Z7     // Z7 =      [D4,D5,D6,D7,D4,D5,D6,D7]
	VPXORD     negHalf<>(SB), Z7, Z7 // Z7 = T2 = [D4..D7, -D4..-D7]
	VADDPS     Z14, Z7, K1, Z7       // lanes 1,3,5,7: T2 += T3 -> (1-i)*D5, (1-i)*D7

	VFMADD231PS coefF<>(SB), Z7, Z6 // Z6 = coefF*T2 + T1 = y0..y7, natural order

	VMOVUPS Z6, (R8) // store the spectrum (safe in place: last read is done)

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

s8r8_512_fwd_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// InverseAVX512Size8Radix8Complex64Asm - inverse size-8 radix-8, complex64
// ===========================================================================
// Identical dataflow with conjugated twiddles (+i, (1+i)/sqrt2). The 1/8
// normalisation is folded into the closing FMA: T1 is scaled by 1/8 (off the
// critical path) and coefI already carries the 1/8 factor.
TEXT ·InverseAVX512Size8Radix8Complex64Asm(SB), NOSPLIT, $0-97
	MOVQ src_len+32(FP), R13 // R13 = len(src)
	CMPQ R13, $8             // this codelet handles exactly n == 8
	JNE  s8r8_512_inv_false

	MOVQ dst_len+8(FP), AX
	CMPQ AX, $8
	JL   s8r8_512_inv_false // dst too short

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $8
	JL   s8r8_512_inv_false // twiddle too short

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $8
	JL   s8r8_512_inv_false // scratch too short

	MOVQ dst+0(FP), R8  // R8 = dst pointer
	MOVQ src+24(FP), R9 // R9 = src pointer

	VMOVUPS permI1A<>(SB), Z10 // stage 1, first operand indices
	VMOVUPS permI1B<>(SB), Z11 // stage 1, second operand indices
	VMOVUPS permI2A<>(SB), Z12 // stage 2, first operand indices
	VMOVUPS permI2B<>(SB), Z13 // stage 2, second operand indices
	VMOVUPS permI3T<>(SB), Z14 // stage 3, third term indices

	MOVL  $0xCCCC, AX // complex lanes 1,3,5,7
	KMOVW AX, K1

	VMOVUPS (R9), Z0 // Z0 = X0..X7, the entire spectrum in one ZMM

	// ---- stage 1: bit-reversal + radix-2 butterfly + stage 2's +i twiddle ----
	VPXORD    negAll<>(SB), Z0, Z1 // Z1 = -Z0
	VPERMI2PS Z1, Z0, Z10          // Z10 = A1
	VPERMI2PS Z1, Z0, Z11          // Z11 = B1
	VADDPS    Z10, Z11, Z2         // Z2 = S1 = [c0,c1,c2,+i*c3,c4,c5,c6,+i*c7]

	// ---- stage 2: radix-2 butterfly + stage 3's +i twiddles ----
	VPXORD    negAll<>(SB), Z2, Z3 // Z3 = -S1
	VPERMI2PS Z3, Z2, Z12          // Z12 = A2
	VPERMI2PS Z3, Z2, Z13          // Z13 = B2
	VADDPS    Z12, Z13, Z4         // Z4 = D = [d0,d1,d2,d3,d4,d5,+i*d6,+i*d7]

	// ---- stage 3 + 1/8 normalisation ----
	VPXORD     negAll<>(SB), Z4, Z5  // Z5 = -D
	VPERMI2PS  Z5, Z4, Z14           // Z14 = T3 = +/-(+i)*D5, +/-(+i)*D7
	VSHUFF32X4 $0x44, Z4, Z4, Z6     // Z6 = [D0,D1,D2,D3,D0,D1,D2,D3]
	VMULPS.BCST ·eighth32(SB), Z6, Z6 // Z6 = T1/8 (off the critical path)
	VSHUFF32X4 $0xEE, Z4, Z4, Z7     // Z7 = [D4,D5,D6,D7,D4,D5,D6,D7]
	VPXORD     negHalf<>(SB), Z7, Z7 // Z7 = T2 = [D4..D7, -D4..-D7]
	VADDPS     Z14, Z7, K1, Z7       // lanes 1,3,5,7: T2 += T3 -> (1+i)*D5, (1+i)*D7

	VFMADD231PS coefI<>(SB), Z7, Z6 // Z6 = coefI*T2 + T1/8 = x0..x7, natural order

	VMOVUPS Z6, (R8) // store the samples (safe in place)

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

s8r8_512_inv_false:
	MOVB $0, ret+96(FP)
	RET

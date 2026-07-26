//go:build amd64 && !purego

// ===========================================================================
// AVX-512 Size-64 Radix-4 FFT Kernels for AMD64 - complex64
// ===========================================================================
//
// 64 complex64 = 512 bytes = eight ZMM registers, so the whole transform is
// register resident: eight loads at the top, eight stores at the bottom, and
// no memory traffic in between (the AVX2 codelets at this size have to spill
// between stages because 64 complex64 fills the entire YMM register file).
// Because every load happens before every store, dst == src works without a
// working buffer and scratch is never touched.
//
// ALGORITHM: 8x8 four-step (Cooley-Tukey with n1 = n2 = 8)
// -------------------------------------------------------
// Row r of the register file holds src[8r .. 8r+7], so with i = l + 8*r the
// register axis is the "outer" index and the lane axis the "inner" one:
//
//	X[k2 + 8*k1] = sum_l w8^(l*k1) * [ w^(l*k2) * sum_r x[l+8r]*w8^(r*k2) ]
//	               \______ step C ______/  \_ step B _/ \____ step A ____/
//
// with w = exp(-2*pi*i/64) and w8 = w^8 = exp(-2*pi*i/8).
//
//	step A    8-point DFT down the register axis (r -> k2). Vertical:
//	          elementwise across registers, no shuffles, and the twiddles are
//	          broadcast scalars read from tw[8] and tw[24].
//	step B    multiply row k2 by the vector w^(lane*k2) (RODATA, see
//	          avx512_f32_size64_tables.s). Row 0 is all ones and is skipped.
//	8x8 transp. move (row k2, lane l) to (row l, lane k2) so that step C is
//	          vertical too. Three bit-exchange levels of eight instructions:
//	          VUNPCK{L,H}PD, VPERMI2PD, VSHUFF64X2.
//	step C    8-point DFT down the register axis (l -> k1). The output row k1
//	          holds X[8*k1 .. 8*k1+7], i.e. natural order: no bit-reversal
//	          pass anywhere in this kernel.
//
// The 8-point sub-FFT is decomposed here as
// one radix-4 stage (two 4-point DFTs over the even and odd
// rows) followed by one radix-2 combine stage: stage radices 4,2 per half,
// 4,2,4,2 for the whole kernel.
//
// The inverse kernels are identical with conjugated twiddles
// (VFMSUBADD231PS instead of VFMADDSUB231PS, +i instead of -i via
// maskNegLoPS instead of maskNegHiPS) plus a final 1/64 scaling.
//
// Only AVX512F instructions are used (VPXORQ rather than the AVX512DQ-only
// VXORPS, VBROADCASTF32X4 rather than VBROADCASTF64X2), so the kernels are
// valid on every AVX-512 CPU, which is what the cpu.Features.HasAVX512 gate
// (CPUID.07H:EBX.AVX512F) promises.
//
// Register use: Z0-Z25 data/temporaries (SSA-allocated by the generator),
// Z26 1/64 (inverse only), Z27 sign mask, Z28-Z31 the w8 and w8^3 broadcasts.
// R8 dst, R9 src, R10 twiddle, R11 n, R12 transpose indices, R13 four-step
// twiddles.
//
// Requires AVX512F only; callers gate on cpu.Features.HasAVX512.
// ===========================================================================

#include "textflag.h"

// Forward transform, size 64, complex64, radix4
TEXT ·ForwardAVX512Size64Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ src_len+32(FP), R11 // R11 = n = len(src)

	// This codelet handles exactly n == 64.
	CMPQ R11, $64
	JNE  avx512_r4_64_fwd_false

	// Validate the remaining slice lengths.
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $64
	JL   avx512_r4_64_fwd_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $64
	JL   avx512_r4_64_fwd_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $64
	JL   avx512_r4_64_fwd_false

	// No working buffer is needed: all 64 values are loaded into ZMM
	// registers before the first store, so dst == src (in-place) is safe and
	// scratch stays untouched.

	VBROADCASTF32X4 ·maskNegHiPS(SB), Z27  // negate imag lane of each complex64 (multiply by -i after swap)
	VBROADCASTSS 64(R10), Z28              // Re(tw[8]) = Re(w8), w8 = exp(-2pi*i/8)
	VBROADCASTSS 68(R10), Z29              // Im(tw[8]) = Im(w8)
	VBROADCASTSS 192(R10), Z30             // Re(tw[24]) = Re(w8^3)
	VBROADCASTSS 196(R10), Z31             // Im(tw[24]) = Im(w8^3)
	LEAQ ·avx512F32Size64TransIdx(SB), R12 // R12 = transpose permute indices
	LEAQ ·avx512F32Size64CrossTw(SB), R13  // R13 = four-step twiddle vectors

	// load all 64 complex64 into 8 ZMM registers (row r = src[8r..8r+7])
	VMOVUPS (R9), Z0    // row 0 -> sub-FFT slot 0
	VMOVUPS 64(R9), Z1  // row 1 -> sub-FFT slot 1
	VMOVUPS 128(R9), Z2 // row 2 -> sub-FFT slot 2
	VMOVUPS 192(R9), Z3 // row 3 -> sub-FFT slot 3
	VMOVUPS 256(R9), Z4 // row 4 -> sub-FFT slot 4
	VMOVUPS 320(R9), Z5 // row 5 -> sub-FFT slot 5
	VMOVUPS 384(R9), Z6 // row 6 -> sub-FFT slot 6
	VMOVUPS 448(R9), Z7 // row 7 -> sub-FFT slot 7

	// ===== step A: 8-point DFT along the register axis (columns) =====
	VADDPS Z4, Z0, Z8           // A dft4(even): t0 = x0+x2
	VSUBPS Z4, Z0, Z9           // A dft4(even): t1 = x0-x2
	VADDPS Z6, Z2, Z10          // A dft4(even): t2 = x1+x3
	VSUBPS Z6, Z2, Z11          // A dft4(even): t3 = x1-x3
	VPERMILPS $0xB1, Z11, Z6    // A dft4(even): swap re/im of t3
	VPXORQ Z27, Z6, Z4          // A dft4(even): u = -i*t3 (fwd) / +i*t3 (inv)
	VADDPS Z10, Z8, Z11         // A dft4(even): X0 = t0+t2
	VSUBPS Z10, Z8, Z6          // A dft4(even): X2 = t0-t2
	VADDPS Z4, Z9, Z2           // A dft4(even): X1 = t1+u
	VSUBPS Z4, Z9, Z0           // A dft4(even): X3 = t1-u
	VADDPS Z5, Z1, Z4           // A dft4(odd): t0 = x0+x2
	VSUBPS Z5, Z1, Z10          // A dft4(odd): t1 = x0-x2
	VADDPS Z7, Z3, Z9           // A dft4(odd): t2 = x1+x3
	VSUBPS Z7, Z3, Z8           // A dft4(odd): t3 = x1-x3
	VPERMILPS $0xB1, Z8, Z7     // A dft4(odd): swap re/im of t3
	VPXORQ Z27, Z7, Z5          // A dft4(odd): u = -i*t3 (fwd) / +i*t3 (inv)
	VADDPS Z9, Z4, Z8           // A dft4(odd): X0 = t0+t2
	VSUBPS Z9, Z4, Z7           // A dft4(odd): X2 = t0-t2
	VADDPS Z5, Z10, Z3          // A dft4(odd): X1 = t1+u
	VSUBPS Z5, Z10, Z1          // A dft4(odd): X3 = t1-u
	VADDPS Z8, Z11, Z5          // A combine w^0: a+b
	VSUBPS Z8, Z11, Z9          // A combine w^0: a-b
	VPERMILPS $0xB1, Z3, Z8     // swap re/im
	VMULPS Z29, Z8, Z11         // t = swap(b)*Im(w8)
	VFMADDSUB231PS Z28, Z3, Z11 // t = w8*b
	VADDPS Z11, Z2, Z3          // A combine w^1: a+t
	VSUBPS Z11, Z2, Z8          // A combine w^1: a-t
	VPERMILPS $0xB1, Z7, Z11    // A combine w^2: swap re/im
	VPXORQ Z27, Z11, Z2         // A combine w^2: t = -i*b (fwd) / +i*b (inv)
	VADDPS Z2, Z6, Z7           // A combine w^2: a+t
	VSUBPS Z2, Z6, Z11          // A combine w^2: a-t
	VPERMILPS $0xB1, Z1, Z2     // swap re/im
	VMULPS Z31, Z2, Z6          // t = swap(b)*Im(w8^3)
	VFMADDSUB231PS Z30, Z1, Z6  // t = w8^3*b
	VADDPS Z6, Z0, Z1           // A combine w^3: a+t
	VSUBPS Z6, Z0, Z2           // A combine w^3: a-t

	// ===== step B: four-step twiddle, row k2 *= w^(lane*k2) =====
	VPERMILPS $0xB1, Z3, Z6          // swap re/im
	VMULPS 64(R13), Z6, Z0           // t = swap(b)*Im(w^(lane*1))
	VFMADDSUB231PS (R13), Z3, Z0     // t = w^(lane*1)*b
	VPERMILPS $0xB1, Z7, Z3          // swap re/im
	VMULPS 192(R13), Z3, Z6          // t = swap(b)*Im(w^(lane*2))
	VFMADDSUB231PS 128(R13), Z7, Z6  // t = w^(lane*2)*b
	VPERMILPS $0xB1, Z1, Z7          // swap re/im
	VMULPS 320(R13), Z7, Z3          // t = swap(b)*Im(w^(lane*3))
	VFMADDSUB231PS 256(R13), Z1, Z3  // t = w^(lane*3)*b
	VPERMILPS $0xB1, Z9, Z1          // swap re/im
	VMULPS 448(R13), Z1, Z7          // t = swap(b)*Im(w^(lane*4))
	VFMADDSUB231PS 384(R13), Z9, Z7  // t = w^(lane*4)*b
	VPERMILPS $0xB1, Z8, Z9          // swap re/im
	VMULPS 576(R13), Z9, Z1          // t = swap(b)*Im(w^(lane*5))
	VFMADDSUB231PS 512(R13), Z8, Z1  // t = w^(lane*5)*b
	VPERMILPS $0xB1, Z11, Z8         // swap re/im
	VMULPS 704(R13), Z8, Z9          // t = swap(b)*Im(w^(lane*6))
	VFMADDSUB231PS 640(R13), Z11, Z9 // t = w^(lane*6)*b
	VPERMILPS $0xB1, Z2, Z11         // swap re/im
	VMULPS 832(R13), Z11, Z8         // t = swap(b)*Im(w^(lane*7))
	VFMADDSUB231PS 768(R13), Z2, Z8  // t = w^(lane*7)*b

	// ===== 8x8 transpose: (row k2, lane l) -> (row l, lane k2) =====

	// transpose level 1: exchange register bit 0 with lane bit 0
	VUNPCKLPD Z0, Z5, Z2  // even lanes of rows 0,1
	VUNPCKHPD Z0, Z5, Z11 // odd lanes of rows 0,1
	VUNPCKLPD Z3, Z6, Z0  // even lanes of rows 2,3
	VUNPCKHPD Z3, Z6, Z5  // odd lanes of rows 2,3
	VUNPCKLPD Z1, Z7, Z3  // even lanes of rows 4,5
	VUNPCKHPD Z1, Z7, Z6  // odd lanes of rows 4,5
	VUNPCKLPD Z8, Z9, Z1  // even lanes of rows 6,7
	VUNPCKHPD Z8, Z9, Z7  // odd lanes of rows 6,7

	// transpose level 2: exchange register bit 1 with lane bit 1
	VMOVDQU64 (R12), Z8    // permute indices
	VPERMI2PD Z0, Z2, Z8   // 128-bit lanes 0,2 of rows 0,2
	VMOVDQU64 64(R12), Z9  // permute indices
	VPERMI2PD Z0, Z2, Z9   // 128-bit lanes 1,3 of rows 0,2
	VMOVDQU64 (R12), Z0    // permute indices
	VPERMI2PD Z5, Z11, Z0  // 128-bit lanes 0,2 of rows 1,3
	VMOVDQU64 64(R12), Z2  // permute indices
	VPERMI2PD Z5, Z11, Z2  // 128-bit lanes 1,3 of rows 1,3
	VMOVDQU64 (R12), Z5    // permute indices
	VPERMI2PD Z1, Z3, Z5   // 128-bit lanes 0,2 of rows 4,6
	VMOVDQU64 64(R12), Z11 // permute indices
	VPERMI2PD Z1, Z3, Z11  // 128-bit lanes 1,3 of rows 4,6
	VMOVDQU64 (R12), Z1    // permute indices
	VPERMI2PD Z7, Z6, Z1   // 128-bit lanes 0,2 of rows 5,7
	VMOVDQU64 64(R12), Z3  // permute indices
	VPERMI2PD Z7, Z6, Z3   // 128-bit lanes 1,3 of rows 5,7

	// transpose level 3: exchange register bit 2 with lane bit 2
	VSHUFF64X2 $0x44, Z5, Z8, Z7  // low halves of rows 0,4
	VSHUFF64X2 $0xEE, Z5, Z8, Z6  // high halves of rows 0,4
	VSHUFF64X2 $0x44, Z1, Z0, Z5  // low halves of rows 1,5
	VSHUFF64X2 $0xEE, Z1, Z0, Z8  // high halves of rows 1,5
	VSHUFF64X2 $0x44, Z11, Z9, Z1 // low halves of rows 2,6
	VSHUFF64X2 $0xEE, Z11, Z9, Z0 // high halves of rows 2,6
	VSHUFF64X2 $0x44, Z3, Z2, Z11 // low halves of rows 3,7
	VSHUFF64X2 $0xEE, Z3, Z2, Z9  // high halves of rows 3,7

	// ===== step C: 8-point DFT along the register axis (rows) =====
	VADDPS Z6, Z7, Z3           // C dft4(even): t0 = x0+x2
	VSUBPS Z6, Z7, Z2           // C dft4(even): t1 = x0-x2
	VADDPS Z0, Z1, Z10          // C dft4(even): t2 = x1+x3
	VSUBPS Z0, Z1, Z4           // C dft4(even): t3 = x1-x3
	VPERMILPS $0xB1, Z4, Z0     // C dft4(even): swap re/im of t3
	VPXORQ Z27, Z0, Z6          // C dft4(even): u = -i*t3 (fwd) / +i*t3 (inv)
	VADDPS Z10, Z3, Z4          // C dft4(even): X0 = t0+t2
	VSUBPS Z10, Z3, Z0          // C dft4(even): X2 = t0-t2
	VADDPS Z6, Z2, Z1           // C dft4(even): X1 = t1+u
	VSUBPS Z6, Z2, Z7           // C dft4(even): X3 = t1-u
	VADDPS Z8, Z5, Z6           // C dft4(odd): t0 = x0+x2
	VSUBPS Z8, Z5, Z10          // C dft4(odd): t1 = x0-x2
	VADDPS Z9, Z11, Z2          // C dft4(odd): t2 = x1+x3
	VSUBPS Z9, Z11, Z3          // C dft4(odd): t3 = x1-x3
	VPERMILPS $0xB1, Z3, Z9     // C dft4(odd): swap re/im of t3
	VPXORQ Z27, Z9, Z8          // C dft4(odd): u = -i*t3 (fwd) / +i*t3 (inv)
	VADDPS Z2, Z6, Z3           // C dft4(odd): X0 = t0+t2
	VSUBPS Z2, Z6, Z9           // C dft4(odd): X2 = t0-t2
	VADDPS Z8, Z10, Z11         // C dft4(odd): X1 = t1+u
	VSUBPS Z8, Z10, Z5          // C dft4(odd): X3 = t1-u
	VADDPS Z3, Z4, Z8           // C combine w^0: a+b
	VSUBPS Z3, Z4, Z2           // C combine w^0: a-b
	VPERMILPS $0xB1, Z11, Z3    // swap re/im
	VMULPS Z29, Z3, Z4          // t = swap(b)*Im(w8)
	VFMADDSUB231PS Z28, Z11, Z4 // t = w8*b
	VADDPS Z4, Z1, Z11          // C combine w^1: a+t
	VSUBPS Z4, Z1, Z3           // C combine w^1: a-t
	VPERMILPS $0xB1, Z9, Z4     // C combine w^2: swap re/im
	VPXORQ Z27, Z4, Z1          // C combine w^2: t = -i*b (fwd) / +i*b (inv)
	VADDPS Z1, Z0, Z9           // C combine w^2: a+t
	VSUBPS Z1, Z0, Z4           // C combine w^2: a-t
	VPERMILPS $0xB1, Z5, Z1     // swap re/im
	VMULPS Z31, Z1, Z0          // t = swap(b)*Im(w8^3)
	VFMADDSUB231PS Z30, Z5, Z0  // t = w8^3*b
	VADDPS Z0, Z7, Z5           // C combine w^3: a+t
	VSUBPS Z0, Z7, Z1           // C combine w^3: a-t

	// store the 8 result rows (natural order)
	VMOVUPS Z8, (R8)    // dst[0..7]
	VMOVUPS Z11, 64(R8) // dst[8..15]
	VMOVUPS Z9, 128(R8) // dst[16..23]
	VMOVUPS Z5, 192(R8) // dst[24..31]
	VMOVUPS Z2, 256(R8) // dst[32..39]
	VMOVUPS Z3, 320(R8) // dst[40..47]
	VMOVUPS Z4, 384(R8) // dst[48..55]
	VMOVUPS Z1, 448(R8) // dst[56..63]

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

avx512_r4_64_fwd_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 64, complex64, radix4
// ===========================================================================
TEXT ·InverseAVX512Size64Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ src_len+32(FP), R11 // R11 = n = len(src)

	// This codelet handles exactly n == 64.
	CMPQ R11, $64
	JNE  avx512_r4_64_inv_false

	// Validate the remaining slice lengths.
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $64
	JL   avx512_r4_64_inv_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $64
	JL   avx512_r4_64_inv_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $64
	JL   avx512_r4_64_inv_false

	// No working buffer is needed: all 64 values are loaded into ZMM
	// registers before the first store, so dst == src (in-place) is safe and
	// scratch stays untouched.

	VBROADCASTF32X4 ·maskNegLoPS(SB), Z27  // negate real lane of each complex64 (multiply by +i after swap)
	VBROADCASTSS 64(R10), Z28              // Re(tw[8]) = Re(w8), w8 = exp(-2pi*i/8)
	VBROADCASTSS 68(R10), Z29              // Im(tw[8]) = Im(w8)
	VBROADCASTSS 192(R10), Z30             // Re(tw[24]) = Re(w8^3)
	VBROADCASTSS 196(R10), Z31             // Im(tw[24]) = Im(w8^3)
	VBROADCASTSS ·sixtyFourth32(SB), Z26   // 1/64 scaling for the inverse
	LEAQ ·avx512F32Size64TransIdx(SB), R12 // R12 = transpose permute indices
	LEAQ ·avx512F32Size64CrossTw(SB), R13  // R13 = four-step twiddle vectors

	// load all 64 complex64 into 8 ZMM registers (row r = src[8r..8r+7])
	VMOVUPS (R9), Z0    // row 0 -> sub-FFT slot 0
	VMOVUPS 64(R9), Z1  // row 1 -> sub-FFT slot 1
	VMOVUPS 128(R9), Z2 // row 2 -> sub-FFT slot 2
	VMOVUPS 192(R9), Z3 // row 3 -> sub-FFT slot 3
	VMOVUPS 256(R9), Z4 // row 4 -> sub-FFT slot 4
	VMOVUPS 320(R9), Z5 // row 5 -> sub-FFT slot 5
	VMOVUPS 384(R9), Z6 // row 6 -> sub-FFT slot 6
	VMOVUPS 448(R9), Z7 // row 7 -> sub-FFT slot 7

	// ===== step A: 8-point DFT along the register axis (columns) =====
	VADDPS Z4, Z0, Z8           // A dft4(even): t0 = x0+x2
	VSUBPS Z4, Z0, Z9           // A dft4(even): t1 = x0-x2
	VADDPS Z6, Z2, Z10          // A dft4(even): t2 = x1+x3
	VSUBPS Z6, Z2, Z11          // A dft4(even): t3 = x1-x3
	VPERMILPS $0xB1, Z11, Z6    // A dft4(even): swap re/im of t3
	VPXORQ Z27, Z6, Z4          // A dft4(even): u = -i*t3 (fwd) / +i*t3 (inv)
	VADDPS Z10, Z8, Z11         // A dft4(even): X0 = t0+t2
	VSUBPS Z10, Z8, Z6          // A dft4(even): X2 = t0-t2
	VADDPS Z4, Z9, Z2           // A dft4(even): X1 = t1+u
	VSUBPS Z4, Z9, Z0           // A dft4(even): X3 = t1-u
	VADDPS Z5, Z1, Z4           // A dft4(odd): t0 = x0+x2
	VSUBPS Z5, Z1, Z10          // A dft4(odd): t1 = x0-x2
	VADDPS Z7, Z3, Z9           // A dft4(odd): t2 = x1+x3
	VSUBPS Z7, Z3, Z8           // A dft4(odd): t3 = x1-x3
	VPERMILPS $0xB1, Z8, Z7     // A dft4(odd): swap re/im of t3
	VPXORQ Z27, Z7, Z5          // A dft4(odd): u = -i*t3 (fwd) / +i*t3 (inv)
	VADDPS Z9, Z4, Z8           // A dft4(odd): X0 = t0+t2
	VSUBPS Z9, Z4, Z7           // A dft4(odd): X2 = t0-t2
	VADDPS Z5, Z10, Z3          // A dft4(odd): X1 = t1+u
	VSUBPS Z5, Z10, Z1          // A dft4(odd): X3 = t1-u
	VADDPS Z8, Z11, Z5          // A combine w^0: a+b
	VSUBPS Z8, Z11, Z9          // A combine w^0: a-b
	VPERMILPS $0xB1, Z3, Z8     // swap re/im
	VMULPS Z29, Z8, Z11         // t = swap(b)*Im(w8)
	VFMSUBADD231PS Z28, Z3, Z11 // t = conj(w8)*b
	VADDPS Z11, Z2, Z3          // A combine w^1: a+t
	VSUBPS Z11, Z2, Z8          // A combine w^1: a-t
	VPERMILPS $0xB1, Z7, Z11    // A combine w^2: swap re/im
	VPXORQ Z27, Z11, Z2         // A combine w^2: t = -i*b (fwd) / +i*b (inv)
	VADDPS Z2, Z6, Z7           // A combine w^2: a+t
	VSUBPS Z2, Z6, Z11          // A combine w^2: a-t
	VPERMILPS $0xB1, Z1, Z2     // swap re/im
	VMULPS Z31, Z2, Z6          // t = swap(b)*Im(w8^3)
	VFMSUBADD231PS Z30, Z1, Z6  // t = conj(w8^3)*b
	VADDPS Z6, Z0, Z1           // A combine w^3: a+t
	VSUBPS Z6, Z0, Z2           // A combine w^3: a-t

	// ===== step B: four-step twiddle, row k2 *= w^(lane*k2) =====
	VPERMILPS $0xB1, Z3, Z6          // swap re/im
	VMULPS 64(R13), Z6, Z0           // t = swap(b)*Im(w^(lane*1))
	VFMSUBADD231PS (R13), Z3, Z0     // t = conj(w^(lane*1))*b
	VPERMILPS $0xB1, Z7, Z3          // swap re/im
	VMULPS 192(R13), Z3, Z6          // t = swap(b)*Im(w^(lane*2))
	VFMSUBADD231PS 128(R13), Z7, Z6  // t = conj(w^(lane*2))*b
	VPERMILPS $0xB1, Z1, Z7          // swap re/im
	VMULPS 320(R13), Z7, Z3          // t = swap(b)*Im(w^(lane*3))
	VFMSUBADD231PS 256(R13), Z1, Z3  // t = conj(w^(lane*3))*b
	VPERMILPS $0xB1, Z9, Z1          // swap re/im
	VMULPS 448(R13), Z1, Z7          // t = swap(b)*Im(w^(lane*4))
	VFMSUBADD231PS 384(R13), Z9, Z7  // t = conj(w^(lane*4))*b
	VPERMILPS $0xB1, Z8, Z9          // swap re/im
	VMULPS 576(R13), Z9, Z1          // t = swap(b)*Im(w^(lane*5))
	VFMSUBADD231PS 512(R13), Z8, Z1  // t = conj(w^(lane*5))*b
	VPERMILPS $0xB1, Z11, Z8         // swap re/im
	VMULPS 704(R13), Z8, Z9          // t = swap(b)*Im(w^(lane*6))
	VFMSUBADD231PS 640(R13), Z11, Z9 // t = conj(w^(lane*6))*b
	VPERMILPS $0xB1, Z2, Z11         // swap re/im
	VMULPS 832(R13), Z11, Z8         // t = swap(b)*Im(w^(lane*7))
	VFMSUBADD231PS 768(R13), Z2, Z8  // t = conj(w^(lane*7))*b

	// ===== 8x8 transpose: (row k2, lane l) -> (row l, lane k2) =====

	// transpose level 1: exchange register bit 0 with lane bit 0
	VUNPCKLPD Z0, Z5, Z2  // even lanes of rows 0,1
	VUNPCKHPD Z0, Z5, Z11 // odd lanes of rows 0,1
	VUNPCKLPD Z3, Z6, Z0  // even lanes of rows 2,3
	VUNPCKHPD Z3, Z6, Z5  // odd lanes of rows 2,3
	VUNPCKLPD Z1, Z7, Z3  // even lanes of rows 4,5
	VUNPCKHPD Z1, Z7, Z6  // odd lanes of rows 4,5
	VUNPCKLPD Z8, Z9, Z1  // even lanes of rows 6,7
	VUNPCKHPD Z8, Z9, Z7  // odd lanes of rows 6,7

	// transpose level 2: exchange register bit 1 with lane bit 1
	VMOVDQU64 (R12), Z8    // permute indices
	VPERMI2PD Z0, Z2, Z8   // 128-bit lanes 0,2 of rows 0,2
	VMOVDQU64 64(R12), Z9  // permute indices
	VPERMI2PD Z0, Z2, Z9   // 128-bit lanes 1,3 of rows 0,2
	VMOVDQU64 (R12), Z0    // permute indices
	VPERMI2PD Z5, Z11, Z0  // 128-bit lanes 0,2 of rows 1,3
	VMOVDQU64 64(R12), Z2  // permute indices
	VPERMI2PD Z5, Z11, Z2  // 128-bit lanes 1,3 of rows 1,3
	VMOVDQU64 (R12), Z5    // permute indices
	VPERMI2PD Z1, Z3, Z5   // 128-bit lanes 0,2 of rows 4,6
	VMOVDQU64 64(R12), Z11 // permute indices
	VPERMI2PD Z1, Z3, Z11  // 128-bit lanes 1,3 of rows 4,6
	VMOVDQU64 (R12), Z1    // permute indices
	VPERMI2PD Z7, Z6, Z1   // 128-bit lanes 0,2 of rows 5,7
	VMOVDQU64 64(R12), Z3  // permute indices
	VPERMI2PD Z7, Z6, Z3   // 128-bit lanes 1,3 of rows 5,7

	// transpose level 3: exchange register bit 2 with lane bit 2
	VSHUFF64X2 $0x44, Z5, Z8, Z7  // low halves of rows 0,4
	VSHUFF64X2 $0xEE, Z5, Z8, Z6  // high halves of rows 0,4
	VSHUFF64X2 $0x44, Z1, Z0, Z5  // low halves of rows 1,5
	VSHUFF64X2 $0xEE, Z1, Z0, Z8  // high halves of rows 1,5
	VSHUFF64X2 $0x44, Z11, Z9, Z1 // low halves of rows 2,6
	VSHUFF64X2 $0xEE, Z11, Z9, Z0 // high halves of rows 2,6
	VSHUFF64X2 $0x44, Z3, Z2, Z11 // low halves of rows 3,7
	VSHUFF64X2 $0xEE, Z3, Z2, Z9  // high halves of rows 3,7

	// ===== step C: 8-point DFT along the register axis (rows) =====
	VADDPS Z6, Z7, Z3           // C dft4(even): t0 = x0+x2
	VSUBPS Z6, Z7, Z2           // C dft4(even): t1 = x0-x2
	VADDPS Z0, Z1, Z10          // C dft4(even): t2 = x1+x3
	VSUBPS Z0, Z1, Z4           // C dft4(even): t3 = x1-x3
	VPERMILPS $0xB1, Z4, Z0     // C dft4(even): swap re/im of t3
	VPXORQ Z27, Z0, Z6          // C dft4(even): u = -i*t3 (fwd) / +i*t3 (inv)
	VADDPS Z10, Z3, Z4          // C dft4(even): X0 = t0+t2
	VSUBPS Z10, Z3, Z0          // C dft4(even): X2 = t0-t2
	VADDPS Z6, Z2, Z1           // C dft4(even): X1 = t1+u
	VSUBPS Z6, Z2, Z7           // C dft4(even): X3 = t1-u
	VADDPS Z8, Z5, Z6           // C dft4(odd): t0 = x0+x2
	VSUBPS Z8, Z5, Z10          // C dft4(odd): t1 = x0-x2
	VADDPS Z9, Z11, Z2          // C dft4(odd): t2 = x1+x3
	VSUBPS Z9, Z11, Z3          // C dft4(odd): t3 = x1-x3
	VPERMILPS $0xB1, Z3, Z9     // C dft4(odd): swap re/im of t3
	VPXORQ Z27, Z9, Z8          // C dft4(odd): u = -i*t3 (fwd) / +i*t3 (inv)
	VADDPS Z2, Z6, Z3           // C dft4(odd): X0 = t0+t2
	VSUBPS Z2, Z6, Z9           // C dft4(odd): X2 = t0-t2
	VADDPS Z8, Z10, Z11         // C dft4(odd): X1 = t1+u
	VSUBPS Z8, Z10, Z5          // C dft4(odd): X3 = t1-u
	VADDPS Z3, Z4, Z8           // C combine w^0: a+b
	VSUBPS Z3, Z4, Z2           // C combine w^0: a-b
	VPERMILPS $0xB1, Z11, Z3    // swap re/im
	VMULPS Z29, Z3, Z4          // t = swap(b)*Im(w8)
	VFMSUBADD231PS Z28, Z11, Z4 // t = conj(w8)*b
	VADDPS Z4, Z1, Z11          // C combine w^1: a+t
	VSUBPS Z4, Z1, Z3           // C combine w^1: a-t
	VPERMILPS $0xB1, Z9, Z4     // C combine w^2: swap re/im
	VPXORQ Z27, Z4, Z1          // C combine w^2: t = -i*b (fwd) / +i*b (inv)
	VADDPS Z1, Z0, Z9           // C combine w^2: a+t
	VSUBPS Z1, Z0, Z4           // C combine w^2: a-t
	VPERMILPS $0xB1, Z5, Z1     // swap re/im
	VMULPS Z31, Z1, Z0          // t = swap(b)*Im(w8^3)
	VFMSUBADD231PS Z30, Z5, Z0  // t = conj(w8^3)*b
	VADDPS Z0, Z7, Z5           // C combine w^3: a+t
	VSUBPS Z0, Z7, Z1           // C combine w^3: a-t

	// inverse scaling by 1/64
	VMULPS Z26, Z8, Z0  // row 0 *= 1/64
	VMULPS Z26, Z11, Z8 // row 1 *= 1/64
	VMULPS Z26, Z9, Z11 // row 2 *= 1/64
	VMULPS Z26, Z5, Z9  // row 3 *= 1/64
	VMULPS Z26, Z2, Z5  // row 4 *= 1/64
	VMULPS Z26, Z3, Z2  // row 5 *= 1/64
	VMULPS Z26, Z4, Z3  // row 6 *= 1/64
	VMULPS Z26, Z1, Z4  // row 7 *= 1/64

	// store the 8 result rows (natural order)
	VMOVUPS Z0, (R8)     // dst[0..7]
	VMOVUPS Z8, 64(R8)   // dst[8..15]
	VMOVUPS Z11, 128(R8) // dst[16..23]
	VMOVUPS Z9, 192(R8)  // dst[24..31]
	VMOVUPS Z5, 256(R8)  // dst[32..39]
	VMOVUPS Z2, 320(R8)  // dst[40..47]
	VMOVUPS Z3, 384(R8)  // dst[48..55]
	VMOVUPS Z4, 448(R8)  // dst[56..63]

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

avx512_r4_64_inv_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

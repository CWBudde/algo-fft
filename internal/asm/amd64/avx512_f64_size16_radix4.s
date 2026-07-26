//go:build amd64 && !purego

// ===========================================================================
// AVX-512 Size-16 Radix-4 FFT Kernels for AMD64 (complex128)
// ===========================================================================
//
// A ZMM register holds 4 complex128, so the whole 16-point transform lives in
// four ZMM registers. The kernel loads once, runs both radix-4 stages entirely
// in registers, and stores once — there is no memory traffic between stages and
// the `scratch` slice is never touched (in-place dst == src therefore works
// without any pointer juggling: every input is consumed before the first
// store).
//
// REGISTER LAYOUT
// ---------------
// Zq (q = 0..3) initially holds src[4q .. 4q+3], i.e. lane m of Zq is
// src[4q+m]. Read the other way round, lane m across Z0..Z3 is
//
//     (src[m], src[4+m], src[8+m], src[12+m])
//
// which is exactly the input quadruple of radix-4 DIT butterfly m at stage 1
// (the radix-4 bit-reversal permutation [0,4,8,12, 1,5,9,13, 2,6,10,14,
// 3,7,11,15] used by dit16_radix4_* is absorbed into this transposed reading of
// the registers, so no permutation instructions are needed for it).
//
// After stage 1, lane m of Zk holds stage1[4m+k]. Stage 2's butterflies run
// over (stage1[j], stage1[j+4], stage1[j+8], stage1[j+12]), which is lane j of
// the *transposed* layout, so one 4x4 exchange of 128-bit lanes
// (8 x VSHUFF64X2) sits between the stages. Stage 2 then writes natural order
// directly: Zk = dst[4k .. 4k+3].
//
// COMPLEX MULTIPLY (4 complex128 per ZMM)
// ---------------------------------------
//   VMOVDDUP      w -> [w.r, w.r, ...]
//   VSHUFPD $0xFF w -> [w.i, w.i, ...]
//   VSHUFPD $0x55 b -> [b.i, b.r, ...]
//   VMULPD + VFMADDSUB231PD  => t = w * b       (forward)
//   VMULPD + VFMSUBADD231PD  => t = conj(w) * b (inverse)
// The twiddle broadcasts are hoisted to the top of the function: they depend
// only on the twiddle table, not on the data.
//
// TWIDDLE VECTORS (lane j of stage 2 needs tw[j], tw[2j], tw[3j])
// ---------------------------------------------------------------
//   w1 = (tw[0], tw[1], tw[2], tw[3]) -- contiguous, one VMOVUPD
//   w2 = (tw[0], tw[2], tw[4], tw[6]) -- XMM load + 3 x VINSERTF32X4
//   w3 = (tw[0], tw[3], tw[6], tw[9]) -- XMM load + 3 x VINSERTF32X4
//
// The (-i) multiply is a swap of the real/imaginary doubles followed by a sign
// flip of the imaginary one; i*t is then just the negation of it, so the two
// odd radix-4 outputs are one VADDPD and one VSUBPD against the same vector.
//
// AVX512F ONLY
// -------------
// cpu.Features.HasAVX512 is set from CPUID.(EAX=7).EBX[16], which is AVX512F,
// so these kernels must not use AVX512DQ encodings. Two substitutions keep
// them inside AVX512F, bit-for-bit identical because no masking is used:
//   VPXORQ       instead of VXORPD on ZMM  (DQ-only)
//   VINSERTF32X4 instead of VINSERTF64X2   (DQ-only; imm[1:0] picks the same
//                                          128-bit lane)
//
// Callers gate on cpu.Features.HasAVX512.
// ===========================================================================

#include "textflag.h"

// negImagPD512x16 flips the sign of the imaginary (odd) double of each of the
// four complex128 in a ZMM register. File-scope: the shared masks in core.s are
// only 16/32 bytes wide.
DATA negImagPD512x16<>+0(SB)/8, $0x0000000000000000
DATA negImagPD512x16<>+8(SB)/8, $0x8000000000000000
DATA negImagPD512x16<>+16(SB)/8, $0x0000000000000000
DATA negImagPD512x16<>+24(SB)/8, $0x8000000000000000
DATA negImagPD512x16<>+32(SB)/8, $0x0000000000000000
DATA negImagPD512x16<>+40(SB)/8, $0x8000000000000000
DATA negImagPD512x16<>+48(SB)/8, $0x0000000000000000
DATA negImagPD512x16<>+56(SB)/8, $0x8000000000000000
GLOBL negImagPD512x16<>(SB), RODATA|NOPTR, $64

// ===========================================================================
// ForwardAVX512Size16Radix4Complex128Asm - forward 16-point FFT, complex128
// ===========================================================================
TEXT ·ForwardAVX512Size16Radix4Complex128Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer

	MOVQ src_len+32(FP), AX
	CMPQ AX, $16
	JNE  s16r4_512_fwd_false // this codelet only handles n == 16

	MOVQ dst_len+8(FP), AX
	CMPQ AX, $16
	JL   s16r4_512_fwd_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $16
	JL   s16r4_512_fwd_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $16
	JL   s16r4_512_fwd_false    // scratch is unused, but keep the contract

	VMOVUPD negImagPD512x16<>(SB), Z16 // Z16 = imaginary-sign mask

	// ------------------------------------------------------------------
	// Twiddle vectors and their broadcast forms (data-independent).
	// ------------------------------------------------------------------
	VMOVUPD 0(R10), Z8 // Z8 = w1 = (tw[0], tw[1], tw[2], tw[3])

	VMOVUPD      0(R10), X9          // X9 = tw[0]
	VINSERTF32X4 $1, 32(R10), Z9, Z9 // lane 1 = tw[2]
	VINSERTF32X4 $2, 64(R10), Z9, Z9 // lane 2 = tw[4]
	VINSERTF32X4 $3, 96(R10), Z9, Z9 // Z9 = w2 = (tw[0],tw[2],tw[4],tw[6])

	VMOVUPD      0(R10), X10            // X10 = tw[0]
	VINSERTF32X4 $1, 48(R10), Z10, Z10  // lane 1 = tw[3]
	VINSERTF32X4 $2, 96(R10), Z10, Z10  // lane 2 = tw[6]
	VINSERTF32X4 $3, 144(R10), Z10, Z10 // Z10 = w3 = (tw[0],tw[3],tw[6],tw[9])

	VMOVDDUP Z8, Z20              // Z20 = [w1.re, w1.re, ...]
	VSHUFPD  $0xFF, Z8, Z8, Z21   // Z21 = [w1.im, w1.im, ...]
	VMOVDDUP Z9, Z22              // Z22 = [w2.re, w2.re, ...]
	VSHUFPD  $0xFF, Z9, Z9, Z23   // Z23 = [w2.im, w2.im, ...]
	VMOVDDUP Z10, Z24             // Z24 = [w3.re, w3.re, ...]
	VSHUFPD  $0xFF, Z10, Z10, Z25 // Z25 = [w3.im, w3.im, ...]

	// ------------------------------------------------------------------
	// Load. Lane m of Z0..Z3 is the stage-1 butterfly quadruple
	// (src[m], src[4+m], src[8+m], src[12+m]).
	// ------------------------------------------------------------------
	VMOVUPD 0(R9), Z0   // Z0 = src[0..3]   -> a0 of each butterfly
	VMOVUPD 64(R9), Z1  // Z1 = src[4..7]   -> a1
	VMOVUPD 128(R9), Z2 // Z2 = src[8..11]  -> a2
	VMOVUPD 192(R9), Z3 // Z3 = src[12..15] -> a3

	// ------------------------------------------------------------------
	// Stage 1: 4 radix-4 butterflies, trivial twiddles (1, -i, -1, i).
	// ------------------------------------------------------------------
	VADDPD Z2, Z0, Z4 // Z4 = t0 = a0 + a2
	VSUBPD Z2, Z0, Z5 // Z5 = t1 = a0 - a2
	VADDPD Z3, Z1, Z6 // Z6 = t2 = a1 + a3
	VSUBPD Z3, Z1, Z7 // Z7 = t3 = a1 - a3

	VSHUFPD $0x55, Z7, Z7, Z11 // Z11 = (t3.im, t3.re) per complex
	VPXORQ  Z16, Z11, Z11      // Z11 = (-i) * t3

	VADDPD Z6, Z4, Z0  // Z0 = y0 = t0 + t2
	VADDPD Z11, Z5, Z1 // Z1 = y1 = t1 + (-i)*t3
	VSUBPD Z6, Z4, Z2  // Z2 = y2 = t0 - t2
	VSUBPD Z11, Z5, Z3 // Z3 = y3 = t1 - (-i)*t3 = t1 + i*t3

	// ------------------------------------------------------------------
	// 4x4 exchange of 128-bit lanes: lane m of Zk (= stage1[4m+k]) becomes
	// lane k of Zm (= stage1[4m+k]), i.e. Zm = stage1[4m .. 4m+3].
	// ------------------------------------------------------------------
	VSHUFF64X2 $0x88, Z1, Z0, Z4 // Z4 = (Z0.l0, Z0.l2, Z1.l0, Z1.l2)
	VSHUFF64X2 $0xDD, Z1, Z0, Z5 // Z5 = (Z0.l1, Z0.l3, Z1.l1, Z1.l3)
	VSHUFF64X2 $0x88, Z3, Z2, Z6 // Z6 = (Z2.l0, Z2.l2, Z3.l0, Z3.l2)
	VSHUFF64X2 $0xDD, Z3, Z2, Z7 // Z7 = (Z2.l1, Z2.l3, Z3.l1, Z3.l3)

	VSHUFF64X2 $0x88, Z6, Z4, Z0 // Z0 = stage1[0..3]   -> a0 of stage 2
	VSHUFF64X2 $0x88, Z7, Z5, Z1 // Z1 = stage1[4..7]   -> a1
	VSHUFF64X2 $0xDD, Z6, Z4, Z2 // Z2 = stage1[8..11]  -> a2
	VSHUFF64X2 $0xDD, Z7, Z5, Z3 // Z3 = stage1[12..15] -> a3

	// ------------------------------------------------------------------
	// Stage 2: 4 radix-4 butterflies with twiddles tw[j], tw[2j], tw[3j].
	// ------------------------------------------------------------------
	VSHUFPD        $0x55, Z1, Z1, Z17 // Z17 = swap(a1)
	VMULPD         Z21, Z17, Z17      // Z17 = swap(a1) * w1.im
	VFMADDSUB231PD Z20, Z1, Z17       // Z17 = a1 * w1

	VSHUFPD        $0x55, Z2, Z2, Z18 // Z18 = swap(a2)
	VMULPD         Z23, Z18, Z18      // Z18 = swap(a2) * w2.im
	VFMADDSUB231PD Z22, Z2, Z18       // Z18 = a2 * w2

	VSHUFPD        $0x55, Z3, Z3, Z19 // Z19 = swap(a3)
	VMULPD         Z25, Z19, Z19      // Z19 = swap(a3) * w3.im
	VFMADDSUB231PD Z24, Z3, Z19       // Z19 = a3 * w3

	VADDPD Z18, Z0, Z4  // Z4 = t0 = a0 + a2*w2
	VSUBPD Z18, Z0, Z5  // Z5 = t1 = a0 - a2*w2
	VADDPD Z19, Z17, Z6 // Z6 = t2 = a1*w1 + a3*w3
	VSUBPD Z19, Z17, Z7 // Z7 = t3 = a1*w1 - a3*w3

	VSHUFPD $0x55, Z7, Z7, Z11 // Z11 = (t3.im, t3.re) per complex
	VPXORQ  Z16, Z11, Z11      // Z11 = (-i) * t3

	VADDPD Z6, Z4, Z0  // Z0 = dst[0..3]   = t0 + t2
	VADDPD Z11, Z5, Z1 // Z1 = dst[4..7]   = t1 + (-i)*t3
	VSUBPD Z6, Z4, Z2  // Z2 = dst[8..11]  = t0 - t2
	VSUBPD Z11, Z5, Z3 // Z3 = dst[12..15] = t1 + i*t3

	VMOVUPD Z0, 0(R8)
	VMOVUPD Z1, 64(R8)
	VMOVUPD Z2, 128(R8)
	VMOVUPD Z3, 192(R8)

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

s16r4_512_fwd_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// InverseAVX512Size16Radix4Complex128Asm - inverse 16-point FFT, complex128
// ===========================================================================
// Same dataflow as the forward kernel with conjugated twiddles
// (VFMSUBADD231PD), the two odd radix-4 outputs swapped (+i instead of -i in
// the trivial stage), and a final 1/16 scaling.
TEXT ·InverseAVX512Size16Radix4Complex128Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer

	MOVQ src_len+32(FP), AX
	CMPQ AX, $16
	JNE  s16r4_512_inv_false

	MOVQ dst_len+8(FP), AX
	CMPQ AX, $16
	JL   s16r4_512_inv_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $16
	JL   s16r4_512_inv_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $16
	JL   s16r4_512_inv_false

	VMOVUPD negImagPD512x16<>(SB), Z16 // Z16 = imaginary-sign mask

	MOVQ         ·sixteenth64(SB), AX // AX = float64(1.0/16.0)
	VMOVQ        AX, X26
	VBROADCASTSD X26, Z26             // Z26 = [1/16 x8]

	// Twiddle vectors and their broadcast forms (data-independent).
	VMOVUPD 0(R10), Z8 // Z8 = w1 = (tw[0], tw[1], tw[2], tw[3])

	VMOVUPD      0(R10), X9          // X9 = tw[0]
	VINSERTF32X4 $1, 32(R10), Z9, Z9 // lane 1 = tw[2]
	VINSERTF32X4 $2, 64(R10), Z9, Z9 // lane 2 = tw[4]
	VINSERTF32X4 $3, 96(R10), Z9, Z9 // Z9 = w2 = (tw[0],tw[2],tw[4],tw[6])

	VMOVUPD      0(R10), X10            // X10 = tw[0]
	VINSERTF32X4 $1, 48(R10), Z10, Z10  // lane 1 = tw[3]
	VINSERTF32X4 $2, 96(R10), Z10, Z10  // lane 2 = tw[6]
	VINSERTF32X4 $3, 144(R10), Z10, Z10 // Z10 = w3 = (tw[0],tw[3],tw[6],tw[9])

	VMOVDDUP Z8, Z20              // Z20 = [w1.re, w1.re, ...]
	VSHUFPD  $0xFF, Z8, Z8, Z21   // Z21 = [w1.im, w1.im, ...]
	VMOVDDUP Z9, Z22              // Z22 = [w2.re, w2.re, ...]
	VSHUFPD  $0xFF, Z9, Z9, Z23   // Z23 = [w2.im, w2.im, ...]
	VMOVDDUP Z10, Z24             // Z24 = [w3.re, w3.re, ...]
	VSHUFPD  $0xFF, Z10, Z10, Z25 // Z25 = [w3.im, w3.im, ...]

	// Load: lane m of Z0..Z3 is stage-1 butterfly m's quadruple.
	VMOVUPD 0(R9), Z0   // Z0 = src[0..3]   -> a0
	VMOVUPD 64(R9), Z1  // Z1 = src[4..7]   -> a1
	VMOVUPD 128(R9), Z2 // Z2 = src[8..11]  -> a2
	VMOVUPD 192(R9), Z3 // Z3 = src[12..15] -> a3

	// Stage 1: 4 radix-4 butterflies, trivial twiddles (1, i, -1, -i).
	VADDPD Z2, Z0, Z4 // Z4 = t0 = a0 + a2
	VSUBPD Z2, Z0, Z5 // Z5 = t1 = a0 - a2
	VADDPD Z3, Z1, Z6 // Z6 = t2 = a1 + a3
	VSUBPD Z3, Z1, Z7 // Z7 = t3 = a1 - a3

	VSHUFPD $0x55, Z7, Z7, Z11 // Z11 = (t3.im, t3.re) per complex
	VPXORQ  Z16, Z11, Z11      // Z11 = (-i) * t3

	VADDPD Z6, Z4, Z0  // Z0 = y0 = t0 + t2
	VSUBPD Z11, Z5, Z1 // Z1 = y1 = t1 + i*t3
	VSUBPD Z6, Z4, Z2  // Z2 = y2 = t0 - t2
	VADDPD Z11, Z5, Z3 // Z3 = y3 = t1 + (-i)*t3

	// 4x4 exchange of 128-bit lanes (see the forward kernel).
	VSHUFF64X2 $0x88, Z1, Z0, Z4 // Z4 = (Z0.l0, Z0.l2, Z1.l0, Z1.l2)
	VSHUFF64X2 $0xDD, Z1, Z0, Z5 // Z5 = (Z0.l1, Z0.l3, Z1.l1, Z1.l3)
	VSHUFF64X2 $0x88, Z3, Z2, Z6 // Z6 = (Z2.l0, Z2.l2, Z3.l0, Z3.l2)
	VSHUFF64X2 $0xDD, Z3, Z2, Z7 // Z7 = (Z2.l1, Z2.l3, Z3.l1, Z3.l3)

	VSHUFF64X2 $0x88, Z6, Z4, Z0 // Z0 = stage1[0..3]   -> a0 of stage 2
	VSHUFF64X2 $0x88, Z7, Z5, Z1 // Z1 = stage1[4..7]   -> a1
	VSHUFF64X2 $0xDD, Z6, Z4, Z2 // Z2 = stage1[8..11]  -> a2
	VSHUFF64X2 $0xDD, Z7, Z5, Z3 // Z3 = stage1[12..15] -> a3

	// Stage 2: 4 radix-4 butterflies with conjugated twiddles.
	VSHUFPD        $0x55, Z1, Z1, Z17 // Z17 = swap(a1)
	VMULPD         Z21, Z17, Z17      // Z17 = swap(a1) * w1.im
	VFMSUBADD231PD Z20, Z1, Z17       // Z17 = a1 * conj(w1)

	VSHUFPD        $0x55, Z2, Z2, Z18 // Z18 = swap(a2)
	VMULPD         Z23, Z18, Z18      // Z18 = swap(a2) * w2.im
	VFMSUBADD231PD Z22, Z2, Z18       // Z18 = a2 * conj(w2)

	VSHUFPD        $0x55, Z3, Z3, Z19 // Z19 = swap(a3)
	VMULPD         Z25, Z19, Z19      // Z19 = swap(a3) * w3.im
	VFMSUBADD231PD Z24, Z3, Z19       // Z19 = a3 * conj(w3)

	VADDPD Z18, Z0, Z4  // Z4 = t0 = a0 + a2*conj(w2)
	VSUBPD Z18, Z0, Z5  // Z5 = t1 = a0 - a2*conj(w2)
	VADDPD Z19, Z17, Z6 // Z6 = t2 = a1*conj(w1) + a3*conj(w3)
	VSUBPD Z19, Z17, Z7 // Z7 = t3 = a1*conj(w1) - a3*conj(w3)

	VSHUFPD $0x55, Z7, Z7, Z11 // Z11 = (t3.im, t3.re) per complex
	VPXORQ  Z16, Z11, Z11      // Z11 = (-i) * t3

	VADDPD Z6, Z4, Z0  // Z0 = t0 + t2
	VSUBPD Z11, Z5, Z1 // Z1 = t1 + i*t3
	VSUBPD Z6, Z4, Z2  // Z2 = t0 - t2
	VADDPD Z11, Z5, Z3 // Z3 = t1 + (-i)*t3

	VMULPD Z26, Z0, Z0 // apply 1/16
	VMULPD Z26, Z1, Z1
	VMULPD Z26, Z2, Z2
	VMULPD Z26, Z3, Z3

	VMOVUPD Z0, 0(R8)
	VMOVUPD Z1, 64(R8)
	VMOVUPD Z2, 128(R8)
	VMOVUPD Z3, 192(R8)

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

s16r4_512_inv_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

//go:build amd64 && !purego

// ===========================================================================
// AVX-512 Size-32 Radix-4-then-2 FFT Kernels for AMD64 (complex64)
// ===========================================================================
//
// A ZMM register holds 8 complex64, so all 32 points live in four ZMM
// registers. The kernel loads once, runs all three stages (radix-4, radix-4,
// radix-2) entirely in registers and stores once: there is no memory traffic
// between stages and the `scratch` slice is never touched. In-place
// (dst == src) therefore works without pointer juggling, because every input
// is consumed before the first store.
//
// DATAFLOW
// --------
// The mixed-radix bit-reversal of dit32_radix4_then2_* is
//   [0,8,16,24, 2,10,18,26, 4,12,20,28, 6,14,22,30,
//    1,9,17,25, 3,11,19,27, 5,13,21,29, 7,15,23,31]
// i.e. work[4b+k] = src[sigma(b) + 8k] with sigma = [0,2,4,6,1,3,5,7]. The
// four contiguous loads Zk = src[8k..8k+7] therefore already hold member k of
// stage-1 butterfly b in lane sigma(b): the permutation costs no instructions
// and needs no table. Even lanes carry group A (work[0..15]), odd lanes group
// B (work[16..31]).
//
// After stage 1, register k holds work[4b+k] in lane pair b. Stage 2 needs
// quadruple j (members work[j], work[j+4], work[j+8], work[j+12] of each
// group) spread across the four registers in one lane pair, which is a 4x4
// exchange of 128-bit lanes: 8 x VSHUFF32X4. Stage 2 then runs all eight
// radix-4 butterflies in parallel and leaves register r holding
// work[16g+4r+j] in lane 2j+g.
//
// Two masked VPERMPD per output register gather the even (group A) and odd
// (group B) lanes back into natural order, after which the final radix-2
// stage pairs work[0..15] with work[16..31] using contiguous twiddles
// tw[0..15] and stores four contiguous ZMM registers.
//
// COMPLEX MULTIPLY (8 complex64 per ZMM)
// --------------------------------------
//   VMOVSLDUP     w -> [w.r, w.r, ...]
//   VMOVSHDUP     w -> [w.i, w.i, ...]
//   VSHUFPS $0xB1 b -> [b.i, b.r, ...]
//   VMULPS + VFMADDSUB231PS  => t = w * b       (forward)
//   VMULPS + VFMSUBADD231PS  => t = conj(w) * b (inverse)
//
// (-i)*t is a swap of the real/imaginary float32 plus a sign flip of the
// imaginary one, and i*t is its negation, so the two odd radix-4 outputs are
// one VADDPS and one VSUBPS against the same vector. The inverse simply
// exchanges those two.
//
// REGISTERS
// ---------
//   Z0-Z7   source loads and per-stage temporaries
//   Z8-Z11  stage-1/stage-2 data (radix-4 members 0..3)
//   Z12-Z15 work[0..7], work[8..15], work[16..23], work[24..31]
//   Z16     imaginary-sign mask
//   Z17/Z18 even/odd qword deinterleave indices
//   Z19     stage-2 twiddle staging register
//   Z20-Z25 broadcast real/imaginary parts of the stage-2 twiddles
//   Z26     1/32 scale (inverse only)
//   K1-K4   merge masks
//
// Requires AVX512F only. Callers gate on cpu.Features.HasAVX512.
// ===========================================================================

#include "textflag.h"

// negImagPS512x32 flips the sign of the imaginary (odd) float32 of each of the
// eight complex64 in a ZMM register. File-scope: the shared masks in core.s are
// only 16/32 bytes wide.
DATA negImagPS512x32<>+0(SB)/8, $0x8000000000000000
DATA negImagPS512x32<>+8(SB)/8, $0x8000000000000000
DATA negImagPS512x32<>+16(SB)/8, $0x8000000000000000
DATA negImagPS512x32<>+24(SB)/8, $0x8000000000000000
DATA negImagPS512x32<>+32(SB)/8, $0x8000000000000000
DATA negImagPS512x32<>+40(SB)/8, $0x8000000000000000
DATA negImagPS512x32<>+48(SB)/8, $0x8000000000000000
DATA negImagPS512x32<>+56(SB)/8, $0x8000000000000000
GLOBL negImagPS512x32<>(SB), RODATA|NOPTR, $64

// deintEvenQ512x32 / deintOddQ512x32 are VPERMPD qword index vectors that
// gather the even respectively odd complex lanes of a register into its lower
// (and, under mask K4, upper) four lanes.
DATA deintEvenQ512x32<>+0(SB)/8, $0
DATA deintEvenQ512x32<>+8(SB)/8, $2
DATA deintEvenQ512x32<>+16(SB)/8, $4
DATA deintEvenQ512x32<>+24(SB)/8, $6
DATA deintEvenQ512x32<>+32(SB)/8, $0
DATA deintEvenQ512x32<>+40(SB)/8, $2
DATA deintEvenQ512x32<>+48(SB)/8, $4
DATA deintEvenQ512x32<>+56(SB)/8, $6
GLOBL deintEvenQ512x32<>(SB), RODATA|NOPTR, $64

DATA deintOddQ512x32<>+0(SB)/8, $1
DATA deintOddQ512x32<>+8(SB)/8, $3
DATA deintOddQ512x32<>+16(SB)/8, $5
DATA deintOddQ512x32<>+24(SB)/8, $7
DATA deintOddQ512x32<>+32(SB)/8, $1
DATA deintOddQ512x32<>+40(SB)/8, $3
DATA deintOddQ512x32<>+48(SB)/8, $5
DATA deintOddQ512x32<>+56(SB)/8, $7
GLOBL deintOddQ512x32<>(SB), RODATA|NOPTR, $64

// ===========================================================================
// ForwardAVX512Size32Radix4Then2Complex64Asm
// ===========================================================================
TEXT ·ForwardAVX512Size32Radix4Then2Complex64Asm(SB), NOSPLIT, $0-97
	MOVQ           dst+0(FP), R8                        // R8  = dst pointer
	MOVQ           src+24(FP), R9                       // R9  = src pointer
	MOVQ           twiddle+48(FP), R10                  // R10 = twiddle pointer

	MOVQ           src_len+32(FP), AX
	CMPQ           AX, $32
	JNE            s32r42_512_fwd_false                 // this codelet only handles n == 32

	MOVQ           dst_len+8(FP), AX
	CMPQ           AX, $32
	JL             s32r42_512_fwd_false                 // dst too short

	MOVQ           twiddle_len+56(FP), AX
	CMPQ           AX, $32
	JL             s32r42_512_fwd_false                 // twiddle too short

	MOVQ           scratch_len+80(FP), AX
	CMPQ           AX, $32
	JL             s32r42_512_fwd_false                 // scratch too short (unused, checked for API parity)

	// Constants: imaginary-sign mask and the two deinterleave index
	// vectors used before the final radix-2 stage.
	VMOVUPS        negImagPS512x32<>(SB), Z16           // Z16 = per-complex (0, -0.0) sign mask
	VMOVUPS        deintEvenQ512x32<>(SB), Z17          // Z17 = qword indices [0,2,4,6,0,2,4,6]
	VMOVUPS        deintOddQ512x32<>(SB), Z18           // Z18 = qword indices [1,3,5,7,1,3,5,7]

	// Merge masks. K1/K2/K3 select complex-lane pairs {2,3}, {4,5}, {6,7}
	// when building the stage-2 twiddle vectors; K4 selects the upper
	// four complex lanes for the deinterleave.
	MOVL           $0x0C, AX
	KMOVW          AX, K1                               // K1 = complex lanes 2,3
	MOVL           $0x30, AX
	KMOVW          AX, K2                               // K2 = complex lanes 4,5
	MOVL           $0xC0, AX
	KMOVW          AX, K3                               // K3 = complex lanes 6,7
	MOVL           $0xF0, AX
	KMOVW          AX, K4                               // K4 = complex lanes 4..7

	// Stage-2 twiddle vectors. Complex lanes 2j and 2j+1 both carry
	// tw[2*j*r] for member r of the radix-4 butterfly, so each vector is
	// four 64-bit broadcasts merged into one register.
	VBROADCASTSD   (R10), Z19                           // w1 = [tw[0] x8]
	VBROADCASTSD   16(R10), K1, Z19                     // lanes 2,3 = tw[2]
	VBROADCASTSD   32(R10), K2, Z19                     // lanes 4,5 = tw[4]
	VBROADCASTSD   48(R10), K3, Z19                     // lanes 6,7 = tw[6]
	VMOVSLDUP      Z19, Z20                             // Z20 = [w1.re]
	VMOVSHDUP      Z19, Z21                             // Z21 = [w1.im]
	VBROADCASTSD   (R10), Z19                           // w2 = [tw[0] x8]
	VBROADCASTSD   32(R10), K1, Z19                     // lanes 2,3 = tw[4]
	VBROADCASTSD   64(R10), K2, Z19                     // lanes 4,5 = tw[8]
	VBROADCASTSD   96(R10), K3, Z19                     // lanes 6,7 = tw[12]
	VMOVSLDUP      Z19, Z22                             // Z22 = [w2.re]
	VMOVSHDUP      Z19, Z23                             // Z23 = [w2.im]
	VBROADCASTSD   (R10), Z19                           // w3 = [tw[0] x8]
	VBROADCASTSD   48(R10), K1, Z19                     // lanes 2,3 = tw[6]
	VBROADCASTSD   96(R10), K2, Z19                     // lanes 4,5 = tw[12]
	VBROADCASTSD   144(R10), K3, Z19                    // lanes 6,7 = tw[18]
	VMOVSLDUP      Z19, Z24                             // Z24 = [w3.re]
	VMOVSHDUP      Z19, Z25                             // Z25 = [w3.im]

	// Load src[0..31] as four contiguous ZMM registers. Lane l of Zk is
	// src[8k+l]; the mixed-radix permutation [0,8,16,24, 2,10,18,26, ...]
	// means work[4b+k] = src[sigma(b)+8k] with sigma = [0,2,4,6,1,3,5,7],
	// so lane l already holds member k of stage-1 butterfly
	// b = l/2 (l even) or 4+(l-1)/2 (l odd): no permutation is needed.
	VMOVUPS        (R9), Z0                             // Z0 = src[0..7]
	VMOVUPS        64(R9), Z1                           // Z1 = src[8..15]
	VMOVUPS        128(R9), Z2                          // Z2 = src[16..23]
	VMOVUPS        192(R9), Z3                          // Z3 = src[24..31]

	// Stage 1: eight radix-4 butterflies with trivial twiddles, one per
	// complex lane. a0..a3 are Z0..Z3.
	VADDPS         Z2, Z0, Z4                           // t0 = a0 + a2
	VSUBPS         Z2, Z0, Z5                           // t1 = a0 - a2
	VADDPS         Z3, Z1, Z6                           // t2 = a1 + a3
	VSUBPS         Z3, Z1, Z7                           // t3 = a1 - a3
	VSHUFPS        $0xB1, Z7, Z7, Z0                    // (t3.im, t3.re) per complex
	VPXORQ         Z16, Z0, Z0                          // Z0 = (-i) * t3
	VADDPS         Z6, Z4, Z8                           // y0 = t0 + t2
	VSUBPS         Z6, Z4, Z10                          // y2 = t0 - t2
	VADDPS         Z0, Z5, Z9                           // y1 = t1 + (-i)*t3
	VSUBPS         Z0, Z5, Z11                          // y3 = t1 + i*t3

	// Stage 1 leaves register k holding work[4b+k] in lane pair b, while
	// stage 2 needs quadruple j of both groups in lane pair j of every
	// register. That is a 4x4 exchange of 128-bit lanes.
	VSHUFF32X4     $0x88, Z9, Z8, Z0
	VSHUFF32X4     $0xDD, Z9, Z8, Z1
	VSHUFF32X4     $0x88, Z11, Z10, Z2
	VSHUFF32X4     $0xDD, Z11, Z10, Z3
	VSHUFF32X4     $0x88, Z2, Z0, Z8                    // Q0 = quadruple member 0
	VSHUFF32X4     $0x88, Z3, Z1, Z9                    // Q1 = quadruple member 1
	VSHUFF32X4     $0xDD, Z2, Z0, Z10                   // Q2 = quadruple member 2
	VSHUFF32X4     $0xDD, Z3, Z1, Z11                   // Q3 = quadruple member 3

	// Stage 2: eight radix-4 butterflies (four per group) with twiddles
	// tw[2j], tw[4j], tw[6j] on members 1, 2, 3.
	VSHUFPS        $0xB1, Z9, Z9, Z0                    // swap re/im of Z9
	VMULPS         Z21, Z0, Z0                          // Z0 *= w.im
	VFMADDSUB231PS Z20, Z9, Z0                          // a1 = w1 * Q1
	VSHUFPS        $0xB1, Z10, Z10, Z1                  // swap re/im of Z10
	VMULPS         Z23, Z1, Z1                          // Z1 *= w.im
	VFMADDSUB231PS Z22, Z10, Z1                         // a2 = w2 * Q2
	VSHUFPS        $0xB1, Z11, Z11, Z2                  // swap re/im of Z11
	VMULPS         Z25, Z2, Z2                          // Z2 *= w.im
	VFMADDSUB231PS Z24, Z11, Z2                         // a3 = w3 * Q3
	VADDPS         Z1, Z8, Z3                           // t0 = a0 + a2
	VSUBPS         Z1, Z8, Z4                           // t1 = a0 - a2
	VADDPS         Z2, Z0, Z5                           // t2 = a1 + a3
	VSUBPS         Z2, Z0, Z6                           // t3 = a1 - a3
	VSHUFPS        $0xB1, Z6, Z6, Z7                    // (t3.im, t3.re) per complex
	VPXORQ         Z16, Z7, Z7                          // Z7 = (-i) * t3
	VADDPS         Z5, Z3, Z8                           // y0 = t0 + t2
	VSUBPS         Z5, Z3, Z10                          // y2 = t0 - t2
	VADDPS         Z7, Z4, Z9                           // y1 = t1 + (-i)*t3
	VSUBPS         Z7, Z4, Z11                          // y3 = t1 + i*t3

	// Register r now holds work[16g+4r+j] in lane 2j+g. Gather the even
	// lanes (group A) and odd lanes (group B) into natural order so the
	// final radix-2 stage sees contiguous halves and contiguous twiddles.
	VPERMPD        Z8, Z17, Z12                         // lanes 0..3 = work[0..3]
	VPERMPD        Z9, Z17, K4, Z12                     // lanes 4..7 = work[4..7]
	VPERMPD        Z10, Z17, Z13                        // lanes 0..3 = work[8..11]
	VPERMPD        Z11, Z17, K4, Z13                    // lanes 4..7 = work[12..15]
	VPERMPD        Z8, Z18, Z14                         // lanes 0..3 = work[16..19]
	VPERMPD        Z9, Z18, K4, Z14                     // lanes 4..7 = work[20..23]
	VPERMPD        Z10, Z18, Z15                        // lanes 0..3 = work[24..27]
	VPERMPD        Z11, Z18, K4, Z15                    // lanes 4..7 = work[28..31]

	// Stage 3: radix-2 over (work[i], work[i+16]) with twiddle tw[i].
	VMOVUPS        (R10), Z0                            // Z0 = tw[0..7]
	VMOVSLDUP      Z0, Z1                               // Z1 = [tw.re]
	VMOVSHDUP      Z0, Z2                               // Z2 = [tw.im]
	VSHUFPS        $0xB1, Z14, Z14, Z3                  // swap re/im of Z14
	VMULPS         Z2, Z3, Z3                           // Z3 *= w.im
	VFMADDSUB231PS Z1, Z14, Z3                          // t = tw * Z14
	VADDPS         Z3, Z12, Z4                          // a + t
	VSUBPS         Z3, Z12, Z5                          // a - t
	VMOVUPS        Z4, (R8)                             // dst[0..7]
	VMOVUPS        Z5, 128(R8)                          // dst[16..23]

	VMOVUPS        64(R10), Z0                          // Z0 = tw[8..15]
	VMOVSLDUP      Z0, Z1                               // Z1 = [tw.re]
	VMOVSHDUP      Z0, Z2                               // Z2 = [tw.im]
	VSHUFPS        $0xB1, Z15, Z15, Z3                  // swap re/im of Z15
	VMULPS         Z2, Z3, Z3                           // Z3 *= w.im
	VFMADDSUB231PS Z1, Z15, Z3                          // t = tw * Z15
	VADDPS         Z3, Z13, Z4                          // a + t
	VSUBPS         Z3, Z13, Z5                          // a - t
	VMOVUPS        Z4, 64(R8)                           // dst[8..15]
	VMOVUPS        Z5, 192(R8)                          // dst[24..31]

	VZEROUPPER                                          // avoid AVX-SSE transition penalties
	MOVB           $1, ret+96(FP)                       // return true
	RET

s32r42_512_fwd_false:
	VZEROUPPER
	MOVB           $0, ret+96(FP)                       // return false
	RET

// ===========================================================================
// InverseAVX512Size32Radix4Then2Complex64Asm
// ===========================================================================
// Same dataflow as the forward kernel with conjugated twiddles
// (VFMSUBADD231PS), the two odd radix-4 outputs exchanged (+i instead of -i
// in the trivial stages) and a final 1/32 scaling folded into the stores.
TEXT ·InverseAVX512Size32Radix4Then2Complex64Asm(SB), NOSPLIT, $0-97
	MOVQ           dst+0(FP), R8                        // R8  = dst pointer
	MOVQ           src+24(FP), R9                       // R9  = src pointer
	MOVQ           twiddle+48(FP), R10                  // R10 = twiddle pointer

	MOVQ           src_len+32(FP), AX
	CMPQ           AX, $32
	JNE            s32r42_512_inv_false                 // this codelet only handles n == 32

	MOVQ           dst_len+8(FP), AX
	CMPQ           AX, $32
	JL             s32r42_512_inv_false                 // dst too short

	MOVQ           twiddle_len+56(FP), AX
	CMPQ           AX, $32
	JL             s32r42_512_inv_false                 // twiddle too short

	MOVQ           scratch_len+80(FP), AX
	CMPQ           AX, $32
	JL             s32r42_512_inv_false                 // scratch too short (unused, checked for API parity)

	// Constants: imaginary-sign mask and the two deinterleave index
	// vectors used before the final radix-2 stage.
	VMOVUPS        negImagPS512x32<>(SB), Z16           // Z16 = per-complex (0, -0.0) sign mask
	VMOVUPS        deintEvenQ512x32<>(SB), Z17          // Z17 = qword indices [0,2,4,6,0,2,4,6]
	VMOVUPS        deintOddQ512x32<>(SB), Z18           // Z18 = qword indices [1,3,5,7,1,3,5,7]

	// Merge masks. K1/K2/K3 select complex-lane pairs {2,3}, {4,5}, {6,7}
	// when building the stage-2 twiddle vectors; K4 selects the upper
	// four complex lanes for the deinterleave.
	MOVL           $0x0C, AX
	KMOVW          AX, K1                               // K1 = complex lanes 2,3
	MOVL           $0x30, AX
	KMOVW          AX, K2                               // K2 = complex lanes 4,5
	MOVL           $0xC0, AX
	KMOVW          AX, K3                               // K3 = complex lanes 6,7
	MOVL           $0xF0, AX
	KMOVW          AX, K4                               // K4 = complex lanes 4..7

	VBROADCASTSS   ·thirtySecond32(SB), Z26            // Z26 = [float32(1/32) x16]

	// Stage-2 twiddle vectors. Complex lanes 2j and 2j+1 both carry
	// tw[2*j*r] for member r of the radix-4 butterfly, so each vector is
	// four 64-bit broadcasts merged into one register.
	VBROADCASTSD   (R10), Z19                           // w1 = [tw[0] x8]
	VBROADCASTSD   16(R10), K1, Z19                     // lanes 2,3 = tw[2]
	VBROADCASTSD   32(R10), K2, Z19                     // lanes 4,5 = tw[4]
	VBROADCASTSD   48(R10), K3, Z19                     // lanes 6,7 = tw[6]
	VMOVSLDUP      Z19, Z20                             // Z20 = [w1.re]
	VMOVSHDUP      Z19, Z21                             // Z21 = [w1.im]
	VBROADCASTSD   (R10), Z19                           // w2 = [tw[0] x8]
	VBROADCASTSD   32(R10), K1, Z19                     // lanes 2,3 = tw[4]
	VBROADCASTSD   64(R10), K2, Z19                     // lanes 4,5 = tw[8]
	VBROADCASTSD   96(R10), K3, Z19                     // lanes 6,7 = tw[12]
	VMOVSLDUP      Z19, Z22                             // Z22 = [w2.re]
	VMOVSHDUP      Z19, Z23                             // Z23 = [w2.im]
	VBROADCASTSD   (R10), Z19                           // w3 = [tw[0] x8]
	VBROADCASTSD   48(R10), K1, Z19                     // lanes 2,3 = tw[6]
	VBROADCASTSD   96(R10), K2, Z19                     // lanes 4,5 = tw[12]
	VBROADCASTSD   144(R10), K3, Z19                    // lanes 6,7 = tw[18]
	VMOVSLDUP      Z19, Z24                             // Z24 = [w3.re]
	VMOVSHDUP      Z19, Z25                             // Z25 = [w3.im]

	// Load src[0..31] as four contiguous ZMM registers. Lane l of Zk is
	// src[8k+l]; the mixed-radix permutation [0,8,16,24, 2,10,18,26, ...]
	// means work[4b+k] = src[sigma(b)+8k] with sigma = [0,2,4,6,1,3,5,7],
	// so lane l already holds member k of stage-1 butterfly
	// b = l/2 (l even) or 4+(l-1)/2 (l odd): no permutation is needed.
	VMOVUPS        (R9), Z0                             // Z0 = src[0..7]
	VMOVUPS        64(R9), Z1                           // Z1 = src[8..15]
	VMOVUPS        128(R9), Z2                          // Z2 = src[16..23]
	VMOVUPS        192(R9), Z3                          // Z3 = src[24..31]

	// Stage 1: eight radix-4 butterflies with trivial twiddles, one per
	// complex lane. a0..a3 are Z0..Z3.
	VADDPS         Z2, Z0, Z4                           // t0 = a0 + a2
	VSUBPS         Z2, Z0, Z5                           // t1 = a0 - a2
	VADDPS         Z3, Z1, Z6                           // t2 = a1 + a3
	VSUBPS         Z3, Z1, Z7                           // t3 = a1 - a3
	VSHUFPS        $0xB1, Z7, Z7, Z0                    // (t3.im, t3.re) per complex
	VPXORQ         Z16, Z0, Z0                          // Z0 = (-i) * t3
	VADDPS         Z6, Z4, Z8                           // y0 = t0 + t2
	VSUBPS         Z6, Z4, Z10                          // y2 = t0 - t2
	VSUBPS         Z0, Z5, Z9                           // y1 = t1 + i*t3
	VADDPS         Z0, Z5, Z11                          // y3 = t1 + (-i)*t3

	// Stage 1 leaves register k holding work[4b+k] in lane pair b, while
	// stage 2 needs quadruple j of both groups in lane pair j of every
	// register. That is a 4x4 exchange of 128-bit lanes.
	VSHUFF32X4     $0x88, Z9, Z8, Z0
	VSHUFF32X4     $0xDD, Z9, Z8, Z1
	VSHUFF32X4     $0x88, Z11, Z10, Z2
	VSHUFF32X4     $0xDD, Z11, Z10, Z3
	VSHUFF32X4     $0x88, Z2, Z0, Z8                    // Q0 = quadruple member 0
	VSHUFF32X4     $0x88, Z3, Z1, Z9                    // Q1 = quadruple member 1
	VSHUFF32X4     $0xDD, Z2, Z0, Z10                   // Q2 = quadruple member 2
	VSHUFF32X4     $0xDD, Z3, Z1, Z11                   // Q3 = quadruple member 3

	// Stage 2: eight radix-4 butterflies (four per group) with twiddles
	// tw[2j], tw[4j], tw[6j] on members 1, 2, 3.
	VSHUFPS        $0xB1, Z9, Z9, Z0                    // swap re/im of Z9
	VMULPS         Z21, Z0, Z0                          // Z0 *= w.im
	VFMSUBADD231PS Z20, Z9, Z0                          // a1 = w1 * Q1
	VSHUFPS        $0xB1, Z10, Z10, Z1                  // swap re/im of Z10
	VMULPS         Z23, Z1, Z1                          // Z1 *= w.im
	VFMSUBADD231PS Z22, Z10, Z1                         // a2 = w2 * Q2
	VSHUFPS        $0xB1, Z11, Z11, Z2                  // swap re/im of Z11
	VMULPS         Z25, Z2, Z2                          // Z2 *= w.im
	VFMSUBADD231PS Z24, Z11, Z2                         // a3 = w3 * Q3
	VADDPS         Z1, Z8, Z3                           // t0 = a0 + a2
	VSUBPS         Z1, Z8, Z4                           // t1 = a0 - a2
	VADDPS         Z2, Z0, Z5                           // t2 = a1 + a3
	VSUBPS         Z2, Z0, Z6                           // t3 = a1 - a3
	VSHUFPS        $0xB1, Z6, Z6, Z7                    // (t3.im, t3.re) per complex
	VPXORQ         Z16, Z7, Z7                          // Z7 = (-i) * t3
	VADDPS         Z5, Z3, Z8                           // y0 = t0 + t2
	VSUBPS         Z5, Z3, Z10                          // y2 = t0 - t2
	VSUBPS         Z7, Z4, Z9                           // y1 = t1 + i*t3
	VADDPS         Z7, Z4, Z11                          // y3 = t1 + (-i)*t3

	// Register r now holds work[16g+4r+j] in lane 2j+g. Gather the even
	// lanes (group A) and odd lanes (group B) into natural order so the
	// final radix-2 stage sees contiguous halves and contiguous twiddles.
	VPERMPD        Z8, Z17, Z12                         // lanes 0..3 = work[0..3]
	VPERMPD        Z9, Z17, K4, Z12                     // lanes 4..7 = work[4..7]
	VPERMPD        Z10, Z17, Z13                        // lanes 0..3 = work[8..11]
	VPERMPD        Z11, Z17, K4, Z13                    // lanes 4..7 = work[12..15]
	VPERMPD        Z8, Z18, Z14                         // lanes 0..3 = work[16..19]
	VPERMPD        Z9, Z18, K4, Z14                     // lanes 4..7 = work[20..23]
	VPERMPD        Z10, Z18, Z15                        // lanes 0..3 = work[24..27]
	VPERMPD        Z11, Z18, K4, Z15                    // lanes 4..7 = work[28..31]

	// Stage 3: radix-2 over (work[i], work[i+16]) with twiddle tw[i].
	VMOVUPS        (R10), Z0                            // Z0 = tw[0..7]
	VMOVSLDUP      Z0, Z1                               // Z1 = [tw.re]
	VMOVSHDUP      Z0, Z2                               // Z2 = [tw.im]
	VSHUFPS        $0xB1, Z14, Z14, Z3                  // swap re/im of Z14
	VMULPS         Z2, Z3, Z3                           // Z3 *= w.im
	VFMSUBADD231PS Z1, Z14, Z3                          // t = tw * Z14
	VADDPS         Z3, Z12, Z4                          // a + t
	VSUBPS         Z3, Z12, Z5                          // a - t
	VMULPS         Z26, Z4, Z4                          // *= 1/32
	VMULPS         Z26, Z5, Z5                          // *= 1/32
	VMOVUPS        Z4, (R8)                             // dst[0..7]
	VMOVUPS        Z5, 128(R8)                          // dst[16..23]

	VMOVUPS        64(R10), Z0                          // Z0 = tw[8..15]
	VMOVSLDUP      Z0, Z1                               // Z1 = [tw.re]
	VMOVSHDUP      Z0, Z2                               // Z2 = [tw.im]
	VSHUFPS        $0xB1, Z15, Z15, Z3                  // swap re/im of Z15
	VMULPS         Z2, Z3, Z3                           // Z3 *= w.im
	VFMSUBADD231PS Z1, Z15, Z3                          // t = tw * Z15
	VADDPS         Z3, Z13, Z4                          // a + t
	VSUBPS         Z3, Z13, Z5                          // a - t
	VMULPS         Z26, Z4, Z4                          // *= 1/32
	VMULPS         Z26, Z5, Z5                          // *= 1/32
	VMOVUPS        Z4, 64(R8)                           // dst[8..15]
	VMOVUPS        Z5, 192(R8)                          // dst[24..31]

	VZEROUPPER                                          // avoid AVX-SSE transition penalties
	MOVB           $1, ret+96(FP)                       // return true
	RET

s32r42_512_inv_false:
	VZEROUPPER
	MOVB           $0, ret+96(FP)                       // return false
	RET


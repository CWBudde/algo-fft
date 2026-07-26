//go:build amd64 && !purego

// ===========================================================================
// AVX-512 Size-32 Radix-4-then-2 FFT Kernels for AMD64 (complex128)
// ===========================================================================
//
// A ZMM register holds 4 complex128, so all 32 points live in eight ZMM
// registers. The kernel loads once, runs all three stages (radix-4, radix-4,
// radix-2) entirely in registers and stores once: no memory traffic between
// stages, and the `scratch` slice is never touched. In-place (dst == src)
// therefore works without pointer juggling, since every input is consumed
// before the first store.
//
// DATAFLOW
// --------
// The mixed-radix bit-reversal of dit32_radix4_then2_* is
//   [0,8,16,24, 2,10,18,26, 4,12,20,28, 6,14,22,30,
//    1,9,17,25, 3,11,19,27, 5,13,21,29, 7,15,23,31]
// so work[4m+k] = src[2m+8k] for m < 4 (group A, work[0..15]) and
// work[16+4m+k] = src[2m+1+8k] (group B, work[16..31]). Stage 1's butterfly
// quadruples are therefore the even (group A) and odd (group B) 128-bit lanes
// of the eight contiguous loads: one VSHUFF64X2 pair per register pair
// replaces the whole permutation.
//
// After stage 1 lane m of register k holds work[4m+k], while stage 2 needs
// (work[j], work[j+4], work[j+8], work[j+12]) in lane j, so one 4x4 exchange
// of 128-bit lanes (8 x VSHUFF64X2) sits between the stages, per group.
// Stage 2 leaves the data in natural order (register q = work[4q..4q+3]), so
// stage 3's radix-2 butterflies pair register q of group A with register q of
// group B and use the contiguous twiddles tw[4q..4q+3].
//
// COMPLEX MULTIPLY (4 complex128 per ZMM)
// ---------------------------------------
//   VMOVDDUP      w -> [w.r, w.r, ...]
//   VSHUFPD $0xFF w -> [w.i, w.i, ...]
//   VSHUFPD $0x55 b -> [b.i, b.r, ...]
//   VMULPD + VFMADDSUB231PD  => t = w * b       (forward)
//   VMULPD + VFMSUBADD231PD  => t = conj(w) * b (inverse)
//
// The (-i) multiply is a swap of the real/imaginary doubles plus a sign flip
// of the imaginary one; i*t is its negation, so the two odd radix-4 outputs
// are one VADDPD and one VSUBPD against the same vector.
//
// REGISTERS
// ---------
//   Z0-Z7   scratch (initially the eight contiguous source loads)
//   Z8-Z11  group A data (work[0..15])
//   Z12-Z15 group B data (work[16..31])
//   Z16     imaginary-sign mask
//   Z20-Z25 broadcast real/imaginary parts of the stage-2 twiddles
//   Z26     1/32 scale (inverse only)
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

// negImagPD512x32 flips the sign of the imaginary (odd) double of each of the
// four complex128 in a ZMM register. File-scope: the shared masks in core.s
// are only 16/32 bytes wide.
DATA negImagPD512x32<>+0(SB)/8, $0x0000000000000000
DATA negImagPD512x32<>+8(SB)/8, $0x8000000000000000
DATA negImagPD512x32<>+16(SB)/8, $0x0000000000000000
DATA negImagPD512x32<>+24(SB)/8, $0x8000000000000000
DATA negImagPD512x32<>+32(SB)/8, $0x0000000000000000
DATA negImagPD512x32<>+40(SB)/8, $0x8000000000000000
DATA negImagPD512x32<>+48(SB)/8, $0x0000000000000000
DATA negImagPD512x32<>+56(SB)/8, $0x8000000000000000
GLOBL negImagPD512x32<>(SB), RODATA|NOPTR, $64

// ===========================================================================
// ForwardAVX512Size32Radix4Then2Complex128Asm
// ===========================================================================
TEXT ·ForwardAVX512Size32Radix4Then2Complex128Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer

	MOVQ src_len+32(FP), AX
	CMPQ AX, $32
	JNE  s32r42_512_fwd_false // this codelet only handles n == 32

	MOVQ dst_len+8(FP), AX
	CMPQ AX, $32
	JL   s32r42_512_fwd_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $32
	JL   s32r42_512_fwd_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $32
	JL   s32r42_512_fwd_false

	VMOVUPD negImagPD512x32<>(SB), Z16 // Z16 = imaginary-sign mask

	// ------------------------------------------------------------------
	// Stage-2 twiddle vectors (lane j needs tw[2j], tw[4j], tw[6j]) and
	// their broadcast forms. Data-independent, so hoisted to the top.
	// ------------------------------------------------------------------
	VMOVUPD      0(R10), X8
	VINSERTF32X4 $1, 32(R10), Z8, Z8
	VINSERTF32X4 $2, 64(R10), Z8, Z8
	VINSERTF32X4 $3, 96(R10), Z8, Z8

	// Z8 = w1 = (tw[0],tw[2],tw[4],tw[6])
	VMOVUPD      0(R10), X9
	VINSERTF32X4 $1, 64(R10), Z9, Z9
	VINSERTF32X4 $2, 128(R10), Z9, Z9
	VINSERTF32X4 $3, 192(R10), Z9, Z9

	// Z9 = w2 = (tw[0],tw[4],tw[8],tw[12])
	VMOVUPD      0(R10), X10
	VINSERTF32X4 $1, 96(R10), Z10, Z10
	VINSERTF32X4 $2, 192(R10), Z10, Z10
	VINSERTF32X4 $3, 288(R10), Z10, Z10

	// Z10 = w3 = (tw[0],tw[6],tw[12],tw[18])

	VMOVDDUP Z8, Z20              // Z20 = [w1.re x8]
	VSHUFPD  $0xFF, Z8, Z8, Z21   // Z21 = [w1.im x8]
	VMOVDDUP Z9, Z22              // Z22 = [w2.re x8]
	VSHUFPD  $0xFF, Z9, Z9, Z23   // Z23 = [w2.im x8]
	VMOVDDUP Z10, Z24             // Z24 = [w3.re x8]
	VSHUFPD  $0xFF, Z10, Z10, Z25 // Z25 = [w3.im x8]

	// ------------------------------------------------------------------
	// Load src[0..31] as eight contiguous ZMM registers.
	// ------------------------------------------------------------------
	VMOVUPD 0(R9), Z0   // Z0 = src[0..3]
	VMOVUPD 64(R9), Z1  // Z1 = src[4..7]
	VMOVUPD 128(R9), Z2 // Z2 = src[8..11]
	VMOVUPD 192(R9), Z3 // Z3 = src[12..15]
	VMOVUPD 256(R9), Z4 // Z4 = src[16..19]
	VMOVUPD 320(R9), Z5 // Z5 = src[20..23]
	VMOVUPD 384(R9), Z6 // Z6 = src[24..27]
	VMOVUPD 448(R9), Z7 // Z7 = src[28..31]

	// ------------------------------------------------------------------
	// Deinterleave into the two stage-1 register groups. The mixed-radix
	// bit-reversal [0,8,16,24, 2,10,18,26, ...] means group A's butterfly
	// m reads (src[2m], src[2m+8], src[2m+16], src[2m+24]) and group B's
	// reads the odd counterpart, so a 128-bit even/odd lane split of each
	// register pair produces both groups outright.
	// ------------------------------------------------------------------
	VSHUFF64X2 $0x88, Z1, Z0, Z8  // A0 = (src[0],src[2],src[4],src[6])
	VSHUFF64X2 $0xDD, Z1, Z0, Z12 // B0 = (src[1],src[3],src[5],src[7])
	VSHUFF64X2 $0x88, Z3, Z2, Z9  // A1 = (src[8],src[10],src[12],src[14])
	VSHUFF64X2 $0xDD, Z3, Z2, Z13 // B1 = (src[9],src[11],src[13],src[15])
	VSHUFF64X2 $0x88, Z5, Z4, Z10 // A2 = (src[16],src[18],src[20],src[22])
	VSHUFF64X2 $0xDD, Z5, Z4, Z14 // B2 = (src[17],src[19],src[21],src[23])
	VSHUFF64X2 $0x88, Z7, Z6, Z11 // A3 = (src[24],src[26],src[28],src[30])
	VSHUFF64X2 $0xDD, Z7, Z6, Z15 // B3 = (src[25],src[27],src[29],src[31])

	// Stage 1, group A (work[0..15]): radix-4 butterfly, trivial twiddles
	VADDPD  Z10, Z8, Z0       // Z0 = t0 = a0 + a2
	VSUBPD  Z10, Z8, Z1       // Z1 = t1 = a0 - a2
	VADDPD  Z11, Z9, Z2       // Z2 = t2 = a1 + a3
	VSUBPD  Z11, Z9, Z3       // Z3 = t3 = a1 - a3
	VSHUFPD $0x55, Z3, Z3, Z4 // Z4 = (t3.im, t3.re) per complex
	VPXORQ  Z16, Z4, Z4       // Z4 = (-i) * t3
	VADDPD  Z2, Z0, Z8        // y0 = t0 + t2
	VADDPD  Z4, Z1, Z9        // y1 = t1 + (-i)*t3
	VSUBPD  Z2, Z0, Z10       // y2 = t0 - t2
	VSUBPD  Z4, Z1, Z11       // y3 = t1 + i*t3

	// Stage 1, group B (work[16..31]): radix-4 butterfly, trivial twiddles
	VADDPD  Z14, Z12, Z0      // Z0 = t0 = a0 + a2
	VSUBPD  Z14, Z12, Z1      // Z1 = t1 = a0 - a2
	VADDPD  Z15, Z13, Z2      // Z2 = t2 = a1 + a3
	VSUBPD  Z15, Z13, Z3      // Z3 = t3 = a1 - a3
	VSHUFPD $0x55, Z3, Z3, Z4 // Z4 = (t3.im, t3.re) per complex
	VPXORQ  Z16, Z4, Z4       // Z4 = (-i) * t3
	VADDPD  Z2, Z0, Z12       // y0 = t0 + t2
	VADDPD  Z4, Z1, Z13       // y1 = t1 + (-i)*t3
	VSUBPD  Z2, Z0, Z14       // y2 = t0 - t2
	VSUBPD  Z4, Z1, Z15       // y3 = t1 + i*t3

	// Group A: 4x4 exchange of 128-bit lanes
	VSHUFF64X2 $0x88, Z9, Z8, Z0
	VSHUFF64X2 $0xDD, Z9, Z8, Z1
	VSHUFF64X2 $0x88, Z11, Z10, Z2
	VSHUFF64X2 $0xDD, Z11, Z10, Z3
	VSHUFF64X2 $0x88, Z2, Z0, Z8
	VSHUFF64X2 $0x88, Z3, Z1, Z9
	VSHUFF64X2 $0xDD, Z2, Z0, Z10
	VSHUFF64X2 $0xDD, Z3, Z1, Z11

	// Group B: 4x4 exchange of 128-bit lanes
	VSHUFF64X2 $0x88, Z13, Z12, Z0
	VSHUFF64X2 $0xDD, Z13, Z12, Z1
	VSHUFF64X2 $0x88, Z15, Z14, Z2
	VSHUFF64X2 $0xDD, Z15, Z14, Z3
	VSHUFF64X2 $0x88, Z2, Z0, Z12
	VSHUFF64X2 $0x88, Z3, Z1, Z13
	VSHUFF64X2 $0xDD, Z2, Z0, Z14
	VSHUFF64X2 $0xDD, Z3, Z1, Z15

	// Stage 2, base 0: radix-4 butterfly with twiddles (w1), (w2), (w3)
	VSHUFPD        $0x55, Z9, Z9, Z0   // Z0 = swap(Z9)
	VMULPD         Z21, Z0, Z0         // Z0 *= w.im
	VFMADDSUB231PD Z20, Z9, Z0         // Z0 = a1 = (w1)*C1
	VSHUFPD        $0x55, Z10, Z10, Z1 // Z1 = swap(Z10)
	VMULPD         Z23, Z1, Z1         // Z1 *= w.im
	VFMADDSUB231PD Z22, Z10, Z1        // Z1 = a2 = (w2)*C2
	VSHUFPD        $0x55, Z11, Z11, Z2 // Z2 = swap(Z11)
	VMULPD         Z25, Z2, Z2         // Z2 *= w.im
	VFMADDSUB231PD Z24, Z11, Z2        // Z2 = a3 = (w3)*C3
	VADDPD         Z1, Z8, Z3          // t0 = a0 + a2
	VSUBPD         Z1, Z8, Z4          // t1 = a0 - a2
	VADDPD         Z2, Z0, Z5          // t2 = a1 + a3
	VSUBPD         Z2, Z0, Z6          // t3 = a1 - a3
	VSHUFPD        $0x55, Z6, Z6, Z2   // Z2 = (t3.im, t3.re) per complex
	VPXORQ         Z16, Z2, Z2         // Z2 = (-i) * t3
	VADDPD         Z5, Z3, Z8          // y0 = t0 + t2
	VADDPD         Z2, Z4, Z9          // y1 = t1 + (-i)*t3
	VSUBPD         Z5, Z3, Z10         // y2 = t0 - t2
	VSUBPD         Z2, Z4, Z11         // y3 = t1 + i*t3

	// Stage 2, base 16: radix-4 butterfly with twiddles (w1), (w2), (w3)
	VSHUFPD        $0x55, Z13, Z13, Z0 // Z0 = swap(Z13)
	VMULPD         Z21, Z0, Z0         // Z0 *= w.im
	VFMADDSUB231PD Z20, Z13, Z0        // Z0 = a1 = (w1)*C1
	VSHUFPD        $0x55, Z14, Z14, Z1 // Z1 = swap(Z14)
	VMULPD         Z23, Z1, Z1         // Z1 *= w.im
	VFMADDSUB231PD Z22, Z14, Z1        // Z1 = a2 = (w2)*C2
	VSHUFPD        $0x55, Z15, Z15, Z2 // Z2 = swap(Z15)
	VMULPD         Z25, Z2, Z2         // Z2 *= w.im
	VFMADDSUB231PD Z24, Z15, Z2        // Z2 = a3 = (w3)*C3
	VADDPD         Z1, Z12, Z3         // t0 = a0 + a2
	VSUBPD         Z1, Z12, Z4         // t1 = a0 - a2
	VADDPD         Z2, Z0, Z5          // t2 = a1 + a3
	VSUBPD         Z2, Z0, Z6          // t3 = a1 - a3
	VSHUFPD        $0x55, Z6, Z6, Z2   // Z2 = (t3.im, t3.re) per complex
	VPXORQ         Z16, Z2, Z2         // Z2 = (-i) * t3
	VADDPD         Z5, Z3, Z12         // y0 = t0 + t2
	VADDPD         Z2, Z4, Z13         // y1 = t1 + (-i)*t3
	VSUBPD         Z5, Z3, Z14         // y2 = t0 - t2
	VSUBPD         Z2, Z4, Z15         // y3 = t1 + i*t3

	// ------------------------------------------------------------------
	// Stage 3: radix-2 over (work[i], work[i+16]) with twiddle tw[i].
	// Both halves and their twiddles are contiguous, so this is four
	// straight-line butterflies with one load and two stores each.
	// ------------------------------------------------------------------
	VMOVUPD        0(R10), Z0          // Z0 = tw[0..3]
	VMOVDDUP       Z0, Z1              // Z1 = [tw.re x8]
	VSHUFPD        $0xFF, Z0, Z0, Z2   // Z2 = [tw.im x8]
	VSHUFPD        $0x55, Z12, Z12, Z3 // Z3 = swap(Z12)
	VMULPD         Z2, Z3, Z3          // Z3 *= w.im
	VFMADDSUB231PD Z1, Z12, Z3         // Z3 = t = tw[0..3] * work[16..19]
	VADDPD         Z3, Z8, Z4          // Z4 = work[0..3] + t
	VSUBPD         Z3, Z8, Z5          // Z5 = work[0..3] - t
	VMOVUPD        Z4, 0(R8)           // dst[0..3]
	VMOVUPD        Z5, 256(R8)         // dst[16..19]

	VMOVUPD        64(R10), Z0         // Z0 = tw[4..7]
	VMOVDDUP       Z0, Z1              // Z1 = [tw.re x8]
	VSHUFPD        $0xFF, Z0, Z0, Z2   // Z2 = [tw.im x8]
	VSHUFPD        $0x55, Z13, Z13, Z3 // Z3 = swap(Z13)
	VMULPD         Z2, Z3, Z3          // Z3 *= w.im
	VFMADDSUB231PD Z1, Z13, Z3         // Z3 = t = tw[4..7] * work[20..23]
	VADDPD         Z3, Z9, Z4          // Z4 = work[4..7] + t
	VSUBPD         Z3, Z9, Z5          // Z5 = work[4..7] - t
	VMOVUPD        Z4, 64(R8)          // dst[4..7]
	VMOVUPD        Z5, 320(R8)         // dst[20..23]

	VMOVUPD        128(R10), Z0        // Z0 = tw[8..11]
	VMOVDDUP       Z0, Z1              // Z1 = [tw.re x8]
	VSHUFPD        $0xFF, Z0, Z0, Z2   // Z2 = [tw.im x8]
	VSHUFPD        $0x55, Z14, Z14, Z3 // Z3 = swap(Z14)
	VMULPD         Z2, Z3, Z3          // Z3 *= w.im
	VFMADDSUB231PD Z1, Z14, Z3         // Z3 = t = tw[8..11] * work[24..27]
	VADDPD         Z3, Z10, Z4         // Z4 = work[8..11] + t
	VSUBPD         Z3, Z10, Z5         // Z5 = work[8..11] - t
	VMOVUPD        Z4, 128(R8)         // dst[8..11]
	VMOVUPD        Z5, 384(R8)         // dst[24..27]

	VMOVUPD        192(R10), Z0        // Z0 = tw[12..15]
	VMOVDDUP       Z0, Z1              // Z1 = [tw.re x8]
	VSHUFPD        $0xFF, Z0, Z0, Z2   // Z2 = [tw.im x8]
	VSHUFPD        $0x55, Z15, Z15, Z3 // Z3 = swap(Z15)
	VMULPD         Z2, Z3, Z3          // Z3 *= w.im
	VFMADDSUB231PD Z1, Z15, Z3         // Z3 = t = tw[12..15] * work[28..31]
	VADDPD         Z3, Z11, Z4         // Z4 = work[12..15] + t
	VSUBPD         Z3, Z11, Z5         // Z5 = work[12..15] - t
	VMOVUPD        Z4, 192(R8)         // dst[12..15]
	VMOVUPD        Z5, 448(R8)         // dst[28..31]

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

s32r42_512_fwd_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// InverseAVX512Size32Radix4Then2Complex128Asm
// ===========================================================================
// Same dataflow as the forward kernel with conjugated twiddles
// (VFMSUBADD231PD), the two odd radix-4 outputs swapped (+i instead of -i in
// the trivial stages), and a final 1/32 scaling.
TEXT ·InverseAVX512Size32Radix4Then2Complex128Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer

	MOVQ src_len+32(FP), AX
	CMPQ AX, $32
	JNE  s32r42_512_inv_false // this codelet only handles n == 32

	MOVQ dst_len+8(FP), AX
	CMPQ AX, $32
	JL   s32r42_512_inv_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $32
	JL   s32r42_512_inv_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $32
	JL   s32r42_512_inv_false

	VMOVUPD negImagPD512x32<>(SB), Z16 // Z16 = imaginary-sign mask

	MOVQ         ·thirtySecond64(SB), AX // AX = float64(1.0/32.0)
	VMOVQ        AX, X26
	VBROADCASTSD X26, Z26                // Z26 = [1/32 x8]

	// ------------------------------------------------------------------
	// Stage-2 twiddle vectors (lane j needs tw[2j], tw[4j], tw[6j]) and
	// their broadcast forms. Data-independent, so hoisted to the top.
	// ------------------------------------------------------------------
	VMOVUPD      0(R10), X8
	VINSERTF32X4 $1, 32(R10), Z8, Z8
	VINSERTF32X4 $2, 64(R10), Z8, Z8
	VINSERTF32X4 $3, 96(R10), Z8, Z8

	// Z8 = w1 = (tw[0],tw[2],tw[4],tw[6])
	VMOVUPD      0(R10), X9
	VINSERTF32X4 $1, 64(R10), Z9, Z9
	VINSERTF32X4 $2, 128(R10), Z9, Z9
	VINSERTF32X4 $3, 192(R10), Z9, Z9

	// Z9 = w2 = (tw[0],tw[4],tw[8],tw[12])
	VMOVUPD      0(R10), X10
	VINSERTF32X4 $1, 96(R10), Z10, Z10
	VINSERTF32X4 $2, 192(R10), Z10, Z10
	VINSERTF32X4 $3, 288(R10), Z10, Z10

	// Z10 = w3 = (tw[0],tw[6],tw[12],tw[18])

	VMOVDDUP Z8, Z20              // Z20 = [w1.re x8]
	VSHUFPD  $0xFF, Z8, Z8, Z21   // Z21 = [w1.im x8]
	VMOVDDUP Z9, Z22              // Z22 = [w2.re x8]
	VSHUFPD  $0xFF, Z9, Z9, Z23   // Z23 = [w2.im x8]
	VMOVDDUP Z10, Z24             // Z24 = [w3.re x8]
	VSHUFPD  $0xFF, Z10, Z10, Z25 // Z25 = [w3.im x8]

	// ------------------------------------------------------------------
	// Load src[0..31] as eight contiguous ZMM registers.
	// ------------------------------------------------------------------
	VMOVUPD 0(R9), Z0   // Z0 = src[0..3]
	VMOVUPD 64(R9), Z1  // Z1 = src[4..7]
	VMOVUPD 128(R9), Z2 // Z2 = src[8..11]
	VMOVUPD 192(R9), Z3 // Z3 = src[12..15]
	VMOVUPD 256(R9), Z4 // Z4 = src[16..19]
	VMOVUPD 320(R9), Z5 // Z5 = src[20..23]
	VMOVUPD 384(R9), Z6 // Z6 = src[24..27]
	VMOVUPD 448(R9), Z7 // Z7 = src[28..31]

	// ------------------------------------------------------------------
	// Deinterleave into the two stage-1 register groups. The mixed-radix
	// bit-reversal [0,8,16,24, 2,10,18,26, ...] means group A's butterfly
	// m reads (src[2m], src[2m+8], src[2m+16], src[2m+24]) and group B's
	// reads the odd counterpart, so a 128-bit even/odd lane split of each
	// register pair produces both groups outright.
	// ------------------------------------------------------------------
	VSHUFF64X2 $0x88, Z1, Z0, Z8  // A0 = (src[0],src[2],src[4],src[6])
	VSHUFF64X2 $0xDD, Z1, Z0, Z12 // B0 = (src[1],src[3],src[5],src[7])
	VSHUFF64X2 $0x88, Z3, Z2, Z9  // A1 = (src[8],src[10],src[12],src[14])
	VSHUFF64X2 $0xDD, Z3, Z2, Z13 // B1 = (src[9],src[11],src[13],src[15])
	VSHUFF64X2 $0x88, Z5, Z4, Z10 // A2 = (src[16],src[18],src[20],src[22])
	VSHUFF64X2 $0xDD, Z5, Z4, Z14 // B2 = (src[17],src[19],src[21],src[23])
	VSHUFF64X2 $0x88, Z7, Z6, Z11 // A3 = (src[24],src[26],src[28],src[30])
	VSHUFF64X2 $0xDD, Z7, Z6, Z15 // B3 = (src[25],src[27],src[29],src[31])

	// Stage 1, group A (work[0..15]): radix-4 butterfly, trivial twiddles
	VADDPD  Z10, Z8, Z0       // Z0 = t0 = a0 + a2
	VSUBPD  Z10, Z8, Z1       // Z1 = t1 = a0 - a2
	VADDPD  Z11, Z9, Z2       // Z2 = t2 = a1 + a3
	VSUBPD  Z11, Z9, Z3       // Z3 = t3 = a1 - a3
	VSHUFPD $0x55, Z3, Z3, Z4 // Z4 = (t3.im, t3.re) per complex
	VPXORQ  Z16, Z4, Z4       // Z4 = (-i) * t3
	VADDPD  Z2, Z0, Z8        // y0 = t0 + t2
	VSUBPD  Z4, Z1, Z9        // y1 = t1 + i*t3
	VSUBPD  Z2, Z0, Z10       // y2 = t0 - t2
	VADDPD  Z4, Z1, Z11       // y3 = t1 + (-i)*t3

	// Stage 1, group B (work[16..31]): radix-4 butterfly, trivial twiddles
	VADDPD  Z14, Z12, Z0      // Z0 = t0 = a0 + a2
	VSUBPD  Z14, Z12, Z1      // Z1 = t1 = a0 - a2
	VADDPD  Z15, Z13, Z2      // Z2 = t2 = a1 + a3
	VSUBPD  Z15, Z13, Z3      // Z3 = t3 = a1 - a3
	VSHUFPD $0x55, Z3, Z3, Z4 // Z4 = (t3.im, t3.re) per complex
	VPXORQ  Z16, Z4, Z4       // Z4 = (-i) * t3
	VADDPD  Z2, Z0, Z12       // y0 = t0 + t2
	VSUBPD  Z4, Z1, Z13       // y1 = t1 + i*t3
	VSUBPD  Z2, Z0, Z14       // y2 = t0 - t2
	VADDPD  Z4, Z1, Z15       // y3 = t1 + (-i)*t3

	// Group A: 4x4 exchange of 128-bit lanes
	VSHUFF64X2 $0x88, Z9, Z8, Z0
	VSHUFF64X2 $0xDD, Z9, Z8, Z1
	VSHUFF64X2 $0x88, Z11, Z10, Z2
	VSHUFF64X2 $0xDD, Z11, Z10, Z3
	VSHUFF64X2 $0x88, Z2, Z0, Z8
	VSHUFF64X2 $0x88, Z3, Z1, Z9
	VSHUFF64X2 $0xDD, Z2, Z0, Z10
	VSHUFF64X2 $0xDD, Z3, Z1, Z11

	// Group B: 4x4 exchange of 128-bit lanes
	VSHUFF64X2 $0x88, Z13, Z12, Z0
	VSHUFF64X2 $0xDD, Z13, Z12, Z1
	VSHUFF64X2 $0x88, Z15, Z14, Z2
	VSHUFF64X2 $0xDD, Z15, Z14, Z3
	VSHUFF64X2 $0x88, Z2, Z0, Z12
	VSHUFF64X2 $0x88, Z3, Z1, Z13
	VSHUFF64X2 $0xDD, Z2, Z0, Z14
	VSHUFF64X2 $0xDD, Z3, Z1, Z15

	// Stage 2, base 0: radix-4 butterfly with twiddles conj(w1), conj(w2), conj(w3)
	VSHUFPD        $0x55, Z9, Z9, Z0   // Z0 = swap(Z9)
	VMULPD         Z21, Z0, Z0         // Z0 *= w.im
	VFMSUBADD231PD Z20, Z9, Z0         // Z0 = a1 = conj(w1)*C1
	VSHUFPD        $0x55, Z10, Z10, Z1 // Z1 = swap(Z10)
	VMULPD         Z23, Z1, Z1         // Z1 *= w.im
	VFMSUBADD231PD Z22, Z10, Z1        // Z1 = a2 = conj(w2)*C2
	VSHUFPD        $0x55, Z11, Z11, Z2 // Z2 = swap(Z11)
	VMULPD         Z25, Z2, Z2         // Z2 *= w.im
	VFMSUBADD231PD Z24, Z11, Z2        // Z2 = a3 = conj(w3)*C3
	VADDPD         Z1, Z8, Z3          // t0 = a0 + a2
	VSUBPD         Z1, Z8, Z4          // t1 = a0 - a2
	VADDPD         Z2, Z0, Z5          // t2 = a1 + a3
	VSUBPD         Z2, Z0, Z6          // t3 = a1 - a3
	VSHUFPD        $0x55, Z6, Z6, Z2   // Z2 = (t3.im, t3.re) per complex
	VPXORQ         Z16, Z2, Z2         // Z2 = (-i) * t3
	VADDPD         Z5, Z3, Z8          // y0 = t0 + t2
	VSUBPD         Z2, Z4, Z9          // y1 = t1 + i*t3
	VSUBPD         Z5, Z3, Z10         // y2 = t0 - t2
	VADDPD         Z2, Z4, Z11         // y3 = t1 + (-i)*t3

	// Stage 2, base 16: radix-4 butterfly with twiddles conj(w1), conj(w2), conj(w3)
	VSHUFPD        $0x55, Z13, Z13, Z0 // Z0 = swap(Z13)
	VMULPD         Z21, Z0, Z0         // Z0 *= w.im
	VFMSUBADD231PD Z20, Z13, Z0        // Z0 = a1 = conj(w1)*C1
	VSHUFPD        $0x55, Z14, Z14, Z1 // Z1 = swap(Z14)
	VMULPD         Z23, Z1, Z1         // Z1 *= w.im
	VFMSUBADD231PD Z22, Z14, Z1        // Z1 = a2 = conj(w2)*C2
	VSHUFPD        $0x55, Z15, Z15, Z2 // Z2 = swap(Z15)
	VMULPD         Z25, Z2, Z2         // Z2 *= w.im
	VFMSUBADD231PD Z24, Z15, Z2        // Z2 = a3 = conj(w3)*C3
	VADDPD         Z1, Z12, Z3         // t0 = a0 + a2
	VSUBPD         Z1, Z12, Z4         // t1 = a0 - a2
	VADDPD         Z2, Z0, Z5          // t2 = a1 + a3
	VSUBPD         Z2, Z0, Z6          // t3 = a1 - a3
	VSHUFPD        $0x55, Z6, Z6, Z2   // Z2 = (t3.im, t3.re) per complex
	VPXORQ         Z16, Z2, Z2         // Z2 = (-i) * t3
	VADDPD         Z5, Z3, Z12         // y0 = t0 + t2
	VSUBPD         Z2, Z4, Z13         // y1 = t1 + i*t3
	VSUBPD         Z5, Z3, Z14         // y2 = t0 - t2
	VADDPD         Z2, Z4, Z15         // y3 = t1 + (-i)*t3

	// ------------------------------------------------------------------
	// Stage 3: radix-2 over (work[i], work[i+16]) with twiddle tw[i].
	// Both halves and their twiddles are contiguous, so this is four
	// straight-line butterflies with one load and two stores each.
	// ------------------------------------------------------------------
	VMOVUPD        0(R10), Z0          // Z0 = tw[0..3]
	VMOVDDUP       Z0, Z1              // Z1 = [tw.re x8]
	VSHUFPD        $0xFF, Z0, Z0, Z2   // Z2 = [tw.im x8]
	VSHUFPD        $0x55, Z12, Z12, Z3 // Z3 = swap(Z12)
	VMULPD         Z2, Z3, Z3          // Z3 *= w.im
	VFMSUBADD231PD Z1, Z12, Z3         // Z3 = t = tw[0..3] * work[16..19]
	VADDPD         Z3, Z8, Z4          // Z4 = work[0..3] + t
	VSUBPD         Z3, Z8, Z5          // Z5 = work[0..3] - t
	VMULPD         Z26, Z4, Z4         // apply 1/32
	VMULPD         Z26, Z5, Z5
	VMOVUPD        Z4, 0(R8)           // dst[0..3]
	VMOVUPD        Z5, 256(R8)         // dst[16..19]

	VMOVUPD        64(R10), Z0         // Z0 = tw[4..7]
	VMOVDDUP       Z0, Z1              // Z1 = [tw.re x8]
	VSHUFPD        $0xFF, Z0, Z0, Z2   // Z2 = [tw.im x8]
	VSHUFPD        $0x55, Z13, Z13, Z3 // Z3 = swap(Z13)
	VMULPD         Z2, Z3, Z3          // Z3 *= w.im
	VFMSUBADD231PD Z1, Z13, Z3         // Z3 = t = tw[4..7] * work[20..23]
	VADDPD         Z3, Z9, Z4          // Z4 = work[4..7] + t
	VSUBPD         Z3, Z9, Z5          // Z5 = work[4..7] - t
	VMULPD         Z26, Z4, Z4         // apply 1/32
	VMULPD         Z26, Z5, Z5
	VMOVUPD        Z4, 64(R8)          // dst[4..7]
	VMOVUPD        Z5, 320(R8)         // dst[20..23]

	VMOVUPD        128(R10), Z0        // Z0 = tw[8..11]
	VMOVDDUP       Z0, Z1              // Z1 = [tw.re x8]
	VSHUFPD        $0xFF, Z0, Z0, Z2   // Z2 = [tw.im x8]
	VSHUFPD        $0x55, Z14, Z14, Z3 // Z3 = swap(Z14)
	VMULPD         Z2, Z3, Z3          // Z3 *= w.im
	VFMSUBADD231PD Z1, Z14, Z3         // Z3 = t = tw[8..11] * work[24..27]
	VADDPD         Z3, Z10, Z4         // Z4 = work[8..11] + t
	VSUBPD         Z3, Z10, Z5         // Z5 = work[8..11] - t
	VMULPD         Z26, Z4, Z4         // apply 1/32
	VMULPD         Z26, Z5, Z5
	VMOVUPD        Z4, 128(R8)         // dst[8..11]
	VMOVUPD        Z5, 384(R8)         // dst[24..27]

	VMOVUPD        192(R10), Z0        // Z0 = tw[12..15]
	VMOVDDUP       Z0, Z1              // Z1 = [tw.re x8]
	VSHUFPD        $0xFF, Z0, Z0, Z2   // Z2 = [tw.im x8]
	VSHUFPD        $0x55, Z15, Z15, Z3 // Z3 = swap(Z15)
	VMULPD         Z2, Z3, Z3          // Z3 *= w.im
	VFMSUBADD231PD Z1, Z15, Z3         // Z3 = t = tw[12..15] * work[28..31]
	VADDPD         Z3, Z11, Z4         // Z4 = work[12..15] + t
	VSUBPD         Z3, Z11, Z5         // Z5 = work[12..15] - t
	VMULPD         Z26, Z4, Z4         // apply 1/32
	VMULPD         Z26, Z5, Z5
	VMOVUPD        Z4, 192(R8)         // dst[12..15]
	VMOVUPD        Z5, 448(R8)         // dst[28..31]

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

s32r42_512_inv_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

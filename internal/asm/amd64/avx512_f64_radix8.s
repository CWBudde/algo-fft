//go:build amd64 && !purego

// ===========================================================================
// AVX-512 radix-8 DIT FFT, 512-bit wide, complex128
// ===========================================================================
//
// The 512-bit twin of avx2_f64_radix8.s: same algorithm, same packed twiddle
// planes, same stage-1 group index table (the permutation is precision- and
// width-independent, so every radix-8 kernel in the tree shares
// internal/kernels' radix8GroupIndices). Read the header of avx2_f32_radix8.s
// for why the ladder is shaped the way it is and for the derivation of the
// eighth-root rotations; neither is re-derived here.
//
// A ZMM holds four complex128 where a YMM holds two, so everything doubles:
//
//   - Stage 1 gathers four groups per iteration. There is still no
//     128-bit-element gather, so each group's element is an XMM load folded in
//     with VINSERTF32X4 (VINSERTF64X2 is AVX512DQ; the F-encodable
//     VINSERTF32X4 selects the same 128-bit lane). Unlike the AVX2 twin the
//     four group base pointers are *held* in GP registers and a single digit
//     offset is stepped, so the gather costs seven ADDQ per iteration rather
//     than the AVX2 kernel's fourteen for half the work.
//   - The output transpose is two 4x4 blocks of 128-bit lanes (digits 0..3 and
//     4..7), eight VSHUFF64X2 each, rather than the AVX2 kernel's eight
//     VPERM2F128.
//   - The inner loop is unrolled two ZMM deep. m is a power of eight and hence
//     always a multiple of eight, so eight butterflies per iteration always
//     divides. The two halves use disjoint register sets (Z0-Z11 and Z12-Z23)
//     and are therefore two independent dependency chains inside one loop
//     body: that, not the width, is the point of this kernel. The 256-bit
//     radix-8 stage measured 1.24-1.56x a radix-4 stage per pass at
//     n = 512..2048 -- wholly inside L1, and out of a $0 frame with no stack
//     traffic -- because eight live streams plus two rotation masks and the
//     sqrt(2)/2 broadcast leave five scratch YMM of sixteen, exactly one
//     butterfly's worth. Thirty-two ZMM leave room for two, and halve the
//     seven-pointer loop overhead per butterfly at the same time.
//   - The stage bound stays `AX <= 2*limit`: AX is the byte stride m*16 (the
//     element size did not change), and the span it must be compared against
//     is 8*m = AX/2.
//
// Only AVX512F is used: VPXORQ rather than the DQ-only VXORPD, VINSERTF32X4
// rather than VINSERTF64X2. Callers gate on cpu.Features.HasAVX512. Every
// instruction is VEX- or EVEX-encoded; no legacy-SSE form appears.
// ===========================================================================

#include "textflag.h"

// r8eNegOdd flips the sign of the high float64 of each 128-bit lane (the
// imaginary part), so xor(swap(v), r8eNegOdd) = -i*v.
GLOBL ·r8eNegOdd<>(SB), RODATA|NOPTR, $64
DATA ·r8eNegOdd<>+0(SB)/8,  $0x0000000000000000
DATA ·r8eNegOdd<>+8(SB)/8,  $0x8000000000000000
DATA ·r8eNegOdd<>+16(SB)/8, $0x0000000000000000
DATA ·r8eNegOdd<>+24(SB)/8, $0x8000000000000000
DATA ·r8eNegOdd<>+32(SB)/8, $0x0000000000000000
DATA ·r8eNegOdd<>+40(SB)/8, $0x8000000000000000
DATA ·r8eNegOdd<>+48(SB)/8, $0x0000000000000000
DATA ·r8eNegOdd<>+56(SB)/8, $0x8000000000000000

// r8eNegEven flips the sign of the low float64 of each 128-bit lane (the real
// part), so xor(swap(v), r8eNegEven) = +i*v.
GLOBL ·r8eNegEven<>(SB), RODATA|NOPTR, $64
DATA ·r8eNegEven<>+0(SB)/8,  $0x8000000000000000
DATA ·r8eNegEven<>+8(SB)/8,  $0x0000000000000000
DATA ·r8eNegEven<>+16(SB)/8, $0x8000000000000000
DATA ·r8eNegEven<>+24(SB)/8, $0x0000000000000000
DATA ·r8eNegEven<>+32(SB)/8, $0x8000000000000000
DATA ·r8eNegEven<>+40(SB)/8, $0x0000000000000000
DATA ·r8eNegEven<>+48(SB)/8, $0x8000000000000000
DATA ·r8eNegEven<>+56(SB)/8, $0x0000000000000000

// r8eRoot2f64 is sqrt(2)/2 as a float64, broadcast from memory rather than
// materialised through a GP register: the MOVQ/VMOVQ/VBROADCASTSD route costs
// a fixed ~100ns per call, which is invisible at n = 32768 and mis-ranks the
// kernel at n = 32.
GLOBL ·r8eRoot2f64<>(SB), RODATA|NOPTR, $8
DATA ·r8eRoot2f64<>+0(SB)/8, $0x3FE6A09E667F3BCD

// func Radix8AVX512Complex128Asm(dst, src, twiddle, scratch []complex128, idx []int32, limit int, inverse bool, scale float64) bool
TEXT ·Radix8AVX512Complex128Asm(SB), NOSPLIT, $0-145
	MOVQ dst+0(FP), R8       // R8  = working buffer (dst, or scratch when in-place)
	MOVQ src+24(FP), R9      // R9  = src
	MOVQ scratch+72(FP), R11 // R11 = scratch
	MOVQ idx+96(FP), R12     // R12 = stage-1 group index table
	MOVQ src_len+32(FP), R13 // R13 = n

	// n >= 32 and a power of two is guaranteed by the Go caller; here we only
	// check that every slice is long enough to be safe to write. Stage 1
	// retires four groups per iteration, so n/8 must be a multiple of four --
	// which every power of two from 32 up is.
	CMPQ R13, $32
	JL   r8e_false

	MOVQ dst_len+8(FP), AX
	CMPQ AX, R13
	JL   r8e_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, R13
	JL   r8e_false

	// The packed table is n+8 elements (see twiddleSizeRadix8); requiring more
	// than n is what makes a caller that passes the plain length-n twiddle
	// table decline here rather than transform against it. The padding also
	// covers the 8-byte-offset VMOVDDUP the imaginary broadcasts use, which on
	// a ZMM reads 64 bytes and so runs eight bytes past its four twiddles.
	MOVQ twiddle_len+56(FP), AX
	LEAQ 8(R13), BX
	CMPQ AX, BX
	JL   r8e_false

	MOVQ idx_len+104(FP), AX // needs n/8 entries
	MOVQ R13, BX
	SHRQ $3, BX
	CMPQ AX, BX
	JL   r8e_false

	// Rotation masks. Forward: the first rotation is -i (negOdd) and the
	// second +i (negEven); the inverse conjugates the butterfly, so they swap.
	MOVBLZX inverse+128(FP), AX
	TESTL   AX, AX
	JNZ     r8e_inverse_masks

	VMOVUPD ·r8eNegOdd<>(SB), Z28
	VMOVUPD ·r8eNegEven<>(SB), Z29
	JMP     r8e_masks_done

r8e_inverse_masks:
	VMOVUPD ·r8eNegEven<>(SB), Z28
	VMOVUPD ·r8eNegOdd<>(SB), Z29

r8e_masks_done:
	VBROADCASTSD ·r8eRoot2f64<>(SB), Z30 // sqrt(2)/2
	VBROADCASTSD scale+136(FP), Z31      // 1/n for inverse, 1.0 for forward

	// In-place transforms work out of scratch and copy back at the end.
	CMPQ R8, R9
	JNE  r8e_permute
	MOVQ R11, R8

r8e_permute:
	// =====================================================================
	// Permutation and stage 1, fused. The eight inputs of group g are
	// src[idx[g] + d*q] for d = 0..7, q = n/8; four groups at a time give each
	// x_d vector directly, so the separate permutation pass and the input
	// transpose both disappear.
	//
	// The four group bases live in AX, BX, DI and BP for the whole iteration
	// and SI carries the digit offset d*q*16, so stepping to the next digit is
	// one ADDQ rather than four.
	// =====================================================================
	MOVQ R13, R10
	SHRQ $3, R10 // R10 = q = n/8 = the group count
	MOVQ R10, R11
	SHLQ $4, R11 // R11 = q*16 bytes, the distance between digit streams

	XORQ CX, CX // group index
	XORQ DX, DX // byte offset into work

r8e_stage1_loop:
	MOVLQZX (R12)(CX*4), AX // idx[g]
	SHLQ    $4, AX          // element index -> byte offset
	ADDQ    R9, AX          // AX = &src[idx[g]]

	MOVLQZX 4(R12)(CX*4), BX
	SHLQ    $4, BX
	ADDQ    R9, BX          // BX = &src[idx[g+1]]

	MOVLQZX 8(R12)(CX*4), DI
	SHLQ    $4, DI
	ADDQ    R9, DI          // DI = &src[idx[g+2]]

	MOVLQZX 12(R12)(CX*4), BP
	SHLQ    $4, BP
	ADDQ    R9, BP          // BP = &src[idx[g+3]]

	XORQ SI, SI // SI = d*q*16, the digit offset

	VMOVUPD      (AX)(SI*1), X0
	VINSERTF32X4 $1, (BX)(SI*1), Z0, Z0
	VINSERTF32X4 $2, (DI)(SI*1), Z0, Z0
	VINSERTF32X4 $3, (BP)(SI*1), Z0, Z0 // x0 = [A0, B0, C0, D0]
	ADDQ         R11, SI

	VMOVUPD      (AX)(SI*1), X1
	VINSERTF32X4 $1, (BX)(SI*1), Z1, Z1
	VINSERTF32X4 $2, (DI)(SI*1), Z1, Z1
	VINSERTF32X4 $3, (BP)(SI*1), Z1, Z1 // x1
	ADDQ         R11, SI

	VMOVUPD      (AX)(SI*1), X2
	VINSERTF32X4 $1, (BX)(SI*1), Z2, Z2
	VINSERTF32X4 $2, (DI)(SI*1), Z2, Z2
	VINSERTF32X4 $3, (BP)(SI*1), Z2, Z2 // x2
	ADDQ         R11, SI

	VMOVUPD      (AX)(SI*1), X3
	VINSERTF32X4 $1, (BX)(SI*1), Z3, Z3
	VINSERTF32X4 $2, (DI)(SI*1), Z3, Z3
	VINSERTF32X4 $3, (BP)(SI*1), Z3, Z3 // x3
	ADDQ         R11, SI

	VMOVUPD      (AX)(SI*1), X4
	VINSERTF32X4 $1, (BX)(SI*1), Z4, Z4
	VINSERTF32X4 $2, (DI)(SI*1), Z4, Z4
	VINSERTF32X4 $3, (BP)(SI*1), Z4, Z4 // x4
	ADDQ         R11, SI

	VMOVUPD      (AX)(SI*1), X5
	VINSERTF32X4 $1, (BX)(SI*1), Z5, Z5
	VINSERTF32X4 $2, (DI)(SI*1), Z5, Z5
	VINSERTF32X4 $3, (BP)(SI*1), Z5, Z5 // x5
	ADDQ         R11, SI

	VMOVUPD      (AX)(SI*1), X6
	VINSERTF32X4 $1, (BX)(SI*1), Z6, Z6
	VINSERTF32X4 $2, (DI)(SI*1), Z6, Z6
	VINSERTF32X4 $3, (BP)(SI*1), Z6, Z6 // x6
	ADDQ         R11, SI

	VMOVUPD      (AX)(SI*1), X7
	VINSERTF32X4 $1, (BX)(SI*1), Z7, Z7
	VINSERTF32X4 $2, (DI)(SI*1), Z7, Z7
	VINSERTF32X4 $3, (BP)(SI*1), Z7, Z7 // x7

	// ---- the eight-point butterfly, x0..x7 in Z0..Z7 -------------------
	VADDPD Z4, Z0, Z8  // a0 = x0 + x4
	VSUBPD Z4, Z0, Z9  // a1 = x0 - x4
	VADDPD Z6, Z2, Z10 // a2 = x2 + x6
	VSUBPD Z6, Z2, Z11 // a3 = x2 - x6
	VADDPD Z5, Z1, Z0  // a4 = x1 + x5
	VSUBPD Z5, Z1, Z2  // a5 = x1 - x5
	VADDPD Z7, Z3, Z4  // a6 = x3 + x7
	VSUBPD Z7, Z3, Z6  // a7 = x3 - x7

	VADDPD    Z10, Z8, Z1 // e0 = a0 + a2
	VSUBPD    Z10, Z8, Z3 // e2 = a0 - a2
	VPERMILPD $0x55, Z11, Z5
	VPXORQ    Z28, Z5, Z7 // rot1(a3)
	VPXORQ    Z29, Z5, Z5 // rot2(a3)
	VADDPD    Z9, Z7, Z7  // e1
	VADDPD    Z9, Z5, Z5  // e3

	VADDPD    Z4, Z0, Z8  // o0 = a4 + a6
	VSUBPD    Z4, Z0, Z10 // o2 = a4 - a6
	VPERMILPD $0x55, Z6, Z9
	VPXORQ    Z28, Z9, Z11 // rot1(a7)
	VPXORQ    Z29, Z9, Z9  // rot2(a7)
	VADDPD    Z2, Z11, Z11 // o1
	VADDPD    Z2, Z9, Z9   // o3

	VPERMILPD $0x55, Z11, Z0
	VPXORQ    Z28, Z0, Z0
	VADDPD    Z11, Z0, Z0 // o1 + rot1(o1)
	VMULPD    Z30, Z0, Z0 // t1 = W_8^1 * o1

	VPERMILPD $0x55, Z10, Z2
	VPXORQ    Z28, Z2, Z2 // t2 = W_8^2 * o2

	VPERMILPD $0x55, Z9, Z4
	VPXORQ    Z28, Z4, Z4
	VSUBPD    Z9, Z4, Z4  // rot1(o3) - o3
	VMULPD    Z30, Z4, Z4 // t3 = W_8^3 * o3

	VADDPD Z8, Z1, Z6  // y0
	VSUBPD Z8, Z1, Z1  // y4
	VADDPD Z0, Z7, Z9  // y1
	VSUBPD Z0, Z7, Z7  // y5
	VADDPD Z2, Z3, Z10 // y2
	VSUBPD Z2, Z3, Z3  // y6
	VADDPD Z4, Z5, Z11 // y3
	VSUBPD Z4, Z5, Z5  // y7

	// Fold the inverse 1/n here: exact for a power of two, and it saves a
	// separate streaming pass over the whole buffer.
	VMULPD Z31, Z6, Z6
	VMULPD Z31, Z9, Z9
	VMULPD Z31, Z10, Z10
	VMULPD Z31, Z11, Z11
	VMULPD Z31, Z1, Z1
	VMULPD Z31, Z7, Z7
	VMULPD Z31, Z3, Z3
	VMULPD Z31, Z5, Z5

	// Transpose the two 4x4 blocks of 128-bit lanes back into group-major
	// order: y_d holds [A_d, B_d, C_d, D_d] for the four groups A..D, and the
	// store needs [A0..A7][B0..B7][C0..C7][D0..D7]. Digits 0..3 first
	// (y0,y1,y2,y3 = Z6,Z9,Z10,Z11), then digits 4..7 (y4..y7 = Z1,Z7,Z3,Z5).
	VSHUFF64X2 $0x44, Z9, Z6, Z12    // [A0, B0, A1, B1]
	VSHUFF64X2 $0xEE, Z9, Z6, Z13    // [C0, D0, C1, D1]
	VSHUFF64X2 $0x44, Z11, Z10, Z14  // [A2, B2, A3, B3]
	VSHUFF64X2 $0xEE, Z11, Z10, Z15  // [C2, D2, C3, D3]
	VSHUFF64X2 $0x88, Z14, Z12, Z16  // [A0, A1, A2, A3]
	VSHUFF64X2 $0xDD, Z14, Z12, Z17  // [B0, B1, B2, B3]
	VSHUFF64X2 $0x88, Z15, Z13, Z18  // [C0, C1, C2, C3]
	VSHUFF64X2 $0xDD, Z15, Z13, Z19  // [D0, D1, D2, D3]

	VSHUFF64X2 $0x44, Z7, Z1, Z12   // [A4, B4, A5, B5]
	VSHUFF64X2 $0xEE, Z7, Z1, Z13   // [C4, D4, C5, D5]
	VSHUFF64X2 $0x44, Z5, Z3, Z14   // [A6, B6, A7, B7]
	VSHUFF64X2 $0xEE, Z5, Z3, Z15   // [C6, D6, C7, D7]
	VSHUFF64X2 $0x88, Z14, Z12, Z20 // [A4, A5, A6, A7]
	VSHUFF64X2 $0xDD, Z14, Z12, Z21 // [B4, B5, B6, B7]
	VSHUFF64X2 $0x88, Z15, Z13, Z22 // [C4, C5, C6, C7]
	VSHUFF64X2 $0xDD, Z15, Z13, Z23 // [D4, D5, D6, D7]

	VMOVUPD Z16, (R8)(DX*1)
	VMOVUPD Z20, 64(R8)(DX*1)
	VMOVUPD Z17, 128(R8)(DX*1)
	VMOVUPD Z21, 192(R8)(DX*1)
	VMOVUPD Z18, 256(R8)(DX*1)
	VMOVUPD Z22, 320(R8)(DX*1)
	VMOVUPD Z19, 384(R8)(DX*1)
	VMOVUPD Z23, 448(R8)(DX*1)

	ADDQ $4, CX
	ADDQ $512, DX
	CMPQ CX, R10
	JL   r8e_stage1_loop

	// =====================================================================
	// Remaining radix-8 stages, m = 8, 64, 512, ...
	// =====================================================================
	MOVQ twiddle+48(FP), R10 // R10 = &w1[0] of the current stage
	MOVQ R13, R11
	SHLQ $4, R11
	ADDQ R8, R11 // R11 = end of work

	MOVQ $128, AX            // AX = m*16 with m = 8
	MOVQ limit+120(FP), DX
	SHLQ $1, DX              // the span 8*m is AX/2, so compare against 2*limit
	CMPQ AX, DX
	JG   r8e_tail

r8e_stage_setup:
	MOVQ AX, R12
	SHLQ $3, R12 // R12 = 8*m*16, the byte stride between groups
	MOVQ R8, R9  // R9 = group base (the src pointer is dead now)

r8e_group_loop:
	MOVQ R9, SI          // SI = &a0, a1 at (SI)(AX*1)
	LEAQ (R9)(AX*2), R14 // R14 = &a2, a3 at (R14)(AX*1)
	LEAQ (R9)(AX*4), DI  // DI = &a4, a5 at (DI)(AX*1)
	LEAQ (DI)(AX*2), BP  // BP = &a6, a7 at (BP)(AX*1)

	MOVQ R10, CX          // CX = &w1, w2 at (CX)(AX*1), w3 at (CX)(AX*2)
	LEAQ (R10)(AX*2), R15
	ADDQ AX, R15          // R15 = &w4, w5 at (R15)(AX*1), w6 at (R15)(AX*2)
	LEAQ (R15)(AX*2), BX
	ADDQ AX, BX           // BX = &w7

	MOVQ AX, DX
	SHRQ $4, DX // DX = m, butterflies remaining in this group

r8e_inner_loop:
	// ---- first four butterflies: load and twiddle, x_d = w_d * a_d -----
	VMOVUPD (SI), Z0

	VMOVUPD        (SI)(AX*1), Z1
	VMOVDDUP       (CX), Z8
	VMOVDDUP       8(CX), Z9
	VPERMILPD      $0x55, Z1, Z10
	VMULPD         Z9, Z10, Z10
	VFMADDSUB213PD Z10, Z8, Z1

	VMOVUPD        (R14), Z2
	VMOVDDUP       (CX)(AX*1), Z8
	VMOVDDUP       8(CX)(AX*1), Z9
	VPERMILPD      $0x55, Z2, Z10
	VMULPD         Z9, Z10, Z10
	VFMADDSUB213PD Z10, Z8, Z2

	VMOVUPD        (R14)(AX*1), Z3
	VMOVDDUP       (CX)(AX*2), Z8
	VMOVDDUP       8(CX)(AX*2), Z9
	VPERMILPD      $0x55, Z3, Z10
	VMULPD         Z9, Z10, Z10
	VFMADDSUB213PD Z10, Z8, Z3

	VMOVUPD        (DI), Z4
	VMOVDDUP       (R15), Z8
	VMOVDDUP       8(R15), Z9
	VPERMILPD      $0x55, Z4, Z10
	VMULPD         Z9, Z10, Z10
	VFMADDSUB213PD Z10, Z8, Z4

	VMOVUPD        (DI)(AX*1), Z5
	VMOVDDUP       (R15)(AX*1), Z8
	VMOVDDUP       8(R15)(AX*1), Z9
	VPERMILPD      $0x55, Z5, Z10
	VMULPD         Z9, Z10, Z10
	VFMADDSUB213PD Z10, Z8, Z5

	VMOVUPD        (BP), Z6
	VMOVDDUP       (R15)(AX*2), Z8
	VMOVDDUP       8(R15)(AX*2), Z9
	VPERMILPD      $0x55, Z6, Z10
	VMULPD         Z9, Z10, Z10
	VFMADDSUB213PD Z10, Z8, Z6

	VMOVUPD        (BP)(AX*1), Z7
	VMOVDDUP       (BX), Z8
	VMOVDDUP       8(BX), Z9
	VPERMILPD      $0x55, Z7, Z10
	VMULPD         Z9, Z10, Z10
	VFMADDSUB213PD Z10, Z8, Z7

	// ---- second four butterflies, 64 bytes on, disjoint registers ------
	VMOVUPD 64(SI), Z12

	VMOVUPD        64(SI)(AX*1), Z13
	VMOVDDUP       64(CX), Z20
	VMOVDDUP       72(CX), Z21
	VPERMILPD      $0x55, Z13, Z22
	VMULPD         Z21, Z22, Z22
	VFMADDSUB213PD Z22, Z20, Z13

	VMOVUPD        64(R14), Z14
	VMOVDDUP       64(CX)(AX*1), Z20
	VMOVDDUP       72(CX)(AX*1), Z21
	VPERMILPD      $0x55, Z14, Z22
	VMULPD         Z21, Z22, Z22
	VFMADDSUB213PD Z22, Z20, Z14

	VMOVUPD        64(R14)(AX*1), Z15
	VMOVDDUP       64(CX)(AX*2), Z20
	VMOVDDUP       72(CX)(AX*2), Z21
	VPERMILPD      $0x55, Z15, Z22
	VMULPD         Z21, Z22, Z22
	VFMADDSUB213PD Z22, Z20, Z15

	VMOVUPD        64(DI), Z16
	VMOVDDUP       64(R15), Z20
	VMOVDDUP       72(R15), Z21
	VPERMILPD      $0x55, Z16, Z22
	VMULPD         Z21, Z22, Z22
	VFMADDSUB213PD Z22, Z20, Z16

	VMOVUPD        64(DI)(AX*1), Z17
	VMOVDDUP       64(R15)(AX*1), Z20
	VMOVDDUP       72(R15)(AX*1), Z21
	VPERMILPD      $0x55, Z17, Z22
	VMULPD         Z21, Z22, Z22
	VFMADDSUB213PD Z22, Z20, Z17

	VMOVUPD        64(BP), Z18
	VMOVDDUP       64(R15)(AX*2), Z20
	VMOVDDUP       72(R15)(AX*2), Z21
	VPERMILPD      $0x55, Z18, Z22
	VMULPD         Z21, Z22, Z22
	VFMADDSUB213PD Z22, Z20, Z18

	VMOVUPD        64(BP)(AX*1), Z19
	VMOVDDUP       64(BX), Z20
	VMOVDDUP       72(BX), Z21
	VPERMILPD      $0x55, Z19, Z22
	VMULPD         Z21, Z22, Z22
	VFMADDSUB213PD Z22, Z20, Z19

	// ---- butterfly of the first four, identical to stage 1's -----------
	VADDPD Z4, Z0, Z8  // a0
	VSUBPD Z4, Z0, Z9  // a1
	VADDPD Z6, Z2, Z10 // a2
	VSUBPD Z6, Z2, Z11 // a3
	VADDPD Z5, Z1, Z0  // a4
	VSUBPD Z5, Z1, Z2  // a5
	VADDPD Z7, Z3, Z4  // a6
	VSUBPD Z7, Z3, Z6  // a7

	VADDPD    Z10, Z8, Z1 // e0
	VSUBPD    Z10, Z8, Z3 // e2
	VPERMILPD $0x55, Z11, Z5
	VPXORQ    Z28, Z5, Z7
	VPXORQ    Z29, Z5, Z5
	VADDPD    Z9, Z7, Z7 // e1
	VADDPD    Z9, Z5, Z5 // e3

	VADDPD    Z4, Z0, Z8  // o0
	VSUBPD    Z4, Z0, Z10 // o2
	VPERMILPD $0x55, Z6, Z9
	VPXORQ    Z28, Z9, Z11
	VPXORQ    Z29, Z9, Z9
	VADDPD    Z2, Z11, Z11 // o1
	VADDPD    Z2, Z9, Z9   // o3

	VPERMILPD $0x55, Z11, Z0
	VPXORQ    Z28, Z0, Z0
	VADDPD    Z11, Z0, Z0
	VMULPD    Z30, Z0, Z0 // t1

	VPERMILPD $0x55, Z10, Z2
	VPXORQ    Z28, Z2, Z2 // t2

	VPERMILPD $0x55, Z9, Z4
	VPXORQ    Z28, Z4, Z4
	VSUBPD    Z9, Z4, Z4
	VMULPD    Z30, Z4, Z4 // t3

	VADDPD Z8, Z1, Z6  // y0
	VSUBPD Z8, Z1, Z1  // y4
	VADDPD Z0, Z7, Z9  // y1
	VSUBPD Z0, Z7, Z7  // y5
	VADDPD Z2, Z3, Z10 // y2
	VSUBPD Z2, Z3, Z3  // y6
	VADDPD Z4, Z5, Z11 // y3
	VSUBPD Z4, Z5, Z5  // y7

	// ---- butterfly of the second four, the same body on Z12..Z23 -------
	VADDPD Z16, Z12, Z20 // a0
	VSUBPD Z16, Z12, Z21 // a1
	VADDPD Z18, Z14, Z22 // a2
	VSUBPD Z18, Z14, Z23 // a3
	VADDPD Z17, Z13, Z12 // a4
	VSUBPD Z17, Z13, Z14 // a5
	VADDPD Z19, Z15, Z16 // a6
	VSUBPD Z19, Z15, Z18 // a7

	VADDPD    Z22, Z20, Z13 // e0
	VSUBPD    Z22, Z20, Z15 // e2
	VPERMILPD $0x55, Z23, Z17
	VPXORQ    Z28, Z17, Z19
	VPXORQ    Z29, Z17, Z17
	VADDPD    Z21, Z19, Z19 // e1
	VADDPD    Z21, Z17, Z17 // e3

	VADDPD    Z16, Z12, Z20 // o0
	VSUBPD    Z16, Z12, Z22 // o2
	VPERMILPD $0x55, Z18, Z21
	VPXORQ    Z28, Z21, Z23
	VPXORQ    Z29, Z21, Z21
	VADDPD    Z14, Z23, Z23 // o1
	VADDPD    Z14, Z21, Z21 // o3

	VPERMILPD $0x55, Z23, Z12
	VPXORQ    Z28, Z12, Z12
	VADDPD    Z23, Z12, Z12
	VMULPD    Z30, Z12, Z12 // t1

	VPERMILPD $0x55, Z22, Z14
	VPXORQ    Z28, Z14, Z14 // t2

	VPERMILPD $0x55, Z21, Z16
	VPXORQ    Z28, Z16, Z16
	VSUBPD    Z21, Z16, Z16
	VMULPD    Z30, Z16, Z16 // t3

	VADDPD Z20, Z13, Z18 // y0
	VSUBPD Z20, Z13, Z13 // y4
	VADDPD Z12, Z19, Z21 // y1
	VSUBPD Z12, Z19, Z19 // y5
	VADDPD Z14, Z15, Z22 // y2
	VSUBPD Z14, Z15, Z15 // y6
	VADDPD Z16, Z17, Z23 // y3
	VSUBPD Z16, Z17, Z17 // y7

	VMOVUPD Z6, (SI)
	VMOVUPD Z9, (SI)(AX*1)
	VMOVUPD Z10, (R14)
	VMOVUPD Z11, (R14)(AX*1)
	VMOVUPD Z1, (DI)
	VMOVUPD Z7, (DI)(AX*1)
	VMOVUPD Z3, (BP)
	VMOVUPD Z5, (BP)(AX*1)

	VMOVUPD Z18, 64(SI)
	VMOVUPD Z21, 64(SI)(AX*1)
	VMOVUPD Z22, 64(R14)
	VMOVUPD Z23, 64(R14)(AX*1)
	VMOVUPD Z13, 64(DI)
	VMOVUPD Z19, 64(DI)(AX*1)
	VMOVUPD Z15, 64(BP)
	VMOVUPD Z17, 64(BP)(AX*1)

	ADDQ $128, SI
	ADDQ $128, R14
	ADDQ $128, DI
	ADDQ $128, BP
	ADDQ $128, CX
	ADDQ $128, R15
	ADDQ $128, BX
	SUBQ $8, DX
	JNZ  r8e_inner_loop

	ADDQ R12, R9
	CMPQ R9, R11
	JL   r8e_group_loop

	// Next stage: advance past this stage's seven m-element planes (7*AX
	// bytes) and multiply m by eight.
	LEAQ (R10)(AX*8), R10
	SUBQ AX, R10
	SHLQ $3, AX
	MOVQ limit+120(FP), DX
	SHLQ $1, DX
	CMPQ AX, DX
	JLE  r8e_stage_setup

r8e_tail:
	// =====================================================================
	// Tail stage: radix-2 for n = 2*8^k, radix-4 for n = 4*8^k, none for 8^k.
	// R10 already points at its planes.
	// =====================================================================
	MOVQ limit+120(FP), AX
	CMPQ AX, R13
	JGE  r8e_copy_out

	SHLQ $1, AX
	CMPQ AX, R13
	JNE  r8e_tail4

	// ---- radix-2 tail ---------------------------------------------------
	// The smallest n that reaches here is 128 (n = 2*8^k and n >= 32), so a
	// half is 64 elements = 1024 bytes and the 64-byte step divides exactly.
	MOVQ R13, DX
	SHRQ $1, DX          // DX = n/2
	MOVQ R8, SI          // SI = &a0
	LEAQ (R8)(DX*8), DI
	LEAQ (DI)(DX*8), DI  // DI = &a1 = work + (n/2)*16
	SHLQ $4, DX          // DX = byte length of one half

	XORQ CX, CX

r8e_tail2_loop:
	VMOVDDUP (R10)(CX*1), Z8
	VMOVDDUP 8(R10)(CX*1), Z9
	VMOVUPD  (SI)(CX*1), Z0
	VMOVUPD  (DI)(CX*1), Z1

	VPERMILPD      $0x55, Z1, Z10
	VMULPD         Z9, Z10, Z10
	VFMADDSUB213PD Z10, Z8, Z1

	VADDPD Z0, Z1, Z2
	VSUBPD Z1, Z0, Z3

	VMOVUPD Z2, (SI)(CX*1)
	VMOVUPD Z3, (DI)(CX*1)

	ADDQ $64, CX
	CMPQ CX, DX
	JL   r8e_tail2_loop

	JMP r8e_copy_out

r8e_tail4:
	// ---- radix-4 tail, m = n/4 ------------------------------------------
	// The smallest n that reaches here is 32, so a quarter is 8 elements =
	// 128 bytes and the 64-byte step divides exactly.
	MOVQ R13, AX
	SHRQ $2, AX
	SHLQ $4, AX         // AX = (n/4)*16, the byte stride between quarters
	MOVQ R8, SI         // SI = &a0, a1 at (SI)(AX*1)
	LEAQ (R8)(AX*2), DI // DI = &a2, a3 at (DI)(AX*1)
	MOVQ R10, CX        // CX = &w1, w2 at (CX)(AX*1), w3 at (CX)(AX*2)
	MOVQ AX, DX         // DX = bytes remaining in a quarter

r8e_tail4_loop:
	VMOVUPD (SI), Z0

	VMOVUPD        (SI)(AX*1), Z1
	VMOVDDUP       (CX), Z8
	VMOVDDUP       8(CX), Z9
	VPERMILPD      $0x55, Z1, Z10
	VMULPD         Z9, Z10, Z10
	VFMADDSUB213PD Z10, Z8, Z1

	VMOVUPD        (DI), Z2
	VMOVDDUP       (CX)(AX*1), Z8
	VMOVDDUP       8(CX)(AX*1), Z9
	VPERMILPD      $0x55, Z2, Z10
	VMULPD         Z9, Z10, Z10
	VFMADDSUB213PD Z10, Z8, Z2

	VMOVUPD        (DI)(AX*1), Z3
	VMOVDDUP       (CX)(AX*2), Z8
	VMOVDDUP       8(CX)(AX*2), Z9
	VPERMILPD      $0x55, Z3, Z10
	VMULPD         Z9, Z10, Z10
	VFMADDSUB213PD Z10, Z8, Z3

	VADDPD Z2, Z0, Z4 // t0 = a0 + a2
	VSUBPD Z2, Z0, Z5 // t1 = a0 - a2
	VADDPD Z3, Z1, Z6 // t2 = a1 + a3
	VSUBPD Z3, Z1, Z7 // t3 = a1 - a3

	VPERMILPD $0x55, Z7, Z10
	VPXORQ    Z28, Z10, Z11 // rot1(t3)
	VPXORQ    Z29, Z10, Z10 // rot2(t3)

	VADDPD Z6, Z4, Z0
	VADDPD Z11, Z5, Z1
	VSUBPD Z6, Z4, Z2
	VADDPD Z10, Z5, Z3

	VMOVUPD Z0, (SI)
	VMOVUPD Z1, (SI)(AX*1)
	VMOVUPD Z2, (DI)
	VMOVUPD Z3, (DI)(AX*1)

	ADDQ $64, SI
	ADDQ $64, DI
	ADDQ $64, CX
	SUBQ $64, DX
	JNZ  r8e_tail4_loop

	// =====================================================================
	// Copy out if the transform ran in scratch.
	// =====================================================================
r8e_copy_out:
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r8e_true

	MOVQ R13, AX
	SHLQ $4, AX
	XORQ CX, CX

r8e_copy_loop:
	VMOVUPD (R8)(CX*1), Z0
	VMOVUPD 64(R8)(CX*1), Z1
	VMOVUPD Z0, (R9)(CX*1)
	VMOVUPD Z1, 64(R9)(CX*1)
	ADDQ    $128, CX
	CMPQ    CX, AX
	JL      r8e_copy_loop

r8e_true:
	VZEROUPPER
	MOVB $1, ret+144(FP)
	RET

r8e_false:
	VZEROUPPER
	MOVB $0, ret+144(FP)
	RET
